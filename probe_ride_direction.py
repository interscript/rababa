"""RIDE direction-transferability probe (TODO.sota-2026/05).

Question: is the r7-over-r6 SFT residual direction (per encoder layer)
domain-general, or news-idiosyncratic? RIDE-style extrapolation
(regressing a student toward h_teacher + λ(h_teacher − h_base)) is only
sound if the direction transfers.

Method: mean-pool per-layer encoder hidden states of BOTH teachers on
two text domains (classical SadeedDiac windows vs news lines), form
d_domain = mean_r7 − mean_r6, then cos(d_classical, d_news) per layer.

Kill rule: max-layer cosine < 0.5 ⇒ direction domain-idiosyncratic ⇒
close the arm without any training compute.

Usage:
    modal run probe_ride_direction.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

BASE_RUN = "rababa_arabic_byt5/run-006-morph"
TEACHER_RUN = "rababa_arabic_byt5/run-007-news"
N_UNITS = 200
KILL_COSINE = 0.5

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "pyarrow", "tqdm")
    .add_local_dir("data/sadeed-diac-25", "/opt/rababa/data/sadeed-diac-25", copy=True)
    .add_local_file("ride_probe_math.py", "/opt/rababa/ride_probe_math.py", copy=True)
    .workdir("/opt/rababa")
)

app = modal.App("rababa-ride-probe", image=image)


@app.function(gpu="A10G", timeout=2 * 60 * 60, volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume})
def probe() -> dict:
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from ride_probe_math import cosine_rows, residual_directions

    checkpoints_volume.reload()
    datasets_volume.reload()

    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    classical = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()[:N_UNITS]]
    news = [l.strip() for l in (Path("/datasets/arabic-news-r5/news.txt")).read_text(encoding="utf-8").splitlines() if l.strip()][:N_UNITS]

    def layer_means(model_dir: str, texts: list[str]) -> list[list[float]]:
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).cuda().eval()
        sums, counts = None, 0
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                batch = [t[:1400] for t in texts[i : i + 8]]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1600).to("cuda")
                mask = enc["attention_mask"].unsqueeze(-1).float()
                out = model.encoder(**enc, output_hidden_states=True)
                hs = out.hidden_states  # tuple: (n_layers+1, B, T, D)
                if sums is None:
                    sums = [torch.zeros(h.shape[-1], device="cuda") for h in hs]
                for li, h in enumerate(hs):
                    sums[li] += (h * mask).sum(dim=(0, 1))
                counts += int(enc["attention_mask"].sum().item())
        del model
        torch.cuda.empty_cache()
        return [(s / counts).tolist() for s in sums]

    base_dir = str(Path("/checkpoints") / BASE_RUN / "best")
    teach_dir = str(Path("/checkpoints") / TEACHER_RUN / "best")

    results = {}
    for domain, texts in (("classical", classical), ("news", news)):
        b = layer_means(base_dir, texts)
        t = layer_means(teach_dir, texts)
        results[domain] = residual_directions(b, t)

    cos = cosine_rows(results["classical"], results["news"])
    table_rows = [(i, c) for i, c in enumerate(cos)]
    best = max(cos)
    verdict = "TRANSFER" if best >= KILL_COSINE else "KILL"
    print("layer cosine(classical-dir, news-dir):")
    for i, c in table_rows:
        print(f"  L{i:02d}  {c:+.4f}")
    print(f"max = {best:+.4f}  ->  {verdict} (kill < {KILL_COSINE})")

    out = {
        "base": BASE_RUN,
        "teacher": TEACHER_RUN,
        "n_units": N_UNITS,
        "cosine_by_layer": cos,
        "max_cosine": best,
        "kill_cosine": KILL_COSINE,
        "verdict": verdict,
    }
    dest = Path("/checkpoints") / "ride_probe_r6_r7.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    checkpoints_volume.commit()
    return out


@app.local_entrypoint()
def main():
    print(probe.remote())
