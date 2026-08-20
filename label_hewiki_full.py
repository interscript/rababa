"""Full-scale Dicta labeling of Hebrew Wikipedia for the s46 weak stage.

batch_distill_hewiki.py proved the recipe (DictaBERT-large-char-menaked
batch predict) but capped at 50K lines and skipped normalization.
This labels ALL of /datasets/hewiki/train.txt (80K lines) and applies
the s45 weak-corpus hygiene: nikud-only targets, length/Hebrew
filters, and 40-char-window decontamination against the Nakdimon test.

Output: /datasets/hebrew-hewiki-dicta/{train,val}.txt + DONE marker.

Usage:
    modal run --detach label_hewiki_full.py
"""

from __future__ import annotations

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.38.0",
        "huggingface_hub>=0.20",
        "sentencepiece>=0.2",
        "tqdm",
    )
    .workdir("/opt/rababa")
)

app = modal.App("rababa-hewiki-label", image=image)

_NIKUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")


def _is_hebrew_letter(c: str) -> bool:
    return "א" <= c <= "ת"


def _normalize(line: str) -> tuple[str, str] | None:
    tgt_chars = []
    for c in line:
        if c in _NIKUD_MARKS or _is_hebrew_letter(c) or c.isspace() or (33 <= ord(c) <= 126):
            tgt_chars.append(c)
    target = "".join(tgt_chars).strip()
    src = "".join(c for c in target if c not in _NIKUD_MARKS).strip()
    if not (10 <= len(src) <= 300):
        return None
    letters = sum(1 for c in src if _is_hebrew_letter(c))
    if letters / max(1, len(src.replace(" ", ""))) < 0.6:
        return None
    nikud = sum(1 for c in target if c in _NIKUD_MARKS)
    if nikud < len(src.replace(" ", "")) / 12:
        return None
    return src, target


@app.function(
    gpu="A10G",
    timeout=12 * 60 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def label() -> dict:
    import time
    from pathlib import Path
    from transformers import AutoModel, AutoTokenizer

    datasets_volume.reload()

    out_dir = Path("/datasets/hebrew-hewiki-dicta")
    if (out_dir / "DONE").exists():
        return {"status": "already-done"}

    # Nakdimon test windows for decontamination
    test_windows: set[str] = set()
    nakdi = Path("/datasets/nakdimon-combined")
    for cand in (nakdi / "test.txt", nakdi / "val.txt"):
        if cand.exists():
            for line in cand.read_text(encoding="utf-8").splitlines():
                t = line.strip()
                for i in range(0, max(1, len(t) - 40), 20):
                    test_windows.add(t[i : i + 40])
    print(f"[decontam] {len(test_windows)} nakdimon test windows", flush=True)

    model_name = "dicta-il/dictabert-large-char-menaked"
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    lines = []
    for line in Path("/datasets/hewiki/train.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if 10 <= len(line) <= 200:
            lines.append(line)
    print(f"[data] {len(lines)} hewiki lines to label", flush=True)

    def contaminated(src: str) -> bool:
        s = src.replace(" ", "")
        return any(s[i : i + 40] in test_windows for i in range(0, max(1, len(s) - 40), 20))

    pairs: list[tuple[str, str]] = []
    batch_size = 32
    t_start = time.time()
    for i in range(0, len(lines), batch_size):
        batch = lines[i : i + batch_size]
        try:
            predictions = model.predict(batch, tokenizer)
        except Exception as e:
            print(f"[warn] batch {i // batch_size} error: {e}", flush=True)
            continue
        for src_line, pred in zip(batch, predictions):
            if not pred or not pred.strip():
                continue
            norm = _normalize(pred.strip())
            if norm is None:
                continue
            src, target = norm
            if contaminated(src):
                continue
            pairs.append((src, target))
        n = i // batch_size + 1
        if n % 100 == 0:
            rate = len(pairs) / max(1.0, time.time() - t_start)
            eta = (len(lines) - i) / max(1.0, rate * batch_size)
            print(f"  [{n}/{(len(lines) + batch_size - 1) // batch_size}] {len(pairs)} kept, "
                  f"{rate:.1f}/s, ETA {eta / 60:.0f}min", flush=True)

    # dedupe by src
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for src, target in pairs:
        if src in seen:
            continue
        seen.add(src)
        deduped.append((src, target))

    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = min(2_000, len(deduped) // 20)
    (out_dir / "val.txt").write_text(
        "\n".join(t for _, t in deduped[:n_val]) + "\n", encoding="utf-8")
    (out_dir / "train.txt").write_text(
        "\n".join(t for _, t in deduped[n_val:]) + "\n", encoding="utf-8")
    (out_dir / "DONE").write_text(f"{len(deduped)} pairs\n", encoding="utf-8")
    datasets_volume.commit()
    return {"pairs": len(deduped), "elapsed_min": (time.time() - t_start) / 60}


@app.local_entrypoint()
def main():
    label.remote()
