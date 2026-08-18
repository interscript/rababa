"""Multi-reference WikiNews evaluation (QCRI EMNLP 2025 protocol).

Python port of qcri advancing-arabic-diacritization EvalDiac.java:
- Word correct if ANY "/" alternate matches letter-by-letter.
- Letters where the reference carries NO diacritic accept any prediction.
- Redundant diacritization normalized away first (fatha+alef, kasra+yaa,
  damma+waw, sukun, shadda order, iltiqa-sakinayn, tanwin-alef order).
- WER = word errors / words; DER = letter errors / letters.
- no_case_ending mode skips each word's final letter (their stem mode).

Contamination note: WikiNews-2014 derives from Tashkeela (in our train
corpus); 2024 is the updated benchmark and the primary number.

Usage:
    modal run --detach eval_wikinews_multiref.py            # r3, 2024
    modal run eval_wikinews_multiref.py --model-dir X --tag Y --bench Z
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

MODEL_DIR = "/checkpoints/rababa_arabic_byt5/run-003-domain/best"
TAG = "r3"
BENCH = "WikiNews_2024_Multi_Ref.txt.diac"

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "pandas", "tqdm")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-wikinews-multiref", image=image)

# ---- Java removeDefaultDiac port (order matters) ----
_DEFAULT_RULES = [
    ("َا", "ا"), ("ِي", "ي"), ("ُو", "و"), ("الْ", "ال"),
    ("ْ", ""),
    ("َّ", "َّ"), ("ِّ", "ِّ"), ("ُّ", "ُّ"),
    ("ًّ", "ًّ"), ("ٍّ", "ٍّ"), ("ٌّ", "ٌّ"),
    ("اَ", "ا"), ("اِ", "ا"), ("لِا", "لا"), ("اً", "ًا"),
]

_COMBOS = ("َّ", "ِّ", "ُّ", "ًّ", "ٍّ", "ٌّ", "ّْ")
_D2C = {
    "َ": "F", "ِ": "Z", "ُ": "N", "ً": "FN", "ٍ": "ZN", "ٌ": "NN",
    "ّ": "SH", "ْ": "SK", "ٰ": "KS", "ٓ": "MD", "ٔ": "HMZ", "ٕ": "HMZI",
}


def remove_default_diac(s: str) -> str:
    for a, b in _DEFAULT_RULES:
        s = s.replace(a, b)
    return s


def get_diac_codes(s: str) -> list[tuple[str, str]]:
    """(letter, diac-code) pairs, multi-char diacritic clusters merged."""
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if DIACRITICS_RE.match(ch):
            i += 1
            continue
        j = i + 1
        while j < len(s) and DIACRITICS_RE.match(s[j]):
            j += 1
        cluster = s[i + 1 : j]
        code = ""
        if cluster in _COMBOS:
            code = "SH" + _D2C[cluster[1:]] if len(cluster) == 2 else "SH"
        else:
            for c in cluster:
                code += _D2C.get(c, "")
        out.append((ch, code))
        i = j
    return out


def word_matches(sys_word: str, ref_alt: str, skip_last: bool) -> tuple[bool, int, int]:
    """(matched, correct_letters, total_letters) for one alternate.

    Mirrors EvalDiac: same letter sequence required; a ref letter with no
    diacritic accepts anything; shadda-only ref accepts shadda+vowel.
    """
    r = get_diac_codes(remove_default_diac(ref_alt))
    s = get_diac_codes(remove_default_diac(sys_word))
    n = len(r)
    if n != len(s):
        return False, 0, n
    end = n - 1 if skip_last else n
    correct = 0
    for j in range(end):
        (rc, rd), (sc, sd) = r[j], s[j]
        d1 = sd if not rd else rd  # empty ref diac accepts anything
        if d1 == sd or (rd == "SH" and sd.startswith("SH")):
            correct += 1
    if skip_last:
        correct += 1
    matched = correct == end or (end == 0)
    return matched, correct, n


def score(preds: list[str], refs: list[str], skip_last: bool) -> dict:
    tot_w = c_w = tot_l = c_l = 0
    for pred, ref in zip(preds, refs):
        pw, rw = pred.split(), ref.split()
        tot_w += len(rw)
        for k, ref_w in enumerate(rw):
            alts = ref_w.split("/") if "/" in ref_w else [ref_w]
            if k >= len(pw):
                # prediction shorter than reference: ref letters count, no credit
                tot_l += len(get_diac_codes(remove_default_diac(alts[0])))
                continue
            scored = [word_matches(pw[k], a, skip_last) for a in alts]
            matched = next((x for x in scored if x[0]), None)
            if matched is not None:
                _, c, t = matched
                c_w += 1
                c_l += c
                tot_l += t
            else:
                _, c, t = max(scored, key=lambda x: x[1])
                tot_l += t
                c_l += c
    return {
        "WER": round(100.0 * (1 - c_w / tot_w), 4) if tot_w else 0.0,
        "DER": round(100.0 * (1 - c_l / tot_l), 4) if tot_l else 0.0,
        "words": tot_w,
        "letters": tot_l,
    }


@app.function(
    gpu="A100",
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def run(model_dir: str = MODEL_DIR, tag: str = TAG, bench: str = BENCH) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    datasets_volume.reload()
    checkpoints_volume.reload()

    lines = [
        l.strip()
        for l in Path(f"/datasets/wikinews/{bench}").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    inputs = [DIACRITICS_RE.sub("", l) for l in lines]
    print(f"[data] {bench}: {len(lines)} lines", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda").eval()
    device = next(model.parameters()).device

    preds: list[str] = []
    with torch.no_grad():
        for i in range(0, len(inputs), 8):
            batch = inputs[i : i + 8]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=1024).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=2048, num_beams=1)
            preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            if (i // 8) % 20 == 0:
                print(f"[gen] {i + len(batch)}/{len(inputs)}", flush=True)

    out: dict = {"model": tag, "bench": bench}
    for skip_last, mode in ((False, "full"), (True, "no_case_ending")):
        out[mode] = score(preds, lines, skip_last)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)

    out_dir = Path("/checkpoints") / "/".join(model_dir.strip("/").split("/")[1:-1])
    (out_dir / f"wikinews_multiref_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    checkpoints_volume.commit()
    return out


@app.local_entrypoint()
def main():
    run.remote()
