"""Hebrew error analysis + v2/v4 ensemble evaluation.

Answers three questions:
1. Where do the 17% DER errors come from? (nikud vs teamim vs consonants)
2. What is nikud-only DER if teamim are stripped from comparison?
3. Does a v2+v4 ensemble beat either model alone?

Usage:
    modal run --detach analyze_hebrew_errors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "rababa"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.40,<5",
        "sentencepiece",
        "protobuf",
        "accelerate>=1.1.0",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name=f"{APP_NAME}-hebrew-analysis", image=image)

_NIKUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")


def _split_chars(s: str) -> list[tuple[str, str]]:
    """Split into (consonant, following-marks) pairs."""
    result = []
    cur_c = None
    cur_marks = []
    for c in s:
        if "֑" <= c <= "ׇ":
            cur_marks.append(c)
        else:
            if cur_c is not None:
                result.append((cur_c, "".join(cur_marks)))
            cur_c = c
            cur_marks = []
    if cur_c is not None:
        result.append((cur_c, "".join(cur_marks)))
    return result


def _is_teamim(mark: str) -> bool:
    return "֑" <= mark <= "֯"  # U+0591-U+05AF


def _is_nikud(mark: str) -> bool:
    return mark in _NIKUD_MARKS


def _char_errors(pred: str, gold: str) -> dict[str, int]:
    """Count errors by type at each consonant position."""
    p = _split_chars(pred)
    g = _split_chars(gold)
    if len(p) != len(g):
        return {"length_mismatch": max(len(p), len(g)), "nikud_wrong": 0, "teamim_wrong": 0, "both": 0, "ok": 0}
    counts = {"nikud_wrong": 0, "teamim_wrong": 0, "both": 0, "ok": 0, "length_mismatch": 0}
    for (pc, pm), (gc, gm) in zip(p, g):
        if pc != gc:
            counts["length_mismatch"] += 1
            continue
        if pm == gm:
            counts["ok"] += 1
            continue
        p_nik = "".join(m for m in pm if _is_nikud(m))
        g_nik = "".join(m for m in gm if _is_nikud(m))
        p_tm = "".join(m for m in pm if _is_teamim(m))
        g_tm = "".join(m for m in gm if _is_teamim(m))
        nik_wrong = p_nik != g_nik
        tm_wrong = p_tm != g_tm
        if nik_wrong and tm_wrong:
            counts["both"] += 1
        elif nik_wrong:
            counts["nikud_wrong"] += 1
        elif tm_wrong:
            counts["teamim_wrong"] += 1
        else:
            counts["ok"] += 1  # other marks match differently but not nikud/teamim
    return counts


def _der_stripped(pred: str, gold: str, strip_teamim: bool) -> tuple[int, int]:
    """DER with optional teamim stripping from both sides."""
    if strip_teamim:
        pred = "".join(c for c in pred if not _is_teamim(c))
        gold = "".join(c for c in gold if not _is_teamim(c))
    p = _split_chars(pred)
    g = _split_chars(gold)
    if len(p) != len(g):
        return max(len(p), len(g)), max(len(p), len(g))
    wrong = sum(1 for a, b in zip(p, g) if a != b)
    return wrong, len(g)


@app.function(
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def analyze() -> dict:
    """Generate with all models, compute error breakdown + ensembles."""
    import torch
    from transformers import T5ForConditionalGeneration
    from rababa.evaluate import seq2seq_der
    from rababa.datasets import _find_nakdimon_root

    checkpoints_volume.reload()
    datasets_volume.reload()

    device = torch.device("cuda")

    ckpts = {
        "v2": "/checkpoints/rababa_hebrew_byt5_v2/run-001/best",
        "v4": "/checkpoints/rababa_hebrew_byt5_v4/run-001/best",
        "s43": "/checkpoints/rababa_hebrew_byt5_s43/run-001/best",
        "s44": "/checkpoints/rababa_hebrew_byt5_s44/run-001/best",
    }

    # ByT5 tokenizer is byte-level and identical everywhere; load from v4
    # checkpoint (saved with current transformers, unlike v2's).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

    test_path = Path(_find_nakdimon_root()) / "test.txt"
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        undiacritized = "".join(c for c in line if c not in _NIKUD_MARKS).strip()
        if 2 <= len(undiacritized) <= 512:
            examples.append((undiacritized, line))
    print(f"[analyze] test examples: {len(examples)}", flush=True)

    # Generate with each model (or load cached predictions)
    cache_dir = Path("/datasets/hebrew-pred-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_preds: dict[str, list[str]] = {}
    models_loaded = {}
    for name, ckpt in ckpts.items():
        cache_file = cache_dir / f"{name}.jsonl"
        if cache_file.is_file():
            preds = []
            for ln in cache_file.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    preds.append(json.loads(ln)["pred"])
            if len(preds) == len(examples):
                all_preds[name] = preds
                print(f"[analyze] {name}: loaded {len(preds)} cached preds", flush=True)
                continue

        if not Path(ckpt).is_dir():
            print(f"[analyze] WARNING: {name} at {ckpt} not found, skipping", flush=True)
            continue
        print(f"[analyze] generating with {name}", flush=True)
        m = T5ForConditionalGeneration.from_pretrained(ckpt).to(device)
        # Old checkpoints may carry generation configs that break under new
        # transformers (empty output). Force clean ByT5 defaults.
        m.generation_config.decoder_start_token_id = 0
        m.generation_config.eos_token_id = 1
        m.generation_config.pad_token_id = 0
        m.eval()

        preds = []
        batch_size = 8
        with torch.no_grad():
            for i in range(0, len(examples), batch_size):
                batch = examples[i : i + batch_size]
                src = [s for s, _ in batch]
                enc = tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                gen = m.generate(
                    **enc,
                    max_new_tokens=512,
                    num_beams=4,
                    decoder_start_token_id=0,
                    eos_token_id=1,
                    pad_token_id=0,
                )
                preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
                if i % 320 == 0 and i > 0:
                    print(f"  [{name} {i}/{len(examples)}]", flush=True)
        del m
        torch.cuda.empty_cache()
        all_preds[name] = preds
        with cache_file.open("w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps({"pred": p}, ensure_ascii=False) + "\n")
        datasets_volume.commit()
        print(f"[analyze] {name}: generated + cached {len(preds)}", flush=True)

    if not all_preds:
        return {"error": "no predictions"}

    # Analysis per model: standard DER (seq2seq_der, comparable to v2's 17.3%)
    # + strict breakdown for diagnosis
    results = {}
    error_totals = {}
    for name, preds in all_preds.items():
        total_wrong = total_pos = 0
        total_nik_wrong = total_nik_pos = 0
        agg = {"nikud_wrong": 0, "teamim_wrong": 0, "both": 0, "ok": 0, "length_mismatch": 0}
        for pred, (_, gold) in zip(preds, examples):
            der, n = seq2seq_der(pred, gold)
            total_wrong += int(der * n)
            total_pos += n
            w2, p2 = _der_stripped(pred, gold, strip_teamim=True)
            total_nik_wrong += w2
            total_nik_pos += p2
            for k, v in _char_errors(pred, gold).items():
                agg[k] += v
        results[name] = {
            "der_standard": total_wrong / max(1, total_pos),
            "der_nikud_only_strict": total_nik_wrong / max(1, total_nik_pos),
            "n_examples": len(examples),
        }
        error_totals[name] = agg

    # Ensembles: majority vote across all models per position
    names = list(all_preds.keys())
    if len(names) >= 2:
        ens_wrong = ens_pos = 0
        ens_agg = {"nikud_wrong": 0, "teamim_wrong": 0, "both": 0, "ok": 0, "length_mismatch": 0}
        for idx, (_, gold) in enumerate(examples):
            splits = []
            for name in names:
                s = _split_chars(all_preds[name][idx])
                if s:
                    splits.append(s)
            if not splits:
                continue
            base_len = len(splits[0])
            if all(len(s) == base_len for s in splits):
                merged = []
                for pos in range(base_len):
                    votes = [s[pos] for s in splits]
                    # majority: pick most common (consonant, marks) pair
                    counts: dict = {}
                    for v in votes:
                        counts[v] = counts.get(v, 0) + 1
                    best = max(counts.items(), key=lambda kv: kv[1])[0]
                    merged.append(best)
                merged_str = "".join(c + m for c, m in merged)
            else:
                merged_str = all_preds[names[0]][idx]
            der, n = seq2seq_der(merged_str, gold)
            ens_wrong += int(der * n)
            ens_pos += n
            for k, v in _char_errors(merged_str, gold).items():
                ens_agg[k] += v
        results[f"ensemble_{len(names)}way"] = {
            "der_standard": ens_wrong / max(1, ens_pos),
            "n_examples": len(examples),
            "members": names,
        }
        error_totals[f"ensemble_{len(names)}way"] = ens_agg

    output = {"results": results, "error_breakdown": error_totals}
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)
    return output


@app.local_entrypoint()
def main():
    result = analyze.remote()
    print(json.dumps(result, indent=2, ensure_ascii=False))
