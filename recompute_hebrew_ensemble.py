"""Recompute Hebrew ensemble from cached predictions (no GPU generation).

Loads cached predictions from the Modal datasets volume, votes across the
3 clean models (v4, s43, s44), reports DER + breakdown.

Usage:
    modal run recompute_hebrew_ensemble.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "rababa"
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4,<3", "numpy>=1.26,<3", "tqdm>=4.66")
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name=f"{APP_NAME}-hebrew-ens-recompute", image=image)

_NIKUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")


def _split_chars(s: str) -> list[tuple[str, str]]:
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
    return "֑" <= mark <= "֯"


def _is_nikud(mark: str) -> bool:
    return mark in _NIKUD_MARKS


def _char_errors(pred: str, gold: str) -> dict[str, int]:
    p = _split_chars(pred)
    g = _split_chars(gold)
    counts = {"nikud_wrong": 0, "teamim_wrong": 0, "both": 0, "ok": 0, "length_mismatch": 0}
    if len(p) != len(g):
        counts["length_mismatch"] += max(len(p), len(g))
        return counts
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
        if p_nik != g_nik and p_tm != g_tm:
            counts["both"] += 1
        elif p_nik != g_nik:
            counts["nikud_wrong"] += 1
        elif p_tm != g_tm:
            counts["teamim_wrong"] += 1
        else:
            counts["ok"] += 1
    return counts


@app.function(
    cpu=2,
    timeout=30 * 60,
    volumes={"/datasets": datasets_volume},
)
def recompute() -> dict:
    from rababa.evaluate import seq2seq_der
    from rababa.datasets import _find_nakdimon_root

    datasets_volume.reload()

    # Load test set
    test_path = Path(_find_nakdimon_root()) / "test.txt"
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        undiacritized = "".join(c for c in line if c not in _NIKUD_MARKS).strip()
        if 2 <= len(undiacritized) <= 512:
            examples.append((undiacritized, line))
    print(f"[ens] test examples: {len(examples)}", flush=True)

    # Load cached predictions for the 3 clean models
    cache_dir = Path("/datasets/hebrew-pred-cache")
    members = ["v4", "s43", "s44"]
    all_preds: dict[str, list[str]] = {}
    for name in members:
        cache_file = cache_dir / f"{name}.jsonl"
        if not cache_file.is_file():
            return {"error": f"no cache for {name}"}
        preds = []
        for ln in cache_file.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                preds.append(json.loads(ln)["pred"])
        if len(preds) != len(examples):
            return {"error": f"{name}: {len(preds)} preds != {len(examples)} examples"}
        all_preds[name] = preds
    print(f"[ens] loaded preds for {members}", flush=True)

    # 3-way majority vote per position
    results = {}
    ens_wrong = ens_pos = 0
    ens_agg = {"nikud_wrong": 0, "teamim_wrong": 0, "both": 0, "ok": 0, "length_mismatch": 0}
    aligned_nik_wrong = aligned_ok = 0
    for idx, (_, gold) in enumerate(examples):
        splits = [_split_chars(all_preds[n][idx]) for n in members]
        lens = [len(s) for s in splits if s]
        if lens and all(l == lens[0] for l in lens):
            merged = []
            for pos in range(lens[0]):
                votes = [s[pos] for s in splits]
                counts: dict = {}
                for v in votes:
                    counts[v] = counts.get(v, 0) + 1
                best = max(counts.items(), key=lambda kv: kv[1])[0]
                merged.append(best)
            merged_str = "".join(c + m for c, m in merged)
        else:
            merged_str = all_preds[members[0]][idx]

        der, n = seq2seq_der(merged_str, gold)
        ens_wrong += int(der * n)
        ens_pos += n
        for k, v in _char_errors(merged_str, gold).items():
            ens_agg[k] += v

    total_aligned = ens_agg["ok"] + ens_agg["nikud_wrong"] + ens_agg["both"]
    results["ensemble_3way"] = {
        "der_standard": ens_wrong / max(1, ens_pos),
        "n_examples": len(examples),
        "members": members,
        "breakdown": ens_agg,
        "nikud_accuracy_on_aligned": (ens_agg["ok"] / max(1, total_aligned)) if total_aligned else None,
    }

    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    return results


@app.local_entrypoint()
def main():
    result = recompute.remote()
    print(json.dumps(result, indent=2, ensure_ascii=False))
