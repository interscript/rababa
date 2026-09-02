"""Error analysis for Arabic diacritization predictions (steers RAFT iters).

Classifies every haraqat mismatch by:
  - position: word-final (case-ending/iʿrāb zone) vs word-internal
  - sentence position: clause-final word vs other
  - confusion pair (gold -> pred)

Reads the two-column CSV (gt, pred) the eval scripts write to the
checkpoints volume. Run locally after `modal volume get`.

Usage:
    python scripts/analyze_errors.py sadeed_preds_beam4.csv [--show 20]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


def letter_haraqat(text: str) -> list[tuple[str, str]]:
    seq: list[tuple[str, str]] = []
    for ch in text:
        if DIACRITICS_RE.match(ch):
            if seq:
                seq[-1] = (seq[-1][0], seq[-1][1] + ch)
        else:
            seq.append((ch, ""))
    return seq


def words(text: str) -> list[list[tuple[str, str]]]:
    out: list[list[tuple[str, str]]] = [[]]
    for ch, h in letter_haraqat(text):
        if ch == " ":
            out.append([])
        else:
            out[-1].append((ch, h))
    return [w for w in out if w]


def analyze(gt_rows: list[str], pred_rows: list[str]) -> dict:
    pos_counter: Counter = Counter()
    confusions: Counter = Counter()
    final_confusions: Counter = Counter()
    sents_final_wrong = sents = 0
    total_err = total_pos = 0

    for gt, pred in zip(gt_rows, pred_rows):
        gw, pw = words(gt), words(pred)
        if len(gw) != len(pw):
            continue
        sents += 1
        final_wrong = False
        for wi, (g_w, p_w) in enumerate(zip(gw, pw)):
            if len(g_w) != len(p_w):
                continue
            for li, ((gc, gh), (pc, ph)) in enumerate(zip(g_w, p_w)):
                if not gh and not ph:
                    continue
                total_pos += 1
                if gh == ph:
                    continue
                total_err += 1
                is_final = li == len(g_w) - 1
                pos_counter["word-final" if is_final else "internal"] += 1
                confusion = (gh or "∅", ph or "∅")
                confusions[confusion] += 1
                if is_final:
                    final_confusions[confusion] += 1
                    final_wrong = True
        if final_wrong:
            sents_final_wrong += 1

    return {
        "positions": total_pos,
        "errors": total_err,
        "der": total_err / max(1, total_pos),
        "by_position": dict(pos_counter.most_common()),
        "final_share": pos_counter.get("word-final", 0) / max(1, total_err),
        "confusions": confusions,
        "final_confusions": final_confusions,
        "sentences": sents,
        "sentences_with_final_error": sents_final_wrong,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args()

    gt_rows: list[str] = []
    pred_rows: list[str] = []
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                gt_rows.append(row[0])
                pred_rows.append(row[1])

    r = analyze(gt_rows, pred_rows)
    print(f"paragraph pairs : {r['sentences']}")
    print(f"scorable spots  : {r['positions']}")
    print(f"errors          : {r['errors']}  (DER {r['der']:.4%})")
    print(f"errors by pos   : {r['by_position']}")
    print(f"word-final share: {r['final_share']:.2%}")
    print(f"sents w/ final err: {r['sentences_with_final_error']}/{r['sentences']}")
    print("\ntop confusions (gold -> pred):")
    for (g, p), n in r["confusions"].most_common(args.show):
        print(f"  {n:6d}  {g} -> {p}")
    print("\ntop word-final confusions (iʿrāb zone):")
    for (g, p), n in r["final_confusions"].most_common(args.show):
        print(f"  {n:6d}  {g} -> {p}")


if __name__ == "__main__":
    sys.exit(main())
