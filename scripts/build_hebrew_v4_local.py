"""Rebuild the hebrew-v4 combined corpus locally (no Modal).

Mirrors train_hebrew_v4.py::build_combined_corpus exactly — same
sources (local data/ dirs), same strip/dedupe/filter order — so the
HF-mounted dataset is byte-comparable to the run-021 volume corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

_NIKUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")


def _strip_nikud(s: str) -> str:
    return "".join(c for c in s if c not in _NIKUD_MARKS)


def _has_consonants(s: str) -> bool:
    hebrew_consonants = set("אבגדהוזחטיכלמנסעפצקרשתךםןףץ")
    return any(c in hebrew_consonants for c in s)


def _load_labeled_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and any("֑" <= c <= "ׇ" for c in line):
            out.append(line)
    return out


def build(out_root: Path) -> dict:
    sources = {
        "nakdimon": DATA / "nakdimon/train.txt",
        "sefaria_tanakh": DATA / "sefaria-tanakh/train.txt",
        "distilled_v1": DATA / "hebrew-distilled/train.txt",
        "distilled_v2": DATA / "hebrew-dictabert-distilled/train.txt",
        "expanded_v2": DATA / "hebrew-expanded-v2/train.txt",
    }
    all_pairs = []
    seen: set[tuple[str, str]] = set()
    counts = {}
    for name, path in sources.items():
        new_count = 0
        for line in _load_labeled_lines(path):
            undiacritized = _strip_nikud(line).strip()
            if not undiacritized or not _has_consonants(undiacritized):
                continue
            if len(undiacritized) < 5 or len(undiacritized) > 500:
                continue
            key = (undiacritized, line)
            if key in seen:
                continue
            seen.add(key)
            all_pairs.append({"src": undiacritized, "tgt": line})
            new_count += 1
        counts[name] = new_count

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "train.jsonl").open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    test_pairs = []
    for line in _load_labeled_lines(DATA / "nakdimon/test.txt"):
        undiacritized = _strip_nikud(line).strip()
        if undiacritized and _has_consonants(undiacritized):
            test_pairs.append({"src": undiacritized, "tgt": line})
    with (out_root / "test.jsonl").open("w", encoding="utf-8") as f:
        for pair in test_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return {"train": len(all_pairs), "test": len(test_pairs), "counts": counts}


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "hebrew-v4"
    stats = build(out)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
