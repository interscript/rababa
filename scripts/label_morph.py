"""Label the Arabic corpus with morphological POS tags (qalsadi, CPU).

The aux-task lever: ByT5 has never seen explicit morphological
supervision (POS/case). Qalsadi analyzes diacritized words offline —
our gold corpus is fully diacritized, so its analyses act as
near-ground-truth morphological labels for an r6 multi-task target.

Phase 1 (small limit) prints a sample analysis for API verification;
phase 2 (full) writes /datasets/arabic-morph/train.jsonl.

Usage:
    modal run scripts/label_morph.py::probe   # verify qalsadi API
    modal run scripts/label_morph.py          # label 300k lines
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
CORPUS = "/datasets/sadeed-decontam/train.txt"
OUT = Path("/datasets/arabic-morph/train.jsonl")
N_LINES = 300_000
N_WORKERS = 8

image = modal.Image.debian_slim(python_version="3.11").pip_install("qalsadi")
app = modal.App("rababa-label-morph", image=image)

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from qalsadi.analex import Analex
        _analyzer = Analex()
    return _analyzer


def _pos_of(word_result) -> str:
    for attr in ("pos", "tags", "tag"):
        v = getattr(word_result, attr, None)
        if v:
            return str(v)[:40]
    for meth in ("get_pos", "get_tags"):
        m = getattr(word_result, meth, None)
        if callable(m):
            try:
                return str(m())[:40]
            except Exception:
                pass
    return "X"


import re as _re
_PUNCT = _re.compile(r"[\u060C\u061B\u061F.()\[\]\u00AB\u00BB\"'\-\u2013\u2014\u2026]+")
_DIAC = _re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def _label_line(line: str) -> tuple[str, list[str]]:
    an = _get_analyzer()
    words = line.split()
    tags: list[str] = []
    for w in words:
        w = _PUNCT.sub("", w)
        if not w:
            tags.append("PUNCT")
            continue
        # Diacritized input carries the case ending -> exact tag when it
        # resolves. qalsadi's stopword list needs bare letters, so misses
        # fall back to a COARSE pos from the stripped form (marked "C:").
        try:
            res = an.check_word(w) or []
        except Exception:
            res = []
        if res:
            tags.append(_pos_of(res[0]))
            continue
        try:
            res2 = an.check_word(_DIAC.sub("", w)) or []
        except Exception:
            res2 = []
        if res2:
            tags.append("C:" + _pos_of(res2[0]).split(":")[0])
        else:
            tags.append("X")
    return line, tags


@app.function(volumes={"/datasets": datasets_volume}, cpu=N_WORKERS, timeout=2 * 60 * 60)
def probe() -> dict:
    from qalsadi.analex import Analex
    an = Analex()
    sample = ["قَالَ", "الْكِتَابُ", "يَكْتُبُونَ", "بِسْمِ"]
    for w in sample:
        res = an.check_word(w) or []
        if res:
            print(f"{w}: attrs={sorted(a for a in dir(res[0]) if not a.startswith('_'))[:25]}", flush=True)
            print(f"   pos={_pos_of(res[0])!r} stem={getattr(res[0], 'stem', '?')}", flush=True)
        else:
            print(f"{w}: no analysis", flush=True)
    return {"ok": True}


@app.function(volumes={"/datasets": datasets_volume}, cpu=N_WORKERS, timeout=11 * 60 * 60)
def label() -> dict:
    import multiprocessing as mp
    from collections import Counter

    lines = [
        l.strip()
        for l in Path(CORPUS).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ][:N_LINES]
    print(f"[data] {len(lines)} lines", flush=True)

    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(_label_line, lines, chunksize=200)

    from datetime import datetime, timezone

    tag_counts = Counter(t for _, tags in results for t in tags)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        import re
        diac = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
        for line, tags in results:
            src = diac.sub("", line)
            f.write(json.dumps({"src": src, "gold": line, "tags": tags}, ensure_ascii=False) + "\n")
    manifest = OUT.parent / "MANIFEST.txt"
    manifest.write_text(
        f"lines: {len(results)}\ntop_tags: {tag_counts.most_common(20)}\n"
        f"labeled: {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    datasets_volume.commit()
    return {"lines": len(results), "top_tags": tag_counts.most_common(10)}
