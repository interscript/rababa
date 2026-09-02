"""Assemble ALL available Hebrew data into one expanded training corpus.

Arabic improved 2.42%→1.30% with 28× more data. Hebrew has only 50K
examples. We have 80 chunks of Dicta-distilled data + Sefaria + Nakdimon.
Let's combine everything for a 200K+ example corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git")
    .pip_install("torch>=2.4,<3", "numpy>=1.26,<3", "tqdm>=4.66", "pyyaml>=6.0")
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name="rababa", image=image)


@app.function(
    cpu=4,
    timeout=30 * 60,
    volumes={"/datasets": datasets_volume},
)
def assemble_hebrew_corpus() -> dict:
    """Combine all Hebrew data sources into one large corpus."""
    from pathlib import Path as _P

    datasets_volume.reload()

    # Collect all diacritized Hebrew text from all sources
    all_lines: set[str] = set()
    source_counts: dict[str, int] = {}

    def add_from_file(path: _P, source: str):
        nonlocal all_lines
        if not path.is_file():
            return
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if len(line) < 5 or len(line) > 512:
                continue
            if line not in all_lines:
                all_lines.add(line)
                count += 1
        source_counts[source] = count
        print(f"  {source}: +{count} unique lines from {path}", flush=True)

    def add_from_dir(dir_path: _P, source: str):
        nonlocal all_lines
        if not dir_path.is_dir():
            return
        count = 0
        for chunk in sorted(dir_path.glob("chunk-*.txt")):
            for line in chunk.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if len(line) < 5 or len(line) > 512:
                    continue
                if line not in all_lines:
                    all_lines.add(line)
                    count += 1
        source_counts[source] = count
        print(f"  {source}: +{count} unique lines from {dir_path}/*", flush=True)

    print("=== Collecting Hebrew data ===", flush=True)

    # 1. Nakdimon original corpus (gold labels)
    for split in ("train", "val", "test"):
        add_from_file(_P(f"/datasets/nakdimon/{split}.txt"), f"nakdimon_{split}")

    # 2. Dicta-distilled data (all chunks)
    add_from_dir(_P("/datasets/hebrew-distilled"), "dicta_distilled")

    # 3. Existing distilled train file
    add_from_file(_P("/datasets/hebrew-distilled/train.txt"), "dicta_distilled_train")

    # 4. Sefaria (Biblical/Rabbinic)
    sefaria = _P("/opt/rababa/data/sefaria")
    for split in ("train", "val", "test"):
        for name in (f"{split}.txt", f"sefaria_{split}/{split}.txt"):
            add_from_file(sefaria / name, f"sefaria_{split}")

    # 5. Hebrew-distilled v2 (if any from DictaBERT)
    add_from_dir(_P("/datasets/hebrew-distilled-v2"), "dictabert_distilled_v2")

    total = len(all_lines)
    print(f"\n=== Total unique lines: {total} ===", flush=True)
    print(f"Sources: {json.dumps(source_counts, indent=2)}", flush=True)

    # Split into train/val/test (90/5/5)
    import random
    lines_list = sorted(all_lines)  # deterministic
    random.seed(42)
    random.shuffle(lines_list)

    n_test = max(1000, total // 20)
    n_val = max(1000, total // 20)
    test_lines = lines_list[:n_test]
    val_lines = lines_list[n_test:n_test + n_val]
    train_lines = lines_list[n_test + n_val:]

    # Write expanded corpus
    out_dir = _P("/datasets/hebrew-expanded")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, data in [("train", train_lines), ("val", val_lines), ("test", test_lines)]:
        out_path = out_dir / f"{split}.txt"
        out_path.write_text("\n".join(data) + "\n", encoding="utf-8")
        print(f"  {split}: {len(data)} lines → {out_path}", flush=True)

    datasets_volume.commit()

    return {
        "total_unique": total,
        "train": len(train_lines),
        "val": len(val_lines),
        "test": len(test_lines),
        "sources": source_counts,
        "output_dir": str(out_dir),
    }


@app.local_entrypoint()
def main():
    result = assemble_hebrew_corpus.remote()
    print(json.dumps(result, indent=2, default=str))
