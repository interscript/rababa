#!/usr/bin/env python3
"""Local DictaBERT distillation — runs on CPU, no Modal preemption.

Generates diacritized Hebrew training data from Hebrew Wikipedia
using the SOTA DictaBERT model. Output feeds into ByT5 retraining.

Usage:
    source .venv-dictabert/bin/activate
    python distill_local.py
"""

import sys
import time
from pathlib import Path

def main():
    print("=== Local DictaBERT Hebrew Distillation ===", flush=True)

    # 1. Load DictaBERT
    print("Loading DictaBERT...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    import transformers
    print(f"transformers version: {transformers.__version__}", flush=True)

    model_name = "dicta-il/dictabert-large-char-menaked"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print(f"Model loaded: {type(model).__name__}", flush=True)

    # 2. Smoke test
    test = "בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת"
    print(f"\nSmoke test:", flush=True)
    print(f"  Input:  {test}", flush=True)
    result = model.predict([test], tokenizer)
    print(f"  Output: {result[0] if result else 'EMPTY'}", flush=True)

    # 3. Load Hebrew Wikipedia
    hewiki_path = Path("data/hewiki/train.txt")
    if not hewiki_path.is_file():
        print(f"ERROR: {hewiki_path} not found", flush=True)
        sys.exit(1)

    lines = []
    for line in hewiki_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if 10 <= len(line) <= 200:
            lines.append(line)
        if len(lines) >= 10000:
            break

    print(f"\nProcessing {len(lines)} Hebrew Wikipedia lines on CPU...", flush=True)

    # 4. Distill
    out_dir = Path("data/hebrew-dictabert-distilled")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.txt"
    checkpoint_path = out_dir / "progress.txt"

    # Resume from checkpoint if exists
    start_idx = 0
    if checkpoint_path.is_file():
        start_idx = int(checkpoint_path.read_text().strip())
        print(f"Resuming from line {start_idx}", flush=True)

    batch_size = 4
    count = start_idx
    t_start = time.time()

    mode = "a" if start_idx > 0 else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for i in range(start_idx, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            try:
                predictions = model.predict(batch, tokenizer)
            except Exception as e:
                print(f"  Error at batch {i}: {e}", flush=True)
                predictions = batch

            for pred in predictions:
                if pred and pred.strip():
                    f.write(pred.strip() + "\n")
                    f.flush()
                    count += 1

            # Checkpoint every 50 lines
            if (count % 50) == 0:
                checkpoint_path.write_text(str(count), encoding="utf-8")
                elapsed = time.time() - t_start
                rate = (count - start_idx) / max(1, elapsed)
                remaining = (len(lines) - count) / max(0.1, rate)
                print(f"  [{count}/{len(lines)}] {rate:.1f} lines/s, "
                      f"ETA: {remaining/60:.0f} min", flush=True)

    checkpoint_path.write_text(str(count), encoding="utf-8")
    elapsed = time.time() - t_start
    print(f"\nDone! {count} lines in {elapsed/60:.1f} min", flush=True)
    print(f"Output: {out_path}", flush=True)
    print(f"Rate: {count/max(1,elapsed):.1f} lines/sec", flush=True)

    # 5. Show sample output
    all_lines = out_path.read_text(encoding="utf-8").splitlines()
    print(f"\nSample distilled lines:", flush=True)
    for line in all_lines[-5:]:
        print(f"  {line[:80]}", flush=True)


if __name__ == "__main__":
    main()
