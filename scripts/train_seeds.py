#!/usr/bin/env python3
"""Multi-seed training launcher.

Trains N copies of the same task in parallel on Modal via `.starmap()`.
Each seed writes to /checkpoints/{task}/run-{seed:03d}/best.pt.

Usage:
    python scripts/train_seeds.py --task rababa_arabic_pro --seeds 42,1337,2026
    python scripts/train_seeds.py --task rababa_arabic_pro --n-seeds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", required=True)
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seed values (default: 1,2,...,n-seeds)")
    p.add_argument("--n-seeds", type=int, default=3,
                   help="Number of seeds if --seeds not given")
    p.add_argument("--init-from-pretrain", default=None,
                   help="Pretrained encoder checkpoint path")
    args = p.parse_args(argv)

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = list(range(1, args.n_seeds + 1))

    print(f"Launching {len(seeds)} parallel trainings on Modal:")
    for s in seeds:
        print(f"  seed {s} → /checkpoints/{args.task}/run-{s:03d}/")

    try:
        import modal
    except ImportError:
        print("ERROR: modal not installed. Run `pip install modal`.", file=sys.stderr)
        return 1

    app_name = "rababa"
    try:
        train_fn = modal.Function.lookup(app_name, "train_with_seed")
    except Exception as e:
        print(f"ERROR: cannot find modal function 'train_with_seed'. "
              f"Did you `modal app deploy` first?\n  {e}", file=sys.stderr)
        return 1

    # Each seed gets its own checkpoint dir.
    ckpt_roots = [f"/checkpoints/{args.task}/run-{s:03d}" for s in seeds]
    results = list(train_fn.starmap([
        (args.task, seed, ckpt_root, args.init_from_pretrain)
        for seed, ckpt_root in zip(seeds, ckpt_roots)
    ]))
    print("All seeds completed:")
    for seed, result in zip(seeds, results):
        print(f"  seed {seed}: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
