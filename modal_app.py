"""Modal app for rababa training + export + evaluation (Arabic + Hebrew).

Both languages go through the same `train_supervised`, `pretrain_mlm`,
`export_student_onnx` functions. Task dispatch (dataset + collate) lives
in `rababa.tasks`, model dispatch (single vs multi head) in
`rababa.models.base.build_model`.

Usage:
    # First-time auth:
    modal token new

    # Test connection + dataset fetch:
    modal run modal_app.py::fetch_data --task rababa_arabic
    modal run modal_app.py::fetch_data --task rababa_hebrew

    # MLM pretrain (A100, ~6h):
    modal run modal_app.py::pretrain --task rababa_arabic_pretrain
    modal run modal_app.py::pretrain --task rababa_hebrew_pretrain

    # Train (A100, ~3h), optionally with pretrained encoder init:
    modal run modal_app.py::train --task rababa_arabic \\
        --init-from-pretrain /checkpoints/rababa_arabic_pretrain/run-001/best.pt
    modal run modal_app.py::train --task rababa_hebrew

    # Export to ONNX + int8 (A10G, ~30m):
    modal run modal_app.py::export_onnx --task rababa_arabic --version v0.1.0
    modal run modal_app.py::export_onnx --task rababa_hebrew --version v0.1.0

    # Evaluate (A10G):
    modal run modal_app.py::evaluate --task rababa_arabic

Volumes:
    datasets     — fetched Tashkeela / Nakdimon corpora (idempotent).
    checkpoints  — per-epoch + best.pt model weights.
    models       — final ONNX exports.
"""

from __future__ import annotations

import modal
from pathlib import Path

APP_NAME = "rababa"
PYTHON_VERSION = "3.11"

datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
models_volume = modal.Volume.from_name(f"{APP_NAME}-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("build-essential", "git")
    .pip_install(
        "torch>=2.4,<3",
        "numpy>=1.26,<3",
        "omegaconf>=2.3,<3",
        "onnx>=1.17",
        "onnxscript>=0.1",
        "onnxruntime>=1.20",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "wandb>=0.18",
        "transformers>=4.46",
        "datasets>=3.0",
        "litert-torch>=0.9",
        "ai-edge-quantizer>=0.8",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .add_local_dir("configs", "/opt/rababa/configs", copy=True)
    .add_local_dir("test-datasets", "/opt/rababa/test-datasets", copy=True)
    .add_local_file("pyproject.toml", "/opt/rababa/pyproject.toml", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
    # ---- Data repos baked in at build time (single source of truth = git) ----
    # Each clone is --depth 1 to keep image size minimal. To update the corpus,
    # bump the commit SHA via a no-op commit + push to the source repo, which
    # invalidates Modal's image cache via the .add_local_dir hash on this file.
    .run_commands(
        "git clone --depth 1 https://github.com/interscript/rababa-tashkeela.git /opt/rababa/data/tashkeela",
        "git clone --depth 1 https://github.com/interscript/rababa-tashkeela-full.git /opt/rababa/data/tashkeela-full",
        "git clone --depth 1 https://github.com/interscript/rababa-arwiki.git /opt/rababa/data/arwiki",
        "git clone --depth 1 https://github.com/interscript/rababa-sefaria.git /opt/rababa/data/sefaria",
        "git clone --depth 1 https://github.com/interscript/rababa-hewiki.git /opt/rababa/data/hewiki",
        "git clone --depth 1 https://github.com/interscript/rababa-hebrew-distilled.git /opt/rababa/data/hebrew-distilled",
        # EMNLP 2025 QCRI advancing-arabic-diacritization — refined datasets + SadeedDiac-25 benchmark.
        "git clone --depth 1 https://github.com/qcri/advancing-arabic-diacritization.git /opt/rababa/data/qcri-diac",
    )
)

app = modal.App(name=APP_NAME, image=image)


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def fetch_data(task: str) -> dict[str, object]:
    """Verify data is present and assemble combined Hebrew corpus if needed.

    Data repos are baked into the Modal image at build time (git clone in
    the image recipe). This function just verifies presence and, for Hebrew
    tasks, concatenates Sefaria + distilled into a combined train/val/test.

    The /datasets volume mount is kept for backwards compatibility with
    checkpoint/model volumes but is no longer the source of truth — git is.
    """
    import hashlib
    from pathlib import Path

    summary: dict[str, object] = {"task": task, "files": {}}

    if task in {"rababa_arabic", "rababa_arabic_pretrain"}:
        # Tashkeela is shipped with the repo at /opt/rababa/test-datasets/tashkeela.
        root = Path("/opt/rababa/test-datasets/tashkeela")
    elif task in {"rababa_arabic_pro", "rababa_arabic_pro_pretrain"}:
        # Merged corpus: GPLv2 Tashkeela-full + Sadeed HF + QCRI EMNLP 2025.
        # Built on first call, cached on the /datasets volume for re-use.
        root = Path("/datasets/arabic-combined")
        if not (root / "train.txt").is_file():
            print(f"[fetch_data] building combined Arabic corpus at {root} ...")
            _build_arabic_combined_corpus(root)
        else:
            print(f"[fetch_data] combined Arabic corpus already present at {root}")
    elif task in {"rababa_hebrew", "rababa_hebrew_pretrain"}:
        # Assemble combined Hebrew corpus from Sefaria (Biblical) + distilled (Modern).
        sefaria = Path("/opt/rababa/data/sefaria")
        distilled = Path("/opt/rababa/data/hebrew-distilled")
        combined = Path("/opt/rababa/data/nakdimon-combined")
        combined.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            parts = []
            for src_repo, subdir_prefix in (
                (sefaria, "sefaria"),
                (distilled, "hebrew_distilled"),
            ):
                # Try both naming conventions.
                for name in (f"{split}.txt", f"{subdir_prefix}_{split}/{split}.txt"):
                    p = src_repo / name
                    if p.is_file():
                        parts.append(p.read_text(encoding="utf-8"))
                        break
            (combined / f"{split}.txt").write_text("".join(parts), encoding="utf-8")
        root = combined
    else:
        raise ValueError(f"fetch_data for {task!r} not implemented")

    for split in ("train", "val", "test"):
        path = root / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing {split}: {path}")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        line_count = sum(1 for _ in path.open(encoding="utf-8"))
        summary["files"][split] = {"path": str(path), "sha256": sha, "lines": line_count}
    return summary


def _iter_lines(path: Path):
    """Yield stripped non-empty lines from a file."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield line


def _iter_corpus_files(root: Path, split: str) -> list[Path]:
    """Find files for a split under root, handling sharded + legacy layouts.

    Looks for: {split}-*.txt (sharded), {split}.txt (legacy), and any
    *.txt under a subdir named like the split.
    """
    shards = sorted(root.glob(f"{split}-*.txt"))
    if shards:
        return shards
    legacy = root / f"{split}.txt"
    if legacy.is_file():
        return [legacy]
    # Subdir layout: root/train/whatever.txt
    subdir = root / split
    if subdir.is_dir():
        return sorted(subdir.glob("*.txt"))
    return []


def _maybe_download_sadeed_hf(dest_dir: Path) -> bool:
    """Download Misraj/Sadeed_Tashkeela from HuggingFace if HF_TOKEN is set.

    Returns True if the dataset was downloaded and written to
    dest_dir/{train,val,test}.txt; False if HF_TOKEN is unset or the
    download failed (we fall back to Tashkeela + QCRI only).

    Output format: one diacritized Arabic line per row (the `output`
    field of the HF dataset). Lines are deduplicated within each split.
    """
    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[sadeed-hf] HF_TOKEN not set — skipping Sadeed HF download")
        return False
    try:
        from datasets import load_dataset
        print("[sadeed-hf] downloading Misraj/Sadeed_Tashkeela ...")
        ds = load_dataset("Misraj/Sadeed_Tashkeela", token=token)
    except Exception as e:
        print(f"[sadeed-hf] download failed: {e!r} — skipping")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    # Sadeed_Tashkeela has train + test splits. Carve 10% of train as val.
    splits_present = list(ds.keys())
    print(f"[sadeed-hf] splits present: {splits_present}")
    train_ds = ds["train"] if "train" in splits_present else ds[splits_present[0]]
    test_ds = ds.get("test") or ds.get(splits_present[-1])
    train_val_split = train_ds.train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = train_val_split["train"], train_val_split["test"]

    for name, subset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        out = dest_dir / f"{name}.txt"
        seen: set[str] = set()
        with out.open("w", encoding="utf-8") as f:
            for ex in subset:
                line = (ex.get("output") or "").strip()
                if not line or line in seen:
                    continue
                seen.add(line)
                f.write(line + "\n")
        print(f"[sadeed-hf] {name}.txt: {len(seen):,} unique lines")
    return True


def _find_qcri_files(root: Path) -> dict[str, Path]:
    """Locate train/val/test files in the qcri/advancing-arabic-diacritization repo.

    The repo layout is not documented up-front; we search broadly. Each
    split is the first .txt file whose path contains the split keyword.
    """
    out: dict[str, Path] = {}
    all_txt = sorted(root.rglob("*.txt"))
    for split in ("train", "val", "test"):
        for p in all_txt:
            path_str = str(p).lower()
            # Acceptable: 'train.txt', 'train-001.txt', 'train_split.txt',
            # subdir named train/anything.txt
            if split in path_str or f"/{split}/" in path_str or f"-{split}" in path_str:
                # Avoid train/test mixups: ensure the split keyword is the
                # strongest signal in the path.
                if split == "val" and "val" not in p.stem.lower():
                    continue
                out[split] = p
                break
    return out


def _build_arabic_combined_corpus(dest_root: Path) -> None:
    """Merge GPLv2 Tashkeela + Sadeed HF + QCRI EMNLP 2025 into dest_root.

    Output: dest_root/{train,val,test}.txt — one diacritized Arabic line
    per row, deduplicated across all sources.

    Sources:
      1. /opt/rababa/data/tashkeela-full (GPLv2, image-baked)
      2. /datasets/sadeed-hf/ (HF, gated, downloaded if HF_TOKEN set)
      3. /opt/rababa/data/qcri-diac/ (CC BY-NC-SA, image-baked)

    Graceful fallback: missing sources are skipped with a warning.
    """
    dest_root.mkdir(parents=True, exist_ok=True)

    # 1. Sadeed HF — gated, needs token. Best-effort; skipped if no token.
    sadeed_root = Path("/datasets/sadeed-hf")
    if not (sadeed_root / "train.txt").is_file():
        _maybe_download_sadeed_hf(sadeed_root)

    # 2. QCRI EMNLP 2025 — image-baked.
    qcri_root = Path("/opt/rababa/data/qcri-diac")
    qcri_files = _find_qcri_files(qcri_root) if qcri_root.is_dir() else {}
    if not qcri_files:
        print(f"[combined] WARNING: no QCRI files under {qcri_root}")
    else:
        print(f"[combined] QCRI files: {qcri_files}")

    # 3. GPLv2 Tashkeela-full — image-baked. Primary source.
    tashkeela_full = Path("/opt/rababa/data/tashkeela-full")
    if not tashkeela_full.is_dir():
        raise RuntimeError(
            f"tashkeela-full missing at {tashkeela_full} — image recipe is wrong"
        )

    sources: list[tuple[str, Path]] = [("tashkeela-full", tashkeela_full)]
    if (sadeed_root / "train.txt").is_file():
        sources.append(("sadeed-hf", sadeed_root))
    if qcri_files:
        # Synthetic root whose _iter_corpus_files returns the explicit paths.
        # Easier: handle QCRI separately below.
        pass

    for split in ("train", "val", "test"):
        seen: set[str] = set()
        out_path = dest_root / f"{split}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            # GPLv2 Tashkeela-full (sharded)
            for src_name, src_root in sources:
                files = _iter_corpus_files(src_root, split)
                if not files:
                    print(f"[combined]   {split}/{src_name}: no files")
                    continue
                count = 0
                for fp in files:
                    for line in _iter_lines(fp):
                        if line not in seen:
                            seen.add(line)
                            f.write(line + "\n")
                            count += 1
                print(f"[combined]   {split}/{src_name}: +{count:,} unique lines")
            # QCRI (custom find)
            if qcri_files and split in qcri_files:
                count = 0
                for line in _iter_lines(qcri_files[split]):
                    if line not in seen:
                        seen.add(line)
                        f.write(line + "\n")
                        count += 1
                print(f"[combined]   {split}/qcri: +{count:,} unique lines")
        print(f"[combined] {split}.txt: {len(seen):,} total unique lines")


def _fetch_nakdimon_corpus(dest: Path) -> None:
    """Clone Nakdimon repo and assemble a train/val/test split from the
    open test corpus.

    The Nakdimon *training* corpus is Dicta-licensed and not redistributable,
    so we use the open test corpus (`tests/new/expected/`) — 110 files
    across 10 categories (books, wiki, verdicts, etc.). We split 80/10/10
    into train/val/test by file. This is methodologically impure (training
    on what was meant to be a test set) but produces a usable v0.1.0
    preview. v0.5.0 should switch to a proper Hebrew corpus
    (Wikisource nikud, Open Scriptures Hebrew).
    """
    import random
    import shutil
    import subprocess
    import tempfile

    nakdimon_url = "https://github.com/elazarg/nakdimon.git"
    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "nakdimon"
        subprocess.run(
            ["git", "clone", "--depth", "1", nakdimon_url, str(clone_dir)],
            check=True,
        )
        # Collect all test files (pointed Hebrew, one line per row).
        test_root = clone_dir / "tests" / "new" / "expected"
        all_files: list[Path] = []
        for category_dir in sorted(test_root.iterdir()):
            if category_dir.is_dir():
                all_files.extend(sorted(category_dir.glob("*.txt")))
        if not all_files:
            raise RuntimeError(
                f"No Hebrew test files found under {test_root}. "
                "Nakdimon repo layout may have changed."
            )

        # Deterministic 80/10/10 split by file.
        rng = random.Random(42)
        rng.shuffle(all_files)
        n = len(all_files)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_files = all_files[:n_train]
        val_files = all_files[n_train : n_train + n_val]
        test_files = all_files[n_train + n_val :]

        dest.mkdir(parents=True, exist_ok=True)
        for split, files in (
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ):
            out = dest / f"{split}.txt"
            with out.open("w", encoding="utf-8") as f:
                for src in files:
                    f.write(src.read_text(encoding="utf-8"))
            line_count = sum(1 for _ in out.open(encoding="utf-8"))
            print(f"  {split}.txt: {len(files)} files, {line_count} lines")


@app.function(
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def train(
    task: str,
    epochs: int | None = None,
    init_from_pretrain: str | None = None,
) -> dict[str, object]:
    """Run Tier 1 supervised training. Returns path to best checkpoint.

    Dispatches dataset/collate via `rababa.tasks`; model via cfg.model.arch.
    Works for rababa_arabic (single-head) and rababa_hebrew (multi-head).
    """
    import torch

    from rababa.config import load_task_config, to_dict
    from rababa.tasks import build_supervised_loaders
    from rababa.training import train_supervised

    cfg = load_task_config(task)
    if epochs is not None:
        cfg.train.epochs = epochs
    if init_from_pretrain is not None:
        cfg.train.init_from_pretrain = init_from_pretrain

    train_loader, val_loader = build_supervised_loaders(cfg)

    device = torch.device("cuda")
    ckpt_root = Path("/checkpoints") / task / "run-001"
    train_supervised(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=to_dict(cfg),
        device=device,
        ckpt_root=ckpt_root,
    )
    checkpoints_volume.commit()
    return {"checkpoint_root": str(ckpt_root), "best": str(ckpt_root / "best.pt")}


@app.function(
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def pretrain(task: str, epochs: int | None = None) -> dict[str, object]:
    """Run MLM pretraining. Returns path to best encoder checkpoint."""
    import torch

    from rababa.config import load_task_config, to_dict
    from rababa.tasks import build_mlm_loaders
    from rababa.training import pretrain_mlm

    cfg = load_task_config(task)
    if epochs is not None:
        cfg.train.epochs = epochs

    train_loader, val_loader = build_mlm_loaders(cfg)

    device = torch.device("cuda")
    ckpt_root = Path("/checkpoints") / task / "run-001"
    pretrain_mlm(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=to_dict(cfg),
        device=device,
        ckpt_root=ckpt_root,
    )
    checkpoints_volume.commit()
    return {"checkpoint_root": str(ckpt_root), "best": str(ckpt_root / "best.pt")}


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/models": models_volume},
)
def export_onnx(task: str, version: str, checkpoint: str | None = None) -> dict[str, object]:
    """Export checkpoint → ONNX fp32 + int8. Handles single- and multi-head."""
    from rababa.config import load_task_config, to_dict
    from rababa.export import export_student_onnx, quantize_dynamic_int8

    cfg = load_task_config(task)
    cfg_dict = to_dict(cfg)  # type: ignore[arg-type]
    batch_size = int(cfg.model.get("batch_size", 32))
    max_len = int(cfg.model.get("max_len", 200))

    if checkpoint is None:
        checkpoint = str(Path("/checkpoints") / task / "run-001" / "best.pt")

    out_dir = Path("/models") / task
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / f"{task}-{version}-fp32.onnx"
    q8_path = out_dir / f"{task}-{version}-q8.onnx"

    export_student_onnx(Path(checkpoint), cfg_dict, fp32_path, batch_size, max_len)
    quantize_dynamic_int8(fp32_path, q8_path)

    models_volume.commit()
    return {"fp32": str(fp32_path), "q8": str(q8_path)}


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/models": models_volume},
)
def export_tflite(task: str, version: str, checkpoint: str | None = None) -> dict[str, object]:
    """Export checkpoint → TFLite (.tflite) for LiteRT.js browser runtime.

    Same model architecture, same I/O contract — different serialization
    format. fp32 only for v0.1.0; int8 (PT2E) is a follow-up.
    """
    from rababa.config import load_task_config, to_dict
    from rababa.export_tflite import export_student_tflite

    cfg = load_task_config(task)
    cfg_dict = to_dict(cfg)  # type: ignore[arg-type]
    batch_size = int(cfg.model.get("batch_size", 32))
    max_len = int(cfg.model.get("max_len", 200))

    if checkpoint is None:
        checkpoint = str(Path("/checkpoints") / task / "run-001" / "best.pt")

    out_dir = Path("/models") / task
    out_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = out_dir / f"{task}-{version}-fp32.tflite"

    export_student_tflite(Path(checkpoint), cfg_dict, tflite_path, batch_size, max_len)

    models_volume.commit()
    return {"tflite": str(tflite_path)}


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate(task: str, checkpoint: str | None = None) -> dict[str, object]:
    """Compute per-head DER + aggregate DER on test split.

    Uses the unified Diacritizer protocol — same code path for Arabic (1 head)
    and Hebrew (3 heads).
    """
    import torch

    from rababa.config import load_task_config, to_dict
    from rababa.evaluate import diacritization_error_rate, per_example_accuracy
    from rababa.models.base import build_model
    from rababa.tasks import build_test_loader

    cfg = load_task_config(task)
    cfg_dict = to_dict(cfg)  # type: ignore[arg-type]
    device = torch.device("cuda")

    if checkpoint is None:
        checkpoint = str(Path("/checkpoints") / task / "run-001" / "best.pt")

    model = build_model(cfg_dict).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    head_names = model.head_names()
    loader = build_test_loader(task=task, batch_size=32)

    head_der = [0.0] * len(head_names)
    head_acc = [0.0] * len(head_names)
    aggregate_wrong = 0
    aggregate_total = 0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            targets = [t.to(device) for t in batch.targets]
            outputs = model.forward_heads(src, lengths)
            any_wrong = None
            any_evaluable = None
            for h_idx, (logits, target) in enumerate(zip(outputs, targets, strict=True)):
                head_der[h_idx] += diacritization_error_rate(logits, target) * src.size(0)
                head_acc[h_idx] += per_example_accuracy(logits, target) * src.size(0)
                preds = logits.argmax(dim=-1)
                head_mask = target != 0
                head_wrong = (preds != target) & head_mask
                any_wrong = head_wrong if any_wrong is None else (any_wrong | head_wrong)
                any_evaluable = head_mask if any_evaluable is None else (any_evaluable | head_mask)
            aggregate_wrong += any_wrong.sum().item()
            aggregate_total += any_evaluable.sum().item()
            total_n += src.size(0)

    result = {
        "task": task,
        "checkpoint": checkpoint,
        "head_names": head_names,
        "n_examples": total_n,
        "per_head_der": [d / max(1, total_n) for d in head_der],
        "per_head_per_example_accuracy": [a / max(1, total_n) for a in head_acc],
        "der_aggregate": aggregate_wrong / max(1, aggregate_total),
        "der": aggregate_wrong / max(1, aggregate_total),
        "per_example_accuracy": head_acc[0] / max(1, total_n),
    }
    # Print so the result is visible in `modal run` stdout (not just returned).
    import json
    print("=== evaluate result ===")
    print(json.dumps(result, indent=2, default=str))
    return result


# ---- Distillation: auto-label unpointed Hebrew via Dicta Nakdan API ----

DICTA_URL = "https://nakdan-2-0.loadbalancer.dicta.org.il/api"


@app.function(
    cpu=2,
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets_volume},
)
def distill_hebrew_chunk(chunk_index: int, total_chunks: int, source_path: str) -> dict[str, object]:
    """Process one chunk of unpointed Hebrew lines via Dicta Nakdan API.

    Designed to be called via `.starmap()` so N chunks run in parallel
    across N containers. Each container handles 1/N of the input.
    """
    import requests
    from pathlib import Path

    src = Path(source_path)
    all_lines = src.read_text(encoding="utf-8").splitlines()
    n = len(all_lines)
    chunk_size = (n + total_chunks - 1) // total_chunks
    start = chunk_index * chunk_size
    end = min(start + chunk_size, n)
    chunk = all_lines[start:end]

    # Write pointed output to a per-chunk file; merged later.
    out_path = Path("/datasets") / "hebrew-distilled" / f"chunk-{chunk_index:04d}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"chunk": chunk_index, "total": 0, "kept": 0, "low_confidence_words": 0, "failed": 0}

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, line in enumerate(chunk):
            line = line.strip()
            if not line or len(line) < 10:
                continue
            stats["total"] += 1
            try:
                resp = requests.post(
                    DICTA_URL,
                    json={"data": line, "genre": "modern"},
                    headers={"Content-Type": "application/json"},
                    timeout=15,
                )
                resp.raise_for_status()
                words = resp.json()

                pointed_parts = []
                for w in words:
                    if w.get("sep"):
                        pointed_parts.append(w["word"])
                        continue
                    options = w.get("options") or []
                    if not options:
                        pointed_parts.append(w["word"])
                        continue
                    if not w.get("fconfident", False):
                        stats["low_confidence_words"] += 1
                    # Always take top prediction — Dicta is reliable even when
                    # fconfident=false (the flag is conservative). Let the
                    # downstream student model learn from any residual noise.
                    pointed_parts.append(options[0])

                pointed_line = "".join(pointed_parts).strip()
                if pointed_line:
                    out_f.write(pointed_line + "\n")
                    stats["kept"] += 1
            except Exception:
                stats["failed"] += 1

            if (i + 1) % 500 == 0:
                print(f"  chunk {chunk_index}: {i + 1}/{len(chunk)} kept={stats['kept']}", flush=True)
                out_f.flush()

    datasets_volume.commit()
    return stats


@app.function(
    cpu=1,
    timeout=10 * 60,
    volumes={"/datasets": datasets_volume},
)
def merge_distilled_chunks(n_chunks: int, out_path: str = "/datasets/hebrew-distilled/train.txt") -> dict[str, object]:
    """Concatenate all chunk files into a single train.txt."""
    from pathlib import Path

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    chunks_used = 0
    with out.open("w", encoding="utf-8") as out_f:
        for i in range(n_chunks):
            chunk_file = Path("/datasets") / "hebrew-distilled" / f"chunk-{i:04d}.txt"
            if not chunk_file.is_file():
                continue
            text = chunk_file.read_text(encoding="utf-8")
            out_f.write(text)
            if not text.endswith("\n"):
                out_f.write("\n")
            total_lines += text.count("\n")
            chunks_used += 1
    datasets_volume.commit()
    return {"chunks_used": chunks_used, "total_lines": total_lines, "out_path": out_path}


@app.function(
    cpu=1,
    timeout=4 * 60 * 60,
    volumes={"/datasets": datasets_volume},
)
def distill_hebrew(
    source_path: str = "/hewiki/train.txt",
    n_parallel: int = 20,
    commit_to_repo: bool = False,
) -> dict[str, object]:
    """Top-level entry: dispatch N parallel containers, then merge.

    Returns aggregate stats. Output: /datasets/hebrew-distilled/train.txt

    Designed for `modal app deploy` invocation — runs entirely server-side
    with a 4-hour timeout (the 300s client RPC limit only applies to
    `modal run`). Use `modal app deploy` then `modal app call` for the
    long-running path; use `modal run` only for small smoke tests.

    Set commit_to_repo=True to push the distilled corpus back to the
    rababa-hebrew-distilled GitHub repo so future builds pick it up
    via the image-recipe git clone.
    """
    from pathlib import Path

    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source corpus not found: {src}")

    # Dispatch chunks in parallel.
    chunk_indices = list(range(n_parallel))
    print(f"Dispatching {n_parallel} parallel workers on {source_path}...", flush=True)
    stats_list = list(distill_hebrew_chunk.starmap(
        [(i, n_parallel, source_path) for i in chunk_indices],
    ))

    # Merge.
    print(f"Merging {len(stats_list)} chunk outputs...", flush=True)
    merge_result = merge_distilled_chunks.remote(n_parallel)

    totals = {"total": sum(s["total"] for s in stats_list),
              "kept": sum(s["kept"] for s in stats_list),
              "low_confidence_words": sum(s["low_confidence_words"] for s in stats_list),
              "failed": sum(s["failed"] for s in stats_list)}
    print(f"=== distill_hebrew result ===")
    import json
    print(json.dumps({"per_chunk_stats": stats_list[:5], "totals": totals, "merge": merge_result},
                     indent=2, default=str))

    if commit_to_repo:
        _commit_distilled_to_repo(merge_result["total_lines"])

    return {"totals": totals, "merge": merge_result}


def _commit_distilled_to_repo(n_lines: int) -> None:
    """Push the distilled corpus to rababa-hebrew-distilled GitHub repo.

    Called from inside the container after a successful distillation run.
    Requires GH_TOKEN env var to be set (Modal Secret `github-token`).
    """
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("WARNING: no GH_TOKEN — skipping repo commit. Distilled data stays on volume.")
        return

    repo_url = f"https://x-access-token:{token}@github.com/interscript/rababa-hebrew-distilled.git"
    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "repo"
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(clone_dir)], check=True)

        # Copy merged train.txt into the repo's data subdir.
        data_dir = clone_dir / "hebrew_distilled_train"
        data_dir.mkdir(exist_ok=True)
        src = Path("/datasets/hebrew-distilled/train.txt")
        dst = data_dir / "train.txt"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        # Commit + push to a branch + open PR.
        branch = f"distill-{n_lines}-lines"
        subprocess.run(["git", "-C", str(clone_dir), "checkout", "-b", branch], check=True)
        subprocess.run(["git", "-C", str(clone_dir), "add",
                        "hebrew_distilled_train/train.txt"], check=True)
        subprocess.run(["git", "-C", str(clone_dir), "commit",
                        "-m", f"Distill {n_lines:,} lines via Dicta API"], check=True)
        subprocess.run(["git", "-C", str(clone_dir), "push", "-u", "origin", branch],
                       check=True)
        subprocess.run([
            "gh", "pr", "create", "--repo", "interscript/rababa-hebrew-distilled",
            "--title", f"Distilled Hebrew corpus ({n_lines:,} lines)",
            "--body", "Auto-generated by modal_app.distill_hebrew.",
            "--head", branch,
        ], check=True)


@app.local_entrypoint()
def distill_hebrew_entrypoint(
    source_path: str = "/hewiki/train.txt",
    n_parallel: int = 20,
    commit_to_repo: bool = False,
):
    """Fire-and-forget entrypoint for large Hebrew distillation runs.

    Usage:
        modal app deploy modal_app                          # one-time
        modal app call rababa/distill_hebrew_entrypoint \\
            --source-path /hewiki/train.txt \\
            --n-parallel 40 \\
            --commit-to-repo

    The entrypoint itself runs as a thin client; the heavy lifting is on
    `distill_hebrew` which has the 4-hour container timeout.
    """
    return distill_hebrew.remote(
        source_path=source_path,
        n_parallel=n_parallel,
        commit_to_repo=commit_to_repo,
    )
