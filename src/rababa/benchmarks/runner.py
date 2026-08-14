"""Benchmark runner — applies a trained model to named test sets.

Returns structured `BenchmarkResult` objects. Supports trie-constrained
decoding via an optional lexicon argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..constants import PAD_ID
from ..decoding.constrained import trie_constrained_decode
from ..evaluate import diacritization_error_rate
from ..tasks import SUPERVISED_DATASETS
from .registry import REGISTRY


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark."""
    benchmark: str
    task: str
    n_examples: int = 0
    der: float = 1.0
    wer: float = 1.0
    per_head_der: list[float] = field(default_factory=list)
    constrained: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "task": self.task,
            "n_examples": self.n_examples,
            "der": self.der,
            "wer": self.wer,
            "per_head_der": self.per_head_der,
            "constrained": self.constrained,
            **({"error": self.error} if self.error else {}),
            **self.extra,
        }


def build_benchmark_loader(
    task: str,
    benchmark: str,
    batch_size: int = 32,
    max_len: int = 200,
) -> DataLoader | None:
    """Build a test DataLoader for a named benchmark.

    Returns None if the benchmark isn't registered, OR if `benchmark`
    is the special value `"in-domain-test"` (caller should use the
    task's own test loader in that case).
    """
    if benchmark == "in-domain-test":
        return None
    root = REGISTRY.get(benchmark)
    if root is None:
        return None
    from ..config import load_task_config
    cfg = load_task_config(task)
    kind = cfg.kind
    if kind not in SUPERVISED_DATASETS:
        raise ValueError(f"unknown task kind: {kind!r}")
    loader_fn, collate = SUPERVISED_DATASETS[kind]
    cleaner = "hebrew" if "hebrew" in task else "arabic"
    if kind == "rababa":
        ds = loader_fn("test", root=root, cleaner=cleaner)
    else:
        ds = loader_fn("test", root=root, cleaner=cleaner, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)


def _build_in_domain_loader(task: str, batch_size: int) -> DataLoader | None:
    """Build the task's own test split as a fallback / baseline.

    Returns None if the test split isn't available locally (e.g. in unit
    tests, or when running outside Modal).
    """
    from ..tasks import build_test_loader
    try:
        return build_test_loader(task, batch_size=batch_size)
    except (FileNotFoundError, ValueError):
        return None


def run_benchmark(
    model: torch.nn.Module,
    task: str,
    benchmark: str,
    device: torch.device,
    lexicon: dict[str, list[list[int]]] | None = None,
    batch_size: int = 32,
    max_len: int = 200,
) -> BenchmarkResult:
    """Run a single benchmark against a model.

    Args:
        model: trained diacritization model (single- or multi-head).
        task: task name (e.g. `rababa_arabic_pro`).
        benchmark: registered benchmark name (or `"in-domain-test"`).
        device: torch device for inference.
        lexicon: optional `{word: [haraqat_seqs]}` for trie-constrained
            decoding on the first head. Other heads use argmax.
        batch_size, max_len: DataLoader config.
    """
    if benchmark == "in-domain-test":
        loader = _build_in_domain_loader(task, batch_size)
        if loader is None:
            return BenchmarkResult(
                benchmark=benchmark, task=task,
                error=f"in-domain test split not available for task {task!r}",
                constrained=lexicon is not None,
            )
    else:
        loader = build_benchmark_loader(task, benchmark, batch_size, max_len)
        if loader is None:
            return BenchmarkResult(
                benchmark=benchmark, task=task,
                error=f"benchmark {benchmark!r} not registered",
                constrained=lexicon is not None,
            )

    model.eval()
    head_names = model.head_names() if hasattr(model, "head_names") else ["output"]
    head_der_acc = [0.0] * len(head_names)
    head_n = [0] * len(head_names)
    aggregate_wrong = 0
    aggregate_total = 0
    exact_match_correct = 0
    total_n = 0

    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            targets = [t.to(device) for t in batch.targets]
            outputs = model.forward_heads(src, lengths)
            any_wrong: torch.Tensor | None = None
            any_evaluable: torch.Tensor | None = None
            for h_idx, (logits, target) in enumerate(zip(outputs, targets, strict=True)):
                # Trie constraint applies only to head 0 (the primary diacritization head).
                head_lex = lexicon if h_idx == 0 else None
                der = diacritization_error_rate(logits, target, src=src, lexicon=head_lex)
                head_der_acc[h_idx] += der * src.size(0)
                head_n[h_idx] += src.size(0)
                preds = (
                    trie_constrained_decode(logits, src, head_lex)
                    if head_lex is not None
                    else logits.argmax(dim=-1)
                )
                head_mask = target != PAD_ID
                head_wrong = (preds != target) & head_mask
                any_wrong = head_wrong if any_wrong is None else (any_wrong | head_wrong)
                any_evaluable = head_mask if any_evaluable is None else (any_evaluable | head_mask)
            aggregate_wrong += int(any_wrong.sum().item())
            aggregate_total += int(any_evaluable.sum().item())
            for i in range(src.size(0)):
                if not any_wrong[i].any():
                    exact_match_correct += 1
            total_n += src.size(0)

    return BenchmarkResult(
        benchmark=benchmark,
        task=task,
        n_examples=total_n,
        der=aggregate_wrong / max(1, aggregate_total),
        wer=1.0 - (exact_match_correct / max(1, total_n)),
        per_head_der=[d / max(1, n) for d, n in zip(head_der_acc, head_n)],
        constrained=lexicon is not None,
    )


def run_all_benchmarks(
    model: torch.nn.Module,
    task: str,
    device: torch.device,
    benchmarks: list[str] | None = None,
    lexicon: dict[str, list[list[int]]] | None = None,
) -> list[BenchmarkResult]:
    """Run multiple benchmarks. Defaults to all registered + in-domain."""
    names = list(benchmarks or REGISTRY.names())
    if "in-domain-test" not in names:
        names.append("in-domain-test")
    return [run_benchmark(model, task, name, device, lexicon=lexicon) for name in names]
