"""In-memory benchmark harness — separate from `benchmark.py` (ONNX-only).

This subpackage:
  - Registers named benchmark datasets (Fadel, SadeedDiac-25, etc.)
  - Runs them against a trained torch model (pre-export)
  - Supports trie-constrained decoding at evaluation
  - Returns structured `BenchmarkResult` objects

Design (OCP): new benchmarks = call `REGISTRY.register(name, path)`,
no edits to existing files.
"""

from .registry import REGISTRY, BenchmarkRegistry
from .runner import BenchmarkResult, build_benchmark_loader, run_all_benchmarks, run_benchmark

__all__ = [
    "BenchmarkRegistry",
    "BenchmarkResult",
    "REGISTRY",
    "build_benchmark_loader",
    "run_all_benchmarks",
    "run_benchmark",
]
