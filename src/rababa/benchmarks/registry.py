"""Benchmark dataset registry.

A benchmark is a directory containing `test.txt` (or sharded
`test-NNN.txt`) in standard Tashkeela/Nakdimon format. The registry
maps a string name to a directory path; the runner builds a DataLoader
on demand.

Adding a new benchmark = call `register(name, path)`. No edits to
existing code (OCP).
"""

from __future__ import annotations

from pathlib import Path


class BenchmarkRegistry:
    """Registry of named benchmark datasets.

    Each entry maps a benchmark name to a directory Path. The directory
    must contain `{split}.txt` (or `{split}-NNN.txt` shards) in standard
    format. Benchmarks are added by external callers via `register()`,
    not hardcoded here.
    """

    def __init__(self) -> None:
        self._entries: dict[str, Path] = {}

    def register(self, name: str, root: Path) -> None:
        """Add or overwrite a benchmark entry."""
        self._entries[name] = Path(root)

    def get(self, name: str) -> Path | None:
        return self._entries.get(name)

    def names(self) -> list[str]:
        return sorted(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


REGISTRY = BenchmarkRegistry()


def _register_default_benchmarks() -> None:
    """Register benchmarks that ship with the repo (best-effort, idempotent).

    Paths are tried in order; first existing path wins. This runs once
    at module import. Additional benchmarks can be registered at any time.
    """
    candidates = [
        ("fadel", [
            Path("/opt/rababa/test-datasets/benchmarks/fadel"),
            Path("test-datasets/benchmarks/fadel"),
        ]),
        ("sadeed-diac-25", [
            Path("/opt/rababa/data/qcri-diac/benchmarks/sadeed-diac-25"),
            Path("test-datasets/benchmarks/sadeed-diac-25"),
        ]),
        ("in-domain-test", None),  # Special — uses task's own test split, no path
    ]
    for name, paths in candidates:
        if name in REGISTRY:
            continue
        if paths is None:
            continue
        for p in paths:
            if p.is_dir():
                REGISTRY.register(name, p)
                break


_register_default_benchmarks()
