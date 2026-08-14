"""Specs for benchmark harness."""

from __future__ import annotations

import torch
import torch.nn as nn

from rababa.benchmarks import (
    BenchmarkResult,
    REGISTRY,
    BenchmarkRegistry,
    run_all_benchmarks,
    run_benchmark,
)


class _StubModel(nn.Module):
    """Single-head model that always predicts class 0."""

    def forward_heads(self, src, lengths):
        B, T = src.shape
        V = 5  # small vocab
        return [torch.zeros(B, T, V)]

    def head_names(self):
        return ["output"]


def test_benchmark_registry_register_and_get():
    reg = BenchmarkRegistry()
    reg.register("test", "/path/to/test")
    assert "test" in reg
    assert reg.get("test") is not None
    assert reg.get("nope") is None


def test_benchmark_registry_names_sorted():
    reg = BenchmarkRegistry()
    reg.register("z_last", "/z")
    reg.register("a_first", "/a")
    assert reg.names() == ["a_first", "z_last"]


def test_run_benchmark_unregistered_returns_error_result():
    model = _StubModel()
    out = run_benchmark(
        model, task="rababa_arabic_pro",
        benchmark="does_not_exist",
        device=torch.device("cpu"),
    )
    assert isinstance(out, BenchmarkResult)
    assert out.error is not None
    assert "not registered" in out.error
    assert out.n_examples == 0


def test_run_all_benchmarks_includes_in_domain(monkeypatch):
    """`run_all_benchmarks` always appends 'in-domain-test' if not in list.

    We mock the in-domain loader so this test doesn't require a real
    dataset on disk.
    """
    model = _StubModel()

    # Stub the in-domain loader to return None (so the runner surfaces an error).
    def _stub_build_in_domain_loader(task, batch_size):
        return None

    import rababa.benchmarks.runner as runner
    monkeypatch.setattr(runner, "_build_in_domain_loader", _stub_build_in_domain_loader)

    results = run_all_benchmarks(
        model, task="rababa_arabic_pro",
        device=torch.device("cpu"),
        benchmarks=["in-domain-test"],
    )
    assert len(results) >= 1
    assert any(r.benchmark == "in-domain-test" for r in results)


def test_benchmark_result_to_dict_roundtrip():
    r = BenchmarkResult(
        benchmark="fadel", task="rababa_arabic_pro",
        n_examples=100, der=0.05, wer=0.20,
        per_head_der=[0.05],
    )
    d = r.to_dict()
    assert d["benchmark"] == "fadel"
    assert d["der"] == 0.05
    assert d["wer"] == 0.20
    assert d["per_head_der"] == [0.05]
    assert d["n_examples"] == 100
