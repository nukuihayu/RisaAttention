"""Benchmark grouping and paired comparisons preserve experimental controls."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load("common")
report = load("report")


def row(mode, seed=0, stage="dense", distribution="normal"):
    return {
        "case": "B1-H4/1-L128/128-D64",
        "shape": {"head_dim": 64},
        "seed": seed,
        "layout": "hnd",
        "qk_distribution": "normal",
        "value_distribution": distribution,
        "implementation": "risa",
        "v_quantization": mode,
        "stage": stage,
        "timing": {"median_ms": 2 if mode == "absmax" else 3, "p10_ms": 1, "p90_ms": 4},
        "accuracy": {
            "nrmse": 0.1 if mode == "absmax" else 0.05,
            "rmse": 0.01,
            "sqnr_db": 20,
            "mae": 0.001,
            "max_abs": 0.05,
            "max_channel_bias": 0.003,
        },
        "memory": {"peak_incremental_mib": 4},
        "sparse": None,
    }


def test_affine_comparison_matches_seed_stage_and_distribution():
    rows = [
        row("absmax"),
        row("midpoint_affine"),
        row("absmax", seed=1),
        row("midpoint_affine", stage="sparse"),
        row("midpoint_affine", distribution="outlier"),
    ]
    result = common.affine_comparisons(rows)
    assert len(result) == 1
    assert result[0]["latency_change_percent"] == 50
    assert result[0]["nrmse_change_percent"] == -50


def test_zero_error_has_no_relative_change():
    rows = [row("absmax"), row("midpoint_affine")]
    for value in rows:
        value["accuracy"]["nrmse"] = 0
    assert common.affine_comparisons(rows)[0]["nrmse_change_percent"] is None


def test_affine_comparison_keeps_implementations_separate():
    before, after = row("absmax"), row("midpoint_affine")
    before["implementation"], after["implementation"] = "before", "after"
    assert common.affine_comparisons([before, after]) == []
    before_on, after_off = row("midpoint_affine"), row("absmax")
    before_on["implementation"], after_off["implementation"] = "before", "after"
    pairs = common.affine_comparisons([before, after, before_on, after_off])
    assert {pair["implementation"] for pair in pairs} == {"before", "after"}


def test_summary_rejects_duplicate_seeds():
    with pytest.raises(ValueError, match="duplicate seed"):
        report.summarize({"schema_version": 2, "results": [row("absmax"), row("absmax")]})


def test_summary_preserves_v_modes_and_stages():
    rows = [
        row("absmax", seed=0),
        row("absmax", seed=1),
        row("midpoint_affine"),
        row("absmax", stage="construct"),
    ]
    result = report.summarize({"schema_version": 2, "results": rows})
    assert len(result) == 3
    assert result[0]["seed_count"] == 2
    assert result[0]["median_ms"] == 2


def test_summary_preserves_unmeasured_memory():
    value = row("absmax")
    value["memory"] = None
    result = report.summarize({"schema_version": 2, "results": [value]})
    assert result[0]["peak_incremental_mib"] is None


def test_affine_table_keeps_before_and_after_controls():
    values = []
    for implementation in ("before", "after"):
        for mode in ("absmax", "midpoint_affine"):
            value = row(mode)
            value["implementation"] = implementation
            values.append(value)
    table = report.affine_table(report.summarize({"schema_version": 2, "results": values}))
    assert "| before |" in table and "| after |" in table
    assert len(table.splitlines()) == 4


def test_timing_alternates_methods_and_reports_per_call_samples(monkeypatch):
    from types import SimpleNamespace

    calls = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 20.0

    monkeypatch.setattr(common.torch.cuda, "Event", Event)
    monkeypatch.setattr(common.torch.cuda, "synchronize", lambda: None)
    args = SimpleNamespace(warmup=1, samples=2, calls_per_sample=2, guard_comfyui=None)
    result = common.measure_methods(
        {"off": lambda: calls.append("off"), "on": lambda: calls.append("on")}, args
    )
    assert calls == ["off", "on", "off", "off", "on", "on", "on", "on", "off", "off"]
    assert result["off"]["samples_ms"] == [10, 10]
    assert result["off"]["calls_per_sample"] == 2


def test_timing_balances_every_method_position(monkeypatch):
    from collections import Counter
    from types import SimpleNamespace

    calls = []
    names = ("before/off", "after/off", "before/on", "after/on")
    event = SimpleNamespace(
        record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 1.0
    )
    monkeypatch.setattr(common.torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(common.torch.cuda, "synchronize", lambda: None)
    args = SimpleNamespace(warmup=1, samples=8, calls_per_sample=1, guard_comfyui=None)
    common.measure_methods({name: lambda name=name: calls.append(name) for name in names}, args)
    measured = calls[len(names) :]
    for position in range(len(names)):
        assert Counter(measured[position :: len(names)]) == dict.fromkeys(names, 2)


@pytest.mark.parametrize(
    "value", ["0,4,1,128,128,64", "1,5,2,128,128,64", "1,4,1,128,128,257", "bad"]
)
def test_shape_rejects_invalid_cases(value):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        common.Shape.parse(value)
