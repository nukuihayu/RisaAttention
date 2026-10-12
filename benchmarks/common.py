# SPDX-License-Identifier: Apache-2.0
"""Shared inputs, paired timing, numerical metrics and benchmark serialization."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

import risa_attention as risa

ROOT = Path(__file__).resolve().parents[1]
VALUE_MODES = {"off": "absmax", "on": "midpoint_affine"}


@dataclass(frozen=True)
class Shape:
    batch: int
    q_heads: int
    kv_heads: int
    q_length: int
    kv_length: int
    head_dim: int

    @classmethod
    def parse(cls, value):
        try:
            fields = [int(x) for x in value.split(",")]
        except ValueError as error:
            raise argparse.ArgumentTypeError("case fields must be integers") from error
        if len(fields) != 6 or min(fields) < 1:
            raise argparse.ArgumentTypeError("case must be B,Hq,Hkv,Lq,Lkv,D")
        shape = cls(*fields)
        if shape.q_heads % shape.kv_heads or shape.head_dim > 256:
            raise argparse.ArgumentTypeError("Hq must be divisible by Hkv; D must be <= 256")
        return shape

    @property
    def label(self):
        return f"B{self.batch}-H{self.q_heads}/{self.kv_heads}-L{self.q_length}/{self.kv_length}-D{self.head_dim}"


def add_arguments(parser):
    parser.add_argument("--case", action="append", type=Shape.parse)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--layout", choices=("hnd", "nhd"), default="hnd", help="physical input storage"
    )
    parser.add_argument("--v-affine", choices=("off", "on", "both"), default="both")
    parser.add_argument(
        "--value-distribution",
        action="append",
        choices=("normal", "shifted", "outlier", "constant"),
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--calls-per-sample",
        "--iterations",
        dest="calls_per_sample",
        type=int,
        default=20,
        help="calls per CUDA-event sample",
    )
    parser.add_argument("--compare-sage", action="store_true")
    parser.add_argument("--compare-kitchen", action="store_true")
    parser.add_argument("--guard-comfyui", help="abort if this ComfyUI server has queued work")
    parser.add_argument("--output", required=True, type=Path, help="JSON result file")


def validate_arguments(parser, args):
    if min(args.warmup, args.samples, args.calls_per_sample) < 1:
        parser.error("warmup, samples and calls-per-sample must be positive")
    if args.compare_sage and args.dtype == "fp32":
        parser.error("SageAttention requires bf16 or fp16 inputs")
    validate_runtime(parser)


def validate_runtime(parser):
    """Require the native extension and Python sources from this checkout."""
    if not risa.int8_attention_is_available():
        parser.error("build and install this checkout's CUDA extension before benchmarking")
    package = Path(risa.__file__).resolve().parent
    for source in (ROOT / "src/risa_attention").rglob("*.py"):
        installed = package / source.relative_to(ROOT / "src/risa_attention")
        if not installed.is_file() or installed.read_bytes() != source.read_bytes():
            parser.error(f"installed Python code differs from this checkout: {source.name}")


def modes(args):
    return tuple(VALUE_MODES.values()) if args.v_affine == "both" else (VALUE_MODES[args.v_affine],)


def make_inputs(shape, args, seed, distribution):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def tensor(heads, tokens):
        result = torch.randn(
            (shape.batch, heads, tokens, shape.head_dim),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        if args.layout == "nhd":
            result = result.transpose(1, 2).contiguous().transpose(1, 2)
        return result

    q, k, v = (
        tensor(shape.q_heads, shape.q_length),
        tensor(shape.kv_heads, shape.kv_length),
        tensor(shape.kv_heads, shape.kv_length),
    )
    if distribution == "shifted":
        v.add_(3)
    elif distribution == "outlier":
        v[..., ::127, :].mul_(12)
    elif distribution == "constant":
        v.fill_(2.75)
    return q, k, v, generator


def fp32_reference(q, k, v):
    # Bound temporary score storage even when FP32 SDPA selects its math backend.
    group = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(group, dim=1) if group > 1 else k.float()
    v = v.float().repeat_interleave(group, dim=1) if group > 1 else v.float()
    output = torch.empty_like(q, dtype=torch.float32)
    chunk = max(1, min(128, 16_777_216 // (q.shape[0] * q.shape[1] * k.shape[2])))
    for start in range(0, q.shape[2], chunk):
        output[:, :, start : start + chunk] = F.scaled_dot_product_attention(
            q[:, :, start : start + chunk].float(), k, v
        )
    return output


def accuracy(output, reference):
    actual = output.float()
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise RuntimeError("non-finite benchmark output or reference")
    difference = actual - reference
    rmse = difference.square().mean().sqrt().item()
    rms = reference.square().mean().sqrt().item()
    nrmse = rmse / max(rms, 1e-12)
    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "sqnr_db": -20 * math.log10(nrmse) if nrmse > 0 else None,
        "mae": difference.abs().mean().item(),
        "max_abs": difference.abs().max().item(),
        "max_channel_bias": difference.mean(dim=2).abs().max().item(),
        "reference_rms": rms,
    }


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def ensure_idle(args):
    if args.guard_comfyui:
        with urllib.request.urlopen(
            args.guard_comfyui.rstrip("/") + "/queue", timeout=5
        ) as response:
            queue = json.load(response)
        if queue["queue_running"] or queue["queue_pending"]:
            raise RuntimeError("ComfyUI has queued work; discard this benchmark run")


def measure_methods(methods, args):
    """Warm methods, then rotate and reverse their positions on identical input."""
    samples = {name: [] for name in methods}
    names = list(methods)
    ensure_idle(args)
    for method in methods.values():
        for _ in range(args.warmup):
            method()
    torch.cuda.synchronize()
    for sample in range(args.samples):
        ensure_idle(args)
        # Forward/reverse alone puts only two methods first when there are
        # three or more. Rotate every position, including the first launch
        # after the queue check, then reverse each complete rotation cycle.
        order = names if (sample // len(names)) % 2 == 0 else names[::-1]
        offset = sample % len(names)
        order = order[offset:] + order[:offset]
        for name in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.calls_per_sample):
                methods[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / args.calls_per_sample)
    ensure_idle(args)
    return {
        name: {
            "median_ms": statistics.median(values),
            "p10_ms": percentile(values, 0.1),
            "p90_ms": percentile(values, 0.9),
            "samples_ms": values,
            "calls_per_sample": args.calls_per_sample,
            "warmup": args.warmup,
        }
        for name, values in samples.items()
    }


def peak_memory(function):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    torch.cuda.synchronize()
    result = {
        "peak_incremental_mib": max(0, torch.cuda.max_memory_allocated() - baseline) / 1024**2
    }
    del output
    return result


def baseline_methods(q, k, v, args):
    methods = {
        "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=q.shape[1] != k.shape[1])
    }
    if args.compare_sage:
        from sageattention import sageattn

        methods["sage"] = lambda: sageattn(q, k, v, tensor_layout="HND")
    if args.compare_kitchen:
        import comfy_kitchen

        methods["kitchen"] = lambda: comfy_kitchen.int8_attention(q, k, v)
    return methods


def record(
    shape, args, seed, distribution, implementation, mode, stage, timing, error, memory, sparse=None
):
    return {
        "case": shape.label,
        "shape": asdict(shape),
        "seed": seed,
        "layout": args.layout,
        "qk_distribution": getattr(args, "pattern", "normal"),
        "value_distribution": distribution,
        "implementation": implementation,
        "v_quantization": mode,
        "stage": stage,
        "timing": timing,
        "accuracy": error,
        "memory": memory,
        "sparse": sparse,
    }


def affine_comparisons(results):
    groups = {}
    for row in results:
        if row["v_quantization"] not in VALUE_MODES.values():
            continue
        key = (
            row["implementation"],
            row["case"],
            row["seed"],
            row["layout"],
            row["qk_distribution"],
            row["value_distribution"],
            row["stage"],
        )
        groups.setdefault(key, {})[row["v_quantization"]] = row
    comparisons = []
    for pair in groups.values():
        if set(pair) != {"absmax", "midpoint_affine"}:
            continue
        off, on = pair["absmax"], pair["midpoint_affine"]

        def change(a, b):
            return (b / a - 1) * 100 if a > 0 else None

        comparisons.append(
            {
                **{
                    key: off[key]
                    for key in (
                        "implementation",
                        "case",
                        "shape",
                        "seed",
                        "layout",
                        "qk_distribution",
                        "value_distribution",
                        "stage",
                    )
                },
                "latency_change_percent": change(
                    off["timing"]["median_ms"], on["timing"]["median_ms"]
                ),
                "nrmse_change_percent": change(off["accuracy"]["nrmse"], on["accuracy"]["nrmse"])
                if off["accuracy"]
                else None,
            }
        )
    return comparisons


def write_results(args, suite, results):
    from risa_attention import _backend

    digest = hashlib.sha256()
    for directory in ("src", "csrc"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.suffix in (".py", ".cpp", ".cu", ".cuh", ".h", ".txt", ".in"):
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    versions = {}
    for package in ("torch", "sageattention", "comfy-kitchen"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = {key: value for key, value in vars(args).items() if key not in ("output", "case")}
    config["cases"] = [asdict(shape) for shape in args.case]
    config["timing_order"] = "rotate-and-reverse"
    payload = {
        "schema_version": 2,
        "suite": suite,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "cuda": torch.version.cuda,
            "versions": versions,
            "risa_version": risa.__version__,
            "package_path": risa.__file__,
            "source_sha256": digest.hexdigest(),
            "extension_sha256": hashlib.sha256(Path(_backend._C.__file__).read_bytes()).hexdigest(),
        },
        "config": config,
        "reference": "FP32 PyTorch SDPA, query chunks limited to 16M score elements",
        "results": results,
        "affine_comparisons": affine_comparisons(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(results)} records: {args.output}")
