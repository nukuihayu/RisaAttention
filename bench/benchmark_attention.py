#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import risa_attention as risa


@dataclass(frozen=True)
class Case:
    batch: int
    q_heads: int
    kv_heads: int
    q_length: int
    kv_length: int
    head_dim: int

    @classmethod
    def parse(cls, value: str) -> Case:
        try:
            fields = [int(item) for item in value.split(",")]
        except ValueError as error:
            raise argparse.ArgumentTypeError("case fields must be integers") from error
        if len(fields) != 6 or any(field <= 0 for field in fields):
            raise argparse.ArgumentTypeError(
                "case must be B,Hq,Hkv,Lq,Lkv,D with positive fields"
            )
        case = cls(*fields)
        if case.q_heads % case.kv_heads:
            raise argparse.ArgumentTypeError("Hq must be divisible by Hkv")
        if case.head_dim > 256:
            raise argparse.ArgumentTypeError("D must be <= 256")
        return case

    @property
    def label(self) -> str:
        return (
            f"B{self.batch}-H{self.q_heads}/{self.kv_heads}-"
            f"L{self.q_length}/{self.kv_length}-D{self.head_dim}"
        )


@dataclass
class Result:
    case: str
    pattern: str
    implementation: str
    median_ms: float
    p20_ms: float
    p80_ms: float
    query_tokens_per_second: float | None
    approximate_tflops: float | None
    nrmse_vs_sdpa: float | None


DEFAULT_CASES = (
    Case(1, 16, 16, 512, 512, 64),
    Case(1, 16, 4, 1024, 1024, 128),
    Case(2, 24, 8, 1024, 2048, 128),
    Case(1, 16, 4, 4096, 4096, 128),
)


def _dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[value.lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            "dtype must be float16, bfloat16, or float32"
        ) from error


def _measure(
    function: Callable[[], torch.Tensor | object], warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _nrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    magnitude = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return (error / magnitude).item()


def _result(
    case: Case, pattern: str, name: str, samples: list[float], error: float | None
) -> Result:
    ordered = sorted(samples)
    median_ms = statistics.median(ordered)
    p20_ms = ordered[int(0.2 * (len(ordered) - 1))]
    p80_ms = ordered[int(0.8 * (len(ordered) - 1))]
    seconds = median_ms / 1_000.0
    is_attention = name != "risa_quantize_only"
    operations = (
        4 * case.batch * case.q_heads * case.q_length * case.kv_length * case.head_dim
    )
    return Result(
        case=case.label,
        pattern=pattern,
        implementation=name,
        median_ms=median_ms,
        p20_ms=p20_ms,
        p80_ms=p80_ms,
        query_tokens_per_second=case.batch * case.q_length / seconds
        if is_attention
        else None,
        approximate_tflops=operations / seconds / 1e12 if is_attention else None,
        nrmse_vs_sdpa=error,
    )


def _run_case(
    case: Case,
    pattern: str,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    compare_comfy_kitchen: bool,
    compare_sage_attention: bool,
) -> list[Result]:
    q = torch.randn(
        case.batch,
        case.q_heads,
        case.q_length,
        case.head_dim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        case.batch,
        case.kv_heads,
        case.kv_length,
        case.head_dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn_like(k)
    if pattern == "channel_outlier":
        q[..., 0].mul_(12)
        k[..., 0].mul_(12)
        q.mul_(q.float().square().mean(-1, keepdim=True).rsqrt().to(dtype))
        k.mul_(k.float().square().mean(-1, keepdim=True).rsqrt().to(dtype))
    elif pattern == "common_key":
        common = torch.randn(
            case.batch,
            case.kv_heads,
            1,
            case.head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        common.mul_(40.0 / common.square().mean(-1, keepdim=True).sqrt())
        k.add_(common.to(dtype))
    group_size = case.q_heads // case.kv_heads
    baseline_k = k.repeat_interleave(group_size, dim=1) if group_size > 1 else k
    baseline_v = v.repeat_interleave(group_size, dim=1) if group_size > 1 else v

    def sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            q, baseline_k, baseline_v
        )

    def fused():
        return risa.int8_attention(q, k, v)

    reference = sdpa()
    fused_output = fused()
    packed = risa.prequantize_int8_attention(q, k, v)

    implementations: list[tuple[str, Callable[[], object], float | None]] = [
        ("torch_sdpa", sdpa, 0.0),
        ("risa_fused_e2e", fused, _nrmse(fused_output, reference)),
        (
            "risa_prequantized_attend",
            lambda: risa.int8_attention_from_prequantized(packed),
            _nrmse(risa.int8_attention_from_prequantized(packed), reference),
        ),
        (
            "risa_quantize_only",
            lambda: risa.prequantize_int8_attention(q, k, v),
            None,
        ),
    ]

    if compare_comfy_kitchen:
        try:
            import comfy_kitchen
        except ImportError as error:
            raise RuntimeError(
                "--compare-comfy-kitchen requires an installed comfy-kitchen"
            ) from error
        if not comfy_kitchen.int8_attention_is_available():
            raise RuntimeError(
                "the installed comfy-kitchen INT8 attention backend is unavailable"
            )
        kitchen_output = comfy_kitchen.int8_attention(q, k, v)
        implementations.append(
            (
                "comfy_kitchen_baseline",
                lambda: comfy_kitchen.int8_attention(q, k, v),
                _nrmse(kitchen_output, reference),
            )
        )

    if compare_sage_attention:
        if dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "official SageAttention comparison requires FP16 or BF16"
            )
        try:
            from sageattention import sageattn
        except ImportError as error:
            raise RuntimeError(
                "--compare-sage-attention requires the sageattention package"
            ) from error
        official_output = sageattn(q, k, v, tensor_layout="HND")
        implementations.append(
            (
                "sageattention_official",
                lambda: sageattn(q, k, v, tensor_layout="HND"),
                _nrmse(official_output, reference),
            )
        )

    results = []
    for name, function, error in implementations:
        samples = _measure(function, warmup, iterations)
        results.append(_result(case, pattern, name, samples, error))
    return results


def _print_table(results: list[Result]) -> None:
    header = (
        f"{'case':<30} {'pattern':<16} {'implementation':<28} {'median ms':>10} "
        f"{'p20/p80 ms':>18} {'query tok/s':>14} {'TFLOP/s':>10} {'NRMSE':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        token_rate = (
            "-"
            if result.query_tokens_per_second is None
            else f"{result.query_tokens_per_second:,.0f}"
        )
        tflops = (
            "-"
            if result.approximate_tflops is None
            else f"{result.approximate_tflops:.2f}"
        )
        error = "-" if result.nrmse_vs_sdpa is None else f"{result.nrmse_vs_sdpa:.4g}"
        print(
            f"{result.case:<30} {result.pattern:<16} {result.implementation:<28} "
            f"{result.median_ms:>10.3f} "
            f"{result.p20_ms:>8.3f}/{result.p80_ms:<8.3f} "
            f"{token_rate:>14} {tflops:>10} {error:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RISA dense INT8 attention")
    parser.add_argument(
        "--case",
        action="append",
        type=Case.parse,
        help="B,Hq,Hkv,Lq,Lkv,D; repeat for multiple cases",
    )
    parser.add_argument("--dtype", type=_dtype, default=torch.bfloat16)
    parser.add_argument(
        "--pattern",
        action="append",
        choices=("normal", "channel_outlier", "common_key"),
        help="input distribution; repeat to benchmark multiple patterns",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-comfy-kitchen", action="store_true")
    parser.add_argument("--compare-sage-attention", action="store_true")
    parser.add_argument("--json", type=Path, help="also write machine-readable results")
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be >= 0 and iterations must be > 0")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        parser.error("an NVIDIA CUDA PyTorch runtime is required")
    if not risa.int8_attention_is_available():
        parser.error("RISA Attention is unavailable or the GPU is older than SM75")

    results = []
    patterns = args.pattern or ["normal"]
    for case in args.case or DEFAULT_CASES:
        for pattern in patterns:
            torch.manual_seed(args.seed)
            results.extend(
                _run_case(
                    case,
                    pattern,
                    args.dtype,
                    args.warmup,
                    args.iterations,
                    args.compare_comfy_kitchen,
                    args.compare_sage_attention,
                )
            )
    _print_table(results)

    if args.json:
        payload = {
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": str(args.dtype),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "patterns": patterns,
            "results": [asdict(result) for result in results],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
