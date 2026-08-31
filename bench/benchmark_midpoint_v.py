#!/usr/bin/env python3
"""Compare residual-zero midpoint-affine V with comfy-kitchen absmax V."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import comfy_kitchen
import risa_attention


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    difference = actual_f - reference_f
    rmse = difference.square().mean().sqrt()
    signal = reference_f.square().mean().sqrt().clamp_min(1e-12)
    return {
        "rmse": float(rmse),
        "nrmse": float(rmse / signal),
        "mae": float(difference.abs().mean()),
        "max_abs": float(difference.abs().max()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_f.flatten(), reference_f.flatten(), dim=0
            )
        ),
    }


def _paired_latency(
    first,
    second,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], dict[str, float]]:
    for iteration in range(warmup):
        functions = (first, second) if iteration % 2 == 0 else (second, first)
        for function in functions:
            function()
    torch.cuda.synchronize()
    samples = [[], []]
    for iteration in range(iterations):
        order = ((0, first), (1, second))
        if iteration % 2:
            order = tuple(reversed(order))
        for index, function in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end))

    def summarize(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "median_ms": statistics.median(ordered),
            "p10_ms": ordered[int(0.1 * (len(ordered) - 1))],
            "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        }

    return summarize(samples[0]), summarize(samples[1])


def _inputs(
    length: int,
    pattern: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    shape_q = (1, 16, length, 128)
    shape_kv = (1, 4, length, 128)
    q = torch.randn(*shape_q, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(*shape_kv, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(*shape_kv, device="cuda", dtype=torch.bfloat16)
    if pattern == "shifted":
        v.add_(3.0)
    elif pattern == "positive":
        v.abs_()
    elif pattern == "channel_shift":
        channel_offset = torch.empty(
            1, 4, 1, 128, device="cuda", dtype=torch.float32
        ).uniform_(-3.0, 3.0)
        v.add_(channel_offset.to(v.dtype))
    return q, k, v


def _fake_quantize_v(
    v: torch.Tensor,
    *,
    midpoint: bool,
) -> torch.Tensor:
    value = v.float()
    maximum = value.amax(dim=2, keepdim=True)
    minimum = value.amin(dim=2, keepdim=True)
    if midpoint:
        center = 0.5 * (minimum + maximum)
        radius = 0.5 * (maximum - minimum)
    else:
        center = torch.zeros_like(maximum)
        radius = torch.maximum(minimum.abs(), maximum.abs())
    scale = (radius / 127.0).clamp_min(1e-12)
    quantized = torch.round((value - center) / scale).clamp_(-127, 127)
    if midpoint:
        center = (value - scale * quantized).mean(dim=2, keepdim=True)
    return (quantized * scale + center).to(v.dtype)


@torch.inference_mode()
def _run_case(
    length: int,
    pattern: str,
    seed: int,
    warmup: int,
    iterations: int,
) -> dict:
    q, k, v = _inputs(length, pattern, seed)
    expanded_k = k.repeat_interleave(4, dim=1)
    expanded_v = v.repeat_interleave(4, dim=1)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, expanded_k, expanded_v
    )

    midpoint_v = _fake_quantize_v(v, midpoint=True)
    absmax_v = _fake_quantize_v(v, midpoint=False)
    midpoint_v_output = torch.nn.functional.scaled_dot_product_attention(
        q, expanded_k, midpoint_v.repeat_interleave(4, dim=1)
    )
    absmax_v_output = torch.nn.functional.scaled_dot_product_attention(
        q, expanded_k, absmax_v.repeat_interleave(4, dim=1)
    )
    midpoint_v_rmse = float((midpoint_v.float() - v.float()).square().mean().sqrt())
    absmax_v_rmse = float((absmax_v.float() - v.float()).square().mean().sqrt())
    midpoint_v_output_rmse = _metrics(midpoint_v_output, reference)["rmse"]
    absmax_v_output_rmse = _metrics(absmax_v_output, reference)["rmse"]
    isolated_v = {
        "midpoint_reconstruction_rmse": midpoint_v_rmse,
        "absmax_reconstruction_rmse": absmax_v_rmse,
        "reconstruction_rmse_reduction": 1.0 - midpoint_v_rmse / absmax_v_rmse,
        "midpoint_output_rmse": midpoint_v_output_rmse,
        "absmax_output_rmse": absmax_v_output_rmse,
        "output_rmse_reduction": 1.0 - midpoint_v_output_rmse / absmax_v_output_rmse,
    }
    del midpoint_v, absmax_v, midpoint_v_output, absmax_v_output

    midpoint_packed = risa_attention.prequantize_int8_attention(q, k, v)
    absmax_packed = comfy_kitchen.prequantize_int8_attention(q, k, v)
    qk_equal = all(
        torch.equal(left, right)
        for left, right in (
            (midpoint_packed.q, absmax_packed.q),
            (midpoint_packed.k, absmax_packed.k),
            (midpoint_packed.q_scale, absmax_packed.q_scale),
            (midpoint_packed.k_scale, absmax_packed.k_scale),
        )
    )
    midpoint_output = risa_attention.int8_attention_from_prequantized(
        midpoint_packed
    )
    absmax_output = comfy_kitchen.int8_attention_from_prequantized(absmax_packed)
    midpoint_metrics = _metrics(midpoint_output, reference)
    absmax_metrics = _metrics(absmax_output, reference)

    midpoint_fused = lambda: risa_attention.int8_attention(q, k, v)
    absmax_fused = lambda: comfy_kitchen.int8_attention(q, k, v)
    midpoint_quantize = lambda: risa_attention.prequantize_int8_attention(q, k, v)
    absmax_quantize = lambda: comfy_kitchen.prequantize_int8_attention(q, k, v)
    midpoint_attend = lambda: risa_attention.int8_attention_from_prequantized(
        midpoint_packed
    )
    absmax_attend = lambda: comfy_kitchen.int8_attention_from_prequantized(
        absmax_packed
    )
    fused_midpoint, fused_absmax = _paired_latency(
        midpoint_fused, absmax_fused, warmup, iterations
    )
    quant_midpoint, quant_absmax = _paired_latency(
        midpoint_quantize, absmax_quantize, warmup, iterations
    )
    attend_midpoint, attend_absmax = _paired_latency(
        midpoint_attend, absmax_attend, warmup, iterations
    )

    return {
        "length": length,
        "pattern": pattern,
        "seed": seed,
        "qk_bitwise_equal": qk_equal,
        "midpoint_center_abs_mean": float(midpoint_packed.v_center.abs().mean()),
        "isolated_v": isolated_v,
        "midpoint": {
            **midpoint_metrics,
            "fused": fused_midpoint,
            "quantize": quant_midpoint,
            "attend": attend_midpoint,
        },
        "comfy_kitchen_absmax": {
            **absmax_metrics,
            "fused": fused_absmax,
            "quantize": quant_absmax,
            "attend": attend_absmax,
        },
        "rmse_reduction": 1.0 - midpoint_metrics["rmse"] / absmax_metrics["rmse"],
        "fused_latency_change": (
            fused_midpoint["median_ms"] / fused_absmax["median_ms"] - 1.0
        ),
        "quantize_latency_change": (
            quant_midpoint["median_ms"] / quant_absmax["median_ms"] - 1.0
        ),
        "attend_latency_change": (
            attend_midpoint["median_ms"] / attend_absmax["median_ms"] - 1.0
        ),
    }


def _print(run: dict) -> None:
    midpoint = run["midpoint"]
    absmax = run["comfy_kitchen_absmax"]
    print(
        f"L={run['length']:<5} {run['pattern']:<13} seed={run['seed']} "
        f"QK_equal={run['qk_bitwise_equal']} center={run['midpoint_center_abs_mean']:.3f} "
        f"NRMSE {midpoint['nrmse']:.5f}/{absmax['nrmse']:.5f} "
        f"RMSE_gain={100.0 * run['rmse_reduction']:+.2f}% "
        f"V_only={100.0 * run['isolated_v']['output_rmse_reduction']:+.2f}% "
        f"fused={midpoint['fused']['median_ms']:.3f}/{absmax['fused']['median_ms']:.3f}ms "
        f"delta={100.0 * run['fused_latency_change']:+.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", action="append", type=int)
    parser.add_argument(
        "--pattern",
        action="append",
        choices=("normal", "shifted", "positive", "channel_shift"),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not risa_attention.int8_attention_is_available():
        parser.error("risa_attention is unavailable on the active GPU")
    if not comfy_kitchen.int8_attention_is_available():
        parser.error("comfy_kitchen INT8 attention is unavailable on the active GPU")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be >= 0 and iterations must be positive")

    runs = []
    for length in args.length or [1024, 4096, 8192, 16384]:
        for pattern in args.pattern or ["normal", "shifted", "positive"]:
            for seed in args.seed or [0]:
                run = _run_case(
                    length, pattern, seed, args.warmup, args.iterations
                )
                runs.append(run)
                _print(run)

    if args.json:
        payload = {
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "runs": runs,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
