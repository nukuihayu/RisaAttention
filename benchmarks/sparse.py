# SPDX-License-Identifier: Apache-2.0
"""Benchmark sparse construction and reuse with paired V-affine modes."""

from __future__ import annotations

import argparse
from copy import copy

import torch
from common import (
    Shape,
    accuracy,
    add_arguments,
    baseline_methods,
    fp32_reference,
    make_inputs,
    measure_methods,
    modes,
    peak_memory,
    record,
    validate_arguments,
    write_results,
)

import risa_attention as risa

DEFAULT_SHAPES = tuple(Shape(1, 16, 4, n, n, 128) for n in (1024, 2048, 4096, 8192, 16384, 32768))


@torch.inference_mode()
def run_case(shape, args, seed, distribution):
    q0, k0, v, generator = make_inputs(shape, args, seed, distribution)
    if args.pattern == "video_blocks":
        prototypes = torch.randn(
            (shape.batch, shape.kv_heads, args.clusters, shape.head_dim),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        prototypes *= args.prototype_norm / prototypes.norm(dim=-1, keepdim=True)
        cluster = (
            torch.arange(shape.q_length, device="cuda").div(128, rounding_mode="floor")
            % args.clusters
        )
        q0.add_(
            prototypes.repeat_interleave(shape.q_heads // shape.kv_heads, dim=1)[:, :, cluster].to(
                q0.dtype
            )
        )
        k0.add_(prototypes[:, :, cluster].to(k0.dtype))
    q = q0 + torch.randn(q0.shape, generator=generator, device="cuda", dtype=q0.dtype) * args.drift
    k = k0 + torch.randn(k0.shape, generator=generator, device="cuda", dtype=k0.dtype) * args.drift
    reference = fp32_reference(q, k, v)
    construction_reference = fp32_reference(q0, k0, v)
    builds = {
        mode: lambda mode=mode: risa.construct_sparse_int8_attention(
            q0, k0, v, theta=args.theta, v_quantization=mode
        )
        for mode in modes(args)
    }
    patterns, build_errors = {}, {}
    for mode, function in builds.items():
        output, pattern = function()
        patterns[mode] = pattern
        build_errors[mode] = accuracy(output, construction_reference)
    shared = next(iter(patterns.values()))
    for pattern in patterns.values():
        if not torch.equal(shared.row_offsets, pattern.row_offsets) or not torch.equal(
            shared.block_indices, pattern.block_indices
        ):
            raise RuntimeError("V quantization unexpectedly changed retained-mass support")
    sparse_info = {
        "coverage": shared.coverage,
        "theta": args.theta,
        "pattern_index_mib": shared.index_bytes / 1024**2,
        "measured_construction_mass": shared.measured_retained_mass,
        "exact_construction_recall": risa.measure_pattern_recall(q0, k0, shared),
        "current_recall": risa.measure_pattern_recall(q, k, shared),
    }
    construction_args = copy(args)
    construction_args.warmup = args.construction_warmup
    construction_args.samples = args.construction_samples
    construction_args.calls_per_sample = 1
    build_timings = measure_methods(builds, construction_args)
    rows = [
        record(
            shape,
            args,
            seed,
            distribution,
            "risa",
            mode,
            "construct",
            build_timings[mode],
            build_errors[mode],
            peak_memory(function),
            sparse_info,
        )
        for mode, function in builds.items()
    ]
    del construction_reference
    methods = baseline_methods(q, k, v, args)
    metadata = {name: (name, None, "dense") for name in methods}
    for mode in modes(args):
        methods[f"dense/{mode}"] = lambda mode=mode: risa.int8_attention(
            q, k, v, v_quantization=mode
        )
        methods[f"sparse/{mode}"] = lambda mode=mode: risa.sparse_int8_attention(
            q, k, v, shared, v_quantization=mode
        )
        metadata[f"dense/{mode}"] = ("risa", mode, "dense")
        metadata[f"sparse/{mode}"] = ("risa", mode, "sparse")
    errors = {name: accuracy(function(), reference) for name, function in methods.items()}
    timings = measure_methods(methods, args)
    rows.extend(
        record(
            shape,
            args,
            seed,
            distribution,
            *metadata[name],
            timings[name],
            errors[name],
            peak_memory(function),
            sparse_info if metadata[name][2] == "sparse" else None,
        )
        for name, function in methods.items()
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--pattern", choices=("normal", "video_blocks"), default="video_blocks")
    parser.add_argument("--theta", type=float, default=0.99)
    parser.add_argument("--drift", type=float, default=0.05)
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--prototype-norm", type=float, default=8)
    parser.add_argument("--construction-warmup", type=int, default=2)
    parser.add_argument("--construction-samples", type=int, default=10)
    args = parser.parse_args()
    validate_arguments(parser, args)
    args.case = args.case or list(DEFAULT_SHAPES)
    args.seed = args.seed or [0, 1, 2]
    args.value_distribution = args.value_distribution or ["normal"]
    if any(shape.q_length != shape.kv_length for shape in args.case):
        parser.error("sparse benchmark requires self-attention")
    if (
        not 0 < args.theta <= 1
        or min(
            args.construction_warmup, args.construction_samples, args.clusters, args.prototype_norm
        )
        <= 0
        or args.drift < 0
    ):
        parser.error("invalid sparse construction parameters")
    results = []
    for shape in args.case:
        for seed in args.seed:
            for distribution in args.value_distribution:
                print(f"{shape.label} seed={seed} QK={args.pattern} V={distribution}", flush=True)
                results.extend(run_case(shape, args, seed, distribution))
    write_results(args, "sparse", results)


if __name__ == "__main__":
    main()
