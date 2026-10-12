# SPDX-License-Identifier: Apache-2.0
"""Paired native comparison for broadcast masks and tile-skipping controls."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from common import (
    Shape,
    accuracy,
    add_arguments,
    make_inputs,
    measure_methods,
    modes,
    record,
    validate_arguments,
    write_results,
)
from compare_native import load_baseline, with_native

import risa_attention as risa
from risa_attention import _backend


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.set_defaults(samples=16)
    parser.add_argument("--baseline-extension", required=True, type=Path)
    parser.add_argument("--empty-fraction", action="append", type=float)
    parser.add_argument("--mask-dtype", choices=("bool", "fp32"), default="bool")
    args = parser.parse_args()
    validate_arguments(parser, args)
    if args.compare_sage or args.compare_kitchen:
        parser.error("this runner compares two RISA native extensions")
    args.case = args.case or [Shape(1, 16, 4, 4096, 4096, 128)]
    args.seed = args.seed or [0, 1, 2]
    fractions = args.empty_fraction or [0.0, 0.25, 0.5, 0.75]
    if any(not 0 <= fraction <= 1 for fraction in fractions):
        parser.error("empty-fraction must be in [0, 1]")
    args.baseline_extension_sha256 = hashlib.sha256(
        args.baseline_extension.read_bytes()
    ).hexdigest()
    baseline = load_baseline(args.baseline_extension)
    current = _backend._C
    args.baseline_extension = str(args.baseline_extension.resolve())
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for shape in args.case:
        for seed in args.seed:
            q, k, v, _ = make_inputs(shape, args, seed, "normal")
            for fraction in fractions:
                mask = torch.ones(1, 1, 1, shape.kv_length, device=q.device, dtype=torch.bool)
                tile_size = 64 if shape.head_dim <= 64 or shape.kv_length <= 1024 else 128
                tiles = (shape.kv_length + tile_size - 1) // tile_size
                empty_tiles = int(tiles * fraction)
                # Interleave excluded tiles rather than measuring a simple prefix.
                order = list(range(0, tiles, 2)) + list(range(1, tiles, 2))
                for tile in order[:empty_tiles]:
                    mask[..., tile * tile_size : (tile + 1) * tile_size] = False
                if args.mask_dtype == "fp32":
                    mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                        ~mask, -torch.inf
                    )
                reference = torch.empty_like(q, dtype=torch.float32)
                for start in range(0, shape.q_length, 128):
                    reference[..., start : start + 128, :] = F.scaled_dot_product_attention(
                        q[..., start : start + 128, :].float(),
                        k.float(),
                        v.float(),
                        attn_mask=mask,
                        enable_gqa=shape.q_heads != shape.kv_heads,
                    )
                methods, errors, comparisons = {}, {}, {}
                for mode in modes(args):

                    def call(mode=mode):
                        return risa.int8_attention(q, k, v, attn_mask=mask, v_quantization=mode)

                    before = with_native(baseline, call)
                    after = with_native(current, call)
                    torch.testing.assert_close(before, after, atol=0.002, rtol=0.01)
                    comparisons[mode] = {
                        "bitwise_equal": torch.equal(before, after),
                        "max_abs_difference": (before.float() - after.float()).abs().max().item(),
                    }
                    for label, native, output in (
                        ("before", baseline, before),
                        ("after", current, after),
                    ):
                        name = f"{label}/{mode}"
                        methods[name] = lambda native=native, call=call: with_native(native, call)
                        errors[name] = accuracy(output, reference)
                timings = measure_methods(methods, args)
                for name, timing in timings.items():
                    label, mode = name.split("/")
                    row = record(
                        shape,
                        args,
                        seed,
                        "normal",
                        label,
                        mode,
                        f"mask_empty_{fraction}",
                        timing,
                        errors[name],
                        None,
                    )
                    row.update(
                        mask_empty_fraction=fraction,
                        mask_dtype=args.mask_dtype,
                        native_comparison=comparisons[mode],
                    )
                    results.append(row)
                print(
                    f"{shape.label} seed={seed} empty={fraction}: "
                    f"{ {name: round(value['median_ms'], 5) for name, value in timings.items()} }",
                    flush=True,
                )
    write_results(args, "masked-native-comparison", results)


if __name__ == "__main__":
    main()
