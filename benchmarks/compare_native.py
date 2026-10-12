# SPDX-License-Identifier: Apache-2.0
"""Compare a preserved CUDA extension with this checkout on identical inputs.

For changes intended to preserve output bits. Load both modules in one process,
check equality before timing, and balance their measurement positions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
from common import (
    Shape,
    accuracy,
    add_arguments,
    fp32_reference,
    make_inputs,
    measure_methods,
    modes,
    record,
    validate_arguments,
    write_results,
)

import risa_attention as risa
from risa_attention import _backend, _execution


def load_baseline(path):
    sys.modules["_risa_benchmark_baseline"] = types.ModuleType("_risa_benchmark_baseline")
    spec = importlib.util.spec_from_file_location("_risa_benchmark_baseline._C", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def with_native(native, function):
    original = _backend._C
    prepare_tiles = _execution.prepare_key_mask_tiles
    _backend._C = native
    # Historical extensions predate mask metadata. Keep comparison support in
    # the benchmark runner rather than adding an old-native path to the library.
    if not hasattr(native, "prepare_key_mask_tiles"):
        _execution.prepare_key_mask_tiles = lambda *args: None
    try:
        return function()
    finally:
        _backend._C = original
        _execution.prepare_key_mask_tiles = prepare_tiles


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.set_defaults(samples=24)
    parser.add_argument("--baseline-extension", required=True, type=Path)
    parser.add_argument("--include-components", action="store_true")
    args = parser.parse_args()
    validate_arguments(parser, args)
    if args.compare_sage or args.compare_kitchen:
        parser.error("use dense.py for comparisons with SageAttention or comfy-kitchen")
    args.case = args.case or [Shape(1, 16, 4, 512, 512, 128)]
    args.seed = args.seed or [0, 1, 2]
    args.value_distribution = args.value_distribution or ["normal"]
    baseline = load_baseline(args.baseline_extension)
    current = _backend._C
    args.baseline_extension_sha256 = hashlib.sha256(
        args.baseline_extension.read_bytes()
    ).hexdigest()
    args.baseline_extension = str(args.baseline_extension.resolve())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    results = []
    for shape in args.case:
        for seed in args.seed:
            for distribution in args.value_distribution:
                q, k, v, _ = make_inputs(shape, args, seed, distribution)
                reference = fp32_reference(q, k, v)
                methods, errors, metadata = {}, {}, {}
                for mode in modes(args):

                    def dense(mode=mode):
                        return risa.int8_attention(q, k, v, v_quantization=mode)

                    before = with_native(baseline, dense)
                    after = with_native(current, dense)
                    if not torch.equal(before, after):
                        raise RuntimeError(f"output bits differ: {shape.label}, {seed}, {mode}")
                    error = accuracy(after, reference)
                    stages = {"dense": dense}
                    if args.include_components:

                        def quantize(mode=mode):
                            return risa.prequantize_int8_attention(q, k, v, v_quantization=mode)

                        packs = [with_native(native, quantize) for native in (baseline, current)]
                        for attribute in (
                            "q",
                            "k",
                            "v",
                            "q_scale",
                            "k_scale",
                            "v_scale",
                            "v_center",
                        ):
                            if not torch.equal(
                                getattr(packs[0], attribute), getattr(packs[1], attribute)
                            ):
                                raise RuntimeError(
                                    f"packed {attribute} bits differ: {shape.label}, {mode}"
                                )
                        stages["quantize"] = quantize
                        stages["prequantized_dense"] = lambda packed=packs[0]: (
                            risa.int8_attention_from_prequantized(packed)
                        )
                    for stage, operation in stages.items():
                        for label, native in (("before", baseline), ("after", current)):
                            name = f"{label}/{mode}/{stage}"
                            methods[name] = lambda native=native, operation=operation: with_native(
                                native, operation
                            )
                            errors[name] = None if stage == "quantize" else error
                            metadata[name] = (label, mode, stage)
                timings = measure_methods(methods, args)
                for name in methods:
                    results.append(
                        record(
                            shape,
                            args,
                            seed,
                            distribution,
                            *metadata[name],
                            timings[name],
                            errors[name],
                            None,
                        )
                    )
                print(
                    shape.label,
                    seed,
                    distribution,
                    {name: round(t["median_ms"], 6) for name, t in timings.items()},
                    flush=True,
                )
    write_results(args, "native-comparison", results)


if __name__ == "__main__":
    main()
