# SPDX-License-Identifier: Apache-2.0
"""Benchmark dense attention and paired V-affine modes."""

from __future__ import annotations

import argparse

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
from risa_attention import _backend

DEFAULT_SHAPES = tuple(
    Shape(1, 16, 4, n, n, 128) for n in (512, 1024, 2048, 4096, 8192, 16384, 32768)
)


def protocol_export(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.__dlpack__(stream=-1)


def with_exporter(exporter, operation):
    """Select capsule export in this single-threaded benchmark process."""
    previous = _backend.wrap_for_dlpack
    _backend.wrap_for_dlpack = exporter
    try:
        return operation()
    finally:
        _backend.wrap_for_dlpack = previous


@torch.inference_mode()
def run_case(shape, args, seed, distribution):
    q, k, v, _ = make_inputs(shape, args, seed, distribution)
    reference = fp32_reference(q, k, v)
    methods = baseline_methods(q, k, v, args)
    metadata = {name: (name, None, "dense") for name in methods}
    for mode in modes(args):
        name = f"risa/{mode}"

        def operation(mode=mode):
            return risa.int8_attention(q, k, v, v_quantization=mode)

        methods[name] = operation
        metadata[name] = ("risa", mode, "dense")
        if args.compare_dlpack_protocol:
            # Give both exports the same dispatch-wrapper overhead.
            exporter = _backend.wrap_for_dlpack
            methods[name] = lambda operation=operation: with_exporter(exporter, operation)
            protocol_name = f"risa_protocol/{mode}"
            methods[protocol_name] = lambda operation=operation: with_exporter(
                protocol_export, operation
            )
            metadata[protocol_name] = ("risa_protocol", mode, "dense")
        if args.include_components:
            packed = risa.prequantize_int8_attention(q, k, v, v_quantization=mode)
            methods[f"quantize/{mode}"] = lambda mode=mode: risa.prequantize_int8_attention(
                q, k, v, v_quantization=mode
            )
            methods[f"attend/{mode}"] = lambda packed=packed: risa.int8_attention_from_prequantized(
                packed
            )
            metadata[f"quantize/{mode}"] = ("risa", mode, "quantize")
            metadata[f"attend/{mode}"] = ("risa", mode, "prequantized_dense")
    errors = {}
    exported_outputs = {}
    original = [tensor.clone() for tensor in (q, k, v)]
    for name, function in methods.items():
        output = function()
        errors[name] = None if metadata[name][2] == "quantize" else accuracy(output, reference)
        if (
            args.compare_dlpack_protocol
            and metadata[name][0] in ("risa", "risa_protocol")
            and metadata[name][2] == "dense"
        ):
            mode = metadata[name][1]
            if mode in exported_outputs:
                if not torch.equal(output, exported_outputs.pop(mode)):
                    raise RuntimeError(f"DLPack export changes {mode} output")
            else:
                exported_outputs[mode] = output
        if any(not torch.equal(a, b) for a, b in zip((q, k, v), original)):
            raise RuntimeError(f"{name} modifies benchmark inputs")
        del output
    del original
    timings = measure_methods(methods, args)
    return [
        record(
            shape,
            args,
            seed,
            distribution,
            *metadata[name],
            timings[name],
            errors[name],
            peak_memory(function),
        )
        for name, function in methods.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--include-components", action="store_true")
    parser.add_argument(
        "--compare-dlpack-protocol", action="store_true", help="compare Python protocol export"
    )
    args = parser.parse_args()
    validate_arguments(parser, args)
    args.case = args.case or list(DEFAULT_SHAPES)
    args.seed = args.seed or [0, 1, 2]
    args.value_distribution = args.value_distribution or ["normal"]
    results = []
    for shape in args.case:
        for seed in args.seed:
            for distribution in args.value_distribution:
                print(f"{shape.label} seed={seed} V={distribution}", flush=True)
                results.extend(run_case(shape, args, seed, distribution))
    write_results(args, "dense", results)


if __name__ == "__main__":
    main()
