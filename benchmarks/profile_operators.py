# SPDX-License-Identifier: Apache-2.0
"""Capture warmed RISA operators with Nsight Systems or Nsight Compute.

Use cudaProfilerApi capture in nsys, or --profile-from-start off in ncu.
This runner emits NVTX ranges and deliberately does not report profiler-inflated
kernel durations as benchmark latency. Use dense.py for latency and accuracy.
"""

from __future__ import annotations

import argparse

import torch
from common import Shape, ensure_idle, make_inputs, modes, validate_runtime

import risa_attention as risa


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Shape.parse, default=Shape(1, 16, 4, 512, 512, 128))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--layout", choices=("hnd", "nhd"), default="hnd")
    parser.add_argument("--v-affine", choices=("off", "on", "both"), default="both")
    parser.add_argument("--stage", choices=("dense", "quantize", "attend"), default="dense")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--guard-comfyui")
    args = parser.parse_args()
    if min(args.warmup, args.calls) < 1:
        parser.error("warmup and calls must be positive")
    validate_runtime(parser)
    ensure_idle(args)
    q, k, v, _ = make_inputs(args.case, args, args.seed, "normal")
    methods = {}
    for mode in modes(args):
        if args.stage == "attend":
            packed = risa.prequantize_int8_attention(q, k, v, v_quantization=mode)
            methods[mode] = lambda packed=packed: risa.int8_attention_from_prequantized(packed)
        else:
            operation = (
                risa.int8_attention if args.stage == "dense" else risa.prequantize_int8_attention
            )
            methods[mode] = lambda mode=mode: operation(q, k, v, v_quantization=mode)
    for method in methods.values():
        for _ in range(args.warmup):
            method()
    torch.cuda.synchronize()
    ensure_idle(args)
    print(f"{args.case.label} {args.layout} {args.stage}: {tuple(methods)}", flush=True)
    torch.cuda.cudart().cudaProfilerStart()
    try:
        for _ in range(args.calls):
            for mode, method in methods.items():
                with torch.cuda.nvtx.range(f"risa/{args.stage}/{mode}"):
                    method()
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    ensure_idle(args)


if __name__ == "__main__":
    main()
