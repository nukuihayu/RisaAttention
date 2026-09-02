#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import gc
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
        if case.q_length != case.kv_length:
            raise argparse.ArgumentTypeError(
                "retained-mass benchmark requires self-attention Lq=Lkv"
            )
        if case.head_dim > 256:
            raise argparse.ArgumentTypeError("D must be <= 256")
        return case

    @property
    def label(self) -> str:
        return (
            f"B{self.batch}-H{self.q_heads}/{self.kv_heads}-"
            f"L{self.q_length}-D{self.head_dim}"
        )


@dataclass(frozen=True)
class ErrorMetrics:
    nrmse_vs_sdpa: float
    relative_l1_vs_sdpa: float
    mae_vs_sdpa: float
    max_abs_vs_sdpa: float
    cosine_vs_sdpa: float
    sqnr_db_vs_sdpa: float
    nrmse_vs_dense_int8: float | None


@dataclass(frozen=True)
class MemoryMetrics:
    live_allocated_mib: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    peak_incremental_mib: float


@dataclass
class Result:
    seed: int
    case: str
    pattern: str
    implementation: str
    median_ms: float
    p10_ms: float
    p90_ms: float
    min_ms: float
    live_allocated_mib: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    peak_incremental_mib: float
    speedup_vs_sdpa: float | None
    query_tokens_per_second: float | None
    dense_equivalent_tflops: float | None
    executed_tflops: float | None
    nrmse_vs_sdpa: float | None
    relative_l1_vs_sdpa: float | None
    mae_vs_sdpa: float | None
    max_abs_vs_sdpa: float | None
    cosine_vs_sdpa: float | None
    sqnr_db_vs_sdpa: float | None
    nrmse_vs_dense_int8: float | None
    coverage: float | None
    sparsity: float | None
    construction_recall: float | None
    exact_construction_recall: float | None
    current_recall: float | None
    pattern_index_mib: float | None


DEFAULT_CASES = (
    Case(1, 16, 4, 1024, 1024, 128),
    Case(1, 16, 4, 2048, 2048, 128),
    Case(1, 16, 4, 4096, 4096, 128),
    Case(1, 16, 4, 8192, 8192, 128),
    Case(1, 16, 4, 16384, 16384, 128),
    Case(1, 16, 4, 32768, 32768, 128),
)


def _dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[value.lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError("dtype must be float16 or bfloat16") from error


def _measure(
    function: Callable[[], object], warmup: int, iterations: int
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


def _peak_memory(function: Callable[[], object]) -> MemoryMetrics:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del output
    mib = 1024**2
    return MemoryMetrics(
        live_allocated_mib=baseline_allocated / mib,
        peak_allocated_mib=peak_allocated / mib,
        peak_reserved_mib=peak_reserved / mib,
        peak_incremental_mib=max(0, peak_allocated - baseline_allocated) / mib,
    )


def _errors(
    actual: torch.Tensor,
    reference: torch.Tensor,
    dense_int8: torch.Tensor | None = None,
) -> ErrorMetrics:
    actual_f = actual.float()
    reference_f = reference.float()
    difference = actual_f - reference_f
    noise = difference.square().mean().sqrt()
    signal = reference_f.square().mean().sqrt().clamp_min(1e-12)
    nrmse = noise / signal
    relative_l1 = difference.abs().mean() / reference_f.abs().mean().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), reference_f.flatten(), dim=0
    )
    sqnr = 20.0 * torch.log10(signal / noise.clamp_min(1e-12))
    incremental = None
    if dense_int8 is not None:
        incremental = float(
            (actual_f - dense_int8.float()).square().mean().sqrt()
            / dense_int8.float().square().mean().sqrt().clamp_min(1e-12)
        )
    return ErrorMetrics(
        nrmse_vs_sdpa=float(nrmse),
        relative_l1_vs_sdpa=float(relative_l1),
        mae_vs_sdpa=float(difference.abs().mean()),
        max_abs_vs_sdpa=float(difference.abs().max()),
        cosine_vs_sdpa=float(cosine),
        sqnr_db_vs_sdpa=float(sqnr),
        nrmse_vs_dense_int8=incremental,
    )


def _video_blocks(
    case: Case,
    dtype: torch.dtype,
    clusters: int,
    prototype_norm: float,
    drift: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    q0 = torch.randn(
        case.batch,
        case.q_heads,
        case.q_length,
        case.head_dim,
        device=device,
        dtype=dtype,
    )
    k0 = torch.randn(
        case.batch,
        case.kv_heads,
        case.kv_length,
        case.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k0)
    prototypes = torch.randn(
        case.batch,
        case.kv_heads,
        clusters,
        case.head_dim,
        device=device,
        dtype=torch.float32,
    )
    prototypes.mul_(prototype_norm / prototypes.norm(dim=-1, keepdim=True))
    q_cluster = torch.arange(case.q_length, device=device).div(
        128, rounding_mode="floor"
    )
    q_cluster.remainder_(clusters)
    k_cluster = torch.arange(case.kv_length, device=device).div(
        128, rounding_mode="floor"
    )
    k_cluster.remainder_(clusters)
    group = case.q_heads // case.kv_heads
    q_prototypes = prototypes.repeat_interleave(group, dim=1)
    q0.add_(q_prototypes[:, :, q_cluster].to(dtype))
    k0.add_(prototypes[:, :, k_cluster].to(dtype))
    q = q0 + torch.randn_like(q0) * drift
    k = k0 + torch.randn_like(k0) * drift
    return q0, k0, q, k, v


def _normal(
    case: Case, dtype: torch.dtype, drift: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q0 = torch.randn(
        case.batch,
        case.q_heads,
        case.q_length,
        case.head_dim,
        device="cuda",
        dtype=dtype,
    )
    k0 = torch.randn(
        case.batch,
        case.kv_heads,
        case.kv_length,
        case.head_dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn_like(k0)
    return (
        q0,
        k0,
        q0 + torch.randn_like(q0) * drift,
        k0 + torch.randn_like(k0) * drift,
        v,
    )


def _make_result(
    seed: int,
    case: Case,
    pattern_name: str,
    name: str,
    samples: list[float],
    memory: MemoryMetrics,
    baseline_ms: float | None,
    errors: ErrorMetrics | None,
    coverage: float | None,
    sparse_pattern: risa.RetainedMassPattern | None,
    current_recall: float | None,
    exact_construction_recall: float | None = None,
) -> Result:
    ordered = sorted(samples)
    median_ms = statistics.median(ordered)
    is_compute = name not in ("retained_mass_pattern_build", "risa_quantize_only")
    dense_operations = (
        4 * case.batch * case.q_heads * case.q_length * case.kv_length * case.head_dim
    )
    executed_operations = dense_operations * (coverage if coverage is not None else 1.0)
    seconds = median_ms / 1000.0
    return Result(
        seed=seed,
        case=case.label,
        pattern=pattern_name,
        implementation=name,
        median_ms=median_ms,
        p10_ms=ordered[int(0.1 * (len(ordered) - 1))],
        p90_ms=ordered[int(0.9 * (len(ordered) - 1))],
        min_ms=ordered[0],
        live_allocated_mib=memory.live_allocated_mib,
        peak_allocated_mib=memory.peak_allocated_mib,
        peak_reserved_mib=memory.peak_reserved_mib,
        peak_incremental_mib=memory.peak_incremental_mib,
        speedup_vs_sdpa=(
            baseline_ms / median_ms if baseline_ms and is_compute else None
        ),
        query_tokens_per_second=(
            case.batch * case.q_length / seconds if is_compute else None
        ),
        dense_equivalent_tflops=(
            dense_operations / seconds / 1e12 if is_compute else None
        ),
        executed_tflops=(executed_operations / seconds / 1e12 if is_compute else None),
        nrmse_vs_sdpa=(errors.nrmse_vs_sdpa if errors else None),
        relative_l1_vs_sdpa=(errors.relative_l1_vs_sdpa if errors else None),
        mae_vs_sdpa=(errors.mae_vs_sdpa if errors else None),
        max_abs_vs_sdpa=(errors.max_abs_vs_sdpa if errors else None),
        cosine_vs_sdpa=(errors.cosine_vs_sdpa if errors else None),
        sqnr_db_vs_sdpa=(errors.sqnr_db_vs_sdpa if errors else None),
        nrmse_vs_dense_int8=(errors.nrmse_vs_dense_int8 if errors else None),
        coverage=coverage,
        sparsity=(1.0 - coverage if coverage is not None else None),
        construction_recall=(
            sparse_pattern.measured_retained_mass if sparse_pattern else None
        ),
        exact_construction_recall=exact_construction_recall,
        current_recall=current_recall,
        pattern_index_mib=(
            sparse_pattern.index_bytes / 1024**2 if sparse_pattern else None
        ),
    )


def _run_case(
    seed: int,
    case: Case,
    pattern_name: str,
    dtype: torch.dtype,
    theta: float,
    drift: float,
    clusters: int,
    prototype_norm: float,
    construction_iterations: int,
    warmup: int,
    iterations: int,
    compare_comfy_kitchen: bool,
    compare_sageattention2: bool,
    compare_sageattention3: bool,
) -> list[Result]:
    if pattern_name == "video_blocks":
        q0, k0, q, k, v = _video_blocks(case, dtype, clusters, prototype_norm, drift)
    else:
        q0, k0, q, k, v = _normal(case, dtype, drift)

    build = lambda: risa.construct_sparse_int8_attention(q0, k0, v, theta=theta)
    build_samples = _measure(build, 1, construction_iterations)
    build_peak = _peak_memory(build)
    _, sparse_pattern = build()
    exact_construction_recall = risa.measure_pattern_recall(q0, k0, sparse_pattern)
    current_recall = risa.measure_pattern_recall(q, k, sparse_pattern)
    build = None
    q0 = None
    k0 = None

    group = case.q_heads // case.kv_heads
    baseline_k = k.repeat_interleave(group, dim=1) if group > 1 else k
    baseline_v = v.repeat_interleave(group, dim=1) if group > 1 else v
    sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(
        q, baseline_k, baseline_v
    )
    dense_fused = lambda: risa.int8_attention(q, k, v)
    sparse_fused = lambda: risa.sparse_int8_attention(q, k, v, sparse_pattern)
    packed = risa.prequantize_int8_attention(q, k, v)
    dense_prequantized = lambda: risa.int8_attention_from_prequantized(packed)
    dense_flatten_hnd = lambda: (
        risa.int8_attention_from_prequantized(packed, output_layout="hnd")
        .transpose(1, 2)
        .reshape(case.batch, case.q_length, -1)
    )
    dense_flatten_nhd = lambda: (
        risa.int8_attention_from_prequantized(packed, output_layout="nhd")
        .transpose(1, 2)
        .reshape(case.batch, case.q_length, -1)
    )
    sparse_prequantized = lambda: risa.int8_attention_from_prequantized(
        packed, sparse_pattern=sparse_pattern
    )
    sparse_flatten_hnd = lambda: (
        risa.int8_attention_from_prequantized(
            packed, sparse_pattern=sparse_pattern, output_layout="hnd"
        )
        .transpose(1, 2)
        .reshape(case.batch, case.q_length, -1)
    )
    sparse_flatten_nhd = lambda: (
        risa.int8_attention_from_prequantized(
            packed, sparse_pattern=sparse_pattern, output_layout="nhd"
        )
        .transpose(1, 2)
        .reshape(case.batch, case.q_length, -1)
    )
    quantize = lambda: risa.prequantize_int8_attention(q, k, v)

    # FP32 SDPA is evaluated once per case and excluded from all timing loops.
    # It isolates approximation error from the BF16 reference rounding itself.
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(), baseline_k.float(), baseline_v.float()
    )
    dense_output = dense_fused()
    sparse_output = sparse_fused()
    implementations = [
        ("torch_sdpa", sdpa, _errors(reference, reference), None),
        ("risa_dense_fused", dense_fused, _errors(dense_output, reference), None),
        (
            "risa_dense_prequantized",
            dense_prequantized,
            _errors(dense_output, reference),
            None,
        ),
        (
            "risa_dense_flatten_hnd",
            dense_flatten_hnd,
            _errors(dense_output, reference),
            None,
        ),
        (
            "risa_dense_flatten_nhd",
            dense_flatten_nhd,
            _errors(dense_output, reference),
            None,
        ),
        (
            "risa_sparse_fused",
            sparse_fused,
            _errors(sparse_output, reference, dense_output),
            sparse_pattern.coverage,
        ),
        (
            "risa_sparse_prequantized",
            sparse_prequantized,
            _errors(sparse_output, reference, dense_output),
            sparse_pattern.coverage,
        ),
        (
            "risa_sparse_flatten_hnd",
            sparse_flatten_hnd,
            _errors(sparse_output, reference, dense_output),
            sparse_pattern.coverage,
        ),
        (
            "risa_sparse_flatten_nhd",
            sparse_flatten_nhd,
            _errors(sparse_output, reference, dense_output),
            sparse_pattern.coverage,
        ),
        ("risa_quantize_only", quantize, None, None),
    ]
    if compare_comfy_kitchen:
        try:
            import comfy_kitchen
        except ImportError as error:
            raise RuntimeError(
                "--compare-comfy-kitchen requires comfy-kitchen"
            ) from error
        if not comfy_kitchen.int8_attention_is_available():
            raise RuntimeError(
                "the comfy-kitchen INT8 attention backend is unavailable"
            )
        initial_output = comfy_kitchen.int8_attention(q, k, v)
        implementations.insert(
            2,
            (
                "comfy_kitchen_baseline",
                lambda: comfy_kitchen.int8_attention(q, k, v),
                _errors(initial_output, reference),
                None,
            ),
        )
    if compare_sageattention2:
        try:
            from sageattention import sageattn
        except ImportError as error:
            raise RuntimeError(
                "--compare-sageattention2 requires the sageattention package"
            ) from error

        def sageattention2() -> torch.Tensor:
            return sageattn(q, k.clone(), v, tensor_layout="HND")

        sage_output = sageattention2()
        implementations.insert(
            2,
            (
                "sageattention2_official",
                sageattention2,
                _errors(sage_output, reference),
                None,
            ),
        )
    if compare_sageattention3:
        if case.q_heads != case.kv_heads:
            raise RuntimeError(
                "--compare-sageattention3 requires Hq=Hkv; SageAttention3 has "
                "no GQA/MQA kernel"
            )
        try:
            from sageattn3 import sageattn3_blackwell
        except ImportError as error:
            raise RuntimeError(
                "--compare-sageattention3 requires the sageattn3 package"
            ) from error

        def sageattention3() -> torch.Tensor:
            # The current official API modifies K in place. Clone to make each
            # timing sample operate on the same inputs and preserve caller data.
            return sageattn3_blackwell(q, k.clone(), v, is_causal=False)

        sage_output = sageattention3()
        implementations.insert(
            2,
            (
                "sageattention3_official",
                sageattention3,
                _errors(sage_output, reference),
                None,
            ),
        )

    results = [
        _make_result(
            seed,
            case,
            pattern_name,
            "retained_mass_pattern_build",
            build_samples,
            build_peak,
            None,
            None,
            sparse_pattern.coverage,
            sparse_pattern,
            current_recall,
            exact_construction_recall,
        )
    ]
    measured: list[
        tuple[str, list[float], float, ErrorMetrics | None, float | None]
    ] = []
    for name, function, errors, coverage in implementations:
        samples = _measure(function, warmup, iterations)
        peak = _peak_memory(function)
        measured.append((name, samples, peak, errors, coverage))
    baseline_ms = statistics.median(measured[0][1])
    for name, samples, peak, errors, coverage in measured:
        is_sparse = name.startswith("risa_sparse_")
        results.append(
            _make_result(
                seed,
                case,
                pattern_name,
                name,
                samples,
                peak,
                baseline_ms,
                errors,
                coverage,
                sparse_pattern if is_sparse else None,
                current_recall if is_sparse else None,
                exact_construction_recall if is_sparse else None,
            )
        )
    return results


def _print_table(results: list[Result]) -> None:
    header = (
        f"{'case':<29} {'pattern':<12} {'implementation':<25} {'median':>9} "
        f"{'p10/p90 ms':>17} {'speedup':>8} {'+peak MiB':>9} {'coverage':>9} "
        f"{'recall':>8} {'NRMSE':>9} {'SQNR dB':>8} {'cosine':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        speedup = (
            "-" if result.speedup_vs_sdpa is None else f"{result.speedup_vs_sdpa:.2f}x"
        )
        coverage = "-" if result.coverage is None else f"{result.coverage:.3f}"
        recall = (
            "-" if result.current_recall is None else f"{result.current_recall:.4f}"
        )
        nrmse = "-" if result.nrmse_vs_sdpa is None else f"{result.nrmse_vs_sdpa:.4f}"
        sqnr = (
            "-" if result.sqnr_db_vs_sdpa is None else f"{result.sqnr_db_vs_sdpa:.2f}"
        )
        cosine = (
            "-" if result.cosine_vs_sdpa is None else f"{result.cosine_vs_sdpa:.6f}"
        )
        print(
            f"{result.case:<29} {result.pattern:<12} {result.implementation:<25} "
            f"{result.median_ms:>8.3f} {result.p10_ms:>7.3f}/{result.p90_ms:<7.3f} "
            f"{speedup:>8} {result.peak_incremental_mib:>9.1f} {coverage:>9} "
            f"{recall:>8} {nrmse:>9} {sqnr:>8} {cosine:>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ConvRot INT8 dense and retained-mass sparse attention"
    )
    parser.add_argument("--case", action="append", type=Case.parse)
    parser.add_argument("--dtype", type=_dtype, default=torch.bfloat16)
    parser.add_argument(
        "--pattern", action="append", choices=("video_blocks", "normal")
    )
    parser.add_argument("--theta", type=float, default=0.99)
    parser.add_argument("--drift", type=float, default=0.05)
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--prototype-norm", type=float, default=8.0)
    parser.add_argument("--construction-iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--compare-comfy-kitchen", action="store_true")
    parser.add_argument("--compare-sageattention2", action="store_true")
    parser.add_argument("--compare-sageattention3", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.theta <= 1.0:
        parser.error("theta must be in (0, 1]")
    if args.drift < 0 or args.clusters <= 0 or args.prototype_norm <= 0:
        parser.error("drift must be >= 0; clusters and prototype-norm must be > 0")
    if args.warmup < 0 or args.iterations <= 0 or args.construction_iterations <= 0:
        parser.error("warmup must be >= 0 and iteration counts must be > 0")
    if not torch.cuda.is_available() or not risa.int8_attention_is_available():
        parser.error("an NVIDIA SM75+ CUDA runtime is required")

    patterns = args.pattern or ["video_blocks"]
    results = []
    seeds = args.seed or [0]
    for seed in seeds:
        for case in args.case or DEFAULT_CASES:
            for pattern_name in patterns:
                torch.manual_seed(seed)
                results.extend(
                    _run_case(
                        seed,
                        case,
                        pattern_name,
                        args.dtype,
                        args.theta,
                        args.drift,
                        args.clusters,
                        args.prototype_norm,
                        args.construction_iterations,
                        args.warmup,
                        args.iterations,
                        args.compare_comfy_kitchen,
                        args.compare_sageattention2,
                        args.compare_sageattention3,
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
            "theta": args.theta,
            "drift": args.drift,
            "clusters": args.clusters,
            "prototype_norm": args.prototype_norm,
            "construction_iterations": args.construction_iterations,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seeds": seeds,
            "compare_comfy_kitchen": args.compare_comfy_kitchen,
            "reference": "float32 PyTorch SDPA",
            "compare_sageattention2": args.compare_sageattention2,
            "compare_sageattention3": args.compare_sageattention3,
            "patterns": patterns,
            "results": [asdict(result) for result in results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
