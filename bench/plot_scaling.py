#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THROUGHPUT_IMPLEMENTATIONS = (
    ("torch_sdpa", "PyTorch SDPA", "#7a7a7a", ""),
    ("sageattention2_official", "SageAttention2", "#e76f51", "//"),
    ("sageattention3_official", "SageAttention3", "#f59e0b", "//"),
    ("comfy_kitchen_baseline", "comfy-kitchen INT8", "#facc15", "\\\\"),
    ("risa_dense_fused", "RISA dense INT8", "#159957", ""),
    ("risa_sparse_fused", "RISA sparse INT8", "#159eaa", "xx"),
)
ACCURACY_IMPLEMENTATIONS = THROUGHPUT_IMPLEMENTATIONS[1:]
PATTERN_LABELS = {
    "normal": "normal Q/K negative control",
    "video_blocks": "block-structured Q/K",
}
REQUIRED_IMPLEMENTATIONS = tuple(
    implementation for implementation, *_ in THROUGHPUT_IMPLEMENTATIONS
)


def _length(case: str) -> int:
    match = re.search(r"-L(\d+)-", case)
    if match is None:
        raise ValueError(f"cannot parse sequence length from case {case!r}")
    return int(match.group(1))


def _length_label(length: int) -> str:
    return f"{length // 1024}K" if length % 1024 == 0 else str(length)


def _load(path: Path) -> tuple[dict, dict[tuple[str, int, str], list[dict]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for result in payload["results"]:
        key = (
            result["pattern"],
            _length(result["case"]),
            result["implementation"],
        )
        rows[key].append(result)
    return payload, dict(rows)


def _validate(
    rows: dict[tuple[str, int, str], list[dict]], pattern: str
) -> list[int]:
    lengths = sorted(
        length
        for row_pattern, length, implementation in rows
        if row_pattern == pattern and implementation == "torch_sdpa"
    )
    if not lengths:
        raise ValueError(f"the JSON contains no {pattern} PyTorch SDPA results")
    missing = [
        (pattern, length, implementation)
        for length in lengths
        for implementation in REQUIRED_IMPLEMENTATIONS
        if (pattern, length, implementation) not in rows
    ]
    if missing:
        preview = ", ".join(str(item) for item in missing[:4])
        raise ValueError(f"benchmark JSON is missing required results: {preview}")
    missing_metrics = [
        (pattern, length, implementation, metric)
        for length in lengths
        for implementation in REQUIRED_IMPLEMENTATIONS
        for metric in ("dense_equivalent_tflops", "sqnr_db_vs_sdpa")
        if any(row.get(metric) is None for row in rows[(pattern, length, implementation)])
    ]
    if missing_metrics:
        preview = ", ".join(str(item) for item in missing_metrics[:4])
        raise ValueError(f"benchmark JSON is missing required metrics: {preview}")
    return lengths


def _aggregate(
    rows: dict[tuple[str, int, str], list[dict]],
    pattern: str,
    length: int,
    implementation: str,
    metric: str,
    *,
    use_median: bool,
) -> tuple[float, float]:
    values = np.asarray(
        [row[metric] for row in rows[(pattern, length, implementation)]],
        dtype=np.float64,
    )
    center = float(np.median(values) if use_median else np.mean(values))
    spread = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return center, spread


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RISA 1K-32K scaling from benchmark_sparse.py JSON"
    )
    parser.add_argument("json", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("bench/risa_scaling_1k_32k.png")
    )
    parser.add_argument("--svg", type=Path)
    parser.add_argument(
        "--memory-output", type=Path, default=Path("bench/risa_memory_1k_32k.png")
    )
    parser.add_argument("--memory-svg", type=Path)
    parser.add_argument(
        "--pattern", choices=tuple(PATTERN_LABELS), default="video_blocks"
    )
    args = parser.parse_args()

    payload, rows = _load(args.json)
    lengths = _validate(rows, args.pattern)
    x = np.arange(len(lengths), dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(17.2, 6.4))

    panel_specs = (
        (
            axes[0],
            ACCURACY_IMPLEMENTATIONS,
            "sqnr_db_vs_sdpa",
            "Attention accuracy",
            "SQNR vs FP32 PyTorch SDPA (dB, higher is better)",
            ".1f",
        ),
        (
            axes[1],
            THROUGHPUT_IMPLEMENTATIONS,
            "dense_equivalent_tflops",
            "Attention speed",
            "Dense-equivalent throughput (TFLOP/s)",
            ".0f",
        ),
    )
    for axis, implementations, metric, title, ylabel, value_format in panel_specs:
        width = 0.76 / len(implementations)
        offsets = (
            np.arange(len(implementations)) - (len(implementations) - 1) / 2
        ) * width
        use_median = metric == "dense_equivalent_tflops"
        panel_aggregates = [
            _aggregate(
                rows,
                args.pattern,
                length,
                implementation,
                metric,
                use_median=use_median,
            )
            for length in lengths
            for implementation, *_ in implementations
        ]
        upper = max(
            center + (0.0 if use_median else spread)
            for center, spread in panel_aggregates
        ) * 1.20
        for offset, (implementation, label, color, hatch) in zip(
            offsets, implementations, strict=True
        ):
            aggregates = [
                _aggregate(
                    rows,
                    args.pattern,
                    length,
                    implementation,
                    metric,
                    use_median=use_median,
                )
                for length in lengths
            ]
            heights = [center for center, _ in aggregates]
            errors = [spread for _, spread in aggregates]
            bars = axis.bar(
                x + offset,
                heights,
                width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=1.4,
                hatch=hatch,
                yerr=errors if not use_median else None,
                capsize=3 if not use_median else 0,
                error_kw={"ecolor": "black", "linewidth": 1.0},
            )
            for bar, height in zip(bars, heights, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + upper * 0.012,
                    format(height, value_format),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=8.5,
                )
        axis.set_title(title, fontsize=15)
        axis.set_ylabel(ylabel, fontsize=12)
        axis.set_xlabel("Sequence length", fontsize=13)
        axis.set_xticks(x, [_length_label(length) for length in lengths])
        axis.set_ylim(0, upper)
        axis.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)

    device = payload.get("device", "NVIDIA GPU")
    theta = payload.get("theta", 0.99)
    dtype_name = str(payload.get("dtype", "torch.bfloat16")).removeprefix("torch.")
    dtype = {"bfloat16": "BF16", "float16": "FP16"}.get(
        dtype_name, dtype_name.upper()
    )
    figure.suptitle(
        "RISA Attention scaling on "
        f"{device} | D=128, {dtype}, theta={theta:g}",
        fontsize=16,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncols=3,
        fontsize=9.5,
        frameon=False,
    )
    figure.text(
        0.5,
        0.018,
        f"{PATTERN_LABELS[args.pattern]}; "
        "SQNR is mean +/- 1 standard deviation; speed is median across seeds.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.16, 1, 0.93))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.svg, bbox_inches="tight")
    plt.close(figure)

    memory_figure, memory_axis = plt.subplots(figsize=(10.2, 6.2))
    memory_width = 0.76 / len(THROUGHPUT_IMPLEMENTATIONS)
    memory_offsets = (
        np.arange(len(THROUGHPUT_IMPLEMENTATIONS))
        - (len(THROUGHPUT_IMPLEMENTATIONS) - 1) / 2
    ) * memory_width
    memory_values = []
    for offset, (implementation, label, color, hatch) in zip(
        memory_offsets, THROUGHPUT_IMPLEMENTATIONS, strict=True
    ):
        heights = [
            _aggregate(
                rows,
                args.pattern,
                length,
                implementation,
                "peak_incremental_mib",
                use_median=True,
            )[0]
            for length in lengths
        ]
        memory_values.extend(heights)
        bars = memory_axis.bar(
            x + offset,
            heights,
            memory_width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=1.3,
            hatch=hatch,
        )
        for bar, height in zip(bars, heights, strict=True):
            memory_axis.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8.0,
            )
    memory_axis.set_title("Peak temporary CUDA allocation", fontsize=15)
    memory_axis.set_ylabel("Incremental allocation (MiB)", fontsize=12)
    memory_axis.set_xlabel("Sequence length", fontsize=13)
    memory_axis.set_xticks(x, [_length_label(length) for length in lengths])
    memory_axis.set_ylim(0, max(memory_values) * 1.20)
    memory_axis.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    memory_axis.set_axisbelow(True)
    memory_figure.suptitle(
        "RISA Attention memory on "
        f"{device} | D=128, {dtype}, theta={theta:g}",
        fontsize=16,
    )
    handles, labels = memory_axis.get_legend_handles_labels()
    memory_figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncols=3,
        fontsize=9.5,
        frameon=False,
    )
    memory_figure.text(
        0.5,
        0.018,
        "Median peak allocation above live inputs; RISA pattern index storage is separate.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    memory_figure.tight_layout(rect=(0, 0.16, 1, 0.93))
    args.memory_output.parent.mkdir(parents=True, exist_ok=True)
    memory_figure.savefig(args.memory_output, dpi=180, bbox_inches="tight")
    if args.memory_svg is not None:
        args.memory_svg.parent.mkdir(parents=True, exist_ok=True)
        memory_figure.savefig(args.memory_svg, bbox_inches="tight")
    plt.close(memory_figure)


if __name__ == "__main__":
    main()
