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
    ("sage", "SageAttention (auto)", "#e76f51", "//"),
    ("comfy_kitchen_baseline", "comfy-kitchen INT8", "#facc15", "\\\\"),
    ("dense/absmax", "RISA dense, affine off", "#159957", ""),
    ("dense/midpoint_affine", "RISA dense, affine on", "#91cf9e", ""),
    ("sparse/absmax", "RISA sparse, affine off", "#159eaa", "xx"),
    ("sparse/midpoint_affine", "RISA sparse, affine on", "#90d9df", "xx"),
)
PATTERN_LABELS = {"normal": "normal Q/K", "video_blocks": "block-structured Q/K"}
REQUIRED_IMPLEMENTATIONS = ("torch_sdpa",)


def _length(case: str) -> int:
    match = re.search(r"-L(\d+)/", case)
    if match is None:
        raise ValueError(f"cannot parse sequence length from case {case!r}")
    return int(match.group(1))


def _length_label(length: int) -> str:
    return f"{length // 1024}K" if length % 1024 == 0 else str(length)


def _load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("expected schema_version=2")
    rows = defaultdict(list)
    geometries = set()
    for source in payload["results"]:
        if source["stage"] not in ("dense", "sparse") or source["value_distribution"] != "normal":
            continue
        result = dict(source)
        implementation = result["implementation"]
        if implementation == "risa":
            implementation = result["stage"] + "/" + result["v_quantization"]
        else:
            implementation = {"sdpa": "torch_sdpa", "kitchen": "comfy_kitchen_baseline"}.get(
                implementation, implementation
            )
        shape = result["shape"]
        if shape["q_length"] != shape["kv_length"]:
            raise ValueError("scaling plots require equal Q/K lengths")
        geometries.add(
            (
                shape["batch"],
                shape["q_heads"],
                shape["kv_heads"],
                shape["head_dim"],
                source["layout"],
            )
        )
        result["dense_equivalent_tflops"] = (
            4
            * shape["batch"]
            * shape["q_heads"]
            * shape["q_length"]
            * shape["kv_length"]
            * shape["head_dim"]
            / result["timing"]["median_ms"]
            / 1e9
        )
        result["sqnr_db_vs_sdpa"] = result["accuracy"]["sqnr_db"]
        result["peak_incremental_mib"] = result["memory"]["peak_incremental_mib"]
        rows[(result["qk_distribution"], shape["q_length"], implementation)].append(result)
    if len(geometries) != 1:
        raise ValueError("plot one batch/head/dimension/layout configuration per file")
    return payload, dict(rows)


def _validate(rows: dict[tuple[str, int, str], list[dict]], pattern: str) -> list[int]:
    lengths = sorted(
        {
            length
            for row_pattern, length, implementation in rows
            if row_pattern == pattern and implementation == "torch_sdpa"
        }
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
        for metric in ("dense_equivalent_tflops",)
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
    parser = argparse.ArgumentParser(description="Plot schema-v2 attention results")
    parser.add_argument("json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--memory-output", type=Path, required=True)
    parser.add_argument("--memory-svg", type=Path)
    parser.add_argument("--pattern", choices=tuple(PATTERN_LABELS), default="video_blocks")
    args = parser.parse_args()

    payload, rows = _load(args.json)
    lengths = _validate(rows, args.pattern)
    throughput_implementations = tuple(
        entry
        for entry in THROUGHPUT_IMPLEMENTATIONS
        if all((args.pattern, length, entry[0]) in rows for length in lengths)
    )
    accuracy_implementations = tuple(
        entry for entry in throughput_implementations if entry[0] != "torch_sdpa"
    )
    x = np.arange(len(lengths), dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(17.2, 6.4))

    panel_specs = (
        (
            axes[0],
            accuracy_implementations,
            "sqnr_db_vs_sdpa",
            "Attention accuracy",
            "SQNR vs FP32 PyTorch SDPA (dB, higher is better)",
            ".1f",
        ),
        (
            axes[1],
            throughput_implementations,
            "dense_equivalent_tflops",
            "Attention speed",
            "Dense-equivalent throughput (TFLOP/s)",
            ".0f",
        ),
    )
    for axis, implementations, metric, title, ylabel, value_format in panel_specs:
        width = 0.76 / len(implementations)
        offsets = (np.arange(len(implementations)) - (len(implementations) - 1) / 2) * width
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
        upper = (
            max(center + (0.0 if use_median else spread) for center, spread in panel_aggregates)
            * 1.20
        )
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
            for bar, height, error in zip(bars, heights, errors, strict=True):
                axis.annotate(
                    format(height, value_format),
                    xy=(
                        bar.get_x() + bar.get_width() / 2,
                        height + (0.0 if use_median else error),
                    ),
                    xytext=(0, 6),
                    textcoords="offset points",
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

    device = payload["environment"]["gpu"]
    theta_label = f", theta={payload['config']['theta']:g}" if "theta" in payload["config"] else ""
    dtype_name = payload["config"]["dtype"]
    dtype = {"bfloat16": "BF16", "float16": "FP16"}.get(dtype_name, dtype_name.upper())
    shape = re.sub(r"-L\d+/\d+", "", payload["results"][0]["case"])
    figure.suptitle(
        f"RISA Attention scaling on {device} | {shape}, {dtype}{theta_label}",
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
    memory_width = 0.76 / len(throughput_implementations)
    memory_offsets = (
        np.arange(len(throughput_implementations)) - (len(throughput_implementations) - 1) / 2
    ) * memory_width
    memory_values = []
    for offset, (implementation, label, color, hatch) in zip(
        memory_offsets, throughput_implementations, strict=True
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
            memory_axis.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
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
        f"RISA Attention memory on {device} | {shape}, {dtype}{theta_label}",
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
