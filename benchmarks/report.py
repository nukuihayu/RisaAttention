# SPDX-License-Identifier: Apache-2.0
"""Export benchmark summaries as CSV and optional Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

GROUP_FIELDS = (
    "case",
    "layout",
    "qk_distribution",
    "value_distribution",
    "implementation",
    "v_quantization",
    "stage",
)


def summarize(payload):
    if payload.get("schema_version") != 2:
        raise ValueError("expected benchmark schema_version=2; rerun the benchmark")
    groups = defaultdict(list)
    for row in payload["results"]:
        groups[tuple(row[key] for key in GROUP_FIELDS)].append(row)
    summaries = []
    for key, rows in groups.items():
        if len({row["seed"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate seed in result group {key}")
        result = dict(zip(GROUP_FIELDS, key))
        result["seed_count"] = len(rows)
        result.update(
            {
                metric: statistics.median(row["timing"][metric] for row in rows)
                for metric in ("median_ms", "p10_ms", "p90_ms")
            }
        )
        errors = [row["accuracy"] for row in rows if row["accuracy"] is not None]
        for metric in ("nrmse", "rmse", "sqnr_db", "mae", "max_abs", "max_channel_bias"):
            values = [error[metric] for error in errors if error[metric] is not None]
            result[metric] = statistics.mean(values) if values else None
        result["nrmse_std"] = (
            statistics.stdev(error["nrmse"] for error in errors)
            if len(errors) > 1
            else 0
            if errors
            else None
        )
        memory = [row["memory"]["peak_incremental_mib"] for row in rows if row["memory"]]
        result["peak_incremental_mib"] = statistics.median(memory) if memory else None
        for metric in (
            "coverage",
            "pattern_index_mib",
            "exact_construction_recall",
            "current_recall",
        ):
            values = [row["sparse"][metric] for row in rows if row["sparse"]]
            result[metric] = statistics.mean(values) if values else None
        summaries.append(result)
    return summaries


def affine_table(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row["v_quantization"] in ("absmax", "midpoint_affine"):
            key = (
                row["implementation"],
                row["case"],
                row["layout"],
                row["qk_distribution"],
                row["value_distribution"],
                row["stage"],
            )
            groups[key][row["v_quantization"]] = row
    lines = [
        "| Implementation | Case | Layout | Q/K | V | Stage | Off (ms) | On (ms) | Time change | NRMSE off | NRMSE on | Error change |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, pair in groups.items():
        if set(pair) != {"absmax", "midpoint_affine"}:
            continue
        off, on = pair["absmax"], pair["midpoint_affine"]
        delta = (on["median_ms"] / off["median_ms"] - 1) * 100
        if off["nrmse"] is not None:
            off_error, on_error = f"{off['nrmse']:.6g}", f"{on['nrmse']:.6g}"
            change = f"{(on['nrmse'] / off['nrmse'] - 1) * 100:+.1f}%" if off["nrmse"] else "—"
        else:
            off_error = on_error = change = "—"
        lines.append(
            "| "
            + " | ".join(
                (
                    *key,
                    f"{off['median_ms']:.4f}",
                    f"{on['median_ms']:.4f}",
                    f"{delta:+.1f}%",
                    off_error,
                    on_error,
                    change,
                )
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.json.read_text())
    summaries = summarize(payload)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    table = affine_table(summaries)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
