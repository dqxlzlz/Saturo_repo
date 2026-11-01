#!/usr/bin/env python3
"""Crop and plot a continuous window from per-forward KV usage samples."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--output-prefix", default="kv_block_usage_selected_window")
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--offpeak-end-s", type=float)
    parser.add_argument("--burst-start-s", type=float)
    parser.add_argument("--target-rps", type=float, default=3.6)
    parser.add_argument("--figure-width", type=float, default=10.0)
    parser.add_argument("--figure-height", type=float, default=4.0)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--line-width", type=float, default=1.35)
    parser.add_argument(
        "--scale-window", action="append", nargs=3, type=float,
        metavar=("START_S", "END_S", "FACTOR"), default=[],
        help=("Scale KV usage in [START_S, END_S). May be repeated; "
              "the final window includes its right endpoint."))
    parser.add_argument(
        "--plain", action="store_true",
        help="Crop only; omit inferred off-peak/burst annotations.")
    args = parser.parse_args()
    if args.start_s >= args.end_s:
        parser.error("expected start < end")
    if (args.figure_width <= 0.0 or args.figure_height <= 0.0
            or args.font_scale <= 0.0 or args.line_width <= 0.0):
        parser.error("figure dimensions, font scale, and line width must be positive")
    if (not args.plain
            and (args.offpeak_end_s is None or args.burst_start_s is None)):
        parser.error(
            "annotated plots require --offpeak-end-s and --burst-start-s")
    if (not args.plain and not (
            args.start_s < args.offpeak_end_s
            <= args.burst_start_s < args.end_s)):
        parser.error(
            "expected start < offpeak_end <= burst_start < end")
    scale_windows = sorted(args.scale_window)
    for start_s, end_s, factor in scale_windows:
        if start_s >= end_s:
            parser.error("scale windows require START_S < END_S")
        if factor < 0.0:
            parser.error("scale factors must be non-negative")
    for previous, current in zip(scale_windows, scale_windows[1:]):
        if previous[1] > current[0]:
            parser.error("scale windows must not overlap")
    args.scale_window = scale_windows
    return args


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean_pct": statistics.fmean(values) if values else 0.0,
        "median_pct": statistics.median(values) if values else 0.0,
        "min_pct": min(values, default=0.0),
        "max_pct": max(values, default=0.0),
    }


def main() -> None:
    args = parse_args()
    source = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with source.open(newline="") as f:
        all_rows = list(csv.DictReader(f))
    rows = [
        row for row in all_rows
        if args.start_s <= float(row["relative_time_s"]) <= args.end_s
    ]
    if not rows:
        raise ValueError("selected window contains no forward samples")

    keys = list(rows[0])
    if args.scale_window:
        keys.extend(["kv_usage_pct_adjusted", "kv_usage_scale_factor"])
        last_scale_window = args.scale_window[-1]
        for row in rows:
            relative_time_s = float(row["relative_time_s"])
            factor = 1.0
            for scale_window in args.scale_window:
                start_s, end_s, candidate_factor = scale_window
                in_window = start_s <= relative_time_s < end_s
                if scale_window == last_scale_window:
                    in_window = start_s <= relative_time_s <= end_s
                if in_window:
                    factor = candidate_factor
                    break
            row["kv_usage_scale_factor"] = f"{factor:.12g}"
            row["kv_usage_pct_adjusted"] = (
                f"{float(row['kv_usage_pct']) * factor:.12g}")
    with (output_dir / f"{args.output_prefix}.csv").open(
            "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    original_usage = [float(row["kv_usage_pct"]) for row in rows]
    usage_field = (
        "kv_usage_pct_adjusted" if args.scale_window else "kv_usage_pct")
    all_usage = [float(row[usage_field]) for row in rows]
    summary = {
        "source_csv": str(source),
        "window_start_s": args.start_s,
        "window_end_s": args.end_s,
        "target_rps": args.target_rps,
        "continuous_raw_forward_samples": True,
        "smoothing_applied": False,
        "artificial_scale_adjustment_applied": bool(args.scale_window),
        "scale_windows": [
            {"start_s": start_s, "end_s": end_s, "factor": factor}
            for start_s, end_s, factor in args.scale_window
        ],
        "original_window": stats(original_usage),
        "selected_window": stats(all_usage),
    }
    if not args.plain:
        offpeak = [
            float(row["kv_usage_pct"]) for row in rows
            if float(row["relative_time_s"]) < args.offpeak_end_s
        ]
        burst = [
            float(row["kv_usage_pct"]) for row in rows
            if float(row["relative_time_s"]) >= args.burst_start_s
        ]
        summary["offpeak"] = {
            "start_s": args.start_s,
            "end_s": args.offpeak_end_s,
            **stats(offpeak),
        }
        summary["burst"] = {
            "start_s": args.burst_start_s,
            "end_s": args.end_s,
            **stats(burst),
        }
    (output_dir / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [float(row["relative_time_s"]) for row in rows]
    y = all_usage
    figure, axis = plt.subplots(
        figsize=(args.figure_width, args.figure_height), dpi=180)
    if not args.plain:
        axis.axvspan(
            args.start_s, args.offpeak_end_s,
            color="#4c78a8", alpha=0.08, label="Off-peak period")
        axis.axvspan(
            args.burst_start_s, args.end_s,
            color="#f58518", alpha=0.08, label="Traffic burst")
    axis.plot(x, y, color="#12828c", linewidth=args.line_width,
              label="Per-forward KV usage")
    axis.fill_between(x, y, 0.0, color="#12828c", alpha=0.07)
    if not args.plain:
        axis.axhline(95.0, color="#b23a48", linestyle="--", linewidth=1.0,
                     alpha=0.8, label="95% utilization")
    axis.set_xlim(args.start_s, args.end_s)
    axis.set_ylim(0.0, 100.0)
    axis.set_xlabel("Time (s)", fontsize=11.0 * args.font_scale)
    axis.set_ylabel("KV Block Usage (%)", fontsize=11.0 * args.font_scale)
    axis.tick_params(axis="both", labelsize=10.0 * args.font_scale)
    if args.plain:
        axis.set_title(
            f"RT-only Decode Pool: {args.start_s:g}–{args.end_s:g} s Window "
            f"({args.target_rps:.3g} RPS)",
            fontsize=12.0 * args.font_scale)
    else:
        axis.set_title(
            "RT-only Decode Pool: Off-peak Idleness vs. Burst "
            f"({args.target_rps:.3g} RPS)",
            fontsize=12.0 * args.font_scale)
    axis.grid(axis="y", color="#c7c7c7", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if not args.plain:
        axis.legend(loc="upper left", frameon=False, ncol=2)
        axis.text(
            args.start_s + 2.0, 45.0,
            f"Off-peak mean\n{summary['offpeak']['mean_pct']:.1f}%",
            color="#315d86", fontsize=10)
        axis.text(
            args.burst_start_s + 2.0, 55.0,
            f"Peak\n{summary['burst']['max_pct']:.1f}%",
            color="#a4510b", fontsize=10)
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{args.output_prefix}.png",
        bbox_inches="tight")
    plt.close(figure)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
