#!/usr/bin/env python3
"""Reproduce the 4.215 RPS queue/execution timeline figure."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "requests_rps4.215.csv"


@dataclass(frozen=True)
class RequestTiming:
    request_index: int
    request_id: str
    arrival_s: float
    start_s: float
    finish_s: float
    queueing_s: float
    execution_s: float
    ttft_s: float
    slo_violated: bool
    prompt_len: int
    output_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot queue/execution timelines from a saved simulator CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--target-rps",
        type=float,
        default=None,
        help="Label shown in the title; default parses requests_rpsX.csv.",
    )
    parser.add_argument("--ttft-slo", type=float, default=0.5)
    parser.add_argument("--slo-percentile", type=float, default=0.99)
    parser.add_argument("--start-index", type=int, default=200)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument(
        "--start-s",
        type=float,
        default=73.043,
        help="Start time in the saved CSV timeline (requires --end-s).",
    )
    parser.add_argument(
        "--end-s",
        type=float,
        default=79.043,
        help="End time in the saved CSV timeline (requires --start-s).",
    )
    parser.add_argument(
        "--preview-duration-s",
        type=float,
        default=10.0,
        help="Also draw the leading time window; 0 disables it.",
    )
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: the input CSV's directory.",
    )
    parser.add_argument(
        "--height-factor",
        type=float,
        default=1.0,
        help="Scale the figure height relative to the default.",
    )
    parser.add_argument(
        "--width-factor",
        type=float,
        default=1.0,
        help="Scale the figure width relative to the default (12 inches).",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_timings(path: Path) -> list[RequestTiming]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"saved request CSV is empty: {path}")

    timings = [
        RequestTiming(
            request_index=int(row["request_index"]),
            request_id=row["request_id"],
            arrival_s=float(row["arrival_s"]),
            start_s=float(row["start_s"]),
            finish_s=float(row["finish_s"]),
            queueing_s=float(row["queueing_s"]),
            execution_s=float(row["execution_s"]),
            ttft_s=float(row["ttft_s"]),
            slo_violated=parse_bool(row["slo_violated"]),
            prompt_len=int(row["prompt_len"]),
            output_len=int(row["output_len"]),
        )
        for row in rows
    ]
    return sorted(timings, key=lambda timing: timing.request_index)


def infer_rps(path: Path) -> float:
    match = re.search(r"requests_rps([0-9]+(?:\.[0-9]+)?)", path.stem)
    if not match:
        raise ValueError(
            "cannot infer RPS from input filename; pass --target-rps"
        )
    return float(match.group(1))


def select_requests(
    timings: Sequence[RequestTiming], start_index: int, count: int
) -> list[RequestTiming]:
    if count <= 0:
        raise ValueError("requests must be positive")
    start = max(0, min(start_index, len(timings) - 1))
    window = list(timings[start : start + count])
    if not window:
        raise ValueError("selected request window is empty")
    return window


def leading_window(
    timings: Sequence[RequestTiming], duration_s: float
) -> list[RequestTiming]:
    if duration_s <= 0:
        return []
    end_s = timings[0].arrival_s + duration_s
    return [timing for timing in timings if timing.arrival_s < end_s]


def select_time_range(
    timings: Sequence[RequestTiming], start_s: float, end_s: float
) -> list[RequestTiming]:
    """Select requests whose queue/execution interval overlaps [start, end)."""
    if start_s < 0 or end_s <= start_s:
        raise ValueError("time range needs 0 <= start_s < end_s")
    window = [
        timing
        for timing in timings
        if timing.finish_s > start_s and timing.arrival_s < end_s
    ]
    if not window:
        raise ValueError(f"no requests overlap time range [{start_s}, {end_s})")
    return window


def merge_busy_intervals(
    timings: Sequence[RequestTiming], origin_s: float, end_s: float
) -> list[tuple[float, float]]:
    intervals = [
        (
            max(timing.start_s, origin_s) - origin_s,
            min(timing.finish_s, end_s) - origin_s,
        )
        for timing in timings
        if timing.finish_s > origin_s and timing.start_s < end_s
    ]
    merged: list[list[float]] = []
    for start, finish in intervals:
        if not merged or start > merged[-1][1] + 1e-12:
            merged.append([start, finish])
        else:
            merged[-1][1] = max(merged[-1][1], finish)
    return [(start, finish - start) for start, finish in merged]


def plot_timeline(
    path: Path,
    window: Sequence[RequestTiming],
    all_timings: Sequence[RequestTiming],
    target_rps: float,
    ttft_slo_s: float,
    slo_percentile: float,
    dpi: int,
    duration_s: float | None = None,
    origin_s_override: float | None = None,
    height_factor: float = 1.0,
    width_factor: float = 1.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not window:
        raise ValueError("cannot plot an empty request window")

    queue_color = "#A9C8E2"
    execution_color = "#3C86BF"
    violation_color = "#C83E4D"
    origin_s = (
        origin_s_override
        if origin_s_override is not None
        else window[0].arrival_s
    )
    natural_end_s = max(timing.finish_s for timing in window)
    display_end_s = origin_s + duration_s if duration_s else natural_end_s
    figure_height = max(3.5, 1.5 + 0.18 * len(window))

    figure, request_axis = plt.subplots(
        1,
        1,
        figsize=(12 * width_factor, figure_height * height_factor),
    )

    for row, timing in enumerate(window):
        arrival = timing.arrival_s - origin_s
        start = timing.start_s - origin_s
        slo_violated = timing.ttft_s > ttft_slo_s
        request_axis.broken_barh(
            [(arrival, timing.queueing_s)],
            (row - 0.38, 0.76),
            facecolors=queue_color,
        )
        request_axis.broken_barh(
            [(start, timing.execution_s)],
            (row - 0.38, 0.76),
            facecolors=execution_color,
            edgecolors=violation_color if slo_violated else "none",
            linewidth=1.5 if slo_violated else 0.0,
        )

    # 稀疏化 y 轴刻度：每 5 个请求显示一个标签
    tick_step = max(1, len(window) // 5)
    tick_positions = list(range(0, len(window), tick_step))
    request_axis.set_yticks(tick_positions, [str(i) for i in tick_positions], fontsize=32)
    request_axis.set_ylim(-0.8, len(window) - 0.2)
    request_axis.set_xlabel("Time(s)", fontsize=32)
    request_axis.set_ylabel("Request Index", fontsize=32)
    request_axis.tick_params(axis="x", labelsize=32)
    request_axis.grid(True, axis="both", linestyle="--", alpha=0.35)
    if duration_s is not None:
        request_axis.set_xlim(0.0, duration_s)
    request_axis.legend(
        handles=[
            Patch(color=queue_color, label="Queueing Time (s)"),
            Patch(color=execution_color, label="Execution Time (s)"),
        ],
        loc="upper left",
        fontsize=30,
    )
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.98)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.ttft_slo <= 0:
        raise ValueError("ttft_slo must be positive")
    if not 0 < args.slo_percentile <= 1:
        raise ValueError("slo_percentile must be in (0, 1]")
    if args.preview_duration_s < 0:
        raise ValueError("preview_duration_s must be non-negative")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")

    timings = load_timings(args.input)
    target_rps = args.target_rps or infer_rps(args.input)
    output_dir = args.output_dir or args.input.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    has_start_s = args.start_s is not None
    has_end_s = args.end_s is not None
    if has_start_s != has_end_s:
        raise ValueError("--start-s and --end-s must be provided together")
    if has_start_s and has_end_s:
        range_window = select_time_range(timings, args.start_s, args.end_s)
        range_path = output_dir / (
            f"queue_timeline_rps{target_rps:.3f}_"
            f"time{args.start_s:g}-{args.end_s:g}s.png"
        )
        plot_timeline(
            range_path,
            range_window,
            timings,
            target_rps,
            args.ttft_slo,
            args.slo_percentile,
            args.dpi,
            args.end_s - args.start_s,
            args.start_s,
            args.height_factor,
            args.width_factor,
        )
        print(
            f"[plot] time range {args.start_s:g}-{args.end_s:g}s "
            f"({len(range_window)} requests): {range_path}"
        )
        return

    window = select_requests(timings, args.start_index, args.requests)
    first_index = window[0].request_index
    last_index = window[-1].request_index
    prefix = f"queue_timeline_saved_rps{target_rps:.3f}"

    full_path = output_dir / f"{prefix}_req{first_index}-{last_index}.png"
    plot_timeline(
        full_path,
        window,
        timings,
        target_rps,
        args.ttft_slo,
        args.slo_percentile,
        args.dpi,
        height_factor=args.height_factor,
        width_factor=args.width_factor,
    )
    print(f"[plot] full: {full_path}")

    if args.preview_duration_s > 0:
        preview = leading_window(window, args.preview_duration_s)
        preview_path = output_dir / (
            f"{prefix}_req{first_index}-{preview[-1].request_index}_"
            f"first{args.preview_duration_s:g}s.png"
        )
        plot_timeline(
            preview_path,
            preview,
            timings,
            target_rps,
            args.ttft_slo,
            args.slo_percentile,
            args.dpi,
            args.preview_duration_s,
            height_factor=args.height_factor,
            width_factor=args.width_factor,
        )
        print(f"[plot] first {args.preview_duration_s:g}s: {preview_path}")


if __name__ == "__main__":
    main()
