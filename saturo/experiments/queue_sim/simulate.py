#!/usr/bin/env python3
"""CPU-only FCFS replay for a measured single-request TTFT trace.

Each successful CSV row is treated as one request.  ``arrival_offset_s`` is
scaled to the requested RPS with the same formula as
``saturo.tools.trace_scaler``.  The measured single-request ``ttft_s`` is the
request's non-preemptive service time on one server.  Consequently::

    start     = max(arrival, previous_finish)
    queue     = start - arrival
    finish    = start + measured_ttft
    total_ttft = queue + measured_ttft

The script evaluates the RPS values passed through ``--target-rps`` and writes
one selected run's per-request timings and a compact timeline around the
request that is just over the SLO.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from saturo.tools.trace_scaler import (  # noqa: E402
    compute_trace_rps,
    scale_to_target_rps_fixed_num,
)


DEFAULT_TRACE = (
    PROJECT_ROOT
    / "experiments/saturo/results/ttft"
    / "nvidia-gb200_qwen2-5-72b_tp1_auto_maxlen20000.csv"
)


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    arrival_offset_s: float
    execution_s: float
    prompt_len: int
    output_len: int


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


@dataclass(frozen=True)
class SweepPoint:
    target_rps: float
    request_count: int
    p50_ttft_s: float
    p95_ttft_s: float
    p99_ttft_s: float
    max_ttft_s: float
    p95_queueing_s: float
    slo_percentile_ttft_s: float
    violation_count: int
    violation_rate: float
    slo_met: bool


def _float_csv(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",")
                  if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--target-rps must be a comma-separated list of numbers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "--target-rps must contain positive values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay isolated TTFT measurements as a CPU FCFS queue."
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--num-req",
        type=int,
        required=True,
        help="Use exactly this many chronological trace requests.",
    )
    parser.add_argument(
        "--ttft-slo",
        type=float,
        default=1.0,
        help="Percentile and per-request TTFT SLO in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--slo-percentile",
        type=float,
        default=0.99,
        help="Required TTFT percentile in (0, 1] (default: 0.99).",
    )
    parser.add_argument(
        "--target-rps",
        type=_float_csv,
        required=True,
        help="Target RPS; accepts comma-separated values for a sweep.",
    )
    parser.add_argument(
        "--timeline-requests",
        type=int,
        default=500,
        help="Number of neighboring requests in the timeline (default: 500).",
    )
    parser.add_argument(
        "--timeline-start-index",
        type=int,
        default=None,
        help="First request index to plot; default centers on a tight violation.",
    )
    parser.add_argument(
        "--timeline-duration-s",
        type=float,
        default=None,
        help="Limit the main timeline to this many seconds (default: unlimited).",
    )
    parser.add_argument(
        "--timeline-preview-duration-s",
        type=float,
        default=10.0,
        help="Also write a leading-duration preview PNG; 0 disables it.",
    )
    parser.add_argument(
        "--timeline-dpi",
        type=int,
        default=120,
        help="Timeline PNG resolution (default: 120).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/saturo/results/queue",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _parse_float(row: dict[str, str], names: Sequence[str]) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return None


def load_trace(path: Path) -> list[TraceRequest]:
    """Load successful requests and keep arrival/service pairs associated."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    requests: list[TraceRequest] = []
    for row_index, row in enumerate(rows):
        if row.get("status", "ok").lower() != "ok":
            continue
        arrival = _parse_float(
            row, ("arrival_offset_s", "arrival_time_s", "arrival_time")
        )
        execution = _parse_float(row, ("ttft_s", "runtime_s", "execution_time_s"))
        if execution is None:
            execution_ms = _parse_float(
                row, ("ttft_ms", "runtime_ms", "execution_time_ms")
            )
            execution = None if execution_ms is None else execution_ms / 1000.0
        if arrival is None or execution is None:
            continue
        if execution < 0:
            raise ValueError(f"negative execution time at CSV row {row_index + 2}")
        requests.append(
            TraceRequest(
                request_id=row.get("request_id") or f"request_{row_index}",
                arrival_offset_s=arrival,
                execution_s=execution,
                prompt_len=int(float(row.get("prompt_len") or 0)),
                output_len=int(float(row.get("output_len") or 0)),
            )
        )

    if not requests:
        raise ValueError(f"no usable successful requests in {path}")
    return sorted(requests, key=lambda request: request.arrival_offset_s)


def replay_fcfs(
    requests: Sequence[TraceRequest], target_rps: float, ttft_slo_s: float
) -> list[RequestTiming]:
    """Replay a non-preemptive, single-server FCFS queue."""
    if target_rps <= 0:
        raise ValueError("target_rps must be positive")
    raw_arrivals = [request.arrival_offset_s for request in requests]
    scaled_arrivals = scale_to_target_rps_fixed_num(
        raw_arrivals,
        num_req=len(requests),
        target_rps=target_rps,
    )

    available_s = 0.0
    timings: list[RequestTiming] = []
    for index, request in enumerate(requests):
        arrival_s = scaled_arrivals[index]
        start_s = max(arrival_s, available_s)
        finish_s = start_s + request.execution_s
        queueing_s = start_s - arrival_s
        ttft_s = queueing_s + request.execution_s
        timings.append(
            RequestTiming(
                request_index=index,
                request_id=request.request_id,
                arrival_s=arrival_s,
                start_s=start_s,
                finish_s=finish_s,
                queueing_s=queueing_s,
                execution_s=request.execution_s,
                ttft_s=ttft_s,
                slo_violated=ttft_s > ttft_slo_s,
                prompt_len=request.prompt_len,
                output_len=request.output_len,
            )
        )
        available_s = finish_s
    return timings


def percentile(values: Iterable[float], quantile: float) -> float:
    """Nearest-rank percentile, matching the repository's simulators."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def summarize(
    timings: Sequence[RequestTiming],
    target_rps: float,
    ttft_slo_s: float,
    slo_percentile: float,
) -> SweepPoint:
    ttfts = [timing.ttft_s for timing in timings]
    queues = [timing.queueing_s for timing in timings]
    violations = sum(timing.slo_violated for timing in timings)
    slo_percentile_ttft_s = percentile(ttfts, slo_percentile)
    return SweepPoint(
        target_rps=target_rps,
        request_count=len(timings),
        p50_ttft_s=percentile(ttfts, 0.50),
        p95_ttft_s=percentile(ttfts, 0.95),
        p99_ttft_s=percentile(ttfts, 0.99),
        max_ttft_s=max(ttfts, default=0.0),
        p95_queueing_s=percentile(queues, 0.95),
        slo_percentile_ttft_s=slo_percentile_ttft_s,
        violation_count=violations,
        violation_rate=violations / len(timings) if timings else 0.0,
        slo_met=slo_percentile_ttft_s <= ttft_slo_s,
    )


def run_sweep(
    requests: Sequence[TraceRequest],
    rps_points: Iterable[float],
    ttft_slo_s: float,
    slo_percentile: float,
) -> list[SweepPoint]:
    return [
        summarize(
            replay_fcfs(requests, rps, ttft_slo_s),
            rps,
            ttft_slo_s,
            slo_percentile,
        )
        for rps in rps_points
    ]


def select_timeline_window(
    timings: Sequence[RequestTiming],
    count: int,
    ttft_slo_s: float,
    start_index: int | None = None,
    duration_s: float | None = None,
) -> list[RequestTiming]:
    """Center the plot on the request nearest to, but above, the SLO."""
    if not timings:
        return []
    count = max(1, min(count, len(timings)))
    if start_index is not None:
        start = max(0, min(start_index, len(timings) - count))
    else:
        above = [timing for timing in timings if timing.ttft_s > ttft_slo_s]
        candidates = above or list(timings)
        center = min(
            candidates, key=lambda timing: abs(timing.ttft_s - ttft_slo_s)
        )
        start = max(0, center.request_index - count // 2)
        start = min(start, len(timings) - count)

    window = list(timings[start : start + count])
    if duration_s is not None:
        end_s = window[0].arrival_s + duration_s
        window = [timing for timing in window if timing.arrival_s < end_s]
    return window


def write_request_csv(path: Path, timings: Sequence[RequestTiming]) -> None:
    fieldnames = list(RequestTiming.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(timing) for timing in timings)


def write_sweep_csv(path: Path, sweep: Sequence[SweepPoint]) -> None:
    fieldnames = list(SweepPoint.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(point) for point in sweep)


def _merge_busy_intervals(
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
    timings: Sequence[RequestTiming],
    all_timings: Sequence[RequestTiming],
    target_rps: float,
    ttft_slo_s: float,
    slo_percentile: float,
    dpi: int,
    duration_s: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not timings:
        return
    queue_color = "#A9C8E2"
    execution_color = "#3C86BF"
    violation_color = "#C83E4D"
    origin_s = timings[0].arrival_s
    natural_end_s = max(timing.finish_s for timing in timings)
    display_end_s = (
        origin_s + duration_s if duration_s is not None else natural_end_s
    )

    figure_height = max(6.5, 2.0 + 0.18 * len(timings))
    main_height = max(6, len(timings))
    figure, (util_axis, request_axis) = plt.subplots(
        2,
        1,
        figsize=(12, figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": [6, main_height], "hspace": 0.01},
    )

    util_axis.broken_barh(
        _merge_busy_intervals(all_timings, origin_s, display_end_s),
        (0, 1),
        facecolors=execution_color,
    )
    util_axis.set_ylim(0, 1)
    util_axis.set_yticks([])
    util_axis.set_ylabel("GPU Util.", fontsize=13)
    util_axis.tick_params(axis="x", labelbottom=False)

    labels: list[str] = []
    for row, timing in enumerate(timings):
        arrival = timing.arrival_s - origin_s
        start = timing.start_s - origin_s
        queue_width = timing.queueing_s
        request_axis.broken_barh(
            [(arrival, queue_width)],
            (row - 0.38, 0.76),
            facecolors=queue_color,
        )
        request_axis.broken_barh(
            [(start, timing.execution_s)],
            (row - 0.38, 0.76),
            facecolors=execution_color,
            edgecolors=violation_color if timing.slo_violated else "none",
            linewidth=1.5 if timing.slo_violated else 0.0,
        )
        labels.append(str(timing.request_index))

    request_axis.set_yticks(range(len(timings)), labels, fontsize=6)
    request_axis.set_ylim(-0.8, len(timings) - 0.2)
    request_axis.set_xlabel("Time since first displayed arrival (s)", fontsize=13)
    request_axis.set_ylabel("Request index", fontsize=13)
    request_axis.grid(True, axis="x", linestyle="--", alpha=0.35)
    if duration_s is not None:
        request_axis.set_xlim(0.0, duration_s)
    request_axis.legend(
        handles=[
            Patch(color=queue_color, label="Queueing Time (s)"),
            Patch(color=execution_color, label="Execution Time (s)"),
            Patch(facecolor="white", edgecolor=violation_color,
                  label=f"TTFT > {ttft_slo_s:g}s"),
        ],
        loc="upper left",
        fontsize=11,
    )
    figure.suptitle(
        f"CPU FCFS replay at {target_rps:.3f} RPS "
        f"(P{100 * slo_percentile:g} TTFT SLO = {ttft_slo_s:g}s)",
        fontsize=14,
    )
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.01, top=0.995)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_sweep(
    path: Path,
    sweep: Sequence[SweepPoint],
    selected: SweepPoint,
    ttft_slo_s: float,
    slo_percentile: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(
        [point.target_rps for point in sweep],
        [point.slo_percentile_ttft_s for point in sweep],
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color="#3C86BF",
        label=f"P{100 * slo_percentile:g} TTFT",
    )
    axis.axhline(
        ttft_slo_s,
        color="#C83E4D",
        linestyle="--",
        linewidth=1.5,
        label=f"SLO = {ttft_slo_s:g}s",
    )
    axis.scatter(
        [selected.target_rps],
        [selected.slo_percentile_ttft_s],
        color="#C83E4D",
        zorder=5,
        label=f"Selected = {selected.target_rps:.3f} RPS",
    )
    axis.set_xlabel("Target RPS")
    axis.set_ylabel(f"P{100 * slo_percentile:g} TTFT (s)")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.ttft_slo <= 0:
        raise ValueError("ttft_slo must be positive")
    if not 0.0 < args.slo_percentile <= 1.0:
        raise ValueError("slo_percentile must be in (0, 1]")
    if args.timeline_dpi <= 0:
        raise ValueError("timeline_dpi must be positive")
    if args.timeline_duration_s is not None and args.timeline_duration_s <= 0:
        raise ValueError("timeline_duration_s must be positive")
    if args.timeline_preview_duration_s < 0:
        raise ValueError("timeline_preview_duration_s must be non-negative")

    percentile_label = f"P{100 * args.slo_percentile:g}"
    percentile_slug = percentile_label.lower().replace(".", "_")

    requests = load_trace(args.trace)
    if args.num_req <= 0:
        raise ValueError("num_req must be positive")
    if len(requests) < args.num_req:
        raise ValueError(
            f"trace has {len(requests)} requests, but num_req={args.num_req}")
    requests = requests[:args.num_req]
    raw_arrivals = [request.arrival_offset_s for request in requests]
    trace_rps = compute_trace_rps(raw_arrivals)
    rps_points = sorted(set(args.target_rps))
    sweep = run_sweep(
        requests, rps_points, args.ttft_slo, args.slo_percentile
    )
    slo_met_points = [point for point in sweep if point.slo_met]
    selected = (
        max(slo_met_points, key=lambda point: point.target_rps)
        if slo_met_points else sweep[0]
    )
    first_violated = next(
        (point for point in sweep if not point.slo_met),
        None,
    )

    timings = replay_fcfs(requests, selected.target_rps, args.ttft_slo)
    window = select_timeline_window(
        timings,
        args.timeline_requests,
        args.ttft_slo,
        args.timeline_start_index,
        args.timeline_duration_s,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = args.output_dir / "rps_sweep.csv"
    request_path = args.output_dir / f"requests_rps{selected.target_rps:.3f}.csv"
    summary_path = args.output_dir / "summary.json"
    write_sweep_csv(sweep_path, sweep)
    write_request_csv(request_path, timings)

    previous_met = max(
        (point for point in sweep if point.slo_met),
        key=lambda point: point.target_rps,
        default=None,
    )
    payload = {
        "model": "single-server non-preemptive FCFS",
        "trace": str(args.trace.resolve()),
        "request_count": len(requests),
        "original_trace_rps": trace_rps,
        "ttft_slo_s": args.ttft_slo,
        "slo_percentile": args.slo_percentile,
        "percentile_method": "nearest-rank",
        "selected": asdict(selected),
        "target_rps": rps_points,
        "first_slo_violated": (
            asdict(first_violated) if first_violated else None
        ),
        "last_slo_met": asdict(previous_met) if previous_met else None,
        "timeline_request_indices": [timing.request_index for timing in window],
        "outputs": {
            "rps_sweep_csv": str(sweep_path.resolve()),
            "per_request_csv": str(request_path.resolve()),
        },
    }

    if not args.no_plot:
        sweep_plot_path = args.output_dir / f"{percentile_slug}_ttft_vs_rps.png"
        timeline_path = (
            args.output_dir
            / f"queue_timeline_{percentile_slug}_rps{selected.target_rps:.3f}_"
            f"{len(window)}req"
            f"{'_' + format(args.timeline_duration_s, 'g') + 's' if args.timeline_duration_s is not None else ''}.png"
        )
        plot_sweep(
            sweep_plot_path,
            sweep,
            selected,
            args.ttft_slo,
            args.slo_percentile,
        )
        plot_timeline(
            timeline_path,
            window,
            timings,
            selected.target_rps,
            args.ttft_slo,
            args.slo_percentile,
            args.timeline_dpi,
            args.timeline_duration_s,
        )
        payload["outputs"].update(
            {
                "rps_sweep_plot": str(sweep_plot_path.resolve()),
                "timeline_plot": str(timeline_path.resolve()),
            }
        )

        if (
            args.timeline_duration_s is None
            and args.timeline_preview_duration_s > 0
        ):
            preview_window = select_timeline_window(
                timings,
                args.timeline_requests,
                args.ttft_slo,
                args.timeline_start_index,
                args.timeline_preview_duration_s,
            )
            preview_path = (
                args.output_dir
                / f"queue_timeline_{percentile_slug}_"
                f"rps{selected.target_rps:.3f}_{len(preview_window)}req_"
                f"first{args.timeline_preview_duration_s:g}s.png"
            )
            plot_timeline(
                preview_path,
                preview_window,
                timings,
                selected.target_rps,
                args.ttft_slo,
                args.slo_percentile,
                args.timeline_dpi,
                args.timeline_preview_duration_s,
            )
            payload["outputs"]["timeline_preview_plot"] = str(
                preview_path.resolve()
            )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        f"Loaded {len(requests)} requests; original trace RPS={trace_rps:.6f}"
    )
    print(f"\nTarget RPS sweep ({percentile_label} TTFT):")
    for point in sweep:
        state = "MET" if point.slo_met else "VIOLATED"
        print(
            f"  {point.target_rps:6.3f} RPS  "
            f"{percentile_label}={point.slo_percentile_ttft_s:8.6f}s  "
            f"request violations={point.violation_rate:7.3%}  {state}"
        )
    print(
        f"\nSelected {selected.target_rps:.3f} RPS: "
        f"{percentile_label}={selected.slo_percentile_ttft_s:.6f}s, "
        f"violations={selected.violation_count}/{selected.request_count} "
        f"({selected.violation_rate:.3%})"
    )
    if first_violated is not None:
        print(
            f"First supplied SLO-violated point "
            f"{first_violated.target_rps:.3f} RPS: "
            f"{percentile_label}="
            f"{first_violated.slo_percentile_ttft_s:.6f}s"
        )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
