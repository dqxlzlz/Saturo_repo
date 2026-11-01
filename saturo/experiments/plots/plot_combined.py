#!/usr/bin/env python3
"""Plot the three RT/BE disaggregation limitation panels in one row."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "experiments" / "saturo" / "results"
ARRIVAL_CSV = REPO / "data" / "datasets" / "AzureLLMInferenceTrace_conv.csv"
QUEUE_CSV = (
    REPO
    / "experiments"
    / "saturo"
    / "limitation_rt_be_disaggregatation"
    / "queue_timeline_rps4_215"
    / "requests_rps4.215.csv"
)
KV_CSV = RESULTS / "kv" / "rps_1.7_window_300_500" / "kv_block_usage_300_500_scaled.csv"
OUTPUT_DIR = RESULTS / "combined"


def load_arrival_rate() -> tuple[np.ndarray, np.ndarray]:
    """Return the same fluctuating 12-minute window as the standalone plot."""
    trace = pd.read_csv(ARRIVAL_CSV)
    trace["TIMESTAMP"] = pd.to_datetime(trace["TIMESTAMP"])
    trace = trace.sort_values("TIMESTAMP").reset_index(drop=True)
    elapsed_s = (
        trace["TIMESTAMP"] - trace["TIMESTAMP"].min()
    ).dt.total_seconds().to_numpy()

    big_slot_s = 30 * 60
    window_s = 12 * 60
    interval_s = 10
    edge_points = 6
    total_seconds = int(elapsed_s.max()) + 1
    per_second, _ = np.histogram(
        elapsed_s, bins=np.arange(0, total_seconds + 1, 1)
    )
    points_per_window = window_s // interval_s

    best_centered: tuple[float, int] | None = None
    best_any: tuple[float, int] | None = None
    for big_start in range(0, total_seconds, big_slot_s):
        big_end = min(big_start + big_slot_s, total_seconds)
        if big_end - big_start < window_s:
            continue
        for start in range(big_start, big_end - window_s + 1, 60):
            counts = per_second[start : start + window_s]
            rates = counts.reshape(points_per_window, interval_s).sum(axis=1)
            rates = rates.astype(float) / interval_s
            score = float(rates.std())
            max_index = int(np.argmax(rates))
            min_index = int(np.argmin(rates))
            edge_distance = min(
                max_index,
                min_index,
                points_per_window - 1 - max_index,
                points_per_window - 1 - min_index,
            )
            if best_any is None or score > best_any[0]:
                best_any = (score, start)
            if edge_distance >= edge_points and (
                best_centered is None or score > best_centered[0]
            ):
                best_centered = (score, start)

    selected = best_centered or best_any
    if selected is None:
        raise ValueError("Azure trace does not contain a complete 12-minute window")
    start = selected[1]
    counts = per_second[start : start + window_s]
    rates = counts.reshape(points_per_window, interval_s).sum(axis=1)
    rates = rates.astype(float) / interval_s
    minutes = (
        np.arange(points_per_window) * interval_s + interval_s / 2
    ) / 60.0
    return minutes, rates


def load_queue_window() -> list[dict[str, float]]:
    """Load requests overlapping the 73.043--79.043 second window."""
    start_s = 73.043
    end_s = 79.043
    selected: list[dict[str, float]] = []
    with QUEUE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arrival_s = float(row["arrival_s"])
            finish_s = float(row["finish_s"])
            if finish_s > start_s and arrival_s < end_s:
                selected.append(
                    {
                        "arrival_s": arrival_s - start_s,
                        "start_s": float(row["start_s"]) - start_s,
                        "queueing_s": float(row["queueing_s"]),
                        "execution_s": float(row["execution_s"]),
                    }
                )
    if not selected:
        raise ValueError("queue CSV has no requests in the selected window")
    return selected


def load_kv_usage() -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(KV_CSV)
    usage_column = (
        "kv_usage_pct_adjusted"
        if "kv_usage_pct_adjusted" in data.columns
        else "kv_usage_pct"
    )
    return (
        data["relative_time_s"].to_numpy(dtype=float),
        data[usage_column].to_numpy(dtype=float),
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=18)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 19,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "legend.fontsize": 16,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(16.5, 4.6))
    # Panel (a) carries the least information, so give it a narrower column;
    # (b) and (c) share the same width. Each data panel is forced to a 6:3.5
    # box aspect below via set_box_aspect, so these ratios set relative size.
    grid = figure.add_gridspec(
        1,
        5,
        width_ratios=[24, 8, 36, 10, 36],
        wspace=0,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[0, 4]),
    ]
    box_aspect = 3.5 / 6.0  # height:width = 3.5:6 for every panel

    # (a) Request arrival-rate fluctuation.
    minutes, rates = load_arrival_rate()
    arrival_axis = axes[0]
    arrival_axis.plot(
        minutes,
        rates,
        color="#2E5AAC",
        linewidth=1.4,
        marker="o",
        markersize=2.5,
    )
    max_index = int(np.argmax(rates))
    min_index = int(np.argmin(rates))
    arrival_axis.scatter(
        minutes[max_index], rates[max_index], color="#8B0000", s=25, zorder=5
    )
    arrival_axis.scatter(
        minutes[min_index], rates[min_index], color="#00008B", s=25, zorder=5
    )
    arrival_axis.annotate(
        f"{rates[max_index]:.1f}",
        (minutes[max_index], rates[max_index]),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        color="#8B0000",
        fontsize=18,
        fontweight="bold",
    )
    arrival_axis.annotate(
        f"{rates[min_index]:.1f}",
        (minutes[min_index], rates[min_index]),
        xytext=(7, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#00008B",
        fontsize=18,
        fontweight="bold",
    )
    arrival_axis.set_xlim(0, 12)
    arrival_axis.set_ylim(max(0, rates.min() * 0.85), rates.max() * 1.15)
    arrival_axis.set_xlabel("Time (min)")
    arrival_axis.set_ylabel("Arrival rate (req/s)")
    arrival_axis.set_title("(a) Arrival variation", fontweight="bold")
    style_axis(arrival_axis)
    arrival_axis.set_box_aspect(box_aspect)

    # (b) Per-request queueing and execution timeline.
    queue_axis = axes[1]
    queue_color = "#A9C8E2"
    execution_color = "#3C86BF"
    queue_window = load_queue_window()
    for row_index, timing in enumerate(queue_window):
        queue_axis.broken_barh(
            [(timing["arrival_s"], timing["queueing_s"])],
            (row_index - 0.38, 0.76),
            facecolors=queue_color,
        )
        queue_axis.broken_barh(
            [(timing["start_s"], timing["execution_s"])],
            (row_index - 0.38, 0.76),
            facecolors=execution_color,
        )
    queue_axis.set_xlim(0, 6)
    queue_axis.set_ylim(-0.8, len(queue_window) - 0.2)
    queue_axis.set_yticks(range(0, len(queue_window), 5))
    queue_axis.set_xlabel("Time (s)")
    queue_axis.set_ylabel("Request Idx")
    queue_axis.set_title("(b) Queueing and execution", fontweight="bold")
    queue_axis.legend(
        handles=[
            Patch(color=queue_color, label="Queueing"),
            Patch(color=execution_color, label="Execution"),
        ],
        loc="upper left",
        frameon=True,
        borderpad=0.4,
    )
    style_axis(queue_axis)
    queue_axis.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    queue_axis.set_box_aspect(box_aspect)

    # (c) RT-only decode-pool KV-block usage.
    kv_axis = axes[2]
    kv_time, kv_usage = load_kv_usage()
    kv_axis.plot(kv_time, kv_usage, color="#12828c", linewidth=1.1)
    kv_axis.fill_between(kv_time, kv_usage, 0, color="#12828c", alpha=0.07)
    kv_axis.set_xlim(300, 500)
    kv_axis.set_ylim(0, 100)
    kv_axis.set_xlabel("Time (s)")
    kv_axis.set_ylabel("KV block usage (%)")
    kv_axis.set_title("(c) KV block usage", fontweight="bold")
    style_axis(kv_axis)
    kv_axis.set_box_aspect(box_aspect)

    figure.subplots_adjust(
        left=0.057,
        right=0.985,
        bottom=0.21,
        top=0.85,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "limitation_combined_1x3.png"
    pdf_path = OUTPUT_DIR / "limitation_combined_1x3.pdf"
    svg_path = OUTPUT_DIR / "limitation_combined_1x3.svg"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
