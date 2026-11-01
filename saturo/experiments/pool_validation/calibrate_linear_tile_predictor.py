#!/usr/bin/env python3
"""Calibrate a stable tile lookup for TimePredictor's linear component."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple


def _load_measurements(
    paths: Sequence[Path],
    metric: str,
    min_batch: int,
    max_batch: int,
) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                batch = int(row["batch_size"])
                if batch < min_batch or batch > max_batch:
                    continue
                value = float(row[metric])
                if math.isfinite(value):
                    points.append((batch, value))
    if not points:
        raise ValueError("no finite measurements in the requested range")
    return points


def _fit_tiles(
    points: Sequence[Tuple[int, float]],
    tile_width: int,
    max_batch: int,
    small_batch_cutoff: int,
    round_ms: float,
) -> Tuple[List[int], List[float], List[float], List[int]]:
    upper_bounds = []
    if 0 < small_batch_cutoff < tile_width:
        upper_bounds.append(small_batch_cutoff)
    upper_bounds.extend(range(tile_width, max_batch + 1, tile_width))
    by_tile: Dict[int, List[float]] = {
        index: [] for index in range(len(upper_bounds))
    }
    for batch, value in points:
        tile_id = next(
            index for index, upper in enumerate(upper_bounds)
            if batch <= upper)
        by_tile[tile_id].append(value)

    missing = [index for index, values in by_tile.items() if not values]
    if missing:
        raise ValueError(f"tiles without measurements: {missing}")
    raw = [median(by_tile[index]) for index in range(len(upper_bounds))]
    if round_ms > 0:
        fitted = [round(value / round_ms) * round_ms for value in raw]
    else:
        fitted = list(raw)
    weights = [len(by_tile[index]) for index in range(len(upper_bounds))]
    return upper_bounds, raw, fitted, weights


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def _write_table(
    path: Path,
    upper_bounds: Sequence[int],
    num_layers: int,
    raw: Sequence[float],
    fitted: Sequence[float],
    weights: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "tile_start_tokens",
            "tile_end_tokens",
            "num_profile_points",
            "raw_full_model_linear_ms",
            "adjusted_full_model_linear_ms",
            "adjusted_per_layer_linear_ms",
        ])
        lower_bound = 1
        for upper_bound, raw_value, fitted_value, weight in zip(
                upper_bounds, raw, fitted, weights):
            writer.writerow([
                lower_bound,
                upper_bound,
                weight,
                raw_value,
                fitted_value,
                fitted_value / num_layers,
            ])
            lower_bound = upper_bound + 1


def _write_plot(
    path: Path,
    points: Sequence[Tuple[int, float]],
    upper_bounds: Sequence[int],
    fitted: Sequence[float],
    legacy_linear: dict,
    num_layers: int,
) -> None:
    import matplotlib.pyplot as plt

    x_raw = [point[0] for point in points]
    y_raw = [point[1] for point in points]
    legacy_width = int(legacy_linear.get("width", 32))
    legacy_coef = legacy_linear.get("coef", {})
    legacy_fallback = legacy_linear.get("fallback", [0.0, 0.0])
    x_legacy = list(range(1, upper_bounds[-1] + 1))
    y_legacy = []
    for token_count in x_legacy:
        coefficient = legacy_coef.get(
            str(math.floor(token_count / legacy_width)), legacy_fallback)
        per_layer = max(
            0.0, float(coefficient[0]) * token_count + float(coefficient[1]))
        y_legacy.append(per_layer * num_layers)
    x_step = [1]
    y_step = [fitted[0]]
    for index, (right, value) in enumerate(zip(upper_bounds, fitted)):
        x_step.extend([right, right + 1])
        y_step.extend([value, fitted[min(index + 1, len(fitted) - 1)]])

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.scatter(x_raw, y_raw, s=13, alpha=0.42,
                 label="measured linear p50 (full model)")
    axis.plot(x_legacy, y_legacy, linewidth=1.2, alpha=0.75,
              linestyle="--", color="#d95f02",
              label="old 32-token piecewise-linear predictor")
    axis.plot(x_step[:-1], y_step[:-1], linewidth=2.4,
              label="tile-median linear predictor")
    axis.set_xlabel("Decode batch size / total linear tokens")
    axis.set_ylabel("Linear time per forward (ms, 88 layers)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--metric", default="actual_linear_ms_p50")
    parser.add_argument("--tile-width", type=int, default=128)
    parser.add_argument("--small-batch-cutoff", type=int, default=0)
    parser.add_argument("--round-ms", type=float, default=0.5)
    parser.add_argument("--min-batch", type=int, default=24)
    parser.add_argument("--max-batch", type=int, default=2048)
    args = parser.parse_args()

    if args.tile_width <= 0:
        raise ValueError("--tile-width must be positive")
    if args.max_batch % args.tile_width:
        raise ValueError("--max-batch must be divisible by --tile-width")

    with args.profile.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    predictor = payload.get("time_predictor", payload)
    num_layers = int(predictor["num_layers"])
    if num_layers <= 0:
        raise ValueError("profile num_layers must be positive")

    points = _load_measurements(
        args.summary, args.metric, args.min_batch, args.max_batch)
    upper_bounds, raw, fitted, weights = _fit_tiles(
        points,
        args.tile_width,
        args.max_batch,
        args.small_batch_cutoff,
        args.round_ms,
    )
    predictor["linear_tile_override"] = {
        "tile_width": args.tile_width,
        "max_tokens": args.max_batch,
        "token_upper_bounds": upper_bounds,
        "per_layer_values_ms": [value / num_layers for value in fitted],
        "full_model_values_ms": fitted,
        "fit": "per-tile median rounded to a stable step level",
        "metric": args.metric,
        "min_profiled_batch": args.min_batch,
        "small_batch_cutoff": args.small_batch_cutoff,
        "round_ms": args.round_ms,
        "source_summaries": [str(path) for path in args.summary],
    }

    output = args.output or args.profile
    _write_json(output, payload)
    if args.table:
        _write_table(
            args.table, upper_bounds, num_layers, raw, fitted, weights)
    if args.plot:
        _write_plot(
            args.plot,
            points,
            upper_bounds,
            fitted,
            predictor.get("linear", {}),
            num_layers,
        )

    print(f"Updated profile: {output}")
    lower_bound = 1
    for upper_bound, raw_value, fitted_value, weight in zip(
            upper_bounds, raw, fitted, weights):
        print(
            f"{lower_bound:4d}-{upper_bound:4d}: "
            f"raw={raw_value:8.3f} ms  adjusted={fitted_value:8.3f} ms  "
            f"n={weight}"
        )
        lower_bound = upper_bound + 1


if __name__ == "__main__":
    main()
