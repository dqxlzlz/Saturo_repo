#!/usr/bin/env python3
"""Calibrate decode iteration wall time from real Saturo event logs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--batch-bin", type=int, default=32)
    parser.add_argument("--decode-hinge", type=float, default=512.0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_events(path: Path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("event") != "iteration_log":
                continue
            try:
                sample = {
                    "source": str(path),
                    "batch": int(row["n_decode_reqs"]),
                    "decode_sum_ctx": float(row["decode_sum_ctx"]),
                    "base_predicted_ms": float(row["predicted_time_ms"]),
                    "forward_ms": float(row["actual_time_ms"]),
                    "step_wall_ms": float(row["step_wall_time_ms"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if sample["batch"] > 0:
                rows.append(sample)
    return rows


def bucket_medians(rows, width):
    points = []
    for start in range(0, max(row["batch"] for row in rows) + 1, width):
        bucket = [row for row in rows if start <= row["batch"] < start + width]
        if len(bucket) < 3:
            continue
        points.append({
            key: float(np.median([row[key] for row in bucket]))
            for key in (
                "batch",
                "decode_sum_ctx",
                "base_predicted_ms",
                "forward_ms",
                "step_wall_ms",
            )
        })
    return points


def fit_nonnegative_overhead(points, quantile, decode_hinge):
    # Keep the existing model intact and fit the correction against total
    # query tokens.  These decode-only calibration traces have one query token
    # per request, so ``batch`` is also ``total_query_tokens``.
    x = np.array([
        [
            point["batch"],
            max(point["batch"] - decode_hinge, 0.0),
        ]
        for point in points
    ], dtype=float)
    y = np.array([
        point["step_wall_ms"] - point["base_predicted_ms"]
        for point in points
    ])
    count = len(y)
    objective = np.r_[
        np.zeros(2),
        np.full(count, quantile),
        np.full(count, 1.0 - quantile),
    ]
    equality = np.c_[x, np.eye(count), -np.eye(count)]
    scales = np.array([500.0, 500.0])
    equality[:, :2] *= scales
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=[(0.0, None)] * (2 + 2 * count),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:2] * scales


def metrics(rows, per_request_ms, hinge_ms, decode_hinge):
    errors = np.array([
        row["base_predicted_ms"]
        + per_request_ms * row["batch"]
        + hinge_ms * max(row["batch"] - decode_hinge, 0.0)
        - row["step_wall_ms"]
        for row in rows
    ])
    return {
        "samples": len(rows),
        "coverage": float(np.mean(errors >= 0.0)),
        "median_error_ms": float(np.median(errors)),
        "median_absolute_error_ms": float(np.median(np.abs(errors))),
        "p90_absolute_error_ms": float(np.percentile(np.abs(errors), 90)),
    }


def main():
    args = parse_args()
    if not 0.0 < args.quantile < 1.0:
        raise SystemExit("--quantile must be between 0 and 1")

    source_rows = {
        str(Path(path)): load_events(Path(path))
        for path in args.events
    }
    points = [
        point
        for rows in source_rows.values()
        for point in bucket_medians(rows, args.batch_bin)
    ]
    per_request_ms, hinge_ms = fit_nonnegative_overhead(
        points, args.quantile, args.decode_hinge)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_profile = Path(args.profile)
    payload = json.loads(source_profile.read_text())
    predictor = payload.get("time_predictor", payload)
    predictor["runtime_overhead"] = {
        "constant_ms": 0.0,
        "per_query_token_ms": float(per_request_ms),
        "query_token_hinge_threshold": args.decode_hinge,
        "per_query_token_above_hinge_ms": float(hinge_ms),
        "per_prefill_request_ms": 0.0,
        "per_decode_request_ms": 0.0,
        "per_active_request_ms": 0.0,
        "per_prefill_token_ms": 0.0,
        "per_decode_token_ms": 0.0,
        "decode_request_hinge_threshold": 0.0,
        "per_decode_request_above_hinge_ms": 0.0,
    }
    predictor.setdefault("meta", {})["end_to_end_decode_calibration"] = {
        "target": "engine_step_wall_time_ms",
        "quantile": args.quantile,
        "batch_bin": args.batch_bin,
        "bucket_count": len(points),
        "source_events": list(source_rows),
    }

    label = f"q{int(round(args.quantile * 100)):02d}"
    profile_path = output_dir / f"{source_profile.stem}.decode_{label}.json"
    profile_path.write_text(json.dumps(payload, indent=2) + "\n")

    report = {
        "profile": str(profile_path),
        "quantile": args.quantile,
        "batch_bin": args.batch_bin,
        "bucket_count": len(points),
        "runtime_overhead": predictor["runtime_overhead"],
        "per_source": {
            source: metrics(
                rows,
                per_request_ms,
                hinge_ms,
                args.decode_hinge,
            )
            for source, rows in source_rows.items()
        },
    }
    report_path = output_dir / f"calibration_{label}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
