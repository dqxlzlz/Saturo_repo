#!/usr/bin/env python3
"""Consolidate real D-pool per-RPS summaries into reference-style files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROW_FIELDS = (
    "rps",
    "window_s",
    "num_req",
    "rt_slo_met",
    "rt_tbt_slo_attainment",
    "rt_tbt_slo_violation_rate",
    "rt_tbt_slo_violations",
    "rt_tbt_sample_count",
    "rt_tbt_p50_ms",
    "rt_tbt_p95_ms",
    "rt_tbt_p99_ms",
    "rt_ttft_p50_ms",
    "rt_ttft_p95_ms",
    "rt_completed",
    "rt_submitted",
    "be_completed_req_per_s",
    "be_fractional_completed_requests",
    "be_fractional_req_per_s",
    "be_token_normalized_req_per_s",
    "be_work_throughput_tok_per_s",
    "be_output_throughput_tok_per_s",
    "be_prefill_throughput_tok_per_s",
    "be_decode_throughput_tok_per_s",
    "scheduler_wall_ms_avg",
    "scheduler_wall_ms_p50",
    "scheduler_wall_ms_p95",
    "scheduler_wall_ms_p99",
    "scheduler_wall_fraction_of_window",
    "avg_actual_iteration_ms",
    "avg_predicted_iteration_ms",
    "avg_target_tbt_ms",
    "verdict",
)


def _row(summary: dict[str, Any]) -> dict[str, Any]:
    tbt_p99 = float(summary.get("rt_tbt_ms_p99", 0.0) or 0.0)
    slo_ms = float(summary.get("rt_tbt_slo_ms", 150.0) or 150.0)
    violation_rate = float(
        summary.get("rt_tbt_slo_violation_rate", 0.0) or 0.0)
    work_tput = float(
        summary.get("be_work_throughput_tokens_per_s", 0.0) or 0.0)
    rt_slo_met = tbt_p99 <= slo_ms
    if not rt_slo_met:
        verdict = "rt_slo_violation"
    elif work_tput <= 1e-9:
        verdict = "critical_be_starved"
    else:
        verdict = "serviceable"
    return {
        "rps": float(summary["target_rps"]),
        "window_s": float(summary["window_s"]),
        "num_req": int(summary["num_req"]),
        "rt_slo_met": rt_slo_met,
        "rt_tbt_slo_attainment": float(
            summary.get("rt_tbt_slo_attainment",
                        1.0 - violation_rate) or 0.0),
        "rt_tbt_slo_violation_rate": violation_rate,
        "rt_tbt_slo_violations": int(
            summary.get("rt_tbt_slo_violations", 0) or 0),
        "rt_tbt_sample_count": int(
            summary.get("rt_tbt_sample_count", 0) or 0),
        "rt_tbt_p50_ms": float(summary.get("rt_tbt_ms_p50", 0.0) or 0.0),
        "rt_tbt_p95_ms": float(summary.get("rt_tbt_ms_p95", 0.0) or 0.0),
        "rt_tbt_p99_ms": tbt_p99,
        "rt_ttft_p50_ms": float(summary.get("rt_ttft_ms_p50", 0.0) or 0.0),
        "rt_ttft_p95_ms": float(summary.get("rt_ttft_ms_p95", 0.0) or 0.0),
        "rt_completed": int(summary.get("rt_completed", 0) or 0),
        "rt_submitted": int(summary.get("rt_submitted", 0) or 0),
        "be_completed_req_per_s": float(
            summary.get("be_saturation_req_per_s", 0.0) or 0.0),
        "be_fractional_completed_requests": float(
            summary.get("be_decode_fractional_completed_requests", 0.0)
            or 0.0),
        "be_fractional_req_per_s": float(
            summary.get("be_decode_throughput_req_per_s", 0.0) or 0.0),
        "be_token_normalized_req_per_s": float(
            summary.get("be_decode_token_normalized_req_per_s", 0.0)
            or 0.0),
        "be_work_throughput_tok_per_s": work_tput,
        "be_output_throughput_tok_per_s": float(
            summary.get("be_throughput_tokens_per_s", 0.0) or 0.0),
        "be_prefill_throughput_tok_per_s": float(
            summary.get("be_prefill_throughput_tokens_per_s", 0.0) or 0.0),
        "be_decode_throughput_tok_per_s": float(
            summary.get("be_decode_throughput_tokens_per_s", 0.0) or 0.0),
        "scheduler_wall_ms_avg": float(
            summary.get("avg_scheduler_wall_ms", 0.0) or 0.0),
        "scheduler_wall_ms_p50": float(
            summary.get("scheduler_wall_ms_p50", 0.0) or 0.0),
        "scheduler_wall_ms_p95": float(
            summary.get("scheduler_wall_ms_p95", 0.0) or 0.0),
        "scheduler_wall_ms_p99": float(
            summary.get("scheduler_wall_ms_p99", 0.0) or 0.0),
        "scheduler_wall_fraction_of_window": float(
            summary.get("scheduler_wall_fraction_of_window", 0.0) or 0.0),
        "avg_actual_iteration_ms": float(
            summary.get("avg_actual_iteration_ms", 0.0) or 0.0),
        "avg_predicted_iteration_ms": float(
            summary.get("avg_predicted_iteration_ms", 0.0) or 0.0),
        "avg_target_tbt_ms": float(
            summary.get("avg_target_tbt_ms", 0.0) or 0.0),
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    summaries = sorted(
        in_dir.glob("rps_*/summary.json"),
        key=lambda path: float(path.parent.name.removeprefix("rps_")),
    )
    if not summaries:
        raise ValueError(f"no rps_*/summary.json files under {in_dir}")

    payloads = [json.loads(path.read_text(encoding="utf-8"))
                for path in summaries]
    rows = [_row(summary) for summary in payloads]
    first = payloads[0]
    result = {
        "label": args.label,
        "workload": {
            "model": first.get("model"),
            "load_format": first.get("load_format"),
            "rt_trace": first.get("rt_workload_csv"),
            "be_trace": first.get("be_workload_csv"),
        },
        "operating_config": {
            "request_mode": first.get("request_mode"),
            "strategy": first.get("saturation_strategy"),
            "lookahead_window_s": first.get("oracle_lookahead_s"),
            "rt_tbt_slo_ms": first.get("rt_tbt_slo_ms", 150.0),
            "be_inflight": first.get("be_inflight"),
            "max_num_seqs": first.get("max_num_seqs"),
            "max_num_batched_tokens": first.get("max_num_batched_tokens"),
            "throughput_metric": (
                "sum(min(per_request_decode_tokens/target_output_tokens, "
                "1))/window_s"),
        },
        "rows": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rps_sweep_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "rps_sweep_summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
