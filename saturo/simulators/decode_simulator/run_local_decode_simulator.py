#!/usr/bin/env python3
"""Run the D-pool simulator with local BE prefill -> decode lifecycles."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from saturo.simulators.decode_simulator.local_simulator import (
    LocalDecodeSimulator,
)
from saturo.simulators.decode_simulator.run_decode_simulator import (
    build_decode_config,
    deep_copy_requests,
    parse_args,
    parse_decode_strategies,
    print_comparison_table,
    print_result_summary,
    result_to_dict,
    setup_logging,
)
from saturo.simulators.decode_simulator.trace_loader import load_rt_trace


def _arg(args, name, default=None):
    return getattr(args, name, default)


def run_local_decode_profile(args, *, print_summaries: bool = True):
    """Run one duration-window RT point with local BE request semantics."""
    duration_s = float(args.duration_s)
    target_rps = _arg(args, "rt_rps")
    time_scale = _arg(args, "time_scale")
    if _arg(args, "rps_sweep"):
        raise ValueError("local decode simulator accepts one RPS per run")
    config, chunk_sizes = build_decode_config(args)
    config.be_prefill_reserve_blocks = 0
    config.balance_be_throughput = False
    strategies = parse_decode_strategies(
        _arg(args, "strategies", "progressive,oracle,greedy"))

    rt_requests = load_rt_trace(
        args.rt_trace,
        duration_s=duration_s,
        target_rps=target_rps,
        time_scale=time_scale,
        max_prompt_len=_arg(args, "rt_max_prompt_len"),
        max_output_len=_arg(args, "rt_max_output_len"),
    )

    results = {}
    label = f"RPS={target_rps}" if target_rps is not None else "RPS=original"
    for strategy in strategies:
        sim = LocalDecodeSimulator(config, strategy=strategy)
        result = sim.run(deep_copy_requests(rt_requests), [])
        if print_summaries:
            print_result_summary(result)
        results[strategy.value] = result
    if print_summaries:
        print_comparison_table(results, label=label)

    runtime_profile = _arg(args, "runtime_profile")
    return {
        "config": {
            "request_mode": "local_prefill_then_decode",
            "runtime_profile": runtime_profile,
            "total_gpu_blocks": config.total_gpu_blocks,
            "block_size": config.model.block_size,
            "rt_tbt_slo_ms": config.slo.tbt_slo_ms,
            "duration_s": duration_s,
            "warmup_s": config.warmup_s,
            "rt_rps": target_rps,
            "time_scale": time_scale,
            "arrival_mode": "scaled_dataset_trace",
            "memory_watermark": config.memory_watermark,
            "be_workload_csv": config.be_workload_csv,
            "be_prefill_reserve_blocks": 0,
            "balance_be_throughput": False,
            "throughput_metric": "completed_local_be_requests_per_s",
            "chunk_sizes": chunk_sizes,
            "lookahead_window_s": config.saturation.lookahead_window_s,
            "min_interval_ms": config.saturation.min_interval_ms,
            "strategies": [strategy.value for strategy in strategies],
        },
        "results": {
            label: {
                name: result_to_dict(result)
                for name, result in results.items()
            }
        },
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    try:
        payload = run_local_decode_profile(args, print_summaries=True)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"local decode simulator: error: {exc}") from exc

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "local_decode_simulation_result.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
    logging.getLogger(__name__).info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
