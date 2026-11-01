#!/usr/bin/env python3
"""Sweep RT RPS × BE chunk-size over the RT-BE colocation simulator.

Produces a CSV grid of (RPS, chunk_latency_ms) → RT violation rate &
BE throughput, plus machine-readable JSON with per-point detail.

Modes
-----

**Simple mode** (``--be-chunk-ms-list``): specify BE chunk latencies
directly.  Each value is the wall-clock duration of one BE chunk on
the GPU.  ``--be-chunk-tokens`` sets how many prefill tokens per chunk.

**Profile mode** (``--runtime-profile`` + ``--be-chunk-tokens-list``):
use a Saturo TimePredictor to convert token counts to latency estimates.

Examples
--------

Simple mode (recommended for quick sweeps)::

    python saturo/simulators/rt_be_colocate/run_simulator.py \\
        --rt-trace data/datasets/rt_workload.csv \\
        --num-req 600 \\
        --target-rps 1,2,4,6,8 \\
        --be-chunk-ms-list 10,25,50,100,200 \\
        --be-chunk-tokens 4096 \\
        --ttft-slo-ms 2000 \\
        --output results/colocate_sim.csv

Profile-based mode::

    python saturo/simulators/rt_be_colocate/run_simulator.py \\
        --rt-trace data/datasets/rt_workload.csv \\
        --runtime-profile saturo/results/time_predictors/tp1_pp1.json \\
        --num-req 600 \\
        --target-rps 1,2,4,6,8 \\
        --be-chunk-tokens-list 1024,2048,4096,8192 \\
        --ttft-slo-ms 2000 \\
        --output results/colocate_sim.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

# Ensure the vllm / saturo packages are importable.
_PROJECT = Path(__file__).resolve().parents[3]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from saturo.simulators.rt_be_colocate.config import ColocateConfig, ColocateResult
from saturo.simulators.rt_be_colocate.simulator import (
    ColocateSimulator,
    _RTRequest,
)


# ---------------------------------------------------------------------------
# Trace loading (reuses trace_loader from the existing prefill simulator)
# ---------------------------------------------------------------------------


def _load_rt_trace(
    csv_path: str,
    num_req: int,
    target_rps: float,
    default_runtime_s: float | None = None,
    runtime_profile: str = "",
) -> List[_RTRequest]:
    """Load an RT trace CSV and scale inter-arrival times to *target_rps*.

    The CSV must have an arrival-time column (``arrival_time``,
    ``arrival_time_s``, ``arrival_offset_s``, or ``TIMESTAMP``).

    Runtime is resolved in this order:
    1. From a runtime column in the CSV (``runtime_s``, ``forward_time``, …).
    2. From ``--default-runtime-s`` if provided (same runtime for all rows).
    3. Estimated from ``prompt_len`` using the TimePredictor profile.
    4. Estimated from ``prompt_len`` at a conservative 5000 tok/s.
    """
    from saturo.simulators.prefill_simulator.trace_loader import (
        load_rt_trace as _pf_load_rt,
    )

    predictor = None
    if runtime_profile:
        try:
            from saturo.predictors.time_predictor import TimePredictor
            predictor = TimePredictor(runtime_profile)
        except Exception:
            pass

    def predict_ms(prompt_len: int) -> float:
        if predictor is not None:
            try:
                return predictor.predict_forward_ms(
                    [(prompt_len, prompt_len)], [])
            except Exception:
                pass
        return prompt_len / 5000.0 * 1000.0

    rt_reqs = _pf_load_rt(
        trace_path=csv_path,
        num_req=num_req,
        target_rps=target_rps,
        default_runtime_s=default_runtime_s,
        runtime_predictor_ms=predict_ms,
    )
    return [
        _RTRequest(arrival_time_s=r.arrival_time_s, runtime_s=r.runtime_s)
        for r in rt_reqs
    ]


# ---------------------------------------------------------------------------
# Chunk latency resolution
# ---------------------------------------------------------------------------


def _resolve_chunk_latencies_ms(
    args: argparse.Namespace,
) -> List[float]:
    """Return a list of BE chunk latencies in milliseconds."""
    # Profile mode: use TimePredictor to map tokens → latency
    if args.runtime_profile and args.be_chunk_tokens_list:
        return _latencies_from_profile(
            args.runtime_profile,
            [int(x) for x in args.be_chunk_tokens_list.split(",") if x.strip()],
        )

    # Simple mode: direct latency values
    if args.be_chunk_ms_list:
        return [float(x) for x in args.be_chunk_ms_list.split(",") if x.strip()]

    raise ValueError(
        "Provide either --be-chunk-ms-list (simple mode) or "
        "--runtime-profile + --be-chunk-tokens-list (profile mode)"
    )


def _latencies_from_profile(
    profile_path: str,
    chunk_tokens: List[int],
) -> List[float]:
    """Map prefill chunk sizes in tokens to latency in ms via TimePredictor."""
    from saturo.predictors.time_predictor import TimePredictor
    from saturo.simulators.prefill_simulator.latency import LatencyPredictor

    tp = TimePredictor(profile_path)
    lp = LatencyPredictor(predictor=tp)

    latencies_ms: List[float] = []
    for tokens in chunk_tokens:
        # decode_batch_size=0 because BE output_len=1 (essentially pure prefill)
        lat_s = lp.predict_be_chunk_s(
            decode_batch_size=0,
            prefill_chunk_size=tokens,
            avg_be_decode_context=0,
        )
        latencies_ms.append(round(lat_s * 1000.0, 3))
    return latencies_ms


def _chunk_tokens_list(args: argparse.Namespace) -> List[int]:
    """Resolve the list of BE chunk token counts."""
    if args.be_chunk_tokens_list:
        return [int(x) for x in args.be_chunk_tokens_list.split(",") if x.strip()]
    if args.be_chunk_tokens:
        return [int(args.be_chunk_tokens)]
    return [4096]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep RT RPS × BE chunk size on the colocation simulator.")

    # ---- traces -----------------------------------------------------------
    p.add_argument("--rt-trace", required=True,
                   help="RT workload CSV (arrival_time, runtime_s).")
    p.add_argument("--default-runtime-s", type=float, default=None,
                   help="Fallback RT runtime if the CSV lacks a runtime column.")

    # ---- sweep grid --------------------------------------------------------
    p.add_argument("--num-req", type=int, required=True)
    p.add_argument("--target-rps", required=True,
                   help="Comma-separated RT RPS values.")

    # Simple mode: direct latency
    p.add_argument("--be-chunk-ms-list", default=None,
                   help="Comma-separated BE chunk latencies in ms "
                        "(simple mode, e.g. 10,25,50,100,200).")

    # Profile mode: token → latency mapping
    p.add_argument("--runtime-profile", default="",
                   help="Path to a Saturo runtime profile JSON for "
                        "TimePredictor-based latency estimation.")
    p.add_argument("--be-chunk-tokens-list", default=None,
                   help="Comma-separated BE chunk prefill tokens "
                        "(profile mode, e.g. 1024,2048,4096,8192).")
    p.add_argument("--be-chunk-tokens", type=int, default=4096,
                   help="Tokens per BE chunk in simple mode (default: 4096).")

    # ---- scheduling -------------------------------------------------------
    p.add_argument("--policy", choices=("greedy", "conservative"),
                   default="greedy",
                   help="BE scheduling policy. 'greedy' (default) starts BE "
                        "chunks whenever idle; 'conservative' only starts when "
                        "a chunk fits before the next RT arrival.")
    p.add_argument("--ttft-slo-ms", type=float, default=2000.0,
                   help="TTFT SLO threshold in ms (default: 2000).")

    # ---- output -----------------------------------------------------------
    p.add_argument("--output", default="",
                   help="Output CSV path.  Also writes a .json sidecar.")
    p.add_argument("--print-all", action="store_true",
                   help="Print every grid point instead of a compact summary.")

    args = p.parse_args()
    if args.num_req <= 0:
        p.error("--num-req must be positive")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return s[idx]


def main() -> None:
    args = parse_args()

    rps_list: List[float] = [
        float(x) for x in args.target_rps.split(",") if x.strip()
    ]
    if not rps_list or min(rps_list) <= 0:
        raise ValueError("--target-rps must contain positive values")
    chunk_ms_list = _resolve_chunk_latencies_ms(args)
    chunk_tokens = _chunk_tokens_list(args)

    print(f"RT-BE Colocation Simulation Sweep")
    print(f"  rt_trace       : {args.rt_trace}")
    print(f"  rps_list       : {rps_list}")
    print(f"  chunk_ms_list  : {chunk_ms_list}")
    print(f"  chunk_tokens   : {chunk_tokens}")
    print(f"  policy         : {args.policy}")
    print(f"  ttft_slo_ms    : {args.ttft_slo_ms}")
    if args.runtime_profile:
        print(f"  profile        : {args.runtime_profile}")
    print()

    results: List[ColocateResult] = []

    for rps in rps_list:
        # Load RT trace scaled to this RPS
        rt_requests = _load_rt_trace(
            csv_path=args.rt_trace,
            num_req=args.num_req,
            target_rps=rps,
            default_runtime_s=args.default_runtime_s,
            runtime_profile=args.runtime_profile,
        )
        if not rt_requests:
            print(f"[skip] RPS={rps}: no RT requests in window")
            continue

        from saturo.tools.trace_scaler import fixed_num_window_s
        window_s = fixed_num_window_s(args.num_req, rps)
        for chunk_ms in chunk_ms_list:
            # Determine token count for this chunk size
            tok = chunk_tokens[0] if len(chunk_tokens) == 1 else (
                chunk_tokens[chunk_ms_list.index(chunk_ms)]
                if chunk_ms in chunk_ms_list else chunk_tokens[0]
            )

            config = ColocateConfig(
                ttft_slo_s=args.ttft_slo_ms / 1000.0,
                duration_s=window_s,
                be_chunk_latency_s=chunk_ms / 1000.0,
                be_chunk_tokens=tok,
                policy=args.policy,
            )

            sim = ColocateSimulator(rt_requests, config)
            result = sim.run()
            results.append(result)

            d = result.to_dict()
            d["rt_requests_in_window"] = len(rt_requests)
            print(
                f"  RPS={rps:5.1f}  chunk={chunk_ms:8.2f}ms  "
                f"RT_viol={result.rt_violation_rate:.4f}  "
                f"BE_tput={result.be_throughput_tok_s:8.1f} tok/s  "
                f"BE_util={result.be_utilization:.3f}  "
                f"RT_p95={result.rt_ttft_ms_p95:.1f}ms"
            )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if not results:
        print("\nNo results to write.", file=sys.stderr)
        return

    # CSV
    csv_path = args.output or "rt_be_colocate_sim.csv"
    json_path = Path(csv_path).with_suffix(".json")

    fieldnames = [
        "target_rps", "be_chunk_latency_ms", "be_chunk_tokens", "policy",
        "rt_total", "rt_completed", "rt_violations", "rt_violation_rate",
        "rt_ttft_ms_p50", "rt_ttft_ms_p95", "rt_ttft_ms_p99",
        "rt_mean_queue_s",
        "be_chunks_completed", "be_total_tokens", "be_throughput_tok_s",
        "be_utilization", "rt_utilization", "idle_time_s",
    ]

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r.to_dict())

    # JSON sidecar
    json_payload = {
        "config": {
            "rt_trace": args.rt_trace,
            "num_req": args.num_req,
            "target_rps": rps_list,
            "chunk_ms_list": chunk_ms_list,
            "chunk_tokens": chunk_tokens,
            "policy": args.policy,
            "ttft_slo_ms": args.ttft_slo_ms,
            "runtime_profile": args.runtime_profile or "none",
        },
        "points": [r.to_dict() for r in results],
    }
    json_path.write_text(json.dumps(json_payload, indent=2))

    print(f"\n[output] CSV: {csv_path}")
    print(f"[output] JSON: {json_path}")

    # Compact summary table (if --print-all was passed)
    if args.print_all and len(results) >= 2:
        print("\n" + "=" * 75)
        print("Summary grid (RT violation rate)")
        print("=" * 75)
        chunk_set = sorted(set(r.config.be_chunk_latency_s * 1000 for r in results))
        header = f"{'RPS':>6s}"
        for c in chunk_set:
            header += f"  {c:>8.1f}ms"
        print(header)
        print("-" * len(header))
        for rps in sorted(set(r.target_rps for r in results)):
            row = f"{rps:>6.1f}"
            for c in chunk_set:
                match = [r for r in results
                         if abs(r.target_rps - rps) < 0.001
                         and abs(r.config.be_chunk_latency_s * 1000 - c) < 0.01]
                if match:
                    row += f"  {match[0].rt_violation_rate:>8.4f}"
                else:
                    row += f"  {'--':>8s}"
            print(row)

        print(f"\n{'RPS':>6s}{'':>6s}BE throughput (tok/s)")
        print("-" * len(header))
        for rps in sorted(set(r.target_rps for r in results)):
            row = f"{rps:>6.1f}"
            for c in chunk_set:
                match = [r for r in results
                         if abs(r.target_rps - rps) < 0.001
                         and abs(r.config.be_chunk_latency_s * 1000 - c) < 0.01]
                if match:
                    row += f"  {match[0].be_throughput_tok_s:>8.1f}"
                else:
                    row += f"  {'--':>8s}"
            print(row)


if __name__ == "__main__":
    main()
