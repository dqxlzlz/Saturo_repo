"""CLI: sweep be_chunk_ms with the local BE-lifecycle P-pool simulator.

For each candidate ``be_chunk_ms`` this runs one FULL time-window simulation
(BE requests go through a real prefill -> decode lifecycle, decode batch is
emergent) and reports the achieved BE throughput plus RT SLO attainment, then
prints the throughput-maximizing chunk within the RT violation tolerance.
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from saturo.predictors.time_predictor import TimePredictor  # noqa: E402
from saturo.simulators.prefill_simulator.local_be_simulator import (  # noqa: E402,E501
    LocalBESimulator,
)
from saturo.simulators.prefill_simulator.trace_loader import (  # noqa: E402
    load_rt_trace,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local BE-lifecycle P-pool simulator (chunk-ms sweep)")
    p.add_argument("--rt-trace", required=True)
    p.add_argument("--be-trace", required=True)
    p.add_argument("--runtime-profile", required=True)
    p.add_argument("--num-req", type=int, required=True,
                   help="Number of chronological RT requests to replay.")
    p.add_argument("--target-rps", type=float, required=True)
    p.add_argument("--ttft-slo", type=float, default=1.0)
    p.add_argument("--max-rt-violation-rate", type=float, default=0.001)
    p.add_argument("--be-input-max-percentile", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--memory-watermark", type=float, default=0.9)
    p.add_argument("--be-decode-block-fraction", type=float, default=0.95)
    p.add_argument("--max-prefill-chunk", type=int, default=1000000,
                   help="Hard cap on a single prefill slice. Default is "
                        "effectively uncapped: the latency budget and the "
                        "request's own remaining prompt bound the slice.")
    p.add_argument("--prefill-search-granularity", type=int, default=16)
    p.add_argument(
        "--sweep-unit", choices=("tokens", "ms"), default="tokens",
        help="Sweep BE chunk in prompt tokens (plain chunked prefill, "
             "default) or in per-step latency budget (ms).")
    p.add_argument(
        "--chunk-token-grid", type=str, default="512:2049:128",
        help="start:stop:step token grid for --sweep-unit=tokens.")
    p.add_argument(
        "--chunk-ms-grid", type=str,
        default="100,150,200,250,300,350,400,450,500,600,700,800",
        help="Comma-separated be_chunk_ms candidates for --sweep-unit=ms.")
    p.add_argument("--be-supply-mult", type=int, default=4,
                   help="Replicate the BE trace this many times as an "
                        "(approx) infinite supply.")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def load_be_supply(
    path: str, input_max_percentile: Optional[float],
    max_model_len: Optional[int], mult: int,
) -> Tuple[List[Tuple[int, int]], int, int, int]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = [(int(float(r["prompt_len"])), int(float(r["output_len"])))
             for r in rows]
    if input_max_percentile is not None:
        q = float(input_max_percentile)
        ordered = sorted(p for p, _ in pairs)
        rank = max(0, min(len(ordered) - 1,
                          int(math.ceil(q * len(ordered))) - 1))
        cutoff = ordered[rank]
        pairs = [p for p in pairs if p[0] <= cutoff]
    if max_model_len is not None:
        limit = int(max_model_len)
        pairs = [(pl, min(ol, limit - pl)) for pl, ol in pairs if pl < limit]
    if not pairs:
        raise ValueError("BE supply empty after filtering")
    avg_in = int(round(sum(p[0] for p in pairs) / len(pairs)))
    avg_out = int(round(sum(p[1] for p in pairs) / len(pairs)))
    max_in = max(p[0] for p in pairs)
    supply = pairs * max(1, mult)
    return supply, avg_in, avg_out, max_in


def slack_stats_ms(slacks) -> dict:
    if not slacks:
        return {"min_ms": 0.0, "p999min_ms": 0.0}
    ordered = sorted(slacks)
    drop = int(0.001 * len(ordered))
    return {
        "min_ms": round(ordered[0] * 1000.0, 1),
        "p999min_ms": round(ordered[min(len(ordered) - 1, drop)] * 1000.0, 1),
    }


def ttft_stats_ms(slacks, ttft_slo_s: float) -> dict:
    """Recover request TTFTs from the simulator's SLO slack samples."""
    ordered = sorted(
        max(0.0, (float(ttft_slo_s) - float(slack)) * 1000.0)
        for slack in slacks
    )
    if not ordered:
        return {
            "count": 0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    def percentile(q: float) -> float:
        rank = int(math.ceil(q * len(ordered))) - 1
        return ordered[min(len(ordered) - 1, max(0, rank))]

    return {
        "count": len(ordered),
        "p50": round(percentile(0.50), 3),
        "p90": round(percentile(0.90), 3),
        "p95": round(percentile(0.95), 3),
        "p99": round(percentile(0.99), 3),
        "max": round(ordered[-1], 3),
    }


def main() -> None:
    args = parse_args()

    with open(args.runtime_profile, "r", encoding="utf-8") as f:
        profile = json.load(f)
    kv = profile.get("kv_cache_block", {})
    total_blocks = int(kv["total_gpu_blocks"])
    block_size = int(kv["block_size"])

    predictor = TimePredictor(args.runtime_profile)

    be_supply, be_avg_in, be_avg_out, be_max_in = load_be_supply(
        args.be_trace, args.be_input_max_percentile, args.max_model_len,
        args.be_supply_mult)

    rt_requests = load_rt_trace(
        trace_path=args.rt_trace,
        num_req=args.num_req,
        target_rps=args.target_rps,
        runtime_predictor_ms=lambda pl: predictor.predict_forward_ms(
            [(max(1, pl), max(1, pl))], []),
    )

    # KV budget for BE: reserve space for the longest RT prefill, then hand the
    # rest (scaled by the block fraction) to BE prefill+decode.
    rt_prompts = [r.prompt_len for r in rt_requests
                  if args.max_model_len is None
                  or r.prompt_len < int(args.max_model_len)]
    max_rt_prompt = max(rt_prompts, default=1)
    reserved_rt = math.ceil(max(1, max_rt_prompt) / block_size)
    usable = math.floor(total_blocks * args.memory_watermark
                        * args.be_decode_block_fraction) - reserved_rt
    usable = max(1, usable)

    if args.sweep_unit == "tokens":
        start, stop, step = (int(x) for x in args.chunk_token_grid.split(":"))
        grid = list(range(start, stop, step))
    else:
        grid = [float(x) for x in args.chunk_ms_grid.split(",") if x.strip()]

    print(f"[local-be-sim] profile total_blocks={total_blocks} "
          f"block_size={block_size} usable_be_blocks={usable}")
    print(f"[local-be-sim] BE avg_in/out={be_avg_in}/{be_avg_out} "
          f"supply={len(be_supply)} reqs | RT n={len(rt_requests)} "
          f"target_rps={args.target_rps} sweep_unit={args.sweep_unit}")
    label = "chunk(tok)" if args.sweep_unit == "tokens" else "chunk(ms)"
    header = (f"{label:>10} {'stepAvg':>8} {'BE req/s':>9} {'BE tok/s':>9} "
              f"{'avgDecB':>8} {'maxDecB':>8} {'preempt':>8} {'preR':>7} "
              f"{'RT-SLO%':>8} {'viol':>5} {'minSlk':>8} {'p999min':>8}")
    print(header)
    print("-" * len(header))

    results = []
    tol = args.max_rt_violation_rate
    for cand in grid:
        sim = LocalBESimulator(
            rt_requests=rt_requests,
            be_supply=be_supply,
            predictor=predictor,
            ttft_slo_s=args.ttft_slo,
            usable_kv_blocks=usable,
            block_size=block_size,
            max_prefill_chunk=args.max_prefill_chunk,
            prefill_search_granularity=args.prefill_search_granularity,
        )
        ctx = be_avg_in + be_avg_out // 2
        if args.sweep_unit == "tokens":
            r = sim.run(0.0, avg_decode_ctx=ctx, chunk_tokens=int(cand))
        else:
            r = sim.run(float(cand), avg_decode_ctx=ctx)
        att = 1.0 - r.rt_violation_rate
        pre_ratio = r.preemptions / max(1, r.preemptions + r.steps)
        st = slack_stats_ms(r.rt_slacks_s)
        step_avg_ms = r.busy_be_time_s / max(1, r.steps) * 1000.0
        results.append((r, pre_ratio, st, step_avg_ms))
        shown = f"{int(cand):>10}" if args.sweep_unit == "tokens" \
            else f"{cand:>10.1f}"
        print(f"{shown} {step_avg_ms:>8.1f} {r.be_req_per_s:>9.4f} "
              f"{r.be_token_per_s:>9.0f} {r.avg_decode_batch:>8.1f} "
              f"{r.max_decode_batch_seen:>8} {r.preemptions:>8} "
              f"{pre_ratio:>7.4f} {att * 100:>7.2f}% {r.rt_violations:>5} "
              f"{st['min_ms']:>8.1f} {st['p999min_ms']:>8.1f}")

    feasible = [x for x in results if x[0].rt_violation_rate <= tol]
    pool = feasible if feasible else results
    # Tie-break: among candidates within 1% of the max throughput, prefer the
    # SMALLEST chunk (finest preemption granularity / largest tail margin),
    # so flat-plateau noise no longer picks razor-edge large chunks.
    top = max(x[0].be_req_per_s for x in pool)
    near = [x for x in pool if x[0].be_req_per_s >= 0.99 * top]
    best, best_ratio, best_st, best_step_ms = min(
        near, key=lambda x: (x[0].chunk_tokens or x[0].be_chunk_ms))
    print("\n[Best within RT tolerance]" if feasible
          else "\n[Best (no candidate met tolerance)]")
    if args.sweep_unit == "tokens":
        print(f"  be_chunk_size     : {best.chunk_tokens} tokens "
              f"(avg step {best_step_ms:.1f} ms)")
    else:
        print(f"  be_chunk_ms       : {best.be_chunk_ms:.1f} ms")
    print(f"  BE throughput     : {best.be_req_per_s:.4f} req/s "
          f"({best.be_token_per_s:.0f} tok/s)")
    print(f"  emergent decode B : avg {best.avg_decode_batch:.1f}, "
          f"max {best.max_decode_batch_seen}")
    print(f"  completed BE      : {best.completed_be}")
    print(f"  RT SLO attain     : {(1 - best.rt_violation_rate) * 100:.3f}% "
          f"({best.rt_violations} viol / {best.rt_total})")
    print(f"  preempt ratio     : {best_ratio:.4f} "
          f"({best.preemptions} / {best.preemptions + best.steps} steps)")
    print(f"  min slack         : {best_st['min_ms']:.1f} ms "
          f"(p999 {best_st['p999min_ms']:.1f} ms)")
    best_ttft = ttft_stats_ms(best.rt_slacks_s, args.ttft_slo)
    print(f"  RT TTFT P99       : {best_ttft['p99']:.3f} ms")

    if args.output:
        payload = {
            "config": {
                "num_req": args.num_req, "target_rps": args.target_rps,
                "ttft_slo_s": args.ttft_slo, "usable_be_blocks": usable,
                "be_avg_in": be_avg_in, "be_avg_out": be_avg_out,
                "max_prefill_chunk": args.max_prefill_chunk,
            },
            "best": {
                "be_chunk_ms": best.be_chunk_ms,
                "chunk_tokens": best.chunk_tokens,
                "avg_step_ms": best_step_ms,
                "be_req_per_s": best.be_req_per_s,
                "be_token_per_s": best.be_token_per_s,
                "avg_decode_batch": best.avg_decode_batch,
                "max_decode_batch": best.max_decode_batch_seen,
                "rt_violation_rate": best.rt_violation_rate,
                "rt_violations": best.rt_violations,
                "rt_ttft_ms": best_ttft,
                "preempt_ratio": best_ratio,
                "preemptions": best.preemptions,
                "min_slack_ms": best_st["min_ms"],
                "p999_min_slack_ms": best_st["p999min_ms"],
                "feasible": bool(feasible),
            },
            "sweep": [
                {"be_chunk_ms": r.be_chunk_ms,
                 "chunk_tokens": r.chunk_tokens,
                 "avg_step_ms": step_ms,
                 "be_req_per_s": r.be_req_per_s,
                 "be_token_per_s": r.be_token_per_s,
                 "avg_decode_batch": r.avg_decode_batch,
                 "max_decode_batch": r.max_decode_batch_seen,
                 "completed_be": r.completed_be,
                 "preemptions": r.preemptions,
                 "preempt_ratio": ratio,
                 "rt_violations": r.rt_violations,
                 "rt_violation_rate": r.rt_violation_rate,
                 "rt_ttft_ms": ttft_stats_ms(
                     r.rt_slacks_s, args.ttft_slo),
                 "min_slack_ms": st["min_ms"],
                 "p999_min_slack_ms": st["p999min_ms"],
                 "window_s": r.window_s}
                for r, ratio, st, step_ms in results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Output] wrote {args.output}")


if __name__ == "__main__":
    main()
