"""CLI entry point for the P-pool prefill saturation simulator."""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from saturo.predictors.time_predictor import TimePredictor  # noqa: E402
from saturo.simulators.prefill_simulator.config import (  # noqa: E402
    BEConfig,
    SLOConfig,
    SimulatorConfig,
)
from saturo.simulators.prefill_simulator.latency import LatencyPredictor  # noqa: E402, E501
from saturo.simulators.prefill_simulator.optimizer import (  # noqa: E402
    ChunkLatencyOptimizer,
    OptimizationResult,
)
from saturo.simulators.prefill_simulator.request import SimulationResult  # noqa: E402
from saturo.simulators.prefill_simulator.trace_loader import (  # noqa: E402
    load_be_trace,
    load_rt_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Saturo P-pool prefill saturation simulator")

    parser.add_argument(
        "--rt-trace",
        required=True,
        help="RT history trace CSV. Needs TIMESTAMP/arrival_time and one "
             "runtime column such as forward_time, runtime_s, or latency_ms.",
    )
    parser.add_argument(
        "--runtime-profile",
        required=True,
        help="Runtime profile JSON (contains time_predictor + "
             "kv_cache_block).",
    )
    parser.add_argument(
        "--be-trace",
        required=True,
        help="BE workload trace CSV with prompt_len, output_len columns.",
    )
    parser.add_argument(
        "--be-input-drop-frac",
        type=float,
        default=0.0,
        help="Drop the shortest-input fraction of BE requests (0,1) to "
             "lengthen the average prompt. Barely moves the optimal chunk.",
    )
    parser.add_argument(
        "--be-input-max-percentile",
        type=float,
        default=None,
        help="Drop BE prompts above this nearest-rank percentile. Uses the "
             "same filtering order as the real P-pool benchmark.",
    )
    parser.add_argument(
        "--be-output-keep-frac",
        type=float,
        default=None,
        help="Keep only the shortest-output fraction of BE requests (0,1]. "
             "Lowers avg_out, which is the effective knob to raise the "
             "optimal P/D balance chunk (C* = B * avg_in / avg_out).",
    )
    parser.add_argument(
        "--be-output-max-len",
        type=int,
        default=None,
        help="Keep only BE requests whose output_len <= this absolute "
             "threshold. Takes precedence over --be-output-keep-frac.",
    )
    parser.add_argument(
        "--num-req",
        type=int,
        required=True,
        help="Replay exactly this many chronological RT requests.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Apply the same prompt filtering/output clamping as the real "
             "vLLM instance.",
    )
    parser.add_argument(
        "--total-gpu-blocks",
        type=int,
        default=None,
        help="Override kv_cache_block.total_gpu_blocks with the real "
             "instance's post-initialization block count.",
    )
    parser.add_argument(
        "--target-rps",
        type=float,
        required=True,
        help="Per-instance RPS for the selected fixed RT request set.",
    )
    parser.add_argument(
        "--default-runtime",
        type=float,
        default=None,
        help="Fallback RT runtime in seconds if the trace lacks runtime cols.",
    )
    parser.add_argument("--ttft-slo", type=float, default=2.0)
    parser.add_argument(
        "--max-rt-violation-rate",
        type=float,
        default=0.01,
        help="If the pure-RT (BE-free) TTFT-SLO violation rate already exceeds "
             "this fraction, skip the BE search and report saturation=0 "
             "(the RPS point is beyond RT capacity). Range [0, 1].",
    )

    parser.add_argument("--memory-watermark", type=float, default=0.9)
    parser.add_argument("--max-prefill-chunk", type=int, default=8192)
    parser.add_argument("--min-prefill-chunk", type=int, default=0)
    parser.add_argument("--prefill-search-granularity", type=int, default=1)

    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument(
        "--max-budget-iterations", type=int, default=1024,
        help="Safety limit for slack-driven budget updates.")
    parser.add_argument(
        "--budget-granularity-ms",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print evaluated candidates instead of the compact summary.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print human-readable diagnostics: pure-RT slack stats, the "
             "KV-derived BE decode cap, the slack-driven search and the "
             "full candidate curve (budget -> chunk -> saturation).",
    )
    args = parser.parse_args()
    if args.num_req <= 0:
        parser.error("--num-req must be positive")
    if args.target_rps <= 0:
        parser.error("--target-rps must be positive")
    if args.max_budget_iterations <= 0:
        parser.error("--max-budget-iterations must be positive")
    return args


def build_config(
    args: argparse.Namespace,
    be_avg_in: int,
    be_avg_out: int,
    *,
    be_decode_free_blocks: int,
    block_size: int,
) -> SimulatorConfig:
    """Build SimulatorConfig from parsed CLI args and loaded data.

    ``be_decode_free_blocks`` is the KV block budget left for BE decode after
    reserving space for the longest RT prefill and the longest BE prefill,
    mirroring the real P-pool engine's dataset-derived decode limit.
    """
    return SimulatorConfig(
        duration_s=args.duration,
        run_to_completion=getattr(args, "run_to_completion", False),
        slo=SLOConfig(
            ttft_slo_s=args.ttft_slo,
            max_rt_violation_rate=args.max_rt_violation_rate,
        ),
        be=BEConfig(
            free_gpu_blocks=int(be_decode_free_blocks),
            block_size=int(block_size),
            memory_watermark=args.memory_watermark,
            avg_be_length=be_avg_in,
            avg_be_output_len=be_avg_out,
            avg_be_decode_context=be_avg_in,
            max_prefill_chunk=args.max_prefill_chunk,
            min_prefill_chunk=args.min_prefill_chunk,
            prefill_search_granularity=args.prefill_search_granularity,
        ),
    )


def result_to_dict(result: SimulationResult) -> Dict[str, Any]:
    chunk = result.chunk
    return {
        "target_latency_ms": round(chunk.target_latency_s * 1000.0, 3),
        "be_chunk_latency_ms": round(chunk.latency_s * 1000.0, 3),
        "be_decode_batch_size": chunk.decode_batch_size,
        "be_prefill_chunk_size": chunk.prefill_chunk_size,
        "completed_be_chunks": result.completed_be_chunks,
        "preemptions": result.preemptions,
        "preempt_ratio": round(result.preempt_ratio, 6),
        "wasted_time_ratio": round(result.wasted_time_ratio, 6),
        "rt_requests": result.total_rt_requests,
        "rt_completed": result.completed_rt_requests,
        "rt_slo_violations": result.rt_slo_violations,
        "min_rt_slack_ms": round(result.min_slack_s * 1000.0, 3),
        "prefill_req_per_s": round(result.prefill_req_per_s, 6),
        "decode_req_per_s": round(result.decode_req_per_s, 6),
        "saturation_req_per_s": round(result.saturation_req_per_s, 6),
        "be_token_per_s": round(result.be_token_per_s, 3),
        "useful_be_time_s": round(result.useful_be_time_s, 6),
        "wasted_be_time_s": round(result.wasted_be_time_s, 6),
        "idle_time_s": round(result.idle_time_s, 6),
        "total_prefill_tokens": result.total_prefill_tokens,
        "total_decode_tokens": result.total_decode_tokens,
    }


def _slack_tail_stats_ms(slacks_s: list) -> Dict[str, Any]:
    """Min slack after dropping the worst q of requests, q in 0..1% grid.

    Mirrors the violation-rate tolerance: dropping floor(q*n) worst slacks
    shows how much headroom the remaining (tolerated) population keeps.
    """
    if not slacks_s:
        return {"num": 0}
    ordered = sorted(slacks_s)
    n = len(ordered)

    def after_drop(q: float) -> float:
        return round(ordered[min(n - 1, int(q * n))] * 1000.0, 3)

    grid = {
        f"{q / 10.0:.1f}pct": after_drop(q / 1000.0)
        for q in range(0, 11)  # 0.0%, 0.1%, ..., 1.0%
    }
    return {
        "num": n,
        "min_ms": round(ordered[0] * 1000.0, 3),
        "min_after_drop_0.1pct_ms": after_drop(0.001),
        "min_after_drop_1pct_ms": after_drop(0.01),
        "min_after_drop_grid_ms": grid,
        "mean_ms": round(sum(ordered) / n * 1000.0, 3),
    }


def optimization_to_dict(opt: OptimizationResult) -> Dict[str, Any]:
    pure_slacks = opt.pure_rt_slacks_s
    return {
        "best": result_to_dict(opt.best_result),
        "rt_infeasible": opt.rt_infeasible,
        "rt_violation_rate": round(opt.rt_violation_rate, 6),
        "decode_only_latency_ms": round(opt.decode_only_latency_s * 1000.0, 3),
        "max_candidate_latency_ms": (
            round(opt.max_candidate_latency_s * 1000.0, 3)
            if opt.max_candidate_latency_s is not None else None
        ),
        "search": {
            "method": "slack",
            "initial_budget_ms": opt.initial_budget_s * 1000.0,
            "evaluated_candidates": opt.evaluated_candidates,
            "stop_reason": opt.stop_reason,
            "budgets_ms": [budget * 1000.0 for budget in opt.budget_history_s],
        },
        "pure_rt": {
            "num_slacks": len(pure_slacks),
            "min_slack_ms": (
                round(min(pure_slacks) * 1000.0, 3) if pure_slacks else 0.0),
            "mean_slack_ms": (
                round(sum(pure_slacks) / len(pure_slacks) * 1000.0, 3)
                if pure_slacks else 0.0),
            "violations": sum(1 for slack in pure_slacks if slack < 0.0),
            "slack_tail": _slack_tail_stats_ms(pure_slacks),
        },
        "best_rt_slack_tail": _slack_tail_stats_ms(
            opt.best_result.rt_slacks_s),
        "candidates": [result_to_dict(r) for r in opt.all_results],
    }


def print_summary(payload: Dict[str, Any], print_all: bool) -> None:
    pure = payload["pure_rt"]
    best = payload["best"]
    print("\n[Pure RT]")
    print(f"  requests          : {pure['num_slacks']}")
    print(f"  min slack         : {pure['min_slack_ms']:.3f} ms")
    print(f"  mean slack        : {pure['mean_slack_ms']:.3f} ms")
    print(f"  pure violations   : {pure['violations']}")
    print(f"  decode-only BE lat: {payload['decode_only_latency_ms']:.3f} ms")
    if payload["max_candidate_latency_ms"] is None:
        print("  candidate lat cap : none")
    else:
        print(f"  candidate lat cap : {payload['max_candidate_latency_ms']:.3f} ms")

    sweep = payload.get("sweep")
    if sweep is not None:
        search = payload["search"]
        print("\n[Slack-driven budget search]")
        print(f"  initial budget    : {search['initial_budget_ms']:.3f} ms")
        print(f"  evaluated         : {search['evaluated_candidates']}")
        print(f"  stop reason       : {search['stop_reason']}")
        print(f"  largest evaluated : {sweep['max_budget_ms']} ms")
        print(f"  candidates kept   : {sweep['num_candidates_kept']}")
        print(f"  BE avg in/out     : {sweep['be_avg_input_len']} / "
              f"{sweep['be_avg_output_len']} tokens")
        print(f"  decode KV cap     : {sweep['decode_capacity_kv_cap']} reqs "
              "(free_blocks*block*watermark / (avg_in+avg_out))")

    if print_all:
        print("\n[Candidates]  (budget -> balanced chunk -> net saturation)")
        header = (
            f"{'budget(ms)':>11} {'lat(ms)':>10} {'dec':>6} {'prefill':>8} "
            f"{'chunks':>7} {'preempt':>8} {'pre_ratio':>10} {'waste%':>7} "
            f"{'viol':>6} {'sat(req/s)':>11} {'min_slack(ms)':>14}"
        )
        print(header)
        print("-" * len(header))
        best_lat = best["be_chunk_latency_ms"]
        for item in payload["candidates"]:
            mark = " <== best" if item["be_chunk_latency_ms"] == best_lat \
                else ""
            print(
                f"{item['target_latency_ms']:>11.3f} "
                f"{item['be_chunk_latency_ms']:>10.3f} "
                f"{item['be_decode_batch_size']:>6} "
                f"{item['be_prefill_chunk_size']:>8} "
                f"{item['completed_be_chunks']:>7} "
                f"{item['preemptions']:>8} "
                f"{item['preempt_ratio']:>10.4f} "
                f"{item['wasted_time_ratio'] * 100.0:>6.2f}% "
                f"{item['rt_slo_violations']:>6} "
                f"{item['saturation_req_per_s']:>11.6f} "
                f"{item['min_rt_slack_ms']:>14.3f}"
                f"{mark}"
            )

    print("\n[Best]")
    print(f"  BE chunk latency  : {best['be_chunk_latency_ms']:.3f} ms")
    print(f"  BE decode batch   : {best['be_decode_batch_size']}")
    print(f"  BE prefill chunk  : {best['be_prefill_chunk_size']} tokens")
    print(f"  completed chunks  : {best['completed_be_chunks']}")
    print(f"  preemptions       : {best['preemptions']}")
    print(f"  preempt ratio     : {best['preempt_ratio']:.4f} "
          "(preempted / launched chunks)")
    print(f"  wasted time ratio : {best['wasted_time_ratio'] * 100.0:.2f}% "
          "(wasted BE time / non-RT time)")
    print(f"  RT SLO violations : {best['rt_slo_violations']}")
    print(f"  prefill tput      : {best['prefill_req_per_s']:.6f} req/s")
    print(f"  decode tput       : {best['decode_req_per_s']:.6f} req/s")
    print(f"  saturation BE tput: {best['saturation_req_per_s']:.6f} req/s")
    print(f"  BE token tput     : {best['be_token_per_s']:.3f} tok/s")


def run_prefill_profile(
    *,
    rt_trace: str,
    runtime_profile: str,
    be_trace: str,
    num_req: int,
    target_rps: float,
    default_runtime: float | None = None,
    ttft_slo: float = 2.0,
    max_rt_violation_rate: float = 0.01,
    memory_watermark: float = 0.9,
    max_prefill_chunk: int = 8192,
    min_prefill_chunk: int = 0,
    prefill_search_granularity: int = 1,
    max_candidates: int = 128,
    budget_granularity_ms: float | None = None,
    max_budget_iterations: int = 1024,
    max_model_len: int | None = None,
    total_gpu_blocks: int | None = None,
    be_input_drop_frac: float = 0.0,
    be_input_max_percentile: float | None = None,
    be_output_keep_frac: float | None = None,
    be_output_max_len: int | None = None,
    be_decode_block_fraction: float = 0.95,
    rt_rows: list[dict] | None = None,
) -> Dict[str, Any]:
    """Run one P-pool saturation profile and return the JSON payload."""

    # Load runtime profile JSON and extract kv_cache_block values
    with open(runtime_profile, "r", encoding="utf-8") as f:
        profile_data = json.load(f)

    tp_data = profile_data.get("time_predictor")
    if not isinstance(tp_data, dict):
        raise ValueError(
            f"runtime_profile {runtime_profile} is missing "
            "time_predictor section")

    kv = profile_data.get("kv_cache_block", {})
    if "total_gpu_blocks" not in kv or "block_size" not in kv:
        raise ValueError(
            "runtime_profile is missing kv_cache_block section "
            "(total_gpu_blocks, block_size)")
    block_size = int(kv["block_size"])
    total_blocks = int(
        total_gpu_blocks if total_gpu_blocks is not None
        else kv["total_gpu_blocks"])

    # Load BE trace -> average token lengths + longest (post-filter) prompt.
    be_avg_in, be_avg_out, be_max_in = load_be_trace(
        be_trace,
        input_drop_frac=be_input_drop_frac,
        input_max_percentile=be_input_max_percentile,
        output_keep_frac=be_output_keep_frac,
        output_max_len=be_output_max_len,
        max_model_len=max_model_len,
    )

    from saturo.tools.trace_scaler import fixed_num_window_s
    window_s = fixed_num_window_s(num_req, target_rps)
    args = argparse.Namespace(
        duration=window_s,
        run_to_completion=True,
        ttft_slo=ttft_slo,
        max_rt_violation_rate=max_rt_violation_rate,
        memory_watermark=memory_watermark,
        max_prefill_chunk=max_prefill_chunk,
        min_prefill_chunk=min_prefill_chunk,
        prefill_search_granularity=prefill_search_granularity,
        total_gpu_blocks=total_gpu_blocks,
    )

    # Construct TimePredictor and LatencyPredictor
    time_predictor = TimePredictor(runtime_profile)
    latency_predictor = LatencyPredictor(predictor=time_predictor)

    # Load RT trace
    rt_requests = load_rt_trace(
        trace_path=rt_trace,
        rows=rt_rows,
        num_req=num_req,
        target_rps=target_rps,
        default_runtime_s=default_runtime,
        runtime_predictor_ms=lambda prompt_len: (
            time_predictor.predict_forward_ms(
                [(max(1, prompt_len), max(1, prompt_len))],
                [],
            )
        ),
    )

    # Peek at the longest RT and BE prefill to reserve KV blocks for them, cap
    # decode at a fraction of total blocks so it never fills the whole KV cache
    # and starves BE prefill, then give the rest to BE decode -- the same
    # dataset-derived split the real P-pool engine applies.
    rt_prompts = [
        r.prompt_len for r in rt_requests
        if max_model_len is None or r.prompt_len < int(max_model_len)
    ]
    max_rt_prompt = max(rt_prompts, default=1)
    reserved_rt_blocks = math.ceil(max(1, max_rt_prompt) / block_size)
    reserved_be_prefill_blocks = math.ceil(max(1, be_max_in) / block_size)
    usable_decode_blocks = math.floor(total_blocks * be_decode_block_fraction)
    be_decode_free_blocks = max(
        1,
        usable_decode_blocks
        - reserved_rt_blocks
        - reserved_be_prefill_blocks)

    config = build_config(
        args, be_avg_in, be_avg_out,
        be_decode_free_blocks=be_decode_free_blocks,
        block_size=block_size,
    )

    # Run optimizer
    optimizer = ChunkLatencyOptimizer(
        config, latency_predictor=latency_predictor)
    opt = optimizer.optimize(
        rt_requests,
        max_candidates=max_candidates,
        budget_granularity_ms=budget_granularity_ms,
        max_iterations=max_budget_iterations,
    )
    payload = optimization_to_dict(opt)
    payload["workload"] = {
        "num_req": num_req,
        "target_rps": target_rps,
        "window_s": window_s,
    }
    payload["sweep"] = {
        "search_method": "slack",
        "budget_granularity_ms": None,
        "max_budget_ms": payload["max_candidate_latency_ms"],
        "num_candidates_kept": len(payload["candidates"]),
        "be_avg_input_len": be_avg_in,
        "be_avg_output_len": be_avg_out,
        "decode_capacity_kv_cap": int(config.be.decode_batch_size),
    }
    payload["block_reservation"] = {
        "total_gpu_blocks": total_blocks,
        "block_size": block_size,
        "max_rt_prompt": max_rt_prompt,
        "max_be_prompt": be_max_in,
        "decode_block_fraction": be_decode_block_fraction,
        "usable_decode_blocks": usable_decode_blocks,
        "reserved_rt_blocks": reserved_rt_blocks,
        "reserved_be_prefill_blocks": reserved_be_prefill_blocks,
        "be_decode_free_blocks": be_decode_free_blocks,
    }

    # Balance the BE-decode active count against BE-prefill throughput under the
    # same be_chunk_ms budget the real engine will use (the best candidate's
    # achieved chunk latency). This yields a max_be_decode_batch cap so the real
    # engine stops greedily filling decode to the block ceiling and starving
    # BE prefill.
    decode_capacity = int(config.be.decode_batch_size)
    target_latency_s = float(opt.best_result.chunk.latency_s)
    if opt.rt_infeasible or target_latency_s <= 0.0 or decode_capacity <= 0:
        payload["balanced_be_decode"] = {
            "max_be_decode_batch": 0,
            "decode_capacity_batch": decode_capacity,
            "be_prefill_chunk_size": 0,
            "be_chunk_latency_ms": 0.0,
            "decode_req_per_s": 0.0,
            "prefill_req_per_s": 0.0,
            "saturation_req_per_s": 0.0,
        }
    else:
        balanced = latency_predictor.balance_decode_batch(
            target_latency_s, config.be, max_decode_batch=decode_capacity)
        payload["balanced_be_decode"] = {
            "max_be_decode_batch": int(balanced["decode_batch"]),
            "decode_capacity_batch": decode_capacity,
            "be_prefill_chunk_size": int(balanced["prefill_chunk"]),
            "be_chunk_latency_ms": round(balanced["latency_s"] * 1000.0, 3),
            "decode_req_per_s": round(balanced["decode_req_s"], 6),
            "prefill_req_per_s": round(balanced["prefill_req_s"], 6),
            "saturation_req_per_s": round(
                min(balanced["decode_req_s"], balanced["prefill_req_s"]), 6),
        }
    return payload


def main() -> None:
    args = parse_args()
    payload = run_prefill_profile(
        rt_trace=args.rt_trace,
        runtime_profile=args.runtime_profile,
        be_trace=args.be_trace,
        num_req=args.num_req,
        target_rps=args.target_rps,
        default_runtime=args.default_runtime,
        ttft_slo=args.ttft_slo,
        max_rt_violation_rate=args.max_rt_violation_rate,
        memory_watermark=args.memory_watermark,
        max_prefill_chunk=args.max_prefill_chunk,
        min_prefill_chunk=args.min_prefill_chunk,
        prefill_search_granularity=args.prefill_search_granularity,
        max_candidates=args.max_candidates,
        budget_granularity_ms=args.budget_granularity_ms,
        max_budget_iterations=args.max_budget_iterations,
        max_model_len=args.max_model_len,
        total_gpu_blocks=args.total_gpu_blocks,
        be_input_drop_frac=args.be_input_drop_frac,
        be_input_max_percentile=args.be_input_max_percentile,
        be_output_keep_frac=args.be_output_keep_frac,
        be_output_max_len=args.be_output_max_len,
    )
    if args.verbose:
        print_summary(payload, print_all=True)
    print(json.dumps(payload if args.print_all else {
        "best": payload["best"],
        "rt_infeasible": payload["rt_infeasible"],
        "rt_violation_rate": payload["rt_violation_rate"],
    }, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Output] wrote {args.output}")


if __name__ == "__main__":
    main()
