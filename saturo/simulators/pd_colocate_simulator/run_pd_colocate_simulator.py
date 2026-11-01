"""CLI entry point for the PD-aggregated RT-BE colocation simulator.

Given a target RT RPS, sweeps candidate chunk sizes (max_num_batched_tokens)
and reports, for each: RT feasibility, per-iteration latency, RT decode/prefill
occupancy, the KV blocks left for BE, and BE goodput = min(BE-Prefill,
BE-Decode). Picks the chunk size that maximizes BE goodput while keeping RT
feasible.

Example:
    python -m saturo.simulators.pd_colocate_simulator.run_pd_colocate_simulator \\
        --runtime-profile <tp1 profile>.json \\
        --rt-workload-csv data/datasets/rt_workload.csv \\
        --be-workload-csv data/datasets/be_workload.csv \\
        --num-req 5000 \\
        --target-rps 1,2,4,6 \\
        --min-chunk-size 64 --max-chunk-size 8192 --chunk-step 64
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from saturo.simulators.pd_colocate_simulator.config import (  # noqa: E402
    ModelConfig, PDColocateConfig, SolverConfig, WorkloadStats,
)
from saturo.simulators.pd_colocate_simulator.latency import (  # noqa: E402
    LatencyPredictor,
)
from saturo.simulators.pd_colocate_simulator.simulator import (  # noqa: E402
    PDColocateSimulator,
)
from saturo.simulators.pd_colocate_simulator.workload import (  # noqa: E402
    load_workload_stats,
)


def _int_csv(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _float_csv(value: str):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Saturo PD-aggregated RT-BE colocation throughput sim")
    p.add_argument("--runtime-profile", required=True,
                   help="Runtime profile JSON (time_predictor + kv_cache_block)")
    p.add_argument("--rt-workload-csv", required=True,
                   help="RT workload CSV.")
    p.add_argument("--be-workload-csv", default=None,
                   help="BE workload CSV; used to compute mean prompt/output.")
    p.add_argument("--num-req", type=_positive_int, required=True,
                   help="Exact chronological RT prefix size.")
    p.add_argument("--target-rps", type=_float_csv, required=True,
                   help="Target RT RPS; accepts a comma-separated sweep.")
    p.add_argument("--chunk-sizes", type=_int_csv, default=None,
                   help="Comma-separated candidate chunk sizes "
                        "(max_num_batched_tokens). Overrides range options.")
    p.add_argument("--min-chunk-size", type=_positive_int, default=64,
                   help="Smallest token budget in the automatic sweep.")
    p.add_argument("--max-chunk-size", type=_positive_int, default=8192,
                   help="Largest token budget in the automatic sweep.")
    p.add_argument("--chunk-step", type=_positive_int, default=64,
                   help="Automatic sweep granularity in tokens (default: 64).")
    p.add_argument("--memory-watermark", type=float, default=0.9)
    balance = p.add_mutually_exclusive_group()
    balance.add_argument(
        "--balance-be-throughput",
        dest="balance_be_throughput",
        action="store_true",
        default=True,
        help="Binary-search the BE-Decode batch that maximizes "
             "min(BE-P, BE-D) request throughput (default).")
    balance.add_argument(
        "--no-balance-be-throughput",
        dest="balance_be_throughput",
        action="store_false",
        help="Restore the legacy decode-first greedy BE split.")
    p.add_argument(
        "--be-balance-search-epsilon",
        type=float,
        default=1e-3,
        help="BE-Decode batch-size binary-search tolerance.")
    p.add_argument("--be-prompt-len", type=float, default=8000.0)
    p.add_argument("--be-output-len", type=float, default=256.0)
    p.add_argument("--output", default=None, help="Output JSON path.")
    args = p.parse_args(argv)
    if args.chunk_sizes is not None:
        if not args.chunk_sizes or any(c <= 0 for c in args.chunk_sizes):
            p.error("--chunk-sizes must contain positive integers")
    elif args.min_chunk_size > args.max_chunk_size:
        p.error("--min-chunk-size must be <= --max-chunk-size")
    if not args.target_rps or any(rps <= 0 for rps in args.target_rps):
        p.error("--target-rps must contain positive values")
    if not 0 < args.memory_watermark <= 1:
        p.error("--memory-watermark must be in (0, 1]")
    if args.be_balance_search_epsilon <= 0:
        p.error("--be-balance-search-epsilon must be positive")
    return args


def build_config(args) -> PDColocateConfig:
    latency = LatencyPredictor(args.runtime_profile)
    total_blocks = latency.total_gpu_blocks
    block_size = latency.kv_block_size or 16
    if total_blocks is None:
        raise ValueError(
            "runtime profile lacks kv_cache_block.total_gpu_blocks; "
            "re-profile or pass a profile with KV capacity.")

    base_stats = WorkloadStats(
        be_prompt_len=args.be_prompt_len,
        be_output_len=args.be_output_len,
    )
    stats = load_workload_stats(
        args.rt_workload_csv,
        args.be_workload_csv,
        base_stats,
        num_req=args.num_req,
        target_rps=args.target_rps[0],
    )

    config = PDColocateConfig(
        runtime_profile=args.runtime_profile,
        model=ModelConfig(
            total_gpu_blocks=int(total_blocks),
            block_size=int(block_size),
            memory_watermark=args.memory_watermark,
        ),
        workload=stats,
        solver=SolverConfig(
            balance_be_throughput=args.balance_be_throughput,
            be_balance_search_epsilon=args.be_balance_search_epsilon,
        ),
        rt_workload_csv=args.rt_workload_csv,
        be_workload_csv=args.be_workload_csv,
    )
    if args.chunk_sizes is not None:
        config.chunk_sizes = sorted(set(args.chunk_sizes))
    else:
        config.chunk_sizes = list(range(
            args.min_chunk_size,
            args.max_chunk_size + 1,
            args.chunk_step,
        ))
    return config, latency


def print_sweep(rps, result, balance_be_throughput=True):
    width = 132
    print(f"\n{'=' * width}")
    policy = "balanced-binary-search" if balance_be_throughput else "greedy"
    print(f"  RPS = {rps}   (BE policy={policy}; "
          "goodput=min(BE-P, BE-D))")
    print("=" * width)
    header = (f"{'chunk':>6} {'feasible':>9} {'t_iter':>8} "
              f"{'RTd_bs':>7} {'RTd_blk':>8} "
              f"{'BEd_bs':>7} {'BEd_cap':>7} {'BEp_tok':>8} {'BE_blk':>8} "
              f"{'BE-P r/s':>9} {'BE-D r/s':>9} {'BE good':>9}  reason")
    print(header)
    print("-" * width)
    for pt in result.all_points:
        mark = "*" if (result.best is not None
                       and pt is result.best) else " "
        print(f"{pt.chunk_size:>6} {str(pt.feasible):>9} "
              f"{pt.t_iter_ms:>8.1f} {pt.rt_decode_batch:>7.1f} "
              f"{pt.rt_decode_blocks:>8.0f} {pt.be_decode_batch:>7.1f} "
              f"{pt.be_decode_capacity_batch:>7.1f} "
              f"{pt.be_prefill_tokens:>8.1f} "
              f"{pt.be_decode_blocks:>8.0f} {pt.be_prefill_req_per_s:>9.3f} "
              f"{pt.be_decode_req_per_s:>9.3f} {pt.be_goodput_req_per_s:>8.3f}{mark} "
              f"{pt.infeasible_reason}")
    if result.best is not None:
        b = result.best
        print("-" * width)
        print(f"  BEST chunk_size={b.chunk_size}: BE goodput="
              f"{b.be_goodput_req_per_s:.3f} req/s "
              f"(BE-P={b.be_prefill_req_per_s:.3f}, "
              f"BE-D={b.be_decode_req_per_s:.3f}), "
              f"D={b.be_decode_batch:.1f}/{b.be_decode_capacity_batch:.1f}, "
              f"P={b.be_prefill_tokens:.1f} tok, "
              f"t_iter={b.t_iter_ms:.1f}ms")
    else:
        print("  No feasible chunk size (RT over capacity at this RPS).")


def main(argv=None):
    args = parse_args(argv)
    config, latency = build_config(args)
    sim = PDColocateSimulator(config, latency)

    w = config.workload
    print(f"[pd_colocate] profile={os.path.basename(config.runtime_profile)}")
    print(f"[pd_colocate] KV: total_blocks={config.model.total_gpu_blocks} "
          f"block_size={config.model.block_size} "
          f"watermark={config.model.memory_watermark} "
          f"=> budget={config.model.kv_block_budget} blocks")
    print(f"[pd_colocate] RT avg prompt/output = "
          f"{w.rt_prompt_len:.0f}/{w.rt_output_len:.0f} tok; "
          f"BE avg prompt/output = {w.be_prompt_len:.0f}/{w.be_output_len:.0f} tok")

    sweep = {}
    for rps in args.target_rps:
        result = sim.optimize(rps)
        print_sweep(
            rps, result, config.solver.balance_be_throughput)
        sweep[str(rps)] = {
            "best_chunk_size": (result.best.chunk_size
                                if result.best else None),
            "best_be_goodput_req_per_s": (
                result.best.be_goodput_req_per_s if result.best else 0.0),
            "points": [pt.to_dict() for pt in result.all_points],
        }

    output = {
        "config": {
            "runtime_profile": config.runtime_profile,
            "total_gpu_blocks": config.model.total_gpu_blocks,
            "block_size": config.model.block_size,
            "memory_watermark": config.model.memory_watermark,
            "kv_block_budget": config.model.kv_block_budget,
            "rt_prompt_len": w.rt_prompt_len,
            "rt_output_len": w.rt_output_len,
            "be_prompt_len": w.be_prompt_len,
            "be_output_len": w.be_output_len,
            "chunk_sizes": config.chunk_sizes,
            "chunk_granularity": (
                args.chunk_step if args.chunk_sizes is None else None),
            "num_req": args.num_req,
            "target_rps": args.target_rps,
            "balance_be_throughput": (
                config.solver.balance_be_throughput),
            "be_balance_search_epsilon": (
                config.solver.be_balance_search_epsilon),
        },
        "sweep": sweep,
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\n[pd_colocate] results saved to {args.output}")
    return output


if __name__ == "__main__":
    main()
