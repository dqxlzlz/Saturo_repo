"""Main entry point for the decode simulator.

Runs three saturation strategies and compares results:
1. Progressive (Saturo): look-ahead memory-based saturation
2. Oracle: perfect knowledge of remain_len and future arrivals
3. Greedy: always saturate to TBT_SLO, ignoring future uncertainty

Supports:
  - Single RPS run (--target-rps)
  - RPS sweep (--rps-sweep "1,2,5,10,20,50")
"""

import argparse
from dataclasses import asdict
import json
import logging
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from saturo.simulators.decode_simulator.config import (
    ModelConfig, SLOConfig, SaturationConfig, SimulatorConfig,
)
from saturo.simulators.decode_simulator.trace_loader import load_rt_trace
from saturo.simulators.decode_simulator.request import Request


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _probability(value: str) -> float:
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("must be in the range (0, 1]")
    return number


def _strategy_csv(value: str) -> str:
    valid = {"progressive", "oracle", "greedy"}
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(names) - valid)
    if not names:
        raise argparse.ArgumentTypeError("must contain at least one strategy")
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown strategies: {', '.join(invalid)}; "
            f"choose from {', '.join(sorted(valid))}")
    return ",".join(dict.fromkeys(names))


def _parse_rps_sweep(value):
    if value is None:
        return None
    try:
        values = [float(item.strip()) for item in value.split(",")
                  if item.strip()]
    except ValueError as exc:
        raise ValueError("--rps-sweep must contain only numbers") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError("--rps-sweep must contain positive RPS values")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Saturo D-pool decode saturation simulator",
    )

    inputs = parser.add_argument_group("required inputs")
    inputs.add_argument(
        "--rt-workload-csv", "--rt-trace",
        dest="rt_trace",
        required=True,
        help="RT workload CSV, Azure trace CSV, or WildChat parquet file.",
    )
    inputs.add_argument(
        "--runtime-profile", "--latency-model",
        dest="runtime_profile",
        required=True,
        help="Runtime profile JSON containing time_predictor and, when "
             "available, kv_cache_block. --latency-model is a legacy alias.",
    )
    inputs.add_argument(
        "--be-workload-csv", required=True,
        help="BE workload CSV with prompt_len and output_len columns.",
    )

    traffic = parser.add_argument_group("traffic and SLO")
    traffic.add_argument(
        "--duration-s", "--duration",
        dest="duration_s", type=_positive_float, default=300.0,
        help="Simulation duration in seconds (default: 300).",
    )
    traffic.add_argument(
        "--warmup-s", "--warmup",
        dest="warmup_s", type=_nonnegative_float, default=10.0,
        help="Warmup excluded from TBT samples, in seconds (default: 10).",
    )
    arrival = traffic.add_mutually_exclusive_group()
    arrival.add_argument(
        "--rt-rps", "--target-rps",
        dest="rt_rps", type=_positive_float, default=None,
        help="Target RT RPS; scales trace inter-arrival times.",
    )
    arrival.add_argument(
        "--time-scale", type=_positive_float, default=None,
        help="Scale trace inter-arrival times directly.",
    )
    arrival.add_argument(
        "--rps-sweep", default=None,
        help="Comma-separated positive RT RPS values.",
    )
    traffic.add_argument(
        "--rt-tbt-slo-ms", "--tbt-slo",
        dest="rt_tbt_slo_ms", type=_positive_float, default=100.0,
        help="RT time-between-token SLO in milliseconds (default: 100).",
    )

    trace_filter = parser.add_argument_group("optional trace filtering")
    trace_filter.add_argument(
        "--rt-max-prompt-len", "--rt-prompt-max",
        dest="rt_max_prompt_len", type=_positive_int, default=None,
        help="Clamp RT prompt lengths after loading the trace.",
    )
    trace_filter.add_argument(
        "--rt-max-output-len", "--rt-output-max",
        dest="rt_max_output_len", type=_positive_int, default=None,
        help="Clamp RT output lengths after loading the trace.",
    )

    policy = parser.add_argument_group("decode saturation policy")
    policy.add_argument(
        "--lookahead-window-s", "--lookahead-window",
        dest="lookahead_window_s", type=_positive_float, default=5.0,
        help="Progressive look-ahead window in seconds (default: 5).",
    )
    policy.add_argument(
        "--model-len-quantile", "--length-quantile",
        dest="model_len_quantile",
        type=_probability,
        default=0.95,
        help="Quantile for Progressive active RT remaining-length prediction.",
    )
    policy.add_argument(
        "--model-len-predictor", "--length-predictor",
        dest="model_len_predictor",
        default=None,
        help="QRF/GBDT remaining-length predictor artifact.",
    )
    policy.add_argument(
        "--model-len-predictor-type", "--length-predictor-type",
        dest="model_len_predictor_type",
        choices=("qrf", "gbdt", "gbpt", "gbrt"),
        default=None,
        help="Optional expected type for --model-len-predictor.",
    )
    policy.add_argument(
        "--min-interval-ms", "--min-interval",
        dest="min_interval_ms", type=_positive_float, default=5.0,
        help="Smallest candidate inter-token interval in ms (default: 5).",
    )
    policy.add_argument(
        "--memory-watermark", type=_probability, default=0.9,
        help="Fraction of profile KV-cache blocks usable for RT and BE.",
    )
    policy.add_argument(
        "--chunk-sizes", default="auto",
        help="Comma-separated BE prefill chunks, or auto (default: auto).",
    )
    policy.add_argument(
        "--balance-be-throughput",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance BE-Prefill and BE-Decode request throughput within each "
             "target-TBT budget (default: enabled).",
    )
    policy.add_argument(
        "--strategies", type=_strategy_csv,
        default="progressive,oracle,greedy",
        help="Comma-separated: progressive, oracle, greedy.",
    )

    # Keep accepting this old no-op so existing scripts do not fail. The
    # runtime profile is now the source of parallelism and KV-cache capacity.
    parser.add_argument(
        "--tensor-parallel", type=_positive_int, default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", default=None, help="Output JSON path.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    if args.warmup_s >= args.duration_s:
        parser.error("--warmup-s must be smaller than --duration-s")
    try:
        _parse_rps_sweep(args.rps_sweep)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def deep_copy_requests(requests):
    """Deep copy request list so each strategy run starts fresh."""
    copied = []
    for r in requests:
        copied.append(Request(
            request_id=r.request_id,
            request_type=r.request_type,
            prompt_len=r.prompt_len,
            target_output_len=r.target_output_len,
            arrival_time_s=r.arrival_time_s,
        ))
    return copied


def print_result_summary(result):
    """Print a formatted summary for one strategy result."""
    print("\n" + "=" * 60)
    print(f"  Strategy: {result.strategy.upper()}")
    print("=" * 60)
    if result.rps_infeasible:
        print("  *** RPS INFEASIBLE FOR THIS CONFIG: RT KV demand exceeds "
              "usable capacity even with all swappable BE reclaimed "
              f"(terminated at {result.termination_reason}) ***")
    print(f"  Total iterations:           {result.total_iterations}")
    print(f"  --- RT Metrics ---")
    print(f"    Total RT requests:        {result.total_rt_requests}")
    print(f"    Completed RT requests:    {result.completed_rt_requests}")
    print(f"    TBT P50:                  {result.rt_tbt_p50_ms:.2f} ms")
    print(f"    TBT P99:                  {result.rt_tbt_p99_ms:.2f} ms")
    print(f"    TBT Max:                  {result.rt_tbt_max_ms:.2f} ms")
    print(f"    TBT violations:           "
          f"{result.rt_tbt_violations}/{result.rt_tbt_total_samples}")
    print(f"    TBT violation rate:       "
          f"{result.rt_tbt_violation_rate*100:.2f}%")
    print(f"    Admission violations:     {result.rt_admission_violations}")
    print(f"    RT memory overflow iters: {result.rt_memory_overflow_events}")
    print(f"    Total RT SLO violations:  {result.rt_slo_violations_total}")
    print(f"    Avg RT batch size:        {result.avg_rt_batch_size:.1f}")
    print(f"    Avg RT queue size:        {result.avg_rt_queue_size:.1f}")
    print(f"    Peak RT blocks:           {result.peak_rt_blocks}")
    print(f"  --- BE Metrics ---")
    print(f"    BE completed:             {result.total_be_completed}")
    print(f"    BE prefill tokens:        {result.total_be_prefill_tokens}")
    print(f"    BE decode tokens:         {result.total_be_decode_tokens}")
    print(f"    BE saturation throughput: "
          f"{result.be_saturation_req_per_s:.6f} req/s")
    print(f"    BE prefill req tput:      "
          f"{result.be_prefill_req_per_s:.6f} req/s")
    print(f"    BE decode req tput:       "
          f"{result.be_decode_req_per_s:.6f} req/s")
    print(f"    Raw BE token throughput:  "
          f"{result.be_throughput_tokens_per_s:.1f} tok/s")
    print(f"    BE prefill token tput:    "
          f"{result.be_prefill_throughput_tokens_per_s:.1f} tok/s")
    print(f"    BE decode token tput:     "
          f"{result.be_decode_throughput_tokens_per_s:.1f} tok/s")
    print(f"    Avg prefill chunk:        {result.avg_be_prefill_chunk:.1f}")
    print(f"    Avg BE decode batch:      {result.avg_be_batch_size:.1f}")
    print(f"  --- Saturation ---")
    print(f"    Avg saturation ratio:     {result.avg_saturation_ratio:.3f}")
    print(f"    Avg iteration latency:    "
          f"{result.avg_iteration_latency_ms:.2f} ms")
    print(f"    Avg target TBT:           {result.avg_target_tbt_ms:.2f} ms")
    print(f"    P50/P99 target TBT:       "
          f"{result.p50_target_tbt_ms:.2f}/"
          f"{result.p99_target_tbt_ms:.2f} ms")
    print(f"    Avg safe TBT:             {result.avg_safe_tbt_ms:.2f} ms")


def result_to_dict(result):
    """Convert SimulationResult to serializable dict."""
    iteration_stats = [asdict(item) for item in result.iteration_stats]
    return {
        "strategy": result.strategy,
        "total_rt_requests": result.total_rt_requests,
        "completed_rt_requests": result.completed_rt_requests,
        "total_be_completed": result.total_be_completed,
        "total_be_prefill_tokens": result.total_be_prefill_tokens,
        "total_be_decode_tokens": result.total_be_decode_tokens,
        "total_iterations": result.total_iterations,
        "rt_tbt_p50_ms": round(result.rt_tbt_p50_ms, 2),
        "rt_tbt_p99_ms": round(result.rt_tbt_p99_ms, 2),
        "rt_tbt_max_ms": round(result.rt_tbt_max_ms, 2),
        "rt_tbt_violations": result.rt_tbt_violations,
        "rt_tbt_total_samples": result.rt_tbt_total_samples,
        "rt_tbt_violation_rate": round(result.rt_tbt_violation_rate, 4),
        "rt_admission_violations": result.rt_admission_violations,
        "rt_memory_overflow_events": result.rt_memory_overflow_events,
        "rt_slo_violations_total": result.rt_slo_violations_total,
        "infeasible_iterations": result.infeasible_iterations,
        "be_swap_out_events": result.be_swap_out_events,
        "be_swap_in_events": result.be_swap_in_events,
        "avg_be_prefill_chunk": round(result.avg_be_prefill_chunk, 1),
        "avg_rt_batch_size": round(result.avg_rt_batch_size, 1),
        "avg_rt_queue_size": round(result.avg_rt_queue_size, 1),
        "avg_be_batch_size": round(result.avg_be_batch_size, 1),
        "avg_iteration_latency_ms": round(result.avg_iteration_latency_ms, 2),
        "avg_target_tbt_ms": round(result.avg_target_tbt_ms, 2),
        "p50_target_tbt_ms": round(result.p50_target_tbt_ms, 2),
        "p99_target_tbt_ms": round(result.p99_target_tbt_ms, 2),
        "avg_safe_tbt_ms": round(result.avg_safe_tbt_ms, 2),
        "peak_rt_blocks": result.peak_rt_blocks,
        "avg_rt_blocks": round(result.avg_rt_blocks, 1),
        "avg_be_blocks": round(result.avg_be_blocks, 1),
        "be_saturation_req_per_s": round(result.be_saturation_req_per_s, 6),
        "be_throughput_req_per_s": round(result.be_saturation_req_per_s, 6),
        "be_prefill_req_per_s": round(result.be_prefill_req_per_s, 6),
        "be_decode_req_per_s": round(result.be_decode_req_per_s, 6),
        "be_raw_throughput_tok_per_s": round(
            result.be_throughput_tokens_per_s, 1),
        "be_throughput_tok_per_s": round(result.be_throughput_tokens_per_s, 1),
        "be_prefill_throughput_tok_per_s": round(
            result.be_prefill_throughput_tokens_per_s, 1),
        "be_decode_throughput_tok_per_s": round(
            result.be_decode_throughput_tokens_per_s, 1),
        "avg_saturation_ratio": round(result.avg_saturation_ratio, 4),
        "wall_run_s": round(result.wall_run_s, 6),
        "wall_enqueue_s": round(result.wall_enqueue_s, 6),
        "wall_admit_s": round(result.wall_admit_s, 6),
        "wall_decision_s": round(result.wall_decision_s, 6),
        "wall_iteration_s": round(result.wall_iteration_s, 6),
        "terminated_early": result.terminated_early,
        "termination_reason": result.termination_reason,
        "rps_infeasible": result.rps_infeasible,
        "iteration_stats": iteration_stats,
    }


def print_comparison_table(all_results, label=""):
    """Print side-by-side comparison of strategy results."""
    if len(all_results) <= 1:
        return
    print(f"\n{'=' * 80}")
    if label:
        print(f"  STRATEGY COMPARISON — {label}")
    else:
        print(f"  STRATEGY COMPARISON")
    print("=" * 80)
    header = "Metric".ljust(30)
    for name in all_results:
        header += f"  {name:>14}"
    print(header)
    print("-" * (30 + 16 * len(all_results)))

    metrics = [
        ("TBT P50 (ms)", "rt_tbt_p50_ms"),
        ("TBT P99 (ms)", "rt_tbt_p99_ms"),
        ("TBT Max (ms)", "rt_tbt_max_ms"),
        ("TBT Violations", "rt_tbt_violations"),
        ("Admission Violations", "rt_admission_violations"),
        ("RT Overflow Iters", "rt_memory_overflow_events"),
        ("Total RT SLO Viol.", "rt_slo_violations_total"),
        ("BE Completed", "total_be_completed"),
        ("BE Saturation (req/s)", "be_saturation_req_per_s"),
        ("Avg Saturation Ratio", "avg_saturation_ratio"),
        ("Avg Iter Latency (ms)", "avg_iteration_latency_ms"),
        ("Avg Target TBT (ms)", "avg_target_tbt_ms"),
        ("Peak RT Blocks", "peak_rt_blocks"),
    ]
    for label_str, attr in metrics:
        row = f"{label_str:<30}"
        for name, res in all_results.items():
            if attr is not None:
                val = getattr(res, attr)
                if isinstance(val, float):
                    row += f"  {val:>14.2f}"
                else:
                    row += f"  {val:>14}"
            else:
                row += f"  {res.rt_tbt_violation_rate*100:>13.2f}%"
        print(row)
    print("=" * 80)


def run_single(config, strategies, rt_requests, be_requests, label="",
               print_summaries: bool = True):
    """Run all strategies for a single configuration."""
    from saturo.simulators.decode_simulator.simulator import DecodeSimulator

    logger = logging.getLogger("simulators.decode_simulator")
    all_results = {}

    for strategy in strategies:
        logger.info("=" * 60)
        logger.info("Running %s: %s", strategy.value, label)
        logger.info("=" * 60)

        rt_copy = deep_copy_requests(rt_requests)
        be_copy = deep_copy_requests(be_requests)

        sim = DecodeSimulator(config, strategy=strategy)
        t0 = time.time()
        result = sim.run(rt_copy, be_copy)
        elapsed = time.time() - t0

        logger.info("Completed in %.1fs", elapsed)
        if print_summaries:
            print_result_summary(result)
        all_results[strategy.value] = result

    if print_summaries:
        print_comparison_table(all_results, label=label)
    return all_results


def _arg(args, canonical, legacy=None, default=None):
    if hasattr(args, canonical):
        return getattr(args, canonical)
    if legacy and hasattr(args, legacy):
        return getattr(args, legacy)
    return default


def _load_runtime_profile_metadata(profile_path):
    if not profile_path:
        raise ValueError("--runtime-profile is required")
    if not os.path.isfile(profile_path):
        raise FileNotFoundError(
            f"runtime profile does not exist: {profile_path}")

    try:
        with open(profile_path, "r", encoding="utf-8") as profile_file:
            payload = json.load(profile_file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"runtime profile is not valid JSON: {profile_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"runtime profile must contain a JSON object: {profile_path}")
    time_predictor = payload.get("time_predictor")
    if "time_predictor" in payload and not isinstance(time_predictor, dict):
        raise ValueError(
            f"runtime profile has an invalid time_predictor section: "
            f"{profile_path}")

    kv = payload.get("kv_cache_block")
    if kv is not None and not isinstance(kv, dict):
        raise ValueError(
            f"runtime profile has an invalid kv_cache_block section: "
            f"{profile_path}")
    kv = kv or {}

    total_gpu_blocks = kv.get("total_gpu_blocks")
    block_size = kv.get("block_size")
    if total_gpu_blocks is not None and int(total_gpu_blocks) <= 0:
        raise ValueError("kv_cache_block.total_gpu_blocks must be positive")
    if block_size is not None and int(block_size) <= 0:
        raise ValueError("kv_cache_block.block_size must be positive")
    return (
        int(total_gpu_blocks) if total_gpu_blocks is not None else None,
        int(block_size) if block_size is not None else None,
    )


def build_decode_config(args):
    chunk_sizes_value = _arg(args, "chunk_sizes", default="auto")
    if chunk_sizes_value.strip().lower() == "auto":
        chunk_sizes = list(range(16, 4097, 16))
    else:
        try:
            chunk_sizes = [
                int(item.strip()) for item in chunk_sizes_value.split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "--chunk-sizes must be 'auto' or comma-separated integers"
            ) from exc
        if not chunk_sizes or any(item <= 0 for item in chunk_sizes):
            raise ValueError("--chunk-sizes values must be positive")

    runtime_profile = _arg(args, "runtime_profile", "latency_model")
    profile_blocks, profile_block_size = _load_runtime_profile_metadata(
        runtime_profile)
    explicit_blocks = _arg(args, "total_gpu_blocks")
    explicit_block_size = _arg(args, "block_size")
    total_gpu_blocks = (
        int(explicit_blocks) if explicit_blocks is not None
        else profile_blocks
    )
    block_size = (
        int(explicit_block_size) if explicit_block_size is not None
        else profile_block_size
    )

    logger = logging.getLogger("simulators.decode_simulator")
    if total_gpu_blocks is None:
        logger.warning(
            "Runtime profile has no kv_cache_block.total_gpu_blocks; "
            "using the simulator fallback. Re-profile for accurate capacity.")
    if block_size is None:
        block_size = 16
        logger.warning(
            "Runtime profile has no kv_cache_block.block_size; using 16.")

    return SimulatorConfig(
        model=ModelConfig(block_size=block_size),
        slo=SLOConfig(
            tbt_slo_ms=_arg(args, "rt_tbt_slo_ms", "tbt_slo", 100.0)),
        saturation=SaturationConfig(
            lookahead_window_s=_arg(
                args, "lookahead_window_s", "lookahead_window", 5.0),
            model_len_quantile=_arg(args, "model_len_quantile", default=0.95),
            min_interval_ms=_arg(
                args, "min_interval_ms", "min_interval", 5.0),
        ),
        rt_trace_path=args.rt_trace,
        total_gpu_blocks=total_gpu_blocks,
        memory_watermark=_arg(args, "memory_watermark", default=0.9),
        be_prefill_reserve_blocks=int(
            _arg(args, "be_prefill_reserve_blocks", default=0) or 0),
        be_workload_csv=_arg(args, "be_workload_csv", default=""),
        balance_be_throughput=bool(
            _arg(args, "balance_be_throughput", default=True)),
        be_max_prompt_len=_arg(
            args, "rt_max_prompt_len", "rt_prompt_max", None),
        be_chunk_sizes=chunk_sizes,
        simulation_duration_s=_arg(args, "duration_s", "duration", 300.0),
        warmup_s=_arg(args, "warmup_s", "warmup", 10.0),
        latency_model_path=runtime_profile,
        model_len_predictor_path=_arg(
            args, "model_len_predictor", default=None),
        model_len_predictor_type=_arg(
            args, "model_len_predictor_type", default=None),
    ), chunk_sizes


def parse_decode_strategies(strategy_csv: str):
    from saturo.simulators.decode_simulator.simulator import DecodeStrategy

    strategy_map = {
        "progressive": DecodeStrategy.PROGRESSIVE,
        "oracle": DecodeStrategy.ORACLE,
        "greedy": DecodeStrategy.GREEDY,
    }
    names = [item.strip().lower() for item in strategy_csv.split(",")
             if item.strip()]
    invalid = sorted(set(names) - set(strategy_map))
    if not names or invalid:
        detail = f": {', '.join(invalid)}" if invalid else ""
        raise ValueError(f"invalid --strategies{detail}")
    return [strategy_map[name] for name in dict.fromkeys(names)]


def run_decode_profile(args, *, print_summaries: bool = True):
    logger = logging.getLogger("simulators.decode_simulator")
    duration_s = float(_arg(args, "duration_s", "duration", 300.0))
    warmup_s = float(_arg(args, "warmup_s", "warmup", 10.0))
    rt_rps = _arg(args, "rt_rps", "target_rps")
    time_scale = _arg(args, "time_scale")
    rps_sweep = _arg(args, "rps_sweep")
    rt_max_prompt_len = _arg(
        args, "rt_max_prompt_len", "rt_prompt_max")
    rt_max_output_len = _arg(
        args, "rt_max_output_len", "rt_output_max")
    if duration_s <= 0:
        raise ValueError("duration must be positive")
    if warmup_s < 0 or warmup_s >= duration_s:
        raise ValueError("warmup must be nonnegative and smaller than duration")
    if rt_rps is not None and float(rt_rps) <= 0:
        raise ValueError("RT RPS must be positive")
    if time_scale is not None and float(time_scale) <= 0:
        raise ValueError("time scale must be positive")
    from saturo.history import is_redis_history
    if not is_redis_history(args.rt_trace) and not os.path.isfile(args.rt_trace):
        raise FileNotFoundError(f"RT trace does not exist: {args.rt_trace}")
    be_workload_csv = _arg(args, "be_workload_csv", default="")
    if be_workload_csv and not os.path.isfile(be_workload_csv):
        raise FileNotFoundError(
            f"BE workload does not exist: {be_workload_csv}")

    base_config, chunk_sizes = build_decode_config(args)
    strategies = parse_decode_strategies(
        _arg(args, "strategies", default="progressive,oracle,greedy"))

    # The simulator loads BE templates from the CSV and cycles them to keep
    # the GPU saturated, so no fixed-size BE request list is passed here.
    be_requests_orig = []

    if rps_sweep:
        rps_values = _parse_rps_sweep(rps_sweep)
    elif rt_rps is not None:
        rps_values = [float(rt_rps)]
    else:
        rps_values = [None]

    sweep_results = {}
    for target_rps in rps_values:
        if target_rps is not None:
            label = f"RPS={target_rps}"
            logger.info("Loading RT trace (target RPS=%.2f) ...", target_rps)
        else:
            label = "RPS=original"
            logger.info("Loading RT trace (original RPS) ...")

        rt_requests = load_rt_trace(
            args.rt_trace,
            duration_s=duration_s,
            target_rps=target_rps,
            time_scale=time_scale if target_rps is None else None,
            max_prompt_len=rt_max_prompt_len,
            max_output_len=rt_max_output_len,
        )
        logger.info("Loaded %d RT requests", len(rt_requests))

        results = run_single(
            base_config,
            strategies,
            rt_requests,
            be_requests_orig,
            label=label,
            print_summaries=print_summaries,
        )
        sweep_results[label] = {
            name: result_to_dict(result) for name, result in results.items()
        }

    if print_summaries and len(sweep_results) > 1:
        print(f"\n{'=' * 100}")
        print("  RPS SWEEP SUMMARY")
        print("=" * 100)
        header = f"{'RPS':<12} {'Strategy':<14}"
        header += f"{'BE_req/s':>10}{'Sat%':>8}{'SLO':>8}"
        header += f"{'Adm':>8}{'RTmem':>8}{'TBT':>9}{'RTblk':>8}{'Feasible':>10}"
        print(header)
        print("-" * 100)
        for rps_label, strategy_results in sweep_results.items():
            for strategy_name, result in strategy_results.items():
                row = f"{rps_label:<12} {strategy_name:<14}"
                row += f"{result['be_saturation_req_per_s']:>10.3f}"
                row += f"{result['avg_saturation_ratio']*100:>7.1f}%"
                row += f"{result['rt_slo_violations_total']:>8}"
                row += f"{result['rt_admission_violations']:>8}"
                row += f"{result['rt_memory_overflow_events']:>8}"
                row += f"{result['avg_target_tbt_ms']:>9.1f}"
                row += f"{result['peak_rt_blocks']:>8}"
                row += (
                    f"{'INFEASIBLE':>10}" if result.get('rps_infeasible')
                    else f"{'yes':>10}")
                print(row)
            print("-" * 100)

    runtime_profile = _arg(args, "runtime_profile", "latency_model")
    return {
        "config": {
            "runtime_profile": runtime_profile,
            "total_gpu_blocks": base_config.total_gpu_blocks,
            "block_size": base_config.model.block_size,
            "rt_tbt_slo_ms": base_config.slo.tbt_slo_ms,
            "duration_s": duration_s,
            "warmup_s": warmup_s,
            "rt_rps": rt_rps,
            "time_scale": time_scale,
            "arrival_mode": "scaled_dataset_trace",
            "memory_watermark": base_config.memory_watermark,
            "be_workload_csv": be_workload_csv or None,
            "balance_be_throughput": base_config.balance_be_throughput,
            "chunk_sizes": chunk_sizes,
            "rt_max_prompt_len": rt_max_prompt_len,
            "rt_max_output_len": rt_max_output_len,
            "lookahead_window_s": (
                base_config.saturation.lookahead_window_s),
            "model_len_quantile": (
                base_config.saturation.model_len_quantile),
            "model_len_predictor": _arg(
                args, "model_len_predictor", default=None),
            "model_len_predictor_type": _arg(
                args, "model_len_predictor_type", default=None),
            "min_interval_ms": base_config.saturation.min_interval_ms,
            "strategies": [strategy.value for strategy in strategies],
        },
        "results": sweep_results,
    }


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("simulators.decode_simulator")

    try:
        output_data = run_decode_profile(args, print_summaries=True)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"decode simulator: error: {exc}") from exc

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "decode_simulation_result.json",
    )
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(output_data, output_file, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
