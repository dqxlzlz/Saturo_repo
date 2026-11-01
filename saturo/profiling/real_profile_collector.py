"""Real-engine profile collection for Saturo resource allocation."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from saturo.history import (configured_history_source, is_redis_history,
                            load_trace_rows, read_redis_history)

from saturo.predictors.qrf import resolve_qrf_path
from saturo.predictors.time_predictor import (
    TimePredictorProfileIdentity,
    time_predictor_cache_path,
)


@dataclass(frozen=True)
class ParallelConfig:
    name: str
    gpus: int
    tp: int
    pp: int


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    arrival_time_s: float
    prompt_len: int
    output_len: int


def collect_real_allocation_profile(args: argparse.Namespace,
                                    profile_path: Path,
                                    config_names: list[str]) -> None:
    """Collect config-specific history profiles and write allocation profile.

    Timing profiles are collected by a local engine before serving workers
    start. CUDA_VISIBLE_DEVICES must select available local GPUs. Historical
    request timings and pool capacities are then modeled by the simulators.
    """

    source = configured_history_source(
        getattr(args, "rt_workload_csv", "") or "")
    if not source:
        raise ValueError(
            "missing --rt-workload-csv; needed as the history trace for profiling")
    trace_path = source if is_redis_history(source) else Path(source)
    if not is_redis_history(source) and not trace_path.exists():
        raise FileNotFoundError(f"RT workload CSV not found: {trace_path}")

    profile_root = profile_path.parent
    if is_redis_history(source):
        # The experiment's request count controls live replay, not the size
        # of the independent 30-minute historical modeling window.
        args = argparse.Namespace(**vars(args))
        snapshot = getattr(args, "_history_snapshot", None)
        if snapshot is None:
            snapshot = read_redis_history(
                source, window_s=float(os.environ.get(
                    "SATURO_HISTORY_WINDOW_S", "1800")), model=args.model)
            args._history_snapshot = snapshot
        args._history_rows = snapshot.workload_rows()
        if not args._history_rows:
            raise ValueError("Redis history has no completed requests; "
                             "bootstrap with an existing allocation profile "
                             "or collect RT traces first")
        args.num_req = len(args._history_rows)
        args._history_metadata = snapshot.metadata()
        # Timing models are reusable, workload-dependent simulation results
        # must belong to the same immutable history snapshot.
        profile_root /= "history_" + args._history_metadata["fingerprint"]

    configs = [parse_parallel_config(name) for name in config_names]
    if not configs:
        raise ValueError("no parallel configs requested for profiling")

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    rps_values = _resource_rps_values(args)

    for config in configs:
        config_dir = profile_root / config.name
        config_dir.mkdir(parents=True, exist_ok=True)
        summary_path = config_dir / "real_profile_summary.json"
        history_profile = config_dir / "history_profile.csv"

        if (summary_path.exists() and history_profile.exists()
                and _history_profile_is_simulator_ready(
                    history_profile, args.num_req)
                and _summary_has_runtime_profile(summary_path)):
            summaries[config.name] = json.loads(summary_path.read_text())
        else:
            summary = run_profile_engine(
                args=args,
                config=config,
                trace_path=trace_path,
                history_profile=history_profile,
            )
            summary_path.write_text(json.dumps(summary, indent=2))
            summaries[config.name] = summary

    profile_path.write_text(json.dumps(
        build_allocation_profile(
            args=args,
            configs=configs,
            summaries=summaries,
            profile_root=profile_root,
            rps_values=rps_values,
        ),
        indent=2,
    ))


def run_profile_engine(
    args: argparse.Namespace,
    config: ParallelConfig,
    trace_path: Path,
    history_profile: Path,
) -> dict[str, Any]:
    """Launch one vLLM engine, fit time_predictor, synthesize history profile.

    Instead of replaying every trace request through the engine (slow), we:
    1. Run a small concurrent batch to generate iteration logs.
    2. Fit the time_predictor from those logs.
    3. Use the predictor to synthesize per-request timing for all trace
       requests (fast, no GPU needed per request beyond the warm-up batch).
    4. Write the synthetic history_profile.csv consumed by the simulators.
    """
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    engine = None
    try:
        if not args.model:
            raise ValueError("missing --model for real profile collection")

        requests = load_history_trace(
            trace_path,
            num_req=args.num_req,
            target_rps=args.target_rps,
            rows=getattr(args, "_history_rows", None),
        )
        if not requests:
            raise ValueError(f"history trace is empty: {trace_path}")

        cached_predictor = _find_cached_time_predictor(args, config)
        if cached_predictor is not None:
            rows = _synthesize_rows_from_predictor(
                cached_predictor,
                requests,
            )
            write_history_profile(history_profile, rows)
            summary = summarize_profile(rows, args, config.name)
            summary["runtime_profile"] = str(cached_predictor)
            summary["runtime_profile_available"] = True
            summary["profile_source"] = "cached_time_predictor"
            return summary

        from vllm.config import CompilationConfig
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine

        engine_kwargs: dict[str, Any] = {
            "model": args.model,
            "trust_remote_code": True,
            "skip_tokenizer_init": True,
            "enforce_eager": True,
            "tensor_parallel_size": config.tp,
            "pipeline_parallel_size": config.pp,
            "load_format": args.load_format,
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "disable_custom_all_reduce": True,
            "scheduling_policy": "priority",
            "time_predictor_profile": str(
                _runtime_profile_root(args)),
        }
        for name in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "swap_space"):
            value = getattr(args, name, None)
            if value is not None:
                engine_kwargs[name] = value
        if args.disable_inductor:
            engine_kwargs["compilation_config"] = CompilationConfig(
                level=0,
                custom_ops=["none"],
                use_cudagraph=False,
                use_inductor=False,
            )

        engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))

        predictor_path = _ensure_iteration_time_predictor(
            engine=engine,
            cache_root=_runtime_profile_root(args),
        )

        # Synthesize per-request timing using the predictor (no GPU per req)
        rows = _synthesize_rows_from_predictor(
            predictor_path,
            requests,
        )

        write_history_profile(history_profile, rows)
        summary = summarize_profile(rows, args, config.name)
        summary["runtime_profile"] = str(predictor_path or "")
        summary["runtime_profile_available"] = predictor_path is not None
        return summary
    finally:
        if engine is not None:
            del engine
        _cleanup_engine_runtime()


def _find_cached_time_predictor(
    args: argparse.Namespace,
    config: ParallelConfig,
) -> Path | None:
    roots = [_runtime_profile_root(args)]
    model_text = str(getattr(args, "model", "")).lower()
    model_name = Path(model_text).name.lower()
    candidates: list[Path] = []
    needle = f"_tp{config.tp}_pp{config.pp}_"
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("time_predictor_*.json"):
            if path.name.endswith(".raw_profile.json"):
                continue
            text = str(path).lower()
            if needle not in path.name:
                continue
            if model_name and model_name not in text and model_text not in text:
                continue
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_allocation_profile(
    *,
    args: argparse.Namespace,
    configs: list[ParallelConfig],
    summaries: dict[str, dict[str, Any]],
    profile_root: Path,
    rps_values: list[float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "configs": {
            cfg.name: {"gpus": cfg.gpus, "tp": cfg.tp, "pp": cfg.pp}
            for cfg in configs
        },
        "pools": {"P": {}, "D": {}, "T": {}},
    }
    if getattr(args, "_history_metadata", None):
        payload["history"] = args._history_metadata
    for cfg in configs:
        config_dir = profile_root / cfg.name
        history_profile = config_dir / "history_profile.csv"
        summary = summaries[cfg.name]
        p_profile = _profile_p_pool(
            args=args,
            config=cfg,
            summaries=summaries,
            history_profile=history_profile,
            rps_values=rps_values,
            config_dir=config_dir,
        )
        d_profile = _profile_d_pool(
            args=args,
            config=cfg,
            summaries=summaries,
            history_profile=history_profile,
            rps_values=rps_values,
            config_dir=config_dir,
        )
        t_tput = _profile_t_pool(
            args=args,
            config=cfg,
            summaries=summaries,
            config_dir=config_dir,
        )
        payload["pools"]["P"][cfg.name] = {
            "mu": p_profile["mu"],
            "saturation": p_profile["saturation"],
            "be_chunk_latency_ms": p_profile.get("be_chunk_latency_ms"),
        }
        payload["pools"]["D"][cfg.name] = {
            "mu": d_profile["mu"],
            "saturation": d_profile["saturation"],
        }
        payload["pools"]["T"][cfg.name] = {
            "saturation": [{"rt_rps": 0.0, "be_throughput": t_tput}],
        }
    return payload


def _profile_t_pool(
    *,
    args: argparse.Namespace,
    config: ParallelConfig,
    summaries: dict[str, dict[str, Any]],
    config_dir: Path,
) -> float:
    """Return T-pool BE saturation (req/s) using tput_simulator."""
    from saturo.simulators.tput_simulator.config import BEConfig, ModelConfig, SimulatorConfig
    from saturo.simulators.tput_simulator.simulator import ThroughputPoolSimulator

    output_path = config_dir / "tput_profile.json"
    if output_path.exists():
        cached = json.loads(output_path.read_text())
        return max(0.0, float(cached.get("be_saturation_req_per_s", 0.0)))

    block_size = _profile_block_size(summaries, config.name)
    profile_blocks = _profile_total_gpu_blocks(summaries, config.name)
    if profile_blocks is None:
        raise ValueError(
            f"missing kv_cache_block.total_gpu_blocks for {config.name}; "
            "rerun resource profiling with a TimePredictor profile that "
            "records KV cache blocks")
    model_cfg = ModelConfig(
        block_size=block_size,
    )
    avg_in, avg_out = _be_workload_avg_lengths(args)
    sim_config = SimulatorConfig(
        duration_s=20.0,
        warmup_s=0.0,
        latency_model_path=_profile_latency_model(summaries, config.name),
        model=model_cfg,
        pipeline_parallel_size=max(int(getattr(config, "pp", 1) or 1), 1),
        be=BEConfig(
            avg_input_len=avg_in,
            avg_output_len=avg_out,
            be_decode_blocks=profile_blocks,
            chunk_size_candidates=_parse_int_csv(
                _arg(args, "resource_tput_chunk_size_candidates",
                     ",".join(str(x) for x in range(128, 24577, 128)))),
        ),
    )
    simulator = ThroughputPoolSimulator(sim_config)
    result = simulator.run()
    best = result.best.to_dict()
    be_tput = max(0.0, float(best.get("be_saturation_req_per_s", 0.0)))
    output_path.write_text(json.dumps({**best, "be_saturation_req_per_s": be_tput},
                                      indent=2))
    return be_tput


def _parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _profile_p_pool(
    *,
    args: argparse.Namespace,
    config: ParallelConfig,
    summaries: dict[str, dict[str, Any]],
    history_profile: Path,
    rps_values: list[float],
    config_dir: Path,
) -> dict[str, Any]:
    from saturo.simulators.prefill_simulator.run_prefill_simulator import (
        run_prefill_profile,
    )

    runtime_profile_path = _profile_latency_model(summaries, config.name)
    points: list[dict[str, float]] = []
    mu = 0.0
    be_chunk_latency_ms = None
    for rps in rps_values:
        target_rps = float(rps)
        output_path = (
            config_dir
            / f"prefill_n{args.num_req}_rps_{_rps_label(rps)}.json"
        )
        if output_path.exists():
            result = json.loads(output_path.read_text())
        else:
            result = run_prefill_profile(
                rt_trace=str(history_profile),
                runtime_profile=runtime_profile_path,
                be_trace=_arg(args, "be_workload_csv", ""),
                num_req=args.num_req,
                target_rps=target_rps,
                ttft_slo=_arg(args, "rt_ttft_slo_ms", 2000.0) / 1000.0,
                max_rt_violation_rate=_arg(args, "rt_max_violation_rate", 0.05),
                memory_watermark=_arg(args, "resource_memory_watermark", 0.9),
                max_prefill_chunk=_arg(args, "resource_max_prefill_chunk", 8192),
                prefill_search_granularity=_arg(
                    args, "resource_prefill_search_granularity", 1),
                max_candidates=_arg(args, "resource_profile_max_candidates", 64),
                be_input_drop_frac=_arg(args, "be_input_drop_frac", 0.0),
                be_output_keep_frac=_arg(args, "be_output_keep_frac", None),
                be_output_max_len=_arg(args, "be_output_max_len", None),
            )
            output_path.write_text(json.dumps(result, indent=2))
        best = result["best"]
        # Capture the optimized BE chunk latency from the lowest-RPS point.
        if be_chunk_latency_ms is None:
            be_chunk_latency_ms = float(best.get("be_chunk_latency_ms", 0.0))
        # A point is infeasible when RT alone exceeds the tolerated TTFT-SLO
        # violation rate (rt_infeasible from the optimizer). Fall back to the
        # strict zero-violation gate for cached results predating this field.
        rt_infeasible = result.get("rt_infeasible")
        if rt_infeasible is None:
            rt_infeasible = int(best.get("rt_slo_violations", 0)) > 0
        be_throughput = 0.0
        if not rt_infeasible:
            be_throughput = max(0.0, float(best["saturation_req_per_s"]))
        points.append({
            "rt_rps": float(rps),
            "be_throughput": be_throughput,
        })
        if not rt_infeasible:
            mu = max(mu, float(rps))
    if mu == 0.0:
        # No sampled RPS point met the TTFT SLO. Infeasible points remain in
        # the output for debugging, but their BE throughput is zeroed.
        print(
            f"[real_profile] WARNING: prefill simulator saw TTFT SLO "
            f"violations at every positive RPS point {rps_values} for config "
            f"{config.name}; setting mu=0 and zeroing infeasible BE points.")
    return {
        "mu": mu,
        "saturation": points,
        "be_chunk_latency_ms": be_chunk_latency_ms,
    }


def _profile_d_pool(
    *,
    args: argparse.Namespace,
    config: ParallelConfig,
    summaries: dict[str, dict[str, Any]],
    history_profile: Path,
    rps_values: list[float],
    config_dir: Path,
) -> dict[str, Any]:
    from saturo.simulators.decode_simulator.run_decode_simulator import run_decode_profile

    points: list[dict[str, float]] = []
    mu = 0.0
    for rps in rps_values:
        target_rps = float(rps)
        output_path = (
            config_dir
            / f"decode_n{args.num_req}_rps_{_rps_label(rps)}.json"
        )
        if output_path.exists():
            result = json.loads(output_path.read_text())
        else:
            sim_args = argparse.Namespace(
                rt_trace=str(history_profile),
                runtime_profile=_profile_latency_model(
                    summaries, config.name),
                num_req=args.num_req,
                target_rps=target_rps,
                duration_s=float(args.num_req) / target_rps,
                warmup_s=_arg(
                    args, "resource_decode_warmup_s", 0.0),
                rt_tbt_slo_ms=_arg(args, "rt_tbt_slo_ms", 100.0),
                rt_max_prompt_len=_arg(args, "max_model_len", 128),
                rt_max_output_len=_arg(args, "max_model_len", 128),
                lookahead_window_s=_arg(
                    args, "resource_decode_lookahead_window_s", 5.0),
                model_len_quantile=_arg(
                    args, "resource_decode_model_len_quantile", 0.95),
                model_len_predictor=resolve_qrf_path(
                    getattr(args, "model_len_predictor", None)),
                model_len_predictor_type="qrf",
                min_interval_ms=_arg(
                    args, "resource_decode_min_interval_ms", 5.0),
                be_workload_csv=_arg(args, "be_workload_csv", ""),
                memory_watermark=_arg(
                    args, "resource_memory_watermark", 0.9),
                chunk_sizes=_arg(
                    args, "resource_decode_chunk_sizes", "64,128"),
                strategies=_arg(
                    args, "resource_decode_strategies", "progressive"),
                verbose=False,
            )
            result = run_decode_profile(sim_args, print_summaries=False)
            output_path.write_text(json.dumps(result, indent=2))
        strategy_result = _decode_progressive_result(result)
        be_throughput = strategy_result.get(
            "be_saturation_req_per_s",
            strategy_result.get("be_throughput_req_per_s", 0.0),
        )
        rt_slo_violations = int(
            strategy_result.get("rt_slo_violations_total", 0))
        if rt_slo_violations > 0:
            be_throughput = 0.0
        points.append({
            "rt_rps": float(rps),
            "be_throughput": max(0.0, float(be_throughput)),
        })
        if rt_slo_violations == 0:
            mu = max(mu, float(rps))
    if mu == 0.0:
        # See the prefill note above: a strict TBT SLO can make every positive
        # RPS point infeasible. Keep the points for plotting/debugging, but
        # zero infeasible BE throughput so the solver does not use them.
        print(
            f"[real_profile] WARNING: decode simulator saw TBT SLO "
            f"violations at every positive RPS point {rps_values} for config "
            f"{config.name}; setting mu=0 and zeroing infeasible BE points.")
    return {"mu": mu, "saturation": points}


def profile_one_request(engine: Any, sampling_params_cls: Any,
                        req: TraceRequest) -> dict[str, Any]:
    """Profile a single request on a live engine (kept for ad-hoc debugging)."""
    request_id = f"profile_{req.request_id}"
    params = sampling_params_cls(
        temperature=0.0,
        max_tokens=req.output_len,
        ignore_eos=True,
        detokenize=False,
    )
    started = time.perf_counter()
    engine.add_request(
        request_id,
        {"prompt_token_ids": [1] * req.prompt_len},
        params,
        priority=0,
    )

    first_token_s = None
    finished_s = None
    last_seen = 0
    token_times: list[float] = []
    while engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter()
        for output in outputs:
            if output.request_id != request_id or not output.outputs:
                continue
            token_count = max(len(candidate.token_ids)
                              for candidate in output.outputs)
            while last_seen < token_count:
                token_times.append(now)
                last_seen += 1
            if first_token_s is None and token_count > 0:
                first_token_s = now
            if output.finished:
                finished_s = now

    finished_s = finished_s or time.perf_counter()
    first_token_s = first_token_s or finished_s
    tbt_ms = [
        (right - left) * 1000.0
        for left, right in zip(token_times, token_times[1:])
    ]
    return {
        "request_id": req.request_id,
        "arrival_time_s": req.arrival_time_s,
        "prompt_len": req.prompt_len,
        "output_len": req.output_len,
        "ContextTokens": req.prompt_len,
        "GeneratedTokens": req.output_len,
        "prefill_time_s": max(first_token_s - started, 0.0),
        "ttft_s": max(first_token_s - started, 0.0),
        "decode_time_s": max(finished_s - first_token_s, 0.0),
        "generated_tokens": last_seen,
        "mean_tbt_ms": sum(tbt_ms) / len(tbt_ms) if tbt_ms else 0.0,
        "p95_tbt_ms": percentile(tbt_ms, 0.95),
    }


def _synthesize_rows_from_predictor(
    predictor_path: Path | None,
    requests: list[TraceRequest],
) -> list[dict[str, Any]]:
    """Build per-request profiling rows using time_predictor predictions.

    For each request we simulate:
    - One prefill iteration: (prefill_tokens=prompt_len, sum_ctx=prompt_len)
    - output_len decode iterations: (n_decode=1, sum_ctx=prompt_len + step)

    If the predictor is unavailable, fall back to rough analytical estimates
    (1 ms/token prefill, 5 ms/token decode) so the pipeline still runs.
    """
    predictor = None
    if predictor_path is not None and Path(predictor_path).exists():
        try:
            from saturo.predictors.time_predictor import TimePredictor
            predictor = TimePredictor(str(predictor_path))
        except Exception:
            predictor = None

    def _predict_ms(prefill_tokens: int, prefill_sum_ctx: int,
                    n_decode: int, decode_sum_ctx: int) -> float:
        if predictor is not None:
            prefill_reqs = (
                [(prefill_tokens, prefill_sum_ctx)]
                if prefill_tokens > 0 else []
            )
            decode_reqs = (
                [(1, decode_sum_ctx)] * max(0, int(n_decode))
                if n_decode > 0 else []
            )
            return max(
                float(predictor.predict_forward_ms(prefill_reqs, decode_reqs)),
                0.1,
            )
        # Fallback: rough linear estimates
        if prefill_tokens > 0:
            return max(prefill_tokens * 1.0, 0.1)
        return max(n_decode * 5.0, 0.1)

    rows: list[dict[str, Any]] = []
    for req in requests:
        prefill_ms = _predict_ms(req.prompt_len, req.prompt_len, 0, 0)
        rows.append({
            "request_id": req.request_id,
            "arrival_time_s": req.arrival_time_s,
            "prompt_len": req.prompt_len,
            "output_len": req.output_len,
            "ContextTokens": req.prompt_len,
            "GeneratedTokens": req.output_len,
            "prefill_time_s": prefill_ms / 1000.0,
            "ttft_s": prefill_ms / 1000.0,
            "generated_tokens": req.output_len,
        })
    return rows


def _profile_small_batch_for_time_predictor(
    engine: Any,
    sampling_params_cls: Any,
    *,
    prompt_len: int,
    output_len: int,
    batch_size: int,
) -> None:
    """Add a tiny concurrent batch so fallback fitting sees batched decode."""

    batch_size = max(1, int(batch_size))
    params = sampling_params_cls(
        temperature=0.0,
        max_tokens=max(1, int(output_len)),
        ignore_eos=True,
        detokenize=False,
    )
    for idx in range(batch_size):
        request_id = f"profile_batch_{idx}"
        engine.add_request(
            request_id,
            {"prompt_token_ids": [1] * max(1, int(prompt_len))},
            params,
            priority=10,
        )
    while engine.has_unfinished_requests():
        engine.step()


def _ensure_iteration_time_predictor(
    *,
    engine: Any,
    cache_root: Path,
) -> Path | None:
    predictor = getattr(engine, "time_predictor", None)
    if predictor is not None and getattr(predictor, "available", False):
        profile_path = str(getattr(predictor, "profile_path", ""))
        return Path(profile_path) if profile_path else None

    logs = list(getattr(engine, "iteration_logs", []))
    rows = []
    for item in logs:
        actual = float(item.get("actual_time_ms", 0.0) or 0.0)
        if actual <= 0.0:
            continue
        rows.append([
            float(item.get("prefill_tokens", 0.0) or 0.0),
            float(item.get("prefill_sum_ctx", 0.0) or 0.0),
            float(item.get("n_decode_reqs", 0.0) or 0.0),
            float(item.get("decode_sum_ctx", 0.0) or 0.0),
            actual,
        ])
    if len(rows) < 2:
        return None

    import numpy as np

    x = np.array([row[:4] + [1.0] for row in rows], dtype=float)
    y = np.array([row[4] for row in rows], dtype=float)
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    feature_coef = np.maximum(coef[:4], 0.0)
    intercept = float(np.median(y - (x[:, :4] @ feature_coef)))
    coef = np.concatenate([feature_coef, [max(intercept, 0.0)]])
    pred = x @ coef
    denom = np.maximum(np.abs(y), 1e-9)
    mape = float(np.mean(np.abs(pred - y) / denom) * 100.0)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    identity = TimePredictorProfileIdentity.from_engine_configs(
        model_config=engine.model_config,
        parallel_config=engine.parallel_config,
    )
    num_layers = _engine_num_layers(engine)
    kv_cache_block = _engine_kv_cache_block(engine)
    profile_path = time_predictor_cache_path(identity, cache_root)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "iteration_level",
        "description": (
            "Fallback fitted from real Saturo profile iteration logs after "
            "module-level profiling failed."),
        "features": [
            "prefill_tokens",
            "prefill_sum_ctx",
            "n_decode_reqs",
            "decode_sum_ctx",
        ],
        "coef": [float(v) for v in coef[:4]],
        "intercept": float(coef[4]),
        "r2": float(r2),
        "mape": mape,
        "n_samples": int(len(rows)),
        "profile_identity": identity.to_dict(),
        "profile_key": identity.digest,
    }
    if num_layers is not None:
        payload["num_layers"] = int(num_layers)
        payload["meta"] = {"total_model_layers": int(num_layers)}
    if kv_cache_block is not None:
        payload["kv_cache_block"] = kv_cache_block
    tmp_path = profile_path.with_suffix(profile_path.suffix + ".tmp")
    wrapped = {"time_predictor": {k: v for k, v in payload.items()
                                   if k != "kv_cache_block"}}
    if kv_cache_block is not None:
        wrapped["kv_cache_block"] = kv_cache_block
    tmp_path.write_text(json.dumps(wrapped, indent=2))
    tmp_path.replace(profile_path)
    return profile_path


def _engine_kv_cache_block(engine: Any) -> dict[str, int] | None:
    cache_config = getattr(engine, "cache_config", None)
    if cache_config is None:
        return None
    total_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
    block_size = getattr(cache_config, "block_size", None)
    if total_gpu_blocks is None or block_size is None:
        return None
    return {
        "total_gpu_blocks": int(total_gpu_blocks),
        "block_size": int(block_size),
    }


def _engine_num_layers(engine: Any) -> int | None:
    model_config = getattr(engine, "model_config", None)
    candidates = [
        getattr(model_config, "hf_config", None),
        getattr(getattr(model_config, "hf_config", None), "text_config", None),
    ]
    for cfg in candidates:
        if cfg is None:
            continue
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(cfg, name, None)
            if value is not None:
                return int(value)
    return None


def summarize_profile(rows: list[dict[str, Any]],
                      args: argparse.Namespace,
                      config_name: str) -> dict[str, Any]:
    prefill_times = [float(row["prefill_time_s"]) for row in rows]
    output_lens = [max(int(row["generated_tokens"]), 1) for row in rows]
    p95_prefill_s = percentile(prefill_times, 0.95)
    avg_output_len = sum(output_lens) / max(len(output_lens), 1)
    prefill_mu = 1.0 / max(p95_prefill_s, 1e-9)
    return {
        "config_name": config_name,
        "requests": len(rows),
        "p95_prefill_s": p95_prefill_s,
        "p95_ttft_ms": p95_prefill_s * 1000.0,
        "avg_output_len": avg_output_len,
        "prefill_mu_rps": prefill_mu,
        "num_req": args.num_req,
        "target_rps": args.target_rps,
        "rt_ttft_slo_ms": args.rt_ttft_slo_ms,
        "rt_tbt_slo_ms": args.rt_tbt_slo_ms,
    }


def write_history_profile(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "TIMESTAMP",
        "request_id",
        "arrival_time_s",
        "prompt_len",
        "output_len",
        "ContextTokens",
        "GeneratedTokens",
        "prefill_time_s",
        "ttft_s",
        "decode_time_s",
        "generated_tokens",
        "mean_tbt_ms",
        "p95_tbt_ms",
    ]
    base = datetime(2026, 1, 1, 0, 0, 0)
    output_rows = []
    for row in rows:
        enriched = dict(row)
        arrival = float(enriched.get("arrival_time_s", 0.0))
        enriched["TIMESTAMP"] = (
            base + timedelta(seconds=arrival)).strftime("%Y-%m-%d %H:%M:%S.%f")
        output_rows.append(enriched)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(output_rows)


def load_history_trace(
    path: Path,
    *,
    num_req: int,
    target_rps: float,
    rows: list[dict] | None = None,
) -> list[TraceRequest]:
    from saturo.tools.trace_scaler import scale_to_target_rps_fixed_num

    if rows is None:
        rows = load_trace_rows(path)
    if not rows:
        return []

    arrivals = parse_arrival_times(rows)
    ordered = sorted(
        enumerate(zip(arrivals, rows)),
        key=lambda item: item[1][0],
    )
    if len(ordered) < num_req:
        raise ValueError(
            f"RT trace has {len(ordered)} requests, but num_req={num_req}")
    selected = ordered[:num_req]
    scaled = scale_to_target_rps_fixed_num(
        [arrival for _, (arrival, _) in selected],
        num_req=num_req,
        target_rps=target_rps,
    )

    requests = []
    for arrival_time_s, (source_idx, (_, row)) in zip(scaled, selected):
        prompt_len = parse_int(row, ("prompt_len", "ContextTokens", "input_len"), 1)
        output_len = parse_int(row, ("output_len", "GeneratedTokens"), 1)
        prompt_len = max(1, prompt_len)
        output_len = max(1, output_len)
        requests.append(TraceRequest(
            request_id=row.get("request_id") or str(source_idx),
            arrival_time_s=arrival_time_s,
            prompt_len=prompt_len,
            output_len=output_len,
        ))
    return requests


def parse_arrival_times(rows: list[dict[str, str]]) -> list[float]:
    if "arrival_offset_s" in rows[0]:
        return [float(row["arrival_offset_s"]) for row in rows]
    if "arrival_time_s" in rows[0]:
        return [float(row["arrival_time_s"]) for row in rows]
    if "arrival_time" in rows[0]:
        return [float(row["arrival_time"]) for row in rows]
    if "TIMESTAMP" in rows[0]:
        times = [parse_timestamp(row["TIMESTAMP"]) for row in rows]
        base = times[0]
        return [(ts - base).total_seconds() for ts in times]
    return [float(i) for i in range(len(rows))]


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if "." in value:
        head, frac = value.rsplit(".", 1)
        return datetime.strptime(f"{head}.{frac[:6]}", "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def parse_int(row: dict[str, str], names: tuple[str, ...], default: int) -> int:
    for name in names:
        if name in row and row[name] not in ("", None):
            return int(float(row[name]))
    return default


def parse_parallel_config(name: str) -> ParallelConfig:
    import re

    match = re.fullmatch(r"tp(\d+)_pp(\d+)", name)
    if not match:
        raise ValueError(f"unsupported config name {name!r}; expected tpN_ppM")
    tp = int(match.group(1))
    pp = int(match.group(2))
    return ParallelConfig(name=name, gpus=tp * pp, tp=tp, pp=pp)


def _local_gpu_slots(cluster_layout: str, gpu_ids: str) -> list[str]:
    """Kept for backward-compatibility; not used in Ray-based profiling."""
    groups = cluster_layout.strip() or f"local:{gpu_ids}"
    slots: list[str] = []
    for group in groups.split(";"):
        if not group.strip():
            continue
        if ":" not in group:
            raise ValueError(f"invalid resource cluster group: {group}")
        host, gpu_csv = group.split(":", 1)
        if host.strip() not in ("", "local", "localhost", "127.0.0.1"):
            continue
        slots.extend(g.strip() for g in gpu_csv.split(",") if g.strip())
    if not slots:
        raise RuntimeError("real profile collector currently needs local GPUs")
    return slots


def _cleanup_engine_runtime() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass


def _history_profile_is_simulator_ready(path: Path, num_req: int) -> bool:
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            row_count = sum(1 for _ in reader)
    except OSError:
        return False
    required = {
        "TIMESTAMP",
        "ContextTokens",
        "GeneratedTokens",
        "prefill_time_s",
    }
    return required.issubset(fields) and row_count >= num_req


def _summary_has_runtime_profile(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    profile = str(payload.get("runtime_profile")
                  or payload.get("time_predictor_profile") or "")
    available = bool(payload.get("runtime_profile_available")
                     or payload.get("time_predictor_available"))
    return bool(available and profile)


def _resource_rps_values(args: argparse.Namespace) -> list[float]:
    target_rps = float(_arg(args, "target_rps", 0.0))
    if target_rps <= 0:
        raise ValueError("target_rps must be positive")
    return sorted({
        round(target_rps / 2.0, 12),
        round(target_rps, 12),
        round(target_rps * 2.0, 12),
    })


def _decode_progressive_result(result: dict[str, Any]) -> dict[str, Any]:
    results = result.get("results", {})
    if not results:
        raise ValueError("decode simulator returned no RPS results")
    first_rps = next(iter(results.values()))
    if "progressive" in first_rps:
        return first_rps["progressive"]
    raise ValueError(
        "Resource allocation requires progressive QRF results; "
        "regenerate the decode profile instead of substituting another strategy.")


def _runtime_profile_root(args: argparse.Namespace) -> Path:
    return Path(_arg(args, "runtime_profile",
                     "saturo/results/runtime_profiles"))


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, default)
    if isinstance(value, str) and value == "":
        return default
    return value


def _summary_runtime_profile(summaries: dict[str, dict[str, Any]],
                             config_name: str) -> str:
    payload = summaries.get(config_name, {})
    return str(payload.get("runtime_profile")
               or payload.get("time_predictor_profile") or "")


def _profile_num_layers(summaries: dict[str, dict[str, Any]],
                        config_name: str) -> int:
    profile = _summary_runtime_profile(summaries, config_name)
    return _profile_num_layers_from_path(profile)


def _profile_num_layers_from_path(
    profile: Path | str | None,
    *,
    fallback_for_latency: bool = False,
) -> int:
    if profile in (None, ""):
        if fallback_for_latency:
            return 1
        raise ValueError("missing runtime_profile; cannot infer model layers")
    try:
        data = json.loads(Path(profile).read_text())
    except Exception as exc:
        if fallback_for_latency:
            return 1
        raise ValueError(
            f"cannot read runtime profile {profile!s}; "
            "cannot infer model layers") from exc

    tp_data = data.get("time_predictor")
    if isinstance(tp_data, dict):
        data = tp_data

    if data.get("num_layers") is not None:
        return int(data["num_layers"])
    meta = data.get("meta")
    if isinstance(meta, dict) and meta.get("total_model_layers") is not None:
        return int(meta["total_model_layers"])
    if data.get("model_type") == "iteration_level" and fallback_for_latency:
        return 1
    raise ValueError(
        f"runtime profile {profile!s} lacks num_layers; "
        "rerun profiling so resource estimates use the real model layer count")


def _profile_latency_model(summaries: dict[str, dict[str, Any]],
                           config_name: str) -> str:
    profile = _summary_runtime_profile(summaries, config_name)
    if not profile:
        raise ValueError(
            f"missing runtime_profile for {config_name}; "
            "run resource profiling with --runtime-profile")
    return profile


def _profile_kv_cache_block_value(
    summaries: dict[str, dict[str, Any]],
    config_name: str,
    key: str,
) -> Any | None:
    profile = _summary_runtime_profile(summaries, config_name)
    if not profile:
        return None
    try:
        data = json.loads(Path(profile).read_text())
    except Exception:
        return None
    kv = data.get("kv_cache_block")
    if isinstance(kv, dict):
        return kv.get(key)
    return None


def _profile_block_size(summaries: dict[str, dict[str, Any]],
                        config_name: str) -> int:
    value = _profile_kv_cache_block_value(summaries, config_name,
                                          "block_size")
    if value is None:
        raise ValueError(
            f"missing kv_cache_block.block_size for {config_name}; "
            "rerun resource profiling with a TimePredictor profile that "
            "records KV cache block metadata")
    return int(value)


def _profile_total_gpu_blocks(summaries: dict[str, dict[str, Any]],
                              config_name: str) -> int | None:
    """Read the profiled total GPU KV-cache blocks for a config.

    The value is persisted by the TimePredictor under the ``kv_cache_block``
    section of its fitted profile JSON (see vllm/engine/time_predictor.py).
    Returns None when the profile is missing or predates this field;
    resource profilers should fail instead of guessing capacity.
    """
    value = _profile_kv_cache_block_value(
        summaries, config_name, "total_gpu_blocks")
    return int(value) if value is not None else None


DEFAULT_BE_LENGTH_DATASET = "saturo/data/be_workload/longbench_gov_report.json"


def _be_dataset_avg_lengths(path: str) -> tuple[int, int]:
    """Average (input_len, output_len) tokens from a pre-tokenized BE length pool.

    The pool file is produced by experiments/saturo/build_be_length_dataset.py
    (LongBench gov_report + gov_report_e). Falls back to (8000, 100) only when
    the file is missing/unreadable so profiling never hard-crashes.
    """
    try:
        payload = json.loads(Path(path).read_text())
        pairs = payload.get("pairs", [])
        if pairs:
            n = len(pairs)
            avg_in = int(round(sum(int(p[0]) for p in pairs) / n))
            avg_out = int(round(sum(int(p[1]) for p in pairs) / n))
            return avg_in, avg_out
    except Exception:  # noqa: BLE001
        pass
    return 8000, 100


def _be_workload_avg_lengths(args: argparse.Namespace) -> tuple[int, int]:
    """Compute average (input_len, output_len) from the BE workload CSV.

    Reads ``--be-workload-csv`` (the same CSV used by the scheduler at runtime)
    and returns the mean prompt/output token counts.  Falls back to the
    pre-tokenized JSON pool when the CSV is unavailable or unreadable.
    """
    be_csv = getattr(args, "be_workload_csv", "") or ""
    if be_csv:
        try:
            prompts: list[int] = []
            outputs: list[int] = []
            with open(be_csv, newline="") as f:
                for row in csv.DictReader(f):
                    prompts.append(int(float(row["prompt_len"])))
                    outputs.append(int(float(row["output_len"])))
            if prompts:
                return (
                    int(round(sum(prompts) / len(prompts))),
                    int(round(sum(outputs) / len(outputs))),
                )
        except Exception:  # noqa: BLE001
            pass
    return _be_dataset_avg_lengths(DEFAULT_BE_LENGTH_DATASET)


def _rps_label(rps: float) -> str:
    return f"{float(rps):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return float(ordered[idx])
