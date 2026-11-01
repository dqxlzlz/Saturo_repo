#!/usr/bin/env python3
"""Benchmark RT-BE colocation on a real GPU instance (P-pool or D-pool).

Drives a Saturo ``AsyncLLMEngine`` with the standard vLLM scheduler
(chunked prefill + priority scheduling).  Requests are submitted the same
way ``benchmarks/benchmark_serving.py`` drives a live server: each request
is its own coroutine calling ``engine.generate(...)`` and RT arrivals are
paced by their dataset inter-arrival times.

Scheduling modes (mutually exclusive; default is vanilla vLLM, i.e. the
limitation study):

* ``--enable-p-pool-saturation``   Saturo P-pool saturation.
* ``--enable-decode-saturation``   Saturo D-pool decode saturation.
  ``--request-mode split-phase`` admits RT and the decode-ready BE
  population directly into decode (PD-disaggregated D node semantics);
  ``--request-mode local`` runs ordinary prefill->decode lifecycles.
  With ``--saturation-strategy oracle`` the engine is rebuilt per
  (chunk, rps) point because the oracle RT schedule is loaded once at
  engine construction.

Sweeps ``max_num_batched_tokens`` (the BE chunk size in tokens) to measure
the empirical trade-off between RT TTFT-SLO violation rate and BE prefill
throughput.

BE requests use ``max_tokens=1`` (output_len=1) so that BE work is
essentially pure prefill.  RT requests are injected according to the
arrival timestamps in the workload CSV (``arrival_offset_s``), preserving
the original arrival pattern.

The engine is re-created for each BE chunk size, so a sweep loads the
model once per chunk size.

Example::

    CUDA_VISIBLE_DEVICES=0 python experiments/saturo/benchmark_rt_be_colocate.py \\
        --model /root/models/Qwen/Qwen2.5-72B-Instruct \\
        --rt-workload-csv data/datasets/rt_workload.csv \\
        --be-workload-csv data/datasets/be_workload.csv \\
        --num-req 600 --target-rps 1,2,4 \\
        --runtime-profile saturo/results/time_predictors/tp1_pp1.json \\
        --max-num-batched-tokens-list 256,512,1024,2048,4096 \\
        --warmup-s 10 \\
        --output-json results/rt_be_colocate_bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

_PROJECT = Path(__file__).resolve().parents[3]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    prompt_len: int
    output_len: int
    arrival_offset_s: float = 0.0


@dataclass
class RunResult:
    max_num_batched_tokens: int
    # RT metrics
    rt_injected: int = 0
    rt_first_token: int = 0
    rt_ttft_ms: List[float] = field(default_factory=list)
    rt_pred_service_ms: List[float] = field(default_factory=list)
    rt_queue_ms: List[float] = field(default_factory=list)
    rt_violations: int = 0
    rt_tbt_ms: List[float] = field(default_factory=list)
    rt_tbt_violations: int = 0
    rt_completed: int = 0
    rt_aborted_on_stop: int = 0
    rt_output_tokens: int = 0
    rt_ttft_slo_passed: int = 0
    rt_tbt_slo_passed: int = 0
    rt_joint_slo_passed: int = 0
    rt_prefill_req_equiv: float = 0.0
    rt_decode_req_equiv: float = 0.0
    rt_requests: List[dict[str, Any]] = field(default_factory=list)
    # BE metrics
    be_started: int = 0
    be_completed: int = 0
    be_prefill_tokens: int = 0
    # BE prefill tokens actually processed per scheduler step inside the
    # measurement window (accurate high-resolution throughput signal; does
    # not depend on how many long BE requests fully finish in the window).
    be_prefill_tokens_processed: int = 0
    be_decode_tokens_processed: int = 0
    be_output_tokens: int = 0
    be_aborted_on_stop: int = 0
    be_inflight_at_stop: int = 0
    be_prefill_req_equiv: float = 0.0
    be_decode_req_equiv: float = 0.0
    # Scheduler/stop diagnostics.  These are cheap to collect and are useful
    # when a long sweep exposes an anomalous point that would otherwise need
    # to be rerun.
    scheduler_steps: int = 0
    scheduler_tokens: int = 0
    scheduler_preempted: int = 0
    scheduler_swap_out_blocks: int = 0
    scheduler_batch_tokens: List[float] = field(default_factory=list)
    stop_overrun_ms: float = 0.0
    # Fixed-request RT completion extends beyond the BE sending window.
    rt_drain_s: float = 0.0
    rt_drain_timed_out: bool = False
    # timing
    window_s: float = 0.0
    started_at: str = ""
    ended_at: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    rank = int(math.ceil(q * len(ordered))) - 1
    return ordered[min(len(ordered) - 1, max(0, rank))]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(v) for v in values]
    if not data:
        return {"count": 0, "mean": 0.0, "p50": 0.0,
                "median": 0.0, "p90": 0.0, "p95": 0.0,
                "p99": 0.0, "max": 0.0}
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "p50": statistics.median(data),
        "median": statistics.median(data),
        "p90": percentile(data, 0.90),
        "p95": percentile(data, 0.95),
        "p99": percentile(data, 0.99),
        "max": max(data),
    }


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def _load_trace(
    path: Path,
    kind: str,
    num_req: int | None = None,
    target_rps: float | None = None,
) -> List[TraceRequest]:
    from saturo.runtime.workload import (
        load_rt_workload_csv,
        load_workload_csv,
    )

    if not path.is_file():
        raise FileNotFoundError(f"workload CSV not found: {path}")
    if kind == "rt":
        if num_req is None or target_rps is None:
            raise ValueError("RT loading requires num_req and target_rps")
        loaded = load_rt_workload_csv(
            str(path), num_req=num_req, target_rps=target_rps)
    else:
        loaded = load_workload_csv(str(path), kind)
    trace = [
        TraceRequest(
            str(r.request_id),
            int(r.prompt_len),
            max(1, int(r.output_len)),
            float(r.arrival_offset_s) if kind == "rt" else 0.0,
        )
        for r in loaded
    ]
    if not trace:
        raise ValueError(f"workload CSV is empty: {path}")
    if kind == "rt":
        trace = sorted(trace, key=lambda r: r.arrival_offset_s)
    return trace


def _filter_be(
    trace: List[TraceRequest], args: argparse.Namespace
) -> List[TraceRequest]:
    """Apply BE length filters matching trace_loader / dataset knobs."""
    items = list(trace)
    if not args.full_generation_workload:
        # Historical limitation-study mode: BE is pure prefill work.
        items = [
            TraceRequest(r.request_id, r.prompt_len, 1, r.arrival_offset_s)
            for r in items
        ]

    if args.be_input_drop_frac:
        if not 0.0 < args.be_input_drop_frac < 1.0:
            raise ValueError("--be-input-drop-frac must be in (0, 1)")
        items.sort(key=lambda r: r.prompt_len)
        items = items[int(args.be_input_drop_frac * len(items)):]
    if args.be_output_max_len is not None:
        # Already clamped to 1 above, but keep for consistency
        pass
    elif args.be_output_keep_frac is not None:
        if not 0.0 < args.be_output_keep_frac <= 1.0:
            raise ValueError("--be-output-keep-frac must be in (0, 1]")
        items.sort(key=lambda r: r.prompt_len)
        items = items[: max(1, int(args.be_output_keep_frac * len(items)))]
    if not items:
        raise ValueError("BE length filter removed all rows")
    return items


def _sanitize(trace: List[TraceRequest], max_model_len: int) -> List[TraceRequest]:
    usable: List[TraceRequest] = []
    for r in trace:
        if r.prompt_len >= max_model_len:
            continue
        out = min(r.output_len, max_model_len - r.prompt_len)
        usable.append(TraceRequest(
            r.request_id, r.prompt_len, max(1, out), r.arrival_offset_s))
    if not usable:
        raise ValueError(f"no request fits max_model_len={max_model_len}")
    return usable


def _point_num_req(args: argparse.Namespace, rps: float) -> tuple[int, float]:
    """RT prefix size and send window for one RPS point."""
    from saturo.tools.trace_scaler import fixed_num_window_s
    if args.duration_s > 0:
        send_window_s = float(args.duration_s)
        # One extra source request prevents the trace scaler's last point
        # (at exactly num_req/rps) from being the only boundary sample. The
        # injector excludes arrivals at the hard stop itself.
        point_num_req = max(
            2, int(math.ceil(send_window_s * float(rps))) + 1)
    else:
        point_num_req = args.num_req
        send_window_s = fixed_num_window_s(point_num_req, rps)
    return point_num_req, send_window_s


def _write_oracle_schedule(
    args: argparse.Namespace,
    rt_path: Path,
    rps: float,
    out_path: Path,
    chunk_tokens: int,
) -> Path:
    """Materialise the exact future-RT schedule for one oracle point.

    The decode-saturation oracle provider loads this file once at engine
    construction (offsets are anchored to the first observed RT arrival),
    so oracle points rebuild the engine per (chunk, rps).
    """
    point_num_req, _ = _point_num_req(args, rps)
    trace = _load_trace(
        rt_path, kind="rt", num_req=point_num_req, target_rps=rps)
    if args.max_model_len is not None:
        trace = _sanitize(trace, args.max_model_len)
    schedule_path = out_path.parent / (
        f"{out_path.stem}_oracle_rt_chunk{chunk_tokens}_rps{rps:g}.json")
    schedule_path.write_text(json.dumps({
        "schema_version": 1,
        "lookahead_window_s": float(args.oracle_lookahead_s),
        "requests": [
            {
                "arrival_offset_s": float(r.arrival_offset_s),
                "prompt_len": int(r.prompt_len),
                "output_len": int(r.output_len),
            }
            for r in trace
        ],
    }, indent=2), encoding="utf-8")
    return schedule_path


# ---------------------------------------------------------------------------
# Async engine creation
# ---------------------------------------------------------------------------


def _build_async_engine(
    args: argparse.Namespace, max_num_batched_tokens: int
) -> Any:
    """Build a Saturo P-pool ``AsyncLLMEngine`` with the standard scheduler.

    Mirrors the PREFILL-role engine configuration used by
    ``saturo.runtime.worker.RayEngineWorker`` (priority scheduling, chunked
    prefill, eager mode), but exposes it through ``AsyncLLMEngine`` so RT and
    BE requests can be streamed concurrently.  P-pool saturation is not
    enabled: this is the limitation study on vanilla vLLM scheduling.
    """
    from vllm.config import CompilationConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "skip_tokenizer_init": True,
        "enforce_eager": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "load_format": args.load_format,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_custom_all_reduce": True,
        "scheduling_policy": "priority",
        # Standard chunked-prefill path: long BE prompts are split into
        # partial-prefill chunks bounded by max_num_batched_tokens.
        "enable_chunked_prefill": True,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "disable_log_requests": True,
        "disable_log_stats": True,
    }
    if sum((
        bool(args.enable_p_pool_saturation),
        bool(getattr(args, "enable_decode_saturation", False)),
    )) > 1:
        raise ValueError(
            "--enable-p-pool-saturation and "
            "--enable-decode-saturation are mutually exclusive")
    if args.enable_p_pool_saturation:
        be_chunk_size = (
            max_num_batched_tokens
            if getattr(args, "be_chunk_size_follows_mnbt", False)
            else args.p_pool_be_chunk_size)
        engine_kwargs.update({
            "enable_p_pool_saturation": True,
            "p_pool_be_chunk_ms": args.p_pool_be_chunk_ms,
            "p_pool_be_chunk_size": be_chunk_size,
            "p_pool_preemption_margin": args.p_pool_preemption_margin,
        })
    if getattr(args, "enable_decode_saturation", False):
        # Mirror the DECODE-role engine configuration used by
        # saturo.runtime.worker.RayEngineWorker. Block fraction, BE-prefill
        # reserve blocks and the oracle schedule travel through the
        # SATURO_DECODE_* environment variables set in main().
        engine_kwargs.update({
            "enable_decode_saturation": True,
            "decode_saturation_tbt_ms": args.tbt_slo_ms,
            "decode_saturation_lookahead_s": (
                0.0 if args.saturation_strategy == "greedy"
                else args.oracle_lookahead_s),
            "decode_saturation_require_model_len_predictor": False,
        })
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    profile = str(args.runtime_profile or "")
    if profile:
        engine_kwargs["time_predictor_profile"] = profile
        if args.enable_p_pool_saturation:
            engine_kwargs["p_pool_time_predictor_profile"] = profile
    if bool(args.disable_inductor):
        engine_kwargs["compilation_config"] = CompilationConfig(
            level=0,
            custom_ops=["none"],
            use_cudagraph=False,
            use_inductor=False,
        )
    return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))


# ---------------------------------------------------------------------------
# Async request drivers
# ---------------------------------------------------------------------------


async def _warmup_fixed_requests(engine, rt_trace, be_trace, args):
    """Warm the same engine with complete batches; never abort warmup work."""
    from vllm import SamplingParams
    deadline = time.perf_counter() + args.warmup_s
    round_id = 0
    async def one(req, kind, serial):
        params = SamplingParams(
            temperature=0.0, max_tokens=16, ignore_eos=True,
            detokenize=False, is_realtime=(kind == "rt"),
            **({"prefill_ddl": time.time() + args.rt_prefill_deadline_s}
               if kind == "rt" else {}))
        count = 0
        async for out in engine.generate(
                {"prompt_token_ids": [0] * req.prompt_len}, params,
                f"warmup_{round_id}_{kind}_{serial}",
                priority=0 if kind == "rt" else 10):
            count = max((len(c.token_ids) for c in out.outputs), default=0)
        if count != 16:
            raise RuntimeError("Warmup request failed to complete")
    while time.perf_counter() < deadline:
        round_id += 1
        await asyncio.wait_for(asyncio.gather(
            *(one(req, "rt", i) for i, req in enumerate(rt_trace[:16])),
            *(one(req, "be", i) for i, req in enumerate(be_trace[:16]))),
            timeout=240)
    # All warmup generators completed naturally, so no request carries over.
    print(f"[fixed-request] warmup complete: {round_id} batches of 16 RT + 16 BE", flush=True)


async def _run_one_async(
    engine: Any,
    rt_trace: List[TraceRequest],
    be_trace: List[TraceRequest],
    max_num_batched_tokens: int,
    args: argparse.Namespace,
    send_window_s: float,
) -> RunResult:
    """Drive one ``max_num_batched_tokens`` point with an AsyncLLMEngine.

    RT requests are streamed at their dataset arrival offsets (like
    ``benchmark_serving.py`` pacing requests by inter-arrival time), while a
    fixed pool of ``--be-inflight`` best-effort prefill requests is kept
    saturated by relaunching a new BE request whenever one finishes.
    """
    from vllm import SamplingParams

    result = RunResult(max_num_batched_tokens=max_num_batched_tokens)
    result.started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()
    fixed_requests = getattr(args, "fixed_request_run", False)
    meas_start = run_start if fixed_requests else run_start + args.warmup_s
    meas_end = run_start + send_window_s
    hard_deadline = run_start + args.max_wall_s

    def _in_window(t: float) -> bool:
        return meas_start <= t < meas_end

    # Split-phase decode saturation: RT and the decode-ready BE population
    # bypass real prefill through direct scheduler admission (the scheduler
    # allocates prompt KV blocks and seeds one synthetic output token; see
    # saturo.runtime.worker._request_inputs).
    split_phase = (
        getattr(args, "enable_decode_saturation", False)
        and getattr(args, "request_mode", "local") == "split-phase")

    def _rt_params(req: TraceRequest) -> Any:
        prefill_deadline_s = args.rt_prefill_deadline_s
        params_kwargs: dict[str, Any] = dict(
            temperature=0.0,
            max_tokens=(
                req.output_len if args.full_generation_workload else 1),
            ignore_eos=True,
            detokenize=False,
            is_realtime=True,
            prefill_ddl=time.time() + prefill_deadline_s,
        )
        if split_phase:
            # Direct admission seeds one synthetic output token as the first
            # decode input. Add one to the cap and hide the seed from token
            # accounting in _rt_task/_be_loop.
            params_kwargs["max_tokens"] += 1
            params_kwargs["extra_args"] = {"saturo_direct_decode": True}
        return SamplingParams(**params_kwargs)

    def _be_params(req: TraceRequest, be_phase: str) -> Any:
        if be_phase == "decode":
            return SamplingParams(
                temperature=0.0,
                max_tokens=req.output_len + 1,
                ignore_eos=True,
                detokenize=False,
                extra_args={"saturo_direct_decode": True},
            )
        if be_phase == "prefill":
            return SamplingParams(
                temperature=0.0,
                max_tokens=1,
                ignore_eos=True,
                detokenize=False,
            )
        return SamplingParams(
            temperature=0.0,
            max_tokens=(
                req.output_len if args.full_generation_workload else 1),
            ignore_eos=True,
            detokenize=False,
        )

    stop = asyncio.Event()
    be_state = {"serial": 0, "idx": 0}
    active_be_request_ids: set[str] = set()
    active_rt_request_ids: set[str] = set()
    request_shapes: dict[str, tuple[str, int, int]] = {}
    measured_request_ids: set[str] = set()
    prefill_complete_ids: set[str] = set()
    # Requests admitted directly into decode (split-phase). Their first
    # stream token is the synthetic admission seed: hide it from token
    # accounting and never credit them with prefill work.
    direct_decode_ids: set[str] = set()
    output_tokens_by_request: dict[str, int] = {}
    rt_records: dict[str, dict[str, Any]] = {}

    # ---- step-level instrumentation -------------------------------------
    # Always-on: accumulate BE prefill tokens actually processed per
    # scheduler step inside the measurement window. This is the accurate,
    # high-resolution BE throughput signal -- it does NOT depend on how many
    # long BE requests happen to fully finish in the window (the completed-
    # request count is too coarse: ~6-9 completions over 15s quantizes the
    # throughput to +/- one ~10k-token request).
    #
    # Set SATURO_RT_DEBUG=1 to additionally record a per-step log and print
    # per-RT queue timing / a scheduler occupancy monitor.
    debug = os.environ.get("SATURO_RT_DEBUG", "0") not in ("0", "", "false")

    step_stats = {
        "be_prefill_tokens": 0,
        "be_decode_tokens": 0,
        "steps": 0,
        "tokens": 0,
        "preempted": 0,
        "swap_out_blocks": 0,
        "batch_tokens": [],
    }

    # Per-step log (debug only). Each entry:
    #   t   : relative time (s) when this step was scheduled
    #   dt  : wall-clock gap to the *next* schedule() call (~= this step's
    #         forward time), filled in lazily
    #   batched      : num_batched_tokens for the step
    #   rt_reqs/be_reqs : #RT / #BE seq-groups in the step
    #   rt_chunk/be_chunk : summed token_chunk_size for RT / BE
    step_log: List[dict[str, Any]] = []
    try:
        for _sched in engine.engine.scheduler:
            # Recover the true original schedule() so repeated runs on a
            # reused engine (RPS sweep) never stack wrappers.
            _orig_schedule = getattr(_sched, "_saturo_orig_schedule", None)
            if _orig_schedule is None:
                _orig_schedule = _sched.schedule
                _sched._saturo_orig_schedule = _orig_schedule

            def _make_wrap(orig):
                def _wrapped():
                    t0 = time.perf_counter()
                    if debug and step_log:
                        step_log[-1]["dt"] = t0 - step_log[-1]["_abs"]
                    out = orig()
                    so = out[1]
                    if meas_start <= t0 < meas_end:
                        step_stats["steps"] += 1
                        step_stats["tokens"] += int(so.num_batched_tokens)
                        step_stats["preempted"] += int(
                            getattr(so, "preempted", 0))
                        step_stats["swap_out_blocks"] += len(
                            getattr(so, "blocks_to_swap_out", []))
                        step_stats["batch_tokens"].append(
                            int(so.num_batched_tokens))
                    rt_reqs = be_reqs = rt_chunk = be_chunk = 0
                    be_detail: List[tuple] = []
                    for ssg in so.scheduled_seq_groups:
                        rid = str(ssg.seq_group.request_id)
                        cs = int(ssg.token_chunk_size)
                        if rid.startswith("rt_"):
                            rt_reqs += 1
                            rt_chunk += cs
                        else:
                            be_reqs += 1
                            be_chunk += cs
                            if meas_start <= t0 < meas_end:
                                if ssg.seq_group.is_prefill():
                                    step_stats["be_prefill_tokens"] += cs
                                else:
                                    # Each sequence emits one token per
                                    # decode step. Current traces use one
                                    # sequence per sequence group.
                                    step_stats["be_decode_tokens"] += cs
                            if debug:
                                try:
                                    seq = ssg.seq_group.get_seqs()[0]
                                    be_detail.append((
                                        rid, cs,
                                        seq.get_num_computed_tokens(),
                                        seq.get_prompt_len()))
                                except Exception:
                                    be_detail.append((rid, cs, -1, -1))
                    if debug:
                        step_log.append({
                            "t": t0 - run_start, "_abs": t0, "dt": 0.0,
                            "batched": int(so.num_batched_tokens),
                            "rt_reqs": rt_reqs, "be_reqs": be_reqs,
                            "rt_chunk": rt_chunk, "be_chunk": be_chunk,
                            "be_detail": be_detail,
                        })
                    return out
                return _wrapped

            _sched.schedule = _make_wrap(_orig_schedule)
    except Exception as exc:
        print(f"[instrument] failed to wrap scheduler.schedule: {exc}",
              flush=True)

    def _steps_between(abs_lo: float, abs_hi: float) -> List[dict[str, Any]]:
        lo = abs_lo - run_start
        hi = abs_hi - run_start
        return [s for s in step_log if lo <= s["t"] < hi]

    def _sched_snapshot() -> dict[str, Any]:
        """Snapshot scheduler queue occupancy split by RT vs BE."""
        try:
            schedulers = engine.engine.scheduler
        except Exception:
            return {}
        run_rt = run_be = wait_rt = wait_be = swap_n = 0
        run_tok = 0
        for sched in schedulers:
            for sg in list(sched.running):
                is_rt = str(sg.request_id).startswith("rt_")
                if is_rt:
                    run_rt += 1
                else:
                    run_be += 1
                try:
                    for seq in sg.get_seqs():
                        run_tok += seq.get_num_uncomputed_tokens()
                except Exception:
                    pass
            for sg in list(sched.waiting):
                if str(sg.request_id).startswith("rt_"):
                    wait_rt += 1
                else:
                    wait_be += 1
            swap_n += len(getattr(sched, "swapped", []))
        return {
            "run_rt": run_rt, "run_be": run_be,
            "wait_rt": wait_rt, "wait_be": wait_be,
            "swapped": swap_n, "run_uncomputed_tok": run_tok,
        }

    async def _monitor() -> None:
        if not debug:
            return
        while not stop.is_set():
            t = time.perf_counter() - run_start
            snap = _sched_snapshot()
            if snap:
                print(
                    f"[mon t={t:6.2f}s] run(rt={snap['run_rt']},be={snap['run_be']}) "
                    f"wait(rt={snap['wait_rt']},be={snap['wait_be']}) "
                    f"swapped={snap['swapped']} "
                    f"run_uncomputed_tok={snap['run_uncomputed_tok']}",
                    flush=True,
                )
            await asyncio.sleep(0.5)
    time_predictor = None
    if args.runtime_profile:
        try:
            from saturo.predictors.time_predictor import TimePredictor
            time_predictor = TimePredictor(str(args.runtime_profile),
                                           warn_missing=False)
        except Exception:
            time_predictor = None

    def _predict_rt_service_ms(req: TraceRequest) -> float:
        if time_predictor is None or not getattr(time_predictor, "available", False):
            return 0.0
        return float(time_predictor.predict_forward_ms(
            [(int(req.prompt_len), int(req.prompt_len))], []))

    async def _rt_task(req: TraceRequest, serial: int) -> None:
        # RT is priority 0 (highest); a single prefill token gives us TTFT.
        rid = f"rt_{serial}:{req.request_id}"
        submit_t = time.perf_counter()
        counted = fixed_requests or _in_window(submit_t)
        request_shapes[rid] = ("rt", req.prompt_len, req.output_len)
        active_rt_request_ids.add(rid)
        output_tokens_by_request[rid] = 0
        if counted:
            result.rt_injected += 1
            measured_request_ids.add(rid)
            rt_records[rid] = {
                "request_id": rid,
                "prompt_len": req.prompt_len,
                "output_len": req.output_len,
                "arrival_s": submit_t - run_start,
                "scheduled_arrival_s": req.arrival_offset_s,
                "completed": False,
                "ttft_ms": None,
                "tbt_ms": [],
                "output_tokens": 0,
            }
        if debug:
            snap = _sched_snapshot()
            print(
                f"[RT submit t={submit_t - run_start:7.2f}s] {rid} "
                f"prompt_len={req.prompt_len} counted={counted} "
                f"| sched run(rt={snap.get('run_rt')},be={snap.get('run_be')}) "
                f"wait(rt={snap.get('wait_rt')},be={snap.get('wait_be')}) "
                f"run_uncomputed_tok={snap.get('run_uncomputed_tok')}",
                flush=True,
            )
        if split_phase:
            direct_decode_ids.add(rid)
        prompt = {"prompt_token_ids": (
            [1] if split_phase else [0]) * max(1, req.prompt_len)}
        first_token_seen = False
        last_token_t: Optional[float] = None
        last_token_count = 0
        try:
            async for out in engine.generate(
                    prompt, _rt_params(req), rid, priority=0):
                raw_count = (
                    max((len(c.token_ids) for c in out.outputs), default=0)
                    if out.outputs else 0)
                # Hide the synthetic direct-decode admission seed.
                token_count = (
                    max(raw_count - 1, 0) if rid in direct_decode_ids
                    else raw_count)
                event_t = time.perf_counter()
                # In strict-stop mode, work reported after the five-minute
                # boundary is intentionally excluded even if abort cleanup
                # takes one more event-loop turn.
                if args.strict_stop and event_t >= meas_end:
                    continue
                output_tokens_by_request[rid] = max(
                    output_tokens_by_request.get(rid, 0), token_count)
                if counted:
                    rt_records[rid]["output_tokens"] = token_count
                if (not first_token_seen and token_count > 0):
                    first_token_seen = True
                    prefill_complete_ids.add(rid)
                    first_t = event_t
                    last_token_t = first_t
                    last_token_count = token_count
                    ttft_ms = (first_t - submit_t) * 1000.0
                    pred_service_ms = _predict_rt_service_ms(req)
                    queue_ms = max(0.0, ttft_ms - pred_service_ms)
                    if debug:
                        seg = _steps_between(submit_t, first_t)
                        be_only = [s for s in seg if s["rt_reqs"] == 0
                                   and s["be_reqs"] > 0]
                        be_fwd_ms = sum(s["dt"] for s in be_only) * 1000.0
                        batched_seq = ",".join(
                            str(s["batched"]) for s in seg[:16])
                        print(
                            f"[RT  first t={first_t - run_start:7.2f}s] {rid} "
                            f"prompt_len={req.prompt_len} ttft={ttft_ms:8.1f}ms "
                            f"pred_service={pred_service_ms:7.1f}ms "
                            f"queue={queue_ms:8.1f}ms counted={counted}",
                            flush=True,
                        )
                        print(
                            f"        -> during queue: steps={len(seg)} "
                            f"be_only_steps={len(be_only)} "
                            f"be_fwd={be_fwd_ms:7.1f}ms "
                            f"batched_tokens_per_step=[{batched_seq}]",
                            flush=True,
                        )
                        for _i, _s in enumerate(seg[:10]):
                            _det = "; ".join(
                                f"{_r.split(':')[0]}(chunk={_c},"
                                f"computed={_cc}/{_pl})"
                                for (_r, _c, _cc, _pl) in _s["be_detail"]
                            ) or "-"
                            print(
                                f"           step[{_i}] t={_s['t']:6.2f}s "
                                f"dt={_s['dt'] * 1000:6.0f}ms "
                                f"batched={_s['batched']:5d} "
                                f"rt={_s['rt_reqs']} be={_det}",
                                flush=True,
                            )
                    if counted:
                        result.rt_first_token += 1
                        result.rt_ttft_ms.append(ttft_ms)
                        result.rt_pred_service_ms.append(pred_service_ms)
                        result.rt_queue_ms.append(queue_ms)
                        rt_records[rid]["ttft_ms"] = ttft_ms
                        if ttft_ms > args.ttft_slo_ms:
                            result.rt_violations += 1
                elif token_count > last_token_count and last_token_t is not None:
                    token_t = event_t
                    tbt_ms = (token_t - last_token_t) * 1000.0
                    # V0 usually emits one token per callback. Preserve the
                    # token count if multiple tokens are co-emitted.
                    tbt_values = [tbt_ms] + [0.0] * max(
                        token_count - last_token_count - 1, 0)
                    if counted:
                        result.rt_tbt_ms.extend(tbt_values)
                        rt_records[rid]["tbt_ms"].extend(tbt_values)
                        result.rt_tbt_violations += sum(
                            value > args.tbt_slo_ms for value in tbt_values)
                    last_token_t = token_t
                    last_token_count = token_count
            done_t = time.perf_counter()
            if (counted and (fixed_requests or done_t <= meas_end)
                    and output_tokens_by_request.get(rid, 0) >= req.output_len):
                result.rt_completed += 1
                if fixed_requests and result.rt_completed % 100 == 0:
                    print(f"[fixed-request] RT completed {result.rt_completed}/{len(rt_trace)}", flush=True)
                rt_records[rid]["completed"] = True
        finally:
            active_rt_request_ids.discard(rid)

    async def _be_loop(loop_idx: int) -> None:
        # BE is priority 10 (lowest); relaunch on completion to stay saturated.
        # Split-phase keeps independent decode-ready / prefill-only
        # populations (50/50 across the inflight loops, mirroring the
        # standalone D-pool benchmark's --enable-be-decode split).
        if split_phase:
            be_phase = (
                "decode"
                if getattr(args, "enable_be_decode", True) and loop_idx % 2 == 0
                else "prefill")
        else:
            be_phase = "local"
        while not stop.is_set():
            req = be_trace[be_state["idx"] % len(be_trace)]
            be_state["idx"] += 1
            rid = f"be_{be_state['serial']}:{req.request_id}"
            be_state["serial"] += 1
            if be_phase == "decode":
                direct_decode_ids.add(rid)
            prompt = {"prompt_token_ids": (
                [1] if be_phase == "decode" else [0]) * max(1, req.prompt_len)}
            token_count = 0
            start_t = time.perf_counter()
            request_shapes[rid] = ("be", req.prompt_len, req.output_len)
            output_tokens_by_request[rid] = 0
            if _in_window(start_t):
                measured_request_ids.add(rid)
            active_be_request_ids.add(rid)
            result.be_started += 1
            try:
                async for out in engine.generate(
                        prompt, _be_params(req, be_phase), rid, priority=10):
                    if out.outputs:
                        raw_count = max(
                            len(c.token_ids) for c in out.outputs)
                        token_count = (
                            max(raw_count - 1, 0)
                            if rid in direct_decode_ids else raw_count)
                        event_t = time.perf_counter()
                        if (not args.strict_stop and not fixed_requests) or event_t < meas_end:
                            output_tokens_by_request[rid] = max(
                                output_tokens_by_request.get(rid, 0),
                                token_count)
                            if raw_count > 0 and be_phase != "decode":
                                prefill_complete_ids.add(rid)
                done_t = time.perf_counter()
                if _in_window(done_t):
                    result.be_completed += 1
                    if be_phase != "decode":
                        # Direct-decode admission never executes a real
                        # prefill, so its prompt is not prefill work.
                        result.be_prefill_tokens += req.prompt_len
                    result.be_output_tokens += token_count
            finally:
                active_be_request_ids.discard(rid)

    rt_tasks: List[asyncio.Task] = []

    async def _rt_injector() -> None:
        for serial, req in enumerate(rt_trace):
            target = run_start + req.arrival_offset_s
            if not fixed_requests and target >= meas_end:
                break
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            rt_tasks.append(asyncio.create_task(_rt_task(req, serial)))

    be_tasks = [
        asyncio.create_task(_be_loop(i))
        for i in range(max(1, args.be_inflight))
    ]
    injector = asyncio.create_task(_rt_injector())
    monitor = asyncio.create_task(_monitor())

    now = time.perf_counter()
    if meas_end > now:
        await asyncio.sleep(meas_end - now)

    # Snapshot effective prefill progress before aborting. Unlike summing
    # scheduled chunks, this does not double-count work that the scheduler rolled
    # back and recomputed after a BE preemption.
    current_prefill_tokens: dict[str, int] = {}
    try:
        for sched in engine.engine.scheduler:
            for queue in (sched.waiting, sched.running,
                          getattr(sched, "swapped", [])):
                for group in list(queue):
                    rid = str(group.request_id)
                    shape = request_shapes.get(rid)
                    if shape is None:
                        continue
                    try:
                        computed = max(
                            seq.get_num_computed_tokens()
                            for seq in group.get_seqs())
                    except Exception:
                        computed = 0
                    current_prefill_tokens[rid] = min(
                        max(int(computed), 0), shape[1])
    except Exception as exc:
        print(f"[instrument] pre-stop progress snapshot failed: {exc}",
              flush=True)

    if fixed_requests:
        # Include the request at exactly N/RPS and keep the BE pool saturated
        # until every RT finishes. BE statistics still stop at meas_end.
        await injector
        await asyncio.sleep(0)
        if result.rt_injected != len(rt_trace):
            raise RuntimeError(f"Expected {len(rt_trace)} RT submissions, got {result.rt_injected}")
        remaining = max(0.0, hard_deadline - time.perf_counter())
        _, pending_rt = await asyncio.wait(rt_tasks, timeout=remaining)
        result.rt_drain_timed_out = bool(pending_rt)
        result.rt_drain_s = max(0.0, time.perf_counter() - meas_end)
        print(f"[fixed-request] submitted={result.rt_injected} "
              f"completed={result.rt_completed} drain_s={result.rt_drain_s:.3f} "
              f"timed_out={result.rt_drain_timed_out}", flush=True)

    # Stop feeding work and abort every request at the measurement boundary.
    # In strict mode there is no late-RT drain: unfinished RT requests count
    # as partial progress and as strict-SLO misses.
    stop.set()
    be_to_abort = list(active_be_request_ids)
    rt_to_abort = list(active_rt_request_ids)
    result.be_inflight_at_stop = len(be_to_abort)
    await injector
    monitor.cancel()
    abort_ids = be_to_abort + rt_to_abort
    if abort_ids:
        abort_results = await asyncio.gather(
            *(engine.abort(rid) for rid in abort_ids),
            return_exceptions=True)
        result.be_aborted_on_stop = sum(
            not isinstance(value, BaseException)
            for value in abort_results[:len(be_to_abort)])
        result.rt_aborted_on_stop = sum(
            not isinstance(value, BaseException)
            for value in abort_results[len(be_to_abort):])

    pending_be_tasks: set[asyncio.Task] = set()
    if args.strict_stop:
        pending = [task for task in be_tasks + rt_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    else:
        _, pending_be_tasks = await asyncio.wait(
            be_tasks, timeout=max(0.0, args.shutdown_grace_s))
        for task in pending_be_tasks:
            task.cancel()
        # Legacy behavior: allow RT submitted in-window to drain, capped by
        # max-wall-s. New fixed-duration sweeps always use --strict-stop.
        if rt_tasks:
            timeout = max(1.0, hard_deadline - time.perf_counter())
            try:
                await asyncio.wait_for(
                    asyncio.gather(*rt_tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                for task in rt_tasks:
                    task.cancel()
                await asyncio.gather(*rt_tasks, return_exceptions=True)
        if pending_be_tasks:
            _, stuck = await asyncio.wait(pending_be_tasks, timeout=5.0)
            if stuck:
                print(f"[cleanup] {len(stuck)} BE tasks still cancelling", flush=True)

    if debug and step_log:
        be_chunks = [c for s in step_log for (_r, c, _cc, _pl)
                     in s["be_detail"]]
        if be_chunks:
            from collections import Counter
            top = Counter(be_chunks).most_common(6)
            steps_with_be = sum(1 for s in step_log if s["be_reqs"] > 0)
            print(
                f"[BE chunk stats] steps_with_be={steps_with_be} "
                f"be_chunk_samples={len(be_chunks)} "
                f"min={min(be_chunks)} max={max(be_chunks)} "
                f"mean={sum(be_chunks) / len(be_chunks):.1f} "
                f"top_values={top}",
                flush=True,
            )

    # Drain the engine so the next RPS point reusing this same engine starts
    # from a clean scheduler state (no leftover BE prefills or KV residency).
    try:
        for _ in range(50):
            leftover: set[str] = set()
            for sched in engine.engine.scheduler:
                for q in (sched.waiting, sched.running,
                          getattr(sched, "swapped", [])):
                    for sg in list(q):
                        leftover.add(str(sg.request_id))
            if not leftover:
                break
            for rid in leftover:
                try:
                    await engine.abort(rid)
                except Exception:
                    pass
            await asyncio.sleep(0.1)
    except Exception:
        pass

    result.window_s = (send_window_s if fixed_requests
                       else max(send_window_s - args.warmup_s, 0.0))
    result.be_decode_tokens_processed = step_stats["be_decode_tokens"]
    result.be_prefill_tokens_processed = step_stats["be_prefill_tokens"]
    result.scheduler_steps = int(step_stats["steps"])
    result.scheduler_tokens = int(step_stats["tokens"])
    result.scheduler_preempted = int(step_stats["preempted"])
    result.scheduler_swap_out_blocks = int(step_stats["swap_out_blocks"])
    result.scheduler_batch_tokens = list(step_stats["batch_tokens"])
    result.stop_overrun_ms = max(
        0.0, (time.perf_counter() - meas_end) * 1000.0)

    # Fractional request equivalents at the hard boundary. A completed
    # prefill contributes 1.0; an unfinished prefill contributes its current
    # computed-token fraction. Decode uses observed output tokens directly.
    for rid in measured_request_ids:
        kind, prompt_len, output_len = request_shapes[rid]
        if rid in direct_decode_ids:
            # No real prefill was executed for direct-decode admissions.
            prefill_fraction = 0.0
        elif rid in prefill_complete_ids:
            prefill_fraction = 1.0
        else:
            prefill_fraction = min(
                current_prefill_tokens.get(rid, 0) / max(prompt_len, 1), 1.0)
        decode_fraction = min(
            output_tokens_by_request.get(rid, 0) / max(output_len, 1), 1.0)
        if kind == "rt":
            result.rt_prefill_req_equiv += prefill_fraction
            result.rt_decode_req_equiv += decode_fraction
            if rid in rt_records:
                record = rt_records[rid]
                record["prefill_fraction"] = prefill_fraction
                record["decode_fraction"] = decode_fraction
        else:
            result.be_prefill_req_equiv += prefill_fraction
            result.be_decode_req_equiv += decode_fraction

    for record in rt_records.values():
        ttft_pass = (
            record["ttft_ms"] is not None
            and record["ttft_ms"] <= args.ttft_slo_ms)
        # Strict request-level TBT attainment requires completion; an
        # unfinished request is right-censored and counted as not attained.
        tbt_pass = (
            record["completed"]
            and all(value <= args.tbt_slo_ms
                    for value in record["tbt_ms"]))
        record["ttft_slo_attained"] = ttft_pass
        record["tbt_slo_attained"] = tbt_pass
        record["joint_slo_attained"] = ttft_pass and tbt_pass
        result.rt_ttft_slo_passed += int(ttft_pass)
        result.rt_tbt_slo_passed += int(tbt_pass)
        result.rt_joint_slo_passed += int(ttft_pass and tbt_pass)
    result.rt_violations = result.rt_injected - result.rt_ttft_slo_passed
    result.rt_output_tokens = sum(
        output_tokens_by_request.get(rid, 0)
        for rid in measured_request_ids
        if request_shapes[rid][0] == "rt")
    result.be_output_tokens = sum(
        output_tokens_by_request.get(rid, 0)
        for rid in measured_request_ids
        if request_shapes[rid][0] == "be")
    compact_rt_records: list[dict[str, Any]] = []
    for record in rt_records.values():
        compact = dict(record)
        samples = compact.pop("tbt_ms")
        compact["tbt_ms"] = {
            key: round(float(value), 3)
            for key, value in summarize(samples).items()
        }
        compact["tbt_violations"] = sum(
            value > args.tbt_slo_ms for value in samples)
        compact_rt_records.append(compact)
    result.rt_requests = compact_rt_records
    result.ended_at = datetime.now().astimezone().isoformat()
    return result


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _result_to_dict(
    r: RunResult,
    args: argparse.Namespace,
    rt_scale_rps: Optional[float] = None,
) -> dict[str, Any]:
    be_rps = r.be_completed / r.window_s if r.window_s > 0 else 0.0
    be_prefill_tok = r.be_prefill_tokens / r.window_s if r.window_s > 0 else 0.0
    be_processed_tok = (
        (r.be_prefill_tokens_processed + r.be_decode_tokens_processed)
        / r.window_s if r.window_s > 0 else 0.0)
    be_prefill_processed_tok = (
        r.be_prefill_tokens_processed / r.window_s
        if r.window_s > 0 else 0.0)
    be_decode_processed_tok = (
        r.be_decode_tokens_processed / r.window_s
        if r.window_s > 0 else 0.0)
    be_output_tok = r.be_output_tokens / r.window_s if r.window_s > 0 else 0.0
    # Exact useful progress at the stop boundary. This uses every request's
    # real prompt/output denominator, including unfinished requests, instead
    # of dividing aggregate tokens by a fixed average request length.
    be_prefill_req_per_s = (
        r.be_prefill_req_equiv / r.window_s
        if r.window_s > 0 else 0.0)
    be_decode_req_per_s = (
        r.be_decode_req_equiv / r.window_s
        if r.window_s > 0 else 0.0)
    be_goodput_req_per_s = min(be_prefill_req_per_s, be_decode_req_per_s)
    ttft_attainment = (
        r.rt_ttft_slo_passed / r.rt_injected if r.rt_injected else 0.0)
    tbt_request_attainment = (
        r.rt_tbt_slo_passed / r.rt_injected if r.rt_injected else 0.0)
    joint_attainment = (
        r.rt_joint_slo_passed / r.rt_injected if r.rt_injected else 0.0)
    viol_rate = 1.0 - ttft_attainment if r.rt_injected else 0.0
    ttft = summarize(r.rt_ttft_ms)
    rt_service = summarize(r.rt_pred_service_ms)
    rt_queue = summarize(r.rt_queue_ms)
    rt_tbt = summarize(r.rt_tbt_ms)
    tbt_token_viol_rate = (
        r.rt_tbt_violations / len(r.rt_tbt_ms) if r.rt_tbt_ms else 0.0)
    return {
        "rt_scale_rps": rt_scale_rps,
        "max_num_batched_tokens": r.max_num_batched_tokens,
        "window_s": round(r.window_s, 3),
        "started_at": r.started_at,
        "ended_at": r.ended_at,
        "be_throughput_req_per_s": round(be_rps, 4),
        "be_goodput_req_per_s": round(be_goodput_req_per_s, 4),
        "be_prefill_req_per_s": round(be_prefill_req_per_s, 4),
        "be_decode_req_per_s": round(be_decode_req_per_s, 4),
        "be_throughput_tok_s": round(be_prefill_tok, 1),
        "be_throughput_prefill_tok_s": round(be_prefill_tok, 1),
        "be_throughput_processed_tok_s": round(be_processed_tok, 1),
        "be_throughput_prefill_processed_tok_s":
            round(be_prefill_processed_tok, 1),
        "be_throughput_decode_processed_tok_s":
            round(be_decode_processed_tok, 1),
        "be_throughput_output_tok_s": round(be_output_tok, 1),
        "be_started": r.be_started,
        "be_completed": r.be_completed,
        "be_prefill_req_equiv": round(r.be_prefill_req_equiv, 6),
        "be_decode_req_equiv": round(r.be_decode_req_equiv, 6),
        "be_prefill_tokens": r.be_prefill_tokens,
        "be_prefill_tokens_processed": r.be_prefill_tokens_processed,
        "be_decode_tokens_processed": r.be_decode_tokens_processed,
        "be_output_tokens": r.be_output_tokens,
        "be_inflight_at_stop": r.be_inflight_at_stop,
        "be_aborted_on_stop": r.be_aborted_on_stop,
        "rt_injected": r.rt_injected,
        "rt_first_token": r.rt_first_token,
        "rt_completed": r.rt_completed,
        "rt_aborted_on_stop": r.rt_aborted_on_stop,
        "rt_drain_s": round(r.rt_drain_s, 6),
        "rt_drain_timed_out": r.rt_drain_timed_out,
        "rt_measurement_scope": "all_selected_requests" if args.fixed_request_run else "time_window",
        "rt_output_tokens": r.rt_output_tokens,
        "rt_prefill_req_equiv": round(r.rt_prefill_req_equiv, 6),
        "rt_decode_req_equiv": round(r.rt_decode_req_equiv, 6),
        "rt_violations": r.rt_violations,
        "rt_violation_rate": round(viol_rate, 4),
        "rt_ttft_slo_passed": r.rt_ttft_slo_passed,
        "rt_tbt_slo_passed": r.rt_tbt_slo_passed,
        "rt_joint_slo_passed": r.rt_joint_slo_passed,
        "rt_ttft_slo_attainment": round(ttft_attainment, 4),
        "rt_tbt_slo_attainment": round(tbt_request_attainment, 4),
        "rt_slo_attainment": round(joint_attainment, 4),
        "rt_tbt_violations": r.rt_tbt_violations,
        "rt_tbt_token_violation_rate": round(tbt_token_viol_rate, 4),
        "rt_tbt_token_slo_attainment": round(
            1.0 - tbt_token_viol_rate, 4),
        "rt_ttft_ms": {k: round(float(v), 3) for k, v in ttft.items()},
        "rt_tbt_ms": {k: round(float(v), 3) for k, v in rt_tbt.items()},
        "rt_pred_service_ms": {
            k: round(float(v), 3) for k, v in rt_service.items()
        },
        "rt_queue_ms": {k: round(float(v), 3) for k, v in rt_queue.items()},
        "scheduler": {
            "steps": r.scheduler_steps,
            "scheduled_tokens": r.scheduler_tokens,
            "mean_tokens_per_step": round(
                r.scheduler_tokens / r.scheduler_steps
                if r.scheduler_steps else 0.0, 3),
            "batch_tokens": {
                k: round(float(v), 3)
                for k, v in summarize(r.scheduler_batch_tokens).items()
            },
            "preempted": r.scheduler_preempted,
            "swap_out_block_events": r.scheduler_swap_out_blocks,
        },
        "stop_overrun_ms": round(r.stop_overrun_ms, 3),
        "rt_requests": r.rt_requests,
    }


def _csv_row(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten the headline metrics while JSON retains full diagnostics."""
    ttft = payload["rt_ttft_ms"]
    tbt = payload["rt_tbt_ms"]
    return {
        "rt_scale_rps": payload["rt_scale_rps"],
        "chunk_size": payload["max_num_batched_tokens"],
        "rt_injected": payload["rt_injected"],
        "rt_first_token": payload["rt_first_token"],
        "rt_completed": payload["rt_completed"],
        "rt_slo_attainment": payload["rt_slo_attainment"],
        "rt_ttft_slo_attainment": payload["rt_ttft_slo_attainment"],
        "rt_tbt_slo_attainment": payload["rt_tbt_slo_attainment"],
        "ttft_p50_ms": ttft["p50"],
        "ttft_p90_ms": ttft["p90"],
        "ttft_p99_ms": ttft["p99"],
        "ttft_max_ms": ttft["max"],
        "tbt_p50_ms": tbt["p50"],
        "tbt_p90_ms": tbt["p90"],
        "tbt_p99_ms": tbt["p99"],
        "tbt_max_ms": tbt["max"],
        "rt_prefill_req_equiv": payload["rt_prefill_req_equiv"],
        "rt_decode_req_equiv": payload["rt_decode_req_equiv"],
        "be_completed": payload["be_completed"],
        "be_prefill_req_equiv": payload["be_prefill_req_equiv"],
        "be_decode_req_equiv": payload["be_decode_req_equiv"],
        "be_goodput_req_per_s": payload["be_goodput_req_per_s"],
        "be_prefill_req_per_s": payload["be_prefill_req_per_s"],
        "be_decode_req_per_s": payload["be_decode_req_per_s"],
        "be_prefill_processed_tok_s":
            payload["be_throughput_prefill_processed_tok_s"],
        "be_decode_processed_tok_s":
            payload["be_throughput_decode_processed_tok_s"],
        "scheduler_preempted": payload["scheduler"]["preempted"],
        "scheduler_mean_tokens_per_step":
            payload["scheduler"]["mean_tokens_per_step"],
        "stop_overrun_ms": payload["stop_overrun_ms"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark RT-BE colocation on a real P-pool GPU.")

    m = p.add_argument_group("model and engine")
    m.add_argument("--model", required=True)
    m.add_argument("--dtype", default="auto")
    m.add_argument("--load-format", default="auto")
    m.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    m.add_argument("--max-model-len", type=int, default=None)
    m.add_argument("--max-num-seqs", type=int, default=256)
    m.add_argument("--disable-inductor", action="store_true", default=True)
    m.add_argument("--runtime-profile", default="")
    m.add_argument("--tensor-parallel-size", type=int, default=1)
    m.add_argument("--be-chunk-size-follows-mnbt", action="store_true",
                   help="Set p_pool_be_chunk_size = max_num_batched_tokens "
                        "for each swept point, so the fixed BE chunk tracks "
                        "the swept value.")

    pp = p.add_argument_group("scheduling policy")
    pp.add_argument("--ttft-slo-ms", type=float, default=1000.0)
    pp.add_argument("--tbt-slo-ms", type=float, default=150.0)
    pp.add_argument("--rt-prefill-deadline-s", type=float, default=3600.0)
    pp.add_argument("--enable-p-pool-saturation", action="store_true")
    pp.add_argument("--p-pool-be-chunk-ms", type=float, default=100.0)
    pp.add_argument("--p-pool-be-chunk-size", type=int, default=0,
                    help="Fixed BE prefill tokens per chunk in P-pool "
                         "saturation. When > 0, overrides --p-pool-be-chunk-ms "
                         "(the two are mutually exclusive).")
    pp.add_argument("--p-pool-preemption-margin", type=float, default=0.0)
    pp.add_argument(
        "--enable-decode-saturation", action="store_true",
        help="Saturo D-pool decode saturation: RT decode is protected by "
             "the TBT SLO while BE work fills the spare KV/compute budget. "
             "Requires --full-generation-workload.")
    pp.add_argument(
        "--request-mode", choices=("split-phase", "local"), default="local",
        help="Decode-saturation only. 'split-phase': RT and the decode-ready "
             "half of the BE population enter direct decode admission "
             "(PD-disaggregated D-node semantics, no real prefill); 'local': "
             "ordinary end-to-end prefill->decode requests.")
    pp.add_argument(
        "--saturation-strategy", choices=("oracle", "progressive", "greedy"),
        default="greedy",
        help="Decode-pool saturation forecast provider. 'oracle' consumes an "
             "explicit future-RT schedule generated per (chunk, rps) point "
             "(the engine is rebuilt per point); 'greedy' uses current state "
             "only (zero look-ahead).")
    pp.add_argument(
        "--oracle-lookahead-s", type=float, default=60.0,
        help="Future-RT horizon for oracle/progressive decode saturation.")
    pp.add_argument(
        "--be-decode-block-fraction", type=float, default=0.95,
        help="Watermark fraction of physical KV blocks the BE-decode budget "
             "may fill; the rest is reserved for the RT peak and the longest "
             "BE-prefill input.")
    pp.add_argument(
        "--kv-block-size", type=int, default=16,
        help="KV cache block size used to convert the longest BE-prefill "
             "prompt into a reserved block count (split-phase only).")
    pp.add_argument(
        "--enable-be-decode", action=argparse.BooleanOptionalAction,
        default=True,
        help="Split-phase only: split the BE population 50/50 between "
             "decode-ready and prefill-only requests. Disable for a "
             "prefill-only BE population.")

    w = p.add_argument_group("workload and sweep")
    w.add_argument("--rt-workload-csv", required=True)
    w.add_argument("--be-workload-csv", required=True)
    w.add_argument("--num-req", type=int, required=True)
    w.add_argument(
        "--target-rps", required=True,
        help="Comma-separated RT RPS values. Every point reuses the same "
             "fixed request prefix.")
    w.add_argument(
        "--max-num-batched-tokens-list", required=True,
        help="Comma-separated max_num_batched_tokens values (BE chunk size "
             "in tokens), e.g. 256,512,1024,2048,4096.")
    w.add_argument("--fixed-request-run", action="store_true",
                   help="Measure every selected RT request through completion; "
                        "warmup completes separate short-output RT/BE batches.")
    w.add_argument("--warmup-s", type=float, default=10.0)
    w.add_argument(
        "--duration-s", type=float, default=0.0,
        help="Fixed wall-clock duration of each RPS/chunk point. When > 0, "
             "the RT prefix is sized automatically for this duration.")
    w.add_argument(
        "--strict-stop", action="store_true",
        help="At the point boundary abort RT and BE immediately; unfinished "
             "requests are retained as fractional progress and strict-SLO "
             "misses instead of being drained.")
    w.add_argument("--max-wall-s", type=float, default=1200.0)
    w.add_argument("--shutdown-grace-s", type=float, default=30.0)
    w.add_argument("--be-inflight", type=int, default=64)
    w.add_argument("--max-source-be", type=int, default=0)
    w.add_argument("--be-input-drop-frac", type=float, default=0.0)
    w.add_argument("--be-output-keep-frac", type=float, default=None)
    w.add_argument("--be-output-max-len", type=int, default=None)
    w.add_argument(
        "--full-generation-workload",
        action="store_true",
        help="Preserve trace output lengths for both RT and BE so the run "
             "contains RT/BE prefill and decode. Without this flag the legacy "
             "prefill-only limitation-study behaviour is retained.")

    hw = p.add_argument_group("hardware")
    hw.add_argument("--gpu-ids", default="")
    hw.add_argument("--pipeline-parallel-size", type=int, default=1)

    p.add_argument("--output-json", default="")
    p.add_argument(
        "--resume", action="store_true",
        help="Resume an existing --output-json, skipping completed points.")
    args = p.parse_args()
    enabled_modes = [
        name for name, flag in (
            ("--enable-p-pool-saturation", args.enable_p_pool_saturation),
            ("--enable-decode-saturation", args.enable_decode_saturation),
        ) if flag
    ]
    if len(enabled_modes) > 1:
        p.error(" and ".join(enabled_modes) + " are mutually exclusive")
    if args.enable_decode_saturation and not args.full_generation_workload:
        p.error("--enable-decode-saturation requires "
                "--full-generation-workload (BE/RT need real output lengths "
                "for decode-side work)")
    if args.request_mode == "split-phase" and not args.enable_decode_saturation:
        p.error("--request-mode split-phase requires "
                "--enable-decode-saturation")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_chunk(
    args: argparse.Namespace,
    chunk_tokens: int,
    rps_points: List[Optional[float]],
    rt_path: Path,
    be_trace_raw: List[TraceRequest],
    all_runs: List[dict[str, Any]],
    csv_writer: Optional[Any] = None,
    csv_fh: Optional[Any] = None,
    checkpoint_path: Optional[Path] = None,
) -> None:
    """Build one AsyncLLMEngine and sweep all RPS points on it."""
    engine: Any = None
    try:
        engine = _build_async_engine(args, chunk_tokens)
        core = engine.engine
        max_model_len = int(core.model_config.max_model_len)
        be_trace = _sanitize(be_trace_raw, max_model_len)
        cache = core.cache_config
        eff_be_chunk = getattr(
            core.scheduler_config, "p_pool_be_chunk_size", 0)
        print(
            f"Engine ready (chunk={chunk_tokens}): "
            f"num_gpu_blocks={cache.num_gpu_blocks}, "
            f"block_size={cache.block_size}, max_model_len={max_model_len}, "
            f"eff_p_pool_be_chunk_size={eff_be_chunk}",
            flush=True,
        )

        async def _run_all_rps() -> None:
            for rps in rps_points:
                if any(
                    int(run.get("max_num_batched_tokens", -1)) == chunk_tokens
                    and math.isclose(
                        float(run.get("rt_scale_rps", -1.0)), float(rps),
                        rel_tol=0.0, abs_tol=1e-9)
                    for run in all_runs
                ):
                    print(f"  -- chunk={chunk_tokens} rps={rps} already "
                          "checkpointed; skipping --", flush=True)
                    continue
                point_num_req, send_window_s = _point_num_req(args, rps)
                rt_trace_raw = _load_trace(
                    rt_path, kind="rt", num_req=point_num_req,
                    target_rps=rps,
                )
                rt_trace = _sanitize(rt_trace_raw, max_model_len)
                print(
                    f"  -- chunk={chunk_tokens} rps={rps} "
                    f"RT={len(rt_trace)} "
                    f"span={rt_trace[-1].arrival_offset_s:.1f}s --",
                    flush=True,
                )
                if args.fixed_request_run:
                    if len(rt_trace) != args.num_req or any(
                            (a.request_id, a.prompt_len, a.output_len) !=
                            (b.request_id, b.prompt_len, b.output_len)
                            for a, b in zip(rt_trace, rt_trace_raw)):
                        raise ValueError("Selected RT requests were filtered or truncated")
                    if args.warmup_s > 0:
                        print("[fixed-request] independent complete-batch warmup", flush=True)
                        await _warmup_fixed_requests(engine, rt_trace, be_trace, args)
                        print("[fixed-request] warmup cleared; starting measured prefix", flush=True)
                result = await _run_one_async(
                    engine, rt_trace, be_trace, chunk_tokens, args,
                    send_window_s)
                payload = _result_to_dict(result, args, rps)
                all_runs.append(payload)
                if checkpoint_path is not None:
                    partial = {
                        "status": "running",
                        "updated_at": datetime.now().astimezone().isoformat(),
                        "runs": all_runs,
                    }
                    tmp_path = checkpoint_path.with_suffix(
                        checkpoint_path.suffix + ".tmp")
                    tmp_path.write_text(
                        json.dumps(partial, indent=2), encoding="utf-8")
                    os.replace(tmp_path, checkpoint_path)
                if csv_writer is not None:
                    csv_writer.writerow(_csv_row(payload))
                    if csv_fh is not None:
                        csv_fh.flush()
                print(
                    f"     -> rps={rps} "
                    f"attainment={payload['rt_slo_attainment']:.3f} "
                    f"p95_ttft={payload['rt_ttft_ms']['p95']:.0f}ms "
                    f"be_tput(completed)={payload['be_throughput_prefill_tok_s']:.0f} "
                    f"be_tput(processed)={payload['be_throughput_processed_tok_s']:.0f} "
                    f"tok/s",
                    flush=True,
                )

        asyncio.run(_run_all_rps())
    finally:
        if engine is not None:
            try:
                engine.shutdown_background_loop()
            except Exception:
                pass
            del engine


def main() -> None:
    args = parse_args()
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if args.disable_inductor:
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    if args.enable_decode_saturation:
        # The decode-saturation manager reads these at engine construction
        # (see saturo.saturation.decode_saturation_core); AsyncEngineArgs
        # does not expose them directly.
        os.environ["SATURO_DECODE_BLOCK_FRACTION"] = str(
            args.be_decode_block_fraction)

    chunk_tokens_points = [
        int(x) for x in str(args.max_num_batched_tokens_list).split(",")
        if x.strip()
    ]
    if not chunk_tokens_points:
        raise ValueError("--max-num-batched-tokens-list had no values")

    if args.num_req <= 0:
        raise ValueError("--num-req must be positive")
    rps_points = [
        float(x) for x in str(args.target_rps).split(",") if x.strip()]
    if not rps_points or min(rps_points) <= 0:
        raise ValueError("--target-rps must contain positive values")
    from saturo.tools.trace_scaler import fixed_num_window_s
    point_windows = [
        args.duration_s if args.duration_s > 0
        else fixed_num_window_s(args.num_req, rps)
        for rps in rps_points
    ]
    if args.fixed_request_run:
        if args.duration_s != 0 or args.strict_stop:
            raise ValueError("--fixed-request-run requires duration-s=0 and no strict-stop")
        if args.max_wall_s <= max(point_windows):
            raise ValueError("--max-wall-s must allow drain after the sending window")
    if args.duration_s < 0:
        raise ValueError("--duration-s cannot be negative")
    if not args.fixed_request_run and any(args.warmup_s >= window for window in point_windows):
        raise ValueError(
            "--warmup-s must be smaller than each point duration")

    rt_path = Path(args.rt_workload_csv).expanduser().resolve()
    be_path = Path(args.be_workload_csv).expanduser().resolve()

    be_trace_raw = _load_trace(be_path, kind="be")
    if args.max_source_be > 0:
        be_trace_raw = be_trace_raw[:args.max_source_be]
    be_trace_raw = _filter_be(be_trace_raw, args)
    if args.enable_decode_saturation:
        # Reserve KV blocks for the longest prefill-only BE prompt so the
        # BE-decode budget cannot starve BE prefill (split-phase only; local
        # lifecycles manage their own prefill KV like any vLLM request).
        if args.request_mode == "split-phase" and args.be_inflight > 0:
            max_be_prompt = max(r.prompt_len for r in be_trace_raw)
            reserved_blocks = math.ceil(
                max_be_prompt / max(int(args.kv_block_size), 1))
        else:
            reserved_blocks = 0
        os.environ["SATURO_DECODE_BE_PREFILL_RESERVE_BLOCKS"] = str(
            reserved_blocks)
        print(
            f"[decode-saturation] request_mode={args.request_mode} "
            f"strategy={args.saturation_strategy} "
            f"block_fraction={args.be_decode_block_fraction} "
            f"reserved_be_prefill_blocks={reserved_blocks}",
            flush=True)
    print(
        f"Loaded {len(be_trace_raw)} BE rows "
        f"(full_generation={args.full_generation_workload}). "
        f"Sweeping {len(chunk_tokens_points)} chunk x {len(rps_points)} rps "
        f"= {len(chunk_tokens_points) * len(rps_points)} points.",
        flush=True,
    )

    out_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else Path(
            f"saturo/results/rt_be_colocate/{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_runs: List[dict[str, Any]] = []
    if args.resume and out_path.is_file():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        all_runs = list(previous.get("runs", []))
        print(f"[resume] loaded {len(all_runs)} completed points from "
              f"{out_path}", flush=True)
    csv_path = out_path.with_suffix(".csv")
    csv_fields = [
        "rt_scale_rps", "chunk_size", "rt_injected", "rt_first_token",
        "rt_completed", "rt_slo_attainment", "rt_ttft_slo_attainment",
        "rt_tbt_slo_attainment", "ttft_p50_ms", "ttft_p90_ms",
        "ttft_p99_ms", "ttft_max_ms", "tbt_p50_ms", "tbt_p90_ms",
        "tbt_p99_ms", "tbt_max_ms", "rt_prefill_req_equiv",
        "rt_decode_req_equiv", "be_completed", "be_prefill_req_equiv",
        "be_decode_req_equiv", "be_goodput_req_per_s",
        "be_prefill_req_per_s", "be_decode_req_per_s",
        "be_prefill_processed_tok_s", "be_decode_processed_tok_s",
        "scheduler_preempted", "scheduler_mean_tokens_per_step",
        "stop_overrun_ms",
    ]
    csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
    csv_writer.writeheader()
    for prior_run in all_runs:
        csv_writer.writerow(_csv_row(prior_run))
    csv_fh.flush()

    try:
        # The oracle provider anchors its future-RT schedule at engine
        # construction, so oracle points cannot share an engine across RPS
        # values: rebuild per (chunk, rps) point instead of per chunk.
        oracle_mode = (
            args.enable_decode_saturation
            and args.saturation_strategy == "oracle")

        def _point_done(chunk_tokens: int, rps: float) -> bool:
            return any(
                int(run.get("max_num_batched_tokens", -1)) == chunk_tokens
                and math.isclose(
                    float(run.get("rt_scale_rps", -1.0)), float(rps),
                    rel_tol=0.0, abs_tol=1e-9)
                for run in all_runs
            )

        for chunk_tokens in chunk_tokens_points:
            chunk_done = all(
                _point_done(chunk_tokens, rps) for rps in rps_points)
            if chunk_done:
                print(f"[resume] chunk={chunk_tokens} is complete; "
                      "skipping engine creation", flush=True)
                continue
            print(
                f"\n{'='*60}\n"
                f"  max_num_batched_tokens = {chunk_tokens}\n"
                f"{'='*60}",
                flush=True,
            )
            point_groups = (
                [[rps] for rps in rps_points] if oracle_mode
                else [rps_points])
            for group in point_groups:
                if oracle_mode:
                    rps = group[0]
                    if _point_done(chunk_tokens, rps):
                        print(f"[resume] chunk={chunk_tokens} rps={rps} "
                              "already checkpointed; skipping", flush=True)
                        continue
                    schedule_path = _write_oracle_schedule(
                        args, rt_path, rps, out_path, chunk_tokens)
                    os.environ["SATURO_DECODE_ORACLE_SCHEDULE"] = str(
                        schedule_path)
                    print(f"[oracle] wrote RT schedule: {schedule_path}",
                          flush=True)
                try:
                    _run_chunk(
                        args, chunk_tokens, group, rt_path, be_trace_raw,
                        all_runs, csv_writer, csv_fh, out_path)
                finally:
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
    finally:
        csv_fh.close()

    # Write output
    output = {
        "config": {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "load_format": args.load_format,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "ttft_slo_ms": args.ttft_slo_ms,
            "tbt_slo_ms": args.tbt_slo_ms,
            "enable_p_pool_saturation": args.enable_p_pool_saturation,
            "enable_decode_saturation": args.enable_decode_saturation,
            "request_mode": args.request_mode,
            "saturation_strategy": args.saturation_strategy,
            "oracle_lookahead_s": args.oracle_lookahead_s,
            "be_decode_block_fraction": args.be_decode_block_fraction,
            "enable_be_decode": args.enable_be_decode,
            "kv_block_size": args.kv_block_size,
            "p_pool_be_chunk_ms": args.p_pool_be_chunk_ms,
            "be_chunk_size_follows_mnbt": args.be_chunk_size_follows_mnbt,
            "p_pool_preemption_margin": args.p_pool_preemption_margin,
            "warmup_s": args.warmup_s,
            "duration_s": args.duration_s,
            "fixed_request_run": args.fixed_request_run,
            "warmup_mode": "complete_batches_16rt_16be_output16" if args.fixed_request_run else "in_window",
            "strict_stop": args.strict_stop,
            "shutdown_grace_s": args.shutdown_grace_s,
            "be_inflight": args.be_inflight,
            "full_generation_workload": args.full_generation_workload,
            "rt_workload_csv": str(rt_path),
            "be_workload_csv": str(be_path),
            "runtime_profile": args.runtime_profile or "none",
            "num_req": args.num_req,
            "target_rps": rps_points,
            "max_num_batched_tokens_list": chunk_tokens_points,
        },
        "runs": all_runs,
    }
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[Output] wrote {out_path}\n[Output] wrote {csv_path}",
          flush=True)


if __name__ == "__main__":
    main()
