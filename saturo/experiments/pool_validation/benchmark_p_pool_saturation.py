#!/usr/bin/env python3
"""Benchmark a single Saturo P-pool (prefill) instance under BE saturation.

This mirrors what ``saturo/simulators/prefill_simulator`` predicts: one P-pool
engine serves RT prefill requests whose dataset arrival timeline is scaled to
a target per-instance RPS (the TTFT-SLO metric), while
``enable_p_pool_saturation`` fills the spare capacity
with best-effort (BE) work (chunked BE prefill + BE decode) up to the
``p_pool_be_chunk_ms`` budget. The benchmark measures the *achieved* BE
throughput (completed BE requests / s) and the RT TTFT-SLO violation rate, so
the simulator's predicted ``saturation_req_per_s`` can be validated on real
hardware.

Unlike ``benchmark_rt_decode_only.py`` (an offline decode burst that ignores
arrival timestamps), RT here is injected on a real clock at the scaled dataset
arrival timestamps so queueing and TTFT match the simulator's FIFO model.
Multiple RPS points can be passed to ``--target-rps`` (comma-separated); the
model is loaded once and each point is measured in turn.

Example::

    CUDA_VISIBLE_DEVICES=0 python \
        saturo/experiments/pool_validation/benchmark_p_pool_saturation.py \
        --model /root/models/Qwen/Qwen2.5-72B-Instruct \
        --rt-workload-csv data/datasets/rt_workload.csv \
        --be-workload-csv data/datasets/be_workload.csv \
        --runtime-profile saturo/results/time_predictors/.../time_predictor_...tp1....json \
        --tensor-parallel-size 1 --load-format dummy \
        --ttft-slo-ms 1000 --p-pool-be-chunk-ms 113 \
        --num-req 600 --target-rps 4.0,4.4,4.7 --warmup-s 10
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    prompt_len: int
    output_len: int
    arrival_offset_s: float = 0.0


@dataclass
class RunResult:
    target_rps: float
    avg_be_input_len: float = 1.0
    avg_be_output_len: float = 1.0
    # RT
    rt_injected: int = 0
    rt_first_token: int = 0
    rt_ttft_ms: list[float] = field(default_factory=list)
    rt_violations: int = 0
    rt_settlement_s: float = 0.0
    # BE
    be_completed: int = 0
    be_output_tokens: int = 0
    be_prefill_tokens: int = 0
    be_decode_tokens: int = 0
    be_dummy_handoff_tokens: int = 0
    be_prefill_completed: int = 0
    be_decode_completed: int = 0
    # diagnostics
    decode_batch_sizes: list[float] = field(default_factory=list)
    be_prefill_chunk_sizes: list[float] = field(default_factory=list)
    be_chunk_block_util: list[float] = field(default_factory=list)
    be_chunk_used_blocks: list[float] = field(default_factory=list)
    num_gpu_blocks_total: int = 0
    be_chunk_actual_ms: list[float] = field(default_factory=list)
    be_chunk_predicted_ms: list[float] = field(default_factory=list)
    scheduler_wall_ms: list[float] = field(default_factory=list)
    runtime_overhead_ms: list[float] = field(default_factory=list)
    mixed_iterations: int = 0
    pure_prefill_iterations: int = 0
    dummy_handoff_iterations: int = 0
    window_s: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure achieved BE saturation throughput and RT TTFT-SLO "
            "violations on one Saturo P-pool instance."))

    m = p.add_argument_group("model and engine")
    m.add_argument("--model", required=True)
    m.add_argument("--dtype", default="auto")
    m.add_argument("--load-format", default="auto",
                   help="dummy = random weights (profiling only, no weights "
                        "needed); auto = load real weights.")
    m.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    m.add_argument("--max-model-len", type=int, default=None)
    m.add_argument("--max-num-seqs", type=int, default=256,
                   help="Keep high so KV memory, not the seq cap, bounds the "
                        "BE decode batch.")
    m.add_argument("--max-num-batched-tokens", type=int, default=None)
    m.add_argument("--disable-inductor", action=argparse.BooleanOptionalAction,
                   default=True)
    m.add_argument("--runtime-profile", default="",
                   help="Saturo time-predictor JSON so the p_pool saturation "
                        "policy can size BE chunks.")

    pp = p.add_argument_group("p-pool saturation policy")
    pp.add_argument("--ttft-slo-ms", type=float, default=1000.0)
    pp.add_argument("--p-pool-be-chunk-ms", type=float, default=113.0,
                    help="Fallback/fixed BE chunk latency budget. Used for all "
                         "RPS when --chunk-ms-source=fixed, and as the fallback "
                         "when the simulator cannot produce a value.")
    pp.add_argument(
        "--p-pool-be-chunk-size", type=int, default=0,
        help="Fixed BE prefill chunk in tokens. When > 0 the scheduler admits "
             "exactly this many BE prefill tokens per step and skips the "
             "latency (ms) binary search, so --p-pool-be-chunk-ms / "
             "--chunk-ms-source are ignored for chunk sizing. 0 = disabled "
             "(use the ms budget).")
    pp.add_argument(
        "--be-request-mode", choices=["split", "normal"], default="normal",
        help="How BE work is injected. 'split' (explicit legacy mode): the constant BE "
             "population is preloaded as a 50/50 decode-ready + prefill-only "
             "split (decode via DummyConnector), which decouples the P and D "
             "resources. 'normal': send plain BE requests that each run their "
             "full prefill->decode lifecycle in this engine (no dummy decode, "
             "no phase split) -- matches the local BE simulator semantics.")
    pp.add_argument("--chunk-ms-source", choices=["sim", "fixed"],
                    default="sim",
                    help="How to set the BE chunk latency budget. 'sim' runs "
                         "the P-pool prefill simulator once per RPS and uses "
                         "its optimal be_chunk_latency_ms; 'fixed' uses "
                         "--p-pool-be-chunk-ms for every RPS.")
    pp.add_argument("--sim-max-rt-violation-rate", type=float, default=0.01,
                    help="Simulator RT TTFT-SLO violation threshold above which "
                         "an RPS is declared RT-infeasible.")
    pp.add_argument("--sim-plan-json", default="",
                    help="Where to write the per-RPS simulator plan (chunk ms + "
                         "predicted saturation). Default: alongside "
                         "--output-json as <output>.sim_plan.json.")
    pp.add_argument("--p-pool-preemption-margin", type=float, default=0.0,
                    help="Safety margin in seconds subtracted from the RT "
                         "prefill deadline.")
    pp.add_argument(
        "--rt-prefill-deadline-s", type=float, default=None,
        help="Per-request engine deadline. Defaults to --ttft-slo-ms / 1000 "
             "so scheduling and reported TTFT use the same SLO.")
    pp.add_argument(
        "--be-decode-block-fraction", type=float, default=0.95,
        help="Fraction of total GPU KV blocks the decode-ready BE population "
             "may occupy. The decode block limit is "
             "floor(num_gpu_blocks * fraction) minus the longest RT and BE "
             "prefill reservations, leaving a safety margin so decode never "
             "consumes every block and starves BE prefill.")
    pp.add_argument(
        "--be-decode-max-batch", type=int, default=0,
        help="Cap on the number of BE decode-ready requests admitted at once, "
             "so BE decode cannot consume the whole be_chunk_ms budget and "
             "starve BE prefill. 0 (default): in --chunk-ms-source=sim mode use "
             "the simulator's balanced max_be_decode_batch per RPS; otherwise "
             "unlimited. A value > 0 forces a fixed cap for every RPS.")

    w = p.add_argument_group("workload and measurement")
    w.add_argument("--rt-workload-csv", required=True)
    w.add_argument("--be-workload-csv", required=True)
    w.add_argument("--num-req", type=int, required=True)
    w.add_argument("--target-rps", required=True,
                   help="Per-instance RT RPS. Comma-separated for several "
                        "points, e.g. 4.0,4.4,4.7.")
    w.add_argument("--warmup-s", type=float, default=10.0)
    w.add_argument("--max-wall-s", type=float, default=1200.0)
    w.add_argument("--drain-timeout-s", type=float, default=30.0,
                   help="Maximum time to wait for aborted request stream "
                        "tasks between runs. This is intentionally separate "
                        "from the measurement wall-clock limit.")
    w.add_argument("--be-inflight", type=int, default=2048,
                   help="BE submitted/running population kept constant to "
                        "saturate spare capacity. Treated as an effectively "
                        "unbounded BE backlog: keep it well above what the KV "
                        "reservation can admit so the scheduler's block limit "
                        "(not this population) is the binding constraint.")
    w.add_argument("--enable-be-decode", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Preload a decode-ready BE population through "
                        "DummyConnector. When enabled the constant BE "
                        "population is split 50/50 decode/prefill and the "
                        "decode KV block limit is derived from the dataset "
                        "(num_gpu_blocks minus the longest RT and BE prefill). "
                        "When disabled the population is prefill-only.")
    w.add_argument("--be-shuffle-seed", type=int, default=None,
                   help="Deterministically shuffle the saturated BE stream. "
                        "RT trace ordering and arrival timestamps are never "
                        "changed.")
    w.add_argument("--max-source-be", type=int, default=0)
    # BE length filters (mirror trace_loader / dataset knobs; default off).
    w.add_argument("--be-input-drop-frac", type=float, default=0.0)
    w.add_argument("--be-input-max-percentile", type=float, default=None,
                   help="Drop BE requests whose prompt length is above this "
                        "dataset percentile, e.g. 0.95 keeps <= P95.")
    w.add_argument("--be-output-keep-frac", type=float, default=None)
    w.add_argument("--be-output-max-len", type=int, default=None)

    hw = p.add_argument_group("hardware")
    hw.add_argument("--gpu-ids", default="")
    hw.add_argument("--tensor-parallel-size", type=int, default=1)
    hw.add_argument("--pipeline-parallel-size", type=int, default=1)
    hw.add_argument("--expert-parallel-size", type=int, default=0)

    p.add_argument("--output-json", default="",
                   help="Where to write the result JSON. Default: "
                        "experiments/saturo/results/p_pool_sat_<ts>.json")
    args = p.parse_args()
    if args.num_req <= 0:
        p.error("--num-req must be positive")
    if args.p_pool_be_chunk_size > 0:
        # With chunked prefill on, vLLM defaults max_num_batched_tokens to
        # 2048, which silently truncates any larger fixed BE chunk (the
        # scheduler admits min(token budget, chunk_size)). Derive a budget
        # that leaves headroom for RT prefill + BE decode in the same step.
        if args.max_num_batched_tokens is None:
            args.max_num_batched_tokens = max(
                8192, 2 * args.p_pool_be_chunk_size)
            print(f"--max-num-batched-tokens not set; auto-raising to "
                  f"{args.max_num_batched_tokens} so the fixed BE chunk "
                  f"({args.p_pool_be_chunk_size} tok) is not truncated.",
                  flush=True)
        elif args.max_num_batched_tokens < args.p_pool_be_chunk_size:
            p.error(
                f"--max-num-batched-tokens ({args.max_num_batched_tokens}) "
                f"< --p-pool-be-chunk-size ({args.p_pool_be_chunk_size}); "
                "the scheduler would truncate every BE chunk.")
    if args.rt_prefill_deadline_s is None:
        args.rt_prefill_deadline_s = args.ttft_slo_ms / 1000.0
    return args


def default_output_path() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "experiments" / "saturo" / "results" /
            f"p_pool_sat_{stamp}.json")


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    rank = int(math.ceil(q * len(ordered))) - 1
    return ordered[min(len(ordered) - 1, max(0, rank))]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(v) for v in values]
    if not data:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0,
                "p99": 0.0, "max": 0.0}
    return {"count": len(data), "mean": statistics.fmean(data),
            "median": statistics.median(data), "p95": percentile(data, 0.95),
            "p99": percentile(data, 0.99), "max": max(data)}


def load_trace(path: Path, kind_const: Any, max_requests: int = 0,
               num_req: int | None = None,
               target_rps: float | None = None,
               ) -> list[TraceRequest]:
    from saturo.runtime.workload import (
        RT,
        load_rt_workload_csv,
        load_workload_csv,
    )

    if not path.is_file():
        raise FileNotFoundError(f"workload CSV not found: {path}")
    if kind_const == RT:
        if num_req is None or target_rps is None:
            raise ValueError("RT loading requires num_req and target_rps")
        loaded = load_rt_workload_csv(
            str(path), num_req=num_req, target_rps=target_rps)
    else:
        loaded = load_workload_csv(str(path), kind_const)
    if max_requests > 0:
        loaded = loaded[:max_requests]
    trace = [TraceRequest(str(r.request_id), int(r.prompt_len),
                          max(1, int(r.output_len)),
                          float(r.arrival_offset_s)) for r in loaded]
    if not trace:
        raise ValueError(f"workload CSV is empty: {path}")
    return trace


def filter_be(trace: list[TraceRequest], args: argparse.Namespace
              ) -> list[TraceRequest]:
    """Apply the same BE length filters as trace_loader.load_be_trace."""
    items = list(trace)
    args.be_input_max_tokens_applied = None
    if args.be_input_max_percentile is not None:
        q = float(args.be_input_max_percentile)
        if not 0.0 < q <= 1.0:
            raise ValueError("--be-input-max-percentile must be in (0, 1]")
        cutoff = int(percentile((r.prompt_len for r in items), q))
        args.be_input_max_tokens_applied = cutoff
        items = [r for r in items if r.prompt_len <= cutoff]
    if args.be_input_drop_frac:
        if not 0.0 < args.be_input_drop_frac < 1.0:
            raise ValueError("--be-input-drop-frac must be in (0, 1)")
        items.sort(key=lambda r: r.prompt_len)
        items = items[int(args.be_input_drop_frac * len(items)):]
    if args.be_output_max_len is not None:
        items = [r for r in items if r.output_len <= args.be_output_max_len]
    elif args.be_output_keep_frac is not None:
        if not 0.0 < args.be_output_keep_frac <= 1.0:
            raise ValueError("--be-output-keep-frac must be in (0, 1]")
        items.sort(key=lambda r: r.output_len)
        items = items[:max(1, int(args.be_output_keep_frac * len(items)))]
    if not items:
        raise ValueError("BE length filter removed all rows")
    return items


def sanitize(trace: list[TraceRequest], max_model_len: int
             ) -> list[TraceRequest]:
    usable: list[TraceRequest] = []
    for r in trace:
        if r.prompt_len >= max_model_len:
            continue
        out = min(r.output_len, max_model_len - r.prompt_len)
        usable.append(TraceRequest(r.request_id, r.prompt_len, max(1, out),
                                   r.arrival_offset_s))
    if not usable:
        raise ValueError(
            f"no request fits max_model_len={max_model_len}")
    return usable


def worker_args(args: argparse.Namespace) -> dict[str, Any]:
    from saturo.runtime.workload import PREFILL

    return {
        "worker_id": 0,
        "worker_role": PREFILL,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "expert_parallel_size": args.expert_parallel_size,
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_format": args.load_format,
        "dtype": args.dtype,
        "rt_prefill_deadline_s": args.rt_prefill_deadline_s,
        "disable_inductor": args.disable_inductor,
        "rt_tbt_slo_ms": 1000.0,  # unused by the prefill role
        "p_pool_be_chunk_ms": args.p_pool_be_chunk_ms,
        "p_pool_be_chunk_size": args.p_pool_be_chunk_size,
        "p_pool_preemption_margin": args.p_pool_preemption_margin,
        "runtime_profile": args.runtime_profile,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        # 'normal' request mode runs each BE request's full prefill->decode
        # lifecycle in-engine, so it must NOT preload a dummy decode split.
        "enable_p_pool_dummy_decode": (
            args.enable_be_decode and args.be_request_mode == "split"),
    }


def create_worker(args: argparse.Namespace) -> Any:
    """Create the same online AsyncLLMEngine worker used by a P-pool node."""
    from saturo.runtime.worker import RayEngineWorker

    return RayEngineWorker(worker_args(args))


def rt_command(req: TraceRequest, serial: int) -> dict[str, Any]:
    from saturo.runtime.workload import PREFILL, RT

    aid = f"rt_{serial}:{req.request_id}"
    return {"cmd": "add", "request_id": aid, "attempt_id": aid, "kind": RT,
            "phase": PREFILL, "prompt_len": req.prompt_len,
            "max_tokens": 1, "base_tokens": 0}


def be_command(req: TraceRequest, serial: int, phase: str,
               full_lifecycle: bool = False) -> dict[str, Any]:
    from saturo.runtime.workload import BE, DECODE

    phase_name = "decode" if phase == DECODE else (
        "normal" if full_lifecycle else "prefill")
    aid = f"be_{phase_name}_{serial}:{req.request_id}"
    # max_tokens semantics:
    # - split/prefill-only: stop on the token sampled by the prefill iteration.
    # - split/decode-ready: preloaded via DummyConnector, run the output length.
    # - normal (full lifecycle): a plain request that prefills then decodes its
    #   full output length in this same engine (no dummy handoff, no split).
    if full_lifecycle:
        max_tokens = req.output_len
    elif phase == DECODE:
        max_tokens = req.output_len
    else:
        max_tokens = 1
    return {"cmd": "add", "request_id": aid, "attempt_id": aid, "kind": BE,
            "phase": phase, "prompt_len": req.prompt_len,
            "max_tokens": max_tokens, "base_tokens": 0}


async def drain_worker(worker: Any, max_wall_s: float) -> None:
    """Abort active requests and wait for their async stream tasks to exit."""
    active = list(worker.active)
    if active:
        await asyncio.gather(*(worker.abort(aid) for aid in active),
                             return_exceptions=True)
    tasks = list(worker._request_tasks.values())
    if tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=max_wall_s,
            )
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    await worker.drain_events()


async def run_one_rps(worker: Any, rt_trace: list[TraceRequest],
                      be_trace: list[TraceRequest], target_rps: float,
                      args: argparse.Namespace) -> RunResult:
    result = RunResult(
        target_rps=target_rps,
        avg_be_input_len=statistics.fmean(
            req.prompt_len for req in be_trace),
        avg_be_output_len=statistics.fmean(
            req.output_len for req in be_trace),
    )
    be_serial = 0
    rt_serial = 0
    rt_idx = 0
    # Optional per-chunk dump: pair the reverse-solved BE chunk size with its
    # predicted vs actual execution time. Gated by SATURO_CHUNK_DUMP_CSV so it
    # is a no-op for normal runs. Line-buffered; flushed/closed at process exit.
    import csv as _csv
    _chunk_csv_path = os.environ.get("SATURO_CHUNK_DUMP_CSV", "")
    _chunk_csv_w = None
    if _chunk_csv_path:
        _new = not (os.path.exists(_chunk_csv_path)
                    and os.path.getsize(_chunk_csv_path) > 0)
        _chunk_csv_f = open(_chunk_csv_path, "a", newline="",
                            encoding="utf-8", buffering=1)
        _chunk_csv_w = _csv.writer(_chunk_csv_f)
        if _new:
            _chunk_csv_w.writerow([
                "target_rps", "be_prefill_tokens", "n_decode_reqs",
                "predicted_time_ms", "actual_time_ms",
                "num_used_gpu_blocks", "num_gpu_blocks_total"])
    # Optional raw per-iteration dump (JSONL) for scheduler-branch diagnosis.
    # Gated by SATURO_ITERLOG_DUMP; a no-op for normal runs.
    _iterlog_path = os.environ.get("SATURO_ITERLOG_DUMP", "")
    _iterlog_f = None
    if _iterlog_path:
        _iterlog_f = open(_iterlog_path, "a", encoding="utf-8", buffering=1)
    from saturo.runtime.workload import DECODE, PREFILL
    be_idx = {PREFILL: 0, DECODE: 0}
    arrival_time: dict[str, float] = {}  # attempt_id -> intended arrival clock
    seen_first: set[str] = set()
    pending_rt: set[str] = set()
    failed_rt: set[str] = set()
    normal_mode = getattr(args, "be_request_mode", "split") == "normal"

    def next_be(phase: str) -> dict[str, Any]:
        nonlocal be_serial
        req = be_trace[be_idx[phase] % len(be_trace)]
        be_idx[phase] += 1
        cmd = be_command(req, be_serial, phase, full_lifecycle=normal_mode)
        be_serial += 1
        return cmd

    # Keep independent, constant populations for both resources. Replacing a
    # completion with the same phase prevents a fast side from draining its
    # half and makes min(prefill throughput, decode throughput) meaningful.
    # 'normal' mode has a single full-lifecycle population (no P/D split).
    if normal_mode:
        decode_inflight = 0
        prefill_inflight = args.be_inflight
    else:
        decode_inflight = (round(args.be_inflight * 0.5)
                           if args.enable_be_decode else 0)
        prefill_inflight = args.be_inflight - decode_inflight
    pending: list[dict[str, Any]] = [
        *(next_be(DECODE) for _ in range(decode_inflight)),
        *(next_be(PREFILL) for _ in range(prefill_inflight)),
    ]

    run_start = time.perf_counter()
    meas_start = run_start + args.warmup_s
    from saturo.tools.trace_scaler import fixed_num_window_s
    send_window_s = fixed_num_window_s(args.num_req, target_rps)
    meas_end = run_start + send_window_s
    deadline = run_start + args.max_wall_s

    while True:
        now = time.perf_counter()
        if now >= deadline:
            raise TimeoutError(f"exceeded --max-wall-s={args.max_wall_s}")

        # Inject all RT requests whose arrival time has passed.
        elapsed = now - run_start
        while (rt_idx < len(rt_trace)
               and rt_trace[rt_idx].arrival_offset_s <= elapsed + 1e-9):
            req = rt_trace[rt_idx]
            rt_idx += 1
            cmd = rt_command(req, rt_serial)
            rt_serial += 1
            intended_arrival = run_start + req.arrival_offset_s
            arrival_time[cmd["attempt_id"]] = intended_arrival
            in_window = meas_start <= intended_arrival < meas_end
            if in_window:
                result.rt_injected += 1
                pending_rt.add(cmd["attempt_id"])
            pending.append(cmd)

        if pending:
            await worker.submit_batch(pending)
        pending = []
        events = await worker.drain_events()
        observed = time.perf_counter()
        finished_be_phases: list[str] = []

        for ev in events:
            event_time = float(ev.get("time_s", observed))
            in_window = meas_start <= event_time < meas_end
            etype = ev.get("event")
            if etype == "iteration_log":
                if _iterlog_f is not None:
                    _iterlog_f.write(json.dumps(ev) + "\n")
                if not in_window:
                    continue
                n_dec = int(ev.get("n_decode_reqs", 0))
                be_prefill_tokens = int(
                    ev.get("be_prefill_tokens", 0) or 0)
                be_decode_tokens = int(
                    ev.get("be_decode_tokens", 0) or 0)
                be_dummy_handoff_tokens = int(
                    ev.get("be_dummy_handoff_tokens", 0) or 0)
                result.be_prefill_tokens += be_prefill_tokens
                result.be_decode_tokens += be_decode_tokens
                result.be_dummy_handoff_tokens += be_dummy_handoff_tokens
                result.scheduler_wall_ms.append(float(
                    ev.get("scheduler_wall_ms", 0.0) or 0.0))
                overhead = ev.get("runtime_overhead_ms")
                if overhead is not None:
                    result.runtime_overhead_ms.append(float(overhead))
                if be_prefill_tokens > 0:
                    result.be_prefill_chunk_sizes.append(
                        float(be_prefill_tokens))
                if be_prefill_tokens > 0 or be_decode_tokens > 0:
                    actual = float(ev.get("actual_time_ms", 0.0) or 0.0)
                    if actual > 0.0:
                        result.be_chunk_actual_ms.append(actual)
                    predicted = ev.get("predicted_time_ms")
                    if predicted is not None:
                        result.be_chunk_predicted_ms.append(float(predicted))
                    if _chunk_csv_w is not None and actual > 0.0:
                        _chunk_csv_w.writerow([
                            f"{target_rps}", be_prefill_tokens, n_dec,
                            (f"{float(predicted):.3f}"
                             if predicted is not None else ""),
                            f"{actual:.3f}",
                            int(ev.get("num_used_gpu_blocks", 0) or 0),
                            int(ev.get("num_gpu_blocks_total", 0) or 0)])
                    total_blocks = int(ev.get("num_gpu_blocks_total", 0) or 0)
                    used_blocks = int(ev.get("num_used_gpu_blocks", 0) or 0)
                    if total_blocks > 0:
                        result.num_gpu_blocks_total = total_blocks
                        result.be_chunk_used_blocks.append(float(used_blocks))
                        result.be_chunk_block_util.append(
                            used_blocks / total_blocks)
                if be_dummy_handoff_tokens > 0:
                    result.dummy_handoff_iterations += 1
                if be_decode_tokens > 0:
                    result.decode_batch_sizes.append(float(n_dec))
                    if be_prefill_tokens > 0:
                        result.mixed_iterations += 1
                elif be_prefill_tokens > 0:
                    result.pure_prefill_iterations += 1
            elif etype == "token":
                aid = str(ev.get("attempt_id", ""))
                kind = ev.get("kind")
                if kind == "rt" and aid not in seen_first:
                    seen_first.add(aid)
                    t0 = arrival_time.get(aid)
                    if t0 is not None and meas_start <= t0 < meas_end:
                        ttft_ms = (event_time - t0) * 1000
                        result.rt_first_token += 1
                        result.rt_ttft_ms.append(ttft_ms)
                        pending_rt.discard(aid)
                        if ttft_ms > args.ttft_slo_ms and aid not in failed_rt:
                            result.rt_violations += 1
                            failed_rt.add(aid)
            elif etype == "finished":
                kind = ev.get("kind")
                if kind == "be":
                    phase = str(ev.get("phase", PREFILL))
                    finished_be_phases.append(phase)
                    if in_window:
                        result.be_completed += 1
                        result.be_output_tokens += int(ev.get("token_count", 0))
                        if phase == DECODE:
                            result.be_decode_completed += 1
                        else:
                            result.be_prefill_completed += 1
                elif kind == "rt":
                    # RT that never emitted a token still counts as a miss.
                    aid = str(ev.get("attempt_id", ""))
                    if aid in pending_rt:
                        pending_rt.remove(aid)
                        failed_rt.add(aid)
                        result.rt_violations += 1

        # Keep the BE population saturated.
        for phase in finished_be_phases:
            pending.append(next_be(phase))

        settlement_end = max(
            (arrival_time[aid] + args.ttft_slo_ms / 1000.0
             for aid in pending_rt), default=meas_end)
        if observed >= meas_end and rt_idx >= len(rt_trace):
            if not pending_rt or observed >= settlement_end:
                result.rt_violations += len(pending_rt)
                break

        next_arrival = (run_start + rt_trace[rt_idx].arrival_offset_s
                        if rt_idx < len(rt_trace)
                        else max(meas_end, settlement_end))
        sleep_s = min(0.005, max(0.0, next_arrival - time.perf_counter()))
        await asyncio.sleep(sleep_s)

    result.rt_settlement_s = max(0.0, observed - meas_end)
    result.window_s = send_window_s - args.warmup_s
    return result


def result_to_dict(r: RunResult, args: argparse.Namespace) -> dict[str, Any]:
    be_rps = r.be_completed / r.window_s if r.window_s > 0 else 0.0
    be_tok = r.be_output_tokens / r.window_s if r.window_s > 0 else 0.0
    prefill_tok_s = (
        r.be_prefill_tokens / r.window_s if r.window_s > 0 else 0.0)
    decode_tok_s = (
        r.be_decode_tokens / r.window_s if r.window_s > 0 else 0.0)
    prefill_req_s = prefill_tok_s / max(r.avg_be_input_len, 1.0)
    decode_req_s = decode_tok_s / max(r.avg_be_output_len, 1.0)
    viol_rate = (r.rt_violations / r.rt_injected) if r.rt_injected else 0.0
    ttft = summarize(r.rt_ttft_ms)
    batch = summarize(r.decode_batch_sizes)
    return {
        "target_rps": r.target_rps,
        "window_s": round(r.window_s, 3),
        "be_saturation_req_per_s": round(
            min(prefill_req_s, decode_req_s), 4),
        "be_completed_req_per_s": round(be_rps, 4),
        "be_prefill_req_per_s": round(prefill_req_s, 4),
        "be_decode_req_per_s": round(decode_req_s, 4),
        "avg_be_input_len": round(r.avg_be_input_len, 3),
        "avg_be_output_len": round(r.avg_be_output_len, 3),
        "be_token_per_s": round(be_tok, 1),
        "be_prefill_token_per_s": round(
            prefill_tok_s, 1),
        "be_decode_token_per_s": round(
            decode_tok_s, 1),
        "be_completed": r.be_completed,
        "be_prefill_completed": r.be_prefill_completed,
        "be_decode_completed": r.be_decode_completed,
        "be_prefill_tokens": r.be_prefill_tokens,
        "be_decode_tokens": r.be_decode_tokens,
        "be_dummy_handoff_tokens": r.be_dummy_handoff_tokens,
        "rt_injected": r.rt_injected,
        "rt_first_token": r.rt_first_token,
        "rt_without_first_token": r.rt_injected - r.rt_first_token,
        "rt_ttft_samples_complete": r.rt_first_token == r.rt_injected,
        "rt_ttft_sample_scope": "observed_first_tokens_only",
        "rt_settlement_s": round(r.rt_settlement_s, 6),
        "rt_ttft_violations": r.rt_violations,
        "rt_ttft_violation_rate": round(viol_rate, 4),
        "rt_ttft_ms": {k: round(float(v), 3) for k, v in ttft.items()},
        "be_decode_batch_size": {k: round(float(v), 3)
                                 for k, v in batch.items()},
        "be_effective_prefill_chunk_tokens": {
            k: round(float(v), 3)
            for k, v in summarize(r.be_prefill_chunk_sizes).items()
        },
        "be_chunk_num_gpu_blocks_total": r.num_gpu_blocks_total,
        "be_chunk_used_gpu_blocks": {
            k: round(float(v), 3)
            for k, v in summarize(r.be_chunk_used_blocks).items()
        },
        "be_chunk_kv_block_utilization": {
            k: round(float(v), 4)
            for k, v in summarize(r.be_chunk_block_util).items()
        },
        "be_chunk_actual_latency_ms": {
            k: round(float(v), 3)
            for k, v in summarize(r.be_chunk_actual_ms).items()
        },
        "be_chunk_predicted_latency_ms": {
            k: round(float(v), 3)
            for k, v in summarize(r.be_chunk_predicted_ms).items()
        },
        "scheduler_wall_ms": {
            k: round(float(v), 3)
            for k, v in summarize(r.scheduler_wall_ms).items()
        },
        "runtime_overhead_ms": {
            k: round(float(v), 3)
            for k, v in summarize(r.runtime_overhead_ms).items()
        },
        "effective_be_chunks": len(r.be_chunk_actual_ms),
        "mixed_iterations": r.mixed_iterations,
        "pure_prefill_iterations": r.pure_prefill_iterations,
        "dummy_handoff_iterations": r.dummy_handoff_iterations,
    }


def build_sim_plan(args: argparse.Namespace, rps_points: list[float], *,
                   total_gpu_blocks: int, max_model_len: int,
                   plan_path: Path) -> dict[float, dict[str, Any]]:
    """Run the P-pool prefill simulator per RPS to derive the chunk budget.

    The simulator is CPU-only: it reuses the same time-predictor profile and BE
    length filters as the real engine, and its optimal ``be_chunk_latency_ms``
    per RPS becomes the ``p_pool_be_chunk_ms`` budget the real run will use.
    """
    from saturo.simulators.prefill_simulator.run_prefill_simulator import (
        run_prefill_profile,
    )

    if not args.runtime_profile:
        raise ValueError(
            "--chunk-ms-source=sim requires --runtime-profile (the simulator "
            "reads time_predictor + kv_cache_block from it).")

    rt_trace = str(Path(args.rt_workload_csv).expanduser().resolve())
    runtime_profile = str(Path(args.runtime_profile).expanduser().resolve())
    be_trace = str(Path(args.be_workload_csv).expanduser().resolve())

    plan: dict[float, dict[str, Any]] = {}
    for rps in rps_points:
        payload = run_prefill_profile(
            rt_trace=rt_trace,
            runtime_profile=runtime_profile,
            be_trace=be_trace,
            num_req=args.num_req,
            target_rps=rps,
            ttft_slo=args.ttft_slo_ms / 1000.0,
            max_rt_violation_rate=args.sim_max_rt_violation_rate,
            max_model_len=max_model_len,
            total_gpu_blocks=total_gpu_blocks,
            be_input_drop_frac=args.be_input_drop_frac,
            be_input_max_percentile=args.be_input_max_percentile,
            be_output_keep_frac=args.be_output_keep_frac,
            be_output_max_len=args.be_output_max_len,
            be_decode_block_fraction=args.be_decode_block_fraction,
        )
        best = payload.get("best", {}) or {}
        balanced = payload.get("balanced_be_decode", {}) or {}
        plan[rps] = {
            "be_chunk_latency_ms": best.get("be_chunk_latency_ms"),
            "be_prefill_chunk_size": best.get("be_prefill_chunk_size"),
            "be_decode_batch_size": best.get("be_decode_batch_size"),
            "predicted_saturation_req_per_s": best.get(
                "saturation_req_per_s"),
            "rt_infeasible": payload.get("rt_infeasible"),
            "rt_violation_rate": payload.get("rt_violation_rate"),
            "max_be_decode_batch": balanced.get("max_be_decode_batch"),
            "decode_capacity_batch": balanced.get("decode_capacity_batch"),
            "balanced_decode_req_per_s": balanced.get("decode_req_per_s"),
            "balanced_prefill_req_per_s": balanced.get("prefill_req_per_s"),
            "balanced_saturation_req_per_s": balanced.get(
                "saturation_req_per_s"),
        }
        print(f"[sim] RPS={rps}: be_chunk_latency_ms="
              f"{plan[rps]['be_chunk_latency_ms']}, "
              f"predicted_saturation_req_per_s="
              f"{plan[rps]['predicted_saturation_req_per_s']}, "
              f"max_be_decode_batch={plan[rps]['max_be_decode_batch']} "
              f"(capacity={plan[rps]['decode_capacity_batch']}), "
              f"rt_infeasible={plan[rps]['rt_infeasible']}", flush=True)

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump({
            "runtime_profile": runtime_profile,
            "total_gpu_blocks": total_gpu_blocks,
            "max_model_len": max_model_len,
            "ttft_slo_ms": args.ttft_slo_ms,
            "num_req": args.num_req,
            "sim_max_rt_violation_rate": args.sim_max_rt_violation_rate,
            "per_rps": {str(rps): plan[rps] for rps in rps_points},
        }, f, indent=2)
    print(f"[sim] wrote per-RPS chunk plan to {plan_path}", flush=True)
    return plan


async def async_main() -> None:
    args = parse_args()
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
    # TORCH_SDPA is a CPU-only backend in vLLM V0; on a GPU run it crashes in
    # TorchSDPAMetadataBuilder.prepare() because the GPU model runner supplies a
    # ModelInputForGPUBuilder (no ``input_data``). Default to FLASH_ATTN, which
    # is the GPU backend vLLM auto-selects on this hardware.
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    rps_points = [float(x) for x in str(args.target_rps).split(",") if x.strip()]
    if not rps_points or min(rps_points) <= 0:
        raise ValueError("--target-rps must contain positive values")
    from saturo.tools.trace_scaler import fixed_num_window_s
    if any(args.warmup_s >= fixed_num_window_s(args.num_req, rps)
           for rps in rps_points):
        raise ValueError(
            "--warmup-s must be smaller than num_req / target_rps")

    rt_path = Path(args.rt_workload_csv).expanduser().resolve()
    be_path = Path(args.be_workload_csv).expanduser().resolve()
    raw_rt_trace = load_trace(
        rt_path, _rt_const(), num_req=args.num_req,
        target_rps=rps_points[0])
    from saturo.runtime.workload import RT, load_workload_csv
    from saturo.tools.trace_scaler import compute_trace_rps
    raw_source = load_workload_csv(str(rt_path), RT)
    original_rt_rps = compute_trace_rps(
        [request.arrival_offset_s for request in raw_source])
    be_trace = filter_be(load_trace(be_path, _be_const(), args.max_source_be),
                         args)
    print(f"Loaded {len(raw_rt_trace)} RT rows "
          f"(original trace RPS={original_rt_rps:.6f}) and "
          f"{len(be_trace)} BE rows (after filters; "
          f"prompt_max={args.be_input_max_tokens_applied or 'none'}).",
          flush=True)
    print(f"Initializing one Saturo P-pool engine: model={args.model}, "
          f"TP={args.tensor_parallel_size}, load_format={args.load_format}",
          flush=True)

    worker = None
    runs: list[dict[str, Any]] = []
    try:
        worker = create_worker(args)
        core = worker.engine.engine
        max_model_len = int(core.model_config.max_model_len)
        be_trace = sanitize(be_trace, max_model_len)
        if args.be_shuffle_seed is not None:
            random.Random(args.be_shuffle_seed).shuffle(be_trace)
        cache = core.cache_config
        print(f"Engine ready: num_gpu_blocks={cache.num_gpu_blocks}, "
              f"block_size={cache.block_size}, max_model_len={max_model_len}",
              flush=True)

        # Derive the BE-decode KV block limit from the dataset: reserve enough
        # blocks for the longest RT prefill and the longest BE prefill so both
        # can always allocate, cap decode at a fraction of total blocks so it
        # never fills the whole KV cache and starves BE prefill, and give the
        # remaining budget to the decode-ready BE population. The scheduler
        # reads this as an absolute block count
        # (SATURO_P_POOL_BE_DECODE_BLOCK_LIMIT); there is no latency budget
        # split -- every admitted decode request runs each step.
        num_gpu_blocks = int(cache.num_gpu_blocks or 0)
        block_size = int(cache.block_size or 1)
        max_rt_prompt = max(
            (r.prompt_len for r in raw_rt_trace if r.prompt_len < max_model_len),
            default=1)
        max_be_prompt = max((r.prompt_len for r in be_trace), default=1)
        reserved_rt_blocks = math.ceil(max_rt_prompt / block_size)
        reserved_be_prefill_blocks = math.ceil(max_be_prompt / block_size)
        decode_block_fraction = float(args.be_decode_block_fraction)
        usable_decode_blocks = math.floor(num_gpu_blocks * decode_block_fraction)
        be_decode_block_limit = max(
            1,
            usable_decode_blocks
            - reserved_rt_blocks
            - reserved_be_prefill_blocks)
        args.reserved_rt_blocks = reserved_rt_blocks
        args.reserved_be_prefill_blocks = reserved_be_prefill_blocks
        args.be_decode_block_limit = be_decode_block_limit
        if args.enable_be_decode:
            os.environ["SATURO_P_POOL_BE_DECODE_BLOCK_LIMIT"] = str(
                be_decode_block_limit)
            print(f"BE-decode KV limit: num_gpu_blocks={num_gpu_blocks}, "
                  f"decode_block_fraction={decode_block_fraction} "
                  f"(usable={usable_decode_blocks}), "
                  f"reserved_rt_blocks={reserved_rt_blocks} "
                  f"(max_rt_prompt={max_rt_prompt}), "
                  f"reserved_be_prefill_blocks={reserved_be_prefill_blocks} "
                  f"(max_be_prompt={max_be_prompt}) -> "
                  f"be_decode_block_limit={be_decode_block_limit}", flush=True)
        else:
            print("BE-decode disabled: prefill-only BE population.", flush=True)

        sim_plan: dict[float, dict[str, Any]] = {}
        if args.chunk_ms_source == "sim":
            if args.sim_plan_json:
                sim_plan_path = Path(args.sim_plan_json).expanduser().resolve()
            elif args.output_json:
                base = Path(args.output_json).expanduser().resolve()
                sim_plan_path = base.with_suffix(base.suffix + ".sim_plan.json")
            else:
                dflt = default_output_path()
                sim_plan_path = dflt.with_suffix(
                    dflt.suffix + ".sim_plan.json")
            print(f"\n=== Running P-pool simulator for RPS points "
                  f"{rps_points} to derive BE chunk budgets ===", flush=True)
            sim_plan = build_sim_plan(
                args, rps_points,
                total_gpu_blocks=num_gpu_blocks,
                max_model_len=max_model_len,
                plan_path=sim_plan_path,
            )

        for rps in rps_points:
            rt_for_run = sanitize(
                load_trace(rt_path, _rt_const(), num_req=args.num_req,
                           target_rps=rps),
                max_model_len,
            )
            if len(rt_for_run) != args.num_req:
                raise ValueError(
                    f"only {len(rt_for_run)} of num_req={args.num_req} RT "
                    f"requests fit max_model_len={max_model_len}")
            total_window_s = fixed_num_window_s(args.num_req, rps)

            sim_entry = sim_plan.get(rps)
            effective_chunk_ms = args.p_pool_be_chunk_ms
            if args.chunk_ms_source == "sim" and sim_entry is not None:
                sim_chunk_ms = sim_entry.get("be_chunk_latency_ms")
                if sim_chunk_ms and float(sim_chunk_ms) > 0.0:
                    effective_chunk_ms = float(sim_chunk_ms)
            effective_chunk_ms = float(effective_chunk_ms)
            core.scheduler_config.p_pool_be_chunk_ms = effective_chunk_ms
            for sched in core.scheduler:
                sched.scheduler_config.p_pool_be_chunk_ms = effective_chunk_ms

            # Fixed token chunk-size override (> 0 wins over the ms budget).
            effective_chunk_size = int(args.p_pool_be_chunk_size or 0)
            core.scheduler_config.p_pool_be_chunk_size = effective_chunk_size
            for sched in core.scheduler:
                sched.scheduler_config.p_pool_be_chunk_size = \
                    effective_chunk_size

            # BE-decode active-count cap: fixed override when > 0, else the
            # simulator's balanced value (sim mode only). 0 => unlimited.
            effective_max_batch = int(args.be_decode_max_batch or 0)
            if effective_max_batch <= 0 and args.chunk_ms_source == "sim" \
                    and sim_entry is not None:
                sim_batch = sim_entry.get("max_be_decode_batch")
                if sim_batch and int(sim_batch) > 0:
                    effective_max_batch = int(sim_batch)
            if effective_max_batch > 0:
                os.environ["SATURO_P_POOL_MAX_BE_DECODE_BATCH"] = str(
                    effective_max_batch)
            else:
                os.environ.pop("SATURO_P_POOL_MAX_BE_DECODE_BATCH", None)
            core.scheduler_config.p_pool_max_be_decode_batch = \
                effective_max_batch
            for sched in core.scheduler:
                sched.scheduler_config.p_pool_max_be_decode_batch = \
                    effective_max_batch

            print(f"\n=== RPS={rps}: injecting {len(rt_for_run)} RT requests "
                  "at scaled dataset arrival timestamps, "
                  f"send_window={total_window_s:.3f}s, "
                  f"warmup={args.warmup_s}s, "
                  f"be_mode={args.be_request_mode}, "
                  f"be_chunk_size={effective_chunk_size or 'off'}tok, "
                  f"be_chunk_ms={effective_chunk_ms:.3f} "
                  f"(source={args.chunk_ms_source}), "
                  f"max_be_decode_batch="
                  f"{effective_max_batch or 'unlimited'} ===", flush=True)
            result = await run_one_rps(
                worker, rt_for_run, be_trace, rps, args)
            payload = result_to_dict(result, args)
            payload["p_pool_be_chunk_ms_used"] = effective_chunk_ms
            payload["p_pool_be_chunk_size_used"] = effective_chunk_size
            payload["be_request_mode"] = args.be_request_mode
            payload["p_pool_max_be_decode_batch_used"] = effective_max_batch
            payload["sim_prediction"] = sim_entry
            payload["rt_trace"] = {
                "arrival_mode": "scaled_dataset_trace",
                "num_req": args.num_req,
                "scaled_first_arrival_s": rt_for_run[0].arrival_offset_s,
                "scaled_last_arrival_s": rt_for_run[-1].arrival_offset_s,
                "effective_total_window_rps": (
                    len(rt_for_run) / max(total_window_s, 1e-9)),
            }
            runs.append(payload)
            print(json.dumps(payload, indent=2), flush=True)
            await drain_worker(worker, args.drain_timeout_s)

        output = {
            "config": {
                "model": args.model,
                "tensor_parallel_size": args.tensor_parallel_size,
                "load_format": args.load_format,
                "ttft_slo_ms": args.ttft_slo_ms,
                "rt_prefill_deadline_s": args.rt_prefill_deadline_s,
                "p_pool_be_chunk_ms": args.p_pool_be_chunk_ms,
                "p_pool_be_chunk_size": args.p_pool_be_chunk_size,
                "be_request_mode": args.be_request_mode,
                "chunk_ms_source": args.chunk_ms_source,
                "sim_max_rt_violation_rate": args.sim_max_rt_violation_rate,
                "p_pool_preemption_margin": args.p_pool_preemption_margin,
                "be_inflight": args.be_inflight,
                "enable_be_decode": args.enable_be_decode,
                "be_decode_kv_block_limit": getattr(
                    args, "be_decode_block_limit", None),
                "be_decode_max_batch": args.be_decode_max_batch,
                "reserved_rt_blocks": getattr(args, "reserved_rt_blocks", None),
                "reserved_be_prefill_blocks": getattr(
                    args, "reserved_be_prefill_blocks", None),
                "be_decode_inflight": (
                    round(args.be_inflight * 0.5)
                    if args.enable_be_decode else 0),
                "be_prefill_inflight": (
                    args.be_inflight - round(args.be_inflight * 0.5)
                    if args.enable_be_decode else args.be_inflight),
                "be_shuffle_seed": args.be_shuffle_seed,
                "warmup_s": args.warmup_s,
                "num_req": args.num_req,
                "target_rps": rps_points,
                "runtime_profile": args.runtime_profile or "none",
                "num_gpu_blocks": int(core.cache_config.num_gpu_blocks),
                "block_size": int(core.cache_config.block_size),
                "rt_workload_csv": str(rt_path),
                "rt_arrival_mode": "scaled_dataset_trace",
                "rt_original_trace_rps": original_rt_rps,
                "be_workload_csv": str(be_path),
                "be_input_drop_frac": args.be_input_drop_frac,
                "be_input_max_percentile": args.be_input_max_percentile,
                "be_input_max_tokens_applied":
                    args.be_input_max_tokens_applied,
                "be_output_keep_frac": args.be_output_keep_frac,
                "be_output_max_len": args.be_output_max_len,
            },
            "runs": runs,
        }
        out_path = (Path(args.output_json).expanduser().resolve()
                    if args.output_json else default_output_path())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\n[Output] wrote {out_path}", flush=True)
    finally:
        if worker is not None:
            try:
                await worker.shutdown()
            except Exception:
                pass
            del worker
        gc.collect()


def _rt_const() -> Any:
    from saturo.runtime.workload import RT
    return RT


def _be_const() -> Any:
    from saturo.runtime.workload import BE
    return BE


if __name__ == "__main__":
    asyncio.run(async_main())
