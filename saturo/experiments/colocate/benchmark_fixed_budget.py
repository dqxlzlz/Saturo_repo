#!/usr/bin/env python3
"""One-point real-GPU RT/BE colocation limitation experiment.

The engine uses vLLM V0 chunked prefill with a fixed
``max_num_batched_tokens`` and a small RT-prefill priority policy. No Saturo
latency/TBT optimizer is enabled.
At startup a finite BE batch (with real prompt and output lengths) is submitted
all at once.  RT requests are then replayed at a scaled trace RPS.

Scheduler instrumentation counts partial work as request equivalents: if 600
tokens of a 3000-token stage are processed, that stage contributes 0.2 req.
BE goodput is min(BE-prefill req/s, BE-decode req/s).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


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
class RequestShape:
    kind: str
    prompt_len: int
    output_len: int


@dataclass
class RTMetric:
    request_id: str
    prompt_len: int
    output_len: int
    submit_s: float
    counted: bool
    completed: bool = False
    ttft_ms: float | None = None
    tbt_ms: list[float] = field(default_factory=list)
    output_tokens: int = 0


@dataclass
class StepStats:
    steps: int = 0
    scheduled_tokens: int = 0
    rt_prefill_tokens: int = 0
    rt_decode_tokens: int = 0
    be_prefill_tokens: int = 0
    be_decode_tokens: int = 0
    rt_prefill_req_eq: float = 0.0
    rt_decode_req_eq: float = 0.0
    be_prefill_req_eq: float = 0.0
    be_decode_req_eq: float = 0.0
    preempted: int = 0
    swap_out_blocks: int = 0
    rt_prefill_step_tokens: list[int] = field(default_factory=list)
    rt_prefill_chunk_tokens: list[int] = field(default_factory=list)
    rt_prefill_chunks_by_request: dict[str, int] = field(
        default_factory=dict)


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    rank = int(math.ceil(q * len(ordered))) - 1
    return ordered[min(max(rank, 0), len(ordered) - 1)]


def summary(values: Iterable[float]) -> dict[str, float | int]:
    data = list(float(v) for v in values)
    if not data:
        return {"count": 0, "mean": 0.0, "p50": 0.0,
                "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "p50": statistics.median(data),
        "p95": percentile(data, 0.95),
        "p99": percentile(data, 0.99),
        "max": max(data),
    }


def load_trace(path: Path, kind: str, *, num_req: int | None = None,
               target_rps: float | None = None, max_requests: int = 0,
               input_scale: float = 1.0) -> list[TraceRequest]:
    from saturo.runtime.workload import (
        load_rt_workload_csv,
        load_workload_csv,
    )

    if kind == "rt":
        if num_req is None or target_rps is None:
            raise ValueError("RT loading requires num_req and target_rps")
        loaded = load_rt_workload_csv(
            str(path), num_req=num_req, target_rps=target_rps)
    else:
        loaded = load_workload_csv(str(path), kind)
    if max_requests > 0:
        loaded = loaded[:max_requests]
    result = [
        TraceRequest(
            request_id=str(req.request_id),
            prompt_len=max(1, int(round(req.prompt_len * input_scale))),
            output_len=max(1, int(req.output_len)),
            arrival_offset_s=(float(req.arrival_offset_s)
                              if kind == "rt" else 0.0),
        )
        for req in loaded
    ]
    if not result:
        raise ValueError(f"empty {kind} workload: {path}")
    return sorted(result, key=lambda r: r.arrival_offset_s)


def sanitize(trace: list[TraceRequest], max_model_len: int) -> list[TraceRequest]:
    result: list[TraceRequest] = []
    for req in trace:
        if req.prompt_len >= max_model_len:
            continue
        output_len = min(req.output_len, max_model_len - req.prompt_len)
        if output_len > 0:
            result.append(TraceRequest(
                req.request_id, req.prompt_len, output_len,
                req.arrival_offset_s))
    if not result:
        raise ValueError(f"no request fits max_model_len={max_model_len}")
    return result


def cap_prompt_percentile(trace: list[TraceRequest],
                          pctl: float) -> list[TraceRequest]:
    """Drop requests whose prompt_len exceeds the given percentile.

    Long-tail prompts dominate KV-cache usage, so only a handful stay
    resident and the fixed token budget cannot be filled. Trimming the tail
    lets many more requests coexist, so a larger budget can actually batch
    more prefill work.
    """
    if pctl >= 100.0 or len(trace) < 2:
        return trace
    cutoff = percentile([r.prompt_len for r in trace], pctl / 100.0)
    kept = [r for r in trace if r.prompt_len <= cutoff]
    if not kept:
        raise ValueError(
            f"prompt-percentile cutoff {pctl} left no BE requests")
    print(f"[be-filter] pctl={pctl} cutoff_prompt_len={int(cutoff)} "
          f"kept={len(kept)}/{len(trace)}", flush=True)
    return kept



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    model = p.add_argument_group("model")
    model.add_argument("--model", required=True)
    model.add_argument("--dtype", default="auto")
    model.add_argument("--load-format", default="auto")
    model.add_argument("--gpu-id", default="1")
    model.add_argument("--tensor-parallel-size", type=int, default=1)
    model.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    model.add_argument("--max-model-len", type=int, default=None)
    model.add_argument("--max-num-seqs", type=int, default=256)
    model.add_argument("--swap-space-gb", type=float, default=4.0)
    model.add_argument(
        "--protect-rt-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let RT prefill bypass already-running BE partial prefills.")

    work = p.add_argument_group("workload")
    work.add_argument("--rt-workload-csv", required=True)
    work.add_argument("--be-workload-csv", required=True)
    work.add_argument("--num-req", type=int, required=True)
    work.add_argument("--target-rps", type=float, required=True)
    work.add_argument("--chunk-size", type=int, required=True,
                      help="Fixed max_num_batched_tokens for every step.")
    work.add_argument("--input-scale", type=float, default=1.0)
    work.add_argument("--be-requests", type=int, default=128,
                      help="Finite BE batch submitted together at startup.")
    work.add_argument("--be-max-prompt-percentile", type=float, default=100.0,
                      help="Drop BE requests whose prompt_len exceeds this "
                           "percentile of the BE prompt-length distribution. "
                           "100 keeps all; e.g. 95 drops the longest 5 percent "
                           "tail that otherwise clogs KV cache and starves the "
                           "token budget.")
    work.add_argument("--max-source-be", type=int, default=0)
    work.add_argument("--be-lead-s", type=float, default=2.0)
    work.add_argument("--min-be-decode-ready", type=int, default=1,
                      help="Wait for at least this many resident BE decode "
                           "requests before starting RT injection.")
    work.add_argument("--be-setup-timeout-s", type=float, default=30.0,
                      help="Maximum BE-only setup time before RT starts.")
    work.add_argument("--warmup-s", type=float, default=3.0)
    work.add_argument("--shutdown-grace-s", type=float, default=30.0)

    slo = p.add_argument_group("RT SLO")
    slo.add_argument("--ttft-slo-ms", type=float, default=1000.0)
    slo.add_argument("--tbt-slo-ms", type=float, default=150.0)

    p.add_argument("--output-json", default="")
    args = p.parse_args()
    if args.num_req <= 0 or args.target_rps <= 0 or args.chunk_size <= 0:
        p.error("--num-req, --target-rps and --chunk-size must be positive")
    if args.be_requests < 0 or args.max_num_seqs <= 0:
        p.error("--be-requests must be non-negative and "
                "--max-num-seqs must be positive")
    if args.min_be_decode_ready < 0 or args.be_setup_timeout_s <= 0:
        p.error("invalid BE decode-ready setup configuration")
    if args.input_scale <= 0:
        p.error("--input-scale must be positive")
    from saturo.tools.trace_scaler import fixed_num_window_s
    if args.warmup_s >= fixed_num_window_s(
            args.num_req, args.target_rps) or args.be_lead_s < 0:
        p.error("invalid experiment duration")
    return args


def build_engine(args: argparse.Namespace) -> Any:
    from vllm.config import CompilationConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "skip_tokenizer_init": True,
        "enforce_eager": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "load_format": args.load_format,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "swap_space": args.swap_space_gb,
        "disable_custom_all_reduce": True,
        "scheduling_policy": "priority",
        "enable_chunked_prefill": True,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.chunk_size,
        "disable_log_requests": True,
        "disable_log_stats": True,
        # Explicitly disable both adaptive Saturo policies.  This experiment
        # deliberately uses only the fixed scheduler token budget.
        "enable_decode_saturation": False,
        "enable_p_pool_saturation": False,
        "enable_fixed_budget_rt_prefill_priority": args.protect_rt_prefill,
        "compilation_config": CompilationConfig(
            level=0, custom_ops=["none"], use_cudagraph=False,
            use_inductor=False),
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))


async def run_experiment(engine: Any, rt_trace: list[TraceRequest],
                         be_trace: list[TraceRequest],
                         args: argparse.Namespace) -> dict[str, Any]:
    from vllm import SamplingParams

    experiment_start = time.perf_counter()
    # These boundaries are assigned after the BE-only setup phase. Keeping
    # them at infinity prevents setup work from leaking into measurements.
    rt_start = math.inf
    measure_start = math.inf
    measure_end = math.inf

    shapes: dict[str, RequestShape] = {}
    active_ids: set[str] = set()
    rt_metrics: list[RTMetric] = []
    steps = StepStats()
    be_completed = 0

    def in_measurement(now: float) -> bool:
        return measure_start <= now < measure_end

    def resident_be_decode_count() -> int:
        count = 0
        for scheduler in engine.engine.scheduler:
            for group in list(scheduler.running):
                shape = shapes.get(str(group.request_id))
                if (shape is not None and shape.kind == "be"
                        and not group.is_prefill()):
                    count += 1
        return count

    # Record actual scheduler work, split by request class and P/D phase.
    for scheduler in engine.engine.scheduler:
        original = scheduler.schedule

        def wrap(orig):
            def scheduled():
                out = orig()
                now = time.perf_counter()
                scheduler_output = out[1]
                if in_measurement(now):
                    steps.steps += 1
                    steps.scheduled_tokens += int(
                        scheduler_output.num_batched_tokens)
                    steps.preempted += int(scheduler_output.preempted)
                    steps.swap_out_blocks += len(
                        scheduler_output.blocks_to_swap_out)
                    rt_prefill_this_step = 0
                    for item in scheduler_output.scheduled_seq_groups:
                        rid = str(item.seq_group.request_id)
                        shape = shapes.get(rid)
                        if shape is None:
                            continue
                        tokens = int(item.token_chunk_size)
                        phase = ("prefill" if item.seq_group.is_prefill()
                                 else "decode")
                        token_attr = f"{shape.kind}_{phase}_tokens"
                        eq_attr = f"{shape.kind}_{phase}_req_eq"
                        denom = (shape.prompt_len if phase == "prefill"
                                 else shape.output_len)
                        setattr(steps, token_attr,
                                getattr(steps, token_attr) + tokens)
                        setattr(steps, eq_attr,
                                getattr(steps, eq_attr) + tokens / denom)
                        if shape.kind == "rt" and phase == "prefill":
                            rt_prefill_this_step += tokens
                            steps.rt_prefill_chunk_tokens.append(tokens)
                            steps.rt_prefill_chunks_by_request[rid] = (
                                steps.rt_prefill_chunks_by_request.get(rid, 0)
                                + 1)
                    steps.rt_prefill_step_tokens.append(
                        rt_prefill_this_step)
                return out
            return scheduled

        scheduler.schedule = wrap(original)

    def rt_params(req: TraceRequest) -> SamplingParams:
        return SamplingParams(
            temperature=0.0,
            max_tokens=req.output_len,
            ignore_eos=True,
            detokenize=False,
            is_realtime=True,
            prefill_ddl=time.time() + 3600.0,
        )

    def be_params(req: TraceRequest) -> SamplingParams:
        return SamplingParams(
            temperature=0.0,
            max_tokens=req.output_len,
            ignore_eos=True,
            detokenize=False,
        )

    async def be_request(req: TraceRequest, serial: int) -> None:
        nonlocal be_completed
        rid = f"be_{serial}:{req.request_id}"
        shapes[rid] = RequestShape("be", req.prompt_len, req.output_len)
        active_ids.add(rid)
        prompt = {"prompt_token_ids": [0] * req.prompt_len}
        try:
            async for _ in engine.generate(
                    prompt, be_params(req), rid, priority=10):
                pass
            if in_measurement(time.perf_counter()):
                be_completed += 1
        finally:
            active_ids.discard(rid)

    async def rt_request(req: TraceRequest, serial: int) -> None:
        rid = f"rt_{serial}:{req.request_id}"
        submit = time.perf_counter()
        metric = RTMetric(
            rid, req.prompt_len, req.output_len,
            submit - experiment_start, in_measurement(submit))
        rt_metrics.append(metric)
        shapes[rid] = RequestShape("rt", req.prompt_len, req.output_len)
        active_ids.add(rid)
        prompt = {"prompt_token_ids": [0] * req.prompt_len}
        previous_tokens = 0
        previous_event: float | None = None
        try:
            async for output in engine.generate(
                    prompt, rt_params(req), rid, priority=0):
                if not output.outputs:
                    continue
                now = time.perf_counter()
                current_tokens = max(
                    len(candidate.token_ids) for candidate in output.outputs)
                if current_tokens <= previous_tokens:
                    continue
                if previous_tokens == 0:
                    metric.ttft_ms = (now - submit) * 1000.0
                elif previous_event is not None:
                    # V0 normally emits one token per callback. If multiple
                    # arrive together, record one wall interval plus zero for
                    # the additional co-emitted tokens.
                    metric.tbt_ms.append((now - previous_event) * 1000.0)
                    metric.tbt_ms.extend(
                        [0.0] * max(current_tokens - previous_tokens - 1, 0))
                previous_tokens = current_tokens
                previous_event = now
                metric.output_tokens = current_tokens
            metric.completed = True
        finally:
            active_ids.discard(rid)

    # One-shot BE flood before any RT request is submitted.
    be_tasks = [
        asyncio.create_task(be_request(
            be_trace[i % len(be_trace)], i))
        for i in range(args.be_requests)
    ]

    if args.be_lead_s > 0:
        await asyncio.sleep(args.be_lead_s)
    setup_deadline = experiment_start + args.be_setup_timeout_s
    while (resident_be_decode_count() < args.min_be_decode_ready
           and time.perf_counter() < setup_deadline):
        await asyncio.sleep(0.05)
    be_decode_at_rt_start = resident_be_decode_count()
    rt_start = time.perf_counter()
    measure_start = rt_start + args.warmup_s
    from saturo.tools.trace_scaler import fixed_num_window_s
    send_window_s = fixed_num_window_s(args.num_req, args.target_rps)
    measure_end = rt_start + send_window_s

    rt_tasks: list[asyncio.Task] = []

    async def inject_rt() -> None:
        for serial, req in enumerate(rt_trace):
            target = rt_start + req.arrival_offset_s
            if target > measure_end:
                break
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            rt_tasks.append(asyncio.create_task(rt_request(req, serial)))

    injector = asyncio.create_task(inject_rt())
    if args.warmup_s > 0:
        await asyncio.sleep(args.warmup_s)
    be_decode_at_measure_start = resident_be_decode_count()
    remaining = measure_end - time.perf_counter()
    if remaining > 0:
        await asyncio.sleep(remaining)
    await injector

    # Keep BE load present while late-window RT requests drain, but cap it.
    if rt_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*rt_tasks, return_exceptions=True),
                timeout=args.shutdown_grace_s)
        except asyncio.TimeoutError:
            pass

    for rid in list(active_ids):
        try:
            await engine.abort(rid)
        except Exception:
            pass
    pending = [task for task in be_tasks + rt_tasks if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    counted = [m for m in rt_metrics if m.counted]
    ttft_samples = [m.ttft_ms for m in counted if m.ttft_ms is not None]
    tbt_samples = [sample for m in counted for sample in m.tbt_ms]
    ttft_passed = [
        m for m in counted
        if (m.completed
            and m.ttft_ms is not None
            and m.ttft_ms <= args.ttft_slo_ms)
    ]
    tbt_passed = [
        m for m in counted
        if m.completed and all(v <= args.tbt_slo_ms for v in m.tbt_ms)
    ]
    passed = [
        m for m in counted
        if (m.completed
            and m.ttft_ms is not None
            and m.ttft_ms <= args.ttft_slo_ms
            and all(v <= args.tbt_slo_ms for v in m.tbt_ms))
    ]
    tbt_violations = sum(v > args.tbt_slo_ms for v in tbt_samples)

    duration = send_window_s - args.warmup_s
    be_p_rps = steps.be_prefill_req_eq / duration
    be_d_rps = steps.be_decode_req_eq / duration
    rt_p_rps = steps.rt_prefill_req_eq / duration
    rt_d_rps = steps.rt_decode_req_eq / duration
    return {
        "timing": {
            "be_lead_s": args.be_lead_s,
            "be_setup_actual_s": rt_start - experiment_start,
            "min_be_decode_ready": args.min_be_decode_ready,
            "be_decode_at_rt_start": be_decode_at_rt_start,
            "be_decode_at_measure_start": be_decode_at_measure_start,
            "warmup_s": args.warmup_s,
            "measurement_s": duration,
        },
        "rt": {
            "num_req": args.num_req,
            "target_rps": args.target_rps,
            "send_window_s": send_window_s,
            "submitted_in_window": len(counted),
            "completed_in_window_cohort": sum(m.completed for m in counted),
            "slo_passed_requests": len(passed),
            "slo_attainment": len(passed) / len(counted) if counted else 0.0,
            "ttft_slo_passed_requests": len(ttft_passed),
            "ttft_slo_attainment": (
                len(ttft_passed) / len(counted) if counted else 0.0),
            "tbt_slo_passed_requests": len(tbt_passed),
            "tbt_slo_attainment": (
                len(tbt_passed) / len(counted) if counted else 0.0),
            "ttft_slo_ms": args.ttft_slo_ms,
            "tbt_slo_ms": args.tbt_slo_ms,
            "ttft_ms": summary(ttft_samples),
            "tbt_ms": summary(tbt_samples),
            "tbt_token_violations": tbt_violations,
            "tbt_token_violation_rate": (
                tbt_violations / len(tbt_samples) if tbt_samples else 0.0),
            "prefill_tokens": steps.rt_prefill_tokens,
            "decode_tokens": steps.rt_decode_tokens,
            "prefill_req_equiv_per_s": rt_p_rps,
            "decode_req_equiv_per_s": rt_d_rps,
            "prefill_chunk_tokens": summary(
                steps.rt_prefill_chunk_tokens),
            "prefill_tokens_on_active_steps": summary(
                value for value in steps.rt_prefill_step_tokens if value > 0),
            "prefill_active_steps": sum(
                value > 0 for value in steps.rt_prefill_step_tokens),
            "prefill_chunks_per_request": summary(
                steps.rt_prefill_chunks_by_request.values()),
        },
        "be": {
            "initial_requests": args.be_requests,
            "completed_in_window": be_completed,
            "prefill_tokens": steps.be_prefill_tokens,
            "decode_tokens": steps.be_decode_tokens,
            "prefill_req_equiv": steps.be_prefill_req_eq,
            "decode_req_equiv": steps.be_decode_req_eq,
            "prefill_req_equiv_per_s": be_p_rps,
            "decode_req_equiv_per_s": be_d_rps,
            "goodput_req_equiv_per_s": min(be_p_rps, be_d_rps),
            "prefill_tokens_per_s": steps.be_prefill_tokens / duration,
            "decode_tokens_per_s": steps.be_decode_tokens / duration,
        },
        "scheduler": {
            "steps": steps.steps,
            "scheduled_tokens": steps.scheduled_tokens,
            "mean_tokens_per_step": (
                steps.scheduled_tokens / steps.steps if steps.steps else 0.0),
            "fixed_token_budget": args.chunk_size,
            "preempted": steps.preempted,
            "swap_out_block_events": steps.swap_out_blocks,
        },
        "rt_requests": [asdict(m) for m in counted],
    }


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    engine = build_engine(args)
    try:
        max_model_len = int(engine.engine.model_config.max_model_len)
        rt_trace = sanitize(load_trace(
            Path(args.rt_workload_csv), "rt", num_req=args.num_req,
            target_rps=args.target_rps,
            input_scale=args.input_scale), max_model_len)
        be_source = load_trace(
            Path(args.be_workload_csv), "be",
            max_requests=args.max_source_be,
            input_scale=args.input_scale)
        be_source = cap_prompt_percentile(
            be_source, args.be_max_prompt_percentile)
        be_trace = sanitize(be_source, max_model_len)
        print(
            f"[fixed-budget] rps={args.target_rps} chunk={args.chunk_size} "
            f"RT={len(rt_trace)} BE_flood={args.be_requests} "
            f"input_scale={args.input_scale} max_model_len={max_model_len}",
            flush=True)
        result = asyncio.run(run_experiment(
            engine, rt_trace, be_trace, args))
    finally:
        try:
            engine.shutdown_background_loop()
        except Exception:
            pass

    output = {
        "config": {
            "model": args.model,
            "gpu_id": args.gpu_id,
            "load_format": args.load_format,
            "rt_workload_csv": str(Path(args.rt_workload_csv).resolve()),
            "be_workload_csv": str(Path(args.be_workload_csv).resolve()),
            "num_req": args.num_req,
            "target_rps": args.target_rps,
            "chunk_size": args.chunk_size,
            "input_scale": args.input_scale,
            "be_requests": args.be_requests,
            "be_max_prompt_percentile": args.be_max_prompt_percentile,
            "adaptive_saturation_enabled": False,
            "fixed_budget_rt_prefill_priority": args.protect_rt_prefill,
        },
        **result,
    }
    if args.output_json:
        output_path = Path(args.output_json)
    else:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_path = (PROJECT_ROOT / "saturo" / "results" /
                       "fixed_budget_rt_be" /
                       f"rps{args.target_rps}_chunk{args.chunk_size}_{stamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n[RT]", json.dumps(output["rt"], indent=2), flush=True)
    print("\n[BE]", json.dumps(output["be"], indent=2), flush=True)
    print("\n[Scheduler]", json.dumps(
        output["scheduler"], indent=2), flush=True)
    print(f"\n[Output] {output_path}", flush=True)


if __name__ == "__main__":
    main()
