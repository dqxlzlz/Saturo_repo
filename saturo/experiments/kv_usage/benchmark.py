#!/usr/bin/env python3
"""Measure per-forward KV-cache occupancy for trace-paced RT decode traffic.

This is intentionally a small, isolated experiment:

* one AsyncLLMEngine configured like a Decode-Pool instance;
* RT requests only (no BE requests);
* decode-only requests use direct scheduler admission without D-pool prefill;
* Saturo decode saturation and all of its predictors are disabled;
* vLLM schedules with one fixed ``max_num_batched_tokens`` budget; and
* allocator-resident GPU KV blocks are sampled for every non-empty forward.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    prompt_len: int
    output_len: int
    arrival_offset_s: float


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(q * len(ordered)) - 1)
    return float(ordered[min(rank, len(ordered) - 1)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="/root/models/Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--rt-workload-csv",
                        default="data/datasets/rt_workload.csv")
    parser.add_argument("--num-req", type=int, required=True)
    parser.add_argument("--target-rps", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=8192,
        help="Fixed vLLM max_num_batched_tokens used on every scheduler step.",
    )
    parser.add_argument(
        "--kv-usage-target-pct",
        type=float,
        default=90.0,
        help="An RPS passes when the P99 per-forward KV block usage stays at "
             "or below this percentage.",
    )
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--shutdown-grace-s", type=float, default=5.0)
    parser.add_argument(
        "--admission-drain-s",
        type=float,
        default=5.0,
        help="After arrivals stop, keep scheduling for this long so requests "
             "near the window end can receive an admission decision.",
    )
    args = parser.parse_args()
    if args.num_req <= 0 or args.target_rps <= 0:
        parser.error("--num-req and --target-rps must be positive")
    if args.token_budget <= 0 or args.max_num_seqs <= 0:
        parser.error("--token-budget and --max-num-seqs must be positive")
    if not 0.0 < args.kv_usage_target_pct <= 100.0:
        parser.error("--kv-usage-target-pct must be in (0, 100]")
    from saturo.tools.trace_scaler import fixed_num_window_s
    if (args.warmup_s < 0 or args.admission_drain_s < 0
            or args.warmup_s >= fixed_num_window_s(
                args.num_req, args.target_rps)):
        parser.error("invalid measurement duration")
    return args


def load_trace(args: argparse.Namespace) -> tuple[list[TraceRequest], dict[str, Any]]:
    from saturo.runtime.workload import load_rt_workload_csv
    from saturo.tools.trace_scaler import compute_trace_rps

    source = Path(args.rt_workload_csv).expanduser().resolve()
    with source.open(newline="") as f:
        raw_arrivals = [
            float(row.get("arrival_offset_s") or 0.0)
            for row in csv.DictReader(f)
        ]
    original_rps = compute_trace_rps(raw_arrivals)
    loaded = load_rt_workload_csv(
        str(source), num_req=args.num_req, target_rps=args.target_rps)
    if not loaded:
        raise ValueError(f"empty RT trace: {source}")
    base = float(loaded[0].arrival_offset_s)
    trace = [
        TraceRequest(
            request_id=str(req.request_id),
            prompt_len=max(1, int(req.prompt_len)),
            output_len=max(1, int(req.output_len)),
            arrival_offset_s=float(req.arrival_offset_s) - base,
        )
        for req in loaded
    ]
    return trace, {
        "source": str(source),
        "source_rows": len(raw_arrivals),
        "loaded_rows": len(trace),
        "num_req": args.num_req,
        "original_trace_rps": original_rps,
        "target_rps": args.target_rps,
    }


def sanitize_trace(
    trace: list[TraceRequest], max_model_len: int,
) -> tuple[list[TraceRequest], dict[str, int]]:
    usable: list[TraceRequest] = []
    skipped = 0
    clamped = 0
    for req in trace:
        if req.prompt_len >= max_model_len:
            skipped += 1
            continue
        output_len = min(req.output_len, max_model_len - req.prompt_len)
        clamped += int(output_len != req.output_len)
        usable.append(TraceRequest(
            req.request_id, req.prompt_len, max(1, output_len),
            req.arrival_offset_s))
    if not usable:
        raise ValueError(f"no RT request fits max_model_len={max_model_len}")
    return usable, {
        "usable_rows": len(usable),
        "skipped_prompt_too_long": skipped,
        "clamped_output_len": clamped,
    }


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
        "disable_custom_all_reduce": True,
        "scheduling_policy": "priority",
        "enable_chunked_prefill": False,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.token_budget,
        "disable_log_requests": True,
        "disable_log_stats": True,
        # Fixed-budget baseline: none of Saturo's arrival, output-length,
        # memory, or latency predictions are constructed or consulted.
        "enable_decode_saturation": False,
        "enable_p_pool_saturation": False,
        "compilation_config": CompilationConfig(
            level=0,
            custom_ops=["none"],
            use_cudagraph=False,
            use_inductor=False,
        ),
    }
    return AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(**kwargs), start_engine_loop=True)


def install_forward_sampler(
    engine: Any,
    rows: list[dict[str, Any]],
    clock: dict[str, float],
    request_records: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    """Sample real allocator occupancy after scheduling every forward batch."""
    core = engine.engine
    total_blocks = int(core.cache_config.num_gpu_blocks)
    block_size = int(core.cache_config.block_size)

    for scheduler_index, scheduler in enumerate(core.scheduler):
        original_schedule = scheduler.schedule

        def wrapped_schedule(
            original: Any = original_schedule,
            sched: Any = scheduler,
            index: int = scheduler_index,
        ) -> Any:
            # A request necessarily enters ``waiting`` before its prompt KV is
            # admitted. Seeing it once is the normal first scheduling
            # opportunity; seeing it on a later schedule means real queueing.
            pending_admission = (
                list(sched.waiting)
                + list(getattr(sched, "waiting_rt_decode_admission", ()))
            )
            for group in pending_admission:
                request_id = str(group.request_id)
                record = request_records.get(request_id)
                if record is not None:
                    record["waiting_schedule_observations"] += 1
            result = original()
            scheduled = result[1]
            if not scheduled.scheduled_seq_groups:
                return result
            now = time.perf_counter()
            used_blocks = total_blocks - int(
                sched.block_manager.get_num_free_gpu_blocks())
            prefill_reqs = 0
            decode_reqs = 0
            rt_reqs = 0
            for item in scheduled.scheduled_seq_groups:
                if item.seq_group.is_prefill():
                    prefill_reqs += 1
                else:
                    decode_reqs += 1
                request_id = str(item.seq_group.request_id)
                rt_reqs += int(request_id.startswith("rt_"))
                record = request_records.get(request_id)
                if (record is not None
                        and record["first_schedule_wall_time_s"] is None):
                    record["first_schedule_wall_time_s"] = now
                    record["first_schedule_relative_time_s"] = (
                        now - clock["measurement_start_s"])
                    record["admission_wait_ms"] = 1000.0 * (
                        now - record["submit_wall_time_s"])
                    observations = int(
                        record["waiting_schedule_observations"])
                    record["queue_steps_beyond_first"] = max(
                        observations - 1, 0)
                    record["immediate_admission"] = observations <= 1
            rows.append({
                "wall_time_s": now,
                "relative_time_s": now - clock["measurement_start_s"],
                "scheduler_index": index,
                "used_gpu_blocks": used_blocks,
                "total_gpu_blocks": total_blocks,
                "kv_usage_pct": 100.0 * used_blocks / total_blocks,
                "num_batched_tokens": int(scheduled.num_batched_tokens),
                "scheduled_reqs": len(scheduled.scheduled_seq_groups),
                "rt_reqs": rt_reqs,
                "prefill_reqs": prefill_reqs,
                "decode_reqs": decode_reqs,
                "running_queue_size": int(scheduled.running_queue_size),
                "waiting_queue_size": len(sched.waiting),
            })
            return result

        scheduler.schedule = wrapped_schedule
    return total_blocks, block_size


async def run_request(
    engine: Any,
    req: TraceRequest,
    serial: int,
    active_ids: set[str],
    request_records: dict[str, dict[str, Any]],
    clock: dict[str, float],
    intended_arrival_s: float,
) -> None:
    from vllm import SamplingParams

    request_id = f"rt_{serial}:{req.request_id}"
    submit_s = time.perf_counter()
    request_records[request_id] = {
        "request_id": request_id,
        "trace_request_id": req.request_id,
        "prompt_len": req.prompt_len,
        "output_len": req.output_len,
        "intended_arrival_relative_s": intended_arrival_s,
        "submit_wall_time_s": submit_s,
        "submit_relative_time_s": (
            submit_s - clock["measurement_start_s"]),
        "first_schedule_wall_time_s": None,
        "first_schedule_relative_time_s": None,
        "admission_wait_ms": None,
        "waiting_schedule_observations": 0,
        "queue_steps_beyond_first": None,
        "immediate_admission": False,
        "completed": False,
    }
    active_ids.add(request_id)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=req.output_len + 1,
        ignore_eos=True,
        detokenize=False,
        is_realtime=True,
        prefill_ddl=time.time() + 3600.0,
        extra_args={"saturo_direct_decode": True},
    )
    prompt = {"prompt_token_ids": [1] * req.prompt_len}
    try:
        async for _ in engine.generate(
                prompt, params, request_id, priority=0):
            pass
        request_records[request_id]["completed"] = True
    finally:
        active_ids.discard(request_id)


async def inject_trace(
    engine: Any,
    trace: list[TraceRequest],
    run_start_s: float,
    active_ids: set[str],
    request_tasks: list[asyncio.Task[Any]],
    counters: dict[str, int],
    request_records: dict[str, dict[str, Any]],
    clock: dict[str, float],
) -> None:
    for serial, req in enumerate(trace):
        target = run_start_s + req.arrival_offset_s
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        request_tasks.append(asyncio.create_task(
            run_request(
                engine, req, serial, active_ids, request_records, clock,
                target - run_start_s)))
        counters["submitted"] += 1


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "kv_block_usage_per_forward.csv"
    keys = list(rows[0]) if rows else ["relative_time_s", "kv_usage_pct"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    request_csv_path = output_dir / "request_admission.csv"
    request_keys = (list(request_rows[0]) if request_rows else ["request_id"])
    with request_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=request_keys)
        writer.writeheader()
        writer.writerows(request_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.0, 3.8), dpi=180)
    x = [float(row["relative_time_s"]) for row in rows]
    y = [float(row["kv_usage_pct"]) for row in rows]
    axis.plot(x, y, color="#12828c", linewidth=1.2)
    axis.fill_between(x, y, 0.0, color="#12828c", alpha=0.08)
    axis.set_xlim(0.0, float(summary["measurement_s"]))
    axis.set_ylim(0.0, 100.0)
    target_pct = float(summary["kv_usage_target_pct"])
    p99_pct = float(summary["kv_usage_pct_p99"])
    axis.axhline(
        target_pct, color="#d1495b", linewidth=1.1, linestyle="-",
        label=f"target {target_pct:.0f}%")
    axis.axhline(
        p99_pct, color="#edae49", linewidth=1.1, linestyle="--",
        label=f"P99 {p99_pct:.1f}%")
    axis.legend(loc="upper right", frameon=False, fontsize=8)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("KV Cache Usage (%)")
    axis.grid(axis="y", color="#c7c7c7", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    verdict = "PASS" if bool(summary["kv_usage_p99_pass"]) else "FAIL"
    axis.set_title(
        f"RT-only Decode Pool, trace-paced {summary['target_rps']:.3g} RPS  "
        f"(P99 {p99_pct:.1f}% vs {target_pct:.0f}% -> {verdict})")
    figure.tight_layout()
    figure.savefig(output_dir / "kv_block_usage.png", bbox_inches="tight")
    plt.close(figure)


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    trace, trace_meta = load_trace(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    engine: Any = None
    try:
        engine = build_engine(args)
        core = engine.engine
        if bool(core.scheduler_config.enable_decode_saturation):
            raise RuntimeError("decode saturation unexpectedly enabled")
        trace, sanitize_meta = sanitize_trace(
            trace, int(core.model_config.max_model_len))
        if len(trace) != args.num_req:
            raise ValueError(
                f"only {len(trace)} of num_req={args.num_req} RT requests "
                "fit the model context")
        clock = {"measurement_start_s": math.inf}
        all_rows: list[dict[str, Any]] = []
        request_records: dict[str, dict[str, Any]] = {}
        total_blocks, block_size = install_forward_sampler(
            engine, all_rows, clock, request_records)

        active_ids: set[str] = set()
        request_tasks: list[asyncio.Task[Any]] = []
        counters = {"submitted": 0}
        run_start = time.perf_counter()
        measurement_start = run_start + args.warmup_s
        from saturo.tools.trace_scaler import fixed_num_window_s
        send_window_s = fixed_num_window_s(
            args.num_req, args.target_rps)
        measurement_duration_s = send_window_s - args.warmup_s
        measurement_end = run_start + send_window_s
        clock["measurement_start_s"] = measurement_start
        injector = asyncio.create_task(inject_trace(
            engine, trace, run_start, active_ids,
            request_tasks, counters, request_records, clock))

        await asyncio.sleep(max(0.0, measurement_end - time.perf_counter()))
        await injector

        # No new arrivals are added here. This grace interval removes the
        # measurement-boundary ambiguity for requests submitted just before
        # ``measurement_end`` while preserving their actual queue state.
        if args.admission_drain_s > 0:
            await asyncio.sleep(args.admission_drain_s)

        for request_id in list(active_ids):
            try:
                await engine.abort(request_id)
            except Exception:
                pass
        if request_tasks:
            _, pending = await asyncio.wait(
                request_tasks, timeout=args.shutdown_grace_s)
            for task in pending:
                task.cancel()
            await asyncio.gather(*request_tasks, return_exceptions=True)

        rows = [
            row for row in all_rows
            if 0.0 <= float(row["relative_time_s"]) <= measurement_duration_s
        ]
        usage = [float(row["kv_usage_pct"]) for row in rows]
        used_blocks = [float(row["used_gpu_blocks"]) for row in rows]
        request_rows = [
            record for record in request_records.values()
            if (0.0 <= float(record["submit_relative_time_s"])
                < measurement_duration_s)
        ]
        request_rows.sort(key=lambda record: record["submit_relative_time_s"])
        immediate_count = sum(
            bool(record["immediate_admission"]) for record in request_rows)
        queued_count = sum(
            record["first_schedule_wall_time_s"] is not None
            and not bool(record["immediate_admission"])
            for record in request_rows)
        not_scheduled_count = sum(
            record["first_schedule_wall_time_s"] is None
            for record in request_rows)
        admission_wait_ms = [
            float(record["admission_wait_ms"])
            for record in request_rows
            if record["admission_wait_ms"] is not None
        ]
        # A not-yet-scheduled request is necessarily non-immediate. Assign at
        # least one queued step for the P99 pass/fail distribution.
        queue_steps = [
            (int(record["queue_steps_beyond_first"])
             if record["queue_steps_beyond_first"] is not None else
             max(int(record["waiting_schedule_observations"]), 1))
            for record in request_rows
        ]
        immediate_rate = immediate_count / max(len(request_rows), 1)
        summary = {
            "model": args.model,
            "pool": "decode",
            "workload": "rt_only",
            "be_requests": 0,
            "arrival_mode": "scaled_original_trace",
            "num_req": args.num_req,
            "target_rps": args.target_rps,
            "warmup_s": args.warmup_s,
            "send_window_s": send_window_s,
            "measurement_s": measurement_duration_s,
            "fixed_token_budget": args.token_budget,
            "max_num_seqs": args.max_num_seqs,
            "decode_saturation_enabled": False,
            "future_arrival_prediction_enabled": False,
            "remaining_output_prediction_enabled": False,
            "total_gpu_blocks": total_blocks,
            "block_size": block_size,
            "forward_steps": len(rows),
            "submitted_requests_total": counters["submitted"],
            "admission_drain_s": args.admission_drain_s,
            "admission_requests_evaluated": len(request_rows),
            "immediate_admission_requests": immediate_count,
            "queued_admission_requests": queued_count,
            "not_scheduled_within_drain_requests": not_scheduled_count,
            "immediate_admission_rate": immediate_rate,
            "p99_queue_steps_beyond_first": percentile(queue_steps, 0.99),
            "p99_admission_wait_ms": percentile(admission_wait_ms, 0.99),
            "p99_no_queue_pass": (
                immediate_rate >= 0.99
                and percentile(queue_steps, 0.99) == 0.0),
            "kv_used_blocks_mean": statistics.fmean(used_blocks)
            if used_blocks else 0.0,
            "kv_usage_pct_p10": percentile(usage, 0.10),
            "kv_usage_pct_p50": percentile(usage, 0.50),
            "kv_usage_pct_p95": percentile(usage, 0.95),
            "kv_usage_pct_p99": percentile(usage, 0.99),
            "kv_usage_pct_peak": max(usage, default=0.0),
            "kv_usage_target_pct": args.kv_usage_target_pct,
            "kv_usage_p99_pass": (
                percentile(usage, 0.99) <= args.kv_usage_target_pct),
            "trace": {**trace_meta, **sanitize_meta},
        }
        write_outputs(output_dir, rows, request_rows, summary)
        print(json.dumps({
            "output_dir": str(output_dir),
            "forward_steps": len(rows),
            "kv_usage_pct_p10": summary["kv_usage_pct_p10"],
            "kv_usage_pct_p50": summary["kv_usage_pct_p50"],
            "kv_usage_pct_p95": summary["kv_usage_pct_p95"],
            "kv_usage_pct_p99": summary["kv_usage_pct_p99"],
            "kv_usage_pct_peak": summary["kv_usage_pct_peak"],
            "kv_usage_target_pct": summary["kv_usage_target_pct"],
            "kv_usage_p99_pass": summary["kv_usage_p99_pass"],
            "immediate_admission_rate": summary[
                "immediate_admission_rate"],
            "p99_queue_steps_beyond_first": summary[
                "p99_queue_steps_beyond_first"],
            "p99_no_queue_pass": summary["p99_no_queue_pass"],
        }, indent=2), flush=True)
        return summary
    finally:
        if engine is not None:
            try:
                engine.shutdown_background_loop()
            except Exception:
                pass
            del engine
        gc.collect()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
