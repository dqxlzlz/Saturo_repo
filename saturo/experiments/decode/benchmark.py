#!/usr/bin/env python3
"""Benchmark an online RT-only decode-pool engine using AsyncLLMEngine.

This is an async/online rewrite of the original saturated decode benchmark.
Exactly ``--num-req`` RT requests are selected from the workload CSV and that
fixed prefix is scaled to ``--target-rps`` while preserving its burstiness.
Decode-only requests enter direct scheduler admission, which allocates prompt
KV blocks without a synthetic prefill hand-off.

The primary output is a self-contained Markdown report with per-request TBT,
decode-iteration latency, decode batch size, and estimated KV-cache occupancy.

Example::

    CUDA_VISIBLE_DEVICES=0,1 python experiments/saturo/benchmark_rt_decode_only.py \
        --model /models/Qwen2.5-72B-Instruct \
        --rt-workload-csv data/rt_workload.csv \
        --tensor-parallel-size 2 \
        --num-req 600 \
        --target-rps 10 \
        --warmup-s 10
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


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
class BenchmarkSamples:
    tbt_ms: list[float] = field(default_factory=list)
    decode_iteration_ms: list[float] = field(default_factory=list)
    decode_batch_sizes: list[float] = field(default_factory=list)
    kv_blocks: list[float] = field(default_factory=list)
    kv_utilization_pct: list[float] = field(default_factory=list)
    decode_token_events: int = 0
    completed_requests: int = 0
    submitted_requests: int = 0
    pure_decode_iterations: int = 0
    mixed_or_prefill_iterations: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an online RT-only decode workload on one Saturo "
            "decode-pool AsyncLLMEngine and write a Markdown TBT report."))

    model = parser.add_argument_group("model and engine")
    model.add_argument("--model", required=True,
                       help="Local model path or Hugging Face model ID.")
    model.add_argument("--dtype", default="auto")
    model.add_argument("--load-format", default="auto")
    model.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    model.add_argument("--max-model-len", type=int, default=None)
    model.add_argument(
        "--max-num-seqs", type=int, default=1024,
        help=(
            "Scheduler sequence limit. Keep this high enough that KV memory, "
            "rather than the default vLLM limit, can determine peak batch size."))
    model.add_argument(
        "--max-num-batched-tokens", type=int, default=None,
        help="Optional vLLM token budget override.")
    model.add_argument(
        "--disable-inductor", action=argparse.BooleanOptionalAction,
        default=True, help="Match the default Saturo worker configuration.")
    model.add_argument(
        "--runtime-profile", default="",
        help="Optional Saturo time-predictor profile path.")
    model.add_argument(
        "--scheduler-tbt-ms", type=float, default=1000.0,
        help=(
            "Internal D-pool saturation-policy target. RT-only requests are "
            "fully admitted, so this does not cap the measured RT batch."))

    workload = parser.add_argument_group("workload and measurement")
    workload.add_argument("--rt-workload-csv", required=True)
    workload.add_argument(
        "--num-req", type=int, required=True,
        help="Replay exactly this many chronological RT requests.")
    workload.add_argument(
        "--target-rps", type=float, required=True,
        help="Target RPS for the selected fixed request set.")
    workload.add_argument("--warmup-s", type=float, default=10.0)
    workload.add_argument(
        "--max-wall-s", type=float, default=600.0,
        help="Safety timeout covering fill, warmup, measurement, and drain.")
    workload.add_argument(
        "--shutdown-grace-s", type=float, default=30.0,
        help="Max time to wait for in-flight requests after measurement end.")
    workload.add_argument(
        "--slo-headroom", type=float, default=1.20,
        help="Multiplier applied to measured request-level P99 for the report's candidate SLO.")
    workload.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (reserved; arrival replay is deterministic).")

    parallel = parser.add_argument_group("hardware and parallelism")
    parallel.add_argument(
        "--gpu-ids", default="",
        help=(
            "Comma-separated physical GPU IDs. If omitted, preserve the "
            "existing CUDA_VISIBLE_DEVICES setting."))
    parallel.add_argument("--tensor-parallel-size", type=int, default=1)
    parallel.add_argument("--pipeline-parallel-size", type=int, default=1)
    parallel.add_argument("--expert-parallel-size", type=int, default=0)

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-md", default="",
        help=(
            "Markdown report path. Default: experiments/saturo/results/"
            "rt_decode_only_<timestamp>.md"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be positive")
    if args.max_num_batched_tokens is not None \
            and args.max_num_batched_tokens <= 0:
        raise ValueError("--max-num-batched-tokens must be positive")
    if args.tensor_parallel_size <= 0 or args.pipeline_parallel_size <= 0:
        raise ValueError("TP and PP sizes must be positive")
    if args.expert_parallel_size < 0:
        raise ValueError("--expert-parallel-size cannot be negative")
    if args.warmup_s < 0 or args.max_wall_s <= 0:
        raise ValueError("warmup must be non-negative; max wall must be positive")
    if args.shutdown_grace_s < 0:
        raise ValueError("--shutdown-grace-s must be non-negative")
    if args.slo_headroom <= 0:
        raise ValueError("--slo-headroom must be positive")
    if args.num_req <= 0:
        raise ValueError("--num-req must be positive")
    if args.target_rps <= 0:
        raise ValueError("--target-rps must be positive")
    from saturo.tools.trace_scaler import fixed_num_window_s
    if args.warmup_s >= fixed_num_window_s(
            args.num_req, args.target_rps):
        raise ValueError(
            "--warmup-s must be smaller than num_req / target_rps")
    required_gpus = args.tensor_parallel_size * args.pipeline_parallel_size
    if args.gpu_ids:
        gpu_ids = [item.strip() for item in args.gpu_ids.split(",")
                   if item.strip()]
        if len(gpu_ids) < required_gpus:
            raise ValueError(
                f"TP*PP requires {required_gpus} visible GPUs, but --gpu-ids "
                f"contains only {len(gpu_ids)}")


def default_output_path() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "experiments" / "saturo" / "results" /
            f"rt_decode_only_{stamp}.md")


def percentile(values: Iterable[float], q: float) -> float:
    """Nearest-rank percentile, matching Saturo's existing report helpers."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    rank = int(math.ceil(q * len(ordered))) - 1
    return ordered[min(len(ordered) - 1, max(0, rank))]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p95": percentile(data, 0.95),
        "p99": percentile(data, 0.99),
        "max": max(data),
    }


def load_trace(path: Path, num_req: int,
               target_rps: float) -> list[TraceRequest]:
    from saturo.runtime.workload import load_rt_workload_csv

    if not path.is_file():
        raise FileNotFoundError(f"RT workload CSV not found: {path}")
    loaded = load_rt_workload_csv(
        str(path), num_req=num_req, target_rps=target_rps)
    trace = [
        TraceRequest(
            request_id=req.request_id,
            prompt_len=req.prompt_len,
            output_len=req.output_len,
            arrival_offset_s=float(req.arrival_offset_s),
        )
        for req in loaded
    ]
    if not trace:
        raise ValueError(f"RT workload CSV is empty: {path}")
    return trace


def sanitize_trace(
    trace: list[TraceRequest],
    max_model_len: int,
) -> tuple[list[TraceRequest], dict[str, int]]:
    """Clamp completion lengths and reject prompts that cannot enter vLLM."""
    usable: list[TraceRequest] = []
    skipped_prompt_too_long = 0
    clamped_output = 0
    for req in trace:
        if req.prompt_len >= max_model_len:
            skipped_prompt_too_long += 1
            continue
        max_output = max_model_len - req.prompt_len
        output_len = min(req.output_len, max_output)
        if output_len != req.output_len:
            clamped_output += 1
        usable.append(TraceRequest(
            request_id=req.request_id,
            prompt_len=req.prompt_len,
            output_len=max(1, output_len),
            arrival_offset_s=req.arrival_offset_s,
        ))
    if not usable:
        raise ValueError(
            "No workload request fits the engine's max_model_len="
            f"{max_model_len}")
    return usable, {
        "source_rows": len(trace),
        "usable_rows": len(usable),
        "skipped_prompt_too_long": skipped_prompt_too_long,
        "clamped_output_len": clamped_output,
    }


def build_async_engine(args: argparse.Namespace) -> Any:
    """Build a Saturo D-pool ``AsyncLLMEngine`` with direct admission.

    Mirrors the DECODE-role engine configuration used by
    ``saturo.runtime.worker.RayEngineWorker`` (priority scheduling, decode
    saturation, eager mode), but exposes it through ``AsyncLLMEngine`` so
    requests can be streamed concurrently like in ``benchmark_serving.py``.
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
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "load_format": args.load_format,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_custom_all_reduce": True,
        "scheduling_policy": "priority",
        "enable_chunked_prefill": False,
        "max_num_seqs": args.max_num_seqs,
        "disable_log_requests": True,
        "disable_log_stats": True,
        "enable_decode_saturation": True,
        "decode_saturation_tbt_ms": args.scheduler_tbt_ms,
        "decode_saturation_require_model_len_predictor": False,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    profile = str(args.runtime_profile or "")
    if profile:
        engine_kwargs["time_predictor_profile"] = profile
    if bool(args.disable_inductor):
        engine_kwargs["compilation_config"] = CompilationConfig(
            level=0,
            custom_ops=["none"],
            use_cudagraph=False,
            use_inductor=False,
        )
    return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))


def blocks_in_decode_iteration(event: dict[str, Any], block_size: int) -> int:
    total = 0
    for _query_len, context_len in event.get("decode_reqs", []):
        total += max(1, math.ceil(int(context_len) / block_size))
    return total


async def rt_request_task(
    engine: Any,
    req: TraceRequest,
    serial: int,
    samples: BenchmarkSamples,
    measurement_start_s: float,
    measurement_end_s: float,
) -> None:
    """Stream one RT decode request and record per-token TBT."""
    from vllm import SamplingParams

    rid = f"rt_{serial}:{req.request_id}"
    submit_s = time.perf_counter()
    in_window = measurement_start_s <= submit_s < measurement_end_s
    if in_window:
        samples.submitted_requests += 1

    params = SamplingParams(
        temperature=0.0,
        # Direct admission seeds one hidden output token as the first decode
        # input; request one extra token and exclude the seed below.
        max_tokens=req.output_len + 1,
        ignore_eos=True,
        detokenize=False,
        is_realtime=True,
        prefill_ddl=time.time() + 3600.0,
        extra_args={"saturo_direct_decode": True},
    )
    # Direct admission uses prompt length for local block allocation; token
    # values are irrelevant because this benchmark does not validate text.
    prompt = {"prompt_token_ids": [1] * max(1, req.prompt_len)}

    last_seen = 0
    last_token_s: Optional[float] = None
    finished_cleanly = False
    try:
        async for out in engine.generate(prompt, params, rid, priority=0):
            if not out.outputs:
                continue
            token_count = max(len(out.outputs[0].token_ids) - 1, 0)
            if token_count <= last_seen:
                continue
            delta = max(1, token_count - last_seen)
            last_seen = token_count
            now_s = time.perf_counter()
            if (last_token_s is not None
                    and measurement_start_s is not None
                    and last_token_s >= measurement_start_s
                    and now_s < measurement_end_s):
                per_token_tbt_ms = (now_s - last_token_s) * 1000.0 / delta
                samples.tbt_ms.extend([per_token_tbt_ms] * delta)
                samples.decode_token_events += delta
            last_token_s = now_s
        finished_cleanly = True
    finally:
        if finished_cleanly:
            done_s = time.perf_counter()
            if (measurement_start_s is not None
                    and measurement_start_s <= done_s < measurement_end_s):
                samples.completed_requests += 1


async def iteration_log_collector(
    engine: Any,
    samples: BenchmarkSamples,
    measurement_start_s: float,
    measurement_end_s: float,
    stop_event: asyncio.Event,
    block_size: int,
    total_gpu_blocks: int,
) -> None:
    """Poll ``engine.engine.iteration_logs`` for decode batch/KV stats."""
    last_idx = 0
    while not stop_event.is_set():
        core = engine.engine
        logs = getattr(core, "iteration_logs", [])
        while last_idx < len(logs):
            event = logs[last_idx]
            last_idx += 1
            observed_s = time.perf_counter()
            if not (measurement_start_s <= observed_s < measurement_end_s):
                continue
            n_decode = int(event.get("n_decode_reqs", 0))
            n_prefill = int(event.get("n_prefill_reqs", 0))
            if n_decode > 0 and n_prefill == 0:
                samples.pure_decode_iterations += 1
                samples.decode_iteration_ms.append(
                    float(event.get("actual_time_ms", 0.0)))
                samples.decode_batch_sizes.append(float(n_decode))
                used_blocks = blocks_in_decode_iteration(event, block_size)
                samples.kv_blocks.append(float(used_blocks))
                if total_gpu_blocks > 0:
                    samples.kv_utilization_pct.append(
                        100.0 * used_blocks / total_gpu_blocks)
            elif n_decode > 0 or n_prefill > 0:
                samples.mixed_or_prefill_iterations += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.01)
        except asyncio.TimeoutError:
            pass


async def request_injector(
    engine: Any,
    trace: list[TraceRequest],
    replay_start_s: float,
    measurement_start_s: float,
    measurement_end_s: float,
    samples: BenchmarkSamples,
    tasks: list[asyncio.Task],
) -> None:
    """Replay each selected RT request exactly once."""
    for serial, req in enumerate(trace):
        target_arrival_s = replay_start_s + req.arrival_offset_s
        now = time.perf_counter()
        if target_arrival_s > now:
            await asyncio.sleep(target_arrival_s - now)
        task = asyncio.create_task(rt_request_task(
            engine, req, serial, samples,
            measurement_start_s, measurement_end_s))
        tasks.append(task)


async def benchmark(
    engine: Any,
    trace: list[TraceRequest],
    args: argparse.Namespace,
) -> tuple[BenchmarkSamples, dict[str, Any]]:
    """Drive the AsyncLLMEngine by replaying the (scaled) trace arrivals."""
    from saturo.tools.trace_scaler import fixed_num_window_s

    core = engine.engine
    cache_config = core.cache_config
    block_size = int(cache_config.block_size)
    total_gpu_blocks = int(cache_config.num_gpu_blocks)

    samples = BenchmarkSamples()
    run_start = time.perf_counter()
    measurement_start_s = run_start + args.warmup_s
    send_window_s = fixed_num_window_s(args.num_req, args.target_rps)
    measurement_end_s = run_start + send_window_s
    hard_deadline = run_start + args.max_wall_s

    stop_collector = asyncio.Event()
    tasks: list[asyncio.Task] = []

    collector = asyncio.create_task(iteration_log_collector(
        engine, samples, measurement_start_s, measurement_end_s,
        stop_collector, block_size, total_gpu_blocks))
    injector = asyncio.create_task(request_injector(
        engine, trace, run_start,
        measurement_start_s, measurement_end_s, samples, tasks))

    # Wait until the measurement window ends.
    now = time.perf_counter()
    if measurement_end_s > now:
        await asyncio.sleep(measurement_end_s - now)

    injector.cancel()
    try:
        await injector
    except asyncio.CancelledError:
        pass

    # Drain in-flight requests.
    timeout = max(0.0, min(hard_deadline, measurement_end_s + args.shutdown_grace_s)
                  - time.perf_counter())
    if tasks:
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    stop_collector.set()
    await collector

    actual_measurement_s = max(
        0.0,
        min(time.perf_counter(), measurement_end_s) - measurement_start_s,
    )
    metadata = {
        "target_rps": args.target_rps,
        "num_req": args.num_req,
        "send_window_s": send_window_s,
        "effective_replay_rps": args.target_rps,
        "actual_measurement_s": actual_measurement_s,
        "total_gpu_blocks": total_gpu_blocks,
        "block_size": block_size,
    }
    return samples, metadata


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    output = completed.stdout.strip()
    return output if completed.returncode == 0 and output else "unavailable"


def collect_environment(engine: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    core = engine.engine
    scheduler_config = core.scheduler_config
    cache_config = core.cache_config
    gpu_rows = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_rows.append({
            "visible_index": index,
            "name": properties.name,
            "memory_gib": properties.total_memory / (1024 ** 3),
            "compute_capability": (
                f"{properties.major}.{properties.minor}"),
        })
    try:
        import vllm
        vllm_version = getattr(vllm, "__version__", "unknown")
    except Exception:
        vllm_version = "unknown"

    git_commit = command_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"])
    git_dirty = command_output(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"])
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda or "unknown",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "gpu_rows": gpu_rows,
        "driver": command_output([
            "nvidia-smi", "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]).splitlines()[0],
        "vllm": vllm_version,
        "git_commit": git_commit,
        "git_dirty": git_dirty != "unavailable" and bool(git_dirty),
        "model": args.model,
        "dtype": str(core.model_config.dtype),
        "load_format": args.load_format,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "expert_parallel_size": args.expert_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": core.model_config.max_model_len,
        "max_num_seqs": scheduler_config.max_num_seqs,
        "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
        "num_gpu_blocks": cache_config.num_gpu_blocks,
        "block_size": cache_config.block_size,
        "enforce_eager": True,
        "disable_inductor": args.disable_inductor,
        "direct_decode_admission": True,
        "decode_saturation_enabled": True,
        "scheduler_tbt_ms": args.scheduler_tbt_ms,
        "runtime_profile": args.runtime_profile or "none",
    }


def markdown_table(summary: dict[str, float | int], unit: str) -> str:
    def value(key: str) -> str:
        number = summary[key]
        if key == "count":
            return str(int(number))
        return f"{float(number):.3f}"

    return (
        f"| Samples | Mean ({unit}) | Median ({unit}) | P95 ({unit}) | "
        f"P99 ({unit}) | Max ({unit}) |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"| {value('count')} | {value('mean')} | {value('median')} | "
        f"{value('p95')} | {value('p99')} | {value('max')} |")


def candidate_slo_ms(p99_ms: float, headroom: float) -> float:
    if p99_ms <= 0:
        return 0.0
    return math.ceil((p99_ms * headroom) / 5.0) * 5.0


def render_report(
    args: argparse.Namespace,
    source_path: Path,
    trace_meta: dict[str, int],
    environment: dict[str, Any],
    samples: BenchmarkSamples,
    benchmark_meta: dict[str, Any],
) -> str:
    tbt = summarize(samples.tbt_ms)
    iteration = summarize(samples.decode_iteration_ms)
    batch = summarize(samples.decode_batch_sizes)
    kv_blocks = summarize(samples.kv_blocks)
    kv_pct = summarize(samples.kv_utilization_pct)
    suggested_slo = candidate_slo_ms(float(tbt["p99"]), args.slo_headroom)
    duration = float(benchmark_meta["actual_measurement_s"])
    throughput = (samples.decode_token_events / duration
                  if duration > 0 else 0.0)
    effective_rps = benchmark_meta["effective_replay_rps"]
    rate_text = f"{effective_rps:.2f}"
    target_rps = benchmark_meta["target_rps"]
    scale_text = (
        f"num_req={benchmark_meta['num_req']}, "
        f"target_rps={target_rps:.2f}")

    gpu_lines = []
    for gpu in environment["gpu_rows"]:
        gpu_lines.append(
            f"- Visible GPU {gpu['visible_index']}: {gpu['name']}, "
            f"{gpu['memory_gib']:.2f} GiB, compute capability "
            f"{gpu['compute_capability']}")
    if not gpu_lines:
        gpu_lines.append("- GPU details unavailable")

    warnings = []
    if int(tbt["count"]) < 1000:
        warnings.append(
            f"Only {tbt['count']} TBT samples were collected; run longer "
            "before using tail percentiles as a production SLO.")
    if samples.pure_decode_iterations < 100:
        warnings.append(
            f"Only {samples.pure_decode_iterations} pure-decode iterations "
            "were collected.")
    if float(kv_pct["max"]) < 90.0:
        warnings.append(
            f"Estimated peak scheduled KV occupancy was {kv_pct['max']:.1f}%; "
            "the run may have hit max_num_seqs before filling KV memory. "
            "Increase --max-num-seqs or --target-rps to raise the load.")
    if trace_meta["clamped_output_len"]:
        warnings.append(
            f"{trace_meta['clamped_output_len']} source rows had output_len "
            "clamped to max_model_len.")
    if trace_meta["skipped_prompt_too_long"]:
        warnings.append(
            f"{trace_meta['skipped_prompt_too_long']} source rows were skipped "
            "because prompt_len >= max_model_len.")
    warning_text = "\n".join(f"- {warning}" for warning in warnings)
    if not warning_text:
        warning_text = "- No automatic validity warning was triggered."

    git_state = f"{environment['git_commit']}"
    if environment["git_dirty"]:
        git_state += " (dirty worktree)"

    return f"""# Saturo RT-only Decode Pool Online TBT Benchmark

Generated at: {environment['timestamp']}

## Result summary

The request-level online TBT P99 was **{tbt['p99']:.3f} ms**. With the
configured `{args.slo_headroom:.2f}x` headroom and rounding up to the next
5 ms, the candidate TBT-SLO is **{suggested_slo:.0f} ms**. This is a baseline
for this exact model/hardware/parallel configuration, not a cross-platform
SLO.

Traffic was replayed from the CSV arrival timestamps at an effective
**{rate_text} RPS** ({scale_text}).
Decode throughput during the measurement window was approximately
**{throughput:.2f} observed RT token events/s**. The pure-decode batch size
averaged **{batch['mean']:.2f}** and peaked at **{batch['max']:.0f}**.

## Request-level TBT

TBT is the wall-clock interval between two adjacent emitted tokens of the same
request. Both tokens must fall after warmup. It includes scheduler and output
processing overhead between `engine.step()` calls.

{markdown_table(tbt, 'ms')}

## Pure-decode iteration latency

This is CUDA-event timing around the model execution for iterations containing
decode requests and no prefill request.

{markdown_table(iteration, 'ms')}

## Decode batch size

{markdown_table(batch, 'requests')}

## Scheduled KV occupancy estimate

Occupancy is reconstructed from the context lengths in each pure-decode
iteration as `sum(ceil(context_len / block_size))`. It describes scheduled RT
sequences; it does not include allocator metadata or transient reservations.

{markdown_table(kv_blocks, 'blocks')}

{markdown_table(kv_pct, '%')}

## Workload and run conditions

- RT workload CSV: `{source_path}`
- CSV rows read / usable: {trace_meta['source_rows']} / {trace_meta['usable_rows']}
- Arrival pattern: trace replay ({scale_text}) at effective {rate_text} RPS
- Online future-arrival prediction: enabled (ARIMA)
- Workload recycling: single pass
- Submitted / completed request instances in window: {samples.submitted_requests} / {samples.completed_requests}
- Warmup / derived sending window: {args.warmup_s:.3f} s / {benchmark_meta['send_window_s']:.3f} s
- Actual measurement window: {duration:.3f} s
- Pure-decode / mixed-or-prefill iterations in window: {samples.pure_decode_iterations} / {samples.mixed_or_prefill_iterations}

## Model and engine configuration

- Model: `{environment['model']}`
- Effective dtype / load format: `{environment['dtype']}` / `{environment['load_format']}`
- Parallelism: TP={environment['tensor_parallel_size']}, PP={environment['pipeline_parallel_size']}, EP={environment['expert_parallel_size']}
- Effective max model length: {environment['max_model_len']} tokens
- Effective max sequences / batched tokens: {environment['max_num_seqs']} / {environment['max_num_batched_tokens']}
- KV cache: {environment['num_gpu_blocks']} GPU blocks, block size {environment['block_size']} tokens
- GPU memory utilization target: {environment['gpu_memory_utilization']:.3f}
- Eager mode / Inductor disabled: {environment['enforce_eager']} / {environment['disable_inductor']}
- Saturo decode saturation enabled: {environment['decode_saturation_enabled']}
- Internal scheduler TBT target: {environment['scheduler_tbt_ms']:.3f} ms (does not throttle RT admission)
- Time predictor profile: `{environment['runtime_profile']}`
- Direct decode admission: {environment['direct_decode_admission']}

Direct admission allocates blocks using each request's full prompt length and
skips prefill/KV transfer entirely. Generated token values are intentionally
invalid; decode scheduling, KV footprint, and model-forward timing remain the
objects under test.

## Hardware and software

- Host: `{environment['hostname']}`
- Platform: `{environment['platform']}`
- CUDA_VISIBLE_DEVICES: `{environment['cuda_visible_devices']}`
- NVIDIA driver / CUDA runtime: `{environment['driver']}` / `{environment['cuda_runtime']}`
{chr(10).join(gpu_lines)}
- Python / PyTorch / vLLM: `{environment['python']}` / `{environment['torch']}` / `{environment['vllm']}`
- Git revision: `{git_state}`

## Validity notes

{warning_text}

- The candidate SLO formula is `ceil_to_5ms(P99 TBT * {args.slo_headroom:.2f})`.
- Repeat the run several times and use the worst stable P99 before fixing the
  later experiment's TBT-SLO.
- Do not compare this result directly with an engine that performs real
  prefill on the decode GPUs or uses a different TP/PP, dtype, model length,
  KV-cache size, CUDA graph mode, or request-length distribution.
"""


async def run_benchmark(args: argparse.Namespace) -> None:
    source_path = Path(args.rt_workload_csv).expanduser().resolve()
    output_path = (Path(args.output_md).expanduser().resolve()
                   if args.output_md else default_output_path())
    trace = load_trace(
        source_path, num_req=args.num_req, target_rps=args.target_rps)
    print(f"Loaded {len(trace)} RT rows from {source_path}", flush=True)
    print(
        f"Initializing one Saturo D-pool AsyncLLMEngine: model={args.model}, "
        f"TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}, "
        f"EP={args.expert_parallel_size}",
        flush=True,
    )

    engine: Any = None
    try:
        engine = build_async_engine(args)
        core = engine.engine
        max_model_len = int(core.model_config.max_model_len)
        trace, trace_meta = sanitize_trace(trace, max_model_len)
        if len(trace) != args.num_req:
            raise ValueError(
                f"only {len(trace)} of num_req={args.num_req} RT requests "
                f"fit max_model_len={max_model_len}")
        cache = core.cache_config
        print(
            f"Engine ready: num_gpu_blocks={cache.num_gpu_blocks}, "
            f"block_size={cache.block_size}, max_model_len={max_model_len}",
            flush=True,
        )
        environment = collect_environment(engine, args)
        samples, benchmark_meta = await benchmark(engine, trace, args)
        report = render_report(
            args=args,
            source_path=source_path,
            trace_meta=trace_meta,
            environment=environment,
            samples=samples,
            benchmark_meta=benchmark_meta,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        tbt = summarize(samples.tbt_ms)
        batch = summarize(samples.decode_batch_sizes)
        print(json.dumps({
            "output_md": str(output_path),
            "tbt_ms": tbt,
            "decode_batch_size": batch,
            "candidate_tbt_slo_ms": candidate_slo_ms(
                float(tbt["p99"]), args.slo_headroom),
        }, indent=2), flush=True)
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
    validate_args(args)
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

    if args.seed:
        np.random.seed(args.seed)

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
