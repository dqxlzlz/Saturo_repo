#!/usr/bin/env python3
"""Profile controlled decode batches on a plain vLLM AsyncLLMEngine.

The engine uses DummyConnector only to materialize the requested prompt
context without running prefill. All measured iterations are real, eager
decode forwards. Saturo's global scheduler and saturation policy are not used.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_COLLECT_REAL_MODULE_TIMES", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("SATURO_DUMMY_CONNECTOR_BROADCAST_HIDDEN", "1")
os.environ.setdefault("SATURO_PROFILE_RUN_MAX_TOKENS", "8192")

from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.usage.usage_lib import UsageContext


METRICS = (
    "actual_time_ms",
    "step_wall_time_ms",
    "runtime_overhead_ms",
    "scheduler_wall_ms",
    "actual_linear_ms",
    "actual_decode_attn_ms",
    "actual_combined_attn_ms",
    "actual_profiled_events_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/root/huggingface_config/Mistral-Large-Instruct-2411")
    parser.add_argument(
        "--batch-sizes", nargs=3, type=int,
        metavar=("START", "STOP", "STEP"), default=(8, 1025, 8))
    parser.add_argument("--batch-size-values", nargs="+", type=int)
    parser.add_argument("--prompt-len", type=int, default=0)
    parser.add_argument(
        "--rt-workload-csv", default="data/datasets/rt_workload.csv")
    parser.add_argument("--decode-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--batch-timeout-s", type=float, default=600.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--load-format", default="dummy")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def mean_prompt_len(path: str) -> int:
    with open(path, newline="") as stream:
        values = [
            int(float(row["prompt_len"]))
            for row in csv.DictReader(stream)
        ]
    if not values:
        raise ValueError(f"empty RT workload: {path}")
    return int(round(statistics.fmean(values)))


def make_engine(args: argparse.Namespace, max_batch: int,
                prompt_len: int) -> AsyncLLMEngine:
    # One dummy-prefill scheduling iteration must be able to admit the entire
    # controlled batch. The actual prefill model forward is bypassed.
    max_num_batched_tokens = max(8192, max_batch * prompt_len)
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,
        load_format=args.load_format,
        dtype=args.dtype,
        enforce_eager=True,
        enable_chunked_prefill=False,
        max_model_len=args.max_model_len,
        max_num_seqs=max_batch,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        disable_log_requests=True,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="DummyConnector",
            kv_role="kv_consumer",
        ),
    )
    return AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.LLM_CLASS)


async def consume(stream: Any) -> int:
    generated = 0
    async for output in stream:
        if output.outputs:
            generated = len(output.outputs[0].token_ids)
    return generated


async def wait_for_final_step(engine: AsyncLLMEngine, start_index: int) -> None:
    deadline = time.perf_counter() + 30.0
    while time.perf_counter() < deadline:
        logs = engine.engine.iteration_logs[start_index:]
        if logs and "step_wall_time_ms" in logs[-1]:
            return
        await asyncio.sleep(0.01)


def flatten_log(batch_size: int, repeat: int, log: dict[str, Any]) -> dict:
    modules = log.get("module_profile_times") or {}
    row = {
        "requested_batch_size": batch_size,
        "repeat": repeat,
        **{
            key: value
            for key, value in log.items()
            if key != "module_profile_times"
        },
    }
    row.update(modules)
    return row


async def run_batch(
    engine: AsyncLLMEngine,
    *,
    batch_size: int,
    prompt_len: int,
    decode_steps: int,
    warmup_steps: int,
    timeout_s: float,
    repeat: int,
) -> tuple[list[dict], list[dict], dict]:
    core = engine.engine
    log_start = len(core.iteration_logs)
    # DummyConnector's handoff/prefill transition samples token 1. Request
    # decode_steps+1 tokens so exactly decode_steps real decode forwards follow.
    params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=decode_steps + 1,
        detokenize=False,
        extra_args={"saturo_dummy_kv_decode": True},
    )
    prompt = {"prompt_token_ids": [1] * prompt_len}

    # add_request() has no internal suspension after the background task is
    # created, so all streams enter RequestTracker before the engine runs.
    streams = []
    for index in range(batch_size):
        stream = await engine.add_request(
            request_id=f"decode-b{batch_size}-r{repeat}-{index}",
            prompt=prompt,
            params=params,
        )
        streams.append(stream)

    started = time.perf_counter()
    consumers = asyncio.gather(*(consume(stream) for stream in streams))
    while not consumers.done():
        done, _ = await asyncio.wait({consumers}, timeout=10.0)
        if done:
            break
        scheduler = core.scheduler[0]
        print(
            f"  waiting for batch={batch_size}: "
            f"waiting={len(scheduler.waiting)} "
            f"running={len(scheduler.running)} "
            f"swapped={len(scheduler.swapped)} "
            f"new_logs={len(core.iteration_logs) - log_start}",
            flush=True,
        )
        if time.perf_counter() - started > timeout_s:
            consumers.cancel()
            raise TimeoutError(
                f"batch {batch_size} exceeded {timeout_s}s")
    generated = await consumers
    elapsed_s = time.perf_counter() - started
    await wait_for_final_step(engine, log_start)

    raw_logs = [
        flatten_log(batch_size, repeat, item)
        for item in core.iteration_logs[log_start:]
    ]
    exact_decode_logs = [
        row for row in raw_logs
        if int(row.get("n_prefill_reqs", 0)) == 0
        and int(row.get("n_decode_reqs", 0)) == batch_size
    ]
    measured = exact_decode_logs[warmup_steps:warmup_steps + decode_steps]
    metadata = {
        "batch_size": batch_size,
        "repeat": repeat,
        "elapsed_s": elapsed_s,
        "generated_tokens_min": min(generated, default=0),
        "generated_tokens_max": max(generated, default=0),
        "total_iteration_logs": len(raw_logs),
        "exact_decode_iteration_logs": len(exact_decode_logs),
        "measured_decode_iteration_logs": len(measured),
        "preempted_total": sum(
            int(row.get("preempted", 0)) for row in raw_logs),
        "decode_batch_min": min(
            (int(row.get("n_decode_reqs", 0)) for row in exact_decode_logs),
            default=0,
        ),
        "decode_batch_max": max(
            (int(row.get("n_decode_reqs", 0)) for row in exact_decode_logs),
            default=0,
        ),
    }
    return raw_logs, measured, metadata


def summarize(batch_size: int, rows: list[dict], metadata: dict) -> dict:
    result = dict(metadata)
    result["batch_size"] = batch_size
    contexts = [
        float(row.get("decode_sum_ctx", 0.0))
        / max(float(row.get("n_decode_reqs", 0.0)), 1.0)
        for row in rows
    ]
    result["context_len_mean"] = (
        statistics.fmean(contexts) if contexts else "")
    for metric in METRICS:
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric) is not None
        ]
        result[f"{metric}_mean"] = (
            statistics.fmean(values) if values else "")
        result[f"{metric}_p50"] = (
            statistics.median(values) if values else "")
        result[f"{metric}_p95"] = (
            sorted(values)[min(len(values) - 1, int(0.95 * len(values)))]
            if values else "")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    args = parse_args()
    batch_sizes = (
        args.batch_size_values
        if args.batch_size_values is not None
        else list(range(*args.batch_sizes))
    )
    if not batch_sizes:
        raise ValueError("--batch-sizes produced an empty range")
    if args.decode_steps <= 0 or args.warmup_steps < 0:
        raise ValueError("decode/warmup steps must be nonnegative")
    prompt_len = args.prompt_len or mean_prompt_len(args.rt_workload_csv)
    if prompt_len + args.decode_steps + 1 > args.max_model_len:
        raise ValueError("prompt + generated tokens exceed max model length")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or (
        "experiments/saturo/results/plain_vllm_decode/"
        f"tp{args.tensor_parallel_size}_{timestamp}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "batch_size_values": batch_sizes,
        "resolved_prompt_len": prompt_len,
        "sampling_max_tokens": args.decode_steps + 1,
        "measurement_note": (
            "The DummyConnector handoff token is excluded; each row averages "
            "real eager decode forwards only."),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n")

    print(f"Output: {output_dir}", flush=True)
    print(
        f"Prompt={prompt_len}; decode_steps={args.decode_steps}; "
        f"batch={batch_sizes[0]}..{batch_sizes[-1]}", flush=True)
    engine = make_engine(args, max(batch_sizes), prompt_len)
    all_iterations: list[dict] = []
    summaries: list[dict] = []
    try:
        for batch_size in batch_sizes:
            raw, measured, metadata = await run_batch(
                engine,
                batch_size=batch_size,
                prompt_len=prompt_len,
                decode_steps=args.decode_steps,
                warmup_steps=args.warmup_steps,
                timeout_s=args.batch_timeout_s,
                repeat=1,
            )
            all_iterations.extend(raw)
            row = summarize(batch_size, measured, metadata)
            summaries.append(row)
            write_csv(output_dir / "summary.csv", summaries)
            with (output_dir / "iterations.jsonl").open("a") as stream:
                for item in raw:
                    stream.write(json.dumps(item) + "\n")
            print(
                f"batch={batch_size:4d} samples="
                f"{row['measured_decode_iteration_logs']:3d} "
                f"forward={row.get('actual_time_ms_mean', 0):8.2f}ms "
                f"linear={row.get('actual_linear_ms_mean', 0):8.2f}ms "
                f"decode_attn="
                f"{row.get('actual_decode_attn_ms_mean', 0):7.2f}ms "
                f"ctx={row.get('context_len_mean', 0):7.1f} "
                f"preempted={row['preempted_total']}",
                flush=True,
            )
    finally:
        engine.shutdown_background_loop()

    print(f"Saved {len(summaries)} batch points to {output_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
