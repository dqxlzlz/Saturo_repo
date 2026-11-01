#!/usr/bin/env python3
"""Profile serial prefill requests on a plain vLLM AsyncLLMEngine.

Two workloads are supported:

* a CSV trace (for example the first 3,000 RT requests); and
* a fixed prompt-length sweep used to measure isolated BE chunk cost.

Requests are submitted and awaited one at a time.  Together with
``max_num_seqs=1`` this guarantees that every recorded prefill iteration
contains exactly one request, so ``actual_time_ms`` is an unambiguous GPU
forward time for that request/chunk.
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "0")
# This benchmark measures the engine itself and must not trigger Saturo's
# automatic latency-profile sweep during engine construction.
os.environ.setdefault("SATURO_SKIP_TIME_PREDICTOR_PROFILE", "1")
# Per-module CUDA events are useful for attribution, but materially perturb a
# short isolated forward.  The engine-level CUDA event that produces
# ``actual_time_ms`` is always present and is the clean timing source here.
os.environ.setdefault("VLLM_COLLECT_REAL_MODULE_TIMES", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
torch_lib = (
    Path(sys.prefix) / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages" / "torch" / "lib"
)
if __name__ == "__main__" and torch_lib.is_dir():
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if current_ld_path.split(":")[0] != str(torch_lib):
        updated_env = os.environ.copy()
        updated_env["LD_LIBRARY_PATH"] = (
            f"{torch_lib}:{current_ld_path}"
            if current_ld_path else str(torch_lib))
        os.execve(sys.executable, [sys.executable, *sys.argv], updated_env)

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.usage.usage_lib import UsageContext


METRICS = (
    "ttft_ms",
    "total_ms",
    "actual_time_ms",
    "step_wall_time_ms",
    "runtime_overhead_ms",
    "scheduler_wall_ms",
    "actual_linear_ms",
    "actual_prefill_attn_ms",
    "actual_combined_attn_ms",
    "actual_profiled_events_ms",
)

DEFAULT_OUTPUT_ROOT = Path("experiments/results/ttft_benchmark")
TRACE_CSV_FIELDS = (
    "request_id",
    "sequence_index",
    "source_request_id",
    "prompt_len",
    "source_output_len",
    "source_arrival_offset_s",
    "repeat",
    "generated_tokens",
    "ttft_ms",
    "total_ms",
    "iter_id",
    "prefill_tokens",
    "decode_tokens",
    "actual_time_ms",
    "step_wall_time_ms",
    "runtime_overhead_ms",
    "scheduler_wall_ms",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/root/huggingface_config/Mistral-Large-Instruct-2411")
    parser.add_argument("--lengths", nargs=3, type=int,
                        metavar=("START", "STOP", "STEP"),
                        default=(128, 1025, 128))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--trace-csv",
        help=("CSV workload with request_id,prompt_len,output_len columns. "
              "When set, --lengths and --repeats are ignored."),
    )
    parser.add_argument("--max-requests", type=int, default=3000)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--warmup-requests", type=int, default=3,
        help="Unrecorded serial requests issued before measurement.")
    parser.add_argument(
        "--log-every", type=int, default=50,
        help="Print progress every N measured requests (0 disables progress).")
    parser.add_argument(
        "--collect-module-times", action="store_true",
        help="Enable intrusive per-module CUDA timing for attribution only.")
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT_ROOT),
        help=("Root for canonical trace results. The default trace path is "
              "<output-root>/<model>/<dataset>/trace_ttft_only.csv."))
    parser.add_argument(
        "--model-name",
        help="Stable model directory name; inferred from --model when omitted.")
    parser.add_argument(
        "--dataset-name",
        help="Stable dataset directory name; inferred from --trace-csv when omitted.")
    parser.add_argument(
        "--output-dir",
        help="Explicit run directory; overrides the canonical default path.")
    return parser.parse_args()


def path_slug(value):
    """Return a stable, filesystem-friendly model or dataset name."""
    slug = "".join(
        character.lower() if character.isalnum() or character in ".-_"
        else "-"
        for character in str(value)
    ).strip("-._")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "unknown"


def inferred_model_name(model):
    model_path = Path(str(model).rstrip("/"))
    name = model_path.name
    if len(name) >= 12 and all(character in "0123456789abcdef"
                               for character in name.lower()):
        snapshots = model_path.parent
        if snapshots.name == "snapshots":
            name = snapshots.parent.name.removeprefix("models--")
            name = name.replace("--", "-")
    return path_slug(name)


def default_output_dir(args, mode, timestamp):
    if args.output_dir:
        return Path(args.output_dir)
    model_name = path_slug(args.model_name or inferred_model_name(args.model))
    if mode == "trace":
        dataset_name = path_slug(
            args.dataset_name or Path(args.trace_csv).stem)
        return Path(args.output_root) / model_name / dataset_name
    return (Path(args.output_root) / model_name / "fixed_lengths" /
            f"tp{args.tensor_parallel_size}_{timestamp}")


def make_engine(args):
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,
        load_format=args.load_format,
        dtype=args.dtype,
        enforce_eager=True,
        enable_chunked_prefill=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        disable_log_requests=True,
        disable_log_stats=True,
    )
    return AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.LLM_CLASS)


async def wait_for_iteration(engine, start_index):
    deadline = time.perf_counter() + 10
    while time.perf_counter() < deadline:
        logs = engine.engine.iteration_logs[start_index:]
        if logs and "step_wall_time_ms" in logs[-1]:
            return logs
        await asyncio.sleep(0.01)
    return engine.engine.iteration_logs[start_index:]


async def run_one(engine, request):
    prompt_len = int(request["prompt_len"])
    request_id = str(request["request_id"])
    log_start = len(engine.engine.iteration_logs)
    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=1,
        detokenize=False,
    )
    started = time.perf_counter()
    first_token_at = None
    final_output = None
    async for output in engine.generate(
            prompt={"prompt_token_ids": [1] * prompt_len},
            sampling_params=sampling,
            request_id=request_id):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        final_output = output
    finished = time.perf_counter()
    iteration_logs = await wait_for_iteration(engine, log_start)
    prefill_log = next(
        (item for item in iteration_logs if item.get("prefill_tokens", 0) > 0),
        iteration_logs[0] if iteration_logs else {},
    )
    modules = prefill_log.get("module_profile_times") or {}
    row = {
        "request_id": request_id,
        "sequence_index": request.get("sequence_index"),
        "source_request_id": request.get("source_request_id"),
        "prompt_len": prompt_len,
        "source_output_len": request.get("source_output_len"),
        "source_arrival_offset_s": request.get("source_arrival_offset_s"),
        "repeat": request.get("repeat"),
        "generated_tokens": (
            len(final_output.outputs[0].token_ids) if final_output else 0),
        "ttft_ms": (
            (first_token_at - started) * 1000
            if first_token_at is not None else None),
        "total_ms": (finished - started) * 1000,
        "iter_id": prefill_log.get("iter_id"),
        "prefill_tokens": prefill_log.get("prefill_tokens"),
        "decode_tokens": prefill_log.get("decode_tokens"),
        **{
            name: prefill_log.get(name)
            for name in (
                "actual_time_ms",
                "step_wall_time_ms",
                "runtime_overhead_ms",
                "scheduler_wall_ms",
            )
        },
        **modules,
    }
    tagged_logs = [
        {
            "request_id": request_id,
            "sequence_index": request.get("sequence_index"),
            "source_request_id": request.get("source_request_id"),
            "prompt_len": prompt_len,
            "repeat": request.get("repeat"),
            **item,
        } for item in iteration_logs
    ]
    return row, tagged_logs


def percentile(values, fraction):
    """Return a nearest-rank percentile without a NumPy dependency."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1,
                       int((len(ordered) - 1) * fraction)))
    return ordered[index]


def metric_stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def throughput_summary(rows):
    total_tokens = sum(int(row.get("prefill_tokens") or 0) for row in rows)

    def rate(metric, numerator):
        seconds = sum(float(row.get(metric) or 0.0) for row in rows) / 1000
        return numerator / seconds if seconds > 0 else None

    return {
        "requests": len(rows),
        "prefill_tokens": total_tokens,
        "forward_request_per_s": rate("actual_time_ms", len(rows)),
        "forward_token_per_s": rate("actual_time_ms", total_tokens),
        "step_wall_request_per_s": rate("step_wall_time_ms", len(rows)),
        "step_wall_token_per_s": rate("step_wall_time_ms", total_tokens),
        "end_to_end_request_per_s": rate("total_ms", len(rows)),
        "end_to_end_token_per_s": rate("total_ms", total_tokens),
    }


def write_aggregate_summary(rows, output_path, mode):
    payload = {
        "mode": mode,
        "serial_submission": True,
        "max_tokens": 1,
        "throughput": throughput_summary(rows),
        "prompt_len": metric_stats([row.get("prompt_len") for row in rows]),
        "metrics": {
            metric: metric_stats([row.get(metric) for row in rows])
            for metric in METRICS
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def write_summary(rows, output_path):
    fields = ["prompt_len", "samples"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_p50"))
    fields.extend((
        "forward_request_per_s",
        "forward_token_per_s",
        "step_wall_request_per_s",
        "step_wall_token_per_s",
        "end_to_end_request_per_s",
        "end_to_end_token_per_s",
    ))
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for prompt_len in sorted({row["prompt_len"] for row in rows}):
            group = [row for row in rows if row["prompt_len"] == prompt_len]
            summary = {"prompt_len": prompt_len, "samples": len(group)}
            for metric in METRICS:
                values = [
                    float(row[metric]) for row in group
                    if row.get(metric) is not None
                ]
                summary[f"{metric}_mean"] = (
                    statistics.fmean(values) if values else "")
                summary[f"{metric}_p50"] = (
                    statistics.median(values) if values else "")
            summary.update(throughput_summary(group))
            summary.pop("requests")
            summary.pop("prefill_tokens")
            writer.writerow(summary)


def load_trace(path, max_requests):
    rows = []
    with Path(path).open(newline="") as file:
        for index, source in enumerate(csv.DictReader(file)):
            if max_requests > 0 and len(rows) >= max_requests:
                break
            try:
                prompt_len = int(source["prompt_len"])
            except KeyError as exc:
                raise ValueError(
                    f"{path} must contain a prompt_len column") from exc
            source_id = source.get("request_id", str(index))
            rows.append({
                "request_id": f"trace-{index:06d}-{source_id}",
                "sequence_index": index,
                "source_request_id": source_id,
                "prompt_len": prompt_len,
                "source_output_len": (
                    int(source["output_len"])
                    if source.get("output_len") not in (None, "") else None),
                "source_arrival_offset_s": (
                    float(source["arrival_offset_s"])
                    if source.get("arrival_offset_s") not in (None, "")
                    else None),
                "repeat": None,
            })
    return rows


def fixed_length_workload(lengths, repeats):
    rows = []
    for prompt_len in lengths:
        for repeat in range(1, repeats + 1):
            rows.append({
                "request_id": f"prefill-{prompt_len}-{repeat}",
                "sequence_index": len(rows),
                "source_request_id": None,
                "prompt_len": prompt_len,
                "source_output_len": None,
                "source_arrival_offset_s": None,
                "repeat": repeat,
            })
    return rows


async def main():
    args = parse_args()
    if args.collect_module_times:
        os.environ["VLLM_COLLECT_REAL_MODULE_TIMES"] = "1"
    if args.trace_csv:
        workload = load_trace(args.trace_csv, args.max_requests)
        mode = "trace"
        lengths = []
    else:
        lengths = list(range(*args.lengths))
        if not lengths:
            raise ValueError("--lengths produced an empty range")
        workload = fixed_length_workload(lengths, args.repeats)
        mode = "fixed_lengths"
    if not workload:
        raise ValueError("workload is empty")
    too_long = [row for row in workload
                if int(row["prompt_len"]) > args.max_model_len]
    if too_long:
        raise ValueError(
            f"{len(too_long)} prompts exceed --max-model-len="
            f"{args.max_model_len}; first length={too_long[0]['prompt_len']}")
    if max(int(row["prompt_len"]) for row in workload) > \
            args.max_num_batched_tokens:
        raise ValueError(
            "A full prefill would be split: set --max-num-batched-tokens "
            f"to at least {max(int(row['prompt_len']) for row in workload)}")
    standard_warmup_len = min(1024, args.max_model_len,
                              args.max_num_batched_tokens)
    warmup_lengths = [standard_warmup_len] * args.warmup_requests
    minimum_workload_len = min(int(row["prompt_len"]) for row in workload)
    # A one-token prompt can select a different attention/kernel path from a
    # normal prefill. Warm that path explicitly so one-time initialization is
    # not charged to the only such request in a trace.
    if warmup_lengths and minimum_workload_len < standard_warmup_len:
        warmup_lengths[0] = minimum_workload_len
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = default_output_dir(args, mode, timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({
            **vars(args),
            "mode": mode,
            "length_values": lengths,
            "measured_requests": len(workload),
            "serial_submission": True,
            "max_tokens": 1,
            "warmup_prompt_lengths": warmup_lengths,
        }, indent=2) + "\n")

    print(f"Output: {output_dir}")
    print(
        f"Mode: {mode}; measured_requests={len(workload)}; "
        f"max_num_seqs={args.max_num_seqs}; max_tokens=1")
    engine = make_engine(args)
    rows = []
    try:
        for index, warmup_len in enumerate(warmup_lengths):
            await run_one(engine, {
                "request_id": f"warmup-{index}",
                "sequence_index": -1,
                "source_request_id": None,
                "prompt_len": warmup_len,
                "source_output_len": None,
                "source_arrival_offset_s": None,
                "repeat": index + 1,
            })
        engine.engine.iteration_logs.clear()
        print(f"Warmup complete: prompt lengths={warmup_lengths}")
        with (output_dir / "requests.jsonl").open("w") as request_file, \
                (output_dir / "iterations.jsonl").open("w") as iteration_file, \
                (output_dir / "trace_ttft_only.csv").open(
                    "w", newline="") as trace_csv_file:
            trace_writer = csv.DictWriter(
                trace_csv_file,
                fieldnames=TRACE_CSV_FIELDS,
                extrasaction="ignore",
            )
            trace_writer.writeheader()
            for measured_index, request in enumerate(workload, 1):
                row, logs = await run_one(engine, request)
                rows.append(row)
                request_file.write(json.dumps(row) + "\n")
                request_file.flush()
                trace_writer.writerow(row)
                trace_csv_file.flush()
                for log in logs:
                    iteration_file.write(json.dumps(log) + "\n")
                iteration_file.flush()
                if (args.log_every > 0 and
                        (measured_index == 1
                         or measured_index % args.log_every == 0
                         or measured_index == len(workload))):
                    print(
                        f"request={measured_index:4d}/{len(workload)} "
                        f"len={row['prompt_len']:4d} "
                        f"ttft={row['ttft_ms']:.2f} ms "
                        f"step={row.get('actual_time_ms', float('nan')):.2f} ms "
                        f"wall={row.get('step_wall_time_ms', float('nan')):.2f} ms")
    finally:
        engine.shutdown_background_loop()

    if mode == "fixed_lengths":
        write_summary(rows, output_dir / "summary.csv")
    write_aggregate_summary(rows, output_dir / "summary.json", mode)
    print(f"Saved {len(rows)} requests to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
