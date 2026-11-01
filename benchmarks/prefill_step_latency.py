#!/usr/bin/env python3
"""Prefill step-time sweep benchmark (vLLM offline engine, V0).

For a given tensor-parallel size this builds ONE offline engine and measures the
per-step forward time of a single, full prefill request at each length in
``--prefill-lens``.

Why one engine can cover every length: with ``max_num_seqs=1`` and
``max_num_batched_tokens`` >= the largest length, a single request of length N
is always prefilled in exactly one engine step (no chunking, no co-batching).
The forward compute for N tokens is identical regardless of the chunk budget,
so we do not need a separate engine per length.

For each request we read the engine's ``iteration_logs`` and pick the prefill
step (``prefill_tokens == N`` and no decode). We record:
  - ``actual_time_ms``    : CUDA-event GPU forward time of that step. This is
                            exactly the quantity the time_predictor predicts,
                            so it is the ground truth for the accuracy check.
  - ``step_wall_time_ms`` : full step wall-clock (scheduler + forward + output).

Auto-profiling of the time_predictor is disabled via
``SATURO_SKIP_TIME_PREDICTOR_PROFILE=1`` (set before importing vllm), so engine
startup never runs the expensive GPU profiling sweep.

Example:
    python benchmarks/prefill_step_latency.py -tp 1 \
        --output-json saturo/results/prefill_step_latency/tp1.json
"""
import argparse
import json
import os
import statistics
import time

# Must be set before importing vllm so the engine skips live profiling.
os.environ.setdefault("SATURO_SKIP_TIME_PREDICTOR_PROFILE", "1")

import torch  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

DEFAULT_LENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _summarize(values):
    if not values:
        return {"count": 0, "mean": None, "median": None,
                "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def measure_length(llm, prefill_len, num_requests, warmup, detokenize=True):
    """Run warmup + timed single-request prefills of length ``prefill_len``.

    Returns (gpu_ms_list, wall_ms_list) collected from the prefill step of each
    request via the engine's iteration_logs.
    """
    engine = llm.llm_engine
    base_ids = [(i % 1000) + 1 for i in range(prefill_len)]
    sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True,
                              detokenize=detokenize)

    def run_once():
        prompt = TokensPrompt(prompt_token_ids=list(base_ids))
        engine.iteration_logs.clear()
        torch.cuda.synchronize()
        llm.generate([prompt], sampling, use_tqdm=False)
        torch.cuda.synchronize()
        gpu_ms = wall_ms = None
        for log in engine.iteration_logs:
            if (log.get("n_prefill_reqs", 0) >= 1
                    and log.get("prefill_tokens", 0) == prefill_len
                    and log.get("n_decode_reqs", 0) == 0):
                gpu_ms = log.get("actual_time_ms")
                wall_ms = log.get("step_wall_time_ms")
                break
        return gpu_ms, wall_ms

    for _ in range(warmup):
        run_once()

    gpu_list, wall_list = [], []
    for _ in range(num_requests):
        g, w = run_once()
        if g is not None:
            gpu_list.append(float(g))
        if w is not None:
            wall_list.append(float(w))
    return gpu_list, wall_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default="/root/models/Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prefill-lens", type=str,
                        default=",".join(str(x) for x in DEFAULT_LENS),
                        help="Comma-separated prefill lengths to sweep.")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", default="float16",
                        help="Match the shipped GB200 profiles (float16).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--load-format", default="dummy",
                        help="'dummy' skips real weights (timing identical).")
    parser.add_argument("--skip-tokenizer-init", action="store_true",
                        help="Skip tokenizer loading. Prompts are supplied as "
                             "raw token ids, so a model directory holding only "
                             "config.json (no tokenizer files) still works.")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    lens = [int(x) for x in args.prefill_lens.split(",") if x.strip()]
    max_len = max(lens)
    # +64 headroom so an N-token prompt plus the single decode token fit.
    max_model_len = max_len + 64
    chunk = max_len  # one full prefill of any swept length fits in one step

    print(f"[bench] building engine: tp={args.tensor_parallel_size} "
          f"dtype={args.dtype} chunk(max_num_batched_tokens)={chunk} "
          f"max_model_len={max_model_len} load_format={args.load_format}",
          flush=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=chunk,
        max_num_seqs=1,
        max_model_len=max_model_len,
        trust_remote_code=True,
        load_format=args.load_format,
        enable_chunked_prefill=True,
        skip_tokenizer_init=args.skip_tokenizer_init,
    )

    results = []
    for n in lens:
        t0 = time.perf_counter()
        gpu_list, wall_list = measure_length(
            llm, n, args.num_requests, args.warmup,
            detokenize=not args.skip_tokenizer_init)
        dt = time.perf_counter() - t0
        gpu = _summarize(gpu_list)
        wall = _summarize(wall_list)
        results.append({"prefill_len": n, "gpu_ms": gpu, "wall_ms": wall})
        gm = gpu["median"]
        wm = wall["median"]
        print(f"[bench] len={n:5d}  gpu_median={gm:8.3f} ms  "
              f"gpu_mean={gpu['mean']:8.3f} ms  wall_median={wm:8.3f} ms  "
              f"({gpu['count']} samples, {dt:.1f}s)", flush=True)

    payload = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "load_format": args.load_format,
        "skip_tokenizer_init": args.skip_tokenizer_init,
        "num_requests": args.num_requests,
        "warmup": args.warmup,
        "max_num_batched_tokens": chunk,
        "max_model_len": max_model_len,
        "metric": "prefill-step actual_time_ms (CUDA event GPU forward time)",
        "results": results,
    }

    print("\n" + "=" * 72)
    print(f"model            : {args.model}")
    print(f"tensor_parallel  : {args.tensor_parallel_size}")
    print(f"num_requests     : {args.num_requests} (warmup {args.warmup})")
    print("-" * 72)
    print(f"{'prefill_len':>12} | {'gpu_median_ms':>14} | "
          f"{'gpu_mean_ms':>12} | {'wall_median_ms':>15}")
    print("-" * 72)
    for r in results:
        print(f"{r['prefill_len']:>12} | {r['gpu_ms']['median']:>14.3f} | "
              f"{r['gpu_ms']['mean']:>12.3f} | {r['wall_ms']['median']:>15.3f}")
    print("=" * 72)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[bench] wrote {args.output_json}")


if __name__ == "__main__":
    main()
