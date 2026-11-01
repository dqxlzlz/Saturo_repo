#!/usr/bin/env python3
"""Probe inference-latency stability of a plain offline vLLM engine.

For each chunk size we submit a single prefill request of that many tokens
(max_tokens=1) and time the end-to-end generate() call. We repeat N times and
report mean/std/CV/min/max so we can see whether small-chunk inference is
inherently jittery, independent of the time_predictor profiling harness.
"""
import argparse
import statistics
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--load-format", default="dummy")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=8448)
    ap.add_argument("--chunks", default="64,128,256,512,1024")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    try:
        from vllm.inputs import TokensPrompt
    except Exception:
        TokensPrompt = None

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        load_format=args.load_format,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )

    sp = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0.0)
    chunks = [int(c) for c in args.chunks.split(",") if c.strip()]

    def make_prompt(n):
        ids = [1] * n
        if TokensPrompt is not None:
            return TokensPrompt(prompt_token_ids=ids)
        return {"prompt_token_ids": ids}

    print(f"\n=== stability probe: model={args.model} TP={args.tensor_parallel_size} "
          f"dtype={args.dtype} repeat={args.repeat} warmup={args.warmup} ===")
    print(f"{'chunk':>6} | {'mean_ms':>9} {'std_ms':>8} {'cv%':>7} "
          f"{'min_ms':>9} {'max_ms':>9} {'spread%':>8}")
    print("-" * 66)

    for n in chunks:
        prompt = make_prompt(n)
        for _ in range(args.warmup):
            llm.generate(prompt, sp, use_tqdm=False)
        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            llm.generate(prompt, sp, use_tqdm=False)
            times.append((time.perf_counter() - t0) * 1000.0)
        mean = statistics.mean(times)
        std = statistics.pstdev(times)
        cv = 100.0 * std / mean if mean else 0.0
        spread = 100.0 * (max(times) - min(times)) / mean if mean else 0.0
        print(f"{n:>6} | {mean:>9.3f} {std:>8.3f} {cv:>7.2f} "
              f"{min(times):>9.3f} {max(times):>9.3f} {spread:>8.2f}")
        each = " ".join(f"{t:.2f}" for t in times)
        print(f"       each: {each}")


if __name__ == "__main__":
    main()
