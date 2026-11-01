#!/usr/bin/env python3
"""Validate mixed chunk-prefill + decode iteration latency.

This replays a short trace into synthetic vLLM ExecuteModelRequest batches.
Every measured iteration contains both prefill chunks and decode tokens.
"""

import argparse
import json
import math
import statistics

import numpy as np


def summarize(rows):
    errs = [
        r["error_pct"] for r in rows
        if math.isfinite(r.get("error_pct", float("nan")))
    ]
    if not errs:
        return {}
    return {
        "n": len(errs),
        "median_error_pct": float(statistics.median(errs)),
        "mean_error_pct": float(sum(errs) / len(errs)),
        "p90_error_pct": float(np.percentile(errs, 90)),
        "max_error_pct": float(max(errs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--output", default="mixed_iteration_validation.json")
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--load-format", default="dummy")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--num-iterations", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--prefill-batch-size", type=int, default=1)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--max-trace-requests", type=int, default=64)
    args = parser.parse_args()

    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        load_format=args.load_format,
        enable_chunked_prefill=True,
    )

    data = llm.llm_engine.model_executor.profile_mixed_iterations(
        trace_path=args.trace,
        predictor_path=args.predictor,
        num_iterations=args.num_iterations,
        warmup=args.warmup,
        repeat=args.repeat,
        prefill_chunk_size=args.prefill_chunk_size,
        prefill_batch_size=args.prefill_batch_size,
        decode_batch_size=args.decode_batch_size,
        max_trace_requests=args.max_trace_requests,
    )
    rows = data["iterations"]
    data["summary"] = summarize(rows)

    print("\nMIXED ITERATION VALIDATION")
    print("=" * 80)
    print(f"{'iter':>4} {'prefill':>7} {'decode':>6} {'p_tok':>6} "
          f"{'d_tok':>6} {'actual_ms':>10} {'pred_ms':>10} {'err%':>7}")
    print("-" * 80)
    for row in rows:
        print(f"{row['iteration']:>4} {row['prefill_count']:>7} "
              f"{row['decode_count']:>6} {row['prefill_tokens']:>6} "
              f"{row['decode_tokens']:>6} {row['actual_ms']:>10.2f} "
              f"{row['pred_ms']:>10.2f} {row['error_pct']:>6.1f}%")

    print("\nSUMMARY")
    print(json.dumps(data["summary"], indent=2))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
