#!/usr/bin/env python3
"""
collect_iteration_data.py — Run 200 trace requests offline with chunked prefill,
collect per-iteration stats (prefill_tokens, decode_tokens, actual_time_ms) to CSV.
"""
import argparse
import csv
import json
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/models/Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--load-format", type=str, default="dummy")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enable-chunked-prefill", action="store_true", default=True)
    parser.add_argument("--trace-csv", type=str, default="data/datasets/AzureLLMInferenceTrace_conv.csv")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--output", type=str, default="iteration_data.csv")
    args = parser.parse_args()

    # Load trace
    requests = []
    with open(args.trace_csv, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= args.num_requests:
                break
            prompt_len = int(row["ContextTokens"])
            output_len = int(row["GeneratedTokens"])
            # Clamp to max_model_len
            if prompt_len + output_len > args.max_model_len:
                output_len = max(1, args.max_model_len - prompt_len)
            if prompt_len >= args.max_model_len:
                prompt_len = args.max_model_len - 1
                output_len = 1
            requests.append({
                "req_id": i,
                "prompt_len": prompt_len,
                "output_len": output_len,
            })
    print(f"Loaded {len(requests)} requests from {args.trace_csv}")

    os.environ["VLLM_COLLECT_ITERATION_LOGS"] = "1"

    # Init vLLM
    from vllm import LLM, SamplingParams

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
        enable_chunked_prefill=args.enable_chunked_prefill,
    )

    # Clear any warmup iteration logs
    llm.llm_engine.iteration_logs.clear()

    # Build prompts and sampling params
    prompts = []
    sampling_params_list = []
    for req in requests:
        prompt_token_ids = [0] * req["prompt_len"]
        sp = SamplingParams(
            temperature=1.0,
            ignore_eos=True,
            max_tokens=req["output_len"],
        )
        prompts.append({"prompt_token_ids": prompt_token_ids})
        sampling_params_list.append(sp)

    print(f"\nRunning {len(requests)} requests with chunked prefill...")
    print(f"  max_num_batched_tokens={args.max_num_batched_tokens}")
    print(f"  max_num_seqs={args.max_num_seqs}")
    print(f"  max_model_len={args.max_model_len}")

    t0 = time.time()
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params_list,
    )
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s, {len(outputs)} outputs")

    # Extract iteration logs
    logs = llm.llm_engine.iteration_logs
    print(f"Collected {len(logs)} iteration logs")

    # Save to CSV
    fieldnames = [
        "iter_id", "n_prefill_reqs", "n_decode_reqs",
        "prefill_tokens", "decode_tokens", "total_tokens",
        "prefill_sum_ctx", "decode_sum_ctx",
        "actual_time_ms", "predicted_time_ms",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for log in logs:
            row = {k: log.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"Saved {len(logs)} iterations to {args.output}")

    # Quick stats
    if logs:
        actual_times = [l["actual_time_ms"] for l in logs if l["actual_time_ms"] is not None]
        print(f"\n--- Quick Stats ---")
        print(f"  Total iterations: {len(logs)}")
        print(f"  Avg actual_time: {sum(actual_times)/len(actual_times):.2f}ms")
        print(f"  Min actual_time: {min(actual_times):.2f}ms")
        print(f"  Max actual_time: {max(actual_times):.2f}ms")

        # Count mixed iterations
        mixed = sum(1 for l in logs if l["prefill_tokens"] > 0 and l["decode_tokens"] > 0)
        pure_prefill = sum(1 for l in logs if l["prefill_tokens"] > 0 and l["decode_tokens"] == 0)
        pure_decode = sum(1 for l in logs if l["prefill_tokens"] == 0 and l["decode_tokens"] > 0)
        print(f"  Pure prefill: {pure_prefill}, Pure decode: {pure_decode}, Mixed: {mixed}")


if __name__ == "__main__":
    main()
