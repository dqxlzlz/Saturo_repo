#!/usr/bin/env python3
"""
validate_trace.py — Run inference on trace CSV and compare predicted vs actual latency.

Pipeline:
  1. Load fitted model from profile_and_fit.py
  2. Initialize vLLM engine
  3. Run first N requests from trace (serial, one at a time)
  4. Measure per-request latency (prefill TTFT, decode TBT)
  5. Compare model predictions vs measurements

The trace CSV can be in two formats:
  - Azure format: TIMESTAMP, ContextTokens, GeneratedTokens
  - Prefill format: req_id, RequestType, ContextTokens, GeneratedTokens

Usage:
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 python benchmarks/latency_model/validate_trace.py \\
      --model /root/models/Qwen/Qwen2.5-0.5B-Instruct \\
      --fitted-model fitted_model.json \\
      --trace-csv data/datasets/AzureLLMInferenceTrace_conv.csv \\
      --num-requests 1000 --enforce-eager --load-format dummy
"""
import argparse
import asyncio
import csv
import json
import time
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_and_fit import LatencyPredictor


# ============================================================================
# Trace Loading (supports multiple CSV formats)
# ============================================================================
def load_trace(csv_path: str, num_requests: int = 1000) -> List[Dict]:
    requests = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_requests:
                break
            req = {
                "req_id": i,
                "prompt_len": int(row["ContextTokens"]),
                "output_len": int(row["GeneratedTokens"]),
            }
            if "RequestType" in row:
                req["req_type"] = row["RequestType"]
            requests.append(req)
    print(f"Loaded {len(requests)} requests from {csv_path}")
    prompt_lens = [r["prompt_len"] for r in requests]
    output_lens = [r["output_len"] for r in requests]
    print(f"  prompt_len: min={min(prompt_lens)}, max={max(prompt_lens)}, "
          f"median={np.median(prompt_lens):.0f}")
    print(f"  output_len: min={min(output_lens)}, max={max(output_lens)}, "
          f"median={np.median(output_lens):.0f}")
    return requests


# ============================================================================
# Engine & Inference
# ============================================================================
def init_engine(args):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.usage.usage_lib import UsageContext

    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        load_format=args.load_format,
    )
    return AsyncLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=UsageContext.LLM_CLASS,
    )


async def run_one_request(engine, req: Dict, args) -> Dict:
    """Run a single request serially and measure timing."""
    from vllm import SamplingParams

    prompt_token_ids = [0] * req["prompt_len"]
    sp = SamplingParams(
        temperature=1.0,
        ignore_eos=True,
        max_tokens=max(1, req["output_len"]),
        prefill_ddl=time.time() + args.prefill_slo,  # finite → triggers timing recording
    )

    submit_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    try:
        async for _ in engine.generate(
            prompt={"prompt_token_ids": prompt_token_ids},
            sampling_params=sp,
            request_id=str(req["req_id"]),
            priority=0,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1
    except Exception as e:
        return {**req, "error": str(e),
                "ttft_ms": float("nan"), "avg_tbt_ms": float("nan"), "total_ms": float("nan")}

    end_time = time.perf_counter()
    ttft_ms = (first_token_time - submit_time) * 1000 if first_token_time else float("nan")
    total_ms = (end_time - submit_time) * 1000

    avg_tbt_ms = float("nan")
    if token_count > 1 and first_token_time:
        avg_tbt_ms = (total_ms - ttft_ms) / (token_count - 1)

    return {
        **req,
        "tokens_generated": token_count,
        "ttft_ms": ttft_ms,
        "avg_tbt_ms": avg_tbt_ms,
        "total_ms": total_ms,
    }


async def run_all_requests(engine, requests: List[Dict], args) -> List[Dict]:
    """Run requests serially for clean per-request measurement."""
    results = []
    total = len(requests)
    for i, req in enumerate(requests):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{total}] prompt={req['prompt_len']}, output={req['output_len']}")
        result = await run_one_request(engine, req, args)
        results.append(result)
    return results


# ============================================================================
# Comparison & Reporting
# ============================================================================
def build_comparison(predictor, results):
    rows = []
    for r in results:
        if "error" in r:
            continue
        ctx = r["prompt_len"]
        gen = r.get("tokens_generated", r["output_len"])

        pred_pf = predictor.predict_request_prefill_ms(ctx)
        pred_dec = predictor.predict_request_decode_per_step_ms(ctx)
        pred_total = predictor.predict_request_total_ms(ctx, gen)

        rows.append({
            "req_id": r["req_id"],
            "prompt_len": ctx,
            "gen_tokens": gen,
            "pred_prefill_ms": pred_pf,
            "actual_prefill_ms": r["ttft_ms"],
            "pred_decode_ms": pred_dec,
            "actual_decode_ms": r["avg_tbt_ms"],
            "pred_total_ms": pred_total,
            "actual_total_ms": r["total_ms"],
        })
    return pd.DataFrame(rows)


def print_metrics(df):
    print("\n" + "=" * 60)
    print("VALIDATION: Predicted vs Actual")
    print("=" * 60)

    for label, true_col, pred_col in [
        ("Prefill (TTFT)", "actual_prefill_ms", "pred_prefill_ms"),
        ("Decode (avg TBT)", "actual_decode_ms", "pred_decode_ms"),
        ("Total latency", "actual_total_ms", "pred_total_ms"),
    ]:
        sub = df.dropna(subset=[true_col, pred_col])
        if len(sub) < 2:
            print(f"\n[{label}] insufficient data ({len(sub)} samples)")
            continue
        yt = sub[true_col].values
        yp = sub[pred_col].values
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        mape = mean_absolute_percentage_error(yt, yp)
        print(f"\n[{label}] n={len(sub)}")
        print(f"  R2={r2:.4f}  MAE={mae:.4f}ms  MAPE={mape*100:.1f}%")
        # Pearson correlation
        if np.std(yt) > 0 and np.std(yp) > 0:
            corr = np.corrcoef(yt, yp)[0, 1]
            print(f"  Pearson r={corr:.4f}")

    print(f"\n--- Sample (first 10) ---")
    cols = ["req_id", "prompt_len", "gen_tokens",
            "actual_prefill_ms", "pred_prefill_ms",
            "actual_total_ms", "pred_total_ms"]
    print(df[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fitted-model", type=str, required=True)
    parser.add_argument("--trace-csv", type=str, required=True)
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--load-format", type=str, default="dummy")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--prefill-slo", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="validation_results.csv")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Load existing results CSV instead of running inference")
    args = parser.parse_args()

    # Load model
    print(f"Loading fitted model from {args.fitted_model}...")
    predictor = LatencyPredictor.load(args.fitted_model)
    print(f"  num_layers={predictor.num_layers}")

    # Quick prediction examples
    print("\nExample predictions (single request, all layers):")
    for ctx in [256, 512, 1024, 2048]:
        pf = predictor.predict_request_prefill_ms(ctx)
        dec = predictor.predict_request_decode_per_step_ms(ctx)
        print(f"  ctx={ctx}: prefill={pf:.2f}ms, decode/step={dec:.3f}ms")

    if args.skip_inference:
        print(f"\nLoading existing results from {args.output}")
        df = pd.read_csv(args.output)
    else:
        requests = load_trace(args.trace_csv, args.num_requests)
        print("\nInitializing engine...")
        engine = init_engine(args)
        print(f"\nRunning {len(requests)} requests (serial)...")
        t0 = time.time()
        results = asyncio.run(run_all_requests(engine, requests, args))
        print(f"Inference done in {time.time()-t0:.1f}s")

        df = build_comparison(predictor, results)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

    print_metrics(df)


if __name__ == "__main__":
    main()
