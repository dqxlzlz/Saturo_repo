#!/usr/bin/env python3
"""Trigger time_predictor profiling for one TP and validate its accuracy.

Building the offline engine with no cached profile makes LLMEngine.__init__ run
the live time_predictor profiling sweep and write the fitted profile to disk.
Afterwards we query the freshly-built predictor for the forward time of a single
full prefill request at each benchmarked length and compare it against the
measured ground truth (prefill-step ``actual_time_ms`` from the sweep JSON).

A single full prefill of N tokens is expressed to the predictor as one prefill
request with (query_len=N, context_len=N) and no decode requests.
"""
import argparse
import json
import os

# Make sure live profiling is NOT skipped for this driver.
os.environ["SATURO_SKIP_TIME_PREDICTOR_PROFILE"] = "0"

from vllm import LLM  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default="/root/models/Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--load-format", default="dummy")
    parser.add_argument("--max-model-len", type=int, default=8448)
    parser.add_argument("--benchmark-json", type=str, required=True,
                        help="prefill_step_latency_tp{TP}.json ground truth.")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    print(f"[profile] building engine tp={args.tensor_parallel_size} "
          f"dtype={args.dtype} (this triggers time_predictor profiling)",
          flush=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        load_format=args.load_format,
    )

    predictor = llm.llm_engine.time_predictor
    if predictor is None or not getattr(predictor, "available", False):
        raise SystemExit("[profile] ERROR: time_predictor unavailable after "
                         "profiling; cannot validate.")
    print(f"[profile] predictor ready: {predictor.profile_path}", flush=True)

    with open(args.benchmark_json, "r", encoding="utf-8") as f:
        bench = json.load(f)

    rows = []
    for r in bench["results"]:
        n = r["prefill_len"]
        measured = r["gpu_ms"]["median"]
        predicted = predictor.predict_forward_ms([(n, n)], [])
        err = predicted - measured
        pct = (err / measured * 100.0) if measured else float("nan")
        rows.append({
            "prefill_len": n,
            "measured_ms": measured,
            "predicted_ms": predicted,
            "abs_err_ms": err,
            "pct_err": pct,
        })

    print("\n" + "=" * 76)
    print(f"time_predictor accuracy  |  tp={args.tensor_parallel_size}  "
          f"dtype={args.dtype}")
    print(f"profile: {predictor.profile_path}")
    print("-" * 76)
    print(f"{'len':>6} | {'measured_ms':>12} | {'predicted_ms':>13} | "
          f"{'abs_err_ms':>11} | {'pct_err':>8}")
    print("-" * 76)
    for row in rows:
        print(f"{row['prefill_len']:>6} | {row['measured_ms']:>12.3f} | "
              f"{row['predicted_ms']:>13.3f} | {row['abs_err_ms']:>11.3f} | "
              f"{row['pct_err']:>7.1f}%")
    print("-" * 76)
    mape = sum(abs(r["pct_err"]) for r in rows) / len(rows)
    print(f"MAPE (mean abs pct err): {mape:.1f}%")
    print("=" * 76)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "tensor_parallel_size": args.tensor_parallel_size,
                "dtype": args.dtype,
                "profile_path": predictor.profile_path,
                "mape_pct": mape,
                "rows": rows,
            }, f, indent=2)
        print(f"[profile] wrote {args.output_json}")


if __name__ == "__main__":
    main()
