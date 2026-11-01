#!/usr/bin/env python3
"""Collect stateless-linear (attention-excluded) per-step time vs num_token.

For num_token in range(START, STOP, STEP) (default 2..4096 step 2) this runs a
single-request prefill of length num_token through profile_qwen2_module_times
(4 profiled layers, ``--repeat`` measurements, median). It records the
``stateless_linear`` group time -- i.e. the linear part with attention dropped:
layernorms, qkv/o projections, rotary, and the MLP.

Two views are recorded per num_token:
  - per_layer_median_ms : stateless total over the profiled layers / layer_count
                          (this is what the time_predictor's linear model fits).
  - total_4layer_median_ms : stateless total summed over the 4 profiled layers.
"""
import argparse
import csv
import json
import os

# We drive profiling manually; skip the engine-init auto-profile.
os.environ.setdefault("SATURO_SKIP_TIME_PREDICTOR_PROFILE", "1")

from vllm import LLM  # noqa: E402

STATELESS = "stateless_linear"
STATEFUL = "stateful_attention"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen/Qwen2.5-72B-Instruct")
    p.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--load-format", default="dummy")
    p.add_argument("--start", type=int, default=2)
    p.add_argument("--stop", type=int, default=4097)  # exclusive -> up to 4096
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--profile-num-layers", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=4160)
    p.add_argument("--output-csv", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    lens = list(range(args.start, args.stop, args.step))
    print(f"[collect] tp={args.tensor_parallel_size} dtype={args.dtype} "
          f"num_token {lens[0]}..{lens[-1]} step {args.step} "
          f"({len(lens)} points), repeat={args.repeat}, "
          f"profile_num_layers={args.profile_num_layers}", flush=True)

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

    configs = [("prefill", n, 1) for n in lens]
    result = llm.llm_engine.model_executor.profile_qwen2_module_times(
        warmup=args.warmup,
        repeat=args.repeat,
        configs=configs,
        profile_num_layers=args.profile_num_layers,
        profile_layer_start=0,
        profile_kv_cache_fraction=0.9,
    )

    rows = []
    for r in result.get("rows", []):
        gpl = r.get("group_per_layer_ms", {})
        gtot = r.get("group_total_ms", {})
        lin = gpl.get(STATELESS, {})
        lin_tot = gtot.get(STATELESS, {})
        attn = gpl.get(STATEFUL, {})
        rows.append({
            "num_token": r["n_tokens"],
            "linear_per_layer_median_ms": lin.get("median"),
            "linear_per_layer_mean_ms": lin.get("mean"),
            "linear_total_4layer_median_ms": lin_tot.get("median"),
            "attn_per_layer_median_ms": attn.get("median"),
            "total_forward_4layer_median_ms":
                r.get("total_forward_ms", {}).get("median"),
        })
    rows.sort(key=lambda x: x["num_token"])

    print(f"[collect] got {len(rows)} rows. Sample:")
    print(f"{'num_token':>9} | {'lin_per_layer_ms':>16} | "
          f"{'lin_4layer_ms':>13} | {'attn_per_layer_ms':>17}")
    for r in rows[:5] + rows[len(rows) // 2:len(rows) // 2 + 2] + rows[-5:]:
        print(f"{r['num_token']:>9} | {r['linear_per_layer_median_ms']:>16.5f} | "
              f"{r['linear_total_4layer_median_ms']:>13.5f} | "
              f"{r['attn_per_layer_median_ms']:>17.5f}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[collect] wrote {args.output_csv}")
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "tensor_parallel_size": args.tensor_parallel_size,
                "dtype": args.dtype,
                "repeat": args.repeat,
                "profile_num_layers": args.profile_num_layers,
                "meta": result.get("meta", {}),
                "rows": rows,
            }, f, indent=2)
        print(f"[collect] wrote {args.output_json}")


if __name__ == "__main__":
    main()
