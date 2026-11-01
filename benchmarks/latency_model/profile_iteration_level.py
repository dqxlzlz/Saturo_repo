#!/usr/bin/env python3
"""
Profile full iteration time and fit iteration-level latency model.

Model structure:
  iter_time = prefill_component + decode_component
  
  For decode-only: iter_time = alpha_d * B + beta_d * sum_ctx + gamma_d
  For prefill-only: iter_time = alpha_p * n_tokens + beta_p * sum_causal_ctx + gamma_p
  For mixed: sum of both components
"""
import argparse
import csv
import json
import math
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("-tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", default="profile_iteration_level.json")
    parser.add_argument("--fitted-output", default="fitted_model_iter_level.json")
    parser.add_argument("--skip-profile", action="store_true", help="Skip profiling, just fit from existing data")
    args = parser.parse_args()

    if not args.skip_profile:
        print("=" * 60)
        print("PHASE 1: Profile full iteration times")
        print("=" * 60)

        from vllm import LLM
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            enforce_eager=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=4096,
            max_num_seqs=256,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            load_format="dummy",
            enable_chunked_prefill=True,
        )

        engine = llm.llm_engine
        worker = None
        if hasattr(engine, 'model_executor'):
            executor = engine.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker

        if worker is None:
            print("ERROR: Cannot access driver_worker for profiling")
            sys.exit(1)

        # Dense decode configs covering the real iteration range
        decode_configs = []
        for B in list(range(1, 11)) + list(range(12, 33, 2)) + list(range(36, 65, 4)) + list(range(72, 201, 8)) + [224, 256]:
            for L in [256, 512, 768, 1024, 1536, 2048, 3072]:
                decode_configs.append(("decode", L, B))

        # Prefill configs
        prefill_configs = []
        for n_tokens in [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]:
            for num_req in [1]:
                prefill_configs.append(("prefill", n_tokens, num_req))
        # Chunked prefill with varying context
        for n_tokens in [128, 256, 512, 1024, 2048, 4096]:
            for num_req in [1, 2, 4, 8]:
                if n_tokens * num_req <= 4096:
                    prefill_configs.append(("prefill", n_tokens, num_req))

        all_configs = decode_configs + prefill_configs
        print(f"Total configs to profile: {len(all_configs)} ({len(decode_configs)} decode, {len(prefill_configs)} prefill)")

        results = worker.profile_full_iteration(
            warmup=args.warmup,
            repeat=args.repeat,
            configs=all_configs,
        )

        if results is None:
            print("ERROR: profile_full_iteration returned None (not driver worker?)")
            sys.exit(1)

        print(f"\nProfiled {len(results)} configs successfully")

        # Save raw profile data
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved profile data to {args.output}")
    else:
        print("Skipping profiling, loading existing data...")
        with open(args.output) as f:
            results = json.load(f)

    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 2: Fit iteration-level model")
    print("=" * 60)

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error

    # Separate decode and prefill
    decode_data = [r for r in results if r["mode"] == "decode"]
    prefill_data = [r for r in results if r["mode"] == "prefill"]

    print(f"Decode data points: {len(decode_data)}")
    print(f"Prefill data points: {len(prefill_data)}")

    # --- Fit decode model ---
    # iter_time = alpha * B + beta * (B * L) + gamma
    # where B = num_req (= decode_tokens), B*L = sum_context
    X_decode = np.array([[d["num_req"], d["num_req"] * d["req_len"]] for d in decode_data])
    y_decode = np.array([d["iter_ms"] for d in decode_data])

    decode_model = LinearRegression()
    decode_model.fit(X_decode, y_decode)
    y_decode_pred = decode_model.predict(X_decode)

    print(f"\nDecode model: iter_ms = {decode_model.coef_[0]:.6e} * B + {decode_model.coef_[1]:.6e} * sum_ctx + {decode_model.intercept_:.4f}")
    print(f"  R² = {r2_score(y_decode, y_decode_pred):.4f}")
    print(f"  MAE = {mean_absolute_error(y_decode, y_decode_pred):.2f} ms")
    print(f"  MAPE = {np.mean(np.abs(y_decode_pred - y_decode) / y_decode * 100):.1f}%")

    # --- Fit prefill model ---
    # iter_time = alpha * n_tokens + beta * (n_tokens * (n_tokens+1)/2 * num_req) + gamma
    # For simple prefill (no cached KV): sum_causal_ctx = num_req * n_tokens * (n_tokens+1) / 2
    X_prefill = np.array([
        [d["num_req"] * d["req_len"],  # total prefill tokens
         d["num_req"] * d["req_len"] * (d["req_len"] + 1) / 2]  # sum_causal_ctx
        for d in prefill_data
    ])
    y_prefill = np.array([d["iter_ms"] for d in prefill_data])

    prefill_model = LinearRegression()
    prefill_model.fit(X_prefill, y_prefill)
    y_prefill_pred = prefill_model.predict(X_prefill)

    print(f"\nPrefill model: iter_ms = {prefill_model.coef_[0]:.6e} * n_tokens + {prefill_model.coef_[1]:.6e} * sum_ctx + {prefill_model.intercept_:.4f}")
    print(f"  R² = {r2_score(y_prefill, y_prefill_pred):.4f}")
    print(f"  MAE = {mean_absolute_error(y_prefill, y_prefill_pred):.2f} ms")
    print(f"  MAPE = {np.mean(np.abs(y_prefill_pred - y_prefill) / y_prefill * 100):.1f}%")

    X_iter = []
    y_iter = []
    for d in prefill_data:
        prefill_tokens = d["num_req"] * d["req_len"]
        prefill_sum_ctx = d["num_req"] * d["req_len"] * (d["req_len"] + 1) / 2
        X_iter.append([prefill_tokens, prefill_sum_ctx, 0, 0])
        y_iter.append(d["iter_ms"])
    for d in decode_data:
        X_iter.append([0, 0, d["num_req"], d["num_req"] * d["req_len"]])
        y_iter.append(d["iter_ms"])

    X_iter = np.array(X_iter)
    y_iter = np.array(y_iter)
    iter_model = LinearRegression()
    iter_model.fit(X_iter, y_iter)
    y_iter_pred = iter_model.predict(X_iter)
    iter_mape = np.mean(np.abs(y_iter_pred - y_iter) / y_iter * 100)

    print("\nCombined iteration model:")
    print("  iter_ms = coef[0]*prefill_tokens + coef[1]*prefill_sum_ctx "
          "+ coef[2]*n_decode_reqs + coef[3]*decode_sum_ctx + intercept")
    print(f"  coef = {iter_model.coef_}")
    print(f"  intercept = {iter_model.intercept_:.4f}")
    print(f"  R² = {r2_score(y_iter, y_iter_pred):.4f}")
    print(f"  MAE = {mean_absolute_error(y_iter, y_iter_pred):.2f} ms")
    print(f"  MAPE = {iter_mape:.1f}%")

    # Save fitted model
    fitted = {
        "model_type": "iteration_level",
        "description": "iter_ms = coef[0]*prefill_tokens + coef[1]*prefill_sum_ctx + coef[2]*n_decode_reqs + coef[3]*decode_sum_ctx + intercept",
        "features": [
            "prefill_tokens",
            "prefill_sum_ctx",
            "n_decode_reqs",
            "decode_sum_ctx",
        ],
        "coef": iter_model.coef_.tolist(),
        "intercept": float(iter_model.intercept_),
        "r2": float(r2_score(y_iter, y_iter_pred)),
        "mape": float(iter_mape),
        "n_samples": int(len(y_iter)),
        "decode": {
            "features": ["batch_size", "sum_context"],
            "coef": decode_model.coef_.tolist(),
            "intercept": float(decode_model.intercept_),
            "r2": float(r2_score(y_decode, y_decode_pred)),
        },
        "prefill": {
            "features": ["total_tokens", "sum_causal_ctx"],
            "coef": prefill_model.coef_.tolist(),
            "intercept": float(prefill_model.intercept_),
            "r2": float(r2_score(y_prefill, y_prefill_pred)),
        },
    }
    with open(args.fitted_output, "w") as f:
        json.dump(fitted, f, indent=2)
    print(f"\nSaved fitted model to {args.fitted_output}")

    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 3: Validate on real iteration data")
    print("=" * 60)

    csv_path = "iteration_data_200.csv"
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found, skipping validation")
        return

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    print(f"Loaded {len(rows)} iterations")

    errors = {"all": [], "pure_decode": [], "mixed": [], "pure_prefill": []}
    for r in rows:
        actual = float(r["actual_time_ms"])
        prefill_tokens = int(float(r["prefill_tokens"]))
        decode_tokens = int(float(r["decode_tokens"]))
        n_decode = int(float(r["n_decode_reqs"]))
        prefill_sum_ctx = float(r["prefill_sum_ctx"])
        decode_sum_ctx = float(r["decode_sum_ctx"])

        # Predict
        pred_decode = 0.0
        if n_decode > 0:
            pred_decode = decode_model.coef_[0] * n_decode + decode_model.coef_[1] * decode_sum_ctx + decode_model.intercept_

        pred_prefill = 0.0
        if prefill_tokens > 0:
            pred_prefill = prefill_model.coef_[0] * prefill_tokens + prefill_model.coef_[1] * prefill_sum_ctx + prefill_model.intercept_

        pred = pred_decode + pred_prefill
        err = abs(pred - actual) / actual * 100

        entry = {"actual": actual, "pred": pred, "err": err}
        errors["all"].append(entry)
        if prefill_tokens > 0 and decode_tokens > 0:
            errors["mixed"].append(entry)
        elif prefill_tokens > 0:
            errors["pure_prefill"].append(entry)
        else:
            errors["pure_decode"].append(entry)

    print(f"\n{'Category':<15} {'N':>5} {'MAPE':>8} {'Med%':>8} {'P90%':>8} {'MAE':>8} {'Bias':>8}")
    print("-" * 65)
    for cat in ["all", "pure_decode", "mixed", "pure_prefill"]:
        entries = errors[cat]
        if not entries:
            continue
        errs = np.array([e["err"] for e in entries])
        actuals = np.array([e["actual"] for e in entries])
        preds = np.array([e["pred"] for e in entries])
        mae = np.mean(np.abs(preds - actuals))
        bias = np.mean(preds - actuals)
        print(f"{cat:<15} {len(entries):>5} {errs.mean():>7.1f}% {np.median(errs):>7.1f}% {np.percentile(errs,90):>7.1f}% {mae:>7.1f}ms {bias:>+7.1f}ms")

    # Show samples
    decode_entries = errors["pure_decode"]
    decode_rows = [r for r in rows if int(float(r["prefill_tokens"]))==0 and int(float(r["decode_tokens"]))>0]
    print(f"\nPure decode samples:")
    for i in range(min(10, len(decode_rows))):
        r = decode_rows[i]
        e = decode_entries[i]
        print(f"  B={int(float(r['n_decode_reqs'])):>4}, sumctx={float(r['decode_sum_ctx']):>9.0f}, actual={e['actual']:>7.1f}ms, pred={e['pred']:>7.1f}ms, err={e['err']:.1f}%")


if __name__ == "__main__":
    main()
