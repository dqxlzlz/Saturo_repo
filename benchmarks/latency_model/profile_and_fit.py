#!/usr/bin/env python3
"""
profile_and_fit.py — Component profiling + model fitting for latency prediction.

Pipeline:
  1. (Optional) Initialize vLLM engine with dummy weights
  2. (Optional) Run component-level profiling → profile_bench_results.json
  3. Fit regression models to profiling data
  4. Save fitted model coefficients to JSON

The model predicts per-layer latency as:
  per_layer_ms = linear(total_tokens) + attn_prefill(n, sum_ctx)   [for prefill]
  per_layer_ms = linear(total_tokens) + attn_decode(n, sum_ctx)    [for decode]

Where:
  - linear: piecewise linear (width=32 tokens) for stateless work
    (RMSNorm, QKV, RoPE, O-proj, MLP)
  - attn: bivariate linear regression on (n, sum_ctx) for the Attention kernel
    - prefill: n=B*L, sum_ctx=B*L*(L+1)/2  (causal attention)
    - decode:  n=B,   sum_ctx=B*L           (single-token attention)

Usage:
  # Full pipeline (profile + fit):
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 python benchmarks/latency_model/profile_and_fit.py \\
      --model /root/models/Qwen/Qwen2.5-0.5B-Instruct --enforce-eager --load-format dummy

  # Fit only (use existing profiling data):
  python benchmarks/latency_model/profile_and_fit.py \\
      --skip-profiling --profile-input profile_bench_results.json --model-output fitted_model.json
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error


# ============================================================================
# Piecewise Linear Model (for stateless modules)
# ============================================================================
@dataclass
class PiecewiseLinearModel:
    """Piecewise linear model: each segment of `width` tokens has its own (a, b).
    latency = a * num_token + b within each segment.
    Segments without enough data fall back to a global linear fit.
    """
    width: int = 32
    coef: Dict[int, Tuple[float, float]] = None
    fallback: Tuple[float, float] = (0.0, 0.0)

    def segment_id(self, n: float) -> int:
        return int(math.floor(n / self.width))

    def fit(self, tokens: np.ndarray, ms: np.ndarray, min_points: int = 2):
        tokens = tokens.astype(float)
        ms = ms.astype(float)
        self.coef = {}

        A = np.vstack([tokens, np.ones_like(tokens)]).T
        result = np.linalg.lstsq(A, ms, rcond=None)[0]
        self.fallback = (float(result[0]), float(result[1]))

        seg_ids = np.array([self.segment_id(t) for t in tokens], dtype=int)
        for sid in np.unique(seg_ids):
            idx = np.where(seg_ids == sid)[0]
            if len(idx) < min_points:
                continue
            A_seg = np.vstack([tokens[idx], np.ones(len(idx))]).T
            res = np.linalg.lstsq(A_seg, ms[idx], rcond=None)[0]
            self.coef[int(sid)] = (float(res[0]), float(res[1]))

    def predict_one(self, n: float) -> float:
        sid = self.segment_id(float(n))
        a, b = self.coef.get(sid, self.fallback)
        return max(0.0, a * float(n) + b)

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(t) for t in tokens], dtype=float)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "coef": {str(k): list(v) for k, v in (self.coef or {}).items()},
            "fallback": list(self.fallback),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PiecewiseLinearModel":
        m = cls(width=d["width"])
        m.coef = {int(k): tuple(v) for k, v in d["coef"].items()}
        m.fallback = tuple(d["fallback"])
        return m


# ============================================================================
# Model Fitting
# ============================================================================
def eval_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {"R2": float("nan"), "MAE": float("nan"), "MAPE": float("nan"), "n_valid": int(mask.sum())}
    return {
        "R2": float(r2_score(yt, yp)),
        "MAE": float(mean_absolute_error(yt, yp)),
        "MAPE": float(mean_absolute_percentage_error(yt, yp)),
        "n_valid": int(mask.sum()),
    }


def fit_prefill_attention(df, latency_col="ms_median"):
    L = df["req_len"].to_numpy(dtype=float)
    B = df["num_req"].to_numpy(dtype=float)
    y = df[latency_col].to_numpy(dtype=float)
    mask = np.isfinite(y)
    L, B, y = L[mask], B[mask], y[mask]
    n = L * B
    sum_ctx = B * (L * (L + 1.0) / 2.0)
    X = np.stack([n, sum_ctx], axis=1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = eval_metrics(y, y_pred)
    info = {
        "type": "prefill_attention",
        "source": latency_col,
        "formula": "tau = alpha*(B*L) + beta*(B*L*(L+1)/2) + gamma",
        "alpha": float(model.coef_[0]),
        "beta": float(model.coef_[1]),
        "gamma": float(model.intercept_),
        "n_samples": int(len(y)),
        "metrics": metrics,
    }
    return model, info


def fit_decode_attention(df, latency_col="ms_median"):
    L = df["req_len"].to_numpy(dtype=float)
    B = df["num_req"].to_numpy(dtype=float)
    y = df[latency_col].to_numpy(dtype=float)
    mask = np.isfinite(y)
    L, B, y = L[mask], B[mask], y[mask]
    n = B
    sum_ctx = B * L
    X = np.stack([n, sum_ctx], axis=1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = eval_metrics(y, y_pred)
    info = {
        "type": "decode_attention",
        "source": latency_col,
        "formula": "tau = alpha*B + beta*(B*L) + gamma",
        "alpha": float(model.coef_[0]),
        "beta": float(model.coef_[1]),
        "gamma": float(model.intercept_),
        "n_samples": int(len(y)),
        "metrics": metrics,
    }
    return model, info


def fit_decode_full_layer(df):
    # Fit decode full-layer latency directly (no attn/linear split)
    L = df["req_len"].to_numpy(dtype=float)
    B = df["num_req"].to_numpy(dtype=float)
    y = df["ms_median"].to_numpy(dtype=float)
    mask = np.isfinite(y)
    L, B, y = L[mask], B[mask], y[mask]
    n = B
    sum_ctx = B * L
    X = np.stack([n, sum_ctx], axis=1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = eval_metrics(y, y_pred)
    info = {
        "type": "decode_full_layer",
        "formula": "tau = alpha*B + beta*(B*L) + gamma",
        "alpha": float(model.coef_[0]),
        "beta": float(model.coef_[1]),
        "gamma": float(model.intercept_),
        "n_samples": int(len(y)),
        "metrics": metrics,
    }
    return model, info


def fit_linear_stateless(df, width=32):
    tokens = df["num_token"].to_numpy(dtype=float)
    ms = df["ms_median"].to_numpy(dtype=float)
    mask = np.isfinite(tokens) & np.isfinite(ms)
    tokens, ms = tokens[mask], ms[mask]
    model = PiecewiseLinearModel(width=width)
    model.fit(tokens, ms)
    y_pred = model.predict(tokens)
    metrics = eval_metrics(ms, y_pred)
    info = {
        "type": "linear_stateless",
        "width": width,
        "n_segments": len(model.coef),
        "fallback_a": model.fallback[0],
        "fallback_b": model.fallback[1],
        "n_samples": int(len(ms)),
        "metrics": metrics,
    }
    return model, info


def _attention_pred_ms(row, prefill_model, decode_model):
    L = float(row["req_len"])
    B = float(row["num_req"])
    if row.get("mode") == "prefill":
        return float(prefill_model.predict(
            np.array([[L * B, B * (L * (L + 1.0) / 2.0)]]))[0])
    return float(decode_model.predict(np.array([[B, B * L]]))[0])


def _linear_calibration_points(results, prefill_model, decode_model):
    tokens = []
    linear_ms = []

    for row in results.get("full_layer_ms", []):
        mode = row.get("mode")
        if mode not in ("prefill", "decode"):
            continue
        token_count = (row["req_len"] * row["num_req"]
                       if mode == "prefill" else row["num_req"])
        if "linear_ms_median" in row:
            ms = row["linear_ms_median"]
        elif "linear_ms" in row:
            ms = row["linear_ms"]
        elif "layer_ms_median" in row and "attn_ms_median" in row:
            ms = row["layer_ms_median"] - row["attn_ms_median"]
        else:
            ms = row.get("ms_median", float("nan")) - _attention_pred_ms(
                row, prefill_model, decode_model)
        if math.isfinite(token_count) and math.isfinite(ms) and ms > 0:
            tokens.append(float(token_count))
            linear_ms.append(float(ms))

    if not tokens:
        for row in results.get("decode_full_layer_ms", []):
            token_count = float(row["num_req"])
            layer_ms = float(row.get("ms_median", float("nan")))
            ms = layer_ms - _attention_pred_ms(
                {**row, "mode": "decode"}, prefill_model, decode_model)
            if math.isfinite(token_count) and math.isfinite(ms) and ms > 0:
                tokens.append(token_count)
                linear_ms.append(ms)

    return np.array(tokens, dtype=float), np.array(linear_ms, dtype=float)


def apply_linear_calibration(linear_model, results, prefill_model,
                             decode_model, width=32):
    tokens, ms = _linear_calibration_points(
        results, prefill_model, decode_model)
    if len(tokens) == 0:
        return None

    seg_ids = np.array([linear_model.segment_id(t) for t in tokens], dtype=int)
    for sid in np.unique(seg_ids):
        idx = np.where(seg_ids == sid)[0]
        t_seg = tokens[idx]
        y_seg = ms[idx]
        if len(idx) >= 2 and len(np.unique(t_seg)) >= 2:
            A_seg = np.vstack([t_seg, np.ones_like(t_seg)]).T
            a, b = np.linalg.lstsq(A_seg, y_seg, rcond=None)[0]
        else:
            a = linear_model.fallback[0]
            b = float(np.median(y_seg) - a * np.median(t_seg))
        linear_model.coef[int(sid)] = (float(a), float(b))

    y_pred = linear_model.predict(tokens)
    return {
        "type": "linear_full_layer_calibration",
        "n_samples": int(len(tokens)),
        "n_segments": int(len(np.unique(seg_ids))),
        "metrics": eval_metrics(ms, y_pred),
    }


# ============================================================================
# Predictor (save/load)
# ============================================================================
class LatencyPredictor:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.prefill_attn_model = None
        self.decode_attn_model = None
        self.decode_full_model = None
        self.use_decode_full_layer = False
        self.linear_model = None

    def predict_prefill_ms(self, req_len, num_req=1):
        L, B = float(req_len), float(num_req)
        n = L * B
        sum_ctx = B * (L * (L + 1.0) / 2.0)
        return float(self.prefill_attn_model.predict(np.array([[n, sum_ctx]]))[0])

    def predict_decode_ms(self, req_len, num_req=1):
        L, B = float(req_len), float(num_req)
        n = B
        sum_ctx = B * L
        return float(self.decode_attn_model.predict(np.array([[n, sum_ctx]]))[0])

    def predict_decode_layer_ms(self, req_len, num_req=1):
        if self.use_decode_full_layer and self.decode_full_model is not None:
            L, B = float(req_len), float(num_req)
            n = B
            sum_ctx = B * L
            return max(0.0, float(self.decode_full_model.predict(np.array([[n, sum_ctx]]))[0]))
        return self.predict_decode_ms(req_len, num_req) + self.predict_linear_ms(num_req)

    def predict_linear_ms(self, num_token):
        return self.linear_model.predict_one(float(num_token))

    def predict_request_prefill_ms(self, context_len):
        per_layer = self.predict_linear_ms(context_len) + self.predict_prefill_ms(context_len, 1)
        return per_layer * self.num_layers

    def predict_request_decode_per_step_ms(self, context_len):
        per_layer = self.predict_decode_layer_ms(context_len, 1)
        return per_layer * self.num_layers

    def predict_request_total_ms(self, context_len, generated_tokens):
        return self.predict_request_prefill_ms(context_len) + \
               self.predict_request_decode_per_step_ms(context_len) * generated_tokens

    def save(self, path):
        data = {
            "num_layers": self.num_layers,
            "module_boundary": "linear=RMSNorm+QKV+RoPE+O-proj+MLP; attn=Attention(q,k,v) kernel only",
            "use_decode_full_layer": self.use_decode_full_layer,
            "prefill_attn": {
                "coef": self.prefill_attn_model.coef_.tolist(),
                "intercept": float(self.prefill_attn_model.intercept_),
            },
            "decode_attn": {
                "coef": self.decode_attn_model.coef_.tolist(),
                "intercept": float(self.decode_attn_model.intercept_),
            },
            "linear": self.linear_model.to_dict(),
        }
        if self.decode_full_model is not None:
            data["decode_full_layer"] = {
                "coef": self.decode_full_model.coef_.tolist(),
                "intercept": float(self.decode_full_model.intercept_),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Fitted model saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        pred = cls(num_layers=data["num_layers"])
        pred.prefill_attn_model = LinearRegression()
        pred.prefill_attn_model.coef_ = np.array(data["prefill_attn"]["coef"])
        pred.prefill_attn_model.intercept_ = data["prefill_attn"]["intercept"]
        pred.decode_attn_model = LinearRegression()
        pred.decode_attn_model.coef_ = np.array(data["decode_attn"]["coef"])
        pred.decode_attn_model.intercept_ = data["decode_attn"]["intercept"]
        pred.linear_model = PiecewiseLinearModel.from_dict(data["linear"])
        pred.use_decode_full_layer = bool(data.get("use_decode_full_layer", False))
        if "decode_full_layer" in data:
            pred.decode_full_model = LinearRegression()
            pred.decode_full_model.coef_ = np.array(data["decode_full_layer"]["coef"])
            pred.decode_full_model.intercept_ = data["decode_full_layer"]["intercept"]
        return pred


# ============================================================================
# Profiling & Fitting Pipeline
# ============================================================================
def init_engine(args):
    """Initialize vLLM engine. Use LLM (sync) for direct model_executor access."""
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
        skip_tokenizer_init=args.skip_tokenizer_init,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )
    return llm


def run_profiling(llm, args):
    print("=" * 60)
    print("Step 1: Component-level profiling")
    print("=" * 60)
    t0 = time.time()

    # LLM.llm_engine is the LLMEngine, which has model_executor
    executor = llm.llm_engine.model_executor
    results = executor.profile_attn_linear(
        warmup=args.warmup,
        repeat=args.repeat,
        num_layer_to_run=args.num_layer_to_run,
    )

    elapsed = time.time() - t0
    print(f"Profiling done in {elapsed:.1f}s")
    print(f"  Prefill configs: {len(results.get('prefill_attn_ms', []))}")
    print(f"  Decode configs:  {len(results.get('decode_attn_ms', []))}")
    print(f"  Linear configs:  {len(results.get('linear_ms', []))}")
    print(f"  Full-layer calibration configs: {len(results.get('full_layer_ms', []))}")

    with open(args.profile_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.profile_output}")
    return results


def fit_and_report(results, width=32):
    print("\n" + "=" * 60)
    print("Step 2: Fitting regression models")
    print("=" * 60)

    full_layer_df = pd.DataFrame(results.get("full_layer_ms", []))
    if not full_layer_df.empty and "attn_ms_median" in full_layer_df.columns:
        prefill_source_df = full_layer_df[full_layer_df["mode"] == "prefill"]
        decode_source_df = full_layer_df[full_layer_df["mode"] == "decode"]
    else:
        prefill_source_df = pd.DataFrame()
        decode_source_df = pd.DataFrame()

    if len(prefill_source_df) >= 8:
        prefill_df = prefill_source_df
        prefill_model, p_info = fit_prefill_attention(
            prefill_df, latency_col="attn_ms_median")
    else:
        prefill_df = pd.DataFrame(results["prefill_attn_ms"])
        prefill_model, p_info = fit_prefill_attention(prefill_df)
    print(f"\n[Prefill Attention]")
    print(f"  {p_info['formula']}")
    print(f"  source={p_info['source']}")
    print(f"  alpha={p_info['alpha']:.6e}, beta={p_info['beta']:.6e}, gamma={p_info['gamma']:.4f}")
    print(f"  R2={p_info['metrics']['R2']:.4f}, MAE={p_info['metrics']['MAE']:.4f}ms, MAPE={p_info['metrics']['MAPE']*100:.1f}%")

    if len(decode_source_df) >= 8:
        decode_df = decode_source_df
        decode_model, d_info = fit_decode_attention(
            decode_df, latency_col="attn_ms_median")
    else:
        decode_df = pd.DataFrame(results["decode_attn_ms"])
        decode_model, d_info = fit_decode_attention(decode_df)
    print(f"\n[Decode Attention]")
    print(f"  {d_info['formula']}")
    print(f"  source={d_info['source']}")
    print(f"  alpha={d_info['alpha']:.6e}, beta={d_info['beta']:.6e}, gamma={d_info['gamma']:.4f}")
    print(f"  R2={d_info['metrics']['R2']:.4f}, MAE={d_info['metrics']['MAE']:.4f}ms, MAPE={d_info['metrics']['MAPE']*100:.1f}%")

    linear_df = pd.DataFrame(results["linear_ms"])
    linear_model, l_info = fit_linear_stateless(linear_df, width=width)
    cal_info = apply_linear_calibration(
        linear_model, results, prefill_model, decode_model, width=width)
    if cal_info is not None:
        tokens = linear_df["num_token"].to_numpy(dtype=float)
        ms = linear_df["ms_median"].to_numpy(dtype=float)
        mask = np.isfinite(tokens) & np.isfinite(ms)
        l_info["synthetic_metrics_after_calibration"] = eval_metrics(
            ms[mask], linear_model.predict(tokens[mask]))
        l_info["calibration"] = cal_info
        l_info["n_segments"] = len(linear_model.coef)

    print(f"\n[Linear/Stateless] piecewise width={width}")
    print(f"  fallback: a={l_info['fallback_a']:.6f}, b={l_info['fallback_b']:.4f}")
    print(f"  segments={l_info['n_segments']}")
    print(f"  R2={l_info['metrics']['R2']:.4f}, MAE={l_info['metrics']['MAE']:.4f}ms, MAPE={l_info['metrics']['MAPE']*100:.1f}%")
    if cal_info is not None:
        print("  full-layer calibration:")
        print(f"    segments={cal_info['n_segments']}, samples={cal_info['n_samples']}")
        print(f"    MAE={cal_info['metrics']['MAE']:.4f}ms, MAPE={cal_info['metrics']['MAPE']*100:.1f}%")

    # Decode full-layer model (if data available)
    decode_full_model = None
    df_info = None
    if "decode_full_layer_ms" in results and results["decode_full_layer_ms"]:
        decode_full_df = pd.DataFrame(results["decode_full_layer_ms"])
        decode_full_model, df_info = fit_decode_full_layer(decode_full_df)
        print(f"\n[Decode Full Layer]")
        print(f"  {df_info['formula']}")
        print(f"  alpha={df_info['alpha']:.6e}, beta={df_info['beta']:.6e}, gamma={df_info['gamma']:.4f}")
        print(f"  R2={df_info['metrics']['R2']:.4f}, MAE={df_info['metrics']['MAE']:.4f}ms, MAPE={df_info['metrics']['MAPE']*100:.1f}%")

    return prefill_model, decode_model, linear_model, decode_full_model, p_info, d_info, l_info, df_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/models/Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--load-format", type=str, default="dummy")
    parser.add_argument("--skip-tokenizer-init",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--disable-custom-all-reduce",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--num-layer-to-run", type=int, default=5)
    parser.add_argument("--piecewise-width", type=int, default=32)
    parser.add_argument("--profile-output", type=str, default="profile_bench_results_new.json")
    parser.add_argument("--profile-input", type=str, default=None)
    parser.add_argument("--model-output", type=str, default="fitted_model.json")
    parser.add_argument("--skip-profiling", action="store_true")
    parser.add_argument("--num-layers", type=int, default=None)
    args = parser.parse_args()

    if args.skip_profiling:
        input_path = args.profile_input or args.profile_output
        print(f"Loading existing profiling data from {input_path}")
        with open(input_path, "r") as f:
            results = json.load(f)
    else:
        llm = init_engine(args)
        results = run_profiling(llm, args)

    prefill_model, decode_model, linear_model, decode_full_model, p_info, d_info, l_info, df_info = fit_and_report(
        results, width=args.piecewise_width
    )

    num_layers = args.num_layers or results.get("meta", {}).get("num_layers", 28)
    print(f"\nUsing num_layers={num_layers}")

    predictor = LatencyPredictor(num_layers=num_layers)
    predictor.prefill_attn_model = prefill_model
    predictor.decode_attn_model = decode_model
    predictor.decode_full_model = decode_full_model
    predictor.linear_model = linear_model
    predictor.save(args.model_output)

    report = {
        "prefill_attn": p_info,
        "decode_full_layer": df_info,
        "decode_attn": d_info,
        "linear": l_info,
        "num_layers": num_layers,
        "piecewise_width": args.piecewise_width,
    }
    report_path = args.model_output.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")
    print("\nDone! Next: run validate_trace.py to test predictions.")


if __name__ == "__main__":
    main()
