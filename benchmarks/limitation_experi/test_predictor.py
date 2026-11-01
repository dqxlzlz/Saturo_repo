import json
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error


# -----------------------------
# Utils: build features
# -----------------------------
def build_prefill_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    For prefill attention:
      n = L * B
      sum_ctx = B * (1+2+...+L) = B * L(L+1)/2
    """
    L = df["req_len"].to_numpy(dtype=float)
    B = df["num_req"].to_numpy(dtype=float)
    n = L * B
    sum_ctx = B * (L * (L + 1.0) / 2.0)
    X = np.stack([n, sum_ctx], axis=1)
    y = df["ms_median"].to_numpy(dtype=float)
    return X, y


def build_decode_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    For decode attention (common per-step measurement):
      n = B
      sum_ctx = B * L
    """
    L = df["req_len"].to_numpy(dtype=float)
    B = df["num_req"].to_numpy(dtype=float)
    n = B
    sum_ctx = B * L
    X = np.stack([n, sum_ctx], axis=1)
    y = df["ms_median"].to_numpy(dtype=float)
    return X, y


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


# -----------------------------
# Piecewise linear for linear_ms
# -----------------------------
@dataclass
class PiecewiseLinearModel:
    width: int = 32
    # segment_id -> (a, b)
    coef: Dict[int, Tuple[float, float]] = None
    # fallback global (a, b) if some segment has too few points
    fallback: Tuple[float, float] = (0.0, 0.0)

    def segment_id(self, n: float) -> int:
        # segment for [k*width, (k+1)*width)
        return int(math.floor(n / self.width))

    def fit(self, tokens: np.ndarray, ms: np.ndarray, min_points_per_segment: int = 2):
        tokens = tokens.astype(float)
        ms = ms.astype(float)
        self.coef = {}

        # global fallback fit
        A = np.vstack([tokens, np.ones_like(tokens)]).T
        a_g, b_g = np.linalg.lstsq(A, ms, rcond=None)[0]
        self.fallback = (float(a_g), float(b_g))

        # per-segment
        seg_ids = np.array([self.segment_id(t) for t in tokens], dtype=int)
        for sid in np.unique(seg_ids):
            idx = np.where(seg_ids == sid)[0]
            if len(idx) < min_points_per_segment:
                continue
            t_seg = tokens[idx]
            y_seg = ms[idx]
            A_seg = np.vstack([t_seg, np.ones_like(t_seg)]).T
            a, b = np.linalg.lstsq(A_seg, y_seg, rcond=None)[0]
            self.coef[int(sid)] = (float(a), float(b))

    def predict_one(self, n: float) -> float:
        sid = self.segment_id(float(n))
        a, b = self.coef.get(sid, self.fallback)
        return a * float(n) + b

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(t) for t in tokens], dtype=float)


# -----------------------------
# Main: load + fit + test
# -----------------------------
def main(json_path: str, test_size: float = 0.2, seed: int = 42):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---- Prefill attn ----
    prefill_df = pd.DataFrame(data["prefill_attn_ms"])
    Xp, yp = build_prefill_features(prefill_df)
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(Xp, yp, test_size=test_size, random_state=seed)

    def debug_bad_y(y, name="y", topk=20):
        y = np.asarray(y)
        print(f"[{name}] shape={y.shape}, dtype={y.dtype}")

        # 找非有限值：nan/inf/-inf
        bad = ~np.isfinite(y)
        n_bad = int(bad.sum())
        print(f"[{name}] non-finite count = {n_bad}")

        if n_bad > 0:
            idx = np.where(bad)[0]
            print(f"[{name}] first {min(topk, len(idx))} bad indices:", idx[:topk])
            print(f"[{name}] first bad values:", y[idx[:topk]])
            # 如果你有对应的样本ID/行号，也在这里打印出来

        # 看看是不是极端大值导致后面溢出（即便是finite也可能非常大）
        finite = y[np.isfinite(y)]
        if finite.size > 0:
            absmax = np.max(np.abs(finite))
            print(f"[{name}] finite min={finite.min()}, max={finite.max()}, absmax={absmax}")
            # 打印最大的几个值
            largest_idx = np.argsort(np.abs(finite))[-topk:]
            print(f"[{name}] largest |y| values:", finite[largest_idx])

    # 在 fit 前调用
    debug_bad_y(yp_tr, "yp_tr")



    prefill_lr = LinearRegression()
    prefill_lr.fit(Xp_tr, yp_tr)
    yp_pred = prefill_lr.predict(Xp_te)
    prefill_metrics = eval_regression(yp_te, yp_pred)

    # ---- Decode attn ----
    decode_df = pd.DataFrame(data["decode_attn_ms"])
    Xd, yd = build_decode_features(decode_df)
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=test_size, random_state=seed)

    decode_lr = LinearRegression()
    decode_lr.fit(Xd_tr, yd_tr)
    yd_pred = decode_lr.predict(Xd_te)
    decode_metrics = eval_regression(yd_te, yd_pred)

    # ---- Linear (stateless modules) piecewise ----
    linear_df = pd.DataFrame(data["linear_ms"])
    tokens = linear_df["num_token"].to_numpy(dtype=float)
    ylin = linear_df["ms_mean"].to_numpy(dtype=float)

    t_tr, t_te, y_tr, y_te = train_test_split(tokens, ylin, test_size=test_size, random_state=seed)
    pw = PiecewiseLinearModel(width=128)
    pw.fit(t_tr, y_tr, min_points_per_segment=2)
    y_pw_pred = pw.predict(t_te)
    linear_metrics = eval_regression(y_te, y_pw_pred)

    # ---- Print results ----
    print("=== Prefill Attention LR: tau = alpha*n + beta*sum_ctx + gamma ===")
    print("alpha (n):        ", float(prefill_lr.coef_[0]))
    print("beta  (sum_ctx):  ", float(prefill_lr.coef_[1]))
    print("gamma (intercept):", float(prefill_lr.intercept_))
    print("metrics:", prefill_metrics)
    print()

    print("=== Decode Attention LR: tau = alpha*n + beta*sum_ctx + gamma ===")
    print("alpha (n):        ", float(decode_lr.coef_[0]))
    print("beta  (sum_ctx):  ", float(decode_lr.coef_[1]))
    print("gamma (intercept):", float(decode_lr.intercept_))
    print("metrics:", decode_metrics)
    print()

    print("=== Linear (stateless) Piecewise width=128: omega(n) ===")
    print("fallback (a,b):", pw.fallback)
    print("num_segments_fit:", len(pw.coef))
    print("metrics:", linear_metrics)
    print()

    # ---- Example: make a predictor function ----
    def predict_prefill_ms(req_len: int, num_req: int) -> float:
        L = float(req_len)
        B = float(num_req)
        n = L * B
        sum_ctx = B * (L * (L + 1.0) / 2.0)
        return float(prefill_lr.predict(np.array([[n, sum_ctx]], dtype=float))[0])

    def predict_decode_ms(req_len: int, num_req: int) -> float:
        L = float(req_len)
        B = float(num_req)
        n = B
        sum_ctx = B * L
        return float(decode_lr.predict(np.array([[n, sum_ctx]], dtype=float))[0])

    def predict_linear_ms(num_token: int) -> float:
        return float(pw.predict_one(float(num_token)))

    print("Example predictions:")
    print("  prefill  L=1408, B=4 ->", predict_prefill_ms(1408, 4), "ms")
    print("  decode   L=1024, B=8 ->", predict_decode_ms(1024, 8), "ms")
    print("  linear   n=192        ->", predict_linear_ms(192), "ms")


if __name__ == "__main__":
    # 改成你的 json 文件路径
    main("bench_results.json", test_size=0.2, seed=42)
