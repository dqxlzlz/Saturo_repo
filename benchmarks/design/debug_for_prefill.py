import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

OUT_DIR = "debug_info"
# JSON_PATH = "bench_results.json"
JSON_PATH = "profile_bench_results_test.json"

RANDOM_SEED = 0
TEST_SIZE = 0.2


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def finite_df(df: pd.DataFrame, col="ms_mean"):
    v = df[col].to_numpy(float)
    return df[np.isfinite(v)].copy()


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def eval_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def scatter_pred_vs_true(y_true, y_pred, title, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12)
    lo = float(np.min([y_true.min(), y_pred.min()]))
    hi = float(np.max([y_true.max(), y_pred.max()]))
    plt.plot([lo, hi], [lo, hi])  # y=x
    plt.xlabel("true (ms)")
    plt.ylabel("pred (ms)")
    plt.title(title)
    savefig(out_path)


def residual_heatmap(df, pred_col, title, out_path):
    # heatmap over req_len x num_req
    tmp = df.copy()
    tmp["residual"] = tmp[pred_col] - tmp["ms_mean"]
    pivot = tmp.pivot_table(index="req_len", columns="num_req", values="residual", aggfunc="mean")
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="pred - true (ms)")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("num_req")
    plt.ylabel("req_len")
    plt.title(title)
    savefig(out_path)


def plot_fixed_numreq(df, kind_prefix):
    # for each num_req: ms_mean vs req_len
    for b in sorted(df["num_req"].unique()):
        sub = df[df["num_req"] == b].sort_values("req_len")
        plt.figure(figsize=(7, 4))
        plt.plot(sub["req_len"], sub["ms_median"], marker="o")
        plt.xlabel("req_len")
        plt.ylabel("ms_mean (ms)")
        plt.title(f"{kind_prefix}: ms_mean vs req_len (num_req={b})")
        savefig(os.path.join(OUT_DIR, f"{kind_prefix}_ms_vs_len_numreq_{int(b)}.png"))


def plot_fixed_reqlen(df, kind_prefix):
    # for each req_len: ms_mean vs num_req
    for l in sorted(df["req_len"].unique()):
        sub = df[df["req_len"] == l].sort_values("num_req")
        plt.figure(figsize=(7, 4))
        plt.plot(sub["num_req"], sub["ms_median"], marker="o")
        plt.xlabel("num_req")
        plt.ylabel("ms_mean (ms)")
        plt.title(f"{kind_prefix}: ms_mean vs num_req (req_len={int(l)})")
        savefig(os.path.join(OUT_DIR, f"{kind_prefix}_ms_vs_numreq_reqlen_{int(l)}.png"))


def plot_normalized_prefill(df):
    # normalized plots to see scaling
    df = df.copy()
    L = df["req_len"].to_numpy(float)
    B = df["num_req"].to_numpy(float)
    t = df["ms_median"].to_numpy(float)

    df["t_div_B"] = t / np.clip(B, 1e-9, None)
    df["t_div_BL"] = t / np.clip(B * L, 1e-9, None)
    df["t_div_BL2"] = t / np.clip(B * (L ** 2), 1e-9, None)

    # plot avg over B for each L
    g = df.groupby("req_len", as_index=False).mean(numeric_only=True).sort_values("req_len")

    plt.figure(figsize=(7, 4))
    plt.plot(g["req_len"], g["t_div_B"], marker="o")
    plt.xlabel("req_len")
    plt.ylabel("ms_mean / B")
    plt.title("prefill: (ms/B) vs req_len (avg over num_req)")
    savefig(os.path.join(OUT_DIR, "prefill_norm_ms_div_B_vs_len.png"))

    plt.figure(figsize=(7, 4))
    plt.plot(g["req_len"], g["t_div_BL"], marker="o")
    plt.xlabel("req_len")
    plt.ylabel("ms_mean / (B*L)")
    plt.title("prefill: (ms/(B*L)) vs req_len (avg over num_req)")
    savefig(os.path.join(OUT_DIR, "prefill_norm_ms_div_BL_vs_len.png"))

    plt.figure(figsize=(7, 4))
    plt.plot(g["req_len"], g["t_div_BL2"], marker="o")
    plt.xlabel("req_len")
    plt.ylabel("ms_mean / (B*L^2)")
    plt.title("prefill: (ms/(B*L^2)) vs req_len (avg over num_req)")
    savefig(os.path.join(OUT_DIR, "prefill_norm_ms_div_BL2_vs_len.png"))


def fit_and_debug_prefill(df):
    # build features
    L = df["req_len"].to_numpy(float)
    B = df["num_req"].to_numpy(float)
    y = df["ms_median"].to_numpy(float)

    # your original: n and sum_ctx
    n = B * L
    sum_ctx = B * (L * (L + 1.0) / 2.0)
    X = np.stack([n, sum_ctx], axis=1)

    # alternative: B + BL + BL^2 (often more stable & interpretable)
    X3 = np.stack([B, B * L, B * (L ** 2)], axis=1)

    # split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    X3_tr, X3_te, _, _ = train_test_split(X3, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # OLS on (n, sum_ctx)
    ols = LinearRegression().fit(X_tr, y_tr)
    pred = ols.predict(X_te)
    m = eval_metrics(y_te, pred)

    scatter_pred_vs_true(
        y_te, pred,
        f"prefill OLS (n,sum_ctx) R2={m['R2']:.3f} MAPE={m['MAPE']:.3f}",
        os.path.join(OUT_DIR, "prefill_fit_ols_n_sumctx_scatter.png"),
    )

    # Ridge on (n, sum_ctx) to mitigate collinearity
    ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
    pred_r = ridge.predict(X_te)
    mr = eval_metrics(y_te, pred_r)

    scatter_pred_vs_true(
        y_te, pred_r,
        f"prefill Ridge (n,sum_ctx) R2={mr['R2']:.3f} MAPE={mr['MAPE']:.3f}",
        os.path.join(OUT_DIR, "prefill_fit_ridge_n_sumctx_scatter.png"),
    )

    # OLS on (B,BL,BL2)
    ols3 = LinearRegression().fit(X3_tr, y_tr)
    pred3 = ols3.predict(X3_te)
    m3 = eval_metrics(y_te, pred3)

    scatter_pred_vs_true(
        y_te, pred3,
        f"prefill OLS (B,BL,BL2) R2={m3['R2']:.3f} MAPE={m3['MAPE']:.3f}",
        os.path.join(OUT_DIR, "prefill_fit_ols_B_BL_BL2_scatter.png"),
    )

    # residual heatmap on full data (pick best among above by R2)
    # compute preds on full df using best model
    best_name = "ols_n_sumctx"
    best_model = ols
    best_X_full = X
    best_m = m

    if mr["R2"] > best_m["R2"]:
        best_name, best_model, best_X_full, best_m = "ridge_n_sumctx", ridge, X, mr
    if m3["R2"] > best_m["R2"]:
        best_name, best_model, best_X_full, best_m = "ols_B_BL_BL2", ols3, X3, m3

    df2 = df.copy()
    df2["pred"] = best_model.predict(best_X_full)
    residual_heatmap(
        df2, "pred",
        f"prefill residual heatmap ({best_name})",
        os.path.join(OUT_DIR, "prefill_residual_heatmap.png"),
    )

    # save text summary
    corr = float(np.corrcoef(n, sum_ctx)[0, 1])
    with open(os.path.join(OUT_DIR, "prefill_fit_summary.txt"), "w") as f:
        f.write(f"corr(n,sum_ctx) = {corr}\n\n")
        f.write("OLS (n,sum_ctx):\n")
        f.write(f"  coef={ols.coef_.tolist()} intercept={float(ols.intercept_)} metrics={m}\n\n")
        f.write("Ridge (n,sum_ctx):\n")
        f.write(f"  coef={ridge.coef_.tolist()} intercept={float(ridge.intercept_)} metrics={mr}\n\n")
        f.write("OLS (B,BL,BL2):\n")
        f.write(f"  coef={ols3.coef_.tolist()} intercept={float(ols3.intercept_)} metrics={m3}\n\n")
        f.write(f"Best picked for heatmap: {best_name} metrics={best_m}\n")


def fit_and_debug_decode(df):
    # decode common feature: n=B, sum_ctx=B*L
    L = df["req_len"].to_numpy(float)
    B = df["num_req"].to_numpy(float)
    y = df["ms_mean"].to_numpy(float)

    n = B
    sum_ctx = B * L
    X = np.stack([n, sum_ctx], axis=1)

    # add B explicitly (often helps when PP comm exists)
    X2 = np.stack([B, B * L], axis=1)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    X2_tr, X2_te, _, _ = train_test_split(X2, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    ols = LinearRegression().fit(X_tr, y_tr)
    pred = ols.predict(X_te)
    m = eval_metrics(y_te, pred)
    scatter_pred_vs_true(
        y_te, pred,
        f"decode OLS (B,BL) R2={m['R2']:.3f} MAPE={m['MAPE']:.3f}",
        os.path.join(OUT_DIR, "decode_fit_ols_scatter.png"),
    )

    ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
    pred_r = ridge.predict(X_te)
    mr = eval_metrics(y_te, pred_r)
    scatter_pred_vs_true(
        y_te, pred_r,
        f"decode Ridge (B,BL) R2={mr['R2']:.3f} MAPE={mr['MAPE']:.3f}",
        os.path.join(OUT_DIR, "decode_fit_ridge_scatter.png"),
    )

    ols2 = LinearRegression().fit(X2_tr, y_tr)
    pred2 = ols2.predict(X2_te)
    m2 = eval_metrics(y_te, pred2)
    scatter_pred_vs_true(
        y_te, pred2,
        f"decode OLS (B,BL) (explicit) R2={m2['R2']:.3f} MAPE={m2['MAPE']:.3f}",
        os.path.join(OUT_DIR, "decode_fit_ols_explicit_scatter.png"),
    )

    # residual heatmap pick best
    best_name, best_model, best_X_full, best_m = "ols", ols, X, m
    if mr["R2"] > best_m["R2"]:
        best_name, best_model, best_X_full, best_m = "ridge", ridge, X, mr
    if m2["R2"] > best_m["R2"]:
        best_name, best_model, best_X_full, best_m = "ols_explicit", ols2, X2, m2

    df2 = df.copy()
    df2["pred"] = best_model.predict(best_X_full)
    residual_heatmap(
        df2, "pred",
        f"decode residual heatmap ({best_name})",
        os.path.join(OUT_DIR, "decode_residual_heatmap.png"),
    )

    with open(os.path.join(OUT_DIR, "decode_fit_summary.txt"), "w") as f:
        f.write("OLS (B,BL):\n")
        f.write(f"  coef={ols.coef_.tolist()} intercept={float(ols.intercept_)} metrics={m}\n\n")
        f.write("Ridge (B,BL):\n")
        f.write(f"  coef={ridge.coef_.tolist()} intercept={float(ridge.intercept_)} metrics={mr}\n\n")
        f.write("OLS explicit (B,BL):\n")
        f.write(f"  coef={ols2.coef_.tolist()} intercept={float(ols2.intercept_)} metrics={m2}\n\n")
        f.write(f"Best picked for heatmap: {best_name} metrics={best_m}\n")


def main():
    ensure_dir(OUT_DIR)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prefill
    prefill = finite_df(pd.DataFrame(data["prefill_attn_ms"]))
    plot_fixed_numreq(prefill, "prefill")
    plot_fixed_reqlen(prefill, "prefill")
    plot_normalized_prefill(prefill)
    fit_and_debug_prefill(prefill)

    # Decode
    decode = finite_df(pd.DataFrame(data["decode_attn_ms"]))
    plot_fixed_numreq(decode, "decode")
    plot_fixed_reqlen(decode, "decode")
    fit_and_debug_decode(decode)

    with open(os.path.join(OUT_DIR, "README.txt"), "w") as f:
        f.write(
            "Generated plots:\n"
            "- prefill_ms_vs_len_numreq_*.png\n"
            "- prefill_ms_vs_numreq_reqlen_*.png\n"
            "- prefill_norm_*.png\n"
            "- prefill_fit_*_scatter.png, prefill_residual_heatmap.png, prefill_fit_summary.txt\n"
            "- decode_ms_vs_len_numreq_*.png\n"
            "- decode_ms_vs_numreq_reqlen_*.png\n"
            "- decode_fit_*_scatter.png, decode_residual_heatmap.png, decode_fit_summary.txt\n"
        )

    print(f"[OK] Saved debug plots to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
