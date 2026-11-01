#!/usr/bin/env python3
"""Remove high-side outliers (values well above a local rolling median) from the
linear profiling points, refit the width=32 piecewise model, and plot kept vs
removed points with the old (all-points) and new (filtered) fits overlaid.
"""
import argparse
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_piecewise(tokens, values, width=32, min_points=2):
    tokens = np.asarray(tokens, float)
    values = np.asarray(values, float)
    fallback = np.linalg.lstsq(
        np.vstack([tokens, np.ones_like(tokens)]).T, values, rcond=None)[0]
    coef = {}
    for sid in sorted(set(int(math.floor(t / width)) for t in tokens)):
        idx = np.array([int(math.floor(t / width)) == sid for t in tokens])
        if idx.sum() >= min_points:
            c = np.linalg.lstsq(
                np.vstack([tokens[idx], np.ones(idx.sum())]).T,
                values[idx], rcond=None)[0]
        else:
            c = fallback
        coef[int(sid)] = [float(c[0]), float(c[1])]
    return {"width": width, "coef": coef, "fallback": [float(fallback[0]), float(fallback[1])]}


def eval_fit(fit, tokens):
    w = fit["width"]
    out = []
    for t in tokens:
        sid = int(math.floor(t / w))
        c = fit["coef"].get(sid, fit["fallback"])
        out.append(c[0] * t + c[1])
    return np.array(out)


def rolling_median(ys, window):
    n = len(ys)
    half = window // 2
    med = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        med[i] = np.median(ys[lo:hi])
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--window", type=int, default=41)
    ap.add_argument("--hi-thresh", type=float, default=1.10,
                    help="drop points whose value > local_median * hi_thresh")
    args = ap.parse_args()

    d = json.load(open(args.points))
    label = d.get("label", "linear")
    xs = np.asarray(d["xs"], float)
    ys = np.asarray(d["ys"], float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    med = rolling_median(ys, args.window)
    keep = ys <= med * args.hi_thresh
    removed = ~keep

    fit_all = fit_piecewise(xs, ys, width=args.width)
    fit_kept = fit_piecewise(xs[keep], ys[keep], width=args.width)

    grid = np.linspace(xs.min(), xs.max(), 4000)
    y_all = eval_fit(fit_all, grid)
    y_kept = eval_fit(fit_kept, grid)

    plt.figure(figsize=(13, 6))
    plt.scatter(xs[keep], ys[keep], s=6, alpha=0.5, color="#4C78A8",
                label=f"kept ({keep.sum()})")
    plt.scatter(xs[removed], ys[removed], s=16, color="red", marker="x",
                label=f"removed high-outliers ({removed.sum()})")
    plt.plot(grid, y_all, color="orange", lw=1.0, ls="--",
             label="fit (all points)")
    plt.plot(grid, y_kept, color="red", lw=1.4,
             label=f"fit (filtered, width={args.width})")
    plt.xlabel("num_token")
    plt.ylabel("stateless_linear per-layer forward time (ms)")
    plt.title(f"{label}  stateless-linear: high-outlier filtered refit "
              f"(window={args.window}, hi_thresh={args.hi_thresh})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"[refit] {label}: kept={keep.sum()} removed={removed.sum()} -> {args.out}")


if __name__ == "__main__":
    main()
