"""
Verify the conservative-then-tightening property of the residual-length quantile
used by Saturo:

    Q_alpha(l_in, t_gen) = alpha-quantile of (L_out - t_gen)
                          conditioned on  L_in = l_in  and  L_out > t_gen.

We want to check, on AzureLLMInferenceTrace_conv.csv, whether
    (1) Q_0.9(l_in, t_gen) shrinks as t_gen grows
                                              (residual quantile is decreasing);
    (2) the "total budget"  B(t) = t_gen + Q_0.9(l_in, t_gen)
        is non-increasing in t_gen
        (i.e. previously reserved capacity is actually reclaimed).

Only (2) is the strong property Saturo relies on. (1) is almost trivial.

Usage:
    python benchmarks/design/verify_remaining_quantile.py \
        --csv data/datasets/AzureLLMInferenceTrace_conv.csv \
        --alpha 0.9
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- helpers ----------------------------------------------------------

def make_input_bins(l_in: np.ndarray, n_bins: int = 4):
    """Equal-frequency bins on L_in (so every bucket has enough samples)."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(l_in, qs))
    # ensure strictly increasing
    if len(edges) < 3:
        edges = np.array([l_in.min(), np.median(l_in), l_in.max() + 1])
    edges[-1] += 1  # make right edge inclusive via < comparison below
    return edges


def residual_quantile_curve(l_out_bucket: np.ndarray,
                            alpha: float,
                            t_grid: np.ndarray,
                            min_samples: int = 30):
    """
    For a fixed L_in bucket, compute Q_alpha(L_out - t | L_out > t) at every t
    in t_grid. Returns (residual_q, total_budget, n_alive).
    """
    residual_q = np.full_like(t_grid, np.nan, dtype=float)
    n_alive = np.zeros_like(t_grid, dtype=int)
    for i, t in enumerate(t_grid):
        alive = l_out_bucket[l_out_bucket > t]
        n_alive[i] = alive.size
        if alive.size >= min_samples:
            residual_q[i] = np.quantile(alive - t, alpha)
    total_budget = t_grid + residual_q
    return residual_q, total_budget, n_alive


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--n-bins", type=int, default=4,
                    help="number of L_in buckets (equal-frequency)")
    ap.add_argument("--t-step", type=int, default=4,
                    help="step size for the t_gen grid")
    ap.add_argument("--min-samples", type=int, default=30,
                    help="minimum alive samples to compute a quantile")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows.  L_in / L_out summary:")
    print(df[["ContextTokens", "GeneratedTokens"]].describe(
        percentiles=[0.5, 0.9, 0.99]))

    l_in = df["ContextTokens"].to_numpy()
    l_out = df["GeneratedTokens"].to_numpy()

    # ---- bucket on L_in --------------------------------------------------
    edges = make_input_bins(l_in, n_bins=args.n_bins)
    print(f"\nL_in bucket edges: {edges}")

    # ---- per-bucket curves -----------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    summary_rows = []

    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        mask = (l_in >= lo) & (l_in < hi)
        l_out_b = l_out[mask]
        if l_out_b.size < 100:
            continue

        t_max = int(np.quantile(l_out_b, 0.99))   # don't go into ultra-tail
        t_grid = np.arange(0, max(t_max, 1) + 1, args.t_step)

        q_res, budget, n_alive = residual_quantile_curve(
            l_out_b, args.alpha, t_grid, args.min_samples)

        label = f"L_in∈[{int(lo)},{int(hi)})  n={l_out_b.size}"
        axes[0].plot(t_grid, q_res, label=label)
        axes[1].plot(t_grid, budget, label=label)

        # diagnostics: is budget monotonically non-increasing?
        valid = ~np.isnan(budget)
        b_valid = budget[valid]
        t_valid = t_grid[valid]
        if b_valid.size >= 2:
            diffs = np.diff(b_valid)
            frac_decreasing = (diffs <= 0).mean()
            max_increase = diffs.max()
            slope = np.polyfit(t_valid, b_valid, 1)[0]
            summary_rows.append({
                "bucket": f"[{int(lo)},{int(hi)})",
                "n": int(l_out_b.size),
                "budget(t=0)": float(b_valid[0]),
                "budget(t=max)": float(b_valid[-1]),
                "delta": float(b_valid[-1] - b_valid[0]),
                "frac_steps_non_increasing": float(frac_decreasing),
                "max_step_increase": float(max_increase),
                "linear_slope": float(slope),
            })

    # baseline: a memoryless distribution would have budget(t) = t + const,
    # i.e. slope = +1.  Saturo's claim corresponds to slope <= 0.
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"residual P{int(args.alpha*100)}  "
                       r"$Q_\alpha(L_{out}-t\mid L_{out}>t)$")
    axes[1].set_ylabel(r"total budget  $t + Q_\alpha$")
    axes[1].set_xlabel(r"$t_{gen}$ (tokens already generated)")
    axes[0].set_title("Conservative-then-tightening test "
                      f"(alpha={args.alpha})")

    # reference line: memoryless = slope 1
    t_ref = np.arange(0, axes[1].get_xlim()[1])
    for row in summary_rows[:1]:
        axes[1].plot(t_ref, t_ref + row["budget(t=0)"], "k--",
                     alpha=0.4, label="memoryless ref (slope=+1)")
        axes[1].legend(fontsize=8)
        break

    out_png = os.path.join(args.outdir, "saturo_remaining_quantile.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"\nSaved figure to {out_png}")

    summary = pd.DataFrame(summary_rows)
    print("\n=== Per-bucket diagnostics ===")
    print(summary.to_string(index=False))

    # ---- overall verdict -------------------------------------------------
    print("\nInterpretation:")
    print(" - delta < 0           => total budget shrinks with t_gen (good for Saturo).")
    print(" - linear_slope <= 0   => 'reclaim' property holds on average.")
    print(" - frac_steps_non_increasing close to 1 => monotonic shrinking.")
    print(" - If slope ≈ +1, the residual is ~memoryless and Saturo's claim FAILS.")


if __name__ == "__main__":
    main()
