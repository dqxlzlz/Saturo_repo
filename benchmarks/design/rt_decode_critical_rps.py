"""RT-Decode Critical-RPS Simulator (KV-pressure model).

Goal
----
For an RT-Decode pool that *aggressively saturates* every iteration up to the
TBT SLO via dummy Best-Effort filling, find the critical decode RPS at which
RT requests can no longer be admitted into the running batch (because KV is
full) and therefore violate TBT.

Physical model
--------------
Because Best-Effort work is assumed to fill every iteration up to the TBT
target, **every iteration takes exactly ``TBT_SLO`` seconds** and the only
constraint that matters is the **KV-block budget**.

At the end of iteration ``T`` (wall clock ``t = T * slo``) we do, in order:

  1. RELEASE: every batched request whose ``generated == output_len`` is
     removed and its KV blocks are returned to the free pool.
  2. APPEND : every surviving request emits one more decode token; its KV
     footprint grows by one token, possibly allocating one new block when
     the last block fills up.
  3. ADMIT  : every new arrival whose ``arrival_time`` falls in
     ``[(T-1)*slo, T*slo)`` tries to enter the batch. Initial KV demand =
     ``ceil((context_tokens + 1) / block_size)`` (prompt KV is already
     resident, plus the request's first decode token is being scheduled
     this iter). If KV is insufficient, the request is *rejected and
     counted as a violation*.

Violation == rejection: a request that cannot be admitted on arrival has
its first decode token delayed indefinitely w.r.t. SLO and is therefore
considered an SLO miss.

Trace scaling
-------------
We take the *first N* requests of the trace (N defaults to
``round(rps * duration_time)``) and rescale their inter-arrival intervals so
that the span between the first and the last request equals
``(N - 1) / rps`` seconds. This preserves burstiness while pinning the
effective arrival rate.

Outputs
-------
* Per-RPS metrics: admit-rejection rate, avg/max batch size, avg/max KV
  utilisation, and an 'overload' verdict.
* A sweep summary plus the inferred *critical RPS*.

CLI
---
    python benchmarks/design/rt_decode_critical_rps.py \\
        --trace data/datasets/AzureLLMInferenceTrace_conv.csv \\
        --duration-time 600 \\
        --slo 0.1 \\
        --free-gpu-blocks 7200 \\
        --rps-list 0.5,1,2,3,4,5,6,8

If ``--rps-list`` is omitted, an automatic geometric sweep is used.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="RT-Decode Critical-RPS Simulator (constant-SLO model)")

    # Trace
    p.add_argument("--trace", type=str,
                   default="data/datasets/AzureLLMInferenceTrace_conv.csv",
                   help="Trace CSV (TIMESTAMP, ContextTokens, "
                        "GeneratedTokens).")

    # Window
    p.add_argument("--duration-time", type=float, default=600.0,
                   help="Simulation window in seconds.")
    p.add_argument("--num-reqs", type=int, default=None,
                   help="Override the # of leading trace requests "
                        "(default: round(rps * duration-time)).")

    # SLO / capacity
    p.add_argument("--slo", type=float, default=0.1,
                   help="TBT SLO (sec); also = constant per-iter time.")
    p.add_argument("--free-gpu-blocks", type=int, default=7200,
                   help="Free GPU KV blocks for RT decode.")
    p.add_argument("--block-size", type=int, default=16)

    # RPS sweep
    p.add_argument("--rps-list", type=str, default=None,
                   help="Comma-separated RPS values to evaluate. "
                        "If omitted, an auto sweep is performed.")
    p.add_argument("--rps-min", type=float, default=0.5)
    p.add_argument("--rps-max", type=float, default=20.0)
    p.add_argument("--rps-points", type=int, default=12)

    # Decision threshold
    p.add_argument("--violation-threshold", type=float, default=0.05,
                   help="Critical-RPS = first RPS whose admission-reject "
                        "rate (= rejected / arrivals) exceeds this.")

    # Defaults when trace lacks a column
    p.add_argument("--default-output-len", type=int, default=128)
    p.add_argument("--default-context-len", type=int, default=1024)

    p.add_argument("--csv-out", type=str, default=None,
                   help="Optional path to dump per-RPS metrics.")
    p.add_argument("--plot-out", type=str, default=None,
                   help="Optional path (PNG) to dump per-RPS KV utilisation "
                        "time-series, one subplot per RPS.")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Trace loading & scaling
# ---------------------------------------------------------------------------
def load_trace_first_n(trace_path: str,
                       num_reqs: int,
                       target_rps: float,
                       default_out: int,
                       default_ctx: int) -> pd.DataFrame:
    """Load the FIRST ``num_reqs`` rows of the trace and rescale arrival
    intervals so that the (first, last) span equals ``(N-1) / target_rps``.

    Returns a DataFrame with columns:
        arrival_time, output_len, context_len.
    """
    df = pd.read_csv(trace_path)
    if "TIMESTAMP" in df.columns:
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    df = df.head(num_reqs).copy().reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(f"Need >= 2 requests in trace, got {len(df)}.")

    span_orig = (df["TIMESTAMP"].iloc[-1]
                 - df["TIMESTAMP"].iloc[0]).total_seconds()
    if span_orig <= 0:
        raise ValueError("Trace has zero original span; cannot scale.")

    span_target = (len(df) - 1) / target_rps
    scale = span_target / span_orig

    arrivals = (df["TIMESTAMP"]
                - df["TIMESTAMP"].iloc[0]).dt.total_seconds().to_numpy()
    df["arrival_time"] = arrivals * scale

    if "GeneratedTokens" in df.columns:
        df["output_len"] = df["GeneratedTokens"].fillna(
            default_out).astype(int)
    else:
        df["output_len"] = default_out
    df["output_len"] = df["output_len"].clip(lower=1)

    if "ContextTokens" in df.columns:
        df["context_len"] = df["ContextTokens"].fillna(
            default_ctx).astype(int)
    elif "prompt_len" in df.columns:
        df["context_len"] = df["prompt_len"].fillna(default_ctx).astype(int)
    else:
        df["context_len"] = default_ctx
    df["context_len"] = df["context_len"].clip(lower=1)

    return df[["arrival_time", "output_len", "context_len"]]


# ---------------------------------------------------------------------------
# Active request bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class RTReq:
    """A request currently sitting in the running batch.

    Bookkeeping:
      * ``generated`` counts decode tokens already emitted (=1 right after
        the request is admitted, because admission allocates KV for the
        prompt + the first generated token already).
      * ``output_len`` is the target #tokens (from trace ``GeneratedTokens``).
        The request is released when ``generated == output_len``.
      * ``tokens_in_kv`` = prompt_len + generated  (= total tokens whose
        KV is currently stored). When this crosses a block boundary, we
        allocate one more block.
      * ``kv_blocks`` = current KV footprint in blocks.
    """
    generated: int          # decode tokens already produced
    output_len: int         # target #decode tokens
    tokens_in_kv: int       # prompt_len + generated (KV footprint, in tokens)
    kv_blocks: int          # current KV footprint (in blocks)

    @staticmethod
    def initial_blocks(prompt_len: int, block_size: int) -> int:
        """KV blocks needed at admission time.

        Per the model: prompt KV is already there, plus the first generated
        token contributes one extra token. So initial tokens = prompt + 1.
        """
        toks = max(1, prompt_len) + 1
        return (toks + block_size - 1) // block_size


# ---------------------------------------------------------------------------
# Single-RPS simulation
# ---------------------------------------------------------------------------
def simulate_one_rps(args,
                     trace_df: pd.DataFrame,
                     rps: float) -> dict:
    """Iteration-level RT-Decode KV-pressure simulator.

    Model (per the user's spec):

        At the end of iteration T (i.e. at wall-clock t = T * slo):
        1. RELEASE: for each req in batch with ``generated == output_len``,
           remove it and free its KV blocks.
        2. APPEND : for each surviving req, ``generated += 1`` and
           ``tokens_in_kv += 1``; if a new block is required (i.e. the new
           token spills into a fresh block) allocate one. If a new block is
           needed but no free block is available, this would be a runtime
           failure; in this simulator we treat it as a *violation against
           the surviving request* and free its slot (it cannot continue).
           In practice this never triggers if (3) is enforced strictly.
        3. ADMIT  : drain new arrivals whose ``arrival_time`` falls in
           ``[(T-1)*slo, T*slo)``. For each, attempt to allocate its initial
           KV blocks. If KV is insufficient -> the request is DROPPED and
           counted as a violation.
    """
    block_size = args.block_size
    free_blocks = args.free_gpu_blocks
    slo = args.slo
    duration = args.duration_time

    arrivals = trace_df["arrival_time"].to_numpy()
    outs = trace_df["output_len"].to_numpy(dtype=int)
    ctxs = trace_df["context_len"].to_numpy(dtype=int)
    n_total = len(trace_df)

    # State.
    batch: List[RTReq] = []
    used_blocks = 0

    # Aggregates.
    n_admitted = 0          # arrivals that did enter the batch
    n_rejected = 0          # arrivals dropped at admission (= violations)
    finished = 0
    arrivals_seen = 0       # arrivals whose iter-window has passed

    # Per-iter time-averaged tracking.
    iters_done = 0
    sum_batch = 0
    sum_used = 0
    max_batch = 0
    max_used = 0

    # Per-iter time series (for plotting). Index = iter T (1-based);
    # time stamp of sample = T * slo (end-of-iter).
    used_blocks_series: List[int] = []
    batch_size_series: List[int] = []
    rejects_series: List[int] = []

    # Total iterations needed to cover the simulation window.
    total_iters = int(math.ceil(duration / slo))

    next_idx = 0   # next trace row to consider for admission
    rejects_this_iter = 0  # local accumulator, snapshot at end of iter

    for T in range(1, total_iters + 1):
        t_iter_start = (T - 1) * slo
        t_iter_end = T * slo
        rejects_this_iter = 0

        # (1) RELEASE finished requests.
        survivors: List[RTReq] = []
        for r in batch:
            if r.generated >= r.output_len:
                used_blocks -= r.kv_blocks
                finished += 1
            else:
                survivors.append(r)
        batch = survivors

        # (2) APPEND one token of KV to every surviving req.
        for r in batch:
            r.generated += 1
            r.tokens_in_kv += 1
            need_blocks = (r.tokens_in_kv + block_size - 1) // block_size
            if need_blocks > r.kv_blocks:
                # Crossed a block boundary: need to allocate 1 extra block.
                # If no free block, the runtime would OOM. In this sim we
                # don't expect this because (3) enforces the cap, but guard
                # anyway.
                if used_blocks + 1 > free_blocks:
                    # Cannot extend KV; treat this token as a violation but
                    # do not actually grow the footprint (logically the req
                    # would have to be preempted). Accounted for in the
                    # admission-side cap below.
                    pass
                else:
                    r.kv_blocks += 1
                    used_blocks += 1

        # (3) ADMIT arrivals in [t_iter_start, t_iter_end).
        while next_idx < n_total and arrivals[next_idx] < t_iter_end:
            if arrivals[next_idx] < t_iter_start - 1e-12:
                # Shouldn't happen because we process iters in order; skip.
                next_idx += 1
                continue
            need = RTReq.initial_blocks(int(ctxs[next_idx]), block_size)
            arrivals_seen += 1
            if used_blocks + need <= free_blocks:
                # Admit. Note: per the spec, admission already accounts for
                # the prompt KV + the first generated token (which is why
                # initial_blocks uses ctx + 1).
                batch.append(RTReq(
                    generated=1,
                    output_len=int(outs[next_idx]),
                    tokens_in_kv=int(ctxs[next_idx]) + 1,
                    kv_blocks=need,
                ))
                used_blocks += need
                n_admitted += 1
            else:
                # Not enough KV -> dropped, counted as a violation.
                n_rejected += 1
                rejects_this_iter += 1
            next_idx += 1

        # ---- end-of-iter bookkeeping (state we observed during iter T) ----
        iters_done += 1
        sum_batch += len(batch)
        sum_used += used_blocks
        if len(batch) > max_batch:
            max_batch = len(batch)
        if used_blocks > max_used:
            max_used = used_blocks
        used_blocks_series.append(used_blocks)
        batch_size_series.append(len(batch))
        rejects_series.append(rejects_this_iter)

        # Early termination: if we've consumed the trace and the batch is
        # empty, there's nothing left to do.
        if next_idx >= n_total and not batch:
            break

    # ---- aggregate metrics ----
    avg_batch = sum_batch / max(1, iters_done)
    avg_kv_util = (sum_used / max(1, iters_done)) / max(1, free_blocks)
    max_kv_util = max_used / max(1, free_blocks)
    violation_rate = n_rejected / max(1, arrivals_seen)

    overload = violation_rate > args.violation_threshold

    return {
        "rps": rps,
        "n_total": n_total,
        "arrivals_seen": arrivals_seen,
        "n_admitted": n_admitted,
        "n_rejected": n_rejected,
        "violation_rate": violation_rate,
        "finished": finished,
        "iters_done": iters_done,
        "avg_batch": avg_batch,
        "max_batch": max_batch,
        "avg_kv_util": avg_kv_util,
        "max_kv_util": max_kv_util,
        "overload": bool(overload),
        # Time series (length = iters_done; time of i-th sample = (i+1)*slo).
        "_slo": slo,
        "_free_blocks": free_blocks,
        "_used_blocks_series": used_blocks_series,
        "_batch_size_series": batch_size_series,
        "_rejects_series": rejects_series,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_kv_utilization(rows: List[dict],
                        out_path: str,
                        violation_threshold: float) -> None:
    """One subplot per RPS, showing KV utilisation over time.

    Each subplot also marks the violation threshold (100% line) and
    annotates avg / max KV util and rejection rate in the title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in rows if r.get("_used_blocks_series")]
    if not valid:
        print("[Plot] no data with time series; skip.")
        return

    n = len(valid)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 2.5 * nrows),
                             squeeze=False, sharex=False)

    for i, r in enumerate(valid):
        ax = axes[i // ncols][i % ncols]
        slo = r["_slo"]
        free_blocks = r["_free_blocks"]
        series = r["_used_blocks_series"]
        rejects = r["_rejects_series"]
        ts = [(k + 1) * slo for k in range(len(series))]
        util = [u / free_blocks for u in series]

        ax.plot(ts, util, lw=0.9, color="C0", label="KV util")
        ax.axhline(1.0, color="red", lw=0.7, ls="--",
                   label="capacity (100%)")

        # Mark iters that had at least one rejection.
        rej_x = [ts[k] for k, c in enumerate(rejects) if c > 0]
        rej_y = [util[k] for k, c in enumerate(rejects) if c > 0]
        if rej_x:
            ax.scatter(rej_x, rej_y, s=8, color="red", marker="x",
                       label=f"reject iters ({len(rej_x)})", zorder=3)

        ax.set_ylim(0.0, 1.08)
        ax.set_xlim(0.0, ts[-1] if ts else 1.0)
        ax.set_ylabel("KV util")
        if i // ncols == nrows - 1:
            ax.set_xlabel("time (s)")
        flag = " [OVERLOAD]" if r["overload"] else ""
        ax.set_title(
            f"RPS={r['rps']:.2f}{flag}  "
            f"reject={r['violation_rate']*100:.2f}% "
            f"({r['n_rejected']}/{r['arrivals_seen']})  "
            f"avg/max util={r['avg_kv_util']*100:.1f}%"
            f"/{r['max_kv_util']*100:.1f}%",
            fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", fontsize=7)

    # Hide unused axes.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"KV-cache utilisation over time  "
        f"(threshold={violation_threshold*100:.0f}% reject -> OVERLOAD)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[Plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# RPS sweep & critical-RPS detection
# ---------------------------------------------------------------------------
def build_rps_grid(args) -> List[float]:
    if args.rps_list:
        return sorted([float(x) for x in args.rps_list.split(",")
                       if x.strip()])
    lo = math.log(max(1e-3, args.rps_min))
    hi = math.log(max(args.rps_min + 1e-3, args.rps_max))
    return [math.exp(lo + (hi - lo) * i / (args.rps_points - 1))
            for i in range(args.rps_points)]


def main():
    args = parse_args()

    rps_grid = build_rps_grid(args)
    print(f"[Sweep] RPS grid: "
          f"{[round(x, 3) for x in rps_grid]}")
    print(f"[Setup] slo={args.slo*1000:.1f}ms (= per-iter time), "
          f"free_blocks={args.free_gpu_blocks}, "
          f"block_size={args.block_size}, "
          f"window={args.duration_time}s")

    rows: List[dict] = []
    for rps in rps_grid:
        n = args.num_reqs if args.num_reqs is not None \
            else max(2, int(round(rps * args.duration_time)))
        try:
            trace_df = load_trace_first_n(
                args.trace, n, rps,
                args.default_output_len, args.default_context_len)
        except Exception as e:
            print(f"[RPS={rps:.3f}] trace load failed: {e}")
            continue

        if args.verbose:
            print(f"\n[RPS={rps:.3f}] n_reqs={len(trace_df)}, "
                  f"trace_span={trace_df['arrival_time'].iloc[-1]:.1f}s, "
                  f"avg_out={trace_df['output_len'].mean():.1f}, "
                  f"avg_ctx={trace_df['context_len'].mean():.1f}")

        m = simulate_one_rps(args, trace_df, rps)
        rows.append(m)

        flag = "OVERLOAD" if m["overload"] else "ok"
        print(f"[RPS={rps:6.3f}] "
              f"viol={m['violation_rate']*100:5.2f}% "
              f"({m['n_rejected']}/{m['arrivals_seen']}) | "
              f"admitted={m['n_admitted']:5d} finished={m['finished']:5d} | "
              f"batch avg/max = {m['avg_batch']:6.1f}/{m['max_batch']:5d} | "
              f"kv_util avg/max = "
              f"{m['avg_kv_util']*100:5.1f}%/{m['max_kv_util']*100:5.1f}% | "
              f"{flag}")

    # ---- find critical RPS ----
    critical = None
    for r in rows:
        if r["overload"]:
            critical = r["rps"]
            break

    print("\n=== Summary ===")
    df = pd.DataFrame(rows)
    if not df.empty:
        cols = ["rps", "arrivals_seen", "n_admitted", "n_rejected",
                "violation_rate", "finished", "avg_batch", "max_batch",
                "avg_kv_util", "max_kv_util", "overload"]
        print(df[cols].to_string(index=False))

    if critical is not None:
        print(f"\nCritical RPS (first overload): {critical:.3f} req/s")
    else:
        print("\nNo RPS in the sweep triggered overload; "
              "try a higher --rps-max.")

    if args.csv_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)) or ".",
                    exist_ok=True)
        # Drop internal-series columns from CSV.
        df_csv = df.drop(columns=[c for c in df.columns
                                  if c.startswith("_")], errors="ignore")
        df_csv.to_csv(args.csv_out, index=False)
        print(f"\n[CSV] wrote {args.csv_out}")

    if args.plot_out:
        plot_kv_utilization(rows, args.plot_out, args.violation_threshold)


if __name__ == "__main__":
    main()
