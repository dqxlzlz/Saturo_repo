#!/usr/bin/env python3
"""Fixed chunk-size limitation experiment (RPS=1, PD-aggregated RT+BE colocate).

Dual-axis plot of BE Goodput and RT P99 TBT versus the fixed scheduler token
budget (chunk size).  Data source: ``chunk_tbt_goodput_rps1.csv`` in this
directory.

Provenance of the data points:
  * chunk=128 is a REAL measured run produced by
    ``saturo/experiments/colocate/benchmark_fixed_budget.py`` at RPS=1
    (see ``chunk128.json`` / ``chunk128.log`` here).
  * the remaining points were reconstructed from an earlier reference figure
    whose raw data was not saved; treat them as approximate.

Regenerate the measured point with, e.g.:
  GPU_ID=0 RT_RPS=1.0 CHUNK_SIZE=128 TBT_SLO_MS=100 WARMUP_S=20 DURATION_S=180 \
  MIN_BE_DECODE_READY=5 BE_LEAD_S=2 BE_REQUESTS=128 \
  OUTPUT_JSON=.../chunk128.json \
  bash saturo/experiments/colocate/run_fixed_budget.sh --max-num-seqs 128
(chunk<256 requires --max-num-seqs <= chunk so vLLM's
 max_num_batched_tokens >= max_num_seqs constraint holds.)
"""
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "chunk_tbt_goodput_rps1.csv")
OUT = os.path.join(HERE, "chunk_tbt_goodput_rps1.png")


def main():
    df = pd.read_csv(CSV).sort_values("chunk_size")
    chunk = df["chunk_size"].tolist()
    goodput = df["be_goodput_req_s"].tolist()
    p99_tbt = df["p99_tbt_ms"].tolist()
    tbt_slo = float(df["tbt_slo_ms"].iloc[0])

    fig, ax1 = plt.subplots(figsize=(8, 5))

    c_orange = "#E8862A"
    c_blue = "#4472C4"

    ax1.set_xlabel("Chunk Size", fontweight="bold")
    ax1.set_ylabel("BE Throughput (req/s)", color=c_orange, fontweight="bold")
    ax1.plot(chunk, goodput, "^-", color=c_orange, linewidth=2.5,
             markersize=9, label="BE Throughput")
    ax1.tick_params(axis="y", labelcolor=c_orange)

    ax2 = ax1.twinx()
    ax2.set_ylabel("P99 TBT (ms)", color=c_blue, fontweight="bold")
    ax2.plot(chunk, p99_tbt, "s-", color=c_blue, linewidth=2.5,
             markersize=8, label="P99 TBT")
    ax2.tick_params(axis="y", labelcolor=c_blue)

    # TBT SLO line lives on the P99 TBT (right) axis.
    ax2.axhline(tbt_slo, color="red", linestyle="--", linewidth=2)
    ax2.text(chunk[-1], tbt_slo + 3, f"TBT SLO = {tbt_slo:.0f}ms",
             color="red", ha="right", va="bottom", fontweight="bold")

    ax1.set_xticks(chunk)
    ax1.set_xticklabels([str(c) for c in chunk])

    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
