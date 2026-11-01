import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
RESULTS = REPO / "experiments" / "saturo" / "results" / "arrival"

# ------------------------------------------------------------------
# Load Azure production trace
# ------------------------------------------------------------------
csv_path = REPO / "data" / "datasets" / "AzureLLMInferenceTrace_conv.csv"
df = pd.read_csv(csv_path)
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
df = df.sort_values("TIMESTAMP").reset_index(drop=True)
df["elapsed_s"] = (df["TIMESTAMP"] - df["TIMESTAMP"].min()).dt.total_seconds()

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
BIG_SLOT = 30 * 60       # 30 minutes: divide the trace into half-hour slots
SUB_SLOT = 12 * 60       # 12 minutes: the window we want to plot
BAR_SLOT = 10            # 10 seconds per bar (finer bars)

# ------------------------------------------------------------------
# Compute per-second arrival rate (for flexible re-binning)
# ------------------------------------------------------------------
total_seconds = int(df["elapsed_s"].max()) + 1
per_sec_counts, _ = np.histogram(df["elapsed_s"], bins=np.arange(0, total_seconds + 1, 1))
per_sec_counts = per_sec_counts.astype(float)

# ------------------------------------------------------------------
# Divide trace into half-hour slots and find the 6-min sub-window
# with the largest difference (max - min) in request arrival rate.
# Prefer windows where both the max and min bars are not right at the
# window edges so the annotated extremes sit roughly in the middle of
# the plot.
# ------------------------------------------------------------------
EDGE_BARS = 6  # require max/min to be at least 1 minute away from edges
n_bars = SUB_SLOT // BAR_SLOT

total_big_slots = int(np.ceil(total_seconds / BIG_SLOT))
best_score_centered = -1.0
best_centered = None
best_score_any = -1.0
best_any = None

for big_idx in range(total_big_slots):
    big_start = big_idx * BIG_SLOT
    big_end = min((big_idx + 1) * BIG_SLOT, total_seconds)
    if big_end - big_start < SUB_SLOT:
        continue

    # slide a 6-minute window inside this half-hour slot
    for sub_start in range(big_start, big_end - SUB_SLOT + 1, 60):  # step 1 min
        sub_counts = per_sec_counts[sub_start:sub_start + SUB_SLOT]
        bar_counts = sub_counts.reshape(n_bars, BAR_SLOT).sum(axis=1)
        bar_rps = bar_counts / BAR_SLOT
        score = bar_rps.std()  # standard deviation as difference metric

        max_idx = int(np.argmax(bar_rps))
        min_idx = int(np.argmin(bar_rps))
        edge_dist = min(max_idx, min_idx, n_bars - 1 - max_idx, n_bars - 1 - min_idx)

        if score > best_score_any:
            best_score_any = score
            best_any = (big_idx, sub_start)

        if edge_dist >= EDGE_BARS and score > best_score_centered:
            best_score_centered = score
            best_centered = (big_idx, sub_start)

if best_centered is not None:
    best_big_idx, best_sub_start = best_centered
    best_score = best_score_centered
    centered = True
else:
    best_big_idx, best_sub_start = best_any
    best_score = best_score_any
    centered = False

print(f"Selected half-hour slot #{best_big_idx} (starts at {best_big_idx * BIG_SLOT}s)")
print(f"Best 12-min sub-window starts at {best_sub_start}s, std={best_score:.2f} req/s"
      f"{' (centered max/min)' if centered else ' (fallback: edge max/min)'}")

# ------------------------------------------------------------------
# Compute 10-second bars for the selected 6-minute window
# ------------------------------------------------------------------
sub_counts = per_sec_counts[best_sub_start:best_sub_start + SUB_SLOT]
n_bars = SUB_SLOT // BAR_SLOT
bar_counts = sub_counts.reshape(n_bars, BAR_SLOT).sum(axis=1)
bar_rps = bar_counts / BAR_SLOT

# Use relative time (minutes within the selected window) for the x-axis so the
# tick labels stay compact even if the selected window is far from trace start.
t_minutes = (np.arange(n_bars) * BAR_SLOT + BAR_SLOT / 2) / 60.0

print(f"  max={bar_rps.max():.2f} req/s, min={bar_rps.min():.2f} req/s, mean={bar_rps.mean():.2f} req/s")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (6, 3.5),
    "figure.dpi": 150,
})

fig, ax = plt.subplots()

# Plot the request arrival rate as a line chart.
ax.plot(
    t_minutes,
    bar_rps,
    color="#2E5AAC",
    linewidth=1.8,
    marker="o",
    markersize=3.5,
)

# Highlight the maximum and minimum points.
max_idx = int(np.argmax(bar_rps))
min_idx = int(np.argmin(bar_rps))
ax.scatter(
    t_minutes[max_idx],
    bar_rps[max_idx],
    color="#8B0000",
    s=38,
    zorder=5,
)
ax.scatter(
    t_minutes[min_idx],
    bar_rps[min_idx],
    color="#00008B",
    s=38,
    zorder=5,
)

# Annotate max/min values; place the labels just above the bars.
offset = bar_rps.max() * 0.02
ax.text(
    t_minutes[max_idx],
    bar_rps[max_idx] + offset,
    f"{bar_rps[max_idx]:.1f}",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#8B0000",
    fontweight="bold",
)
ax.text(
    t_minutes[min_idx],
    bar_rps[min_idx] + offset,
    f"{bar_rps[min_idx]:.1f}",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#00008B",
    fontweight="bold",
)

ax.set_xlabel("Time (min)")
ax.set_ylabel("Request arrival rate (req/s)")
ax.set_title("12-minute detailed view (10-second intervals)", pad=12)
ax.set_xlim(0, SUB_SLOT / 60.0)
ax.set_xticks(np.arange(0, SUB_SLOT / 60.0 + 1, 2))
# Raise the y-axis bottom so small differences are more visible.
y_bottom = max(0, bar_rps.min() * 0.85)
ax.set_ylim(y_bottom, bar_rps.max() * 1.15)
ax.grid(True, linestyle="--", alpha=0.4)

# Data source caption
fig.text(
    0.5,
    0.02,
    "Data source: Azure production traces~\\cite{stojkovic2025dynamollm}",
    ha="center",
    fontsize=9,
    style="italic",
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

out_path = SCRIPT_DIR / "request_arrival_rate_12min.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved plot to {out_path}")
