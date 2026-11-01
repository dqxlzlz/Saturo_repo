"""Saturo BE Saturation & Preemption Mechanism Illustration.

Paper-style mechanism figure:

    (a) Conservative : small BE chunks leave idle gaps between RT requests.
    (b) Saturo       : BE chunks grow to fill the idle window; on RT
                       arrival the scheduler tests the SLO predicate
                          T_rem + T_queue + T_RT  <=  SLO
                       to decide "finish" vs "preempt".

Each BE chunk is drawn as a stacked composite of Prefill (c_p) on top
and Decode (c_d) at the bottom, matching the Saturo C = c_p + c_d
decomposition.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 1.1,
    "axes.edgecolor": "#222",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_RT = "#6BAED6"          # RT prefill
C_BE_P = "#2171B5"        # BE prefill
C_BE_D = "#74C476"        # BE decode
C_IDLE = "#ECECEC"
C_AXIS = "#222222"
C_OK = "#1E7F34"
C_BAD = "#C0392B"
C_GUIDE = "#555555"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def draw_timeline_axis(ax, x0, x1, y, label):
    """Axis line with arrow head."""
    ax.annotate("", xy=(x1 + 0.15, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=C_AXIS, lw=1.4),
                annotation_clip=False)
    ax.text(x1 + 0.25, y, "Timeline",
            va="center", ha="left", fontsize=10, color=C_AXIS)
    ax.text(x0 - 0.15, y + 0.62, label, ha="right", va="center",
            fontsize=11, color=C_AXIS, fontweight="bold")


def rect(ax, x, y, w, h, fc, ec, lw=0.9, alpha=1.0, ls="-", zorder=3,
         hatch=None):
    r = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                  alpha=alpha, ls=ls, zorder=zorder, hatch=hatch)
    ax.add_patch(r)
    return r


def draw_rt(ax, x, w, y, h, label=None):
    rect(ax, x, y, w, h, C_RT, "#1F4E79")
    if label:
        ax.text(x + w / 2, y + h + 0.06, label,
                ha="center", va="bottom", fontsize=9.5,
                color="#1F4E79", fontweight="bold")


def draw_be(ax, x, w, y, h, p_frac=0.5, alpha=1.0, label=None,
            hatched=False):
    """BE chunk = prefill (top) + decode (bottom)."""
    if w <= 0:
        return
    h_p = h * p_frac
    h_d = h - h_p
    rect(ax, x, y,       w, h_d, C_BE_D, "#1B5E20", alpha=alpha)
    rect(ax, x, y + h_d, w, h_p, C_BE_P, "#0B3D66", alpha=alpha)
    if hatched:
        rect(ax, x, y, w, h, "none", "#7a0000", lw=0.8,
             hatch="////", zorder=4)
    if label and w > 0.55:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold", zorder=5)


def draw_idle(ax, x, w, y, h, label=None):
    if w <= 0.02:
        return
    rect(ax, x, y, w, h, C_IDLE, "#9A9A9A", lw=0.7, ls="--", zorder=2)
    if label and w > 0.8:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=8.5,
                color="#666", style="italic", zorder=5)


def draw_arrival(ax, x, y_top, y_bot, color=C_GUIDE, label=None):
    ax.plot([x, x], [y_bot, y_top], ls=(0, (3, 2)), color=color, lw=1.1,
            zorder=1)
    if label:
        ax.text(x, y_top + 0.05, label, ha="center", va="bottom",
                fontsize=8.7, color=color)


def brace(ax, x0, x1, y, label, color=C_GUIDE, up=True, fs=9,
          tick_frac=0.06):
    sgn = 1 if up else -1
    tick = tick_frac * sgn
    mid = (x0 + x1) / 2
    ax.plot([x0, x1], [y, y], color=color, lw=1.0)
    ax.plot([x0, x0], [y, y + tick], color=color, lw=1.0)
    ax.plot([x1, x1], [y, y + tick], color=color, lw=1.0)
    ax.text(mid, y + tick + 0.02 * sgn, label,
            ha="center", va="bottom" if up else "top",
            fontsize=fs, color=color)


# ---------------------------------------------------------------------------
# Panel (a): Conservative (small chunk -> fragmentation)
# ---------------------------------------------------------------------------
def panel_a(ax, t_end=9.2):
    y = 0.55
    h = 0.55
    draw_timeline_axis(ax, -0.05, t_end, y + h / 2,
                       label="(a) Conservative\n(small BE chunk)")

    # RT1
    x = 0.5
    draw_rt(ax, x, 0.55, y, h, label="RT$_1$")
    x += 0.55 + 0.10
    # small BE chunks
    for _ in range(2):
        draw_be(ax, x, 0.45, y, h, p_frac=0.5)
        x += 0.45 + 0.10
    # idle gap
    rt2_x = 4.2
    draw_idle(ax, x, rt2_x - x, y, h, label="idle  (fragment)")
    draw_rt(ax, rt2_x, 0.55, y, h, label="RT$_2$")
    x = rt2_x + 0.55 + 0.10
    draw_be(ax, x, 0.45, y, h, p_frac=0.5)
    x += 0.45 + 0.10
    rt3_x = 8.2
    draw_idle(ax, x, rt3_x - x, y, h, label="idle  (fragment)")
    draw_rt(ax, rt3_x, 0.55, y, h, label="RT$_3$")

    # Utilization loss annotations
    brace(ax, x - 3.2, rt3_x - 0.02, y + h + 0.20,
          "under-utilization", color=C_BAD, up=True)

    # Utilization badge
    ax.text(t_end + 1.15, y + h / 2,
            "BE tput  $\\downarrow$",
            ha="left", va="center", fontsize=10.5, color=C_BAD,
            fontweight="bold")


# ---------------------------------------------------------------------------
# Panel (b): Saturo (saturated + SLO-aware preemption)
# ---------------------------------------------------------------------------
def panel_b(ax, t_end=9.2):
    y = 0.55
    h = 0.55
    draw_timeline_axis(ax, -0.05, t_end, y + h / 2,
                       label="(b) Saturo\n(saturated + SLO-aware)")

    # ---- BE chunk filling the initial idle window ----------------------
    rt1_arr = 2.55
    be1_x, be1_w = 0.15, 2.30
    draw_be(ax, be1_x, be1_w, y, h, p_frac=0.5,
            label="BE chunk  (C = c$_p$ + c$_d$)")
    draw_idle(ax, be1_x + be1_w, rt1_arr - (be1_x + be1_w), y, h)
    draw_rt(ax, rt1_arr, 0.55, y, h, label="RT$_1$")
    # RT1 arrival guide
    draw_arrival(ax, rt1_arr, y + h + 0.25, y - 0.05, color=C_OK,
                 label="RT$_1$ arrives")

    # ---- Scenario I: RT2 arrives mid-chunk, predicate PASSES -----------
    be2_x = rt1_arr + 0.55 + 0.08
    be2_w = 2.20
    rt2_arr = be2_x + 1.65     # arrives while BE chunk is still running
    rt2_w = 0.55
    draw_be(ax, be2_x, be2_w, y, h, p_frac=0.5)          # chunk finishes
    rt2_start = be2_x + be2_w
    draw_rt(ax, rt2_start, rt2_w, y, h, label="RT$_2$")

    draw_arrival(ax, rt2_arr, y + h + 0.55, y - 0.05, color=C_OK,
                 label="RT$_2$ arrives")
    # T_rem and T_RT braces (placed ABOVE the chunk)
    brace(ax, rt2_arr, rt2_start, y + h + 0.08,
          "$T_{rem}$", color=C_OK, up=True, fs=9, tick_frac=0.06)
    brace(ax, rt2_start, rt2_start + rt2_w, y + h + 0.08,
          "$T_{RT}$", color=C_OK, up=True, fs=9, tick_frac=0.06)

    # Predicate tag (below timeline) -- placed with a leader line
    pred_ok_x = (rt2_arr + rt2_start + rt2_w) / 2
    ax.plot([pred_ok_x, pred_ok_x], [y - 0.02, y - 0.22],
            color=C_OK, lw=0.9, ls=":")
    ax.text(pred_ok_x, y - 0.24,
            r"$T_{rem}\!+\!T_{queue}\!+\!T_{RT}\,\leq\,\mathrm{SLO}$"
            "\n$\\checkmark$  chunk allowed to finish",
            ha="center", va="top", fontsize=9, color=C_OK)

    # ---- Scenario II: predicate FAILS -> PREEMPTION --------------------
    be3_x = rt2_start + rt2_w + 0.10
    be3_w_full = 2.35          # would-be full chunk
    rt3_arr = be3_x + 0.45     # RT3 arrives very early inside this chunk
    rt3_w = 0.55

    # Executed part (up to rt3_arr)
    draw_be(ax, be3_x, rt3_arr - be3_x, y, h, p_frac=0.5)
    # Wasted (preempted) tail: draw faded + hatched, shifted below axis
    draw_be(ax, rt3_arr, be3_w_full - (rt3_arr - be3_x),
            y, h, p_frac=0.5, alpha=0.35, hatched=True,
            label="preempted")

    # Because of preemption, RT3 is served right away, OVER the hatched
    # region. To avoid visual clash, nudge RT3 block slightly to the
    # right on its own row (same y, but clearly after preemption cut).
    # We keep the original 'paper' semantics: preempted work is wasted
    # and RT3 starts at its arrival time. Visually we use transparency on
    # the wasted tail so RT3 can sit on top cleanly.
    draw_rt(ax, rt3_arr, rt3_w, y, h, label="RT$_3$")

    draw_arrival(ax, rt3_arr, y + h + 0.55, y - 0.05, color=C_BAD,
                 label="RT$_3$ arrives")

    pred_bad_x = rt3_arr + (be3_w_full - (rt3_arr - be3_x)) / 2
    ax.plot([pred_bad_x, pred_bad_x], [y - 0.02, y - 0.22],
            color=C_BAD, lw=0.9, ls=":")
    ax.text(pred_bad_x, y - 0.24,
            r"$T_{rem}\!+\!T_{queue}\!+\!T_{RT}\,>\,\mathrm{SLO}$"
            "\n$\\times$  preempt BE chunk",
            ha="center", va="top", fontsize=9, color=C_BAD)

    # Tail: saturate again after RT3
    tail_x = rt3_arr + rt3_w + 0.10
    tail_w = max(0.0, t_end - 0.2 - tail_x)
    if tail_w > 0.4:
        draw_be(ax, tail_x, tail_w, y, h, p_frac=0.5)

    # Top-level "saturated" annotation bracket
    brace(ax, 0.0, t_end - 0.3, y + h + 0.95,
          "GPU kept saturated: BE chunks fill every RT gap",
          color=C_OK, up=True, fs=10, tick_frac=0.08)

    # Utilization badge
    ax.text(t_end + 1.15, y + h / 2,
            "BE tput  $\\uparrow$",
            ha="left", va="center", fontsize=10.5, color=C_OK,
            fontweight="bold")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(out_dir="debug_info"):
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(12.6, 6.0),
        gridspec_kw=dict(height_ratios=[1, 1.45], hspace=0.60),
    )

    panel_a(ax_a, t_end=9.2)
    ax_a.set_xlim(-0.8, 10.9)
    ax_a.set_ylim(0.15, 1.85)

    panel_b(ax_b, t_end=9.2)
    ax_b.set_xlim(-0.8, 10.9)
    ax_b.set_ylim(-0.85, 2.35)

    for ax in (ax_a, ax_b):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Unified top legend
    handles = [
        mpatches.Patch(facecolor=C_RT,   edgecolor="#1F4E79",
                       label="RT-Prefill"),
        mpatches.Patch(facecolor=C_BE_P, edgecolor="#0B3D66",
                       label="BE-Prefill ($c_p$)"),
        mpatches.Patch(facecolor=C_BE_D, edgecolor="#1B5E20",
                       label="BE-Decode ($c_d$)"),
        mpatches.Patch(facecolor=C_IDLE, edgecolor="#9A9A9A",
                       label="Idle"),
        mpatches.Patch(facecolor="white", edgecolor="#7a0000",
                       hatch="////", label="Preempted (wasted)"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02),
               ncol=5, frameon=False, fontsize=10)

    fig.suptitle(
        "Saturo: saturation-aware BE chunking with SLO-driven preemption",
        y=1.07, fontsize=13, fontweight="bold")

    out_png = os.path.join(out_dir, "saturo_mechanism.png")
    out_pdf = os.path.join(out_dir, "saturo_mechanism.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")
    print(f"[saved] {out_pdf}")


if __name__ == "__main__":
    build_figure()
