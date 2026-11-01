"""Shared RPS scaling logic for trace/arrival-time manipulation.

When a dataset records request arrival times (e.g. an Azure production trace
with RPS=60), but the experiment needs a different target RPS (e.g. RPS=10),
the inter-arrival gaps must be **dilated** by a scale factor so that the
relative arrival pattern (burstiness, gaps) is preserved while the overall
rate matches the target.

Core formula
------------
    trace_rps  =  num_requests / time_span
    scale      =  trace_rps / target_rps
    new_arrival = (original_arrival - base) * scale

The duration-based scaler preserves the complete trace's relative arrival
gaps. Fixed-request helpers remain available for experiments that explicitly
need a request budget, but Saturo's decode simulator does not use them.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def compute_trace_rps(arrival_offsets: Sequence[float]) -> float:
    """Compute the original RPS of a trace from its arrival offsets.

    Args:
        arrival_offsets: arrival times in seconds (relative or absolute,
            only differences matter).  Need not be sorted.

    Returns:
        The trace's effective RPS.  Returns 1.0 for degenerate cases
        (empty, single request, zero span).
    """
    if len(arrival_offsets) < 2:
        return 1.0
    ordered = sorted(float(t) for t in arrival_offsets)
    span = ordered[-1] - ordered[0]
    if span <= 0:
        return 1.0
    return len(ordered) / span


def compute_scale_factor(
    trace_rps: float,
    target_rps: Optional[float] = None,
    time_scale: Optional[float] = None,
) -> float:
    """Compute a uniform dilation factor for all inter-arrival gaps."""
    if time_scale is not None:
        return float(time_scale)
    if target_rps is not None and target_rps > 0:
        return trace_rps / target_rps
    return 1.0


def scale_arrivals(
    arrival_offsets: Sequence[float],
    target_rps: Optional[float] = None,
    time_scale: Optional[float] = None,
) -> List[float]:
    """Uniformly scale the complete trace while preserving burstiness."""
    if not arrival_offsets:
        return []
    ordered = sorted(float(t) for t in arrival_offsets)
    base = ordered[0]
    if len(ordered) < 2:
        return [0.0]

    trace_rps = compute_trace_rps(ordered)
    scale = compute_scale_factor(trace_rps, target_rps, time_scale)

    if target_rps is not None and target_rps > 0:
        print(
            f"[trace_scaler] Trace RPS={trace_rps:.2f}, "
            f"target RPS={target_rps:.2f}, scale factor={scale:.4f}x"
        )
    elif time_scale is not None:
        print(
            f"[trace_scaler] Trace RPS={trace_rps:.2f}, "
            f"time_scale={time_scale:.4f}x"
        )

    return [(t - base) * scale for t in ordered]


def scale_to_target_rps(
    arrival_offsets: Sequence[float],
    target_rps: Optional[float] = None,
    time_scale: Optional[float] = None,
    duration_s: Optional[float] = None,
) -> List[float]:
    """Scale the complete trace and optionally truncate by time."""
    scaled = scale_arrivals(arrival_offsets, target_rps, time_scale)
    if duration_s is not None:
        scaled = [t for t in scaled if t <= duration_s]
    return scaled


def fixed_num_window_s(num_req: int, target_rps: float) -> float:
    """Return the sending window implied by a fixed request count and RPS."""
    if num_req <= 0:
        raise ValueError("num_req must be positive")
    if target_rps <= 0:
        raise ValueError("target_rps must be positive")
    return float(num_req) / float(target_rps)


def scale_to_target_rps_fixed_num(
    arrival_offsets: Sequence[float],
    num_req: int,
    target_rps: float,
) -> List[float]:
    """Take exactly ``num_req`` arrivals and scale that prefix to target RPS.

    The first ``num_req`` chronological requests are selected before the
    prefix's native RPS is computed.  Their relative burst shape is preserved
    and the scaled sending window is always ``num_req / target_rps``.
    """
    duration_s = fixed_num_window_s(num_req, target_rps)
    ordered = sorted(float(t) for t in arrival_offsets)
    if len(ordered) < num_req:
        raise ValueError(
            f"RT trace has {len(ordered)} requests, but num_req={num_req}")

    subset = ordered[:num_req]
    base = subset[0]

    if num_req < 2:
        result = [0.0]
        orig_span = 0.0
    else:
        orig_span = subset[-1] - base
        if orig_span <= 0:
            step = duration_s / (num_req - 1)
            result = [i * step for i in range(num_req)]
        else:
            scale = duration_s / orig_span
            result = [min((t - base) * scale, duration_s) for t in subset]

    orig_rps = (num_req / orig_span) if orig_span > 0 else float("inf")
    print(
        f"[trace_scaler] fixed-num: num_req={num_req}, "
        f"target_rps={target_rps:.6g}, "
        f"orig_rps={orig_rps:.2f}, orig_span={orig_span:.2f}s "
        f"-> window={duration_s:.6f}s"
    )
    return result
