"""Trace loading and per-instance RPS scaling for the P-pool simulator."""

import csv
import math
from datetime import datetime
from typing import Callable, Iterable, List, Optional

from .request import RTRequest
from saturo.history import load_trace_rows


_RUNTIME_COLUMNS = (
    "forward_time",
    "runtime",
    "runtime_s",
    "execution_time",
    "execution_time_s",
    "prefill_time",
    "prefill_time_s",
    "ttft",
    "ttft_s",
    "latency",
    "latency_s",
)

_RUNTIME_MS_COLUMNS = (
    "forward_time_ms",
    "runtime_ms",
    "execution_time_ms",
    "prefill_time_ms",
    "ttft_ms",
    "latency_ms",
)

_PROMPT_COLUMNS = ("prompt_len", "input_len", "ContextTokens", "context_tokens")
_OUTPUT_COLUMNS = ("output_len", "GeneratedTokens", "generated_tokens")


def load_rt_trace(
    trace_path: str,
    num_req: int,
    target_rps: float,
    default_runtime_s: Optional[float] = None,
    runtime_predictor_ms: Optional[Callable[[int], float]] = None,
    rows: Optional[List[dict]] = None,
) -> List[RTRequest]:
    if rows is None:
        rows = load_trace_rows(trace_path, window_s=600, completed_only=False)
    if not rows:
        raise ValueError(f"empty trace: {trace_path}")

    raw_arrivals = _parse_arrival_times(rows)
    ordered = sorted(zip(raw_arrivals, rows), key=lambda pair: pair[0])
    if len(ordered) < num_req:
        raise ValueError(
            f"RT trace has {len(ordered)} requests, but num_req={num_req}")
    selected = ordered[:num_req]
    rows = [row for _, row in selected]
    from saturo.tools.trace_scaler import scale_to_target_rps_fixed_num
    arrivals = scale_to_target_rps_fixed_num(
        [arrival for arrival, _ in selected],
        num_req=num_req,
        target_rps=target_rps,
    )
    requests: List[RTRequest] = []
    for idx, (row, arrival_s) in enumerate(zip(rows, arrivals)):
        prompt_len = _parse_int(row, _PROMPT_COLUMNS, 0)
        runtime_s = _parse_runtime_s(
            row,
            default_runtime_s,
            prompt_len=prompt_len,
            runtime_predictor_ms=runtime_predictor_ms,
        )
        requests.append(RTRequest(
            request_id=idx,
            arrival_time_s=arrival_s,
            runtime_s=runtime_s,
            prompt_len=prompt_len,
            output_len=max(1, _parse_int(row, _OUTPUT_COLUMNS, 1)),
        ))
    return requests


def load_be_trace(
    trace_path: str,
    input_drop_frac: float = 0.0,
    input_max_percentile: Optional[float] = None,
    output_keep_frac: Optional[float] = None,
    output_max_len: Optional[int] = None,
    max_model_len: Optional[int] = None,
) -> tuple[int, int, int]:
    """Read BE workload CSV and return (avg_in, avg_out, max_in).

    Expected CSV columns: prompt_len, output_len
    (request_id and arrival_offset_s are ignored.)

    Optional request filtering (applied in this order) shifts the BE length
    distribution, which moves the P/D throughput balance chunk
    ``C* = B * avg_in / avg_out``:

    - ``input_drop_frac`` in (0, 1): drop the shortest-input fraction, so the
      average prompt gets longer.
    - ``output_max_len``: keep only requests whose ``output_len`` is at most
      this absolute threshold.
    - ``output_keep_frac`` in (0, 1]: keep only the shortest-output fraction.
      Ignored when ``output_max_len`` is given.

    Because ``avg_in >> avg_out`` here, ``C*`` is dominated by ``avg_out``, so
    lowering the average output length (output filters) is the effective knob
    for raising the optimal chunk; ``input_drop_frac`` alone barely moves it.
    """
    rows = _read_csv(trace_path)
    if not rows:
        raise ValueError(f"empty BE trace: {trace_path}")

    prompts = [_parse_int(r, ("prompt_len",), 0) for r in rows]
    outputs = [_parse_int(r, ("output_len",), 1) for r in rows]
    pairs = list(zip(prompts, outputs))

    # 1. Remove the long-input tail using the same nearest-rank percentile as
    # the real P-pool benchmark.
    if input_max_percentile is not None:
        q = float(input_max_percentile)
        if not 0.0 < q <= 1.0:
            raise ValueError("input_max_percentile must be in (0, 1]")
        ordered = sorted(prompt for prompt, _ in pairs)
        rank = max(0, min(len(ordered) - 1,
                          int(math.ceil(q * len(ordered))) - 1))
        cutoff = ordered[rank]
        pairs = [p for p in pairs if p[0] <= cutoff]

    # 2. Drop the shortest-input fraction to lengthen the average prompt.
    if input_drop_frac:
        if not 0.0 < input_drop_frac < 1.0:
            raise ValueError("input_drop_frac must be in (0, 1)")
        pairs.sort(key=lambda p: p[0])
        pairs = pairs[int(input_drop_frac * len(pairs)):]

    # 3. Keep only short-output requests to lower the average output length.
    if output_max_len is not None:
        pairs = [p for p in pairs if p[1] <= output_max_len]
    elif output_keep_frac is not None:
        if not 0.0 < output_keep_frac <= 1.0:
            raise ValueError("output_keep_frac must be in (0, 1]")
        pairs.sort(key=lambda p: p[1])
        pairs = pairs[:max(1, int(output_keep_frac * len(pairs)))]

    if not pairs:
        raise ValueError(f"BE trace filter removed all rows: {trace_path}")

    # Match the real vLLM runner: prompts that cannot fit are discarded and
    # outputs are clamped to the remaining model context. Without this, a
    # trace containing prompts longer than max_model_len inflates avg_in and
    # can under-report request throughput by several times.
    if max_model_len is not None:
        limit = int(max_model_len)
        pairs = [(prompt, min(output, limit - prompt))
                 for prompt, output in pairs if prompt < limit]
        if not pairs:
            raise ValueError(
                f"no BE request fits max_model_len={max_model_len}")

    avg_in = int(round(sum(p[0] for p in pairs) / len(pairs)))
    avg_out = int(round(sum(p[1] for p in pairs) / len(pairs)))
    max_in = max(p[0] for p in pairs)
    return avg_in, avg_out, max_in


def compute_pure_rt_slacks(
    requests: Iterable[RTRequest],
    ttft_slo_s: float,
) -> List[float]:
    slacks: List[float] = []
    current_time_s = 0.0
    for req in sorted(requests, key=lambda r: r.arrival_time_s):
        start_s = max(current_time_s, req.arrival_time_s)
        queue_s = start_s - req.arrival_time_s
        slacks.append(ttft_slo_s - queue_s - req.runtime_s)
        current_time_s = start_s + req.runtime_s
    return slacks


def _read_csv(path: str) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_arrival_times(
    rows: List[dict],
) -> List[float]:
    # Extract raw arrival offsets from whichever column is present
    if "arrival_time" in rows[0]:
        raw = [float(row["arrival_time"]) for row in rows]
    elif "arrival_time_s" in rows[0]:
        raw = [float(row["arrival_time_s"]) for row in rows]
    elif "arrival_offset_s" in rows[0]:
        raw = [float(row["arrival_offset_s"]) for row in rows]
    elif "TIMESTAMP" in rows[0]:
        times = [_parse_timestamp(row["TIMESTAMP"].strip()) for row in rows]
        base = times[0]
        raw = [(ts - base).total_seconds() for ts in times]
    else:
        raise ValueError(
            "trace needs an original TIMESTAMP, arrival_time, "
            "arrival_time_s, or arrival_offset_s column")

    return raw


def _parse_runtime_s(
    row: dict,
    default_runtime_s: Optional[float],
    *,
    prompt_len: int,
    runtime_predictor_ms: Optional[Callable[[int], float]],
) -> float:
    for name in _RUNTIME_COLUMNS:
        if name in row and row[name] not in ("", None):
            return max(float(row[name]), 0.0)
    for name in _RUNTIME_MS_COLUMNS:
        if name in row and row[name] not in ("", None):
            return max(float(row[name]) / 1000.0, 0.0)
    if default_runtime_s is not None:
        return max(float(default_runtime_s), 0.0)
    if runtime_predictor_ms is not None:
        return max(float(runtime_predictor_ms(prompt_len)) / 1000.0, 0.0)
    raise ValueError(
        "trace needs one RT runtime column: "
        + ", ".join(_RUNTIME_COLUMNS + _RUNTIME_MS_COLUMNS))


def _parse_int(row: dict, names: Iterable[str], default: int) -> int:
    for name in names:
        if name in row and row[name] not in ("", None):
            return int(float(row[name]))
    return default


def _parse_timestamp(ts_str: str) -> datetime:
    if "." in ts_str:
        date_part, frac = ts_str.rsplit(".", 1)
        frac = frac[:6]
        return datetime.strptime(f"{date_part}.{frac}", "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
