"""Load RT request traces and generate BE workloads.

Supports RPS scaling: if the trace has RPS=5.53 but the target environment
can only serve RPS=1, inter-arrival times are scaled by 5.53x.
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import tiktoken

from .request import Request, RequestType
from saturo.history import load_trace_rows
from saturo.runtime.workload import _token_ids


def compute_trace_rps(
    trace_path: str,
    duration_limit_s: Optional[float] = None,
) -> float:
    """Compute the original RPS of a supported RT trace."""
    if _is_parquet(trace_path):
        records = _collect_wildchat_records(Path(trace_path))
        return _rps_from_records(records, duration_limit_s)

    rows = load_trace_rows(trace_path)
    if _is_workload_csv_rows(rows):
        arrival_field = (
            "arrival_offset_s"
            if "arrival_offset_s" in rows[0]
            else "arrival_time_s"
        )
        offsets = sorted(float(row.get(arrival_field) or 0.0)
                         for row in rows)
        if duration_limit_s is not None:
            count = sum(offset <= duration_limit_s for offset in offsets)
            return count / duration_limit_s if duration_limit_s > 0 else 1.0
        if len(offsets) < 2:
            return 1.0
        span = offsets[-1] - offsets[0]
        return len(offsets) / span if span > 0 else 1.0

    timestamps = [
        _parse_timestamp(row["TIMESTAMP"].strip()) for row in rows
    ]
    if len(timestamps) < 2:
        return 1.0

    base = timestamps[0]
    total_s = (timestamps[-1] - base).total_seconds()
    if duration_limit_s is not None:
        count = sum(
            1 for timestamp in timestamps
            if (timestamp - base).total_seconds() <= duration_limit_s
        )
        return count / duration_limit_s if duration_limit_s > 0 else 1.0
    return len(timestamps) / total_s if total_s > 0 else 1.0


def load_rt_trace(
    trace_path: str,
    duration_s: float = 300.0,
    target_rps: Optional[float] = None,
    time_scale: Optional[float] = None,
    max_prompt_len: Optional[int] = None,
    max_output_len: Optional[int] = None,
) -> List[Request]:
    """Load RT requests from a Saturo workload, Azure CSV, or WildChat.

    Saturo workload CSV: request_id,prompt_len,output_len,arrival_offset_s.
    Azure CSV: TIMESTAMP,ContextTokens,GeneratedTokens.

    WildChat parquet format: one request is built from each assistant turn.
    The prompt is the preceding conversation and the output is the assistant
    message. Arrival time is the latest preceding user-message timestamp when
    present, otherwise the assistant/conversation timestamp.

    RPS scaling: If target_rps is specified, inter-arrival times are scaled
    so that the effective RPS matches the target. For example, if the trace
    has RPS=5.53 and target_rps=1.0, inter-arrival times are multiplied by
    5.53, spreading the requests out.

    If time_scale is specified directly, it overrides the auto-computed scale.

    Args:
        trace_path: supported RT workload or trace path
        duration_s: how many seconds of (scaled) trace to load
        target_rps: desired RPS after scaling (auto-computes scale factor)
        time_scale: direct scale factor for inter-arrival times (overrides target_rps)
    Returns:
        List of RT Request objects with scaled arrival times
    """
    if _is_parquet(trace_path):
        return _load_wildchat_parquet_trace(
            Path(trace_path),
            duration_s=duration_s,
            target_rps=target_rps,
            time_scale=time_scale,
            max_prompt_len=max_prompt_len,
            max_output_len=max_output_len,
        )

    rows = load_trace_rows(trace_path)

    if not rows:
        raise ValueError(f"Empty trace file: {trace_path}")

    if _is_workload_csv_rows(rows):
        return _load_workload_csv_trace(
            rows,
            trace_path=trace_path,
            duration_s=duration_s,
            target_rps=target_rps,
            time_scale=time_scale,
            max_prompt_len=max_prompt_len,
            max_output_len=max_output_len,
        )

    # Extract raw arrival offsets from timestamps
    base_time = _parse_timestamp(rows[0]["TIMESTAMP"].strip())
    raw_offsets = [
        (_parse_timestamp(row["TIMESTAMP"].strip()) - base_time).total_seconds()
        for row in rows
    ]

    # Scale using shared trace_scaler
    from saturo.tools.trace_scaler import scale_to_target_rps
    scaled = scale_to_target_rps(raw_offsets, target_rps, time_scale)

    requests = []
    for idx, (row, arrival_s) in enumerate(zip(rows, scaled)):
        if arrival_s > duration_s:
            break

        prompt_len, output_len = _clamp_request_lens(
            int(row["ContextTokens"]),
            int(row["GeneratedTokens"]),
            max_prompt_len=max_prompt_len,
            max_output_len=max_output_len,
        )

        req = Request(
            request_id=idx,
            request_type=RequestType.RT,
            prompt_len=prompt_len,
            target_output_len=max(output_len, 1),
            arrival_time_s=arrival_s,
        )
        requests.append(req)

    print(f"[trace_loader] Loaded {len(requests)} RT requests over {duration_s:.0f}s "
          f"(effective RPS={len(requests)/duration_s:.2f})")
    return requests


def _is_workload_csv_rows(rows) -> bool:
    if not rows:
        return False
    fields = set(rows[0])
    return (
        {"prompt_len", "output_len"}.issubset(fields)
        and ("arrival_offset_s" in fields or "arrival_time_s" in fields)
    )


def _load_workload_csv_trace(
    rows,
    *,
    trace_path: str,
    duration_s: float,
    target_rps: Optional[float],
    time_scale: Optional[float],
    max_prompt_len: Optional[int],
    max_output_len: Optional[int],
) -> List[Request]:
    """Load the CSV schema shared with global_scheduler."""
    arrival_field = (
        "arrival_offset_s"
        if "arrival_offset_s" in rows[0]
        else "arrival_time_s"
    )
    ordered = sorted(
        rows,
        key=lambda row: float(row.get(arrival_field) or 0.0),
    )

    raw_offsets = [
        float(row.get(arrival_field) or 0.0) for row in ordered
    ]
    # Delegate scaling to the shared trace_scaler
    from saturo.tools.trace_scaler import scale_to_target_rps
    offsets = scale_to_target_rps(raw_offsets, target_rps, time_scale)

    requests = []
    for idx, (row, arrival_s) in enumerate(zip(ordered, offsets)):
        if arrival_s > duration_s:
            break
        try:
            prompt_len, output_len = _clamp_request_lens(
                int(float(row["prompt_len"])),
                int(float(row["output_len"])),
                max_prompt_len=max_prompt_len,
                max_output_len=max_output_len,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid workload row {idx + 2} in {trace_path}: {exc}"
            ) from exc
        p_ids = o_ids = None
        if "prompt_token_ids" in row:
            original_prompt = int(float(row["prompt_len"]))
            original_output = int(float(row["output_len"]))
            if original_prompt <= 0 or original_output <= 0:
                raise ValueError(f"Nonpositive content lengths in {trace_path}")
            location = f"{trace_path}: row {idx + 2}"
            p_ids = tuple(_token_ids(row["prompt_token_ids"], original_prompt,
                                    "prompt_token_ids", location))
            if "output_token_ids" in row:
                o_ids = tuple(_token_ids(row["output_token_ids"], original_output,
                                        "output_token_ids", location))
            if (prompt_len, output_len) != (original_prompt, original_output):
                raise ValueError(f"{location}: content trace would be truncated by length caps; increase the caps")
        requests.append(Request(
            request_id=row.get("request_id") or idx,
            request_type=RequestType.RT,
            prompt_len=prompt_len,
            target_output_len=output_len,
            arrival_time_s=max(float(arrival_s), 0.0),
            prompt_token_ids=p_ids,
            output_token_ids=o_ids,
        ))

    print(
        f"[trace_loader] Loaded {len(requests)} workload RT requests over "
        f"{duration_s:.0f}s (effective RPS={len(requests)/duration_s:.2f})")
    return requests


def _load_wildchat_parquet_trace(
    trace_path: Path,
    duration_s: float,
    target_rps: Optional[float],
    time_scale: Optional[float],
    max_prompt_len: Optional[int],
    max_output_len: Optional[int],
) -> List[Request]:
    records = _collect_wildchat_records_with_ids(trace_path)
    if not records:
        raise ValueError(f"No usable WildChat requests found in {trace_path}")

    # Extract raw arrival offsets and delegate scaling to trace_scaler
    base_time = records[0][0]
    raw_offsets = [
        (timestamp - base_time).total_seconds()
        for timestamp, _, _ in records
    ]
    from saturo.tools.trace_scaler import scale_to_target_rps
    scaled = scale_to_target_rps(raw_offsets, target_rps, time_scale)

    requests: List[Request] = []
    for idx, ((_, p_ids, o_ids), arrival_s) in enumerate(zip(records, scaled)):
        if arrival_s > duration_s:
            break
        prompt_len, output_len = _clamp_request_lens(
            len(p_ids),
            len(o_ids),
            max_prompt_len=max_prompt_len,
            max_output_len=max_output_len,
        )
        p_ids_clamped = p_ids[:prompt_len]
        o_ids_clamped = o_ids[:output_len]
        requests.append(Request(
            request_id=idx,
            request_type=RequestType.RT,
            prompt_len=max(int(prompt_len), 1),
            target_output_len=max(int(output_len), 1),
            arrival_time_s=max(float(arrival_s), 0.0),
            prompt_token_ids=p_ids_clamped,
            output_token_ids=o_ids_clamped,
        ))

    print(f"[trace_loader] Loaded {len(requests)} WildChat RT requests over "
          f"{duration_s:.0f}s (effective RPS={len(requests)/duration_s:.2f})")
    return requests
def _clamp_request_lens(
    prompt_len: int,
    output_len: int,
    *,
    max_prompt_len: Optional[int],
    max_output_len: Optional[int],
) -> Tuple[int, int]:
    prompt_len = max(int(prompt_len), 1)
    output_len = max(int(output_len), 1)
    if max_prompt_len is not None and max_prompt_len > 0:
        prompt_len = min(prompt_len, int(max_prompt_len))
    if max_output_len is not None and max_output_len > 0:
        output_len = min(output_len, int(max_output_len))
    return prompt_len, output_len


@lru_cache(maxsize=4)
def _collect_wildchat_records(trace_path: Path) -> List[Tuple[datetime, int, int]]:
    """Return (timestamp, prompt_len, output_len) tuples — used for RPS calc."""
    return [(ts, len(p), len(o)) for ts, p, o in _collect_wildchat_records_with_ids(trace_path)]


@lru_cache(maxsize=4)
def _collect_wildchat_records_with_ids(
    trace_path: Path,
) -> List[Tuple[datetime, Tuple[int, ...], Tuple[int, ...]]]:
    """Return (timestamp, prompt_token_ids, output_token_ids) tuples.

    Token IDs are the actual tiktoken cl100k_base encodings, enabling
    TF-IDF-based remaining-length prediction in the simulator.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    parquet_file = pq.ParquetFile(trace_path)
    columns = ["conversation", "timestamp", "toxic", "redacted"]
    records: List[Tuple[datetime, Tuple[int, ...], Tuple[int, ...]]] = []

    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=columns)
        for row in table.to_pylist():
            if bool(row.get("toxic", False)) or bool(row.get("redacted", False)):
                continue
            conversation = row.get("conversation")
            if conversation is None:
                continue
            conversation = list(conversation)
            row_ts = _parse_timestamp(row.get("timestamp"))
            for idx, message in enumerate(conversation):
                if _message_role(message) != "assistant":
                    continue
                output = _message_content(message).strip()
                if not output:
                    continue
                prompt = _serialize_prompt(conversation[:idx])
                if not prompt:
                    continue

                timestamp = (
                    _latest_user_timestamp(conversation[:idx])
                    or _parse_timestamp(_message_timestamp(message))
                    or row_ts
                )
                if timestamp is None:
                    continue

                p_ids = tuple(encoder.encode(prompt))
                o_ids = tuple(encoder.encode(output))
                if not o_ids:
                    continue
                records.append((timestamp, p_ids, o_ids))

    records.sort(key=lambda item: item[0])
    return records


def _rps_from_records(
    records: List[Tuple[datetime, int, int]],
    duration_limit_s: Optional[float] = None,
) -> float:
    if len(records) < 2:
        return 1.0
    base = records[0][0]
    if duration_limit_s is not None:
        count = sum(
            1 for ts, _, _ in records
            if (ts - base).total_seconds() <= duration_limit_s
        )
        return count / duration_limit_s if duration_limit_s > 0 else 1.0
    total_s = (records[-1][0] - base).total_seconds()
    return len(records) / total_s if total_s > 0 else 1.0


def _message_role(message) -> str:
    if not isinstance(message, dict):
        return ""
    role = message.get("role")
    return role if isinstance(role, str) else ""


def _message_content(message) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _message_timestamp(message):
    if not isinstance(message, dict):
        return None
    return message.get("timestamp")


def _serialize_prompt(messages: Iterable[dict]) -> str:
    parts = []
    for message in messages:
        content = _message_content(message).strip()
        if content:
            parts.append(f"{_message_role(message) or 'unknown'}: {content}")
    return "\n".join(parts)


def _latest_user_timestamp(messages: Iterable[dict]):
    for message in reversed(list(messages)):
        if _message_role(message) == "user":
            timestamp = _parse_timestamp(_message_timestamp(message))
            if timestamp is not None:
                return timestamp
    return None


def generate_be_workload(
    num_requests: int = 500,
    input_len_range: Tuple[int, int] = (512, 8192),
    output_len_range: Tuple[int, int] = (256, 4096),
    seed: int = 42,
) -> List[Request]:
    """Generate synthetic BE requests with random input/output lengths.

    Uses log-uniform distribution for more realistic length distribution
    (many short requests, fewer long ones).
    """
    rng = np.random.RandomState(seed)
    requests = []
    for idx in range(num_requests):
        # Log-uniform distribution for input length
        log_min_in = np.log(input_len_range[0])
        log_max_in = np.log(input_len_range[1])
        prompt_len = int(np.exp(rng.uniform(log_min_in, log_max_in)))

        # Log-uniform for output length
        log_min_out = np.log(output_len_range[0])
        log_max_out = np.log(output_len_range[1])
        output_len = int(np.exp(rng.uniform(log_min_out, log_max_out)))

        req = Request(
            request_id=100000 + idx,
            request_type=RequestType.BE,
            prompt_len=prompt_len,
            target_output_len=output_len,
            arrival_time_s=0.0,
        )
        requests.append(req)

    avg_in = np.mean([r.prompt_len for r in requests])
    avg_out = np.mean([r.target_output_len for r in requests])
    print(f"[trace_loader] Generated {num_requests} BE requests "
          f"(avg_input={avg_in:.0f}, avg_output={avg_out:.0f})")
    return requests


def _parse_timestamp(ts_str) -> Optional[datetime]:
    """Parse timestamp string, handling variable fractional second precision."""
    if ts_str is None:
        return None
    if isinstance(ts_str, datetime):
        return ts_str.replace(tzinfo=None)
    ts_str = str(ts_str)
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(
            tzinfo=None)
    except ValueError:
        pass
    if "." in ts_str:
        date_part, frac = ts_str.rsplit(".", 1)
        frac = frac.replace("+00:00", "").replace("Z", "")
        frac = frac[:6]  # Python %f supports up to 6 digits
        ts_str = date_part + "." + frac
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")


def _is_parquet(trace_path: str) -> bool:
    return str(trace_path).lower().endswith(".parquet")
