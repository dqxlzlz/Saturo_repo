"""Shared historical workload input for planning, simulators and training."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from saturo.trace_collection import RedisTraceStore

REDIS_HISTORY_PREFIX = "redis-history:"


def is_redis_history(source) -> bool:
    return str(source).startswith(REDIS_HISTORY_PREFIX)


def configured_history_source(csv_path: str = "") -> str:
    """Select historical input independently of the live/replay workload CSV."""
    mode = os.environ.get("SATURO_HISTORY_SOURCE", "csv").strip().lower()
    if mode == "csv":
        return csv_path
    if mode != "redis":
        raise ValueError("SATURO_HISTORY_SOURCE must be csv or redis")
    key = (os.environ.get("SATURO_HISTORY_KEY")
           or os.environ.get("SATURO_TRACE_KEY"))
    if not key or not os.environ.get("SATURO_REDIS_URL"):
        raise ValueError("Redis history requires SATURO_REDIS_URL and "
                         "SATURO_HISTORY_KEY (or SATURO_TRACE_KEY)")
    return REDIS_HISTORY_PREFIX + key


@dataclass(frozen=True)
class HistorySnapshot:
    key: str
    model: str
    start_s: float
    end_s: float
    records: list[dict[str, Any]]
    source: str = "redis"
    provenance: dict[str, Any] | None = None

    def workload_rows(self, *, completed_only: bool = True) -> list[dict]:
        origin = (self.records[0]["arrival_time_s"]
                  if self.records else self.start_s)
        return [{
            "request_id": f"{row['run_id']}:{row['request_id']}",
            "prompt_len": row["prompt_len"],
            "output_len": row.get("output_len"),
            "arrival_offset_s": row["arrival_time_s"] - origin,
        } for row in self.records
                if not completed_only or row.get("output_len") is not None]

    def statistics(self) -> dict[str, float]:
        outputs = [r["output_len"] for r in self.records
                   if r.get("output_len") is not None]
        return {
            # Pending arrivals still count toward load; max_tokens is never
            # treated as an observed output length.
            "arrival_rate": (len(self.records) / (self.end_s - self.start_s)
                             if self.end_s > self.start_s else 0.0),
            "mean_prompt_len": (sum(r["prompt_len"] for r in self.records)
                                / max(len(self.records), 1)),
            "mean_output_len": sum(outputs) / max(len(outputs), 1),
            "completed_requests": len(outputs),
            "arrival_requests": len(self.records),
            "observed_until_s": self.end_s,
        }

    def metadata(self) -> dict:
        data = json.dumps({"key": self.key, "model": self.model,
                           "source": self.source,
                           "start_s": self.start_s, "end_s": self.end_s,
                           "window_s": self.end_s - self.start_s,
                           "records": self.records}, sort_keys=True,
                          separators=(",", ":"))
        result = {"source": self.source, "key": self.key, "model": self.model,
                "start_s": self.start_s, "end_s": self.end_s,
                "window_s": self.end_s - self.start_s,
                "fingerprint": hashlib.sha256(data.encode()).hexdigest()[:16]}
        if self.provenance:
            result["provenance"] = dict(self.provenance)
        return result


def read_redis_history(source: str, *, window_s: float = 1800,
                       model: str = "", end_s: float | None = None
                       ) -> HistorySnapshot:
    if window_s <= 0:
        raise ValueError("history window must be positive")
    if not is_redis_history(source):
        raise ValueError("expected redis-history:<key>")
    url = os.environ.get("SATURO_REDIS_URL")
    if not url:
        raise ValueError("Redis history requires SATURO_REDIS_URL")
    key = str(source)[len(REDIS_HISTORY_PREFIX):]
    end_s = time.time() if end_s is None else float(end_s)
    store = RedisTraceStore(url, key)
    try:
        records = store.read_window(end_s - window_s, end_s)
    finally:
        store.client.close()
    selected = []
    for original in records:
        if model and original.get("model") != model:
            continue
        row = dict(original)
        finished_s = row.get("finished_time_s")
        if finished_s is not None and float(finished_s) > end_s:
            # Historical queries must not see completions from their future.
            row["output_len"] = None
            row["finished_time_s"] = None
        selected.append(row)
    return HistorySnapshot(key, model, end_s - window_s, end_s, selected)


def load_trace_rows(source, *, window_s: float = 1800, model: str = "",
                    completed_only: bool = True) -> list[dict]:
    """Keep existing CSV formats; Redis input is explicit and needs no export."""
    if is_redis_history(source):
        return read_redis_history(
            str(source), window_s=window_s, model=model).workload_rows(
                completed_only=completed_only)
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    with open(source, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))
