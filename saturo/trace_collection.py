"""Optional Redis storage for RT arrival and token-length traces.

Set SATURO_REDIS_URL to enable runtime collection. Export a collected window
with ``python -m saturo.trace_collection --key KEY --output trace.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RedisTraceStore:
    """An arrival-time index and JSON records, isolated by trace key.

    Both keys expire after ``ttl_s`` of inactivity. Set a separate key per
    model/deployment when sharing Redis. No prompts or generated text are kept.
    """

    def __init__(self, url: str, key: str, ttl_s: int = 86400):
        if not key or ttl_s <= 0:
            raise ValueError("trace key must be nonempty and TTL positive")
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "Redis trace collection requires: "
                "pip install -r saturo/requirements-traces.txt") from exc
        self.client = redis.Redis.from_url(
            url, decode_responses=True, socket_connect_timeout=0.2,
            socket_timeout=0.2, retry_on_timeout=False)
        self.key = key
        self.ttl_s = ttl_s
        self.index_key = f"{key}:arrivals"
        self.records_key = f"{key}:records"

    def write(self, record: dict[str, Any]) -> None:
        # A completion replaces the arrival record; retries never add a new
        # request. run_id keeps repeated experiment request IDs independent.
        member = f"{record['run_id']}:{record['request_id']}"
        with self.client.pipeline() as pipe:
            pipe.hset(self.records_key, member, json.dumps(record))
            pipe.zadd(self.index_key, {member: record["arrival_time_s"]})
            pipe.expire(self.records_key, self.ttl_s)
            pipe.expire(self.index_key, self.ttl_s)
            pipe.execute()

    def read_window(self, start_s: float, end_s: float) -> list[dict]:
        """Read an inclusive Unix-time interval in arrival order."""
        if start_s > end_s:
            raise ValueError("window start must not exceed end")
        members = self.client.zrangebyscore(self.index_key, start_s, end_s)
        if not members:
            return []
        values = self.client.hmget(self.records_key, members)
        return [json.loads(value) for value in values if value is not None]

    def export_csv(self, path: Path, start_s: float, end_s: float) -> int:
        """Export completed requests as the runtime/profiler workload CSV.

        Incomplete/dropped requests retain arrival statistics in Redis but do
        not supply a fictitious output length to history replay.
        """
        rows = self.read_window(start_s, end_s)
        origin = rows[0]["arrival_time_s"] if rows else start_s
        fields = ["request_id", "prompt_len", "output_len", "arrival_offset_s"]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                finished_s = row.get("finished_time_s")
                if (row["output_len"] is None
                        or (finished_s is not None and finished_s > end_s)):
                    continue
                writer.writerow({
                    "request_id": f"{row['run_id']}:{row['request_id']}",
                    "prompt_len": row["prompt_len"],
                    "output_len": row["output_len"],
                    "arrival_offset_s": row["arrival_time_s"] - origin,
                })
                count += 1
        return count


class TraceRecorder:
    """Keep Redis I/O off the scheduler thread with one bounded FIFO writer."""

    def __init__(self, store: RedisTraceStore, model: str,
                 started_at_s: float):
        self.store = store
        self.model = model
        self.started_at_s = started_at_s
        self.run_id = uuid.uuid4().hex
        self.dropped_records = 0
        self.failed_records = 0
        self._queue: queue.Queue = queue.Queue(maxsize=4096)
        self._closing = threading.Event()
        self._close_deadline_s = float("inf")
        self._thread = threading.Thread(
            target=self._write_loop, name="saturo-trace-writer", daemon=True)
        self._thread.start()

    def record(self, request) -> None:
        # Snapshot on the scheduler thread: requests mutate during decoding.
        completed = request.observed_tokens >= request.output_len
        record = {
            "run_id": self.run_id,
            "model": self.model,
            "request_id": request.request_id,
            # The workload source supplies arrival offsets; dispatch delay is
            # not part of the arrival trace.
            "arrival_time_s": self.started_at_s + request.arrival_offset_s,
            "prompt_len": request.prompt_len,
            "requested_output_len": request.output_len,
            "output_len": request.observed_tokens if completed else None,
            "finished_time_s": (
                self.started_at_s + request.finished_s
                if completed and request.finished_s is not None else None),
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped_records += 1
            if self.dropped_records == 1:
                logger.warning("RT trace queue full; dropping trace records")

    def _write_loop(self) -> None:
        while time.monotonic() < self._close_deadline_s:
            try:
                record = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._closing.is_set():
                    break
                continue
            try:
                self.store.write(record)
            except Exception as exc:
                self.failed_records += 1
                if self.failed_records == 1:
                    # Exception text can contain connection credentials.
                    logger.warning("Redis RT trace write failed (%s)",
                                   type(exc).__name__)
            finally:
                self._queue.task_done()
        self.dropped_records += self._queue.qsize()
        self.store.client.close()

    def close(self) -> None:
        """Allow up to five seconds to drain, even when Redis is unavailable."""
        if not self._closing.is_set():
            self._close_deadline_s = time.monotonic() + 5.0
            self._closing.set()
        self._thread.join(timeout=6.0)
        if self._thread.is_alive():
            logger.warning("RT trace writer did not stop before shutdown")
        if self.dropped_records or self.failed_records:
            logger.warning(
                "RT trace collection lost records: queue=%d, redis=%d",
                self.dropped_records, self.failed_records)


def store_from_env() -> Optional[RedisTraceStore]:
    """Disabled by default; importing Saturo does not require redis-py."""
    url = os.environ.get("SATURO_REDIS_URL", "")
    if not url:
        return None
    key = (os.environ.get("SATURO_TRACE_KEY")
           or os.environ.get("SATURO_HISTORY_KEY")
           or f"saturo:traces:{uuid.uuid4().hex}")
    return RedisTraceStore(
        url, key, ttl_s=int(os.environ.get("SATURO_TRACE_TTL_S", "86400")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--end-time-s", type=float, default=None,
                        help="Unix timestamp; defaults to the current time")
    args = parser.parse_args()
    if args.window_s <= 0:
        parser.error("--window-s must be positive")
    url = os.environ.get("SATURO_REDIS_URL")
    if not url:
        parser.error("set SATURO_REDIS_URL to the Redis connection URL")
    store = RedisTraceStore(url, args.key)
    end_s = time.time() if args.end_time_s is None else args.end_time_s
    try:
        count = store.export_csv(args.output, end_s - args.window_s, end_s)
    finally:
        store.client.close()
    print(f"Exported {count} completed RT requests to {args.output}")


if __name__ == "__main__":
    main()
