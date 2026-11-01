"""Versioned, atomic P/D history updates away from the serving loop.

A single daemon computes both controls from one observation cutoff. Failures,
late results and old layouts never partially replace the last usable bundle.
CSV replay supplies observations through the ingress API; its future rows are
never opened here. Redis remains an explicitly configured durable source.
"""

from __future__ import annotations

import copy
import heapq
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from saturo.history import (HistorySnapshot, configured_history_source,
                            is_redis_history, read_redis_history)
from saturo.runtime.workload import DECODE, PREFILL

logger = logging.getLogger(__name__)


def _setting(args, name, env, default):
    value = getattr(args, name, None)
    value = os.environ.get(env, default) if value is None and env else value
    value = default if value is None else value
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class RuntimeHistoryController:
    def __init__(self, args, worker_specs, *, autostart=True, clock=None,
                 monotonic=None, run_id=None):
        self.args = args
        self.source = configured_history_source()
        self.window_s = _setting(args, "saturation_history_window_s",
                                 "SATURO_RUNTIME_HISTORY_WINDOW_S", 600)
        self.refresh_s = _setting(args, "saturation_update_interval_s",
                                  "SATURO_HISTORY_REFRESH_S", 40)
        self.timeout_s = _setting(args, "saturation_update_timeout_s", None, 30)
        self._clock = clock or time.time
        self._monotonic = monotonic or time.monotonic
        self._observation_started_s = self._clock()
        self._retention_s = max(self.window_s, _setting(
            args, "resource_reallocation_history_window_s", None, 1800))
        self._run_id = str(run_id or uuid.uuid4().hex)
        self._records = {}
        self._arrival_heap = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self.worker_specs = copy.deepcopy(worker_specs)
        self._layout_key = self._spec_key(self.worker_specs)
        self._layout_generation = 0
        self._next_version = 0
        self._pending = None
        self._latest = None  # Immutable JSON bytes, never caller-owned dicts.
        self._latest_version = 0
        self._latest_layout_generation = 0
        self._delivered_version = 0
        self._next_due_s = self._monotonic()
        self._requested = True
        self._state = "starting"
        self._last_error_type = None
        self._last_attempt = None
        self._counters = Counter()
        self._thread = threading.Thread(
            target=self._loop, name="saturo-history-refresh", daemon=True)
        if autostart:
            self._thread.start()

    @staticmethod
    def _spec_key(specs):
        # Worker identity matters even when a replacement has the same shape.
        return json.dumps(sorted((dict(spec) for spec in specs),
                                 key=lambda spec: json.dumps(spec, sort_keys=True)),
                          sort_keys=True, separators=(",", ":"))

    def record_arrival(self, request_id, prompt_len, arrival_time_s=None):
        """Record an actual ingress event, never a CSV row not yet arrived."""
        arrival = self._clock() if arrival_time_s is None else float(arrival_time_s)
        prompt = int(prompt_len)
        if not math.isfinite(arrival) or prompt < 0:
            raise ValueError("arrival time must be finite and prompt nonnegative")
        key = str(request_id)
        with self._lock:
            if self._stop.is_set():
                return False
            if key in self._records:
                old = self._records[key]
                if old["prompt_len"] != prompt or old["arrival_time_s"] != arrival:
                    raise ValueError("request identity reused with a different arrival")
                return False
            self._records[key] = {
                "run_id": self._run_id, "request_id": key,
                "model": str(self.args.model), "arrival_time_s": arrival,
                "prompt_len": prompt, "output_len": None,
                "finished_time_s": None,
            }
            heapq.heappush(self._arrival_heap, (arrival, key))
            self._prune_locked(self._clock())
            return True

    def record_completion(self, request_id, observed_output_len,
                          finished_time_s=None):
        """Supply an observed successful completion, not requested max_tokens."""
        finished = (self._clock() if finished_time_s is None
                    else float(finished_time_s))
        observed = int(observed_output_len)
        if not math.isfinite(finished) or observed < 0:
            raise ValueError("completion time must be finite and output nonnegative")
        with self._lock:
            row = self._records.get(str(request_id))
            if self._stop.is_set() or row is None:
                return False
            if finished < row["arrival_time_s"]:
                raise ValueError("completion precedes arrival")
            if row["finished_time_s"] is not None:
                if (row["finished_time_s"] != finished
                        or row["output_len"] != observed):
                    raise ValueError("conflicting observed completion")
                return False
            # Published records are never mutated, allowing snapshot copying
            # outside the short ingress lock without mixed completion fields.
            completed = dict(row)
            completed["finished_time_s"] = finished
            completed["output_len"] = observed
            self._records[str(request_id)] = completed
            return True

    def _prune_locked(self, now):
        oldest = now - self._retention_s
        while self._arrival_heap and self._arrival_heap[0][0] < oldest:
            _, key = heapq.heappop(self._arrival_heap)
            self._records.pop(key, None)

    def snapshot(self, window_s=1800, end_s=None):
        """Return an isolated observation cutoff; safe for background planners.

        Redis I/O can block, so callers use this from their background job.
        Live CSV history includes only ingress/completion events. Its startup
        rate uses actual coverage, not an invented full window of idle time.
        """
        window_s = float(window_s)
        cutoff = self._clock() if end_s is None else float(end_s)
        now = self._clock()
        if (not math.isfinite(window_s) or window_s <= 0
                or not math.isfinite(cutoff) or cutoff > now):
            raise ValueError("snapshot needs a positive window and observed cutoff")
        if window_s > self._retention_s:
            raise ValueError("requested window exceeds retained ingress history")
        durable = None
        if is_redis_history(self.source):
            durable = read_redis_history(
                self.source, window_s=window_s,
                model=str(self.args.model), end_s=cutoff)
        with self._lock:
            self._prune_locked(now)
            retained_start = max(self._observation_started_s,
                                 now - self._retention_s)
            start = max(cutoff - window_s, retained_start)
            if cutoff < start:
                if durable is None:
                    raise ValueError("snapshot cutoff predates live observation")
                start = cutoff
            observed = list(self._records.values())
        rows = [copy.deepcopy(row) for row in observed
                if start <= row["arrival_time_s"] <= cutoff]
        if durable is not None:
            merged = {}
            for original in durable.records:
                row = copy.deepcopy(original)
                finished = row.get("finished_time_s")
                if finished is None or finished > cutoff:
                    row["output_len"] = row["finished_time_s"] = None
                merged[(str(row["run_id"]), str(row["request_id"]))] = row
            for local in rows:
                key = (local["run_id"], local["request_id"])
                previous = merged.get(key)
                if previous is not None:
                    if previous["prompt_len"] != local["prompt_len"]:
                        raise ValueError("conflicting durable/live prompt length")
                    if (previous.get("output_len") is not None
                            and local.get("output_len") is not None
                            and previous["output_len"] != local["output_len"]):
                        raise ValueError("conflicting durable/live completion")
                    if local["output_len"] is None:
                        local["output_len"] = previous.get("output_len")
                        local["finished_time_s"] = previous.get("finished_time_s")
                merged[key] = local
            rows = list(merged.values())
            # A Redis query boundary does not prove observation coverage before
            # the first stored event. Report the actual known coverage instead
            # of diluting a new ten-second key by an invented 1800 idle seconds.
            earliest = min((row["arrival_time_s"] for row in durable.records),
                           default=start)
            start = max(cutoff - window_s, min(start, earliest))
        rows.sort(key=lambda row: (row["arrival_time_s"], row["run_id"],
                                   row["request_id"]))
        for row in rows:
            finished = row.get("finished_time_s")
            if finished is None or finished > cutoff:
                row["output_len"] = row["finished_time_s"] = None
        return HistorySnapshot(
            durable.key if durable is not None else "live-ingress:" + self._run_id,
            str(self.args.model), start, cutoff, rows,
            source="redis+live_ingress" if durable is not None else "live_ingress",
            provenance={
                "requested_window_s": window_s,
                "effective_window_s": max(cutoff - start, 0),
                "observation_started_s": self._observation_started_s,
                "retained_start_s": retained_start,
                "cold_start": cutoff - start < window_s,
                "no_evaluation_csv_seed": True,
                "durable_records": len(durable.records) if durable else 0,
                "durable_coverage_inferred_from_first_arrival": durable is not None,
                "live_run_id": self._run_id,
            })

    def refresh_once(self, worker_specs=None):
        """Compute one whole bundle, without publishing any partial control."""
        # Preserve the existing synchronous CPU-test entry point constructed via
        # __new__; serving instances always take the isolated snapshot branch.
        snapshot = (self.snapshot(window_s=self.window_s)
                    if hasattr(self, "_lock") else read_redis_history(
                        self.source, window_s=self.window_s,
                        model=str(self.args.model)))
        if (snapshot.end_s <= snapshot.start_s or not snapshot.records
                or not any(row.get("output_len") is not None
                           for row in snapshot.records)):
            return None
        specs = copy.deepcopy(self.worker_specs if worker_specs is None
                              else worker_specs)
        counts = Counter(spec["role"] for spec in specs)
        stats = snapshot.statistics()
        stats["arrival_rate"] /= max(counts[DECODE], 1)
        stats["valid_until_s"] = snapshot.end_s + max(2 * self.refresh_s, 5)
        budgets = {}
        for spec in specs:
            name = spec["config_name"]
            if spec["role"] != PREFILL or name in budgets:
                continue
            budget = self._optimize_p_budget(snapshot, spec, counts[PREFILL])
            if budget is None and getattr(self.args, "enable_prefill_saturation", True):
                raise ValueError("history optimizer did not produce a P budget")
            if budget is not None:
                chunk = float(budget["chunk_ms"])
                if not math.isfinite(chunk) or chunk <= 0:
                    raise ValueError("history optimizer returned an invalid P budget")
                budgets[name] = {"chunk_ms": chunk}
        return {"decode_statistics": stats, "p_budgets": budgets,
                "history": snapshot.metadata(), "history_end_s": snapshot.end_s}

    def _optimize_p_budget(self, snapshot, spec, pool_size):
        if not getattr(self.args, "enable_prefill_saturation", True):
            return None
        from saturo.simulators.prefill_simulator.run_prefill_simulator import (
            run_prefill_profile)

        profile = Path(spec.get("runtime_profile") or self.args.runtime_profile)
        if not profile.is_file():
            raise FileNotFoundError("P history update requires a measured profile")
        # P uses arrivals/input lengths, including in-flight requests; unknown
        # final RT outputs are not inputs to the prefill budget optimization.
        rows = snapshot.workload_rows(completed_only=False)
        result = run_prefill_profile(
            rt_trace=self.source, rt_rows=rows, runtime_profile=str(profile),
            be_trace=self.args.be_workload_csv, num_req=len(rows),
            target_rps=len(rows) / (snapshot.end_s - snapshot.start_s)
                       / max(pool_size, 1),
            ttft_slo=self.args.rt_ttft_slo_ms / 1000,
            max_rt_violation_rate=self.args.rt_max_violation_rate,
            memory_watermark=self.args.resource_memory_watermark,
            max_prefill_chunk=self.args.resource_max_prefill_chunk,
            prefill_search_granularity=(
                self.args.resource_prefill_search_granularity),
            max_candidates=self.args.resource_profile_max_candidates,
            max_model_len=spec["max_model_len"],
            total_gpu_blocks=spec["num_gpu_blocks"] or None,
            be_input_drop_frac=self.args.be_input_drop_frac,
            be_output_keep_frac=self.args.be_output_keep_frac,
            be_output_max_len=self.args.be_output_max_len,
        )
        if result.get("rt_infeasible"):
            raise ValueError("history optimizer found RT-infeasible P budget")
        return {"chunk_ms": float(result["best"]["be_chunk_latency_ms"])}

    def _expire_locked(self, now):
        if self._stop.is_set():
            return
        pending = self._pending
        if (pending is not None and not pending["invalidated"]
                and now > pending["deadline_monotonic_s"]):
            pending["invalidated"] = True
            self._counters["timeouts"] += 1
            self._state = "timeout_retaining_previous"

    def _run_refresh(self):
        with self._lock:
            if self._stop.is_set():
                return
            self._next_version += 1
            started = self._monotonic()
            pending = {"version": self._next_version,
                       "pool_layout_generation": self._layout_generation,
                       "started_monotonic_s": started,
                       "deadline_monotonic_s": started + self.timeout_s,
                       "invalidated": False}
            self._pending = pending
            specs = copy.deepcopy(self.worker_specs)
            self._requested = False
            self._state = "refreshing"
            self._counters["attempts"] += 1
        update, error = None, None
        try:
            update = self.refresh_once(specs)
        except Exception as exc:
            error = type(exc).__name__
        with self._lock:
            finished = self._monotonic()
            self._expire_locked(finished)
            if self._stop.is_set():
                self._state = "closed"
            elif (pending["invalidated"]
                    or pending["pool_layout_generation"] != self._layout_generation):
                self._counters["discarded_late_or_layout"] += 1
                if pending["pool_layout_generation"] != self._layout_generation:
                    self._state = "discarded_layout_retaining_previous"
            elif error is not None:
                self._last_error_type = error
                self._counters["failures"] += 1
                self._state = "failed_retaining_previous"
                logger.warning("History update failed (%s); retaining previous "
                               "P/D bundle", error)
            elif update is None:
                self._counters["cold_start_skips"] += 1
                self._state = "cold_start_no_completed_history"
            else:
                update.update({
                    "version": pending["version"],
                    "pool_layout_generation": pending["pool_layout_generation"],
                    "provenance": {
                        "started_monotonic_s": started,
                        "finished_monotonic_s": finished,
                        "duration_s": finished - started,
                        "deadline_monotonic_s": pending["deadline_monotonic_s"],
                        "update_interval_s": self.refresh_s,
                        "timeout_s": self.timeout_s,
                    },
                })
                try:
                    encoded = json.dumps(update, sort_keys=True,
                                         allow_nan=False).encode()
                except (TypeError, ValueError) as exc:
                    self._last_error_type = type(exc).__name__
                    self._counters["failures"] += 1
                    self._state = "failed_retaining_previous"
                else:
                    self._latest = encoded
                    self._latest_version = pending["version"]
                    self._latest_layout_generation = (
                        pending["pool_layout_generation"])
                    self._counters["published"] += 1
                    self._last_error_type = None
                    self._state = "ready"
            self._pending = None
            self._last_attempt = {
                "version": pending["version"],
                "pool_layout_generation": pending["pool_layout_generation"],
                "started_monotonic_s": started,
                "finished_monotonic_s": finished,
                "deadline_monotonic_s": pending["deadline_monotonic_s"],
                "state": self._state,
                "error_type": (self._last_error_type
                               if self._state == "failed_retaining_previous"
                               else error),
                "history_end_s": update.get("history_end_s") if update else None,
            }
            self._next_due_s = started + self.refresh_s

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                wait_s = (0 if self._requested else
                          max(self._next_due_s - self._monotonic(), 0))
            self._wake.wait(wait_s)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                due = self._requested or self._monotonic() >= self._next_due_s
            if due:
                self._run_refresh()

    def request_refresh(self):
        """Coalesce an explicit request; never create another refresh thread."""
        with self._lock:
            if self._stop.is_set():
                return False
            if self._requested:
                self._counters["coalesced_requests"] += 1
            self._requested = True
        self._wake.set()
        return True

    def poll(self, worker_specs=None, *, pool_layout_generation=None):
        with self._lock:
            if worker_specs is not None:
                key = self._spec_key(worker_specs)
                generation = (self._layout_generation if pool_layout_generation is None
                              else int(pool_layout_generation))
                if generation < self._layout_generation:
                    raise ValueError("pool layout generation must not move backwards")
                if key != self._layout_key or generation != self._layout_generation:
                    if (key != self._layout_key
                            and generation == self._layout_generation):
                        generation += 1
                    self.worker_specs = copy.deepcopy(worker_specs)
                    self._layout_key, self._layout_generation = key, generation
                    if self._pending is not None:
                        self._pending["invalidated"] = True
                    self._requested = True
                    self._counters["layout_changes"] += 1
                    self._wake.set()
            self._expire_locked(self._monotonic())
            if (self._stop.is_set() or self._latest is None
                    or self._latest_version <= self._delivered_version
                    or self._latest_layout_generation != self._layout_generation):
                return None
            update = json.loads(self._latest)
            self._delivered_version = update["version"]
            if update["decode_statistics"]["valid_until_s"] < self._clock():
                self._counters["expired_before_delivery"] += 1
                self._state = "expired_retaining_previous"
                return None
            return update

    def status(self):
        with self._lock:
            self._expire_locked(self._monotonic())
            return {"state": self._state,
                    "pool_layout_generation": self._layout_generation,
                    "latest_version": self._latest_version or None,
                    "pending_version": (self._pending["version"]
                                        if self._pending is not None else None),
                    "last_error_type": self._last_error_type,
                    "last_attempt": copy.deepcopy(self._last_attempt),
                    "closed": self._stop.is_set(),
                    "thread_alive": self._thread.is_alive(),
                    "observed_requests": len(self._records),
                    "counters": dict(self._counters)}

    def close(self, timeout=1.0):
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(float(timeout), 0))
        with self._lock:
            self._state = "closed"
        return self.status()
