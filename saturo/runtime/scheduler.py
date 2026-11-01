"""Saturo global scheduler for asynchronous Ray engine workers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional

from saturo.runtime.pool_controls import pool_accepts_be
from saturo.runtime.kv_handoff import new_handoff, real_handoff, remove_handoff
from saturo.runtime.launcher import WorkerLauncherMixin
from saturo.runtime.updates import RuntimeUpdatesMixin
from saturo.runtime.deployment import PeriodicDeploymentMixin
from saturo.runtime.reservations import (CONTROL_COMMANDS, DecodeReservationMixin)
from saturo.runtime.worker import WorkerPlacement, WorkerState
from saturo.runtime.workload import (
    BE,
    DECODE,
    PREFILL,
    RT,
    TPUT,
    REQUEST_TBT_DEFINITION,
    SaturoRequest,
    load_workload_csv,
    load_rt_workload_csv,
    percentile,
)
from saturo.trace_collection import TraceRecorder, store_from_env
from saturo.history import configured_history_source, is_redis_history


MAX_REQUEST_ERROR_RETRIES = 3


def _is_dead_engine_exception(exc: BaseException) -> bool:
    """Recognize both local and Ray-wrapped AsyncEngineDeadError causes."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if "AsyncEngineDeadError" in type(exc).__name__:
            return True
        exc = (getattr(exc, "cause", None)
               or getattr(exc, "__cause__", None)
               or getattr(exc, "__context__", None))
    return False


class SaturoGlobalScheduler(RuntimeUpdatesMixin, PeriodicDeploymentMixin,
                            DecodeReservationMixin, WorkerLauncherMixin):
    """Route requests while each worker's AsyncLLMEngine runs independently."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._kv_transfer_namespace = uuid.uuid4().hex
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._perf_events_enabled = (
            os.environ.get("SATURO_STEP_TIMING", "").strip().lower()
            in ("1", "true", "yes", "on"))
        self.worker_log_dir = self.output_dir / "worker_logs"
        self.worker_log_dir.mkdir(parents=True, exist_ok=True)

        self.workers: dict[int, WorkerState] = {}
        self.next_worker_id = 0
        self._pool_layout_generation = 0
        self._history_rollout = None
        self._epoch_pending = {}
        # Maps config_name -> be_chunk_latency_ms for P-pool workers, populated
        # from the allocation profile's prefill simulator optimized BE chunk
        # latency from the resource profile.
        self._p_pool_be_chunk_ms_map: dict[str, float] = {}
        self._allocation_profiles = None
        self.worker_plan = deque(self._build_worker_plan())
        self.started_s = 0.0
        self.trace_recorder: Optional[TraceRecorder] = None
        self.history_controller = None

        # Request state
        self.all_rt: list[SaturoRequest] = []
        self.all_be: list[SaturoRequest] = []
        self._request_index: dict[str, SaturoRequest] = {}
        self.next_rt_idx = 0
        self.pending_decode: Deque[SaturoRequest] = deque()
        self.pending_prefill: Deque[SaturoRequest] = deque()
        self.pending_be: Deque[SaturoRequest] = deque()
        self.role_switcher = None
        if (getattr(args, "adaptive_role_switching", True)
                and self._allocation_profiles is not None):
            from saturo.runtime.role_switching import AdaptiveRoleSwitcher
            self.role_switcher = AdaptiveRoleSwitcher(
                self, self._allocation_profiles)

        # Metrics
        self.rt_tbt_samples_ms: list[float] = []
        self.rt_ttft_samples_ms: list[float] = []
        self.rt_tbt_violations = 0
        self.rt_ttft_violations = 0
        self.be_tokens = 0
        self.be_tokens_in_window = 0
        self.rt_tokens = 0
        self.prefill_handoffs = 0
        self.request_errors = 0
        self.engine_failure: Optional[dict[str, Any]] = None
        self.request_failure: Optional[dict[str, Any]] = None
        self._request_error_counts: dict[str, int] = {}
        self.stop_reason = "running"
        self.history_updates = 0

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        if not self.args.model:
            raise ValueError("missing --model for Saturo run")
        if ((getattr(self.args, "enable_decode_saturation", True)
             or getattr(self.args, "adaptive_role_switching", False))
                and any(p.role == DECODE for p in self.worker_plan)):
            from saturo.predictors.qrf import load_required_qrf, validate_qrf_workload
            predictor = load_required_qrf()  # Validate before any serving workers start.
            validate_qrf_workload(predictor, self.args)
        self._build_workload()
        if (self.all_be and not any(pool_accepts_be(p.role, self.args)
                                   for p in self.worker_plan)):
            raise ValueError("No BE-eligible pool in the allocation; this ablation needs a T instance")
        # Validate optional trace configuration before starting GPU workers.
        trace_store = store_from_env()
        self.started_s = time.perf_counter()
        try:
            self._start_initial_workers()
            if self.role_switcher is not None:
                for worker in self._live_workers():
                    self.role_switcher._predictor(worker)
            if self.args.worker_warmup:
                self._warmup_workers()
            self._preload_be_to_workers()
            self.started_s = time.perf_counter()
            # The scaled trace is observed at the ingress clock. D-pool RPC
            # arrival times also contain P execution/transport batching delay.
            self._source_arrival_epoch_s = time.time()
            if trace_store is not None:
                self.trace_recorder = TraceRecorder(
                    trace_store, self.args.model, started_at_s=time.time())
                print(f"[scheduler] Redis RT trace key: {trace_store.key}", flush=True)
            self._init_runtime_updates()
            self._init_periodic_reallocation()
            while not self._check_done():
                self._poll_history()
                # 1) 按到达时间注入 RT 请求到 P-pool
                self._inject_due_requests()
                # 2) 将 prefill 完成的请求路由到 D-pool
                self._drain_pending_decode()
                self._drain_pending_be()
                # 3) 提交新请求并收集 AsyncLLMEngine 的流式事件
                self._service_workers()
                self._poll_periodic_reallocation()
                if (self.role_switcher is not None
                        and not self._has_terminal_failure()
                        and not self._deployment_needs_workers()
                        and not self._epoch_pending
                        and self._history_rollout is None):
                    self.role_switcher.maintain()
                time.sleep(0.002)
        finally:
            self._close_periodic_reallocation()
            if self.history_controller is not None:
                self.history_controller.close()
            try:
                self._shutdown_workers()
            finally:
                if self.trace_recorder is not None:
                    self.trace_recorder.close()
        self._write_outputs()
        return self._summary()

    def _poll_history(self) -> None:
        self._poll_history_update_protocol()

    def _history_worker_specs(self) -> list[dict]:
        return [{"worker_id": w.worker_id, "role": w.role, "config_name": w.config_name,
                 "runtime_profile": w.runtime_profile_path,
                 "num_gpu_blocks": w.num_gpu_blocks,
                 "max_model_len": w.max_model_len}
                for w in self._live_workers()]

    # ------------------------------------------------------------------
    # Request injection (time-based)
    # ------------------------------------------------------------------

    def _inject_due_requests(self) -> None:
        """Observe ingress even while a deployment buffers new work."""
        elapsed = self._elapsed()
        if not hasattr(self, "pending_prefill"):
            self.pending_prefill = deque()
        while (self.next_rt_idx < len(self.all_rt)
               and self.all_rt[self.next_rt_idx].arrival_offset_s <= elapsed):
            req = self.all_rt[self.next_rt_idx]
            self.next_rt_idx += 1
            self._observe_runtime_arrival(req)
            if self.trace_recorder is not None:
                self.trace_recorder.record(req)
            if getattr(self, "role_switcher", None) is not None:
                self.role_switcher.observe_arrival(req.arrival_offset_s)
            self.pending_prefill.append(req)
        self._drain_pending_prefill()

    def _select_rt_worker(self, role, req):
        switcher = getattr(self, "role_switcher", None)
        if switcher is not None:
            return switcher.select(role, req)
        return self._least_loaded(role)

    def _drain_pending_prefill(self) -> None:
        if self._deployment_pauses_new_work() or getattr(self, "_epoch_pending", {}):
            return
        for _ in range(len(self.pending_prefill)):
            req = self.pending_prefill.popleft()
            if req.is_finished or req.active_attempt_id is not None:
                continue
            # Reserve D before borrowing P when T workers are scarce.
            if (getattr(self, "role_switcher", None) is not None
                    and not self._reserve_decode_for_prefill(req)):
                self.pending_prefill.append(req)
                continue
            worker = self._select_rt_worker(PREFILL, req)
            if worker is None or not self._assign(worker, req, PREFILL):
                self.pending_prefill.append(req)

    def _drain_pending_be(self) -> None:
        if self._deployment_pauses_new_work() or getattr(self, "_epoch_pending", {}):
            return
        for _ in range(len(self.pending_be)):
            req = self.pending_be.popleft()
            self._requeue(req, TPUT)

    # ------------------------------------------------------------------
    # Async worker submission and event collection
    # ------------------------------------------------------------------

    def _service_workers(self) -> None:
        """Deliver ready workers without waiting behind a busy peer actor."""
        if self._has_terminal_failure():
            return
        ray = self._ensure_ray()
        if callable(getattr(ray, "wait", None)):
            self._service_workers_ready(ray)
        else:
            # Embedding clients and older get-only test doubles retain the
            # synchronous API; actual Ray always supports readiness polling.
            self._service_workers_blocking()

    def _drop_worker_rpc_refs(self, worker_id: int) -> None:
        for table in getattr(self, "_worker_rpcs", {}).values():
            table.pop(worker_id, None)

    def _handle_worker_rpc_failure(self, worker, exc, operation) -> None:
        self._drop_worker_rpc_refs(worker.worker_id)
        print(f"[scheduler] Worker {worker.worker_id} {operation} error: {exc}",
              flush=True)
        if not self._fail_dead_engine_rpc(worker, exc, operation):
            self._retire_worker(worker, requeue_active=True)

    def _poll_worker_rpcs(self, ray, kind: str) -> None:
        table = self._worker_rpcs[kind]
        for wid in list(table):
            worker = self.workers.get(wid)
            if worker is None or worker.retired:
                table.pop(wid, None)
        by_ref = {entry["ref"]: (wid, entry) for wid, entry in table.items()}
        if by_ref:
            ready, _ = ray.wait(list(by_ref), num_returns=len(by_ref), timeout=0)
            for ref in ready:
                wid, entry = by_ref[ref]
                worker = self.workers.get(wid)
                if (table.get(wid) is not entry or worker is None
                        or worker.retired):
                    continue
                table.pop(wid)
                operation = ("decode_reservations" if kind == "submit"
                             and entry["commands"]
                             and all(c.get("cmd") in CONTROL_COMMANDS
                                     for c in entry["commands"])
                             else "submit" if kind == "submit" else "drain_events")
                try:
                    result = ray.get(ref, timeout=0)
                    if kind == "submit":
                        self._acknowledge_decode_reservations(
                            worker, entry["commands"], result)
                        self._acknowledge_runtime_controls(worker, entry["commands"], result)
                    else:
                        self._write_perf_events(worker, result, time.perf_counter())
                        self._consume_worker_events(result)
                except Exception as exc:
                    self._handle_worker_rpc_failure(worker, exc, operation)
                if self._has_terminal_failure():
                    return
        # A busy actor is not failed merely because a peer is ready first.
        # Keep the same five-second RPC deadline, measured from dispatch.
        now_s = time.perf_counter()
        for wid, entry in list(table.items()):
            if now_s - entry["sent_s"] >= 5.0:
                worker = self.workers.get(wid)
                if worker is not None and not worker.retired:
                    self._handle_worker_rpc_failure(
                        worker, TimeoutError("worker RPC exceeded 5.0 seconds"),
                        "submit" if kind == "submit" else "drain_events")
                else:
                    table.pop(wid, None)
                if self._has_terminal_failure():
                    return

    def _consume_worker_events(self, events: list[dict]) -> None:
        for event in events:
            if event.get("event") in ("finished", "error", "aborted"):
                with (self.output_dir / "request_events.jsonl").open("a") as log:
                    log.write(json.dumps({**event, "scheduler_elapsed_s": self._elapsed()}) + "\n")
            self._handle_event(event)
            if self._has_terminal_failure():
                return

    def _service_workers_ready(self, ray) -> None:
        if not hasattr(self, "_worker_rpcs"):
            self._worker_rpcs = {"submit": {}, "drain": {}}
        # Install early-claim acknowledgements before considering P work,
        # then consume all ready events before dispatching another batch.
        self._poll_worker_rpcs(ray, "submit")
        if self._has_terminal_failure():
            return
        self._poll_worker_rpcs(ray, "drain")
        if self._has_terminal_failure():
            return
        submits = self._worker_rpcs["submit"]
        drains = self._worker_rpcs["drain"]
        for worker in self._live_workers():
            wid = worker.worker_id
            if wid in submits or not worker.pending_commands:
                continue
            queued = worker.pending_commands
            worker.pending_commands = []
            controls, additions, held = [], [], []
            for command in queued:
                if command.get("cmd") in CONTROL_COMMANDS:
                    controls.append(command)
                elif self._prefill_reservation_ready(command):
                    additions.append(command)
                else:
                    req = self._request_by_id(str(command.get("request_id", "")))
                    if req is None or not req.is_finished:
                        held.append(command)
            # Reservation checks can append fresh controls to other workers;
            # do not replace those queues while retaining held P additions.
            worker.pending_commands.extend(held)
            commands = controls + additions
            if commands:
                try:
                    sent_s = time.perf_counter()
                    ref = worker.actor.submit_batch.remote(commands)
                    submits[wid] = {"ref": ref, "sent_s": sent_s,
                                    "commands": commands}
                except Exception as exc:
                    self._handle_worker_rpc_failure(worker, exc, "submit")
                if self._has_terminal_failure():
                    return
        for worker in self._live_workers():
            wid = worker.worker_id
            if wid in drains or (not worker.active_attempts and wid not in submits):
                continue
            try:
                sent_s = time.perf_counter()
                ref = worker.actor.drain_events.remote()
                drains[wid] = {"ref": ref, "sent_s": sent_s}
            except Exception as exc:
                self._handle_worker_rpc_failure(worker, exc, "drain_events")
            if self._has_terminal_failure():
                return

    def _service_workers_blocking(self) -> None:
        """Flush control-plane commands and drain async request events.

        ``submit_batch`` only creates per-request asyncio tasks in the Ray
        actor. Model execution is owned by AsyncLLMEngine's background loop;
        this method never calls, or indirectly asks the worker to call,
        ``LLMEngine.step``.
        """
        if self._has_terminal_failure():
            return
        ray = self._ensure_ray()
        failed_workers = self._flush_decode_reservations()
        if self._has_terminal_failure():
            return
        submit_refs: dict[Any, WorkerState] = {}
        submitted_worker_ids: set[int] = set()
        for w in self._live_workers():
            if not w.pending_commands:
                continue
            commands = []
            held = []
            for command in w.pending_commands:
                if command.get("cmd") in CONTROL_COMMANDS:
                    held.append(command)
                elif self._prefill_reservation_ready(command):
                    commands.append(command)
                else:
                    req = self._request_by_id(str(command.get("request_id", "")))
                    if req is None or not req.is_finished:
                        held.append(command)
            w.pending_commands = held
            if not commands:
                continue
            submit_refs[w.actor.submit_batch.remote(commands)] = w
            submitted_worker_ids.add(w.worker_id)

        for ref, worker in submit_refs.items():
            try:
                # This waits for admission only, not for model completion.
                ray.get(ref, timeout=5.0)
            except Exception as exc:
                print(f"[scheduler] Worker {worker.worker_id} submit error: "
                      f"{exc}\n" + traceback.format_exc(),
                      flush=True)
                if self._fail_dead_engine_rpc(worker, exc, "submit"):
                    return
                self._retire_worker(worker, requeue_active=True)
                failed_workers.add(worker.worker_id)

        drain_refs: dict[Any, WorkerState] = {}
        for worker in self._live_workers():
            if worker.worker_id in failed_workers:
                continue
            if (not worker.active_attempts
                    and worker.worker_id not in submitted_worker_ids):
                continue
            drain_refs[worker.actor.drain_events.remote()] = worker

        for ref, worker in drain_refs.items():
            try:
                events = ray.get(ref, timeout=5.0)
                self._write_perf_events(worker, events, time.perf_counter())
                self._consume_worker_events(events)
                if self._has_terminal_failure():
                    return
            except Exception as exc:
                print(f"[scheduler] Worker {worker.worker_id} event error: "
                      f"{exc}\n" + traceback.format_exc(),
                      flush=True)
                if self._fail_dead_engine_rpc(worker, exc, "drain_events"):
                    return
                self._retire_worker(worker, requeue_active=True)

    def _write_perf_events(self, worker: WorkerState, events: list[dict],
                           received_perf_s: float) -> None:
        """Opt-in scalar diagnostics; preserve production metric clocks."""
        if not getattr(self, "_perf_events_enabled", False):
            return
        received_elapsed_s = self._elapsed()
        scalar_types = (str, int, float, bool, type(None))
        records = []
        for event in events:
            if event.get("event") not in (
                    "token", "iteration_log", "finished", "error", "aborted",
                    "request_stage", "admission_snapshot"):
                continue
            # Avoid serializing per-request/context arrays on every iteration.
            record = {key: value for key, value in event.items()
                      if isinstance(value, scalar_types)}
            decision = event.get("decode_saturation_decision")
            if isinstance(decision, dict):
                record.update({"decision_" + key: value
                               for key, value in decision.items()
                               if isinstance(value, scalar_types)})
            planner = event.get("planner_timing")
            if isinstance(planner, dict):
                record.update({("planner_" + key if not key.startswith("planner_")
                                else key): value
                               for key, value in planner.items()
                               if isinstance(value, scalar_types)})
            record.update(worker_id=worker.worker_id,
                          worker_role=worker.role,
                          scheduler_elapsed_s=received_elapsed_s,
                          scheduler_received_perf_s=received_perf_s)
            records.append(json.dumps(record) + "\n")
        if records:
            # One open/write per drained batch rather than per token/event.
            with (self.output_dir / "perf_events.jsonl").open("a") as log:
                log.writelines(records)

    # ------------------------------------------------------------------
    # Decode queue drain
    # ------------------------------------------------------------------

    def _drain_pending_decode(self) -> None:
        """将 pending_decode 中的请求发送到 D-pool。"""
        for _ in range(len(self.pending_decode)):
            req = self.pending_decode.popleft()
            if req.is_finished:
                self._release_decode_reservation(req)
                continue
            if req.active_attempt_id is not None:
                continue
            worker = self._reserved_decode_worker(req)
            if worker is None:
                self.pending_decode.append(req)
            else:
                self._assign(worker, req, DECODE)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_event(self, event: dict[str, Any]) -> None:
        worker = self.workers.get(int(event.get("worker_id", -1)))
        kind = event.get("event")
        if worker is not None and worker.retired:
            return
        if kind == "ready" and worker:
            worker.ready = True
            return
        if kind in ("log", "iteration_log") or worker is None:
            return

        # Engine death belongs to the actor, not one request attempt. The
        # stream that observed it may already have been aborted/reassigned;
        # discarding that event would leave an irrecoverable engine live.
        if kind == "error" and (event.get("engine_fatal")
                                or event.get("error_type") == "AsyncEngineDeadError"):
            self._fail_engine_worker(worker, event)
            return

        request_id = str(event.get("request_id", ""))
        warmup = request_id.startswith("__warmup__")
        req = self._request_by_id(request_id)
        if req is None and not warmup:
            return
        phase = str(event.get("phase", ""))
        attempt_id = str(event.get("attempt_id", ""))
        if (kind in ("finished", "aborted", "error") and attempt_id
                and ((warmup and attempt_id not in worker.active_attempts)
                     or (not warmup and req.active_attempt_id != attempt_id))):
            worker.active_attempts.pop(attempt_id, None)
            return  # Late events must not cancel a newer D reservation.

        if warmup:
            if kind in ("finished", "aborted", "error"):
                worker.active_attempts.pop(attempt_id, None)
            return

        now_s = self._elapsed()
        if real_handoff(getattr(self, "args", None)) and req.kind == RT and kind in ("token", "finished"):
            if kind == "token" and attempt_id != req.active_attempt_id:
                return  # An old actor must not overwrite a newer real context.
            sampled = event.get("sampled_token_ids")
            if sampled is None:
                raise RuntimeError("Real KV handoff received an RT event without sampled token IDs")
            base = int(event["base_tokens"])
            req.generated_token_ids[base:] = [int(t) for t in sampled]
            expected = min(int(event["global_token_count"]), req.output_len)
            if len(req.generated_token_ids) != expected:
                raise RuntimeError("Real KV handoff sampled token count mismatch")
        if kind == "token":
            self._on_token(req, event, now_s)
        elif kind == "finished":
            worker.active_attempts.pop(attempt_id, None)
            if req.active_attempt_id == attempt_id:
                req.active_attempt_id = None
                req.assigned_worker_id = None
            global_tokens = int(
                event.get("global_token_count", req.observed_tokens))
            old_tokens = req.observed_tokens
            req.observed_tokens = min(max(req.observed_tokens, global_tokens),
                                      req.output_len)
            if req.kind == RT and req.observed_tokens > old_tokens:
                req.tbt_unobserved_intervals += (
                    max(req.observed_tokens - 1, 0) - max(old_tokens - 1, 0))
            if phase == PREFILL:
                req.prefill_finished_s = now_s
                self.prefill_handoffs += 1
                if req.is_finished:
                    req.finished_s = now_s
                    self._release_decode_reservation(req)
                else:
                    self.pending_decode.append(req)
            else:
                if req.is_finished:
                    req.finished_s = now_s
                elif phase in (DECODE, TPUT):
                    # A native terminal output can still be below the replay
                    # target (for example, at a context-length boundary).
                    # It is neither a completion nor a retriable preemption.
                    self._fail_incomplete_generation(worker, req, event)
            if req.is_finished and req.kind == RT and real_handoff(getattr(self, "args", None)):
                remove_handoff(self.args.kv_transfer_dir, self._kv_transfer_namespace, req.kv_handoff)
            self._observe_runtime_completion(req)
            if (self.trace_recorder is not None and req.kind == RT
                    and req.finished_s is not None):
                self.trace_recorder.record(req)
        elif kind in ("aborted", "error"):
            if kind == "error":
                self.request_errors += 1
                print(
                    f"[scheduler] Request {request_id} failed on worker "
                    f"{worker.worker_id}: {event.get('error_type', 'Error')}: "
                    f"{event.get('message', '')}",
                    flush=True,
                )
            worker.active_attempts.pop(attempt_id, None)
            if req.active_attempt_id == attempt_id:
                req.active_attempt_id = None
                req.assigned_worker_id = None
            if phase == PREFILL:
                self._release_decode_reservation(req)
            if kind == "error":
                counts = getattr(self, "_request_error_counts", None)
                if counts is None:
                    self._request_error_counts = counts = {}
                counts[request_id] = counts.get(request_id, 0) + 1
                if counts[request_id] > MAX_REQUEST_ERROR_RETRIES:
                    self.request_failure = {
                        "request_id": request_id, "worker_id": worker.worker_id,
                        "error_type": event.get("error_type"),
                        "message": event.get("message", ""),
                        "errors": counts[request_id],
                        "retry_limit": MAX_REQUEST_ERROR_RETRIES,
                    }
                    return
            if not req.is_finished:
                self._requeue(req, phase)

    def _has_terminal_failure(self) -> bool:
        return bool(getattr(self, "engine_failure", None)
                    or getattr(self, "request_failure", None))

    def _fail_incomplete_generation(self, worker: WorkerState,
                                    req: SaturoRequest, event: dict) -> None:
        """Fail immediately when a live final-phase stream ends below target."""
        if self._has_terminal_failure():
            return
        self.request_errors += 1
        self.request_failure = {
            "error_type": "IncompleteGeneration",
            "request_id": req.request_id, "kind": req.kind,
            "attempt_id": event.get("attempt_id"),
            "worker_id": worker.worker_id, "phase": event.get("phase"),
            "observed_tokens": req.observed_tokens,
            "requested_tokens": req.output_len,
            "missing_tokens": req.remaining_tokens,
            "message": "Worker finished before the requested output token target",
            "elapsed_s": self._elapsed(),
        }
        print(f"[scheduler] Incomplete generation; stopping run: {self.request_failure}",
              flush=True)

    def _fail_dead_engine_rpc(self, worker: WorkerState, exc: BaseException,
                              operation: str) -> bool:
        if not _is_dead_engine_exception(exc):
            return False
        self._fail_engine_worker(worker, {
            "error_type": "AsyncEngineDeadError",
            "message": str(exc), "operation": operation})
        return True

    def _fail_engine_worker(self, worker: WorkerState, event: dict) -> None:
        """Stop the run on a dead engine, without replaying failed streams."""
        if self._has_terminal_failure():
            return
        self.request_errors += 1
        self.engine_failure = {
            "worker_id": worker.worker_id, "role": worker.role,
            "error_type": event.get("error_type"),
            "message": event.get("message", ""),
            "request_id": event.get("request_id"),
            "attempt_id": event.get("attempt_id"),
            "operation": event.get("operation", "generate"),
            "elapsed_s": self._elapsed(),
        }
        print(f"[scheduler] Engine failed; stopping run: {self.engine_failure}",
              flush=True)
        # This run is terminating. Do not reserve replacement D capacity or
        # requeue requests on another worker while processing the error fanout.
        for req in self.all_rt + self.all_be:
            self._release_decode_reservation(req, notify=False)
            if req.assigned_worker_id == worker.worker_id:
                req.active_attempt_id = None
                req.assigned_worker_id = None
        worker.active_attempts.clear()
        worker.preempting_attempts.clear()
        worker.pending_commands.clear()
        worker.ready = False
        self._retire_worker(worker, requeue_active=False)

    def _on_token(self, req: SaturoRequest, event: dict, now_s: float) -> None:
        global_tokens = int(
            event.get("global_token_count", req.observed_tokens))
        old = req.observed_tokens
        req.observed_tokens = min(max(req.observed_tokens, global_tokens),
                                  req.output_len)
        delta = max(req.observed_tokens - old, 0)
        if delta == 0:
            return
        if req.kind == RT:
            self.rt_tokens += delta
            req.tbt_unobserved_intervals += delta - 1
            if req.first_token_s is None:
                req.first_token_s = now_s
                ttft_ms = (now_s - req.arrival_offset_s) * 1000.0
                self.rt_ttft_samples_ms.append(ttft_ms)
                if ttft_ms > self.args.rt_ttft_slo_ms:
                    self.rt_ttft_violations += 1
            if req.last_token_s is not None:
                tbt_ms = (now_s - req.last_token_s) * 1000.0
                req.tbt_ms = max(req.tbt_ms or 0.0, tbt_ms)
                req.tbt_samples += 1
                self.rt_tbt_samples_ms.append(tbt_ms)
                if tbt_ms > self.args.rt_tbt_slo_ms:
                    self.rt_tbt_violations += 1
            req.last_token_s = now_s
        else:
            self.be_tokens += delta
            if now_s <= self.args.window_s:
                self.be_tokens_in_window += delta

    # ------------------------------------------------------------------
    # Assign request to worker
    # ------------------------------------------------------------------

    def _assign(self, worker: WorkerState, req: SaturoRequest,
                phase: str) -> bool:
        if (req.is_finished or worker.retired or not worker.ready
                or self._has_terminal_failure()):
            return False
        if (self._deployment_pauses_new_work()
                and (phase == PREFILL or req.kind == BE)):
            return False
        if req.kind == BE and not pool_accepts_be(worker.role, self.args):
            return False
        elapsed = self._elapsed()
        if phase == PREFILL:
            if req.kind == RT and not self._reserve_decode_for_prefill(req):
                return False
            req.prefill_attempts += 1
            attempt_id = f"{req.request_id}:p{req.prefill_attempts}"
            max_tokens = 1
            prompt_len = req.prompt_len
            if req.kind == RT and real_handoff(getattr(self, "args", None)):
                context = list(req.prompt_token_ids or [0] * req.prompt_len) + req.generated_token_ids
                prompt_len = len(context)
                remove_handoff(self.args.kv_transfer_dir, self._kv_transfer_namespace, req.kv_handoff)
                req.kv_handoff = new_handoff(worker, context)
            req.prefill_submitted_s = req.prefill_submitted_s or elapsed
        else:
            if req.remaining_tokens <= 0:
                req.finished_s = elapsed
                return
            req.decode_attempts += 1
            attempt_id = f"{req.request_id}:d{req.decode_attempts}"
            max_tokens = req.remaining_tokens
            prompt_len = req.decode_prompt_len
            req.decode_submitted_s = req.decode_submitted_s or elapsed

        req.active_attempt_id = attempt_id
        req.assigned_worker_id = worker.worker_id
        worker.active_attempts[attempt_id] = req.request_id
        payload: dict[str, Any] = {
            "cmd": "add",
            "request_id": req.request_id,
            "attempt_id": attempt_id,
            "kind": req.kind,
            "phase": phase,
            "prompt_len": prompt_len,
            "max_tokens": max_tokens,
            "base_tokens": req.observed_tokens,
        }
        if req.kind == RT:
            epoch = getattr(self, "_source_arrival_epoch_s", None)
            if epoch is None:
                # Also support embedding callers which use _assign directly.
                epoch = time.time() - elapsed
                self._source_arrival_epoch_s = epoch
            payload["source_request_id"] = req.request_id
            payload["source_arrival_time_s"] = epoch + req.arrival_offset_s
            if getattr(self, "args", None) is not None:
                payload["absolute_prefill_deadline_s"] = (
                    epoch + req.arrival_offset_s
                    + getattr(self.args, "rt_ttft_slo_ms", 2000) / 1000.0)
        if req.kind == RT and real_handoff(getattr(self, "args", None)):
            payload["prompt_token_ids"] = list(req.prompt_token_ids or [0] * req.prompt_len) + req.generated_token_ids
            payload["prompt_len"] = len(payload["prompt_token_ids"])
            if not req.kv_handoff:
                raise RuntimeError("Real RT decode has no completed P-pool handoff")
            payload["kv_handoff"] = dict(req.kv_handoff)
        elif req.prompt_token_ids is not None and phase == PREFILL:
            payload["prompt_token_ids"] = req.prompt_token_ids[:prompt_len]
        if phase == DECODE and req.decode_reservation_id:
            payload["reservation_id"] = req.decode_reservation_id
            self._release_decode_reservation(req, notify=False)
        worker.pending_commands.append(payload)
        return True

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup_workers(self) -> None:
        pending: set[str] = set()
        for w in self._live_workers():
            aid = f"__warmup__:{w.worker_id}"
            pending.add(aid)
            w.active_attempts[aid] = aid
            w.pending_commands.append({
                "cmd": "add",
                "request_id": aid,
                "attempt_id": aid,
                "kind": BE if pool_accepts_be(w.role, self.args) else RT,
                "phase": TPUT,
                "prompt_len": self.args.worker_warmup_prompt_len,
                "max_tokens": self.args.worker_warmup_output_len,
                "base_tokens": 0,
            })
        deadline = time.perf_counter() + self.args.worker_warmup_timeout_s
        while pending and time.perf_counter() < deadline:
            self._service_workers()
            if self._has_terminal_failure():
                return
            pending = {
                aid
                for aid in pending
                if any(aid in w.active_attempts for w in self.workers.values())
            }
            if pending:
                time.sleep(0.01)
        if pending:
            raise RuntimeError(f"warmup timed out: {sorted(pending)}")

    # ------------------------------------------------------------------
    # BE pre-loading
    # ------------------------------------------------------------------

    def _preload_be_to_workers(self) -> None:
        """将 BE 请求均分到所有 worker。"""
        if self._has_terminal_failure():
            return
        ready = [w for w in self._live_workers()
                 if w.ready and pool_accepts_be(w.role, self.args)]
        if not ready:
            return
        for idx, req in enumerate(self.all_be):
            worker = ready[idx % len(ready)]
            req.decode_attempts += 1
            attempt_id = f"{req.request_id}:t{req.decode_attempts}"
            req.active_attempt_id = attempt_id
            req.assigned_worker_id = worker.worker_id
            req.prefill_submitted_s = 0.0
            req.decode_submitted_s = 0.0
            worker.active_attempts[attempt_id] = req.request_id
            prompt_len = req.prompt_len
            max_tokens = req.remaining_tokens
            payload: dict[str, Any] = {
                "cmd": "add",
                "request_id": req.request_id,
                "attempt_id": attempt_id,
                "kind": req.kind,
                "phase": TPUT,
                "prompt_len": prompt_len,
                "max_tokens": max_tokens,
                "base_tokens": 0,
            }
            if req.prompt_token_ids is not None:
                payload["prompt_token_ids"] = req.prompt_token_ids[:prompt_len]
            worker.pending_commands.append(payload)
        print(
            f"[scheduler] Pre-loaded {len(self.all_be)} BE requests "
            f"across {len(ready)} workers",
            flush=True)

    # ------------------------------------------------------------------
    # Workload loading
    # ------------------------------------------------------------------

    def _build_workload(self) -> None:
        if self.args.rt_workload_csv:
            self.all_rt = load_rt_workload_csv(
                self.args.rt_workload_csv,
                num_req=self.args.num_req,
                target_rps=self.args.target_rps,
                expected_model=getattr(self.args, "model", None),
                max_model_len=getattr(self.args, "max_model_len", None),
            )
        else:
            raise ValueError("No RT workload CSV provided. "
                             "Specify --rt-workload-csv.")
        if self.args.be_workload_csv:
            self.all_be = load_workload_csv(
                self.args.be_workload_csv, BE,
                expected_model=getattr(self.args, "model", None),
                max_model_len=getattr(self.args, "max_model_len", None))
        else:
            raise ValueError("No BE workload CSV provided. "
                             "Specify --be-workload-csv.")
        self.all_rt.sort(key=lambda r: r.arrival_offset_s)
        self._request_index = {}
        for req in self.all_rt + self.all_be:
            if req.request_id in self._request_index:
                raise ValueError(f"Duplicate request_id across RT/BE workloads: {req.request_id}")
            self._request_index[req.request_id] = req
        print(
            f"[scheduler] Loaded fixed-num RT: {len(self.all_rt)} reqs, "
            f"target_rps={self.args.target_rps:.6g}, "
            f"window_s={self.args.window_s:.6f}",
            flush=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _elapsed(self) -> float:
        return time.perf_counter() - self.started_s

    def _check_done(self) -> bool:
        elapsed = self._elapsed()
        if getattr(self, "engine_failure", None):
            self.stop_reason = "engine_failed"
        elif getattr(self, "request_failure", None):
            self.stop_reason = (
                "incomplete_generation"
                if self.request_failure.get("error_type") == "IncompleteGeneration"
                else "request_error_limit")
        elif elapsed >= self.args.max_wall_s:
            self.stop_reason = "max_wall_s"
        elif not self._has_live_worker() and not self._deployment_needs_workers():
            self.stop_reason = "no_live_workers"
        elif (elapsed >= self.args.window_s
              and all(r.finished_s is not None for r in self.all_rt + self.all_be)
              and not self._deployment_needs_workers()):
            self.stop_reason = "all_requests_finished"
        else:
            return False
        self._finished_elapsed_s = elapsed
        return True

    def _live_workers(self) -> list[WorkerState]:
        return [w for w in self.workers.values() if not w.retired]

    def _least_loaded(self, role: str) -> Optional[WorkerState]:
        workers = [
            w for w in self._live_workers() if w.role == role and w.ready
        ]
        if not workers:
            return None
        return min(workers, key=lambda w: len(w.active_attempts))

    def _requeue(self, req: SaturoRequest, phase: str) -> None:
        """Requeue an attempt after its Ray actor is retired."""
        if (req.is_finished or req.active_attempt_id is not None
                or self._has_terminal_failure()):
            return
        if req.kind == RT:
            if phase == DECODE and real_handoff(getattr(self, "args", None)):
                self._release_decode_reservation(req)
                phase = PREFILL
            if (phase == PREFILL
                    and getattr(self, "role_switcher", None) is None):
                worker = self._least_loaded(PREFILL)
                if worker is not None and self._assign(worker, req, PREFILL):
                    return
            if not hasattr(self, "pending_prefill"):
                self.pending_prefill = deque()
            queue = (self.pending_prefill if phase == PREFILL
                     else self.pending_decode)
            if not any(item is req for item in queue):
                queue.append(req)
            return
        candidates = [w for w in self._live_workers()
                      if w.ready and pool_accepts_be(w.role, self.args)]
        worker = min(candidates, key=lambda w: len(w.active_attempts),
                     default=None)
        if worker is not None:
            self._assign(worker, req, TPUT)
        elif not any(item is req for item in self.pending_be):
            self.pending_be.append(req)

    def _switch_worker_role(self, worker, role, *, borrowed, reason):
        """Commit the control-plane role only after a worker acknowledges."""
        if self._deployment_needs_workers():
            # Admission may continue for a controls-only refresh, but every
            # adaptive role writer must preserve its frozen worker layout.
            return False
        if (worker.retired or not worker.ready
                or not worker.role_switching_enabled):
            return False
        old_role = worker.role
        if old_role == role:
            return True
        if (worker.worker_id in getattr(self, "_worker_rpcs", {}).get("submit", {})
                or any(c.get("cmd") in CONTROL_COMMANDS
                       for c in worker.pending_commands)):
            # A delayed claim/release must execute under the role in which it
            # was issued. Drain RPCs carry no mutations and need not block it.
            return False
        try:
            reply = self._ensure_ray().get(
                worker.actor.switch_role.remote(role),
                timeout=getattr(self.args, "role_switch_timeout_s", 10.0))
        except Exception as exc:
            print(f"[scheduler] Role switch failed on {worker.worker_id}: "
                  f"{exc}; retiring and requeueing requests", flush=True)
            self._retire_worker(worker, requeue_active=True)
            return False
        if not reply.get("accepted") or reply.get("role") != role:
            return False
        worker.role = role
        worker.borrowed_role = borrowed
        event = {"time_s": self._elapsed(), "worker_id": worker.worker_id,
                 "from": old_role, "to": role, "reason": reason,
                 "config_name": worker.config_name}
        self.role_switcher.events.append(event)
        if hasattr(self, "_pool_layout_generation"):
            self._advance_layout_generation("adaptive_role_switch")
        print(f"[scheduler] Role switch: {event}", flush=True)
        return True

    def _request_by_id(self, request_id: str) -> Optional[SaturoRequest]:
        index = getattr(self, "_request_index", None)
        if index is not None:
            return index.get(request_id)
        for req in self.all_rt:
            if req.request_id == request_id:
                return req
        for req in self.all_be:
            if req.request_id == request_id:
                return req
        return None

    def _build_worker_plan(self) -> list[WorkerPlacement]:
        from saturo.resource_allocation.planner import SaturoResourcePlanner
        planner = SaturoResourcePlanner(
            self.args,
            lambda role, tp, pp, ep=0, gpu_id="": WorkerPlacement(
                role=role,
                host="",
                gpu_id=gpu_id,
                config_name=f"tp{tp}_pp{pp}" + (f"_ep{ep}" if ep else ""),
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                expert_parallel_size=ep,
            ),
        )
        placements = planner.build_worker_plan()
        from saturo.resource_allocation import load_profiles
        self._allocation_profiles = load_profiles(planner._profile_path())
        # Extract P-pool BE chunk latency from the resource profile so the
        # launcher can pass it to P-pool workers at runtime.
        self._p_pool_be_chunk_ms_map = self._extract_p_pool_be_chunk_ms(
            planner._profile_path())
        return placements

    def _extract_p_pool_be_chunk_ms(self,
                                    profile_path: Path) -> dict[str, float]:
        """Return {config_name: be_chunk_latency_ms} for P-pool configs."""
        result: dict[str, float] = {}
        try:
            if not profile_path.exists():
                return result
            data = json.loads(profile_path.read_text())
            p_pools = data.get("pools", {}).get("P", {})
            for config_name, pool_data in p_pools.items():
                value = pool_data.get("be_chunk_latency_ms")
                if value is not None:
                    result[config_name] = float(value)
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _summary(self) -> dict[str, Any]:
        elapsed_s = max(getattr(self, "_finished_elapsed_s", self._elapsed()), 1e-9)
        completed_rt = [r for r in self.all_rt if r.finished_s is not None
                        and r.observed_tokens >= r.output_len]
        completed_be = [r for r in self.all_be if r.finished_s is not None
                        and r.observed_tokens >= r.output_len]
        completed_be_in_window = [r for r in completed_be
                                  if r.finished_s <= self.args.window_s]
        switcher = getattr(self, "role_switcher", None)
        diagnostics = getattr(switcher, "admission_diagnostics", None)
        joint = [req.joint_slo_met(self.args.rt_ttft_slo_ms,
                                  self.args.rt_tbt_slo_ms) for req in self.all_rt]
        missing_joint = sum(value is None for value in joint)
        return {
            "rt_joint_attainment": (sum(value is True for value in joint) / len(joint)
                                    if joint and not missing_joint else None),
            "rt_joint_slo_status": ("missing_per_request_latency" if missing_joint
                                    else "measured" if joint else "no_rt"),
            "rt_joint_slo_missing_requests": missing_joint,
            "rt_joint_slo_scope": "all_replayed_rt_requests",
            "rt_tbt_definition": REQUEST_TBT_DEFINITION,
            "status": ("completed" if len(completed_rt) == len(self.all_rt)
                       and len(completed_be) == len(self.all_be)
                       and self.request_errors == 0
                       and not self._has_terminal_failure() else "incomplete"),
            "stop_reason": self.stop_reason,
            "request_errors": self.request_errors,
            "engine_failure": getattr(self, "engine_failure", None),
            "request_failure": getattr(self, "request_failure", None),
            "history_updates": self.history_updates,
            "pool_layout_generation": getattr(self, "_pool_layout_generation", 0),
            "runtime_update_events": dict(getattr(self, "_runtime_event_counts", {})),
            "history_update_protocol": (self.history_controller.status()
                if getattr(self, "history_controller", None) is not None else None),
            "load_format": self.args.load_format,
            "kv_handoff": ("real_shared_storage" if real_handoff(getattr(self, "args", None)) else "length_only_dummy_kv"),
            "kv_transfer_namespace": getattr(self, "_kv_transfer_namespace", None),
            "enable_prefill_saturation": getattr(self.args, "enable_prefill_saturation", True),
            "enable_decode_saturation": getattr(self.args, "enable_decode_saturation", True),
            "global_admission": (diagnostics() if callable(diagnostics) else
                                 {"length_source": "rt_slot_count", "uses_trace_output_len": False}),
            "workload_mode": (
                "tokenized_prompt_replay" if any(r.prompt_token_ids is not None
                                                 for r in self.all_rt + self.all_be)
                else "length_only_trace_replay"),
            "workload_inputs": {
                kind: {"path": getattr(self.args, option, None), "requests": len(rows),
                       "tokenized_requests": sum(r.prompt_token_ids is not None for r in rows)}
                for kind, option, rows in ((RT, "rt_workload_csv", self.all_rt),
                                           (BE, "be_workload_csv", self.all_be))},
            "strategy":
            "saturo_global_scheduler",
            "model":
            self.args.model,
            "num_req":
            self.args.num_req,
            "window_s":
            self.args.window_s,
            "elapsed_s":
            elapsed_s,
            "target_rps":
            self.args.target_rps,
            "rt_requests": len(self.all_rt),
            "rt_submitted": sum(req.prefill_attempts > 0 for req in self.all_rt),
            "rt_generated_tokens": self.rt_tokens,
            "rt_completed":
            len(completed_rt),
            "rt_tbt_ms_p50":
            percentile(self.rt_tbt_samples_ms, 0.50),
            "rt_tbt_ms_p95":
            percentile(self.rt_tbt_samples_ms, 0.95),
            "rt_tbt_ms_p99": percentile(self.rt_tbt_samples_ms, 0.99),
            "rt_tbt_ms_max": max(self.rt_tbt_samples_ms, default=0.0),
            "rt_tbt_samples": len(self.rt_tbt_samples_ms),
            "rt_tbt_slo_violation_count": self.rt_tbt_violations,
            "rt_tbt_slo_violation_rate":
            (self.rt_tbt_violations / max(len(self.rt_tbt_samples_ms), 1)),
            "rt_ttft_ms_p95":
            percentile(self.rt_ttft_samples_ms, 0.95),
            "rt_ttft_ms_max": max(self.rt_ttft_samples_ms, default=0.0),
            "rt_ttft_slo_ms": self.args.rt_ttft_slo_ms,
            "rt_tbt_slo_ms": self.args.rt_tbt_slo_ms,
            "rt_ttft_slo_violation_rate": self.rt_ttft_violations / max(len(self.all_rt), 1),
            "be_requests":
            len(self.all_be),
            "be_completed":
            len(completed_be),
            "be_completed_in_window": len(completed_be_in_window),
            "be_throughput_rps":
            (len(completed_be_in_window) / max(self.args.window_s, 1e-9)),
            "be_total_throughput_rps": len(completed_be) / elapsed_s,
            "be_generated_tokens":
            self.be_tokens,
            "be_tokens_in_window":
            self.be_tokens_in_window,
            "prefill_handoffs":
            self.prefill_handoffs,
            "role_switches":
            (len(self.role_switcher.events)
             if getattr(self, "role_switcher", None) else 0),
            "redis_trace_key":
            self.trace_recorder.store.key if self.trace_recorder else None,
            "trace_dropped_records":
            self.trace_recorder.dropped_records if self.trace_recorder else 0,
            "trace_failed_records":
            self.trace_recorder.failed_records if self.trace_recorder else 0,
        }

    def _write_outputs(self) -> None:
        with (self.output_dir / "role_switches.json").open("w") as f:
            switcher = getattr(self, "role_switcher", None)
            json.dump(switcher.events if switcher else [], f, indent=2)
        with (self.output_dir / "summary.json").open("w") as f:
            json.dump(self._summary(), f, indent=2)
        with (self.output_dir / "requests.csv").open("w", newline="") as f:
            fields = [
                "request_id",
                "kind",
                "prompt_len",
                "output_len",
                "arrival_offset_s",
                "prefill_submitted_s",
                "prefill_finished_s",
                "decode_submitted_s",
                "first_token_s",
                "finished_s",
                "ttft_ms",
                "tbt_ms",
                "tbt_samples",
                "tbt_unobserved_intervals",
                "tbt_complete",
                "tbt_definition",
                "joint_slo_met",
                "observed_tokens",
                "generated_token_ids",
                "source_id",
                "dataset",
                "azure_prompt_len",
                "azure_output_len",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for req in self.all_rt + self.all_be:
                writer.writerow({
                    "request_id": req.request_id,
                    "kind": req.kind,
                    "prompt_len": req.prompt_len,
                    "output_len": req.output_len,
                    "arrival_offset_s": req.arrival_offset_s,
                    "prefill_submitted_s": req.prefill_submitted_s,
                    "prefill_finished_s": req.prefill_finished_s,
                    "decode_submitted_s": req.decode_submitted_s,
                    "first_token_s": req.first_token_s,
                    "finished_s": req.finished_s,
                    "ttft_ms": req.ttft_ms,
                    "tbt_ms": req.tbt_ms,
                    "tbt_samples": req.tbt_samples,
                    "tbt_unobserved_intervals": req.tbt_unobserved_intervals,
                    "tbt_complete": req.tbt_complete,
                    "tbt_definition": REQUEST_TBT_DEFINITION,
                    "joint_slo_met": (req.joint_slo_met(self.args.rt_ttft_slo_ms,
                                                       self.args.rt_tbt_slo_ms)
                                      if req.kind == RT else None),
                    "observed_tokens": req.observed_tokens,
                    "generated_token_ids": json.dumps(req.generated_token_ids),
                    "source_id": req.source_id,
                    "dataset": req.dataset,
                    "azure_prompt_len": req.azure_prompt_len,
                    "azure_output_len": req.azure_output_len,
                })
