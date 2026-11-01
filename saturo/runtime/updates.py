"""Versioned live observations and ordered P/D control publication."""
from __future__ import annotations

import json
import time
from collections import Counter

from saturo.runtime.workload import DECODE, PREFILL, RT


class RuntimeUpdatesMixin:
    def _runtime_event(self, event, **fields):
        if not hasattr(self, "_runtime_event_counts"):
            self._runtime_event_counts = Counter()
        self._runtime_event_counts[event] += 1
        row = {"event": event, "elapsed_s": self._elapsed(),
               "pool_layout_generation": getattr(self, "_pool_layout_generation", 0),
               **fields}
        if getattr(self, "output_dir", None) is not None:
            with (self.output_dir / "runtime_updates.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def _init_runtime_updates(self):
        from saturo.runtime.history import RuntimeHistoryController
        self._history_rollout = None
        self._epoch_pending = {}
        self._observed_runtime_arrivals = set()
        self.history_controller = RuntimeHistoryController(
            self.args, self._history_worker_specs(),
            run_id=getattr(self.trace_recorder, "run_id", None))
        self._runtime_event("update_protocol_started",
            interval_s=getattr(self.args, "saturation_update_interval_s", 40),
            history_window_s=getattr(self.args, "saturation_history_window_s", 600),
            decode_async=getattr(self.args, "decode_target_async", False))

    def _observe_runtime_arrival(self, req):
        observed = getattr(self, "_observed_runtime_arrivals", None)
        if observed is None or req.request_id in observed:
            return
        observed.add(req.request_id)
        if self.history_controller is not None:
            self.history_controller.record_arrival(
                req.request_id, req.prompt_len,
                self._source_arrival_epoch_s + req.arrival_offset_s)

    def _observe_runtime_completion(self, req):
        if (req.kind == RT and req.finished_s is not None
                and req.observed_tokens >= req.output_len
                and req.request_id in getattr(self, "_observed_runtime_arrivals", ())
                and self.history_controller is not None):
            self.history_controller.record_completion(
                req.request_id, req.observed_tokens,
                self._source_arrival_epoch_s + req.finished_s)

    def _advance_layout_generation(self, reason):
        self._pool_layout_generation = getattr(self, "_pool_layout_generation", 0) + 1
        if getattr(self, "_history_rollout", None) is not None:
            self._runtime_event("history_rollout_invalidated", reason=reason,
                                version=self._history_rollout["version"])
        self._history_rollout = None
        self._epoch_pending = {}
        for worker in self._live_workers():
            if not worker.ready:
                continue
            worker.pending_commands = [c for c in worker.pending_commands
                if c.get("cmd") not in ("history_prepare", "history_commit", "update_epoch")]
            worker.pending_commands.append({"cmd": "update_epoch",
                "pool_layout_generation": self._pool_layout_generation})
            self._epoch_pending[worker.worker_id] = self._pool_layout_generation
        if getattr(self, "history_controller", None) is not None:
            self.history_controller.poll(self._history_worker_specs(),
                pool_layout_generation=self._pool_layout_generation)
            self.history_controller.request_refresh()
        self._runtime_event("layout_generation_advanced", reason=reason)
        return self._pool_layout_generation

    def _abort_history_rollout(self, reason):
        rollout = getattr(self, "_history_rollout", None)
        if rollout is None:
            return
        self._runtime_event("history_rollout_aborted", version=rollout["version"],
                            phase=rollout["phase"], reason=reason)
        self._history_rollout = None
        if rollout["phase"] == "committing":
            # Some actors may already have applied the new bundle. Invalidate
            # D targets until a fresh, all-acknowledged epoch is installed.
            self._advance_layout_generation("history_commit_incomplete")
        elif self.history_controller is not None:
            self.history_controller.request_refresh()

    def _poll_history_update_protocol(self):
        controller = self.history_controller
        if controller is None:
            return
        rollout = getattr(self, "_history_rollout", None)
        if rollout is not None:
            if time.monotonic() >= rollout["deadline"]:
                self._abort_history_rollout("ack_timeout")
            return
        if (getattr(self, "_epoch_pending", {})
                or self._deployment_needs_workers()):
            return
        update = controller.poll(self._history_worker_specs(),
            pool_layout_generation=getattr(self, "_pool_layout_generation", 0))
        if update is None:
            return
        if update["pool_layout_generation"] != getattr(self, "_pool_layout_generation", 0):
            self._runtime_event("history_bundle_stale", version=update["version"])
            return
        workers = [w for w in self._live_workers() if w.ready]
        if not workers:
            return
        commands = []
        for worker in workers:
            cmd = {"cmd": "history_prepare", "version": update["version"],
                   "pool_layout_generation": update["pool_layout_generation"]}
            if (worker.role == PREFILL
                    and getattr(self.args, "enable_prefill_saturation", True)):
                budget = update["p_budgets"].get(worker.config_name)
                if budget is None:
                    self._runtime_event("history_bundle_rejected", version=update["version"],
                                        reason="missing_p_budget")
                    return
                cmd["p_chunk_ms"] = budget["chunk_ms"]
            elif worker.role == DECODE:
                cmd["statistics"] = update["decode_statistics"]
            commands.append((worker, cmd))
        self._history_rollout = {"version": update["version"],
            "epoch": update["pool_layout_generation"], "phase": "preparing",
            "pending": {w.worker_id for w in workers}, "workers": workers,
            "bundle": update, "deadline": time.monotonic() + getattr(self.args, "saturation_update_timeout_s", 30)}
        for worker, cmd in commands:
            worker.pending_commands.append(cmd)
        self._runtime_event("history_prepare_started", version=update["version"],
                            history_end_s=update["history_end_s"], workers=sorted(self._history_rollout["pending"]))

    def _acknowledge_runtime_controls(self, worker, commands, replies):
        if len(commands) != len(replies):
            raise RuntimeError("Worker returned an incomplete command acknowledgement")
        for command, reply in zip(commands, replies):
            operation = command.get("cmd")
            if operation == "update_epoch":
                expected = getattr(self, "_epoch_pending", {}).get(worker.worker_id)
                if expected != command["pool_layout_generation"]:
                    continue
                if not reply.get("accepted") or reply.get("pool_layout_generation") != expected:
                    raise RuntimeError("Worker rejected the current layout epoch")
                self._epoch_pending.pop(worker.worker_id, None)
                self._runtime_event("layout_epoch_ack", worker_id=worker.worker_id, epoch=expected)
                continue
            if operation not in ("history_prepare", "history_commit"):
                continue
            rollout = getattr(self, "_history_rollout", None)
            if (rollout is None or rollout["version"] != command["version"]
                    or rollout["epoch"] != command["pool_layout_generation"]):
                continue
            expected_operation = "history_prepare" if rollout["phase"] == "preparing" else "history_commit"
            if operation != expected_operation:
                continue
            if (not reply.get("accepted") or reply.get("version") != rollout["version"]
                    or reply.get("pool_layout_generation") != rollout["epoch"]):
                self._abort_history_rollout(str(reply.get("reason", "invalid_ack")))
                continue
            rollout["pending"].discard(worker.worker_id)
            self._runtime_event("history_control_ack", worker_id=worker.worker_id,
                operation=operation, version=rollout["version"])
            if rollout["pending"]:
                continue
            if rollout["phase"] == "preparing":
                rollout["phase"] = "committing"
                rollout["pending"] = {w.worker_id for w in rollout["workers"]}
                rollout["deadline"] = time.monotonic() + getattr(self.args, "saturation_update_timeout_s", 30)
                for target in rollout["workers"]:
                    target.pending_commands.append({"cmd": "history_commit",
                        "version": rollout["version"], "pool_layout_generation": rollout["epoch"]})
                self._runtime_event("history_commit_started", version=rollout["version"])
            else:
                self.history_updates += 1
                for name, budget in rollout["bundle"]["p_budgets"].items():
                    self._p_pool_be_chunk_ms_map[name] = budget["chunk_ms"]
                with (self.output_dir / "history_updates.jsonl").open("a") as stream:
                    stream.write(json.dumps(rollout["bundle"]) + "\n")
                self._runtime_event("history_bundle_committed", version=rollout["version"])
                self._history_rollout = None
