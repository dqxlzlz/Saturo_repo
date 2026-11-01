"""Nonblocking main-thread executor for periodic allocation proposals.

The scheduler continues observing ingress while this mixin quiesces only new
P/BE assignments. Existing P→D handoffs and resident BE finish on their current
engines. No request or partially generated KV state is migrated or replayed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from saturo.resource_allocation import load_profiles
from saturo.runtime.pool_controls import pool_accepts_be
from saturo.runtime.reallocation import (DeploymentTransaction, DrainCensus,
    PeriodicReallocationService, PlacementSpec, ReallocationProposal)


def _cost(worker: Any) -> int:
    return int(worker.tensor_parallel_size) * int(worker.pipeline_parallel_size)


def _spec(worker: Any) -> PlacementSpec:
    return PlacementSpec(str(worker.role), int(worker.tensor_parallel_size),
        int(worker.pipeline_parallel_size), int(getattr(worker, "expert_parallel_size", 0)),
        str(getattr(worker, "gpu_id", "")), str(worker.config_name))


def _signature(item: PlacementSpec) -> tuple:
    return (item.tensor_parallel_size, item.pipeline_parallel_size,
            item.expert_parallel_size, item.gpu_id)


def _profile_gpu_name(path: str) -> str:
    try:
        payload = json.loads(Path(path).read_text())
        model = payload.get("time_predictor", payload)
        identity = model.get("profile_identity") or model.get("meta", {}).get("profile_identity") or {}
        name = str(identity.get("gpu_name", "")).strip()
    except (ValueError, OSError, TypeError) as exc:
        raise ValueError(f"cannot determine allocation hardware from worker timing profile {path}: {exc}") from exc
    if not name or name.lower() == "unknown":
        raise ValueError(f"worker timing profile has no GPU identity: {path}")
    return name


@dataclasses.dataclass
class _Deployment:
    proposal: ReallocationProposal
    tx: DeploymentTransaction
    old_workers: list[Any]
    old_specs: tuple[PlacementSpec, ...]
    old_profiles: Any
    old_p_budgets: dict[str, float]
    profiles: Any
    p_budgets: dict[str, float]
    targets: dict[int, PlacementSpec]
    release_available: float
    old_active_plan: bytes | None = None
    commit_started: bool = False
    old_actors: dict[int, Any] = dataclasses.field(default_factory=dict)
    rpc_refs: dict[Any, tuple[Any, str, Any]] = dataclasses.field(default_factory=dict)
    lingering_refs: dict[Any, tuple[Any, str, Any]] = dataclasses.field(default_factory=dict)
    starts: list[Any] = dataclasses.field(default_factory=list)
    shutdown_workers: list[Any] = dataclasses.field(default_factory=list)
    killed: set[int] = dataclasses.field(default_factory=set)
    shutdown_deadline: float = 0
    failure: str | None = None
    role_stage: str = ""
    start_kind: str = ""
    stop_kind: str = ""
    last_census: DrainCensus | None = None


class PeriodicDeploymentMixin:
    """Glue for scheduler/launcher hooks; every Ray get follows wait(timeout=0)."""

    def _deployment_clock(self) -> float:
        return time.monotonic()

    def _deployment_event(self, event: str, **fields: Any) -> None:
        self._runtime_event("resource_" + event, **fields)

    def _init_periodic_reallocation(self) -> None:
        self._periodic_deployment: _Deployment | None = None
        self._deployment_wait_epoch = False
        self._reallocation_service = None
        if not getattr(self.args, "periodic_reallocation", False):
            return
        if self.history_controller is None:
            raise ValueError("periodic reallocation requires a live history controller")
        args = argparse.Namespace(**vars(self.args))
        if int(getattr(args, "total_gpus", 0) or 0) <= 0:
            args.total_gpus = sum(_cost(w) for w in self._live_workers())
        timing_profiles = {}
        gpu_names = set()
        for worker in self._live_workers():
            path = str(getattr(worker, "runtime_profile_path", ""))
            if not path:
                raise ValueError("periodic reallocation requires measured timing profiles for every current worker")
            gpu_names.add(_profile_gpu_name(path))
            timing_profiles.setdefault(worker.config_name, path)
        if len(gpu_names) != 1:
            raise ValueError("periodic reallocation does not support mixed/unknown worker GPU timing hardware")
        gpu_name = next(iter(gpu_names))
        explicit_gpu = str(getattr(args, "resource_reallocation_gpu_name", "") or "")
        if explicit_gpu and explicit_gpu != gpu_name:
            raise ValueError("explicit reallocation GPU constraint differs from actual worker timing hardware")
        args.resource_reallocation_gpu_name = gpu_name
        self._reallocation_service = PeriodicReallocationService(
            args, self.history_controller.snapshot, timing_profiles=timing_profiles)
        self._deployment_event("service_started", interval_s=self._reallocation_service.interval_s,
            history_window_s=self._reallocation_service.history_window_s,
            total_gpus=args.total_gpus, timing_profiles=timing_profiles, gpu_name=gpu_name)

    def _deployment_pauses_new_work(self) -> bool:
        context = getattr(self, "_periodic_deployment", None)
        return ((context is not None and not context.tx.controls_only) or
                bool(getattr(self, "_deployment_wait_epoch", False)))

    def _deployment_needs_workers(self) -> bool:
        """A stopped old layout is an intentional phase, not workload completion."""
        return (getattr(self, "_periodic_deployment", None) is not None or
                bool(getattr(self, "_deployment_wait_epoch", False)))

    def _deployment_census(self) -> DrainCensus:
        workers = self._live_workers()
        context = getattr(self, "_periodic_deployment", None)
        retain = bool(context is not None and context.tx.in_place)
        blocked = retained = 0
        lookup = getattr(self, "_request_by_id", lambda request_id: None)
        for worker in workers:
            preemptions = set(getattr(worker, "preempting_attempts", ()))
            attempts = set(worker.active_attempts) | preemptions
            for attempt in attempts:
                request = lookup(worker.active_attempts.get(attempt, ""))
                target = context.targets.get(worker.worker_id) if retain else None
                if (retain and attempt not in preemptions and request is not None
                        and getattr(request, "kind", None) == "be" and target is not None
                        and pool_accepts_be(target.role, self.args)):
                    retained += 1
                else:
                    # Unknown request kinds and all preemptions stay blocking.
                    blocked += 1
        # Unassigned P/BE requests remain in their original queues. Pending D
        # is already partially generated and must finish before engine stop.
        return DrainCensus(
            active_attempts=blocked,
            local_pending_requests=(sum(not r.is_finished for r in self.pending_decode) +
                sum(not r.is_finished and (getattr(r, "observed_tokens", 0) > 0 or
                    getattr(r, "decode_attempts", 0) > 0) for r in self.pending_be)),
            queued_commands=sum(len(w.pending_commands) for w in workers),
            inflight_rpcs=sum(len(table) for kind, table in getattr(self, "_worker_rpcs", {}).items()
                              if not (retain and kind == "drain")),
            reservations=sum(len(w.decode_reservations) for w in workers),
            retained_local_be=retained)

    def _deployment_prepare(self, proposal: ReallocationProposal) -> None:
        """Validate all staged artifacts before pausing ingress assignment."""
        proposal.verify()
        profiles = load_profiles(proposal.profile_path)
        plan = proposal.plan()
        controls = plan.get("deployment_controls", {}).get("p_budgets", {})
        p_budgets = {}
        for name, entry in profiles.pools.get("P", {}).items():
            value = controls.get(name, {}).get("chunk_ms", entry.be_chunk_latency_ms)
            if value is not None and math.isfinite(float(value)) and float(value) > 0:
                p_budgets[name] = float(value)
        old_workers = list(self._live_workers())
        old_specs = tuple(_spec(w) for w in old_workers)
        same_engines = (len(old_specs) == len(proposal.placements) and
            Counter(map(_signature, old_specs)) == Counter(map(_signature, proposal.placements)))
        budget = int(getattr(self.args, "total_gpus", 0) or sum(_cost(w) for w in old_workers))
        targets = {}
        if same_engines:
            remaining = list(proposal.placements)
            # Match existing engine/GPU first, preserving roles where possible.
            for worker, old in zip(old_workers, old_specs):
                candidates = [p for p in remaining if _signature(p) == _signature(old)]
                target = min(candidates, key=lambda p: p.role != worker.role)
                remaining.remove(target)
                targets[worker.worker_id] = target
        unchanged_roles = same_engines and all(targets[w.worker_id].role == w.role and
            targets[w.worker_id].config_name == w.config_name for w in old_workers)
        can_switch = all(w.role_switching_enabled for w in old_workers)
        in_place = same_engines and all(w.ready for w in old_workers) and (unchanged_roles or can_switch)
        disabled_reason = ("role_switching_not_supported" if same_engines and
                           not unchanged_roles and not can_switch else None)
        if not in_place:
            targets = {}
        if in_place:
            lookup = getattr(self, "_request_by_id", lambda request_id: None)
            for worker in old_workers:
                target = targets[worker.worker_id]
                if pool_accepts_be(target.role, self.args):
                    continue
                if any(getattr(lookup(request_id), "kind", None) == "be"
                       for request_id in worker.active_attempts.values()):
                    in_place = False
                    targets = {}
                    disabled_reason = "target_excludes_resident_be"
                    break
        controls_only = in_place and unchanged_roles
        tx = DeploymentTransaction(proposal,
            current_generation=int(self._pool_layout_generation), gpu_budget=budget,
            drain_timeout_s=float(getattr(self.args, "resource_reallocation_drain_timeout_s", 180)),
            start_timeout_s=float(getattr(self.args, "worker_start_timeout_s", 300)),
            clock=self._deployment_clock, in_place=in_place, old_placements=old_specs,
            controls_only=controls_only)
        ray = self._ensure_ray()
        available = float(ray.available_resources().get("GPU", 0))
        old_budgets = dict(self._p_pool_be_chunk_ms_map)
        resolver = getattr(self, "_resolve_p_pool_be_chunk_ms", lambda name: 100.0)
        for worker in old_workers:
            old_budgets.setdefault(worker.config_name, float(resolver(worker.config_name)))
        context = _Deployment(proposal, tx, old_workers, old_specs,
            self._allocation_profiles, old_budgets, profiles,
            p_budgets, targets, available + sum(_cost(w) for w in old_workers))
        active_path = Path(self.output_dir) / "active_resource_plan.json"
        context.old_active_plan = active_path.read_bytes() if active_path.exists() else None
        context.old_actors = {worker.worker_id: worker.actor for worker in old_workers}
        # Claims attached to buffered, never-started P work must not prevent
        # draining forever; release through the normal acknowledged channel.
        if not controls_only:
            for req in self.pending_prefill:
                if req.active_attempt_id is None:
                    self._release_decode_reservation(req)
        self._periodic_deployment = context
        self._deployment_event("controls_refresh" if controls_only else "quiesce", proposal_id=proposal.proposal_id,
            base_generation=proposal.base_generation, in_place=in_place,
            controls_only=controls_only, pauses_new_work=not controls_only,
            in_place_disabled_reason=disabled_reason,
            old_placements=[dataclasses.asdict(p) for p in old_specs],
            desired_placements=proposal.placement_dicts(),
            buffered_prefill=len(self.pending_prefill), buffered_be=len(self.pending_be))
        if controls_only:
            context.role_stage = "info"
            for worker in old_workers:
                try:
                    ref = worker.actor.info.remote()
                    context.rpc_refs[ref] = (worker, "info", worker.role)
                except Exception as exc:
                    context.failure = f"info dispatch worker={worker.worker_id}: {exc}"

    def _deployment_ready_refs(self, refs: dict) -> list:
        if not refs:
            return []
        ray = self._ensure_ray()
        ready, _ = ray.wait(list(refs), num_returns=len(refs), timeout=0)
        return ready

    def _deployment_layout_error(self, context: _Deployment) -> str | None:
        if self._pool_layout_generation != context.proposal.base_generation:
            return "layout generation changed during live in-place update"
        current = {worker.worker_id: worker for worker in self._live_workers()}
        expected = {worker.worker_id for worker in context.old_workers}
        if set(current) != expected:
            return f"live worker set changed during in-place update: expected={sorted(expected)}, actual={sorted(current)}"
        for original, spec in zip(context.old_workers, context.old_specs):
            worker = current[original.worker_id]
            if (worker is not original or worker.retired or not worker.ready or
                    worker.actor is not context.old_actors[worker.worker_id] or _spec(worker) != spec):
                return f"worker identity/readiness/config changed during in-place update: {worker.worker_id}"
        return None

    def _deployment_poll_rpcs(self, context: _Deployment, *, lingering=False) -> None:
        refs = context.lingering_refs if lingering else context.rpc_refs
        ray = self._ensure_ray()
        for ref in self._deployment_ready_refs(refs):
            worker, operation, expected = refs.pop(ref)
            try:
                result = ray.get(ref, timeout=0)
                if operation == "role":
                    if not result.get("accepted") or result.get("role") != expected:
                        raise RuntimeError(f"role rejected: {result}")
                elif operation == "info":
                    if result.get("worker_id") != worker.worker_id or result.get("role") != expected:
                        raise RuntimeError(f"unchanged-layout verification failed: {result}")
                elif operation == "budget":
                    if len(result) != 1 or not result[0].get("accepted"):
                        raise RuntimeError(f"P budget rejected: {result}")
                elif operation == "start":
                    devices = [part for part in str(result.get("gpu_id", "")).split(",") if part.strip()]
                    if len(devices) != _cost(worker):
                        raise RuntimeError("worker actual GPU allocation differs from requested width")
                    self._finish_worker_start(worker, result)
                self._deployment_event("ack", proposal_id=context.proposal.proposal_id,
                    phase=context.tx.phase, worker_id=worker.worker_id, operation=operation,
                    expected=expected, reply=result, lingering=lingering)
            except Exception as exc:
                self._deployment_event("rpc_failed", proposal_id=context.proposal.proposal_id,
                    phase=context.tx.phase, worker_id=worker.worker_id, operation=operation,
                    error=repr(exc), lingering=lingering)
                if not lingering:
                    context.failure = f"{operation} worker={worker.worker_id}: {exc}"

    def _deployment_send_roles(self, context: _Deployment, *, restore: bool) -> None:
        context.role_stage = "roles"
        specs = {w.worker_id: p for w, p in zip(context.old_workers, context.old_specs)} if restore else context.targets
        for worker in context.old_workers:
            target = specs[worker.worker_id]
            # Send even unchanged roles: an ACK proves the actor's current
            # role after a partially failed previous transition.
            try:
                ref = worker.actor.switch_role.remote(target.role)
                context.rpc_refs[ref] = (worker, "role", target.role)
            except Exception as exc:
                context.failure = f"role dispatch worker={worker.worker_id}: {exc}"
        self._deployment_event("roles_dispatched", proposal_id=context.proposal.proposal_id,
                               restore=restore, count=len(context.rpc_refs))

    def _deployment_send_budgets(self, context: _Deployment, *, restore: bool,
                                 pre_restore=False) -> None:
        context.role_stage = "rollback_prebudgets" if pre_restore else "budgets"
        mapping = context.old_p_budgets if restore else context.p_budgets
        specs = ({w.worker_id: p for w, p in zip(context.old_workers, context.old_specs)}
                 if restore and not pre_restore else context.targets)
        for worker in context.old_workers:
            target = specs[worker.worker_id]
            if target.role != "prefill" or target.config_name not in mapping:
                continue
            value = mapping[target.config_name]
            try:
                ref = worker.actor.submit_batch.remote([{"cmd": "history", "p_chunk_ms": value}])
                context.rpc_refs[ref] = (worker, "budget", value)
            except Exception as exc:
                context.failure = f"budget dispatch worker={worker.worker_id}: {exc}"

    def _deployment_role_progress(self, context: _Deployment, *, restore=False) -> bool:
        # Do not allow an old delayed role/budget RPC to execute after a
        # successful restore. Wait for every original future to settle first.
        if restore and context.lingering_refs:
            self._deployment_poll_rpcs(context, lingering=True)
            if context.lingering_refs:
                return False
            self._deployment_send_budgets(context, restore=True, pre_restore=True)
        self._deployment_poll_rpcs(context)
        if context.rpc_refs or context.failure:
            return False
        if context.role_stage == "await_lingering":
            self._deployment_send_budgets(context, restore=True, pre_restore=True)
            return False
        if context.role_stage == "rollback_prebudgets":
            # Reset the cached P budget while actors that temporarily became
            # P are still in that role; then restore roles and original P
            # controls. Future adaptive borrowing must not inherit new-budget
            # residue from a transaction that rolled back.
            self._deployment_send_roles(context, restore=True)
            return False
        if context.role_stage == "roles":
            self._deployment_send_budgets(context, restore=restore)
            return not context.rpc_refs and not context.failure
        return context.role_stage == "budgets"

    def _deployment_controls_progress(self, context: _Deployment, *, restore=False) -> bool:
        if restore and context.lingering_refs:
            self._deployment_poll_rpcs(context, lingering=True)
            if context.lingering_refs:
                return False
        if restore and context.role_stage == "await_lingering":
            self._deployment_send_budgets(context, restore=True)
        self._deployment_poll_rpcs(context)
        if context.rpc_refs or context.failure:
            return False
        if context.role_stage == "info":
            self._deployment_send_budgets(context, restore=False)
            return not context.rpc_refs and not context.failure
        return context.role_stage == "budgets"

    def _deployment_stop(self, context: _Deployment, workers: list, kind: str) -> None:
        context.stop_kind = kind
        context.shutdown_workers = list(workers)
        context.rpc_refs = {}
        context.killed = set()
        context.shutdown_deadline = self._deployment_clock() + min(5.0,
            float(getattr(self.args, "worker_start_timeout_s", 300)))
        for worker in workers:
            worker.ready = False
            try:
                ref = worker.actor.shutdown.remote()
                context.rpc_refs[ref] = (worker, "shutdown", None)
            except Exception:
                self._deployment_kill(context, worker)
        self._deployment_event("shutdown_dispatched", proposal_id=context.proposal.proposal_id,
            kind=kind, worker_ids=[w.worker_id for w in workers],
            release_available=context.release_available)

    def _deployment_kill(self, context: _Deployment, worker: Any) -> None:
        if worker.worker_id in context.killed:
            return
        try:
            self._ensure_ray().kill(worker.actor, no_restart=True)
            context.killed.add(worker.worker_id)
            worker.ready = False
            worker.retired = True
            self._drop_worker_rpc_refs(worker.worker_id)
            try:
                worker.stderr_file.close()
            except Exception:
                pass
            self._deployment_event("actor_killed", proposal_id=context.proposal.proposal_id,
                                   worker_id=worker.worker_id, kind=context.stop_kind)
        except Exception as exc:
            context.failure = f"actor kill worker={worker.worker_id}: {exc}"

    def _deployment_stop_progress(self, context: _Deployment) -> bool:
        ray = self._ensure_ray()
        for ref in self._deployment_ready_refs(context.rpc_refs):
            worker, _, _ = context.rpc_refs.pop(ref)
            try:
                ray.get(ref, timeout=0)
                self._deployment_event("shutdown_ack", proposal_id=context.proposal.proposal_id,
                                       worker_id=worker.worker_id)
            except Exception as exc:
                self._deployment_event("shutdown_rpc_failed", proposal_id=context.proposal.proposal_id,
                                       worker_id=worker.worker_id, error=repr(exc))
            self._deployment_kill(context, worker)
        if self._deployment_clock() >= context.shutdown_deadline:
            for ref, (worker, _, _) in list(context.rpc_refs.items()):
                context.rpc_refs.pop(ref)
                self._deployment_kill(context, worker)
        if any(w.worker_id not in context.killed for w in context.shutdown_workers):
            return False
        available = float(ray.available_resources().get("GPU", 0))
        if available + 1e-6 < context.release_available:
            return False
        self._deployment_event("gpu_release_verified", proposal_id=context.proposal.proposal_id,
            kind=context.stop_kind, available_gpus=available,
            required_available_gpus=context.release_available)
        return True

    def _deployment_start(self, context: _Deployment, *, old: bool) -> None:
        from saturo.runtime.worker import WorkerPlacement
        context.start_kind = "old" if old else "new"
        context.starts = []
        context.rpc_refs = {}
        specs = context.old_specs if old else context.proposal.placements
        budgets = context.old_p_budgets if old else context.p_budgets
        hosts = [str(w.host) for w in context.old_workers] if old else [""] * len(specs)
        # Launcher consults this staging map only while creating actor args;
        # the active profile/budget map is unchanged until the final commit.
        self._deployment_start_p_budgets = budgets
        before_ids = set(self.workers)
        try:
            for spec, host in zip(specs, hosts):
                placement = WorkerPlacement(role=spec.role, host=host, gpu_id=spec.gpu_id,
                    config_name=spec.config_name, tensor_parallel_size=spec.tensor_parallel_size,
                    pipeline_parallel_size=spec.pipeline_parallel_size,
                    expert_parallel_size=spec.expert_parallel_size)
                state, ref = self._begin_worker_start(placement)
                context.starts.append(state)
                context.rpc_refs[ref] = (state, "start", spec.role)
        except Exception as exc:
            context.failure = f"{context.start_kind} actor allocation: {exc}"
            # The launcher can have registered an actor before its info RPC
            # dispatch raises. Include it in cleanup even without a return.
            tracked = {w.worker_id for w in context.starts}
            context.starts.extend(w for wid, w in self.workers.items()
                if wid not in before_ids and wid not in tracked)
        finally:
            del self._deployment_start_p_budgets
        self._deployment_event("start_dispatched", proposal_id=context.proposal.proposal_id,
            kind=context.start_kind, worker_ids=[w.worker_id for w in context.starts],
            requested_gpus=sum(_cost(w) for w in context.starts), p_budgets=budgets)

    def _deployment_commit(self, context: _Deployment) -> None:
        context.commit_started = True
        if context.tx.in_place:
            for worker in context.old_workers:
                worker.role = context.targets[worker.worker_id].role
                worker.borrowed_role = False
        self._allocation_profiles = context.profiles
        self._p_pool_be_chunk_ms_map = dict(context.p_budgets)
        switcher = getattr(self, "role_switcher", None)
        if switcher is not None:
            switcher.profiles = context.profiles
            switcher._predictors.clear()
        generation = self._advance_layout_generation("resource_reallocation_commit")
        self._deployment_wait_epoch = True
        record = {**context.proposal.plan(), "committed_generation": generation,
            "manifest_path": context.proposal.manifest_path,
            "manifest_sha256": context.proposal.manifest_sha256,
            "actual_placements": [dataclasses.asdict(_spec(w)) for w in self._live_workers()],
            "committed_time_s": time.time(), "p_budgets": context.p_budgets}
        path = Path(self.output_dir) / "active_resource_plan.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True))
        temporary.replace(path)
        self._deployment_event("committed", proposal_id=context.proposal.proposal_id,
            generation=generation, active_plan=str(path),
            active_plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            manifest_sha256=context.proposal.manifest_sha256,
            in_place=context.tx.in_place, actual_placements=record["actual_placements"],
            controls_only=context.tx.controls_only,
            p_budgets=context.p_budgets)
        self._periodic_deployment = None

    def _deployment_finish_old(self, context: _Deployment) -> None:
        # Roles in control-plane state were never changed before commit;
        # actor rollback has now acknowledged every original role/budget.
        for worker, spec in zip(context.old_workers, context.old_specs):
            if not worker.retired:
                worker.role = spec.role
        self._allocation_profiles = context.old_profiles
        self._p_pool_be_chunk_ms_map = dict(context.old_p_budgets)
        switcher = getattr(self, "role_switcher", None)
        if switcher is not None:
            switcher.profiles = context.old_profiles
            switcher._predictors.clear()
        if context.commit_started:
            path = Path(self.output_dir) / "active_resource_plan.json"
            if context.old_active_plan is None:
                path.unlink(missing_ok=True)
            else:
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(context.old_active_plan)
                temporary.replace(path)
        if context.tx.phase == "rolled_back":
            self._advance_layout_generation("resource_reallocation_rollback")
            self._deployment_wait_epoch = True
        self._deployment_event("resumed_old", proposal_id=context.proposal.proposal_id,
            phase=context.tx.phase, reason=context.tx.error,
            actual_placements=[dataclasses.asdict(_spec(w)) for w in self._live_workers()])
        self._periodic_deployment = None

    def _deployment_fail(self, context: _Deployment) -> None:
        self.engine_failure = {"error_type": "ResourceReallocationFailure",
            "message": context.tx.error or context.failure or "deployment transaction failed",
            "proposal_id": context.proposal.proposal_id, "phase": context.tx.phase,
            "operation": "resource_reallocation"}
        self._deployment_event("terminal_failure", **self.engine_failure)
        # Leave context visible for final diagnostics; root terminal-failure
        # path stops all actors and reports buffered requests as unfinished.

    def _poll_periodic_reallocation(self) -> None:
        service = getattr(self, "_reallocation_service", None)
        if service is None:
            return
        for event in service.drain_events():
            payload = dict(event)
            name = payload.pop("event")
            self._deployment_event(name, **payload)
        context = getattr(self, "_periodic_deployment", None)
        if context is None:
            if getattr(self, "_deployment_wait_epoch", False):
                if getattr(self, "_epoch_pending", {}):
                    return
                self._deployment_wait_epoch = False
                self._deployment_event("epoch_acknowledged", generation=self._pool_layout_generation)
            if getattr(self, "_history_rollout", None) is not None or getattr(self, "_epoch_pending", {}):
                return
            proposal = service.poll(self._pool_layout_generation)
            if proposal is None:
                return
            try:
                self._deployment_prepare(proposal)
            except Exception as exc:
                self._deployment_event("proposal_rejected", proposal_id=proposal.proposal_id,
                                       error=repr(exc))
            return
        if context.tx.phase in context.tx.TERMINAL:
            return
        phase = context.tx.phase
        if context.tx.in_place and phase in ("draining", "applying_roles", "verifying_controls"):
            changed = self._deployment_layout_error(context)
            if changed:
                context.tx.abort(changed)
                self._deployment_fail(context)
                return
        kwargs = {}
        if phase == "draining":
            census = self._deployment_census()
            if census != context.last_census:
                self._deployment_event("drain_census", proposal_id=context.proposal.proposal_id,
                                       **dataclasses.asdict(census))
                context.last_census = census
            kwargs["census"] = census
        elif phase in ("verifying_controls", "restoring_controls"):
            census = self._deployment_census()
            if census != context.last_census:
                self._deployment_event("control_census", proposal_id=context.proposal.proposal_id,
                    policy="observe_only_unchanged_layout", **dataclasses.asdict(census))
                context.last_census = census
            ready = self._deployment_controls_progress(context, restore=phase == "restoring_controls")
            kwargs["controls_ready" if phase == "verifying_controls" else "rollback_ready"] = ready
        elif phase in ("applying_roles", "restoring_roles"):
            ready = self._deployment_role_progress(context, restore=phase == "restoring_roles")
            kwargs["roles_ready" if phase == "applying_roles" else "rollback_ready"] = ready
        elif phase in ("stopping_old", "stopping_new"):
            released = self._deployment_stop_progress(context)
            kwargs["old_gpu_claims" if phase == "stopping_old" else "new_gpu_claims"] = 0 if released else None
        elif phase in ("starting_new", "starting_old"):
            self._deployment_poll_rpcs(context)
            expected = context.old_specs if phase == "starting_old" else context.proposal.placements
            ready = (not context.rpc_refs and not context.failure and
                     len(context.starts) == len(expected) and all(w.ready for w in context.starts))
            kwargs["new_ready" if phase == "starting_new" else "rollback_ready"] = ready
            kwargs["new_gpu_claims"] = sum(_cost(w) for w in context.starts if w.ready)
        if context.failure:
            kwargs["failure"] = context.failure
            context.failure = None
        action = context.tx.advance(**kwargs)
        if action == "wait":
            return
        self._deployment_event("phase", proposal_id=context.proposal.proposal_id,
                               phase=context.tx.phase, action=action, error=context.tx.error)
        if action == "apply_roles":
            self._deployment_send_roles(context, restore=False)
        elif action == "restore_roles":
            context.lingering_refs = context.rpc_refs
            context.rpc_refs = {}
            context.role_stage = "await_lingering"
            if not context.lingering_refs:
                self._deployment_send_budgets(context, restore=True, pre_restore=True)
        elif action == "restore_controls":
            context.lingering_refs = context.rpc_refs
            context.rpc_refs = {}
            context.role_stage = "await_lingering"
        elif action == "stop_old":
            self._deployment_stop(context, context.old_workers, "old")
        elif action == "stop_new":
            # Pending startup RPCs cannot add requests; stop every partially
            # created actor before allocating the old layout again.
            self._deployment_stop(context, context.starts, "new")
        elif action == "start_new":
            self._deployment_start(context, old=False)
        elif action == "start_old":
            self._deployment_start(context, old=True)
        elif action == "commit":
            try:
                self._deployment_commit(context)
            except Exception as exc:
                rollback = context.tx.commit_failed(f"commit publication failed: {exc}")
                self._deployment_event("commit_failed", proposal_id=context.proposal.proposal_id,
                                       error=repr(exc), rollback_action=rollback)
                if rollback in ("restore_roles", "restore_controls"):
                    context.lingering_refs = context.rpc_refs
                    context.rpc_refs = {}
                    context.role_stage = "await_lingering"
                    if rollback == "restore_roles" and not context.lingering_refs:
                        self._deployment_send_budgets(context, restore=True, pre_restore=True)
                else:
                    self._deployment_stop(context, context.starts, "new")
        elif action == "resume_old":
            self._deployment_finish_old(context)
        elif action == "fail":
            self._deployment_fail(context)

    def _close_periodic_reallocation(self) -> None:
        service = getattr(self, "_reallocation_service", None)
        if service is not None:
            service.close()
            for event in service.drain_events():
                payload = dict(event)
                name = payload.pop("event")
                self._deployment_event(name, **payload)
        self._reallocation_service = None
