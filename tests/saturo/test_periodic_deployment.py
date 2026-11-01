"""Exercise the real deployment executor with controllable Ray futures."""
from collections import deque
import dataclasses
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from saturo.runtime.deployment import PeriodicDeploymentMixin
from saturo.runtime import deployment as deployment_module
from saturo.runtime import reallocation as allocation
from saturo.runtime.reallocation import PlacementSpec, ReallocationProposal


class Ref:
    def __init__(self, value=None, *, ready=True, error=None):
        self.value, self.ready, self.error = value, ready, error


class Remote:
    def __init__(self, call):
        self.remote = call


class FakeActor:
    def __init__(self, worker, ray):
        self.worker, self.ray = worker, ray
        self.actual_role = worker.role
        self.actual_p_budget = 100
        self.role_replies = deque()
        self.role_refs = []
        self.budget_refs = []
        self.shutdown_ref = Ref(None)
        self.dead = False
        self.switch_role = Remote(self._role)
        self.submit_batch = Remote(self._budget)
        self.shutdown = Remote(lambda: self.shutdown_ref)
        self.info_refs = []
        self.info = Remote(self._info)

    def _info(self):
        ref = Ref({"worker_id": self.worker.worker_id, "role": self.actual_role,
                   "gpu_id": self.worker.gpu_id})
        self.info_refs.append(ref)
        return ref

    def _role(self, role):
        reply = self.role_replies.popleft() if self.role_replies else Ref({"accepted": True, "role": role})
        if reply.value and reply.value.get("accepted"):
            self.actual_role = role
        self.role_refs.append((role, reply))
        return reply

    def _budget(self, commands):
        if self.actual_role == "prefill":
            self.actual_p_budget = commands[0]["p_chunk_ms"]
        ref = Ref([{"accepted": True, "operation": "history"}])
        self.budget_refs.append((commands, ref))
        return ref


class FakeRay:
    def __init__(self):
        self.available = 0
        self.kills = []
        self.gets = []
        self.delay_release = False
        self.held_release = 0

    def available_resources(self):
        return {"GPU": self.available}

    def wait(self, refs, num_returns, timeout):
        assert timeout == 0
        ready = [r for r in refs if r.ready]
        return ready, [r for r in refs if not r.ready]

    def get(self, ref, timeout):
        assert timeout == 0 and ref.ready, "serving thread attempted a blocking get"
        self.gets.append(ref)
        if ref.error:
            raise ref.error
        return ref.value

    def kill(self, actor, no_restart):
        assert no_restart
        if actor.dead:
            return
        actor.dead = True
        self.kills.append(actor.worker.worker_id)
        count = actor.worker.tensor_parallel_size * actor.worker.pipeline_parallel_size
        if self.delay_release:
            self.held_release += count
        else:
            self.available += count


class Service:
    def __init__(self, proposal):
        self.proposals = deque([proposal])
        self.polls = 0
        self.closed = False

    def poll(self, generation):
        self.polls += 1
        return self.proposals.popleft() if self.proposals else None

    def drain_events(self):
        return []

    def close(self):
        self.closed = True


def worker(wid, role, ray, *, width=1, gpu_id=None, capable=True):
    state = SimpleNamespace(worker_id=wid, role=role, host="local",
        config_name=f"tp{width}_pp1", tensor_parallel_size=width, pipeline_parallel_size=1,
        expert_parallel_size=0, gpu_id=str(wid) if gpu_id is None else gpu_id,
        ready=True, retired=False, role_switching_enabled=capable,
        borrowed_role=False, pending_commands=[], active_attempts={},
        preempting_attempts=set(), decode_reservations=set(), stderr_file=io.StringIO(),
        runtime_profile_path="", num_gpu_blocks=100, max_model_len=1024)
    state.actor = FakeActor(state, ray)
    return state


class Harness(PeriodicDeploymentMixin):
    def __init__(self, tmp_path, proposal):
        self.args = SimpleNamespace(total_gpus=3, worker_start_timeout_s=10,
            resource_reallocation_drain_timeout_s=3, periodic_reallocation=True,
            adaptive_role_switching=True)
        self.output_dir = tmp_path
        self._pool_layout_generation = 4
        self._history_rollout = None
        self._epoch_pending = {}
        self.ray = FakeRay()
        self.workers = {i: worker(i, role, self.ray) for i, role in enumerate(
            ("prefill", "decode", "throughput"))}
        self.pending_prefill, self.pending_decode, self.pending_be = deque(), deque(), deque()
        self._worker_rpcs = {"submit": {}, "drain": {}}
        self._allocation_profiles = object()
        self._p_pool_be_chunk_ms_map = {"tp1_pp1": 100}
        self.role_switcher = SimpleNamespace(profiles=self._allocation_profiles, _predictors={"old": 1})
        self.events, self.starts, self.releases = [], [], []
        self._periodic_deployment = None
        self._deployment_wait_epoch = False
        self._reallocation_service = Service(proposal)
        self.engine_failure = None
        self.now = 0
        self.next_worker_id = 10
        self.start_faults = deque()
        self.start_dispatch_failure = False
        self.start_budgets = []
        self.requests = {}

    def _deployment_clock(self):
        return self.now

    def _live_workers(self):
        return [w for w in self.workers.values() if not w.retired]

    def _ensure_ray(self):
        return self.ray

    def _request_by_id(self, request_id):
        return self.requests.get(request_id)

    def _runtime_event(self, event, **fields):
        self.events.append({"event": event, **fields})

    def _release_decode_reservation(self, req):
        self.releases.append(req.request_id)
        reservation = getattr(req, "decode_reservation_id", None)
        for state in self.workers.values():
            state.decode_reservations.discard(reservation)
        req.decode_reservation_id = None

    def _drop_worker_rpc_refs(self, wid):
        for table in self._worker_rpcs.values():
            table.pop(wid, None)

    def _begin_worker_start(self, placement):
        width = placement.tensor_parallel_size * placement.pipeline_parallel_size
        assert self.ray.available >= width, "new actors overlap the old GPU budget"
        self.ray.available -= width
        wid = self.next_worker_id; self.next_worker_id += 1
        state = worker(wid, placement.role, self.ray, width=width, gpu_id=placement.gpu_id,
                       capable=self.args.adaptive_role_switching)
        state.ready = False
        self.workers[wid] = state
        self.starts.append(state)
        self.start_budgets.append(dict(self._deployment_start_p_budgets))
        if self.start_dispatch_failure:
            self.start_dispatch_failure = False
            raise RuntimeError("info dispatch failed after actor registration")
        if self.start_faults:
            ref = self.start_faults.popleft()
        else:
            ref = Ref({"worker_id": wid, "role": placement.role, "gpu_id": placement.gpu_id})
        return state, ref

    def _finish_worker_start(self, state, info):
        if info.get("worker_id") != state.worker_id or info.get("role") != state.role:
            raise RuntimeError("wrong worker ready identity")
        state.ready = True

    def _advance_layout_generation(self, reason):
        self._pool_layout_generation += 1
        self._epoch_pending = {w.worker_id: object() for w in self._live_workers()}
        return self._pool_layout_generation


def make_proposal(tmp_path, *, restart=False, unchanged=False):
    specs = ((PlacementSpec("prefill", 1, 1, gpu_id="0", config_name="tp1_pp1"),
              PlacementSpec("decode", 2, 1, gpu_id="1,2", config_name="tp2_pp1")) if restart else
             tuple(PlacementSpec(role, 1, 1, gpu_id=str(i), config_name="tp1_pp1")
                for i, role in enumerate(("prefill", "decode", "throughput") if unchanged
                                         else ("prefill", "prefill", "decode"))))
    directory = tmp_path / "proposal"; directory.mkdir()
    profile = {"configs": {name: {"gpus": n, "tp": n, "pp": 1}
                           for name, n in (("tp1_pp1", 1), ("tp2_pp1", 2))},
        "pools": {pool: {name: {**({"mu": 2} if pool != "T" else {}),
            "be_chunk_latency_ms": 99 if pool == "P" else None,
            "saturation": [{"rt_rps": 0, "be_throughput": 1}]}
            for name in ("tp1_pp1", "tp2_pp1")} for pool in ("P", "D", "T")}}
    profile_path = directory / "allocation_profiles.json"
    allocation._write(profile_path, profile)
    plan = {"placements": [dataclasses.asdict(p) for p in specs], "base_generation": 4,
            "deployment_controls": {"p_budgets": {"tp1_pp1": {"chunk_ms": 17, "rt_rps": 0.15}}}}
    plan_path = directory / "resource_worker_plan.json"; allocation._write(plan_path, plan)
    manifest_path = directory / "manifest.json"
    allocation._write(manifest_path, {"files": [
        {"path": str(p), "sha256": allocation._sha(p)} for p in (profile_path, plan_path)]})
    return ReallocationProposal("test", 4, str(profile_path), str(plan_path), str(manifest_path),
        specs, allocation._json(plan), allocation._sha(manifest_path))


def tick_until(harness, predicate, limit=30):
    for _ in range(limit):
        harness._poll_periodic_reallocation()
        if predicate():
            return
    raise AssertionError(f"phase did not progress: {getattr(harness._periodic_deployment, 'tx', None)}")


def test_proposal_waits_for_history_and_epoch_barriers(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    h._history_rollout = {}
    h._poll_periodic_reallocation(); assert h._reallocation_service.polls == 0
    h._history_rollout = None; h._epoch_pending = {1: object()}
    h._poll_periodic_reallocation(); assert h._reallocation_service.polls == 0
    h._epoch_pending.clear(); h._poll_periodic_reallocation()
    assert h._deployment_pauses_new_work() and h._deployment_needs_workers()


def test_drain_preserves_buffered_requests_and_releases_only_unstarted_claims(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path, restart=True))
    req = SimpleNamespace(request_id="buffered", active_attempt_id=None,
        decode_reservation_id="claim", arrival_offset_s=12.5, observed_tokens=0)
    h.pending_prefill.append(req); h.workers[1].decode_reservations.add("claim")
    in_progress = SimpleNamespace(request_id="inflight", active_attempt_id="a",
                                  decode_reservation_id="activeclaim", is_finished=False)
    h.pending_prefill.append(in_progress)
    h.pending_decode.append(SimpleNamespace(is_finished=False))
    h.pending_be.append(SimpleNamespace(request_id="newbe", arrival_offset_s=0, observed_tokens=0,
                                       is_finished=False))
    h.workers[0].active_attempts["a"] = "inflight"
    h._poll_periodic_reallocation()
    assert h.releases == ["buffered"]
    h._poll_periodic_reallocation()
    assert not h.ray.kills and h._periodic_deployment.tx.phase == "draining"
    h.workers[0].active_attempts.clear(); h.pending_decode.clear()
    h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.phase == "stopping_old"
    assert h.pending_prefill[0] is req and req.arrival_offset_s == 12.5 and req.observed_tokens == 0
    assert len(h.pending_be) == 1


@pytest.mark.parametrize("obstacle", ["commands", "submit", "reservations", "preemptions"])
def test_all_native_control_work_must_drain(tmp_path, obstacle):
    h = Harness(tmp_path, make_proposal(tmp_path))
    if obstacle == "commands": h.workers[0].pending_commands.append({"cmd": "release_decode"})
    elif obstacle in ("submit", "drain"): h._worker_rpcs[obstacle][0] = object()
    elif obstacle == "reservations": h.workers[0].decode_reservations.add("claim")
    else: h.workers[0].preempting_attempts.add("BEstillstopping")
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.phase == "draining"
    assert not h.ray.kills


def test_in_place_roles_and_selected_budget_all_ack_before_commit_and_epoch(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    old_profiles = h._allocation_profiles
    startup = tmp_path / "resource_worker_plan.json"; startup.write_text("original startup")
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.phase == "applying_roles"
    assert [w.role for w in h._live_workers()] == ["prefill", "decode", "throughput"]
    h._poll_periodic_reallocation()  # roles ACK, dispatch exact selected P budget
    assert h._allocation_profiles is old_profiles
    assert h._p_pool_be_chunk_ms_map["tp1_pp1"] == 100
    assert h.workers[1].actor.budget_refs[0][0] == [{"cmd": "history", "p_chunk_ms": 17}]
    h._poll_periodic_reallocation()
    assert h._periodic_deployment is None
    assert [w.role for w in h._live_workers()] == ["prefill", "prefill", "decode"]
    assert h._p_pool_be_chunk_ms_map["tp1_pp1"] == 17
    assert h.role_switcher.profiles is h._allocation_profiles is not old_profiles
    assert h._deployment_pauses_new_work()  # epoch not acknowledged yet
    assert h._deployment_needs_workers()
    h._epoch_pending.clear(); h._poll_periodic_reallocation()
    assert not h._deployment_pauses_new_work()
    assert not h._deployment_needs_workers()
    assert startup.read_text() == "original startup"
    assert json.loads((tmp_path / "active_resource_plan.json").read_text())["committed_generation"] == 5
    assert not h.ray.kills and not h.starts


def test_in_place_retains_resident_be_and_kv_on_same_actor_during_progress(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    request = SimpleNamespace(request_id="be", kind="be", observed_tokens=3)
    h.requests["be"] = request
    resident = h.workers[2]
    resident.active_attempts["be:a"] = "be"
    actor = resident.actor
    h._worker_rpcs["drain"][2] = {"ref": Ref([], ready=False)}
    h._poll_periodic_reallocation()
    census = h._deployment_census()
    assert census.active_attempts == 0 and census.retained_local_be == 1
    assert census.inflight_rpcs == 0  # read-only streaming does not prevent role barrier
    for _ in range(3):
        request.observed_tokens += 1  # same actor continues its BE stream
        h._poll_periodic_reallocation()
    assert h._periodic_deployment is None
    assert resident.actor is actor and resident.active_attempts == {"be:a": "be"}
    assert request.observed_tokens == 6
    assert not h.ray.kills and not h.starts
    assert any(e["event"] == "resource_drain_census" and e["retained_local_be"] == 1
               for e in h.events)


@pytest.mark.parametrize("capable", [True, False])
def test_unchanged_layout_refresh_keeps_rt_handoffs_and_claims_live_until_info_budget_ack(tmp_path, capable):
    h = Harness(tmp_path, make_proposal(tmp_path, unchanged=True))
    for state in h._live_workers(): state.role_switching_enabled = capable
    h.requests["rt"] = SimpleNamespace(kind="rt", observed_tokens=9)
    h.workers[1].active_attempts["rt:a"] = "rt"
    h.workers[1].decode_reservations.add("claim")
    waiting = SimpleNamespace(request_id="new", active_attempt_id=None,
        decode_reservation_id="claim", arrival_offset_s=10, observed_tokens=0)
    handoff = SimpleNamespace(is_finished=False)
    h.pending_prefill.append(waiting); h.pending_decode.append(handoff)
    h._worker_rpcs["submit"][1] = {"ref": Ref([], ready=False)}
    h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.controls_only
    assert not h._deployment_pauses_new_work()
    assert h._deployment_needs_workers()
    assert h.releases == [] and waiting.decode_reservation_id == "claim"
    assert len(h.workers[0].actor.info_refs) == 1
    assert all(not w.actor.role_refs for w in h._live_workers())
    old_profiles = h._allocation_profiles
    h._poll_periodic_reallocation()  # info ACK -> selected P budget dispatch
    assert h._allocation_profiles is old_profiles
    budget_ref = h.workers[0].actor.budget_refs[0][1]
    budget_ref.ready = False
    h._poll_periodic_reallocation()
    assert h._allocation_profiles is old_profiles and not h._deployment_pauses_new_work()
    assert h.workers[1].active_attempts == {"rt:a": "rt"}
    assert h.pending_decode[0] is handoff and h.pending_prefill[0] is waiting
    assert any(e["event"] == "resource_control_census" and e["active_attempts"] == 1 and
               e["reservations"] == 1 and e["local_pending_requests"] == 1 for e in h.events)
    budget_ref.ready = True; h._poll_periodic_reallocation()
    assert h._periodic_deployment is None and h._allocation_profiles is not old_profiles
    assert h._deployment_pauses_new_work()  # only epoch ACK barrier pauses ingress
    assert h.workers[1].active_attempts == {"rt:a": "rt"}
    assert h.workers[1].decode_reservations == {"claim"}
    assert h.requests["rt"].observed_tokens == 9
    assert all(not w.actor.role_refs for w in h._live_workers())


def test_static_role_change_uses_restart_after_full_be_drain_and_preserves_buffered_work(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))  # same engines, new P/D role counts
    h.args.adaptive_role_switching = False
    for state in h._live_workers(): state.role_switching_enabled = False
    h.requests["be"] = SimpleNamespace(kind="be", observed_tokens=3)
    h.workers[2].active_attempts["be:a"] = "be"
    pending = SimpleNamespace(request_id="queued", active_attempt_id=None,
        decode_reservation_id=None, arrival_offset_s=7.5, observed_tokens=0)
    h.pending_prefill.append(pending)
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    assert not h._periodic_deployment.tx.in_place
    assert not h._periodic_deployment.tx.controls_only
    assert h._deployment_pauses_new_work()
    assert h._deployment_census().active_attempts == 1
    assert h._deployment_census().retained_local_be == 0
    assert not any(state.actor.role_refs for state in h._live_workers())
    assert not h.ray.kills
    h.workers[2].active_attempts.clear()  # original engine finishes its BE locally
    tick_until(h, lambda: h._periodic_deployment is None)
    assert len(h.ray.kills) == len(h.starts) == 3
    assert [state.role for state in h._live_workers()] == ["prefill", "prefill", "decode"]
    assert all(not state.role_switching_enabled for state in h._live_workers())
    assert h.pending_prefill[0] is pending and pending.arrival_offset_s == 7.5
    assert pending.observed_tokens == 0
    assert any(e.get("in_place_disabled_reason") == "role_switching_not_supported" for e in h.events)


def test_static_role_change_drain_timeout_retains_original_actors(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    for state in h._live_workers(): state.role_switching_enabled = False
    actors = [state.actor for state in h._live_workers()]
    h.requests["be"] = SimpleNamespace(kind="be")
    h.workers[2].active_attempts["be:a"] = "be"
    h._poll_periodic_reallocation(); h.now = 4; h._poll_periodic_reallocation()
    assert h._periodic_deployment is None
    assert [state.actor for state in h._live_workers()] == actors
    assert not h.ray.kills and not h.starts
    assert h.workers[2].active_attempts == {"be:a": "be"}


def test_controls_info_failure_restores_budget_without_any_role_rpc(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path, unchanged=True))
    h.workers[1].active_attempts["rt:a"] = "rt"
    h._poll_periodic_reallocation()
    h.workers[1].actor.info_refs[0].value["role"] = "wrong"
    tick_until(h, lambda: h._periodic_deployment is None)
    assert h._p_pool_be_chunk_ms_map == {"tp1_pp1": 100}
    assert h.workers[1].active_attempts == {"rt:a": "rt"}
    assert not h.ray.kills and not h.starts
    assert all(not w.actor.role_refs for w in h._live_workers())


@pytest.mark.parametrize("change", ["retired", "not_ready", "actor_replaced", "epoch_changed"])
def test_controls_cannot_commit_if_layout_changes_after_info_before_budget_ack(tmp_path, change):
    h = Harness(tmp_path, make_proposal(tmp_path, unchanged=True))
    old_profiles = h._allocation_profiles
    h._poll_periodic_reallocation()
    h._poll_periodic_reallocation()  # all info ACK; P budget still in flight
    assert h.workers[0].actor.budget_refs
    if change == "retired": h.workers[1].retired = True
    elif change == "not_ready": h.workers[1].ready = False
    elif change == "actor_replaced": h.workers[1].actor = FakeActor(h.workers[1], h.ray)
    else: h._pool_layout_generation += 1
    h._poll_periodic_reallocation()
    assert h._allocation_profiles is old_profiles
    assert not any(e["event"] == "resource_committed" for e in h.events)
    assert h.engine_failure["error_type"] == "ResourceReallocationFailure"
    assert h._periodic_deployment.tx.phase == "failed"


def test_rt_and_handoffs_still_block_while_resident_be_is_retained(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    h.requests.update(be=SimpleNamespace(kind="be"), rt=SimpleNamespace(kind="rt"))
    h.workers[0].active_attempts["rt:a"] = "rt"
    h.workers[2].active_attempts["be:a"] = "be"
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    census = h._deployment_census()
    assert census.active_attempts == 1 and census.retained_local_be == 1
    assert h._periodic_deployment.tx.phase == "draining"
    h.workers[0].active_attempts.clear()
    h.pending_decode.append(SimpleNamespace(is_finished=False))
    h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.phase == "draining"
    h.pending_decode.clear(); h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.phase == "applying_roles"


def test_in_place_waits_mutations_but_not_readonly_streaming_rpc(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    h.requests["be"] = SimpleNamespace(kind="be")
    h.workers[2].active_attempts["be:a"] = "be"
    h._worker_rpcs["drain"][2] = {"ref": Ref([], ready=False)}
    h._worker_rpcs["submit"][2] = {"ref": Ref([], ready=False)}
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    assert h._deployment_census().inflight_rpcs == 1
    assert h._periodic_deployment.tx.phase == "draining"
    h._worker_rpcs["submit"].clear(); h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.phase == "applying_roles"


def test_restart_still_waits_resident_be_and_readonly_drain_rpc(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path, restart=True))
    h.requests["be"] = SimpleNamespace(kind="be")
    h.workers[2].active_attempts["be:a"] = "be"
    h._worker_rpcs["drain"][2] = {"ref": Ref([], ready=False)}
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    census = h._deployment_census()
    assert census.active_attempts == 1 and census.retained_local_be == 0
    assert census.inflight_rpcs == 1
    h.workers[2].active_attempts.clear(); h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.phase == "draining"
    h._worker_rpcs["drain"].clear(); h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.phase == "stopping_old"


def test_target_excluding_resident_be_falls_back_to_full_drain(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    h.args.enable_decode_saturation = False
    h.requests["be"] = SimpleNamespace(kind="be")
    h.workers[2].active_attempts["be:a"] = "be"  # proposed T→D excludes BE
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    assert not h._periodic_deployment.tx.in_place
    assert h._deployment_census().active_attempts == 1
    assert h._deployment_census().retained_local_be == 0
    assert not any(w.actor.role_refs for w in h._live_workers())
    assert any(e.get("in_place_disabled_reason") == "target_excludes_resident_be" for e in h.events)


@pytest.mark.parametrize("kind", [None, "unknown"])
def test_unknown_request_kind_remains_blocking(tmp_path, kind):
    h = Harness(tmp_path, make_proposal(tmp_path))
    if kind is not None: h.requests["mystery"] = SimpleNamespace(kind=kind)
    h.workers[2].active_attempts["unknown:a"] = "mystery"
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    assert h._deployment_census().active_attempts == 1
    assert h._deployment_census().retained_local_be == 0
    assert h._periodic_deployment.tx.phase == "draining"


def test_preempting_be_attempt_remains_blocking_and_not_double_counted(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    h.requests["be"] = SimpleNamespace(kind="be")
    h.workers[2].active_attempts["be:a"] = "be"
    h.workers[2].preempting_attempts.add("be:a")
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    census = h._deployment_census()
    assert census.active_attempts == 1 and census.retained_local_be == 0
    assert h._periodic_deployment.tx.phase == "draining"


def test_failed_role_ack_waits_lingering_rpc_before_restoring_original_roles(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    delayed = Ref({"accepted": True, "role": "prefill"}, ready=False)
    h.workers[1].actor.role_replies.append(delayed)
    h.workers[2].actor.role_replies.append(Ref({"accepted": False, "role": "throughput"}))
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.phase == "restoring_roles"
    h._poll_periodic_reallocation()
    assert len(h.workers[1].actor.role_refs) == 1  # no restore racing old pending RPC
    delayed.ready = True
    tick_until(h, lambda: h._periodic_deployment is None)
    assert [w.actor.actual_role for w in h._live_workers()] == ["prefill", "decode", "throughput"]
    assert all(w.actor.actual_p_budget == 100 for w in h._live_workers())
    assert h._p_pool_be_chunk_ms_map == {"tp1_pp1": 100}
    assert any(e["event"] == "resource_resumed_old" for e in h.events)
    assert not (tmp_path / "active_resource_plan.json").exists()


def test_restart_never_allocates_before_actual_ray_gpu_release(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path, restart=True))
    h.ray.delay_release = True
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    assert len(h.ray.kills) == 3 and not h.starts
    assert h._deployment_needs_workers()
    h.ray.available += h.ray.held_release; h.ray.held_release = 0
    h.ray.delay_release = False
    tick_until(h, lambda: h._periodic_deployment is None)
    assert sum(w.tensor_parallel_size for w in h._live_workers()) == 3
    assert len(h._live_workers()) == 2
    assert h.start_budgets == [{"tp1_pp1": 17, "tp2_pp1": 99}] * 2
    assert any(e["event"] == "resource_gpu_release_verified" for e in h.events)
    assert h.engine_failure is None


def test_new_start_failure_cleans_every_partial_actor_and_restarts_old_layout(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path, restart=True))
    h.start_faults.extend([Ref(error=RuntimeError("new engine failed"))])
    tick_until(h, lambda: h._periodic_deployment is None)
    assert len(h.starts) == 5  # two new actors, then original three
    assert all(w.retired for w in h.starts[:2])
    assert [w.role for w in h._live_workers()] == ["prefill", "decode", "throughput"]
    assert h._p_pool_be_chunk_ms_map == {"tp1_pp1": 100}
    assert h.start_budgets[-3:] == [{"tp1_pp1": 100}] * 3
    assert h.engine_failure is None
    assert not (tmp_path / "active_resource_plan.json").exists()


def test_dispatch_failure_after_registration_does_not_leak_untracked_actor(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path, restart=True))
    h.start_dispatch_failure = True
    tick_until(h, lambda: h._periodic_deployment is None)
    assert h.starts[0].retired
    assert h.starts[0].worker_id in h.ray.kills
    assert len(h._live_workers()) == 3 and h.engine_failure is None


def test_drain_timeout_resumes_live_workers_and_keeps_buffered_identity(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    h.workers[0].active_attempts["a"] = "rt"
    h._poll_periodic_reallocation(); h.now = 4; h._poll_periodic_reallocation()
    assert h._periodic_deployment is None and not h._deployment_pauses_new_work()
    assert h.workers[0].active_attempts == {"a": "rt"}
    assert not h.ray.kills and h._pool_layout_generation == 4


def test_started_be_retry_cannot_be_migrated_as_new_buffered_work(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path, restart=True))
    request = SimpleNamespace(request_id="partial_be", observed_tokens=2,
                              decode_attempts=1, is_finished=False)
    h.pending_be.append(request)
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    assert h._periodic_deployment.tx.phase == "draining"
    assert h._deployment_census().local_pending_requests == 1
    assert not h.ray.kills
    h.now = 4; h._poll_periodic_reallocation()
    assert h._periodic_deployment is None and h.pending_be[0] is request


def test_shutdown_timeout_forces_kill_after_drain_then_restarts(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path, restart=True))
    for w in h._live_workers(): w.actor.shutdown_ref.ready = False
    h._poll_periodic_reallocation(); h._poll_periodic_reallocation(); h._poll_periodic_reallocation()
    assert not h.ray.kills
    h.now = 6
    tick_until(h, lambda: h._periodic_deployment is None)
    assert len(h.ray.kills) == 3 and h.engine_failure is None


def test_rollback_failure_is_terminal_and_preserves_unfinished_requests(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path, restart=True))
    h.start_faults.extend([Ref(error=RuntimeError("new failed")), Ref(error=RuntimeError("new2 failed")),
                           Ref(error=RuntimeError("old rollback failed"))])
    request = SimpleNamespace(request_id="buffered", active_attempt_id=None,
        decode_reservation_id=None, arrival_offset_s=8, observed_tokens=0)
    h.pending_prefill.append(request)
    tick_until(h, lambda: h.engine_failure is not None)
    assert h.engine_failure["error_type"] == "ResourceReallocationFailure"
    assert h.pending_prefill[0] is request and request.observed_tokens == 0
    assert h._periodic_deployment.tx.phase == "failed"


def test_invalid_actual_gpu_width_rolls_back_without_commit(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path, restart=True))
    h.start_faults.extend([Ref({"worker_id": 10, "role": "prefill", "gpu_id": "0"}),
                           Ref({"worker_id": 11, "role": "decode", "gpu_id": "1"})])
    tick_until(h, lambda: h._periodic_deployment is None)
    assert h._p_pool_be_chunk_ms_map == {"tp1_pp1": 100}
    assert len(h._live_workers()) == 3
    assert any(e["event"] == "resource_rpc_failed" and "GPU allocation" in e["error"] for e in h.events)


def test_tampered_proposal_is_rejected_before_quiesce(tmp_path):
    proposal = make_proposal(tmp_path)
    Path(proposal.profile_path).write_text("modified")
    h = Harness(tmp_path, proposal); h._poll_periodic_reallocation()
    assert not h._deployment_pauses_new_work() and not h.ray.kills
    assert h.events[-1]["event"] == "resource_proposal_rejected"


def test_commit_publication_failure_rolls_back_roles_and_old_profiles(tmp_path, monkeypatch):
    h = Harness(tmp_path, make_proposal(tmp_path))
    old_profiles = h._allocation_profiles
    real = h._deployment_commit
    def fail_after_apply(context):
        real(context)
        h._periodic_deployment = context  # emulate failure before clearing transaction
        raise OSError("publication failed")
    monkeypatch.setattr(h, "_deployment_commit", fail_after_apply)
    tick_until(h, lambda: h._periodic_deployment is None)
    assert h._allocation_profiles is old_profiles
    assert [w.actor.actual_role for w in h._live_workers()] == ["prefill", "decode", "throughput"]
    assert all(w.actor.actual_p_budget == 100 for w in h._live_workers())
    assert not (tmp_path / "active_resource_plan.json").exists()
    assert h.engine_failure is None


def test_close_stops_background_service(tmp_path):
    h = Harness(tmp_path, make_proposal(tmp_path))
    service = h._reallocation_service
    h._close_periodic_reallocation()
    assert service.closed and h._reallocation_service is None


def test_init_derives_required_hardware_constraint_from_actual_worker_profiles(tmp_path, monkeypatch):
    h = Harness(tmp_path, make_proposal(tmp_path))
    cache = tmp_path / "measured.json"
    cache.write_text(json.dumps({"time_predictor": {"profile_identity": {"gpu_name": "RTX4090"}}}))
    for w in h._live_workers(): w.runtime_profile_path = str(cache)
    h.history_controller = SimpleNamespace(snapshot=lambda **kw: None)
    h.args.output_dir = str(tmp_path)
    captured = []
    def service(args, loader, **kwargs):
        captured.append((args, kwargs))
        return SimpleNamespace(interval_s=1800, history_window_s=1800)
    monkeypatch.setattr(deployment_module, "PeriodicReallocationService", service)
    h._init_periodic_reallocation()
    assert captured[0][0].resource_reallocation_gpu_name == "RTX4090"
    assert captured[0][1]["timing_profiles"] == {"tp1_pp1": str(cache)}
    assert not hasattr(h.args, "resource_reallocation_gpu_name")


@pytest.mark.parametrize("case", ["mixed", "missing", "explicit_wrong"])
def test_init_rejects_unbound_or_inconsistent_hardware(tmp_path, case):
    h = Harness(tmp_path, make_proposal(tmp_path))
    h.history_controller = SimpleNamespace(snapshot=lambda **kw: None)
    cache = tmp_path / "measured.json"
    cache.write_text(json.dumps({"time_predictor": {"profile_identity": {"gpu_name": "RTX4090"}}}))
    for w in h._live_workers(): w.runtime_profile_path = str(cache)
    if case == "mixed":
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"profile_identity": {"gpu_name": "GB200"}}))
        h.workers[1].runtime_profile_path = str(other)
    elif case == "missing": h.workers[1].runtime_profile_path = ""
    else: h.args.resource_reallocation_gpu_name = "GB200"
    with pytest.raises(ValueError):
        h._init_periodic_reallocation()
