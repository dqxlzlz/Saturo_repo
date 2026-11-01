"""CPU checks for adaptive routing, role RPCs and iteration-boundary safety."""

import asyncio
import json
from collections import deque
from types import SimpleNamespace

import pytest

from saturo.resource_allocation import AllocationProfiles
from saturo.runtime.role_switching import AdaptiveRoleSwitcher
from saturo.runtime.scheduler import SaturoGlobalScheduler
from saturo.runtime.worker import RayEngineWorker, WorkerState
from saturo.runtime.workload import BE, DECODE, PREFILL, RT, TPUT, SaturoRequest


class Ref:
    def __init__(self, value):
        self.value = value


class Remote:
    def __init__(self, func):
        self.func = func

    def remote(self, *args):
        return Ref(self.func(*args))


class Actor:
    def __init__(self):
        self.roles = []
        self.commands = []
        self.events = []
        self.switch_role = Remote(self.switch)
        self.submit_batch = Remote(self.submit)
        self.drain_events = Remote(self.drain)
        self.shutdown = Remote(lambda: None)

    def switch(self, role):
        self.roles.append(role)
        return {"accepted": True, "role": role}

    def submit(self, commands):
        self.commands.extend(commands)
        for cmd in commands:
            if cmd.get("kind") == RT:
                self.events.append(dict(
                    event="token", worker_id=self.worker_id,
                    request_id=cmd["request_id"], attempt_id=cmd["attempt_id"],
                    phase=cmd["phase"], global_token_count=(
                        cmd["base_tokens"] + cmd["max_tokens"])))
                self.events.append(dict(
                    event="finished", worker_id=self.worker_id,
                    request_id=cmd["request_id"], attempt_id=cmd["attempt_id"],
                    phase=cmd["phase"], global_token_count=(
                        cmd["base_tokens"] + cmd["max_tokens"])))
        return [{"accepted": True} for _ in commands]

    def drain(self):
        events, self.events = self.events, []
        return events


class Ray:
    @staticmethod
    def get(reply, timeout=None):
        return reply.value

    @staticmethod
    def kill(actor, no_restart=True):
        pass


class Predictor:
    def predict_forward_ms(self, prefill, decode):
        return sum(tokens * 0.1 for tokens, _ in prefill) + len(decode) * 10


class RemainingPredictor:
    config = SimpleNamespace(feature_mode="len")

    def __init__(self, remaining=8):
        self.remaining = remaining
        self.rows = []

    def predict_remaining(self, **kwargs):
        self.rows.append(kwargs)
        return self.remaining


def profiles(p=1, d=1, t=10, mu=10):
    return AllocationProfiles.from_dict({
        "configs": {"tp1_pp1": {"gpus": 1}},
        "pools": {
            pool: {"tp1_pp1": {"mu": mu, "saturation": [
                {"rt_rps": rate, "be_throughput": value}
                for rate in ([0] if pool == "T" else [0, 100])
            ]}}
            for pool, value in [("P", p), ("D", d), ("T", t)]
        },
    })


def scheduler(profile=None, *, reservations=False):
    s = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    s.args = SimpleNamespace(
        role_switch_interval_s=60, role_switch_window_s=60,
        role_switch_timeout_s=10, target_rps=1,
        rt_ttft_slo_ms=2000, rt_tbt_slo_ms=100,
        resource_memory_watermark=0.9)
    s.now = 0.0
    s._elapsed = lambda: s.now
    s._ensure_ray = lambda: Ray()
    s._resolve_p_pool_be_chunk_ms = lambda config: 20
    s.all_rt, s.all_be = [], []
    s.pending_prefill = deque()
    s.pending_decode, s.pending_be = deque(), deque()
    s.workers = {}
    s.next_rt_idx = 0
    s.trace_recorder = None
    s.prefill_handoffs = 0
    s.request_errors, s.stop_reason = 0, "running"
    s.rt_tokens = s.rt_tbt_violations = s.rt_ttft_violations = 0
    s.rt_tbt_samples_ms, s.rt_ttft_samples_ms = [], []
    s.role_switcher = AdaptiveRoleSwitcher(s, profile or profiles())
    s.role_switcher._predictor = lambda w: Predictor()
    s.role_switcher._length_predictor = RemainingPredictor()
    if not reservations:
        # Isolate policy cases from the separately tested reservation protocol.
        s._reserve_decode_for_prefill = lambda req: True
        s._prefill_reservation_ready = lambda command: True
        s._reserved_decode_worker = lambda req: s._select_rt_worker(DECODE, req)
    return s


def worker(s, role, **kwargs):
    actor = Actor()
    defaults = dict(
        worker_id=len(s.workers), role=role, host="local",
        config_name="tp1_pp1", tensor_parallel_size=1,
        pipeline_parallel_size=1, stderr_file=None, actor=actor,
        ready=True, num_gpu_blocks=1024, block_size=16,
        max_model_len=8192, max_num_seqs=32, role_switching_enabled=True)
    defaults.update(kwargs)
    w = WorkerState(**defaults)
    actor.worker_id = w.worker_id
    s.workers[w.worker_id] = w
    return w


def request(s, worker=None, role=PREFILL, *, kind=RT, prompt=32,
            output=8, arrival=0):
    requests = s.all_rt if kind == RT else s.all_be
    req = SaturoRequest(
        f"{kind}_{len(requests)}", kind, prompt, output, arrival)
    requests.append(req)
    if worker is not None:
        s._assign(worker, req, role)
    return req


@pytest.mark.parametrize("role", [PREFILL, DECODE])
def test_burst_borrows_t_worker_preserving_be_and_parallel_configuration(role):
    s = scheduler()
    busy = worker(s, role, max_num_seqs=1)
    request(s, busy, role)
    t = worker(s, TPUT)
    be = request(s, t, TPUT, kind=BE)
    be_attempt = be.active_attempt_id
    incoming = request(s)
    selected = s._select_rt_worker(role, incoming)
    assert selected is t and t.role == role and t.borrowed_role
    s._assign(selected, incoming, role)
    assert be.active_attempt_id == be_attempt
    assert be_attempt in t.active_attempts
    assert incoming.assigned_worker_id == t.worker_id
    assert (t.tensor_parallel_size, t.pipeline_parallel_size) == (1, 1)
    assert len(s.workers) == 2
    assert s.role_switcher.events[-1]["reason"] == "rt_burst"


def test_routing_uses_rt_prefill_tokens_and_decode_kv_not_be_request_count():
    s = scheduler()
    first, second = worker(s, PREFILL), worker(s, PREFILL)
    request(s, first, prompt=400)
    request(s, second, prompt=16)
    for _ in range(5):
        request(s, second, TPUT, kind=BE)
    assert s._select_rt_worker(PREFILL, request(s)) is second
    first.role = second.role = DECODE
    assert s.role_switcher.load(first, DECODE) > s.role_switcher.load(
        second, DECODE)
    assert s._select_rt_worker(DECODE, request(s)) is second


@pytest.mark.parametrize("field,value", [
    ("num_gpu_blocks", 0), ("max_model_len", 16),
    ("ready", False), ("retired", True),
])
def test_no_admission_to_unavailable_or_memory_infeasible_workers(field, value):
    s = scheduler()
    w = worker(s, TPUT, **{field: value})
    assert s._select_rt_worker(PREFILL, request(s)) is None
    assert not w.actor.roles


def test_decode_reserves_predicted_output_and_uses_tbt_prediction():
    s = scheduler()
    w = worker(s, DECODE, num_gpu_blocks=10)
    s.role_switcher._length_predictor = RemainingPredictor(1000)
    assert not s.role_switcher.can_accept(w, DECODE, request(s, output=8))
    s.role_switcher._length_predictor = RemainingPredictor(8)
    s.args.rt_tbt_slo_ms = 5
    assert not s.role_switcher.can_accept(w, DECODE, request(s))


def test_trace_output_budget_does_not_change_decode_admission():
    s = scheduler()
    w = worker(s, DECODE, num_gpu_blocks=10)
    short, long = request(s, output=8), request(s, output=1000)
    assert s.role_switcher.can_accept(w, DECODE, short)
    assert s.role_switcher.can_accept(w, DECODE, long)
    assert all("output_len" not in r and "max_tokens" not in r
               for r in s.role_switcher._length_predictor.rows)
    assert s.role_switcher.admission_diagnostics()["uses_trace_output_len"] is False


def test_admission_cannot_read_trace_future_length_at_all():
    class NoFutureLength(SaturoRequest):
        def __getattribute__(self, name):
            if name in ("output_len", "remaining_tokens"):
                raise AssertionError("Trace future length leaked into admission")
            return super().__getattribute__(name)
    s = scheduler()
    w = worker(s, DECODE)
    incoming = NoFutureLength("blind", RT, 32, 9999, 0)
    assert s.role_switcher.can_accept(w, DECODE, incoming)


def test_content_prediction_cache_tracks_observed_tokens_and_model_revision():
    s = scheduler()
    p = RemainingPredictor(12)
    p.config = SimpleNamespace(feature_mode="tfidf")
    s.role_switcher._length_predictor = p
    w = worker(s, DECODE)
    req = request(s, output=100)
    req.prompt_token_ids = [11] * req.prompt_len
    assert s.role_switcher.can_accept(w, DECODE, req)
    assert len(p.rows) == 1
    assert p.rows[0]["prompt_token_ids"] == tuple(req.prompt_token_ids)
    assert p.rows[0]["generated_token_ids"] == ()
    assert s.role_switcher.can_accept(w, DECODE, req)
    assert len(p.rows) == 1
    req.generated_token_ids.append(22)
    req.observed_tokens = 1
    assert s.role_switcher.can_accept(w, DECODE, req)
    assert len(p.rows) == 2 and p.rows[-1]["generated_token_ids"] == (22,)
    req.prompt_token_ids[0] = 33
    assert s.role_switcher.can_accept(w, DECODE, req)
    assert len(p.rows) == 3
    p._prediction_revision = 1
    assert s.role_switcher.can_accept(w, DECODE, req)
    assert len(p.rows) == 4
    s.role_switcher._remaining_cache.clear()
    s.role_switcher._remaining_cache_bytes = 0
    req.generated_token_ids = []
    with pytest.raises(ValueError, match="observed-prefix"):
        s.role_switcher.can_accept(w, DECODE, req)


def test_prediction_failure_never_uses_trace_as_fallback():
    s = scheduler()
    w = worker(s, DECODE)
    def fail(**kwargs):
        raise ValueError("model prediction failed")
    s.role_switcher._length_predictor.predict_remaining = fail
    with pytest.raises(ValueError, match="model prediction failed"):
        s.role_switcher.can_accept(w, DECODE, request(s))


def test_missing_timing_uses_capacity_fallback_and_serial_decode():
    s = scheduler(profiles(mu=10))
    s.role_switcher._predictor = lambda w: None
    w = worker(s, DECODE)
    assert s.role_switcher.can_accept(w, DECODE, request(s))
    request(s, w, DECODE)
    assert not s.role_switcher.can_accept(w, DECODE, request(s))
    w.role = PREFILL
    assert s.role_switcher.can_accept(w, PREFILL, request(s))


def test_unsupported_target_profile_never_borrows_worker():
    s = scheduler()
    w = worker(s, TPUT, config_name="unprofiled")
    assert s._select_rt_worker(PREFILL, request(s)) is None
    assert w.role == TPUT


def test_requests_wait_when_no_capacity_instead_of_being_marked_finished():
    s = scheduler()
    busy = worker(s, PREFILL, max_num_seqs=1)
    active = request(s, busy)
    incoming = request(s)
    # Only the incoming request is a newly arriving trace entry.
    s.next_rt_idx = 1
    s._inject_due_requests()
    assert list(s.pending_prefill) == [incoming]
    assert incoming.finished_s is None and incoming.active_attempt_id is None
    busy.active_attempts.pop(active.active_attempt_id)
    active.active_attempt_id = None
    active.finished_s = 0.1
    s._drain_pending_prefill()
    assert not s.pending_prefill
    assert incoming.assigned_worker_id == busy.worker_id


def test_disabled_controller_routes_without_role_changes():
    s = scheduler()
    w = worker(s, PREFILL)
    worker(s, TPUT)
    s.role_switcher = None
    assert s._select_rt_worker(PREFILL, request(s)) is w


def test_timing_profiles_are_prepared_before_workload_submission(monkeypatch):
    import saturo.runtime.scheduler as module
    s = scheduler()
    worker(s, TPUT)
    s.args.model, s.args.enable_decode_saturation = 'm', False
    s.args.worker_warmup = False
    s.worker_plan, s.history_controller = deque(), None
    calls = []
    monkeypatch.setattr(module, 'store_from_env', lambda: None)
    monkeypatch.setattr(module, 'configured_history_source', lambda: 'csv')
    monkeypatch.setattr(module, 'is_redis_history', lambda source: False)
    s._build_workload = lambda: None
    s._start_initial_workers = lambda: calls.append('workers')
    s.role_switcher._predictor = lambda w: calls.append('predictor')
    s._preload_be_to_workers = lambda: calls.append('be')
    s._check_done = lambda: True
    s._shutdown_workers = lambda: None
    s._write_outputs = lambda: None
    s._summary = lambda: {}
    s.run()
    assert calls == ['workers', 'predictor', 'be']


def test_stale_error_does_not_duplicate_an_active_retry():
    s = scheduler()
    w = worker(s, PREFILL)
    req = request(s, w)
    s._requeue(req, PREFILL)
    assert not s.pending_prefill


def test_expired_ttft_is_served_best_effort_without_bypassing_memory():
    s = scheduler()
    w = worker(s, PREFILL)
    s.now = 3
    req = request(s)
    assert s._select_rt_worker(PREFILL, req) is w
    w.num_gpu_blocks = 0
    assert s._select_rt_worker(PREFILL, req) is None


@pytest.mark.parametrize("blocks,borrowed", [(1024, True), (0, False)])
def test_expired_ttft_can_borrow_t_when_p_is_missing(blocks, borrowed):
    s = scheduler()
    w = worker(s, TPUT, num_gpu_blocks=blocks)
    s.now = 3
    selected = s._select_rt_worker(PREFILL, request(s))
    assert (selected is w) == borrowed
    assert w.role == (PREFILL if borrowed else TPUT)


def test_borrowed_worker_returns_after_rt_drain_even_with_live_be():
    s = scheduler()
    w = worker(s, PREFILL, borrowed_role=True)
    request(s, w, TPUT, kind=BE)
    rt = request(s, w)
    s.role_switcher.maintain()
    assert w.role == PREFILL
    w.active_attempts.pop(rt.active_attempt_id)
    rt.active_attempt_id = None
    rt.finished_s = 0.1
    s.pending_prefill.append(request(s))
    s.role_switcher.maintain()
    assert w.role == PREFILL
    s.pending_prefill.clear()
    s.role_switcher.maintain()
    assert w.role == TPUT and not w.borrowed_role
    assert len(w.active_attempts) == 1
    assert s.role_switcher.events[-1]["reason"] == "rt_drained"


def test_borrowed_d_keeps_early_reservation_until_it_is_released():
    s = scheduler(reservations=True)
    d = worker(s, DECODE, borrowed_role=True)
    req = request(s)
    assert s._reserve_decode_for_prefill(req)
    s.role_switcher.maintain()
    assert d.role == DECODE and d.decode_reservations
    s._release_decode_reservation(req)
    s._service_workers()
    s.role_switcher.maintain()
    assert d.role == TPUT and not d.decode_reservations


def test_early_d_claims_count_toward_admission_before_prefill_finishes():
    s = scheduler(reservations=True)
    d = worker(s, DECODE, max_num_seqs=1)
    t = worker(s, TPUT)
    first, second = request(s), request(s)
    assert s._reserve_decode_for_prefill(first)
    assert first.reserved_decode_worker_id == d.worker_id
    assert s.role_switcher.load(d, DECODE) > 0
    assert s._reserve_decode_for_prefill(second)
    assert second.reserved_decode_worker_id == t.worker_id
    assert t.role == DECODE and t.borrowed_role


def test_last_t_is_reserved_for_d_before_attempting_to_borrow_p():
    s = scheduler(reservations=True)
    p = worker(s, PREFILL, max_num_seqs=1)
    d = worker(s, DECODE, max_num_seqs=1)
    t = worker(s, TPUT)
    request(s, p)
    request(s, d, DECODE)
    req = request(s)
    s.next_rt_idx = 2
    s._inject_due_requests()
    assert t.role == DECODE
    assert req.reserved_decode_worker_id == t.worker_id
    assert req.prefill_attempts == 0 and list(s.pending_prefill) == [req]


def test_periodic_reclaim_requires_be_gain_and_keeps_one_p_and_d():
    s = scheduler()
    worker(s, PREFILL)
    worker(s, PREFILL)
    worker(s, DECODE)
    s.now = 59
    s.role_switcher.maintain()
    assert not s.role_switcher.events
    s.now = 60
    s.role_switcher.maintain()
    assert [w.role for w in s.workers.values()].count(PREFILL) == 1
    assert s.role_switcher.events[-1]["reason"] == "be_gain"
    s.now = 120
    s.role_switcher.maintain()
    assert len(s.role_switcher.events) == 1


@pytest.mark.parametrize("why", ["capacity", "no_gain", "active_rt"])
def test_reclaim_never_removes_needed_or_busy_capacity(why):
    s = scheduler(profiles(mu=2, t=0 if why == "no_gain" else 10))
    p = worker(s, PREFILL)
    worker(s, PREFILL, role_switching_enabled=False)
    worker(s, DECODE)
    if why == "active_rt":
        request(s, p)
    s.now = 60
    if why == "capacity":
        s.role_switcher.arrivals.extend([60] * 240)
    s.role_switcher.maintain()
    assert not s.role_switcher.events


def test_mixed_configuration_reclaim_keeps_sufficient_rt_capacity():
    payload = profiles(mu=2).to_dict()
    payload["configs"]["tp2_pp1"] = {"gpus": 2}
    for pool in ("P", "D", "T"):
        payload["pools"][pool]["tp2_pp1"] = dict(
            payload["pools"][pool]["tp1_pp1"], mu=4)
    s = scheduler(AllocationProfiles.from_dict(payload))
    small = worker(s, PREFILL)
    worker(s, PREFILL, config_name="tp2_pp1", tensor_parallel_size=2)
    worker(s, DECODE, config_name="tp2_pp1", tensor_parallel_size=2)
    s.now = 60
    s.role_switcher.arrivals.extend([60] * 180)
    s.role_switcher.maintain()
    assert small.role == TPUT
    assert [w.role for w in s.workers.values()].count(PREFILL) == 1
    assert sum(w.tensor_parallel_size * w.pipeline_parallel_size
               for w in s.workers.values()) == 5


def test_rejected_role_rpc_does_not_commit_control_plane_change():
    s = scheduler()
    w = worker(s, TPUT)
    w.actor.switch_role = Remote(lambda role: {"accepted": False, "role": TPUT})
    assert s._select_rt_worker(PREFILL, request(s)) is None
    assert w.role == TPUT and not s.role_switcher.events


@pytest.mark.parametrize("reason", ["rt_burst", "return_borrowed"])
def test_deployment_freezes_all_adaptive_role_writers(reason):
    s = scheduler()
    w = worker(s, TPUT)
    s._deployment_needs_workers = lambda: True
    calls = []
    w.actor.switch_role = Remote(lambda role: calls.append(role))
    assert not s._switch_worker_role(
        w, PREFILL, borrowed=True, reason=reason)
    assert not calls and w.role == TPUT and not s.role_switcher.events


def test_failed_role_rpc_retires_uncertain_actor_and_requeues_be():
    s = scheduler()
    w = worker(s, TPUT)
    be = request(s, w, TPUT, kind=BE)

    def fail(role):
        raise TimeoutError("uncertain role")

    w.actor.switch_role = Remote(fail)
    assert not s._switch_worker_role(
        w, PREFILL, borrowed=True, reason="rt_burst")
    assert w.retired and be.active_attempt_id is None
    assert list(s.pending_be) == [be]


@pytest.mark.parametrize("reservations", [False, True])
def test_burst_handoff_decode_and_return_lifecycle_over_fake_ray(
        reservations, tmp_path):
    s = scheduler(reservations=reservations)
    s.output_dir = tmp_path
    p, d = worker(s, PREFILL, max_num_seqs=1), worker(
        s, DECODE, max_num_seqs=1)
    tp, td = worker(s, TPUT), worker(s, TPUT)
    # Native pools stay occupied by earlier attempts throughout this burst.
    request(s, p)
    request(s, d, DECODE)
    p.pending_commands.clear()
    d.pending_commands.clear()
    be = request(s, tp, TPUT, kind=BE)
    incoming = request(s)
    s.next_rt_idx = 2
    s._inject_due_requests()
    assert incoming.assigned_worker_id in (tp.worker_id, td.worker_id)
    borrowed_p = s.workers[incoming.assigned_worker_id]
    s._service_workers()
    assert list(s.pending_decode) == [incoming]
    s._drain_pending_decode()
    borrowed_d = s.workers[incoming.assigned_worker_id]
    assert borrowed_d is not borrowed_p and borrowed_d.borrowed_role
    s._service_workers()
    s.role_switcher.maintain()
    assert incoming.is_finished
    assert incoming.observed_tokens == incoming.output_len
    assert s.rt_tokens == incoming.output_len
    assert borrowed_p.role == borrowed_d.role == TPUT
    assert be.active_attempt_id in tp.active_attempts
    assert len(s.workers) == 4
    assert [e["reason"] for e in s.role_switcher.events].count("rt_burst") == 2
    reasons = [e["reason"] for e in s.role_switcher.events]
    assert reasons.count("rt_drained") == 2


class Policy:
    fail = False

    def __init__(self, block_size=16, total_gpu_blocks=100,
                 scheduler_config=None):
        if self.fail:
            raise ValueError("QRF load failed")
        self.enabled = False
        self.block_size, self.total_gpu_blocks = block_size, total_gpu_blocks
        self.scheduler_config = scheduler_config


def async_worker():
    w = RayEngineWorker.__new__(RayEngineWorker)
    w.worker_role = TPUT
    w._role_switching_enabled = True
    w._shutting_down = w._use_dummy_kv = False
    w._decode_role_initialized = False
    w._role_args = dict(rt_tbt_slo_ms=75, p_pool_be_chunk_ms=30)
    w.active = {"be:t1": {"kind": BE}}
    schedulers = [SimpleNamespace(
        scheduler_config=SimpleNamespace(enable_p_pool_saturation=False,
                                         enable_decode_saturation=False),
        decode_saturation=Policy()) for _ in range(2)]

    async def step(*args):
        await asyncio.sleep(0)
        return True

    w.engine = SimpleNamespace(
        engine_step=step, engine=SimpleNamespace(
            scheduler=schedulers,
            scheduler_config=schedulers[0].scheduler_config))
    w._install_role_barrier()
    return w


def test_worker_changes_all_pp_stages_without_rebuilding_engine_or_losing_be():
    async def run():
        w = async_worker()
        engine = w.engine
        for role in (PREFILL, DECODE, TPUT, DECODE):
            assert (await w.switch_role(role))["accepted"]
            for stage in engine.engine.scheduler:
                assert stage.scheduler_config.enable_p_pool_saturation == (
                    role == PREFILL)
                assert stage.decode_saturation.enabled == (role == DECODE)
            assert w.engine is engine and "be:t1" in w.active
        config = w.engine.engine.scheduler[0].scheduler_config
        assert config.decode_saturation_tbt_ms == 75
    asyncio.run(run())


def test_decode_policy_warms_before_traffic_and_is_reused_on_switch():
    async def run():
        w = async_worker()
        w._warm_decode_role()
        policies = [stage.decode_saturation for stage in w.engine.engine.scheduler]
        assert w.worker_role == TPUT and w._decode_role_initialized
        assert all(not policy.enabled for policy in policies)
        Policy.fail = True  # A role switch must not deserialize QRF again.
        try:
            assert (await w.switch_role(DECODE))["accepted"]
        finally:
            Policy.fail = False
        assert policies == [s.decode_saturation for s in w.engine.engine.scheduler]
    asyncio.run(run())


def test_worker_rejects_role_change_until_rt_is_drained():
    async def run():
        w = async_worker()
        w.active["rt:d1"] = {"kind": RT}
        assert not (await w.switch_role(DECODE))["accepted"]
        assert w.worker_role == TPUT
        w.active.pop("rt:d1")
        assert (await w.switch_role(DECODE))["accepted"]
    asyncio.run(run())


def test_worker_rejects_downgrade_with_native_decode_claims():
    async def run():
        w = async_worker()
        assert (await w.switch_role(DECODE))["accepted"]
        for stage in w.engine.engine.scheduler:
            stage._decode_reservations = SimpleNamespace(tokens={"rt:p1": 32})
        assert not (await w.switch_role(TPUT))["accepted"]
        assert w.worker_role == DECODE
        for stage in w.engine.engine.scheduler:
            stage._decode_reservations.tokens.clear()
        assert (await w.switch_role(TPUT))["accepted"]
    asyncio.run(run())


def test_new_submissions_wait_for_role_barrier():
    async def run():
        w = async_worker()
        w.worker_id, w._events, w._request_tasks = 0, [], {}
        w._role_iterations = 1
        switch = asyncio.create_task(w.switch_role(PREFILL))
        await asyncio.sleep(0)
        submission = asyncio.create_task(w.submit(dict(
            cmd="add", request_id="empty", attempt_id="empty:p1",
            kind=RT, phase=PREFILL, max_tokens=0, base_tokens=0)))
        await asyncio.sleep(0)
        assert not submission.done()
        async with w._role_condition:
            w._role_iterations = 0
            w._role_condition.notify_all()
        assert (await switch)["accepted"]
        assert (await submission)["accepted"]
        assert w.worker_role == PREFILL
    asyncio.run(run())


def test_cancelled_switch_releases_iteration_barrier():
    async def run():
        w = async_worker()
        w._role_iterations = 1
        task = asyncio.create_task(w.switch_role(PREFILL))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not w._role_changing and w.worker_role == TPUT
        w._role_iterations = 0
        assert (await w.switch_role(PREFILL))["accepted"]
    asyncio.run(run())


def test_switchable_t_engine_starts_with_chunking_enabled(monkeypatch):
    import sys
    from saturo.runtime.worker import _build_engine_config
    monkeypatch.setitem(sys.modules, "vllm",
                        SimpleNamespace(SamplingParams=object))
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(
        CompilationConfig=object, KVTransferConfig=object))
    args = dict(worker_role=TPUT, model="m", tensor_parallel_size=1,
                pipeline_parallel_size=1, load_format="auto", dtype="float16",
                gpu_memory_utilization=0.9)
    before, _, _ = _build_engine_config(args)
    after, _, _ = _build_engine_config(dict(args, adaptive_role_switching=True))
    assert not before["enable_chunked_prefill"]
    assert after["enable_chunked_prefill"]


def test_cli_role_switch_flags_and_validation(monkeypatch):
    import sys
    from saturo.cli.global_scheduler import parse_args
    base = ["saturo", "--num-req", "10", "--target-rps", "1"]
    monkeypatch.setattr(sys, "argv", base)
    assert parse_args().adaptive_role_switching
    monkeypatch.setattr(sys, "argv", base + ["--no-adaptive-role-switching"])
    assert not parse_args().adaptive_role_switching
    monkeypatch.setattr(sys, "argv", base + ["--role-switch-interval-s", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def test_constructor_loads_role_profiles_without_starting_gpu_workers(
        tmp_path, monkeypatch):
    import sys
    from saturo.cli.global_scheduler import parse_args
    monkeypatch.setattr(
        "saturo.resource_allocation.planner.configured_history_source",
        lambda source: "")
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(profiles().to_dict()))
    monkeypatch.setattr(sys, "argv", [
        "saturo", "--num-req", "10", "--target-rps", "1",
        "--total-gpus", "4", "--model", "m", "--resource-profile", str(path),
        "--output-dir", str(tmp_path / "output"),
    ])
    args = parse_args()
    s = SaturoGlobalScheduler(args)
    assert s.role_switcher is not None
    assert len(s.worker_plan) == 4
    assert not s.workers and not s.pending_prefill
    queued = SaturoRequest("queued", RT, 16, 4, 0)
    submitted = SaturoRequest("submitted", RT, 16, 4, 0, prefill_attempts=1)
    s.all_rt = [submitted, queued]
    s.next_rt_idx = 2
    assert s._summary()["rt_submitted"] == 1
    args.adaptive_role_switching = False
    assert SaturoGlobalScheduler(args).role_switcher is None


def test_role_switch_waits_for_all_pp_iterations_and_reopens_after_failure():
    async def run():
        w = async_worker()
        release = [asyncio.Event(), asyncio.Event()]
        entered = [asyncio.Event(), asyncio.Event()]

        async def step(ve):
            entered[ve].set()
            await release[ve].wait()

        w.engine.engine_step = step
        w._install_role_barrier()
        tasks = [asyncio.create_task(w.engine.engine_step(i)) for i in range(2)]
        await asyncio.gather(*(event.wait() for event in entered))
        switch = asyncio.create_task(w.switch_role(PREFILL))
        await asyncio.sleep(0)
        assert not switch.done() and w.worker_role == TPUT
        release[0].set()
        await tasks[0]
        assert not switch.done()
        release[1].set()
        await tasks[1]
        assert (await switch)["accepted"]
        Policy.fail = True
        try:
            with pytest.raises(ValueError, match="QRF"):
                await w.switch_role(DECODE)
        finally:
            Policy.fail = False
        assert w.worker_role == PREFILL
        assert all(stage.scheduler_config.enable_p_pool_saturation
                   for stage in w.engine.engine.scheduler)
        await asyncio.wait_for(w.engine.engine_step(0), 1)
        assert not w._role_changing
    asyncio.run(run())
