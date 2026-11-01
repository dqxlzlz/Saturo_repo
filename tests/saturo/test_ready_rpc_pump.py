from collections import deque
from types import SimpleNamespace

import pytest

from saturo.runtime.scheduler import SaturoGlobalScheduler
from saturo.runtime.worker import WorkerState
from saturo.runtime.workload import BE, DECODE, PREFILL, RT, TPUT, SaturoRequest


class Ref:
    def __init__(self, kind, wid, value, ready=False, error=None):
        self.kind, self.wid, self.value = kind, wid, value
        self.ready, self.error = ready, error


class Remote:
    def __init__(self, function):
        self.remote = function


class ReadyRay:
    def __init__(self):
        self.refs, self.log = [], []
        self.get_pending = 0
        self.submit_ready = {}
        self.drain_ready = {}

    def wait(self, refs, num_returns, timeout):
        assert timeout == 0
        ready = [ref for ref in refs if ref.ready][:num_returns]
        return ready, [ref for ref in refs if ref not in ready]

    def get(self, ref, timeout=None):
        self.log.append(('get', ref.kind, ref.wid))
        if not ref.ready:
            self.get_pending += 1
            raise TimeoutError('not ready')
        if ref.error is not None:
            raise ref.error
        return ref.value

    def kill(self, actor, no_restart=True):
        actor.killed = True


class Actor:
    def __init__(self, wid, ray):
        self.wid, self.ray = wid, ray
        self.events, self.submissions = [], []
        self.submit_batch = Remote(self.submit)
        self.drain_events = Remote(self.drain)
        self.switch_role = Remote(self.switch)
        self.killed = False

    def switch(self, role):
        self.ray.log.append(('send', 'switch', self.wid))
        return Ref('switch', self.wid, {'accepted': True, 'role': role}, True)

    def submit(self, commands):
        commands = [dict(c) for c in commands]
        self.submissions.append(commands)
        self.ray.log.append(('send', 'submit', self.wid))
        ref = Ref('submit', self.wid, [{'accepted': True} for _ in commands],
                  self.ray.submit_ready.get(self.wid, False))
        self.ray.refs.append(ref)
        return ref

    def drain(self):
        events, self.events = self.events, []
        self.ray.log.append(('send', 'drain', self.wid))
        ref = Ref('drain', self.wid, events,
                  self.ray.drain_ready.get(self.wid, False))
        self.ray.refs.append(ref)
        return ref


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from saturo.runtime import scheduler as module
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock.now)
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler._elapsed = lambda: clock.now
    scheduler.output_dir = tmp_path
    scheduler._perf_events_enabled = False
    scheduler.args = SimpleNamespace(max_wall_s=200, window_s=60)
    scheduler.stop_reason = 'running'
    scheduler.request_errors = 0
    scheduler.trace_recorder = None
    scheduler.pending_prefill, scheduler.pending_decode, scheduler.pending_be = (
        deque(), deque(), deque())
    scheduler.all_rt, scheduler.all_be = [], []
    scheduler._request_index = {}
    scheduler.prefill_handoffs = 0
    scheduler.next_rt_idx = 0
    ray = ReadyRay()
    scheduler._ensure_ray = lambda: ray
    scheduler.workers = {}
    scheduler.retirements = []

    def retire(worker, requeue_active=False):
        scheduler.retirements.append((worker.worker_id, requeue_active))
        worker.retired = True
        worker.ready = False

    scheduler._retire_worker = retire
    for wid, role in [(0, PREFILL), (1, DECODE), (2, DECODE)]:
        scheduler.workers[wid] = WorkerState(
            wid, role, 'local', 'tp1_pp1', 1, 1, None,
            ready=True, actor=Actor(wid, ray))
    return scheduler, ray, clock


def add_be(scheduler, wid):
    req = SaturoRequest(f'be_{wid}', BE, 16, 3, 0,
                        active_attempt_id=f'be_{wid}:d1',
                        assigned_worker_id=wid, decode_attempts=1)
    scheduler.all_be.append(req)
    scheduler._request_index[req.request_id] = req
    scheduler.workers[wid].active_attempts[req.active_attempt_id] = req.request_id
    return req


def finish(wid, req):
    return dict(event='finished', worker_id=wid, request_id=req.request_id,
                attempt_id=req.active_attempt_id, phase=DECODE,
                global_token_count=3)


def test_ready_worker_events_are_delivered_while_peer_drain_is_pending(setup):
    scheduler, ray, clock = setup
    slow, fast = add_be(scheduler, 1), add_be(scheduler, 2)
    scheduler.workers[2].actor.events = [finish(2, fast)]
    ray.drain_ready[2] = True
    scheduler._service_workers()
    assert not fast.is_finished
    scheduler._service_workers()
    assert fast.is_finished and not slow.is_finished
    assert ray.get_pending == 0
    assert not scheduler.retirements
    assert len([r for r in ray.refs if r.kind == 'drain' and r.wid == 1]) == 1
    for _ in range(5):
        scheduler._service_workers()
    assert len([r for r in ray.refs if r.kind == 'drain' and r.wid == 1]) == 1


def test_pending_reservation_ack_holds_p_and_ready_ack_releases_it(setup):
    scheduler, ray, clock = setup
    req = SaturoRequest('rt', RT, 16, 3, 0)
    scheduler.all_rt = [req]
    scheduler._request_index[req.request_id] = req
    scheduler._inject_due_requests()
    scheduler._service_workers()
    d = scheduler.workers[1].actor
    assert d.submissions[0][0]['cmd'] == 'reserve_decode'
    assert not req.decode_reservation_confirmed
    assert not scheduler.workers[0].actor.submissions
    # The same control must not be resubmitted while its ACK is pending.
    scheduler._service_workers()
    assert len(d.submissions) == 1 and ray.get_pending == 0
    ref = scheduler._worker_rpcs['submit'][1]['ref']
    ref.ready = True
    scheduler._service_workers()
    assert req.decode_reservation_confirmed
    assert scheduler.workers[0].actor.submissions[0][0]['phase'] == PREFILL
    ack_index = ray.log.index(('get', 'submit', 1))
    p_index = ray.log.index(('send', 'submit', 0))
    assert ack_index < p_index


def test_pending_submit_retains_new_commands_and_limits_inflight(setup):
    scheduler, ray, clock = setup
    worker = scheduler.workers[1]
    worker.pending_commands = [dict(cmd='add', kind=BE, phase=TPUT, request_id='a')]
    scheduler._service_workers()
    first = scheduler._worker_rpcs['submit'][1]['ref']
    worker.pending_commands.append(dict(cmd='add', kind=BE, phase=TPUT, request_id='b'))
    scheduler._service_workers()
    assert len(worker.actor.submissions) == 1
    assert [c['request_id'] for c in worker.pending_commands] == ['b']
    first.ready = True
    scheduler._service_workers()
    assert len(worker.actor.submissions) == 2
    assert worker.actor.submissions[1][0]['request_id'] == 'b'
    assert not worker.pending_commands
    assert len([r for r in ray.refs if r.kind == 'drain' and r.wid == 1]) == 1


def test_controls_precede_additions_in_one_actor_batch(setup):
    scheduler, ray, clock = setup
    worker = scheduler.workers[1]
    worker.pending_commands = [dict(cmd='add', kind=BE, phase=TPUT, request_id='a'),
                               dict(cmd='release_decode', reservation_id='r', request_id='rt')]
    scheduler._service_workers()
    assert [c['cmd'] for c in worker.actor.submissions[0]] == ['release_decode', 'add']


def test_cancelled_inflight_claim_is_not_confirmed_and_release_is_sent(setup):
    scheduler, ray, clock = setup
    req = SaturoRequest('rt', RT, 16, 3, 0)
    scheduler.all_rt = [req]
    scheduler._request_index[req.request_id] = req
    scheduler._inject_due_requests()
    scheduler._service_workers()
    first = scheduler._worker_rpcs['submit'][1]['ref']
    scheduler._release_decode_reservation(req)
    req.finished_s = 1.0
    first.ready = True
    scheduler._service_workers()
    assert req.decode_reservation_id is None and not req.decode_reservation_confirmed
    assert not scheduler.workers[0].actor.submissions
    assert scheduler.workers[1].actor.submissions[1][0]['cmd'] == 'release_decode'


def test_old_claim_ack_cannot_confirm_replacement_on_same_worker(setup):
    scheduler, ray, clock = setup
    req = SaturoRequest('rt', RT, 16, 3, 0)
    scheduler.all_rt = [req]
    scheduler._request_index[req.request_id] = req
    scheduler._inject_due_requests()
    scheduler._service_workers()
    old_id = req.decode_reservation_id
    old_ref = scheduler._worker_rpcs['submit'][1]['ref']
    scheduler._release_decode_reservation(req)
    assert scheduler._reserve_decode_for_prefill(req)
    assert req.reserved_decode_worker_id == 1
    assert req.decode_reservation_id != old_id
    replacement_id = req.decode_reservation_id
    old_ref.ready = True
    scheduler._service_workers()
    assert not req.decode_reservation_confirmed
    assert not scheduler.workers[0].actor.submissions
    commands = scheduler.workers[1].actor.submissions[1]
    assert [(c['cmd'], c['reservation_id']) for c in commands] == [
        ('release_decode', old_id), ('reserve_decode', replacement_id)]
    scheduler._worker_rpcs['submit'][1]['ref'].ready = True
    scheduler._service_workers()
    assert req.decode_reservation_confirmed
    assert scheduler.workers[0].actor.submissions[0][0]['phase'] == PREFILL


def test_ack_from_old_worker_cannot_confirm_same_id_on_new_worker(setup):
    scheduler, ray, clock = setup
    req = SaturoRequest('rt', RT, 16, 3, 0)
    scheduler.all_rt = [req]
    scheduler._request_index[req.request_id] = req
    scheduler._inject_due_requests()
    scheduler._service_workers()
    old_id = req.decode_reservation_id
    old_ref = scheduler._worker_rpcs['submit'][1]['ref']
    scheduler._release_decode_reservation(req)
    # Also guard against IDs restored by an embedding client: actor identity
    # remains part of ACK matching even if the string is reused externally.
    req.decode_reservation_id = old_id
    req.reserved_decode_worker_id = 2
    scheduler.workers[2].decode_reservations.add(old_id)
    old_ref.ready = True
    scheduler._service_workers()
    assert not req.decode_reservation_confirmed
    assert not scheduler.workers[0].actor.submissions


@pytest.mark.parametrize('kind', ['submit', 'drain'])
def test_pending_rpc_keeps_original_five_second_deadline(setup, kind):
    scheduler, ray, clock = setup
    worker = scheduler.workers[1]
    add_be(scheduler, 1)
    if kind == 'submit':
        worker.pending_commands = [dict(cmd='add', kind=BE, phase=TPUT, request_id='a')]
    scheduler._service_workers()
    ref = scheduler._worker_rpcs[kind][1]['ref']
    clock.now = 4.999
    scheduler._service_workers()
    assert not worker.retired and not scheduler.retirements
    clock.now = 5.0
    scheduler._service_workers()
    assert worker.retired and scheduler.retirements == [(1, True)]
    assert 1 not in scheduler._worker_rpcs['submit']
    assert 1 not in scheduler._worker_rpcs['drain']
    assert ray.get_pending == 0
    # A result arriving after retirement must not mutate or replay the request.
    before = list(ray.log)
    ref.ready = True
    scheduler._service_workers()
    assert ('get', kind, 1) not in ray.log[len(before):]
    assert len(scheduler.retirements) == 1


@pytest.mark.parametrize('kind', ['submit', 'drain'])
def test_ready_result_is_consumed_before_pending_deadline_check(setup, kind):
    scheduler, ray, clock = setup
    worker = scheduler.workers[1]
    req = add_be(scheduler, 1)
    if kind == 'submit':
        worker.pending_commands = [dict(cmd='add', kind=BE, phase=TPUT, request_id='a')]
        # The companion drain is already complete and must not expire either.
        ray.drain_ready[1] = True
    scheduler._service_workers()
    ref = scheduler._worker_rpcs[kind][1]['ref']
    ref.ready = True
    if kind == 'drain':
        ref.value = [finish(1, req)]
    clock.now = 5.0
    scheduler._service_workers()
    assert ('get', kind, 1) in ray.log
    assert not worker.retired and not scheduler.retirements


def test_external_retirement_drops_stale_ready_drain_without_consuming_it(setup):
    scheduler, ray, clock = setup
    req = add_be(scheduler, 1)
    scheduler._service_workers()
    ref = scheduler._worker_rpcs['drain'][1]['ref']
    ref.value, ref.ready = [finish(1, req)], True
    scheduler.workers[1].retired = True
    scheduler._service_workers()
    assert not req.is_finished
    assert not any(item == ('get', 'drain', 1) for item in ray.log)


@pytest.mark.parametrize('kind', ['submit', 'drain'])
def test_dead_engine_rpc_is_terminal_before_new_work_is_dispatched(setup, kind):
    scheduler, ray, clock = setup
    worker = scheduler.workers[1]
    add_be(scheduler, 1)
    if kind == 'submit':
        worker.pending_commands = [dict(cmd='add', kind=BE, phase=TPUT, request_id='a')]
    scheduler._service_workers()
    ref = scheduler._worker_rpcs[kind][1]['ref']
    class AsyncEngineDeadError(RuntimeError):
        pass
    ref.ready, ref.error = True, AsyncEngineDeadError('background loop dead')
    scheduler.workers[2].pending_commands.append(dict(cmd='add', kind=BE, phase=TPUT, request_id='later'))
    scheduler._service_workers()
    assert worker.retired and scheduler._check_done()
    assert scheduler.stop_reason == 'engine_failed' and scheduler.request_errors == 1
    assert not scheduler.workers[2].actor.submissions
    scheduler._service_workers()
    assert scheduler.request_errors == 1


def test_ready_warmup_finish_is_drained_without_request_lookup(setup):
    scheduler, ray, clock = setup
    worker = scheduler.workers[1]
    worker.active_attempts['__warmup__:1'] = '__warmup__:1'
    worker.actor.events = [dict(event='finished', worker_id=1,
                               request_id='__warmup__:1', attempt_id='__warmup__:1',
                               phase=TPUT, global_token_count=2)]
    ray.drain_ready[1] = True
    scheduler._service_workers()
    scheduler._service_workers()
    assert not worker.active_attempts
    assert not scheduler.pending_decode
    assert scheduler.request_errors == 0


def test_role_switch_waits_for_control_ack_but_not_drain(setup):
    scheduler, ray, clock = setup
    worker = scheduler.workers[1]
    worker.role_switching_enabled = True
    scheduler.role_switcher = SimpleNamespace(events=[])
    worker.pending_commands = [dict(cmd='release_decode',
                                    request_id='rt', reservation_id='old')]
    switch = lambda: scheduler._switch_worker_role(
        worker, PREFILL, borrowed=True, reason='test')
    assert not switch()
    assert not any(item[1] == 'switch' for item in ray.log)
    scheduler._service_workers()
    assert not switch()
    scheduler._worker_rpcs['submit'][1]['ref'].ready = True
    scheduler._service_workers()
    assert 1 in scheduler._worker_rpcs['drain']
    assert switch()
    assert worker.role == PREFILL and worker.borrowed_role
    assert len(scheduler.role_switcher.events) == 1
