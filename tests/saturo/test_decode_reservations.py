import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

from saturo.runtime.scheduler import SaturoGlobalScheduler
from saturo.runtime.worker import RayEngineWorker, WorkerState
from saturo.runtime.workload import DECODE, PREFILL, RT, SaturoRequest
from saturo.saturation.reservations import DecodeReservations


class Remote:
    def __init__(self, function):
        self.remote = function


class FakeRay:
    def __init__(self):
        self.log = []
        self.results = {}
        self.fail_worker = None

    def get(self, ref, timeout=None):
        if ref[0] == 'submit':
            self.log.append(('ack', ref[1]))
            if ref[1] == self.fail_worker:
                raise RuntimeError('worker unavailable')
        return self.results[ref]


class Actor:
    def __init__(self, worker_id, ray):
        self.worker_id, self.ray = worker_id, ray
        self.count = 0
        self.submit_batch = Remote(self.submit)
        self.drain_events = Remote(self.drain)

    def submit(self, commands):
        self.count += 1
        self.ray.log.append(('submit', self.worker_id,
                             [dict(c) for c in commands]))
        ref = ('submit', self.worker_id, self.count)
        self.ray.results[ref] = [{'accepted': True} for c in commands]
        return ref

    def drain(self):
        ref = ('drain', self.worker_id)
        self.ray.results[ref] = []
        return ref


def setup(output_len=4):
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler._elapsed = lambda: 1.0
    scheduler.trace_recorder = None
    scheduler.pending_decode = deque()
    scheduler.prefill_handoffs = 0
    scheduler.next_rt_idx = 0
    request = SaturoRequest('rt_0', RT, 32, output_len, 0)
    scheduler.all_rt, scheduler.all_be = [request], []
    ray = FakeRay()
    scheduler._ensure_ray = lambda: ray
    scheduler.workers = {}
    for wid, role in [(0, PREFILL), (1, DECODE), (2, DECODE)]:
        scheduler.workers[wid] = WorkerState(
            wid, role, 'local', 'tp1_pp1', 1, 1, None,
            ready=True, actor=Actor(wid, ray))
    return scheduler, request, ray


def finish_prefill(scheduler, request):
    scheduler._handle_event(dict(
        event='finished', worker_id=0, request_id=request.request_id,
        attempt_id=request.active_attempt_id, phase=PREFILL,
        global_token_count=1))


def test_d_acknowledges_reservation_before_p_is_submitted():
    scheduler, req, ray = setup()
    scheduler._inject_due_requests()
    assert req.reserved_decode_worker_id == 1
    assert not req.decode_reservation_confirmed
    scheduler._service_workers()
    assert req.decode_reservation_confirmed
    d_submit = next(i for i,e in enumerate(ray.log)
                    if e[0] == 'submit' and e[1] == 1)
    d_ack = ray.log.index(('ack', 1))
    p_submit = next(i for i,e in enumerate(ray.log)
                    if e[0] == 'submit' and e[1] == 0)
    assert d_submit < d_ack < p_submit
    assert ray.log[d_submit][2][0]['cmd'] == 'reserve_decode'
    assert ray.log[d_submit][2][0]['prompt_len'] == 33


def test_prefill_handoff_stays_on_reserved_d_even_if_another_d_is_idle():
    scheduler, req, ray = setup()
    scheduler._inject_due_requests()
    scheduler._service_workers()
    reservation_id = req.decode_reservation_id
    scheduler.workers[1].active_attempts.update({str(i): 'other' for i in range(10)})
    finish_prefill(scheduler, req)
    scheduler._drain_pending_decode()
    assert req.assigned_worker_id == 1
    command = scheduler.workers[1].pending_commands[-1]
    assert command['cmd'] == 'add' and command['phase'] == DECODE
    assert command['reservation_id'] == reservation_id
    assert not scheduler.workers[1].decode_reservations
    assert req.decode_reservation_id is None


def test_prefill_that_finishes_request_releases_early_claim():
    scheduler, req, ray = setup(output_len=1)
    scheduler._inject_due_requests()
    scheduler._service_workers()
    finish_prefill(scheduler, req)
    assert req.is_finished and not scheduler.pending_decode
    assert req.decode_reservation_id is None
    assert scheduler.workers[1].pending_commands[-1]['cmd'] == 'release_decode'


def test_release_is_idempotent_and_removes_unsent_reservation():
    scheduler, req, ray = setup()
    scheduler._inject_due_requests()
    req.finished_s = 2
    scheduler._release_decode_reservation(req)
    scheduler._release_decode_reservation(req)
    controls = scheduler.workers[1].pending_commands
    assert [c['cmd'] for c in controls] == ['release_decode']
    scheduler._service_workers()
    assert not any(e[0] == 'submit' and e[1] == 0 for e in ray.log)


def test_failed_d_is_replaced_before_pending_p_request_can_start():
    scheduler, req, ray = setup()
    scheduler._inject_due_requests()
    ray.fail_worker = 1
    scheduler._service_workers()
    assert scheduler.workers[1].retired
    assert req.reserved_decode_worker_id == 2
    assert not any(e[0] == 'submit' and e[1] == 0 for e in ray.log)
    ray.fail_worker = None
    scheduler._service_workers()
    assert req.decode_reservation_confirmed
    assert any(e[0] == 'submit' and e[1] == 0 for e in ray.log)


def test_no_available_d_defers_p_without_dropping_request():
    scheduler, req, ray = setup()
    scheduler.workers[1].ready = scheduler.workers[2].ready = False
    scheduler._inject_due_requests()
    # Ingress is observed at its source clock even while D admission waits.
    assert scheduler.next_rt_idx == 1
    assert list(scheduler.pending_prefill) == [req]
    assert req.active_attempt_id is None and not req.is_finished
    scheduler.workers[2].ready = True
    scheduler._inject_due_requests()
    assert scheduler.next_rt_idx == 1 and req.reserved_decode_worker_id == 2


def test_reservation_rounding_and_retries_do_not_duplicate_capacity():
    claims = DecodeReservations()
    claims.reserve('rt:p1', 32, 16, 2, 100)
    claims.reserve('rt:p1', 32, 16, 2, 100)
    assert len(claims) == 1 and claims.blocks(16, 2) == 3
    claims.release('rt:p1')
    claims.release('rt:p1')
    assert claims.blocks(16, 2) == 0
    with pytest.raises(ValueError, match='exceeds GPU KV'):
        claims.reserve('too_large', 99999, 16, 2, 100)


def test_worker_reservation_control_calls_native_scheduler_without_generation():
    calls = []
    native = SimpleNamespace(
        reserve_decode_request=lambda rid,n: calls.append(('reserve',rid,n)),
        release_decode_reservation=lambda rid: calls.append(('release',rid)))
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._shutting_down = False
    worker.worker_role = DECODE
    worker.engine = SimpleNamespace(engine=SimpleNamespace(scheduler=[native]))
    async def run():
        await worker.submit(dict(cmd='reserve_decode',reservation_id='r:p1',prompt_len=32))
        await worker.submit(dict(cmd='release_decode',reservation_id='r:p1'))
    asyncio.run(run())
    assert calls == [('reserve','r:p1',32), ('release','r:p1')]


def test_prefill_retry_releases_old_claim_and_ignores_late_events():
    scheduler, req, ray = setup()
    scheduler._inject_due_requests()
    scheduler._service_workers()
    old_attempt = req.active_attempt_id
    old_reservation = req.decode_reservation_id
    scheduler._handle_event(dict(
        event='aborted', worker_id=0, request_id=req.request_id,
        attempt_id=old_attempt, phase=PREFILL))
    new_reservation = req.decode_reservation_id
    assert new_reservation != old_reservation
    controls = scheduler.workers[1].pending_commands
    assert any(c['cmd'] == 'release_decode' and c['reservation_id'] == old_reservation
               for c in controls)
    scheduler._handle_event(dict(
        event='aborted', worker_id=0, request_id=req.request_id,
        attempt_id=old_attempt, phase=PREFILL))
    assert req.decode_reservation_id == new_reservation
    assert req.prefill_attempts == 2


def test_zero_token_decode_releases_reservation_without_starting_generation():
    released = []
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._shutting_down = False
    worker.worker_id, worker.worker_role = 1, DECODE
    worker.active, worker._request_tasks, worker._events = {}, {}, []
    native = SimpleNamespace(release_decode_reservation=released.append)
    worker.engine = SimpleNamespace(engine=SimpleNamespace(scheduler=[native]))
    result = asyncio.run(worker.submit(dict(
        cmd='add', reservation_id='rt:p1', attempt_id='rt:d1',
        request_id='rt', kind=RT, phase=DECODE, max_tokens=0, base_tokens=1)))
    assert result['accepted']
    assert released == ['rt:p1']
    assert worker._events[0]['event'] == 'finished'
