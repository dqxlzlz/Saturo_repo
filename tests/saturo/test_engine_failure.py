import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

from saturo.runtime.scheduler import SaturoGlobalScheduler
from saturo.runtime.worker import RayEngineWorker, WorkerState
from saturo.runtime.workload import BE, DECODE, RT, TPUT, SaturoRequest


class Remote:
    def __init__(self, result=None):
        self.remote = lambda *args: result


class FakeRay:
    def get(self, ref, timeout=None):
        return ref

    def kill(self, actor, no_restart=True):
        actor.killed = True


def scheduler_with_attempt():
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler._elapsed = lambda: 1.0
    scheduler.args = SimpleNamespace(max_wall_s=200, window_s=60)
    scheduler.stop_reason = 'running'
    scheduler.request_errors = 0
    scheduler.trace_recorder = None
    scheduler.pending_prefill = deque()
    scheduler.pending_decode = deque()
    scheduler.pending_be = deque()
    scheduler._ensure_ray = FakeRay
    actor = SimpleNamespace(shutdown=Remote(), killed=False)
    worker = WorkerState(0, DECODE, 'local', 'tp1_pp1', 1, 1, None,
                         ready=True, actor=actor)
    worker2 = WorkerState(1, DECODE, 'local', 'tp1_pp1', 1, 1, None,
                          ready=True, actor=SimpleNamespace(shutdown=Remote()))
    scheduler.workers = {0: worker, 1: worker2}
    req = SaturoRequest('be_0', BE, 16, 8, 0, active_attempt_id='be_0:d1',
                        assigned_worker_id=0, decode_attempts=1)
    worker.active_attempts[req.active_attempt_id] = req.request_id
    worker.pending_commands.append({'cmd': 'add', 'attempt_id': 'queued'})
    rt = SaturoRequest('rt_0', RT, 16, 8, 0,
                       reserved_decode_worker_id=0,
                       decode_reservation_id='rt_0:p1',
                       decode_reservation_confirmed=True)
    worker.decode_reservations.add('rt_0:p1')
    scheduler.all_rt, scheduler.all_be = [rt], [req]
    scheduler._request_index = {r.request_id: r for r in [rt, req]}
    return scheduler, worker, req, rt


def event(req, error_type='AssertionError', **extra):
    return dict(event='error', worker_id=0, request_id=req.request_id,
                attempt_id=req.active_attempt_id, phase=DECODE,
                error_type=error_type, message='scheduler failed', **extra)


@pytest.mark.parametrize('error_type,tag', [
    ('AssertionError', {'engine_fatal': True}),
    ('AsyncEngineDeadError', {}),
])
def test_dead_engine_is_retired_and_stops_before_resubmission(error_type, tag):
    scheduler, worker, req, rt = scheduler_with_attempt()
    fatal = event(req, error_type, **tag)

    scheduler._handle_event(fatal)

    assert worker.retired and not worker.ready
    assert worker.actor.killed
    assert not worker.active_attempts and not worker.pending_commands
    assert req.active_attempt_id is None and req.assigned_worker_id is None
    assert not req.is_finished
    assert rt.decode_reservation_id is None
    assert not worker.decode_reservations
    assert not scheduler.pending_be
    assert not scheduler.workers[1].active_attempts
    assert scheduler._check_done()
    assert scheduler.stop_reason == 'engine_failed'
    assert scheduler.engine_failure['error_type'] == error_type
    assert scheduler.request_errors == 1

    # A single engine failure fans out to many request streams. Late events
    # must neither resubmit work nor inflate the error count indefinitely.
    scheduler._handle_event(fatal)
    assert scheduler.request_errors == 1
    assert not scheduler._assign(worker, req, TPUT)


def test_stale_request_error_cannot_cancel_a_new_attempt():
    scheduler, worker, req, rt = scheduler_with_attempt()
    stale = event(req, 'IncompleteStreamError')
    stale['attempt_id'] = 'be_0:d0'
    scheduler._handle_event(stale)
    assert not worker.retired and req.active_attempt_id == 'be_0:d1'


@pytest.mark.parametrize('request_id', ['be_0', 'expired_request'])
def test_engine_failure_retires_actor_even_if_reporting_attempt_is_stale(request_id):
    scheduler, worker, req, rt = scheduler_with_attempt()
    fatal = event(req, engine_fatal=True)
    fatal.update(request_id=request_id, attempt_id='be_0:d0')
    scheduler._handle_event(fatal)
    assert worker.retired
    assert scheduler._check_done() and scheduler.stop_reason == 'engine_failed'
    assert req.active_attempt_id is None
    assert not scheduler.pending_be


def test_engine_failure_during_warmup_is_not_discarded():
    scheduler, worker, req, rt = scheduler_with_attempt()
    worker.active_attempts['__warmup__:0'] = '__warmup__:0'
    scheduler._handle_event(dict(
        event='error', worker_id=0, request_id='__warmup__:0',
        attempt_id='__warmup__:0', error_type='AssertionError',
        message='background failure during warmup', engine_fatal=True))
    assert worker.retired
    assert scheduler._check_done()
    assert scheduler.stop_reason == 'engine_failed'
    assert scheduler.engine_failure['request_id'] == '__warmup__:0'


def test_warmup_stops_on_engine_failure_without_waiting_for_other_workers():
    scheduler, worker, req, rt = scheduler_with_attempt()
    scheduler.args.worker_warmup_prompt_len = 16
    scheduler.args.worker_warmup_output_len = 2
    scheduler.args.worker_warmup_timeout_s = 60
    scheduler._service_workers = lambda: scheduler._handle_event(dict(
        event='error', worker_id=0, request_id='__warmup__:0',
        attempt_id='__warmup__:0', error_type='AssertionError',
        message='warmup engine failed', engine_fatal=True))
    scheduler._warmup_workers()
    assert scheduler.engine_failure
    assert scheduler.workers[1].active_attempts  # Still pending, not timed out.
    assert scheduler._check_done() and scheduler.stop_reason == 'engine_failed'


@pytest.mark.parametrize('command,operation', [
    ('add', 'submit'), ('reserve_decode', 'decode_reservations')])
def test_ray_wrapped_dead_engine_rpc_stops_the_run(command, operation):
    class AsyncEngineDeadError(RuntimeError):
        pass

    class RayTaskError(RuntimeError):
        cause = AsyncEngineDeadError('background loop is dead')

    class FailingRay(FakeRay):
        def get(self, ref, timeout=None):
            if ref == 'submit':
                raise RayTaskError('remote submit failed')
            if ref == 'drain':
                return []
            return ref

    scheduler, worker, req, rt = scheduler_with_attempt()
    worker.actor.submit_batch = Remote('submit')
    worker.pending_commands = [dict(cmd=command, request_id=req.request_id)]
    other_submissions = []
    scheduler.workers[1].pending_commands = [dict(cmd='add')]
    scheduler.workers[1].actor.submit_batch = SimpleNamespace(
        remote=lambda commands: other_submissions.append(commands))
    scheduler.workers[1].actor.drain_events = Remote('drain')
    scheduler._ensure_ray = FailingRay
    scheduler._service_workers()
    assert worker.retired and worker.actor.killed
    assert scheduler._check_done()
    assert scheduler.stop_reason == 'engine_failed'
    assert scheduler.engine_failure['operation'] == operation
    assert scheduler.request_errors == 1
    assert not scheduler.workers[1].active_attempts
    # Reservation RPCs run before request admissions: a fatal control error
    # must not be followed by submissions to otherwise healthy workers.
    if command == 'reserve_decode':
        assert not other_submissions


def test_nonfatal_request_errors_have_a_retry_limit():
    scheduler, worker, req, rt = scheduler_with_attempt()
    scheduler._release_decode_reservation(rt)
    retry_calls = []

    def reassign(request, phase):
        retry_calls.append(phase)
        scheduler._assign(worker, request, phase)

    scheduler._requeue = reassign
    for _ in range(4):
        scheduler._handle_event(event(req, 'IncompleteStreamError'))
    assert retry_calls == [DECODE] * 3
    assert scheduler._check_done()
    assert scheduler.stop_reason == 'request_error_limit'
    assert not worker.retired
    assert scheduler.request_failure['request_id'] == req.request_id


class FailingEngine:
    def __init__(self):
        self.engine = SimpleNamespace(scheduler=[], iteration_logs=[])
        self.errored = False
        self.is_stopped = False

    async def generate(self, *args, **kwargs):
        self.errored = True
        raise AssertionError('resident sequences exceeded the limit')
        yield  # Keep this an async generator like AsyncLLMEngine.generate.


def test_initial_background_assertion_is_tagged_as_engine_fatal():
    async def run():
        worker = RayEngineWorker.__new__(RayEngineWorker)
        worker.worker_id = 0
        worker.engine = FailingEngine()
        worker.active = {'be_0:d1': dict(request_id='be_0', kind=BE,
                                        phase=DECODE)}
        worker._events, worker._request_tasks = [], {}
        worker._abort_requested = set()
        worker._admission_diagnostics = False
        worker._last_iteration_log_idx = 0
        await worker._consume_request('be_0:d1', {}, None, 10)
        fatal = worker._events[0]
        assert fatal['error_type'] == 'AssertionError'
        assert fatal['engine_fatal'] is True
        assert not worker.active

    asyncio.run(run())


def test_worker_rejects_new_work_after_engine_failure():
    from vllm.engine.async_llm_engine import AsyncEngineDeadError

    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._shutting_down = False
    worker.worker_role = DECODE
    worker.engine = FailingEngine()
    worker.engine.errored = True
    worker.engine.engine.scheduler = [SimpleNamespace(
        reserve_decode_request=lambda *args: None,
        release_decode_reservation=lambda *args: None)]
    with pytest.raises(AsyncEngineDeadError):
        asyncio.run(worker.submit({'cmd': 'reserve_decode',
                                   'reservation_id': 'r:p1', 'prompt_len': 16}))
