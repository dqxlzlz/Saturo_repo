from collections import deque
from types import SimpleNamespace

import pytest

from saturo.runtime.scheduler import SaturoGlobalScheduler
from saturo.runtime.worker import WorkerState
from saturo.runtime.workload import BE, DECODE, PREFILL, RT, TPUT, SaturoRequest


def active_request(kind=RT, phase=DECODE, observed=89, target=90):
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler._elapsed = lambda: 3.0
    scheduler.args = SimpleNamespace(max_wall_s=600, window_s=400)
    scheduler.request_errors = 0
    scheduler.stop_reason = 'running'
    scheduler.trace_recorder = None
    scheduler.pending_prefill = deque()
    scheduler.pending_decode = deque()
    scheduler.pending_be = deque()
    scheduler.prefill_handoffs = 0
    scheduler._release_decode_reservation = lambda request: None
    worker = WorkerState(0, phase, 'local', 'tp1_pp1', 1, 1, None, ready=True)
    aid = 'request:p1' if phase == PREFILL else 'request:d1'
    req = SaturoRequest('request', kind, 934, target, 0,
                        observed_tokens=observed, active_attempt_id=aid,
                        assigned_worker_id=0)
    worker.active_attempts[aid] = req.request_id
    scheduler.workers = {0: worker}
    scheduler.all_rt = [req] if kind == RT else []
    scheduler.all_be = [req] if kind == BE else []
    scheduler._request_index = {req.request_id: req}
    return scheduler, worker, req, dict(event='finished', worker_id=0,
        request_id=req.request_id, attempt_id=aid, phase=phase,
        global_token_count=observed)


@pytest.mark.parametrize('kind,phase', [(RT, DECODE), (BE, DECODE), (BE, TPUT)])
def test_short_current_final_phase_fails_before_wall_limit_without_retry(kind, phase):
    scheduler, worker, req, event = active_request(kind, phase)
    scheduler._handle_event(event)
    assert scheduler._check_done()
    assert scheduler.stop_reason == 'incomplete_generation'
    assert req.finished_s is None and req.observed_tokens == 89
    assert req.active_attempt_id is None and req.assigned_worker_id is None
    assert not worker.active_attempts and not worker.retired
    assert not scheduler.pending_decode and not scheduler.pending_be
    assert scheduler.request_errors == 1
    failure = scheduler.request_failure
    assert failure['error_type'] == 'IncompleteGeneration'
    assert failure['request_id'] == req.request_id
    assert failure['attempt_id'] == event['attempt_id']
    assert failure['worker_id'] == 0 and failure['phase'] == phase
    assert failure['observed_tokens'] == 89 and failure['requested_tokens'] == 90
    assert failure['missing_tokens'] == 1 and failure['elapsed_s'] == 3
    # The duplicate terminal notification is now stale and cannot inflate errors.
    scheduler._handle_event(event)
    assert scheduler.request_errors == 1
    assert not scheduler._assign(worker, SaturoRequest('new', kind, 1, 1, 0), phase)


def test_prefill_partial_finish_is_still_a_decode_handoff():
    scheduler, worker, req, event = active_request(phase=PREFILL, observed=1)
    scheduler._handle_event(event)
    assert not scheduler._has_terminal_failure()
    assert req.finished_s is None and req.prefill_finished_s == 3
    assert list(scheduler.pending_decode) == [req]
    assert scheduler.prefill_handoffs == 1 and scheduler.request_errors == 0
    assert not worker.active_attempts


@pytest.mark.parametrize('kind,phase', [(RT, DECODE), (BE, DECODE), (BE, TPUT)])
def test_full_target_finish_remains_complete(kind, phase):
    scheduler, worker, req, event = active_request(kind, phase, observed=90)
    scheduler._handle_event(event)
    assert req.finished_s == 3 and req.is_finished
    assert not scheduler._has_terminal_failure() and scheduler.request_errors == 0
    assert not worker.active_attempts


def test_stale_short_finished_does_not_fail_or_clear_replacement():
    scheduler, worker, req, event = active_request()
    event['attempt_id'] = 'request:d0'
    scheduler._handle_event(event)
    assert not scheduler._has_terminal_failure() and scheduler.request_errors == 0
    assert req.active_attempt_id == 'request:d1' and req.assigned_worker_id == 0
    assert worker.active_attempts == {'request:d1': req.request_id}


@pytest.mark.parametrize('phase', [DECODE, TPUT])
def test_abort_with_partial_progress_still_requeues_normally(phase):
    scheduler, worker, req, event = active_request(BE, phase)
    requeued = []
    scheduler._requeue = lambda request, source_phase: requeued.append((request, source_phase))
    event['event'] = 'aborted'
    scheduler._handle_event(event)
    assert not scheduler._has_terminal_failure() and scheduler.request_errors == 0
    assert req.finished_s is None and requeued == [(req, phase)]


def test_failure_stops_event_batch_before_following_request_is_handled(tmp_path):
    scheduler, worker, req, event = active_request()
    scheduler.output_dir = tmp_path
    following = dict(event, global_token_count=90)
    scheduler._consume_worker_events([event, following])
    assert scheduler.request_failure['error_type'] == 'IncompleteGeneration'
    assert req.observed_tokens == 89 and req.finished_s is None
    assert len((tmp_path / 'request_events.jsonl').read_text().splitlines()) == 1


def test_engine_failure_and_request_retry_limit_stop_reasons_keep_precedence():
    scheduler, worker, req, event = active_request()
    scheduler.request_failure = {'error_type': 'IncompleteStreamError'}
    assert scheduler._check_done() and scheduler.stop_reason == 'request_error_limit'
    scheduler.request_failure = {'error_type': 'IncompleteGeneration'}
    scheduler.engine_failure = {'error_type': 'AsyncEngineDeadError'}
    assert scheduler._check_done() and scheduler.stop_reason == 'engine_failed'
