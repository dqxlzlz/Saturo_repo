import asyncio
import csv
import json
from collections import deque
from types import SimpleNamespace

import pytest

from saturo.runtime.scheduler import SaturoGlobalScheduler
from saturo.runtime.worker import (OfflineEngineWorker, RayEngineWorker,
                                   WorkerState)
from saturo.runtime.workload import DECODE, PREFILL, RT, SaturoRequest


def _scheduler() -> SaturoGlobalScheduler:
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler.started_s = 0.0
    scheduler.trace_recorder = None
    scheduler.pending_decode = deque()
    scheduler.all_rt = []
    scheduler.all_be = []
    scheduler.workers = {}
    return scheduler


def _worker(**kwargs) -> WorkerState:
    defaults = {
        "worker_id": 0,
        "role": DECODE,
        "host": "local",
        "config_name": "tp1_pp1",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "stderr_file": None,
        "ready": True,
    }
    defaults.update(kwargs)
    return WorkerState(**defaults)


def test_assign_builds_prefill_and_decode_async_submission_commands():
    scheduler = _scheduler()
    scheduler._elapsed = lambda: 1.0
    scheduler._source_arrival_epoch_s = 100.0
    worker = _worker(role=PREFILL)
    request = SaturoRequest("rt_0", RT, 16, 4, 0.0)
    decode_worker = _worker(worker_id=1, role=DECODE)
    scheduler.workers = {0: worker, 1: decode_worker}
    scheduler.all_rt = [request]

    scheduler._assign(worker, request, PREFILL)

    prefill = worker.pending_commands.pop()
    assert prefill == {
        "cmd": "add",
        "request_id": "rt_0",
        "attempt_id": "rt_0:p1",
        "kind": RT,
        "phase": PREFILL,
        "prompt_len": 16,
        "max_tokens": 1,
        "base_tokens": 0,
        "source_request_id": "rt_0",
        "source_arrival_time_s": 100.0,
    }

    request.observed_tokens = 1
    scheduler._assign(decode_worker, request, DECODE)

    decode = decode_worker.pending_commands.pop()
    assert decode["attempt_id"] == "rt_0:d1"
    assert decode["prompt_len"] == 17
    assert decode["max_tokens"] == 3
    assert decode["base_tokens"] == 1


class _RemoteMethod:

    def __init__(self, func):
        self.func = func

    def remote(self, *args):
        return self.func(*args)


class _FakeActor:

    def __init__(self):
        self.submissions = []
        self.submit_batch = _RemoteMethod(self._submit_batch)
        self.drain_events = _RemoteMethod(self._drain_events)

    def _submit_batch(self, commands):
        self.submissions.append(commands)
        return ("submit", len(self.submissions))

    @staticmethod
    def _drain_events():
        return ("events", 0)


class _FakeRay:

    @staticmethod
    def get(ref, timeout=None):
        del timeout
        if ref[0] == "events":
            return [{"event": "iteration_log", "worker_id": 0}]
        return [{"accepted": True}]


def test_service_workers_submits_without_externally_driving_engine_steps():
    scheduler = _scheduler()
    actor = _FakeActor()
    worker = _worker(actor=actor)
    worker.pending_commands.append({"cmd": "add", "attempt_id": "rt_0:p1"})
    scheduler.workers = {0: worker}
    scheduler._ensure_ray = lambda: _FakeRay()
    handled = []
    scheduler._handle_event = handled.append

    scheduler._service_workers()

    assert actor.submissions == [[{"cmd": "add", "attempt_id": "rt_0:p1"}]]
    assert worker.pending_commands == []
    assert handled == [{"event": "iteration_log", "worker_id": 0}]
    assert not hasattr(actor, "drive")
    assert not hasattr(RayEngineWorker, "drive")
    assert hasattr(OfflineEngineWorker, "drive")


class _FakeAsyncEngine:

    def __init__(self):
        self.engine = SimpleNamespace(iteration_logs=[])
        self.is_running = False
        self.generator_exit = False

    async def generate(self, prompt, params, request_id, priority=0):
        del prompt, params, request_id, priority
        self.is_running = True
        try:
            yield SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[1])],
                finished=False,
            )
            await asyncio.sleep(0)
            yield SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[1, 2])],
                finished=True,
            )
        except GeneratorExit:
            self.generator_exit = True
            raise

    async def abort(self, request_id):
        del request_id

    def shutdown_background_loop(self):
        self.is_running = False


def test_ray_worker_consumes_async_generate_stream_without_step_calls():

    async def run_test():
        worker = RayEngineWorker.__new__(RayEngineWorker)
        worker.worker_id = 0
        worker.worker_role = PREFILL
        worker.rt_prefill_deadline_s = 10.0
        worker.engine = _FakeAsyncEngine()
        worker.SamplingParams = lambda **kwargs: kwargs
        worker._use_dummy_kv = False
        worker.active = {}
        worker._request_tasks = {}
        worker._abort_requested = set()
        worker._events = []
        worker._last_iteration_log_idx = 0
        worker._admission_diagnostics = False
        worker._shutting_down = False

        await worker.submit({
            "cmd": "add",
            "request_id": "rt_0",
            "attempt_id": "rt_0:p1",
            "kind": RT,
            "phase": PREFILL,
            "prompt_len": 4,
            "max_tokens": 2,
            "base_tokens": 0,
        })
        await asyncio.gather(*list(worker._request_tasks.values()))
        events = await worker.drain_events()

        assert [event["event"]
                for event in events] == ["token", "token", "finished"]
        token_deltas = [
            event["delta"] for event in events if event["event"] == "token"
        ]
        assert token_deltas == [1, 1]
        assert events[-1]["global_token_count"] == 2
        assert worker.active == {}
        assert not worker.engine.generator_exit
        assert not hasattr(worker.engine, "step")

    asyncio.run(run_test())


def _latency_scheduler():
    scheduler = _scheduler()
    scheduler.args = SimpleNamespace(
        window_s=10, model='smoke', load_format='auto', num_req=3,
        target_rps=1, rt_ttft_slo_ms=1000, rt_tbt_slo_ms=150)
    scheduler._elapsed = lambda: 2.0
    scheduler.rt_tokens = scheduler.be_tokens = scheduler.be_tokens_in_window = 0
    scheduler.rt_tbt_samples_ms = []
    scheduler.rt_ttft_samples_ms = []
    scheduler.rt_tbt_violations = scheduler.rt_ttft_violations = 0
    scheduler.request_errors = scheduler.history_updates = scheduler.prefill_handoffs = 0
    scheduler.stop_reason = 'all_requests_finished'
    scheduler._observe_runtime_completion = lambda request: None
    return scheduler


def test_request_tbt_includes_prefill_to_decode_gap():
    scheduler = _latency_scheduler()
    prefill, decode = _worker(role=PREFILL), _worker(worker_id=1)
    scheduler.workers = {0: prefill, 1: decode}
    request = SaturoRequest('r', RT, 16, 3, 0, active_attempt_id='r:p1')
    scheduler.all_rt = [request]
    prefill.active_attempts['r:p1'] = 'r'
    scheduler._elapsed = lambda: .1
    scheduler._handle_event(dict(event='token', worker_id=0, request_id='r',
        phase=PREFILL, attempt_id='r:p1', global_token_count=1))
    scheduler._elapsed = lambda: .11
    scheduler._handle_event(dict(event='finished', worker_id=0, request_id='r',
        phase=PREFILL, attempt_id='r:p1', global_token_count=1))
    request.active_attempt_id = 'r:d1'
    decode.active_attempts['r:d1'] = 'r'
    for count, now in [(2, .3), (3, .31)]:
        scheduler._elapsed = lambda: now
        scheduler._handle_event(dict(event='token', worker_id=1, request_id='r',
            phase=DECODE, attempt_id='r:d1', global_token_count=count))
    scheduler._elapsed = lambda: .32
    scheduler._handle_event(dict(event='finished', worker_id=1, request_id='r',
        phase=DECODE, attempt_id='r:d1', global_token_count=3))
    assert request.tbt_ms == pytest.approx(200)
    assert request.tbt_samples == 2 and request.tbt_complete
    assert request.joint_slo_met(1000, 150) is False


@pytest.mark.parametrize('counts,missing', [([3, 4], 2), ([1, 3, 4], 1)])
def test_coalesced_tokens_leave_joint_slo_unknown(counts, missing):
    scheduler = _latency_scheduler()
    request = SaturoRequest('r', RT, 16, 4, 0)
    for index, count in enumerate(counts):
        scheduler._on_token(request, {'global_token_count': count}, .1 * (index + 1))
    request.finished_s = 1
    assert request.tbt_unobserved_intervals == missing
    assert not request.tbt_complete
    assert request.joint_slo_met(1000, 150) is None


def test_duplicate_token_events_do_not_change_latency():
    scheduler = _latency_scheduler()
    request = SaturoRequest('r', RT, 16, 2, 0)
    scheduler._on_token(request, {'global_token_count': 0}, .01)
    assert request.first_token_s is None
    scheduler._on_token(request, {'global_token_count': 1}, .1)
    scheduler._on_token(request, {'global_token_count': 1}, .15)
    scheduler._on_token(request, {'global_token_count': 2}, .2)
    assert request.tbt_ms == pytest.approx(100)
    assert request.tbt_samples == 1 and request.tbt_complete
    assert request.first_token_s == .1


def test_terminal_only_progress_marks_missing_intervals():
    scheduler = _latency_scheduler()
    request = SaturoRequest('r', RT, 16, 2, 0, active_attempt_id='r:d1')
    scheduler.all_rt = [request]
    worker = _worker()
    worker.active_attempts['r:d1'] = 'r'
    scheduler.workers = {0: worker}
    scheduler._on_token(request, {'global_token_count': 1}, .1)
    scheduler._handle_event(dict(event='finished', worker_id=0, request_id='r',
        phase=DECODE, attempt_id='r:d1', global_token_count=2))
    assert request.finished_s == 2
    assert request.tbt_unobserved_intervals == 1
    assert request.joint_slo_met(1000, 150) is None


def test_request_joint_slo_survives_csv_round_trip(tmp_path):
    from saturo.experiments.analysis.summarize import summarize
    from saturo.runtime.workload import REQUEST_TBT_DEFINITION
    scheduler = _latency_scheduler()
    regular = SaturoRequest('regular', RT, 16, 3, 0)
    single = SaturoRequest('single', RT, 16, 1, 0)
    unfinished = SaturoRequest('unfinished', RT, 16, 3, 0)
    for count, now in [(1, .1), (2, .2), (3, .3)]:
        scheduler._on_token(regular, {'global_token_count': count}, now)
    scheduler._on_token(single, {'global_token_count': 1}, .1)
    regular.finished_s = single.finished_s = 1
    scheduler.all_rt = [regular, single, unfinished]
    scheduler.output_dir = tmp_path
    scheduler._write_outputs()
    with (tmp_path / 'requests.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert result['rt_joint_attainment'] == pytest.approx(2 / 3)
    assert result['rt_tbt_definition'] == REQUEST_TBT_DEFINITION
    assert rows[1]['tbt_ms'] == '' and rows[1]['tbt_complete'] == 'True'
    assert rows[2]['joint_slo_met'] == 'False'
    assert summarize(rows, 0, 10, 1000, 150)['rt_joint_attainment'] == pytest.approx(2 / 3)
    regular.tbt_unobserved_intervals = 1
    scheduler._write_outputs()
    with (tmp_path / 'requests.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert scheduler._summary()['rt_joint_attainment'] is None
    assert summarize(rows, 0, 10, 1000, 150)['rt_joint_attainment'] is None
