import csv
from collections import deque
import threading
from types import SimpleNamespace

import pytest

from saturo.runtime.scheduler import SaturoGlobalScheduler
from saturo.runtime.worker import WorkerState
from saturo.runtime.workload import BE, DECODE, PREFILL, RT, SaturoRequest
from saturo.trace_collection import (RedisTraceStore, TraceRecorder,
                                     store_from_env)


@pytest.fixture
def store(monkeypatch):
    redis = pytest.importorskip("redis")
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis.Redis, "from_url", lambda *a, **kw: client)
    result = RedisTraceStore("redis://localhost", "test:rt", ttl_s=60)
    yield result
    client.close()


def _record(request_id="rt_0", run_id="run_1", arrival=100.0, output_len=None):
    return dict(run_id=run_id, request_id=request_id, arrival_time_s=arrival,
                prompt_len=16, output_len=output_len)


def test_arrival_completion_upsert_and_inclusive_window(store):
    store.write(_record(arrival=99.0))
    store.write(_record("rt_1", arrival=100.0))
    store.write(_record("rt_2", arrival=101.0))
    store.write(_record("rt_1", arrival=100.0, output_len=4))

    rows = store.read_window(100.0, 101.0)

    assert [row["request_id"] for row in rows] == ["rt_1", "rt_2"]
    assert [row["output_len"] for row in rows] == [4, None]
    assert store.client.zcard(store.index_key) == 3
    assert store.client.hlen(store.records_key) == 3
    assert 0 < store.client.ttl(store.index_key) <= 60
    assert 0 < store.client.ttl(store.records_key) <= 60


def test_reused_request_ids_in_separate_runs_are_independent(store):
    store.write(_record(run_id="first", output_len=4))
    store.write(_record(run_id="second", arrival=101.0, output_len=5))

    assert len(store.read_window(0, 200)) == 2


def test_export_replays_completed_lengths_without_inventing_pending_output(
        store, tmp_path):
    store.write(_record("pending", arrival=100.0))
    store.write(_record("done_1", arrival=101.5, output_len=4))
    store.write(_record("done_2", arrival=104.0, output_len=6))
    path = tmp_path / "history.csv"

    assert store.export_csv(path, 100, 104) == 2
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["output_len"] for row in rows] == ["4", "6"]
    assert [float(row["arrival_offset_s"]) for row in rows] == [1.5, 4.0]
    from saturo.runtime.workload import load_workload_csv
    replay = load_workload_csv(str(path), RT)
    values = [(r.prompt_len, r.output_len, r.arrival_offset_s) for r in replay]
    assert values == [(16, 4, 1.5), (16, 6, 4.0)]


def test_recorder_snapshots_requests_before_background_write():
    entered = threading.Event()
    release = threading.Event()
    records = []

    def write(record):
        entered.set()
        assert release.wait(2)
        records.append(record)

    store = SimpleNamespace(
        write=write, client=SimpleNamespace(close=lambda: None))
    recorder = TraceRecorder(store, "model", started_at_s=100)
    request = SaturoRequest("rt_0", RT, 16, 4, 1.5)
    try:
        recorder.record(request)
        assert entered.wait(2)
        # Mutating the request must not turn the queued arrival into a
        # completed record or leak a mutable request into the writer.
        request.observed_tokens = 4
        request.finished_s = 2.5
        recorder.record(request)
    finally:
        release.set()
        recorder.close()

    assert [r["output_len"] for r in records] == [None, 4]
    assert [r["arrival_time_s"] for r in records] == [101.5, 101.5]
    assert records[1]["finished_time_s"] == 102.5
    assert records[0]["run_id"] == records[1]["run_id"]


def test_redis_failure_does_not_escape_to_scheduler_or_log_credentials(caplog):
    def write(record):
        raise ConnectionError("redis://private:secret@localhost")

    store = SimpleNamespace(
        write=write, client=SimpleNamespace(close=lambda: None))
    recorder = TraceRecorder(store, "model", started_at_s=100)
    recorder.record(SaturoRequest("rt_0", RT, 16, 4, 0))
    recorder.close()

    assert recorder.failed_records == 1
    assert "ConnectionError" in caplog.text
    assert "secret" not in caplog.text


def test_collection_disabled_without_url(monkeypatch):
    monkeypatch.delenv("SATURO_REDIS_URL", raising=False)
    assert store_from_env() is None


def test_scheduler_records_arrival_and_final_completion_only():
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler.pending_prefill = deque()
    snapshots = []
    scheduler.trace_recorder = SimpleNamespace(
        record=lambda r: snapshots.append((r.request_id, r.observed_tokens)))
    scheduler._elapsed = lambda: 2.0
    scheduler.next_rt_idx = 0
    request = SaturoRequest("rt_0", RT, 16, 4, 1.0)
    scheduler.all_rt = [request]
    scheduler.all_be = [SaturoRequest("be_0", BE, 16, 4, 0)]
    scheduler.prefill_handoffs = 0
    worker = WorkerState(0, PREFILL, "local", "tp1_pp1", 1, 1, None, ready=True)
    scheduler.workers = {
        0: worker,
        1: WorkerState(1, DECODE, "local", "tp1_pp1", 1, 1, None, ready=True),
    }

    scheduler._inject_due_requests()
    assert snapshots == [("rt_0", 0)]
    event = dict(event="finished", worker_id=0, request_id="rt_0",
                 attempt_id=request.active_attempt_id, phase=PREFILL,
                 global_token_count=1)
    scheduler.pending_decode = []
    scheduler._handle_event(event)
    assert snapshots == [("rt_0", 0)]

    scheduler._assign(worker, request, DECODE)
    event.update(phase=DECODE, attempt_id=request.active_attempt_id,
                 global_token_count=4)
    scheduler._handle_event(event)
    assert snapshots == [("rt_0", 0), ("rt_0", 4)]

    event.update(request_id="be_0")
    scheduler._handle_event(event)
    assert snapshots == [("rt_0", 0), ("rt_0", 4)]


def test_single_token_rt_completion_is_recorded_in_prefill():
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    recorded = []
    scheduler.trace_recorder = SimpleNamespace(record=recorded.append)
    scheduler._elapsed = lambda: 1.0
    request = SaturoRequest("rt_one", RT, 16, 1, 0)
    scheduler.all_rt, scheduler.all_be = [request], []
    scheduler.prefill_handoffs = 0
    scheduler.workers = {
        0: WorkerState(0, PREFILL, "local", "tp1_pp1", 1, 1, None)}

    scheduler._handle_event(dict(event="finished", worker_id=0,
                                 request_id="rt_one", phase=PREFILL,
                                 global_token_count=1))

    assert recorded == [request]
    assert request.finished_s == 1.0


def test_export_does_not_leak_completions_after_window_end(store, tmp_path):
    record = _record(arrival=100, output_len=4)
    record["finished_time_s"] = 200
    store.write(record)
    assert store.export_csv(tmp_path / "past.csv", 90, 150) == 0
    assert store.export_csv(tmp_path / "later.csv", 90, 250) == 1
