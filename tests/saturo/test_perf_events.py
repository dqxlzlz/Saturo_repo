import json
from types import SimpleNamespace

from saturo.runtime.scheduler import SaturoGlobalScheduler


def make_scheduler(tmp_path, enabled=True):
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler._perf_events_enabled = enabled
    scheduler.output_dir = tmp_path
    scheduler._elapsed = lambda: 1.5
    return scheduler


def test_perf_events_preserve_event_clocks_and_flatten_decision(tmp_path):
    scheduler = make_scheduler(tmp_path)
    worker = SimpleNamespace(worker_id=2, role="decode")
    events = [
        dict(event="token", time_s=80.1, request_id="rt0", delta=1,
             stream_put_to_worker_ms=1.2),
        dict(event="iteration_log", step_started_s=79.9,
             scheduler_wall_ms=20.0, actual_time_ms=3.0,
             prefill_reqs=[(1, 32)], decode_reqs=[(1, 64)],
             planner_timing=dict(active_state_ms=30.0, planner_total_ms=35.0),
             decode_saturation_decision=dict(target_interval_ms=100.0,
                                             current_rt_blocks=50)),
        dict(event="finished", request_id="rt0", time_s=80.2),
    ]
    scheduler._write_perf_events(worker, events, 80.3)
    rows = [json.loads(line) for line in
            (tmp_path / "perf_events.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["time_s"] == 80.1
    assert rows[0]["stream_put_to_worker_ms"] == 1.2
    assert rows[1]["step_started_s"] == 79.9
    assert rows[1]["decision_target_interval_ms"] == 100.0
    assert rows[1]["decision_current_rt_blocks"] == 50
    assert rows[1]["planner_active_state_ms"] == 30.0
    assert rows[1]["planner_total_ms"] == 35.0
    assert "prefill_reqs" not in rows[1] and "decode_reqs" not in rows[1]
    assert "decode_saturation_decision" not in rows[1]
    assert all(row["worker_id"] == 2 and row["worker_role"] == "decode"
               and row["scheduler_elapsed_s"] == 1.5
               and row["scheduler_received_perf_s"] == 80.3 for row in rows)
    # Diagnostics must not rewrite the event consumed by production metrics.
    assert "scheduler_elapsed_s" not in events[0]
    assert events[1]["decode_reqs"] == [(1, 64)]


def test_perf_events_are_disabled_by_default(tmp_path):
    scheduler = make_scheduler(tmp_path, enabled=False)
    scheduler._write_perf_events(SimpleNamespace(worker_id=1, role="prefill"),
                                 [dict(event="token", time_s=1.0)], 2.0)
    assert not (tmp_path / "perf_events.jsonl").exists()


def test_perf_events_omit_unrelated_empty_log_records(tmp_path):
    scheduler = make_scheduler(tmp_path)
    scheduler._write_perf_events(SimpleNamespace(worker_id=1, role="prefill"),
                                 [dict(event="log", message="hello")], 2.0)
    assert not (tmp_path / "perf_events.jsonl").exists()
