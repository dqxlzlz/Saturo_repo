import asyncio
import json
from types import SimpleNamespace

import pytest

from saturo.history import configured_history_source, read_redis_history
from saturo.runtime.worker import RayEngineWorker
from saturo.saturation.decode_saturation_core import (
    DefaultDecodeSaturationStateProvider)
from saturo.trace_collection import RedisTraceStore, store_from_env


@pytest.fixture
def history(monkeypatch):
    redis = pytest.importorskip("redis")
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    monkeypatch.setattr(redis.Redis, "from_url", lambda *a, **kw:
                        fakeredis.FakeRedis(server=server,
                                            decode_responses=True))
    monkeypatch.setenv("SATURO_REDIS_URL", "redis://localhost")
    monkeypatch.setenv("SATURO_HISTORY_SOURCE", "redis")
    monkeypatch.setenv("SATURO_HISTORY_KEY", "test:history")
    monkeypatch.delenv("SATURO_TRACE_KEY", raising=False)
    monkeypatch.setattr("saturo.history.time.time", lambda: 1000.0)
    store = RedisTraceStore("redis://localhost", "test:history")

    def write(request_id, arrival, output=10, model="m", finish=None):
        store.write(dict(
            run_id="run", request_id=request_id, model=model,
            arrival_time_s=arrival, prompt_len=16,
            requested_output_len=10000, output_len=output,
            finished_time_s=(arrival + 0.1 if finish is None else finish)))

    yield write
    store.client.close()


def test_history_counts_pending_arrivals_and_masks_future_completions(history):
    history("old", 399)
    history("boundary", 400, output=8)
    history("done", 500, output=12)
    history("pending", 900, output=None)
    history("future_finish", 950, output=100, finish=1100)
    history("other_model", 995, model="other")
    history("future_arrival", 1001)

    snap = read_redis_history(configured_history_source(), window_s=600,
                              model="m")
    assert len(snap.records) == 4
    assert len(snap.workload_rows()) == 2
    stats = snap.statistics()
    assert stats["arrival_rate"] == pytest.approx(4 / 600)
    assert stats["mean_output_len"] == 10
    assert len(snap.workload_rows(completed_only=False)) == 4
    assert store_from_env().key == "test:history"


def test_simulators_and_profile_loader_consume_redis_without_export(history):
    history("a", 500, output=10)
    history("b", 600, output=20)
    source = configured_history_source()
    from saturo.profiling.real_profile_collector import load_history_trace
    from saturo.simulators.prefill_simulator.trace_loader import load_rt_trace
    requests = load_rt_trace(source, 2, 1.0, default_runtime_s=0.01)
    profile_requests = load_history_trace(source, num_req=2, target_rps=1.0)
    assert [r.prompt_len for r in requests] == [16, 16]
    assert [r.output_len for r in profile_requests] == [10, 20]
    from saturo.simulators.decode_simulator.trace_loader import (
        load_rt_trace as load_decode)
    decode = load_decode(source, duration_s=1000, time_scale=1)
    assert [r.target_output_len for r in decode] == [10, 20]


def test_decode_uses_history_averages_and_expires_to_local_observations():
    provider = DefaultDecodeSaturationStateProvider(load_predictor=False)
    provider.set_history_statistics(dict(
        arrival_rate=2.0, mean_prompt_len=64, mean_output_len=12,
        completed_requests=10, valid_until_s=110))
    future = list(provider.get_future_arrivals(100, 1))
    assert [r.arrival_time_s for r in future] == [100.5, 101]
    assert all(r.prompt_len == 64 and r.remaining_tokens == 12 for r in future)
    assert list(provider.get_future_arrivals(111, 1)) == []


def test_worker_applies_history_to_scheduler_without_engine_step_or_redis():
    applied = []
    config = SimpleNamespace(p_pool_be_chunk_ms=100)
    scheduler = SimpleNamespace(
        scheduler_config=config,
        decode_saturation=SimpleNamespace(provider=SimpleNamespace(
            set_history_statistics=applied.append)))
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._shutting_down = False
    worker.engine = SimpleNamespace(engine=SimpleNamespace(scheduler=[scheduler]))

    async def run():
        worker.worker_role = "prefill"
        await worker.submit({"cmd": "history", "p_chunk_ms": 75})
        worker.worker_role = "decode"
        await worker.submit({"cmd": "history", "statistics": {"rate": 2}})

    asyncio.run(run())
    assert config.p_pool_be_chunk_ms == 75
    assert applied == [{"rate": 2}]


def test_controller_divides_load_by_decode_pool_and_optimizes_p_once(
        history, monkeypatch):
    from saturo.runtime.history import RuntimeHistoryController
    history("a", 500)
    history("b", 600)
    # Exercise one refresh deterministically, without starting a daemon.
    controller = RuntimeHistoryController.__new__(RuntimeHistoryController)
    controller.args = SimpleNamespace(model="m")
    controller.source = configured_history_source()
    controller.window_s, controller.refresh_s = 600, 40
    controller.worker_specs = [
        {"role": role, "config_name": "tp1_pp1"}
        for role in ("prefill", "prefill", "decode", "decode")]
    calls = []

    def optimize(snapshot, spec, count):
        calls.append((len(snapshot.records), count))
        return {"chunk_ms": 80.0}

    monkeypatch.setattr(controller, "_optimize_p_budget", optimize)
    update = controller.refresh_once()
    assert update["decode_statistics"]["arrival_rate"] == pytest.approx(1 / 600)
    assert calls == [(2, 2)]
    assert update["p_budgets"]["tp1_pp1"]["chunk_ms"] == 80


def test_planner_invalidates_workload_tables_but_reuses_matching_snapshot(
        history, tmp_path, monkeypatch):
    from saturo.resource_allocation.planner import SaturoResourcePlanner
    history("a", 500)
    profile = tmp_path / "allocation.json"
    data = {"configs": {"tp1_pp1": {"gpus": 1, "tp": 1, "pp": 1}},
            "pools": {"P": {}, "D": {}, "T": {}}}
    profile.write_text(json.dumps(data))
    args = SimpleNamespace(output_dir=str(tmp_path), model="m",
                           resource_config_filter="tp1_pp1", rt_workload_csv="")
    planner = SaturoResourcePlanner(args, lambda *a, **kw: None)
    collected = []

    def collect(path, configs_override=None):
        collected.append(configs_override)
        path.write_text(json.dumps(data))

    monkeypatch.setattr(planner, "_collect_real_allocation_profile", collect)
    planner._load_or_collect_profile(profile)
    assert len(collected) == 1
    planner._load_or_collect_profile(profile)
    assert len(collected) == 1
    history("b", 600)
    planner._load_or_collect_profile(profile)
    assert len(collected) == 2
    assert "history" in json.loads(profile.read_text())


def test_profile_engine_uses_frozen_history_rows(history):
    from saturo.profiling.real_profile_collector import load_history_trace
    history("a", 500, output=8)
    snapshot = read_redis_history(configured_history_source(), model="m")
    history("b", 600, output=12)
    requests = load_history_trace("not-a-file", num_req=1, target_rps=1,
                                  rows=snapshot.workload_rows())
    assert len(requests) == 1
    assert requests[0].output_len == 8


def test_qrf_can_train_from_length_only_history(history, tmp_path):
    from saturo.predictors.history_training import train_history_qrf
    from saturo.predictors.request_length_predictor import load_length_predictor
    for i in range(4):
        history(str(i), 500 + i, output=10 + i)
    output = tmp_path / "length_qrf.joblib"
    train_history_qrf(configured_history_source(), output, min_requests=4,
                       n_estimators=2, max_steps=4, model="m")
    predictor = load_length_predictor(output)
    assert predictor.config.feature_mode == "len"
    assert predictor.metadata["requests"] == 4
    assert predictor.predict_remaining(prompt_len=16, generated_len=2) >= 0


def test_resource_history_count_is_independent_of_live_replay_count(
        history, tmp_path, monkeypatch):
    from saturo.profiling import real_profile_collector as collector
    history("a", 500)
    history("b", 600)
    args = SimpleNamespace(model="m", rt_workload_csv="", num_req=999)
    seen = []

    def profile(**kwargs):
        local = kwargs["args"]
        seen.append((local.num_req, len(local._history_rows)))
        return {}

    monkeypatch.setattr(collector, "_resource_rps_values", lambda a: [1])
    monkeypatch.setattr(collector, "run_profile_engine", profile)
    monkeypatch.setattr(collector, "build_allocation_profile", lambda **kw: {})
    collector.collect_real_allocation_profile(
        args, tmp_path / "profile.json", ["tp1_pp1"])
    assert seen == [(2, 2)]
    assert args.num_req == 999
