"""CPU tests for observed-history rebuilds and lossless deployment gates."""
import argparse
import dataclasses
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from saturo.history import HistorySnapshot
from saturo.resource_allocation.planner import SaturoResourcePlanner
from saturo.runtime import reallocation as module
from saturo.runtime.reallocation import (
    DeploymentTransaction, DrainCensus, InsufficientHistory, PlacementSpec,
    PeriodicReallocationService, ReallocationProposal, freeze_history,
    validate_measured_timing)


def snapshot(output=8):
    return HistorySnapshot("live", "org/model", 100, 110, [
        {"run_id": "r", "request_id": "a", "arrival_time_s": 101,
         "prompt_len": 16, "output_len": output, "finished_time_s": 105},
        {"run_id": "r", "request_id": "b", "arrival_time_s": 102,
         "prompt_len": 32, "output_len": output + 1, "finished_time_s": 109},
        {"run_id": "r", "request_id": "pending", "arrival_time_s": 108,
         "prompt_len": 64, "output_len": None, "max_tokens": 9999,
         "finished_time_s": None},
    ], source="live_ingress")


def table(mu=2):
    return {"configs": {"tp1_pp1": {"gpus": 1, "tp": 1, "pp": 1}},
            "pools": {pool: {"tp1_pp1": {
                **({"mu": mu} if pool != "T" else {}),
                "saturation": [{"rt_rps": 0, "be_throughput": 1},
                               {"rt_rps": mu, "be_throughput": 0.5}]}}
                for pool in ("P", "D", "T")}}


def args(tmp_path):
    return argparse.Namespace(output_dir=str(tmp_path), model="org/model", dtype="half",
        total_gpus=3, target_rps=9, resource_config_filter="tp1_pp1", gpu_ids="0,1,2",
        resource_allocation_policy="saturation_aware", enable_prefill_saturation=True,
        enable_decode_saturation=True, resource_reallocation_interval_s=1,
        resource_reallocation_build_timeout_s=10, resource_reallocation_min_completed=2)


def timing(tmp_path):
    path = tmp_path / "timing.json"
    path.write_text(json.dumps({"time_predictor": {
        "num_layers": 2,
        "profile_identity": {"model": "org/model", "dtype": "torch.float16",
            "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
            "quantization": "unknown", "gpu_name": "test GPU"},
        "meta": {"repeat": 3, "pass_row_counts": {
            "linear": 1, "prefill_attn": 1, "decode_attn": 1}}},
        "kv_cache_block": {"block_size": 16, "total_gpu_blocks": 1000}}))
    return path


@pytest.fixture
def proposal(tmp_path):
    plan_path = tmp_path / "plan.json"
    specs = (PlacementSpec("prefill", 1, 1, gpu_id="0", config_name="tp1_pp1"),
             PlacementSpec("decode", 1, 1, gpu_id="1", config_name="tp1_pp1"))
    plan = {"placements": [dataclasses.asdict(p) for p in specs]}
    module._write(plan_path, plan)
    profile = tmp_path / "profile.json"
    module._write(profile, table())
    manifest_path = tmp_path / "manifest.json"
    module._write(manifest_path, {"files": [
        {"path": str(path), "sha256": module._sha(path)} for path in (plan_path, profile)]})
    return ReallocationProposal("test", 4, str(profile), str(plan_path), str(manifest_path),
                                specs, module._json(plan), module._sha(manifest_path))


def test_history_masks_future_and_requested_output_without_mutation():
    original = snapshot()
    original.records.append({"run_id": "r", "request_id": "futurefinish",
        "arrival_time_s": 109, "prompt_len": 8, "output_len": 700,
        "finished_time_s": 111})
    frozen = freeze_history(original, min_completed=2)
    assert frozen["statistics"]["arrival_rate"] == 0.4
    assert frozen["statistics"]["completed_requests"] == 2
    assert frozen["records"][-1]["output_len"] is None
    assert "max_tokens" not in frozen["records"][-2]
    assert original.records[-1]["output_len"] == 700
    original.records[0]["output_len"] = 1234
    assert frozen["records"][0]["output_len"] == 8


def test_cold_start_minimum_and_duplicate_gate():
    with pytest.raises(InsufficientHistory, match="minimum=8"):
        freeze_history(snapshot(), min_completed=8)
    old = snapshot()
    old.records.append(dict(old.records[0]))
    with pytest.raises(ValueError, match="duplicate"):
        freeze_history(old)
    with pytest.raises(InsufficientHistory):
        freeze_history(dataclasses.replace(snapshot(), start_s=110))


@pytest.mark.parametrize("change,message", [
    ({"model": "different/model"}, "identity"),
    ({"tensor_parallel_size": 2}, "tensor_parallel"),
    ({"pipeline_parallel_size": 2}, "pipeline_parallel"),
    ({"dtype": "torch.bfloat16"}, "dtype"),
    ({"quantization": "awq"}, "quantization"),
])
def test_timing_identity_is_checked_not_filename(tmp_path, change, message):
    path = timing(tmp_path)
    data = json.loads(path.read_text())
    data["time_predictor"]["profile_identity"].update(change)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        validate_measured_timing(path, model="org/model", dtype="half",
                                 config=SimpleNamespace(tp=1, pp=1))


def test_timing_requires_measured_provenance_and_kv_capacity(tmp_path):
    path = timing(tmp_path)
    data = json.loads(path.read_text())
    validate_measured_timing(path, model="org/model", dtype="half",
                             config=SimpleNamespace(tp=1, pp=1))
    data["time_predictor"]["meta"]["repeat"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="measured"):
        validate_measured_timing(path, model="org/model", config=SimpleNamespace(tp=1, pp=1))
    data["time_predictor"]["meta"]["repeat"] = 1
    data["kv_cache_block"]["total_gpu_blocks"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="KV"):
        validate_measured_timing(path, model="org/model", config=SimpleNamespace(tp=1, pp=1))


def test_existing_full_iteration_fit_provenance_needs_no_fake_repeat(tmp_path):
    path = timing(tmp_path)
    data = json.loads(path.read_text())
    inner = data["time_predictor"]
    inner.update(model_type="iteration_level", n_samples=128,
        calibration_shapes=[[16, 1], [128, 4]],
        features=["prefill_tokens", "prefill_sum_ctx", "n_decode_reqs", "decode_sum_ctx"],
        coef=[0.002, 0.0, 0.07, 0.0001], intercept=24.0,
        meta={"total_model_layers": 2})
    path.write_text(json.dumps(data))
    result = validate_measured_timing(path, model="org/model", dtype="float16",
                                      config=SimpleNamespace(tp=1, pp=1))
    assert result["measurement"]["schema"] == "measured_full_model_iteration_fit"
    assert "repeat" not in result["measurement"]["meta"]
    inner["calibration_shapes"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="sample/shape"):
        validate_measured_timing(path, model="org/model", config=SimpleNamespace(tp=1, pp=1))


def test_staged_solver_rejects_duplicate_physical_gpus(tmp_path):
    supplied = args(tmp_path); supplied.gpu_ids = "0,0,1"
    planner = SaturoResourcePlanner(supplied, lambda *a, **kw: None)
    with pytest.raises(ValueError, match="duplicate GPU"):
        planner.solve_staged_profile("unused", total_gpus=3, target_rps=1)


def test_staged_solver_is_real_and_does_not_mutate_live_state(tmp_path):
    supplied_args = args(tmp_path)
    old = vars(supplied_args).copy()
    profile = tmp_path / "new.json"
    module._write(profile, table())
    before = profile.read_bytes()
    planner = SaturoResourcePlanner(supplied_args, lambda role, tp, pp, ep, **kw:
        PlacementSpec(role, tp, pp, ep, **kw))
    plan = planner.solve_staged_profile(profile, total_gpus=3, target_rps=0.3)
    assert plan["allocation"]["p"]["rt_workload"] == 0.3
    assert plan["allocation"]["d"]["rt_workload"] == 0.3
    assert sum(p["tensor_parallel_size"] * p["pipeline_parallel_size"]
               for p in plan["placements"]) <= 3
    assert all(p["config_name"] == "tp1_pp1" for p in plan["placements"])
    assert vars(supplied_args) == old
    assert profile.read_bytes() == before
    assert not (tmp_path / "resource_worker_plan.json").exists()


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(DrainCensus)
                                  if f.name != "retained_local_be"])
def test_each_pending_census_component_prevents_restart(proposal, field):
    tx = DeploymentTransaction(proposal, current_generation=4, gpu_budget=2)
    assert tx.advance(DrainCensus(**{field: 1})) == "wait"
    assert tx.phase == "draining"
    assert tx.advance(DrainCensus()) == "stop_old"


def test_restart_requires_released_old_and_verified_new_claims(proposal):
    tx = DeploymentTransaction(proposal, current_generation=4, gpu_budget=2)
    assert tx.advance() == "stop_old"
    assert tx.advance(old_gpu_claims=1) == "wait"
    assert tx.advance(old_gpu_claims=0) == "start_new"
    assert tx.advance(new_ready=True, new_gpu_claims=1) == "wait"
    assert tx.advance(new_ready=True, new_gpu_claims=2) == "commit"
    assert tx.generation == 5
    assert tx.advance() == "wait"


def test_retained_be_is_nonblocking_only_for_same_engine_transaction(proposal):
    census = DrainCensus(retained_local_be=3)
    assert census.drained
    tx = DeploymentTransaction(proposal, current_generation=4, gpu_budget=2,
        in_place=True, old_placements=proposal.placements)
    assert tx.advance(census) == "apply_roles"
    restart = DeploymentTransaction(proposal, current_generation=4, gpu_budget=2)
    assert restart.advance(census) == "wait"
    with pytest.raises(ValueError, match="negative"):
        DrainCensus(retained_local_be=-1).drained


def test_controls_only_validates_unchanged_roles_and_can_refresh_while_busy(proposal):
    tx = DeploymentTransaction(proposal, current_generation=4, gpu_budget=2,
        in_place=True, controls_only=True, old_placements=proposal.placements)
    assert tx.phase == "verifying_controls"
    assert tx.advance(DrainCensus(active_attempts=5, reservations=3), controls_ready=False) == "wait"
    assert tx.advance(DrainCensus(active_attempts=5, reservations=3), controls_ready=True) == "commit"
    changed = tuple(dataclasses.replace(p, role="throughput") for p in proposal.placements)
    with pytest.raises(ValueError, match="identical roles"):
        DeploymentTransaction(proposal, current_generation=4, gpu_budget=2,
            in_place=True, controls_only=True, old_placements=changed)


def test_drain_timeout_resumes_old_without_stop(proposal):
    now = [0]
    tx = DeploymentTransaction(proposal, current_generation=4, gpu_budget=2,
                                drain_timeout_s=3, clock=lambda: now[0])
    now[0] = 3
    assert tx.advance(DrainCensus(active_attempts=1)) == "resume_old"
    assert tx.phase == "resumed_old" and tx.generation == 4
    assert all(e["action"] != "stop_old" for e in tx.events)


def test_partial_new_failure_requires_cleanup_before_rollback(proposal):
    tx = DeploymentTransaction(proposal, current_generation=4, gpu_budget=2)
    tx.advance()
    tx.advance(old_gpu_claims=0)
    assert tx.advance(failure="one actor failed") == "stop_new"
    assert tx.advance(new_gpu_claims=1) == "wait"
    assert tx.advance(new_gpu_claims=0) == "start_old"
    assert tx.advance(rollback_ready=True) == "resume_old"
    assert tx.phase == "rolled_back" and tx.generation == 4


def test_rollback_timeout_is_terminal_failure(proposal):
    now = [0]
    tx = DeploymentTransaction(proposal, current_generation=4, gpu_budget=2,
                               start_timeout_s=1, clock=lambda: now[0])
    tx.advance(); tx.advance(old_gpu_claims=0)
    tx.advance(failure="new failed"); tx.advance(new_gpu_claims=0)
    now[0] = 2
    assert tx.advance() == "fail"
    assert tx.phase == "failed"


def test_in_place_role_ack_commit_and_rollback(proposal):
    old = tuple(dataclasses.replace(p, role="throughput") for p in proposal.placements)
    tx = DeploymentTransaction(proposal, current_generation=4, gpu_budget=2,
                                in_place=True, old_placements=old)
    assert tx.advance() == "apply_roles"
    assert tx.advance() == "wait"
    assert tx.advance(roles_ready=True) == "commit"
    tx = DeploymentTransaction(proposal, current_generation=4, gpu_budget=2,
                                in_place=True, old_placements=old)
    tx.advance()
    assert tx.advance(failure="partial role ACK") == "restore_roles"
    assert tx.advance(rollback_ready=True) == "resume_old"
    assert tx.generation == 4


def test_stale_overbudget_changed_config_or_tampered_proposal_rejected(proposal):
    with pytest.raises(ValueError, match="stale"):
        DeploymentTransaction(proposal, current_generation=5, gpu_budget=2)
    with pytest.raises(ValueError, match="budget"):
        DeploymentTransaction(proposal, current_generation=4, gpu_budget=1)
    with pytest.raises(ValueError, match="multiset"):
        DeploymentTransaction(proposal, current_generation=4, gpu_budget=2,
                              in_place=True, old_placements=())
    Path(proposal.profile_path).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        proposal.verify()


def test_periodic_poll_is_nonblocking_and_does_not_overlap(tmp_path, monkeypatch):
    now = [0]
    release = threading.Event()
    entered = threading.Event()
    calls = []
    service = PeriodicReallocationService(args(tmp_path), lambda **kw: snapshot(),
                                          clock=lambda: now[0])
    def build(generation):
        calls.append(generation)
        entered.set()
        release.wait(2)
    monkeypatch.setattr(service, "_build", build)
    assert service.poll(4) is None
    now[0] = 1
    started = time.monotonic()
    assert service.poll(4) is None
    assert time.monotonic() - started < 0.1
    assert entered.wait(1)
    now[0] = 10
    service.poll(5)
    assert calls == [4]
    service.close(); release.set()


def test_service_uses_configured_history_window_and_records_skip(tmp_path):
    supplied = args(tmp_path)
    supplied.resource_reallocation_history_window_s = 77
    seen = []
    def loader(**kw):
        seen.append(kw)
        return dataclasses.replace(snapshot(), records=[])
    service = PeriodicReallocationService(supplied, loader, wall_clock=lambda: 110)
    service._build(4)
    assert seen == [{"window_s": 77, "end_s": 110}]
    assert service.drain_events()[-1]["event"] == "build_skipped"
    service.close()


def test_cpu_builder_freezes_inputs_rebuilds_fresh_table_and_runs_solver(tmp_path, monkeypatch):
    from saturo.profiling import real_profile_collector as collector
    from saturo.predictors import time_predictor
    cache = timing(tmp_path)
    be = tmp_path / "be.csv"; be.write_text("prompt_len,output_len\n16,8\n")
    qrf = tmp_path / "qrf.joblib"; qrf.write_bytes(b"frozen-artifact")
    supplied_args = args(tmp_path)
    supplied_args.be_workload_csv = str(be)
    supplied_args.model_len_predictor = str(qrf)
    built = []
    def model(**kwargs):
        local = kwargs["args"]
        built.append((local.num_req, local.target_rps, local._history_metadata))
        assert local.num_req == 2 and local.target_rps == 0.3
        assert Path(local.model_len_predictor).read_bytes() == qrf.read_bytes()
        assert kwargs["summaries"]["tp1_pp1"]["runtime_profile"].endswith("tp1_pp1.timing.json")
        return table(mu=2 if len(built) == 1 else 0.5)
    def forbidden(*a, **kw):
        raise AssertionError("GPU collection is forbidden in a background rebuild")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(collector, "build_allocation_profile", model)
    selected_rates = []
    def selected_profile(**kw):
        selected_rates.append(kw["rps_values"][0])
        return {"mu": 2, "be_chunk_latency_ms": 17,
                "saturation": [{"rt_rps": kw["rps_values"][0], "be_throughput": 1}]}
    monkeypatch.setattr(collector, "_profile_p_pool", selected_profile)
    monkeypatch.setattr(collector, "collect_real_allocation_profile", forbidden)
    monkeypatch.setattr(collector, "run_profile_engine", forbidden)
    monkeypatch.setattr(time_predictor, "TimePredictor", lambda p:
        SimpleNamespace(predict_forward_ms=lambda prefill, decode: 7.0))
    monkeypatch.setattr(module, "_source_manifest", lambda root: [])
    manifests = []
    for index in range(2):
        directory = tmp_path / str(index); directory.mkdir()
        spec = {"proposal_id": str(index), "base_generation": 4,
                "args": vars(supplied_args),
                "history": freeze_history(snapshot(output=8 + index), min_completed=2),
                "timing_profiles": {"tp1_pp1": str(cache)}}
        spec_path = directory / "build_spec.json"; module._write(spec_path, spec)
        manifests.append(json.loads(module.build_reallocation(spec_path).read_text()))
        assert (directory / "resource_worker_plan.json").is_file()
        plan = json.loads((directory / "resource_worker_plan.json").read_text())
        assert selected_rates[-1] == plan["allocation"]["p"]["per_instance_rt_workload"]
        assert plan["deployment_controls"]["p_budgets"]["tp1_pp1"]["chunk_ms"] == 17
        assert (directory / "allocation_profiles.json").is_file()
    assert len(built) == 2
    assert manifests[0]["history"]["fingerprint"] != manifests[1]["history"]["fingerprint"]
    assert manifests[0]["gpu_profiling"] is False
    assert manifests[0]["inputs"][-1]["source_sha256"] == module._sha(cache)


def test_builder_refuses_gpu_visibility_before_importing_profiler(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        module.build_reallocation(tmp_path / "nonexistent.json")


def test_intentional_shutdown_sigterm_is_cancelled_not_failed(tmp_path, monkeypatch):
    service = PeriodicReallocationService(args(tmp_path), lambda **kw: snapshot())
    class CancelledProcess:
        def wait(self, timeout):
            service._closed.set()  # close terminates the running CPU subprocess
            return -15
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: CancelledProcess())
    service._build(4)
    events = service.drain_events()
    assert events[-1]["event"] == "build_cancelled"
    assert events[-1]["returncode"] == -15
    assert events[-1]["reason"] == "service_shutdown"
    assert not any(e["event"] in ("build_failed", "proposal_ready") for e in events)
    assert service._results.empty()


def test_close_waits_boundedly_for_builder_cancellation_event(tmp_path):
    service = PeriodicReallocationService(args(tmp_path), lambda **kw: snapshot())
    entered = threading.Event()
    def build():
        entered.set()
        service._closed.wait(1)
        service._event("build_cancelled", reason="service_shutdown")
    service._thread = threading.Thread(target=build)
    service._thread.start()
    assert entered.wait(1)
    service.close()
    assert not service._thread.is_alive()
    assert service.drain_events()[-1]["event"] == "build_cancelled"
