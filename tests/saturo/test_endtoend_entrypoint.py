import io
import json
import sys
from types import SimpleNamespace

import pytest

from saturo.cli import endtoend_experiment as e2e


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_entrypoint_forwards_runtime_options_and_rejects_partial_success(
        monkeypatch, tmp_path, status):
    monkeypatch.setattr(sys, "argv", ["e2e", "--output-dir", str(tmp_path),
        "--model", "model", "--num-req", "24", "--target-rps", "4",
        "--resource-config-filter", "tp1_pp1", "--model-len-predictor", "qrf.joblib",
        "--max-model-len", "1024", "--max-num-seqs", "16",
        "--rt-workload-csv", "rt.csv", "--be-workload-csv", "be.csv"])
    args = e2e.parse_args()
    assert args.load_format == "auto"
    assert args.resource_decode_strategies == "progressive"
    def launch(cmd, **kwargs):
        for name, value in (("--model-len-predictor", str(e2e.repo_root()/"qrf.joblib")),
                            ("--max-model-len", "1024"),
                            ("--max-num-seqs", "16")):
            assert cmd[cmd.index(name) + 1] == value
        (tmp_path / "summary.json").write_text(json.dumps(
            {"status": status, "stop_reason": "max_wall_s"}))
        return SimpleNamespace(stdout=io.StringIO("live log\n"), returncode=0,
                               wait=lambda: 0, poll=lambda: 0)
    monkeypatch.setattr(e2e.subprocess, "Popen", launch)
    if status == "completed":
        assert e2e.run(args)["status"] == status
    else:
        with pytest.raises(RuntimeError, match="Incomplete e2e"):
            e2e.run(args)
    assert (tmp_path / "runner_stdout.log").read_text() == "live log\n"
    assert (tmp_path / "runner_summary.json").exists()


def test_resource_decode_profile_uses_fixed_num_window_and_qrf(monkeypatch, tmp_path):
    from saturo.profiling import real_profile_collector as collector
    from saturo.simulators.decode_simulator import run_decode_simulator as decode
    seen = []
    def simulate(args, **kwargs):
        seen.append(args)
        return {"results": {"RPS": {"progressive": {
            "be_saturation_req_per_s": 1, "rt_slo_violations_total": 0}}}}
    monkeypatch.setattr(decode, "run_decode_profile", simulate)
    monkeypatch.setattr(collector, "_profile_latency_model", lambda *args: "time.json")
    args = SimpleNamespace(num_req=24, max_model_len=1024,
                           model_len_predictor="qrf.joblib")
    collector._profile_d_pool(args=args, config=SimpleNamespace(name="tp1_pp1"),
                             summaries={}, history_profile=tmp_path/"history.csv",
                             rps_values=[2, 4], config_dir=tmp_path)
    assert [a.duration_s for a in seen] == [12, 6]
    assert all(a.model_len_predictor == "qrf.joblib" for a in seen)
    assert all(a.model_len_predictor_type == "qrf" for a in seen)
    assert all(a.strategies == "progressive" for a in seen)


def test_impossible_p_pool_budget_fails_before_warmup_can_hang():
    from saturo.runtime.worker import _validate_p_pool_budget
    core = SimpleNamespace(
        scheduler_config=SimpleNamespace(p_pool_be_chunk_ms=30.69,
                                         p_pool_be_chunk_size=0),
        time_predictor=SimpleNamespace(available=True,
            predict_forward_ms=lambda *args: 36.08))
    with pytest.raises(ValueError, match="would stall BE requests"):
        _validate_p_pool_budget(core, "prefill")
    core.scheduler_config.p_pool_be_chunk_ms = 40
    _validate_p_pool_budget(core, "prefill")


def test_resource_profile_never_substitutes_a_greedy_baseline_for_qrf():
    from saturo.profiling.real_profile_collector import _decode_progressive_result
    with pytest.raises(ValueError, match="requires progressive QRF"):
        _decode_progressive_result({"results": {"RPS": {"greedy": {}}}})


def test_measured_iteration_profile_has_shared_simulator_schema(monkeypatch, tmp_path):
    from saturo.profiling import real_profile_collector as collector
    from saturo.predictors.time_predictor import TimePredictor
    identity = SimpleNamespace(to_dict=lambda: {}, digest="smoke")
    monkeypatch.setattr(collector, "TimePredictorProfileIdentity", SimpleNamespace(
        from_engine_configs=lambda **kwargs: identity))
    monkeypatch.setattr(collector, "time_predictor_cache_path", lambda *args: tmp_path/"time.json")
    core = SimpleNamespace(time_predictor=None,
        iteration_logs=[dict(prefill_tokens=n, prefill_sum_ctx=n*n,
            n_decode_reqs=0,decode_sum_ctx=0,actual_time_ms=2+n*.01)
            for n in (16,64,128)],
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=24)),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        cache_config=SimpleNamespace(num_gpu_blocks=100,block_size=16))
    path = collector._ensure_iteration_time_predictor(engine=core, cache_root=tmp_path)
    payload = json.loads(path.read_text())
    assert payload["time_predictor"]["model_type"] == "iteration_level"
    assert payload["kv_cache_block"] == {"total_gpu_blocks": 100, "block_size": 16}
    assert TimePredictor(str(path)).available


def test_be_window_throughput_excludes_requests_finished_during_drain():
    from saturo.runtime.scheduler import SaturoGlobalScheduler
    from saturo.runtime.workload import SaturoRequest
    scheduler=SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler._elapsed=lambda: 10
    scheduler.args=SimpleNamespace(window_s=2,model="smoke",load_format="auto",
        num_req=0,target_rps=1,rt_ttft_slo_ms=2000,rt_tbt_slo_ms=100)
    scheduler.all_rt=[]
    scheduler.all_be=[SaturoRequest("be_1","be",16,2,0,finished_s=1,observed_tokens=2),
                      SaturoRequest("be_2","be",16,2,0,finished_s=8,observed_tokens=2)]
    scheduler.request_errors=0;scheduler.history_updates=0
    scheduler.stop_reason="all_requests_finished";scheduler.next_rt_idx=0
    scheduler.rt_tokens=0;scheduler.rt_tbt_samples_ms=[];scheduler.rt_ttft_samples_ms=[]
    scheduler.rt_tbt_violations=0;scheduler.rt_ttft_violations=0
    scheduler.be_tokens=4;scheduler.be_tokens_in_window=2
    scheduler.prefill_handoffs=0;scheduler.trace_recorder=None
    result=scheduler._summary()
    assert result["be_completed"]==2
    assert result["be_completed_in_window"]==1
    assert result["be_throughput_rps"]==.5
    assert result["be_total_throughput_rps"]==.2
