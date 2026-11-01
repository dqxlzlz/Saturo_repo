import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from saturo.experiments.pool_validation.benchmark_d_pool_saturation import (
    REQUEST_MODE_LOCAL as D_REQUEST_MODE_LOCAL,
    be_request_command,
    run_decode_simulation,
)
from saturo.experiments.pool_validation.benchmark_t_pool_saturation import (
    REQUEST_MODE_VLLM,
    RunResult,
    TraceRequest,
    be_lifecycle_command,
    be_decode_kv_plan,
    build_sim_plan,
    filter_be,
    result_to_dict,
    run_sweep_subprocesses,
)


def test_d_pool_simulation_plan_is_written_before_worker_inputs(
    monkeypatch,
    tmp_path,
):
    from saturo.simulators.decode_simulator import run_decode_simulator

    simulator_payload = {
        "config": {"num_req": 60, "window_s": 30.0},
        "results": {
            "RPS=2.0": {
                "progressive": {
                    "be_saturation_req_per_s": 4.5,
                    "be_prefill_req_per_s": 4.7,
                    "be_decode_req_per_s": 4.5,
                    "rt_tbt_violation_rate": 0.01,
                    "rt_slo_violations_total": 2,
                },
            },
        },
    }
    monkeypatch.setattr(
        run_decode_simulator,
        "run_decode_profile",
        lambda args, print_summaries: simulator_payload,
    )
    args = argparse.Namespace(
        runtime_profile="profile.json",
        rt_workload_csv="rt.csv",
        be_workload_csv="be.csv",
        num_req=60,
        sim_warmup_s=5.0,
        target_rps=2.0,
        rt_tbt_slo_ms=100.0,
        max_model_len=None,
        gpu_memory_utilization=0.9,
        be_decode_block_fraction=0.95,
        sim_total_gpu_blocks=None,
        sim_chunk_sizes="512,1024",
        request_mode="split-phase",
    )

    plan = run_decode_simulation(args, tmp_path)

    assert plan["predicted_be_saturation_req_per_s"] == 4.5
    assert plan["rt_slo_violations_total"] == 2
    assert json.loads((tmp_path / "simulation.json").read_text()) == (
        simulator_payload)


def test_d_pool_local_command_is_one_end_to_end_request():
    command = be_request_command(
        "be_local_7",
        request_mode=D_REQUEST_MODE_LOCAL,
        phase="local",
        prompt_len=1234,
        output_len=56,
    )

    assert command["phase"] == "prefill"
    assert command["prompt_len"] == 1234
    assert command["max_tokens"] == 56
    assert command["request_id"] == "be_local_7"


def test_t_pool_simulator_selects_chunk_and_writes_candidates(
    monkeypatch,
    tmp_path,
):
    from saturo.simulators.tput_simulator import simulator as simulator_module

    captured = {}

    class Candidate:
        def __init__(self, chunk_size, throughput):
            self.chunk_size = chunk_size
            self.throughput = throughput

        def to_dict(self):
            return {
                "chunk_size": self.chunk_size,
                "be_saturation_req_per_s": self.throughput,
                "raw_tokens_per_s": self.throughput * 100,
            }

    class FakeSimulator:
        def __init__(self, config):
            captured["config"] = config

        def run_split_trace(self, trace):
            captured["trace"] = trace
            candidates = (Candidate(512, 3.0), Candidate(1024, 4.0))
            return SimpleNamespace(
                mode="split_trace",
                best=candidates[1],
                candidates=candidates,
            )

    monkeypatch.setattr(
        simulator_module, "ThroughputPoolSimulator", FakeSimulator)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "kv_cache_block": {
            "total_gpu_blocks": 4096,
            "block_size": 16,
        },
    }))
    plan_path = tmp_path / "t_plan.json"
    args = argparse.Namespace(
        runtime_profile=str(profile_path),
        be_workload_csv="be.csv",
        sim_duration_s=60.0,
        sim_warmup_s=0.0,
    )
    trace = [TraceRequest("0", 100, 20), TraceRequest("1", 300, 40)]

    payload = build_sim_plan(args, [512, 1024], trace, plan_path)

    assert payload["best"]["chunk_size"] == 1024
    assert payload["best"]["be_saturation_req_per_s"] == 4.0
    assert captured["config"].be.chunk_size_candidates == [512, 1024]
    assert captured["config"].be.be_inflight == 128
    assert captured["config"].be.total_gpu_blocks == 4096
    assert captured["config"].be.avg_input_len == 200
    assert captured["config"].be.be_decode_blocks == 3872
    assert payload["config"]["be_decode_kv"] == {
        "total_gpu_blocks": 4096,
        "kv_utilization": 0.95,
        "usable_gpu_blocks": 3891,
        "max_be_prompt": 300,
        "reserved_be_prefill_blocks": 19,
        "be_decode_block_limit": 3872,
    }
    assert json.loads(plan_path.read_text())["best"]["chunk_size"] == 1024


def test_t_pool_decode_kv_plan_reserves_utilization_and_max_prefill():
    trace = [TraceRequest("0", 100, 20), TraceRequest("1", 30377, 40)]

    plan = be_decode_kv_plan(5764, 16, trace, 0.95)

    assert plan["usable_gpu_blocks"] == 5475
    assert plan["reserved_be_prefill_blocks"] == 1899
    assert plan["be_decode_block_limit"] == 3576


def test_t_pool_p80_filter_keeps_cutoff_ties():
    args = argparse.Namespace(
        be_input_max_percentile=0.8,
        be_input_drop_frac=0.0,
        be_output_keep_frac=None,
        be_output_max_len=None,
    )
    trace = [
        TraceRequest("0", 10, 1),
        TraceRequest("1", 20, 1),
        TraceRequest("2", 30, 1),
        TraceRequest("3", 40, 1),
        TraceRequest("4", 40, 1),
        TraceRequest("5", 50, 1),
    ]

    filtered = filter_be(trace, args)

    assert [request.prompt_len for request in filtered] == [10, 20, 30, 40, 40]
    assert args.be_input_max_tokens_applied == 40


def test_t_pool_vllm_command_is_one_end_to_end_request():
    cmd = be_lifecycle_command(TraceRequest("row-7", 1234, 56), serial=9)

    assert cmd["phase"] == "prefill"
    assert cmd["prompt_len"] == 1234
    assert cmd["max_tokens"] == 56
    assert cmd["attempt_id"].startswith("be_vllm_9:")


def test_t_pool_vllm_throughput_counts_only_end_to_end_completions():
    result = RunResult(
        chunk_size=1024,
        be_inflight=64,
        request_mode=REQUEST_MODE_VLLM,
        be_completed=17,
        be_e2e_completed=17,
        # These deliberately imply a different phase-equivalent rate.
        be_prefill_tokens=9000,
        be_decode_tokens=100,
        avg_be_input_len=1000,
        avg_be_output_len=100,
        window_s=10.0,
    )

    payload = result_to_dict(result)

    assert payload["be_throughput_req_per_s"] == 1.7
    assert payload["be_saturation_req_per_s"] == 1.7
    assert payload["be_e2e_completed"] == 17


def test_t_pool_vllm_sim_plan_uses_lifecycle_trace(monkeypatch, tmp_path):
    from saturo.simulators.tput_simulator import simulator as simulator_module

    captured = {}

    class Candidate:
        def __init__(self, chunk_size, throughput):
            self.chunk_size = chunk_size
            self.throughput = throughput

        def to_dict(self):
            return {
                "chunk_size": self.chunk_size,
                "be_saturation_req_per_s": self.throughput,
                "raw_tokens_per_s": self.throughput * 100,
            }

    class FakeSimulator:
        def __init__(self, config):
            captured["config"] = config

        def run_trace(self, trace):
            captured["trace"] = trace
            candidates = (Candidate(512, 2.0), Candidate(1024, 3.0))
            return SimpleNamespace(
                mode="trace",
                best=candidates[1],
                candidates=candidates,
            )

        def run_split_trace(self, trace):
            raise AssertionError("vllm mode must not use split trace")

    monkeypatch.setattr(
        simulator_module, "ThroughputPoolSimulator", FakeSimulator)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "kv_cache_block": {
            "total_gpu_blocks": 4096,
            "block_size": 16,
        },
    }))
    args = argparse.Namespace(
        request_mode=REQUEST_MODE_VLLM,
        runtime_profile=str(profile_path),
        be_workload_csv="be.csv",
        sim_duration_s=60.0,
        sim_warmup_s=0.0,
        be_inflight="64,128",
    )
    trace = [TraceRequest("0", 100, 20), TraceRequest("1", 300, 40)]

    payload = build_sim_plan(
        args, [512, 1024], trace, tmp_path / "plan.json")

    assert payload["mode"] == "trace"
    assert payload["config"]["request_mode"] == REQUEST_MODE_VLLM
    assert captured["config"].be.be_decode_blocks == 4096
    assert captured["config"].be.prefill_chunk_candidates == [512, 1024]


def test_t_pool_real_sweep_uses_one_process_per_chunk(
    monkeypatch,
    tmp_path,
):
    calls = []

    def fake_run(command, env, check):
        chunk = int(command[command.index("--chunk-sizes") + 1])
        output = Path(command[-1])
        output.write_text(json.dumps({
            "config": {
                "chunk_size_candidates": [chunk],
                "selected_chunk_size": chunk,
                "num_gpu_blocks": 10000 - chunk,
            },
            "simulator": None,
            "runs": [{"chunk_size": chunk}],
        }))
        calls.append((chunk, env["SATURO_T_POOL_SWEEP_CHILD"], check))

    monkeypatch.setattr(
        "saturo.experiments.pool_validation."
        "benchmark_t_pool_saturation.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "saturo.experiments.pool_validation."
        "benchmark_t_pool_saturation.sys.argv",
        ["benchmark.py", "--model", "fake"],
    )
    output = tmp_path / "combined.json"

    run_sweep_subprocesses(output, [1024, 2048])

    combined = json.loads(output.read_text())
    assert calls == [(1024, "1", True), (2048, "1", True)]
    assert [run["chunk_size"] for run in combined["runs"]] == [1024, 2048]
    assert combined["config"]["isolated_chunk_processes"] is True
    assert combined["config"]["num_gpu_blocks"] is None
    assert combined["config"]["num_gpu_blocks_by_chunk"] == {
        "1024": 8976,
        "2048": 7952,
    }


def _run_p_window(monkeypatch, module, arrivals, emissions, extra_events=(),
                  warmup_s=0.0, max_wall_s=2.0):
    clock = SimpleNamespace(now=100.0)
    trace = [module.TraceRequest(str(i), 16, 1, arrival)
             for i, arrival in enumerate(arrivals)]
    events = [(100.0 + ready, dict(event, time_s=100.0 + event['time_s']))
              for ready, event in extra_events]
    submitted = []

    async def sleep(delay):
        clock.now += max(delay, .001)

    class Worker:
        async def submit_batch(self, commands):
            submitted.extend(commands)
            for command in commands:
                if command['kind'] != 'rt':
                    continue
                index = int(command['attempt_id'].split(':')[1])
                for ready, kind in emissions.get(index, ()):
                    events.append((100.0 + ready, dict(
                        event=kind, kind='rt', phase='prefill',
                        attempt_id=command['attempt_id'], time_s=100.0 + ready)))

        async def drain_events(self):
            ready = [event for at, event in events if at <= clock.now + 1e-9]
            events[:] = [(at, event) for at, event in events
                         if at > clock.now + 1e-9]
            return ready

    args = argparse.Namespace(num_req=len(trace), warmup_s=warmup_s,
        max_wall_s=max_wall_s, ttft_slo_ms=20, be_inflight=1,
        enable_be_decode=True, be_request_mode='normal')
    monkeypatch.setattr(module, 'time', SimpleNamespace(perf_counter=lambda: clock.now))
    monkeypatch.setattr(module, 'asyncio', SimpleNamespace(sleep=sleep))
    result = asyncio.run(module.run_one_rps(
        Worker(), trace, [module.TraceRequest('be', 16, 4)], len(trace), args))
    return result, clock.now - 100.0, submitted


def test_p_window_counts_starved_requests(monkeypatch):
    from saturo.experiments.pool_validation import benchmark_p_pool_saturation as p
    arrivals = [.1] * 70 + [.2] * 10 + [.3] * 20
    emissions = {i: [(.105 if i < 70 else .23, 'token')]
                 for i in range(80)}
    result, elapsed, _ = _run_p_window(monkeypatch, p, arrivals, emissions)
    assert result.rt_injected == 100 and result.rt_first_token == 80
    assert result.rt_violations == 30
    assert elapsed < 1.01
    payload = p.result_to_dict(result, argparse.Namespace())
    assert payload['rt_ttft_violation_rate'] == .3
    assert payload['rt_without_first_token'] == 20
    assert payload['rt_ttft_samples_complete'] is False


@pytest.mark.parametrize('emissions,misses,first', [
    ({0: [(1.005, 'token')]}, 0, 1),
    ({0: [(1.015, 'token')]}, 1, 0),
    ({}, 1, 0),
    ({0: [(.991, 'finished'), (.992, 'finished')]}, 1, 0),
])
def test_p_window_settles_late_arrivals(monkeypatch, emissions, misses, first):
    from saturo.experiments.pool_validation import benchmark_p_pool_saturation as p
    result, elapsed, _ = _run_p_window(monkeypatch, p, [.99], emissions)
    assert result.rt_violations == misses and result.rt_first_token == first
    assert 1 <= elapsed <= 1.021
    assert result.window_s == 1


def test_p_window_ignores_warmup_and_duplicate_events(monkeypatch):
    from saturo.experiments.pool_validation import benchmark_p_pool_saturation as p
    result, _, _ = _run_p_window(monkeypatch, p, [.1, .6], {
        0: [(.15, 'token')],
        1: [(.63, 'token'), (.64, 'token'), (.65, 'finished'), (.66, 'finished')],
    }, warmup_s=.5)
    assert result.rt_injected == result.rt_first_token == result.rt_violations == 1
    assert result.window_s == .5


def test_p_window_excludes_tail_be_and_uses_event_time(monkeypatch):
    from saturo.experiments.pool_validation import benchmark_p_pool_saturation as p
    extra = [
        (1, dict(event='finished', kind='be', phase='prefill',
                 token_count=4, time_s=.995)),
        (1, dict(event='iteration_log', be_prefill_tokens=16,
                 be_decode_tokens=4, time_s=.995)),
        (1.005, dict(event='finished', kind='be', phase='prefill',
                     token_count=9, time_s=1.005)),
        (1.005, dict(event='iteration_log', be_prefill_tokens=99,
                     be_decode_tokens=99, time_s=1.005)),
    ]
    result, _, submitted = _run_p_window(monkeypatch, p, [.99], {}, extra)
    assert result.be_completed == 1 and result.be_output_tokens == 4
    assert result.be_prefill_tokens == 16 and result.be_decode_tokens == 4
    assert sum(command['kind'] == 'be' for command in submitted) == 3
    assert result.rt_violations == 1


def test_p_window_keeps_max_wall_limit(monkeypatch):
    from saturo.experiments.pool_validation import benchmark_p_pool_saturation as p
    with pytest.raises(TimeoutError, match='max-wall-s'):
        _run_p_window(monkeypatch, p, [.99], {}, max_wall_s=1.005)


def test_p_window_defaults_unchanged(monkeypatch):
    from saturo.experiments.pool_validation import benchmark_p_pool_saturation as p
    monkeypatch.setattr(p.sys, 'argv', ['p_pool', '--model', 'fake',
        '--rt-workload-csv', 'rt.csv', '--be-workload-csv', 'be.csv',
        '--num-req', '100', '--target-rps', '100'])
    args = p.parse_args()
    assert args.warmup_s == 10 and args.max_wall_s == 1200
    assert args.drain_timeout_s == 30 and args.ttft_slo_ms == 1000
    assert args.be_inflight == 2048 and args.be_request_mode == 'normal'
    assert args.chunk_ms_source == 'sim' and args.p_pool_be_chunk_ms == 113
