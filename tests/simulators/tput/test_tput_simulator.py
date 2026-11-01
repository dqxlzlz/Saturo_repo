import json

from saturo.simulators.tput_simulator.config import BEConfig, ModelConfig, SimulatorConfig
from saturo.simulators.tput_simulator.run_tput_simulator import run_tput_profile
from saturo.simulators.tput_simulator.simulator import (
    ThroughputPoolSimulator,
    TraceRequest,
)


def _write_iteration_model(path):
    path.write_text(json.dumps({
        "model_type": "iteration_level",
        "coef": [1.0, 0.0, 1.0, 0.0],
        "intercept": 0.0,
    }))


def test_tput_simulator_uses_bottleneck_request_throughput(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    _write_iteration_model(model_path)
    config = SimulatorConfig(
        duration_s=10.0,
        warmup_s=0.0,
        latency_model_path=str(model_path),
        min_iteration_ms=0.0,
        model=ModelConfig(
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            tensor_parallel=1,
        ),
        be=BEConfig(
            avg_input_len=100,
            avg_output_len=10,
            avg_decode_context_len=100,
            be_decode_blocks=70,
            decode_batch_candidates=[10],
            prefill_chunk_candidates=[100],
            chunk_size_candidates=[110],
            prefill_context_policy="chunk",
        ),
    )

    result = ThroughputPoolSimulator(config).run()
    best = result.best

    assert best.iteration_latency_ms == 110.0
    assert best.prefill_req_per_s == best.decode_req_per_s
    assert round(best.be_saturation_req_per_s, 6) == round(1.0 / 0.11, 6)
    assert best.raw_tokens_per_s > best.be_saturation_req_per_s


def test_tput_simulator_selects_best_balanced_candidate(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    _write_iteration_model(model_path)
    config = SimulatorConfig(
        duration_s=10.0,
        warmup_s=0.0,
        latency_model_path=str(model_path),
        min_iteration_ms=0.0,
        model=ModelConfig(
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            tensor_parallel=1,
        ),
        be=BEConfig(
            avg_input_len=100,
            avg_output_len=10,
            avg_decode_context_len=100,
            be_decode_blocks=70,
            decode_batch_candidates=[10],
            prefill_chunk_candidates=[50, 100, 200],
            chunk_size_candidates=[60, 110, 210],
            prefill_context_policy="chunk",
        ),
    )

    result = ThroughputPoolSimulator(config).run()

    assert result.best.prefill_chunk_size == 100


def test_tput_profile_exports_allocator_entry(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    _write_iteration_model(model_path)

    class Args:
        latency_model = str(model_path)
        duration = 10.0
        warmup = 0.0
        latency_scale = 1.0
        latency_offset_ms = 0.0
        min_iteration_ms = 0.0
        avg_input_len = 100
        avg_output_len = 10
        avg_decode_context_len = 100
        be_decode_blocks = 70
        decode_batch_candidates = [10]
        prefill_chunk_candidates = [100]
        chunk_size_candidates = [110]
        prefill_context_policy = "chunk"
        num_layers = 1
        num_kv_heads = 1
        head_dim = 1
        dtype_bytes = 2
        tensor_parallel = 1
        block_size = 16
        config_name = "tp1_pp1"
        trace_csv = None
        trace_limit = None
        trace_output_len = None
        print_all = False

    payload = run_tput_profile(Args())

    assert payload["pool"] == "T"
    assert payload["allocation_profile_entry"]["saturation"][0]["rt_rps"] == 0.0
    assert (
        payload["allocation_profile_entry"]["saturation"][0]["be_throughput"]
        == payload["best"]["be_saturation_req_per_s"]
    )


def test_trace_driven_simulator_replays_chunk_prefill_and_decode(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    _write_iteration_model(model_path)
    config = SimulatorConfig(
        duration_s=100.0,
        warmup_s=0.0,
        latency_model_path=str(model_path),
        min_iteration_ms=0.0,
        model=ModelConfig(
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            tensor_parallel=1,
        ),
        be=BEConfig(
            avg_input_len=100,
            avg_output_len=2,
            avg_decode_context_len=100,
            be_decode_blocks=14,
            decode_batch_candidates=[2],
            prefill_chunk_candidates=[4],
            prefill_context_policy="chunk",
        ),
    )
    trace = [
        TraceRequest(prompt_len=6, output_len=2, request_id="a"),
        TraceRequest(prompt_len=2, output_len=2, request_id="b"),
    ]

    result = ThroughputPoolSimulator(config).run_trace(trace)
    best = result.best

    assert result.mode == "trace"
    assert best.completed_requests == 2
    assert best.completed_all
    assert best.total_token_budget == 4
    assert best.total_prefill_tokens == 8
    assert best.total_decode_tokens == 4
    assert best.completed_iterations > 2
    assert best.be_throughput_req_per_s == (
        best.completed_requests / best.measured_time_s)


def test_trace_mode_derives_decode_batch_from_resident_blocks(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    _write_iteration_model(model_path)
    config = SimulatorConfig(
        duration_s=100.0,
        warmup_s=0.0,
        latency_model_path=str(model_path),
        min_iteration_ms=0.0,
        model=ModelConfig(
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            tensor_parallel=1,
        ),
        be=BEConfig(
            avg_input_len=16,
            avg_output_len=1,
            avg_decode_context_len=16,
            be_decode_blocks=4,
            decode_batch_candidates=[1],
            prefill_chunk_candidates=[32],
            prefill_context_policy="chunk",
        ),
    )
    trace = [
        TraceRequest(prompt_len=16, output_len=1, request_id="a"),
        TraceRequest(prompt_len=16, output_len=1, request_id="b"),
    ]

    result = ThroughputPoolSimulator(config).run_trace(trace)
    best = result.best

    assert best.completed_all
    assert best.peak_decode_batch_size == 2
    assert best.to_dict()["decode_batch_size"] == 2


def test_trace_mode_swaps_short_decode_when_blocks_are_exhausted(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    _write_iteration_model(model_path)
    config = SimulatorConfig(
        duration_s=100.0,
        warmup_s=0.0,
        latency_model_path=str(model_path),
        min_iteration_ms=0.0,
        model=ModelConfig(
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            tensor_parallel=1,
        ),
        be=BEConfig(
            avg_input_len=16,
            avg_output_len=1,
            avg_decode_context_len=16,
            be_decode_blocks=3,
            decode_batch_candidates=[1],
            prefill_chunk_candidates=[32],
            prefill_context_policy="chunk",
        ),
    )
    trace = [
        TraceRequest(prompt_len=16, output_len=1, request_id="a"),
        TraceRequest(prompt_len=16, output_len=1, request_id="b"),
    ]

    result = ThroughputPoolSimulator(config).run_trace(trace)
    best = result.best

    assert best.completed_all
    assert best.swap_out_count == 1
    assert best.swap_in_count == 1
    assert best.peak_kv_blocks <= 3


def test_tput_profile_loads_trace_csv(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    trace_path = tmp_path / "trace.csv"
    _write_iteration_model(model_path)
    trace_path.write_text(
        "prompt_len,output_len\n6,2\n3,1\n"
    )

    class Args:
        latency_model = str(model_path)
        duration = 100.0
        warmup = 0.0
        latency_scale = 1.0
        latency_offset_ms = 0.0
        min_iteration_ms = 0.0
        avg_input_len = 100
        avg_output_len = 2
        avg_decode_context_len = 100
        be_decode_blocks = 14
        decode_batch_candidates = [2]
        prefill_chunk_candidates = [4]
        chunk_size_candidates = [128]
        prefill_context_policy = "chunk"
        trace_csv = str(trace_path)
        trace_limit = None
        trace_output_len = None
        num_layers = 1
        num_kv_heads = 1
        head_dim = 1
        dtype_bytes = 2
        tensor_parallel = 1
        block_size = 16
        config_name = "tp1_pp1_trace"
        print_all = False

    payload = run_tput_profile(Args())

    assert payload["mode"] == "trace"
    assert payload["config"]["trace_requests"] == 2
    assert payload["best"]["completed_requests"] == 2
    assert (
        payload["allocation_profile_entry"]["saturation"][0]["be_throughput"]
        == payload["best"]["be_saturation_req_per_s"]
    )
