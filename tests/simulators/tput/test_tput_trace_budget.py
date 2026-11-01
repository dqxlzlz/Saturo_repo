import json

from saturo.simulators.tput_simulator.config import (
    BEConfig,
    ModelConfig,
    SimulatorConfig,
)
from saturo.simulators.tput_simulator.simulator import (
    ThroughputPoolSimulator,
    TraceRequest,
)


class _RecordingPredictor:

    def __init__(self):
        self.batches = []

    def predict_forward_ms(self, prefill_reqs, decode_reqs):
        self.batches.append((list(prefill_reqs), list(decode_reqs)))
        total_tokens = (
            sum(query_len for query_len, _ in prefill_reqs)
            + sum(query_len for query_len, _ in decode_reqs)
        )
        return float(total_tokens)


def test_trace_decode_and_prefill_share_total_token_budget(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    model_path.write_text(json.dumps({
        "model_type": "iteration_level",
        "coef": [1.0, 0.0, 1.0, 0.0],
        "intercept": 0.0,
    }))
    config = SimulatorConfig(
        duration_s=100.0,
        warmup_s=0.0,
        latency_model_path=str(model_path),
        min_iteration_ms=0.0,
        model=ModelConfig(block_size=16),
        be=BEConfig(
            avg_input_len=4,
            avg_output_len=2,
            avg_decode_context_len=4,
            be_decode_blocks=16,
            prefill_chunk_candidates=[4],
            prefill_context_policy="chunk",
        ),
    )
    simulator = ThroughputPoolSimulator(config)
    predictor = _RecordingPredictor()
    simulator.predictor = predictor

    result = simulator.run_trace([
        TraceRequest(prompt_len=4, output_len=2, request_id="a"),
        TraceRequest(prompt_len=4, output_len=2, request_id="b"),
    ])

    assert result.best.completed_all
    assert result.best.total_token_budget == 4
    assert predictor.batches

    observed_mixed_batch = False
    for prefill_reqs, decode_reqs in predictor.batches:
        prefill_tokens = sum(query_len for query_len, _ in prefill_reqs)
        decode_tokens = sum(query_len for query_len, _ in decode_reqs)
        assert prefill_tokens + decode_tokens <= 4
        if prefill_tokens == 3 and decode_tokens == 1:
            observed_mixed_batch = True

    assert observed_mixed_batch


def test_trace_incremental_kv_accounting_preserves_swap_behavior(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    model_path.write_text(json.dumps({
        "model_type": "iteration_level",
        "coef": [1.0, 0.0, 1.0, 0.0],
        "intercept": 0.0,
    }))
    config = SimulatorConfig(
        duration_s=100.0,
        warmup_s=0.0,
        latency_model_path=str(model_path),
        min_iteration_ms=0.0,
        model=ModelConfig(block_size=16),
        be=BEConfig(
            avg_input_len=16,
            avg_output_len=1,
            avg_decode_context_len=16,
            be_decode_blocks=3,
            prefill_chunk_candidates=[32],
            prefill_context_policy="chunk",
        ),
    )

    result = ThroughputPoolSimulator(config).run_trace([
        TraceRequest(prompt_len=16, output_len=1, request_id="a"),
        TraceRequest(prompt_len=16, output_len=1, request_id="b"),
    ])

    assert result.best.completed_all
    assert result.best.total_prefill_tokens == 32
    assert result.best.total_decode_tokens == 2
    assert result.best.swap_out_count == 1
    assert result.best.swap_in_count == 1
    assert result.best.peak_kv_blocks <= 3


def test_lifecycle_trace_honors_inflight_limit(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    model_path.write_text(json.dumps({
        "model_type": "iteration_level",
        "coef": [1.0, 0.0, 1.0, 0.0],
        "intercept": 0.0,
    }))
    config = SimulatorConfig(
        duration_s=100.0,
        warmup_s=0.0,
        latency_model_path=str(model_path),
        min_iteration_ms=0.0,
        model=ModelConfig(block_size=16),
        be=BEConfig(
            avg_input_len=16,
            avg_output_len=1,
            be_decode_blocks=64,
            be_inflight=2,
            max_num_seqs=8,
            prefill_chunk_candidates=[64],
        ),
    )

    result = ThroughputPoolSimulator(config).run_trace([
        TraceRequest(prompt_len=16, output_len=1, request_id=str(index))
        for index in range(6)
    ])

    assert result.best.completed_all
    assert result.best.peak_active_sequences == 2


def test_lifecycle_trace_excludes_warmup_completions(tmp_path):
    model_path = tmp_path / "time_predictor.json"
    model_path.write_text(json.dumps({
        "model_type": "iteration_level",
        "coef": [1.0, 0.0, 1.0, 0.0],
        "intercept": 0.0,
    }))
    config = SimulatorConfig(
        duration_s=0.1,
        warmup_s=0.05,
        latency_model_path=str(model_path),
        min_iteration_ms=0.0,
        model=ModelConfig(block_size=16),
        be=BEConfig(
            avg_input_len=10,
            avg_output_len=1,
            be_decode_blocks=64,
            be_inflight=1,
            max_num_seqs=1,
            prefill_chunk_candidates=[10],
        ),
    )

    result = ThroughputPoolSimulator(config).run_trace([
        TraceRequest(prompt_len=10, output_len=1, request_id=str(index))
        for index in range(10)
    ])

    assert result.best.completed_requests < 10
    assert result.best.total_prefill_tokens < 100


def test_split_trace_keeps_prefill_and_decode_as_independent_populations(
    tmp_path,
):
    model_path = tmp_path / "time_predictor.json"
    model_path.write_text(json.dumps({
        "model_type": "iteration_level",
        "coef": [1.0, 0.0, 1.0, 0.0],
        "intercept": 0.0,
    }))
    config = SimulatorConfig(
        duration_s=0.2,
        warmup_s=0.0,
        latency_model_path=str(model_path),
        min_iteration_ms=1.0,
        model=ModelConfig(block_size=4),
        be=BEConfig(
            avg_input_len=4,
            avg_output_len=2,
            total_gpu_blocks=16,
            be_decode_blocks=12,
            be_inflight=4,
            max_num_seqs=4,
            enable_be_decode=True,
            chunk_size_candidates=[8],
        ),
    )
    simulator = ThroughputPoolSimulator(config)
    predictor = _RecordingPredictor()
    simulator.predictor = predictor

    result = simulator.run_split_trace([
        TraceRequest(prompt_len=4, output_len=2, request_id="a"),
        TraceRequest(prompt_len=4, output_len=2, request_id="b"),
    ])
    best = result.best

    assert result.mode == "split_trace"
    assert best.prefill_inflight == 2
    assert best.decode_inflight == 2
    assert best.completed_prefill_requests > 0
    assert best.completed_decode_requests > 0
    assert best.prefill_req_per_s > 0
    assert best.decode_req_per_s > 0
    assert best.be_saturation_req_per_s == min(
        best.prefill_req_per_s, best.decode_req_per_s)
    assert best.dummy_handoff_iterations > 0
    assert best.peak_decode_kv_blocks <= best.peak_kv_blocks <= 16
    assert any(prefill and decode for prefill, decode in predictor.batches)
