from saturo.simulators.decode_simulator.config import SimulatorConfig
from saturo.simulators.decode_simulator.predictors.memory import (
    DPoolSaturationPredictor,
    sampling_offsets,
)
from saturo.predictors.time_predictor import TimePredictor
from saturo.simulators.decode_simulator.request import Request, RequestState, RequestType
from saturo.simulators.decode_simulator.simulator import (
    DecodeSimulator,
    DecodeStrategy,
    IterationStats,
)
from saturo.predictors import (
    build_feature_matrix,
    create_length_predictor,
)


def test_decode_public_imports():
    assert DecodeSimulator.__name__ == "DecodeSimulator"
    assert DecodeStrategy.PROGRESSIVE.value == "progressive"


def test_sampling_offsets_follow_tau_steps_and_include_window_end():
    assert sampling_offsets(window_s=0.35, tau_s=0.1, max_samples=99) == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.35,
    ]


def test_sampling_offsets_zero_window():
    assert sampling_offsets(window_s=0.0, tau_s=0.1, max_samples=99) == [0.0]


def test_throughput_uses_elapsed_time_when_run_is_shorter_than_warmup():
    config = SimulatorConfig(
        simulation_duration_s=1.0,
        warmup_s=10.0,
        latency_model_path="/tmp/missing-decode-simulator-model.json",
    )
    simulator = DecodeSimulator(config)
    simulator.current_time_s = 1.0

    result = simulator._compile_results([
        IterationStats(
            iteration_id=0,
            time_s=0.0,
            be_prefill_chunk_size=100,
            num_be_decode=20,
        )
    ])

    assert result.be_throughput_tokens_per_s == 120.0
    assert result.be_prefill_throughput_tokens_per_s == 100.0
    assert result.be_decode_throughput_tokens_per_s == 20.0


def test_be_saturation_throughput_uses_request_bottleneck():
    config = SimulatorConfig(
        simulation_duration_s=1.0,
        warmup_s=0.0,
        latency_model_path="/tmp/missing-decode-simulator-model.json",
    )
    simulator = DecodeSimulator(config)
    simulator.current_time_s = 1.0
    simulator._all_be_requests = [
        Request(
            request_id=1,
            request_type=RequestType.BE,
            prompt_len=100,
            target_output_len=10,
            arrival_time_s=0.0,
        )
    ]

    result = simulator._compile_results([
        IterationStats(
            iteration_id=0,
            time_s=0.0,
            be_prefill_chunk_size=100,
            num_be_decode=5,
        )
    ])

    assert result.be_prefill_req_per_s == 1.0
    assert result.be_decode_req_per_s == 0.5
    assert result.be_saturation_req_per_s == 0.5
    assert result.be_throughput_tokens_per_s == 105.0


def test_throughput_uses_post_warmup_samples_when_available():
    config = SimulatorConfig(
        simulation_duration_s=12.0,
        warmup_s=10.0,
        latency_model_path="/tmp/missing-decode-simulator-model.json",
    )
    simulator = DecodeSimulator(config)
    simulator.current_time_s = 12.0

    result = simulator._compile_results([
        IterationStats(
            iteration_id=0,
            time_s=9.0,
            be_prefill_chunk_size=100,
            num_be_decode=20,
        ),
        IterationStats(
            iteration_id=1,
            time_s=10.5,
            be_prefill_chunk_size=30,
            num_be_decode=10,
        ),
    ])

    assert result.be_throughput_tokens_per_s == 20.0
    assert result.be_prefill_throughput_tokens_per_s == 15.0
    assert result.be_decode_throughput_tokens_per_s == 5.0


def test_decode_simulator_runs_be_work_without_rt_or_initial_be():
    config = SimulatorConfig(
        simulation_duration_s=0.01,
        warmup_s=0.0,
        latency_model_path="/tmp/missing-decode-simulator-model.json",
    )
    simulator = DecodeSimulator(config)

    result = simulator.run([], [])

    assert result.total_iterations > 0
    assert result.total_be_prefill_tokens + result.total_be_decode_tokens > 0
    assert result.avg_target_tbt_ms == config.slo.tbt_slo_ms


def test_rt_admission_uses_block_allocator_capacity():
    config = SimulatorConfig(latency_model_path="/tmp/missing-decode-simulator-model.json")
    simulator = DecodeSimulator(config)
    simulator._reset_runtime_state([], [])

    huge_rt = Request(
        request_id=1,
        request_type=RequestType.RT,
        prompt_len=10**9,
        target_output_len=1,
        arrival_time_s=0.0,
    )
    simulator.rt_waiting.append(huge_rt)

    violations = simulator._admit_waiting_rt()

    assert violations == 1
    assert simulator.rt_active == []
    assert list(simulator.rt_waiting) == [huge_rt]
    assert simulator._rt_allocated_blocks == 0


def test_block_allocator_releases_finished_be_decode():
    config = SimulatorConfig(latency_model_path="/tmp/missing-decode-simulator-model.json")
    simulator = DecodeSimulator(config)
    simulator._reset_runtime_state([], [])

    be = Request(
        request_id=2,
        request_type=RequestType.BE,
        prompt_len=16,
        target_output_len=1,
        arrival_time_s=0.0,
        state=RequestState.DECODING,
        prefill_tokens_processed=16,
    )
    target_blocks = simulator._blocks_for_tokens(be.current_seq_len + 1)
    assert simulator._try_account_be_blocks(be, target_blocks)
    simulator.be_decoding.append(be)

    simulator._advance_be_decode([be], end_time_s=0.1)

    assert be in simulator.finished_be
    assert simulator._be_allocated_blocks == 0
    assert be.request_id not in simulator._request_blocks


def test_rt_admission_does_not_move_be_decode_blocks():
    config = SimulatorConfig(latency_model_path="/tmp/missing-decode-simulator-model.json")
    simulator = DecodeSimulator(config)
    simulator._reset_runtime_state([], [])
    simulator.total_gpu_blocks = 2

    be = Request(
        request_id=2,
        request_type=RequestType.BE,
        prompt_len=16,
        target_output_len=8,
        arrival_time_s=0.0,
        state=RequestState.DECODING,
        prefill_tokens_processed=16,
    )
    assert simulator._try_account_be_blocks(be, 2)
    simulator.be_decoding.append(be)

    rt = Request(
        request_id=1,
        request_type=RequestType.RT,
        prompt_len=16,
        target_output_len=1,
        arrival_time_s=0.0,
    )
    simulator.rt_waiting.append(rt)

    violations = simulator._admit_waiting_rt()

    assert violations == 1
    assert simulator.rt_active == []
    assert simulator.be_decoding == [be]
    assert simulator._rt_allocated_blocks == 0
    assert simulator._be_allocated_blocks == 2


def test_rt_can_be_admitted_when_physical_space_available():
    """RT should be admitted as long as total_gpu_blocks is not exceeded."""
    config = SimulatorConfig(latency_model_path="/tmp/missing-decode-simulator-model.json")
    simulator = DecodeSimulator(config)
    simulator._reset_runtime_state([], [])
    simulator.total_gpu_blocks = 4

    be = Request(
        request_id=2,
        request_type=RequestType.BE,
        prompt_len=32,
        target_output_len=8,
        arrival_time_s=0.0,
        state=RequestState.DECODING,
        prefill_tokens_processed=32,
    )
    assert simulator._try_account_be_blocks(be, 2)
    simulator.be_decoding.append(be)

    rt = Request(
        request_id=1,
        request_type=RequestType.RT,
        prompt_len=32,
        target_output_len=1,
        arrival_time_s=0.0,
    )
    simulator.rt_waiting.append(rt)

    violations = simulator._admit_waiting_rt()

    assert violations == 0
    assert len(simulator.rt_active) == 1
    assert simulator.rt_active[0].request_id == 1
    assert simulator.be_decoding == [be]
    assert simulator._rt_allocated_blocks == 1
    assert simulator._be_allocated_blocks == 2


def test_detects_terminal_rt_memory_stall(tmp_path):
    be_workload = tmp_path / "be_workload.csv"
    be_workload.write_text(
        "request_id,prompt_len,output_len,arrival_offset_s\n"
        "be_0,16,8,0\n",
        encoding="utf-8",
    )
    config = SimulatorConfig(
        latency_model_path="/tmp/missing-decode-simulator-model.json",
        be_workload_csv=str(be_workload),
    )
    simulator = DecodeSimulator(config)
    simulator._reset_runtime_state([], [])
    simulator.total_gpu_blocks = 2
    simulator.usable_gpu_blocks = 2

    requests = []
    for request_id in (1, 2):
        req = Request(
            request_id=request_id,
            request_type=RequestType.RT,
            prompt_len=16,
            target_output_len=8,
            arrival_time_s=0.0,
            state=RequestState.DECODING,
            prefill_tokens_processed=16,
        )
        assert simulator._try_account_rt_blocks(req, 1)
        requests.append(req)
    simulator.rt_active = requests

    simulator._admit_waiting_rt()

    assert simulator._iteration_rt_memory_failures == 2
    assert simulator._iteration_rt_ready_ids == set()
    assert simulator._has_terminal_rt_memory_stall()


def test_rt_block_growth_swaps_be_to_front_and_restores_progress():
    config = SimulatorConfig(
        latency_model_path="/tmp/missing-decode-simulator-model.json",
        be_workload_csv="data/datasets/be_workload.csv",
    )
    simulator = DecodeSimulator(config)
    simulator._reset_runtime_state([], [])
    simulator.be_backlog.clear()
    simulator.usable_gpu_blocks = 4

    fresh_be = Request(
        request_id=3,
        request_type=RequestType.BE,
        prompt_len=64,
        target_output_len=8,
        arrival_time_s=0.0,
    )
    simulator.be_backlog.append(fresh_be)

    be = Request(
        request_id=2,
        request_type=RequestType.BE,
        prompt_len=32,
        target_output_len=8,
        arrival_time_s=0.0,
        state=RequestState.DECODING,
        tokens_generated=1,
        prefill_tokens_processed=32,
    )
    assert simulator._try_account_be_blocks(be, 3)
    simulator.be_decoding.append(be)

    rt = Request(
        request_id=1,
        request_type=RequestType.RT,
        prompt_len=16,
        target_output_len=8,
        arrival_time_s=0.0,
        state=RequestState.DECODING,
        prefill_tokens_processed=16,
    )
    assert simulator._try_account_rt_blocks(rt, 1)
    simulator.rt_active.append(rt)

    violations = simulator._admit_waiting_rt()

    assert violations == 0
    assert simulator._rt_allocated_blocks == 2
    assert simulator._be_allocated_blocks == 0
    assert simulator.be_backlog[0] is be
    assert simulator.be_backlog[1] is fresh_be
    assert be.tokens_generated == 1
    assert simulator.be_swap_out_events == 1

    simulator._release_request_blocks(rt)
    simulator.rt_active.remove(rt)
    simulator._restore_swapped_be_requests(be_budget_blocks=4)

    assert simulator.be_backlog[0] is fresh_be
    assert be in simulator.be_decoding
    assert be.state == RequestState.DECODING
    assert be.tokens_generated == 1
    assert simulator._be_allocated_blocks == 3
    assert simulator.be_swap_in_events == 1


def test_be_block_growth_is_prepared_by_swapping_shortest_selected_be():
    config = SimulatorConfig(
        latency_model_path="/tmp/missing-decode-simulator-model.json",
        be_workload_csv="data/datasets/be_workload.csv",
    )
    simulator = DecodeSimulator(config)
    simulator._reset_runtime_state([], [])
    simulator.be_backlog.clear()
    simulator.usable_gpu_blocks = 4

    short_be = Request(
        request_id=1,
        request_type=RequestType.BE,
        prompt_len=16,
        target_output_len=8,
        arrival_time_s=0.0,
        state=RequestState.DECODING,
        prefill_tokens_processed=16,
    )
    long_be = Request(
        request_id=2,
        request_type=RequestType.BE,
        prompt_len=32,
        target_output_len=8,
        arrival_time_s=0.0,
        state=RequestState.DECODING,
        prefill_tokens_processed=32,
    )
    assert simulator._try_account_be_blocks(short_be, 1)
    assert simulator._try_account_be_blocks(long_be, 2)
    simulator.be_decoding.extend([short_be, long_be])

    prepared = simulator._prepare_be_decode_blocks(
        [short_be, long_be], be_budget_blocks=4)

    assert prepared == [long_be]
    assert simulator.be_backlog[0] is short_be
    assert short_be not in simulator.be_decoding
    assert simulator._request_blocks[long_be.request_id] == 3
    assert simulator._be_allocated_blocks == 3
    assert simulator.be_swap_out_events == 1


def _write_toy_model_len_predictor(tmp_path, model_type: str) -> str:
    rows = []
    targets = []
    for prompt_len, output_len in [
        (128, 80),
        (128, 120),
        (128, 160),
        (128, 220),
        (4096, 700),
        (4096, 900),
        (4096, 1100),
        (4096, 1300),
    ]:
        for generated_len in (0, 32, 64):
            if generated_len >= output_len:
                continue
            rows.append({
                "prompt_len": prompt_len,
                "generated_len": generated_len,
            })
            targets.append(output_len - generated_len)

    x = build_feature_matrix(rows, feature_mode="len", hash_bins=8)
    kwargs = {
        "feature_mode": "len",
        "hash_bins": 8,
        "min_samples_leaf": 1,
        "random_state": 7,
        "quantiles": (0.5, 0.95),
    }
    if model_type == "qrf":
        kwargs["n_estimators"] = 8
    else:
        kwargs["max_iter"] = 32
        kwargs["max_leaf_nodes"] = 7
    predictor = create_length_predictor(model_type, **kwargs)
    predictor.fit(x, targets)
    path = tmp_path / f"{model_type}_length.joblib"
    predictor.save(path)
    return str(path)


def test_progressive_active_remaining_requires_model_len_predictor():
    config = SimulatorConfig(
        latency_model_path="/tmp/missing-decode-simulator-model.json")
    predictor = DPoolSaturationPredictor(config.model, config.saturation)

    req = Request(
        request_id=10,
        request_type=RequestType.RT,
        prompt_len=128,
        target_output_len=9999,
        arrival_time_s=0.0,
    )

    try:
        predictor.predict_active_remaining_len(req)
    except RuntimeError as exc:
        assert "length predictor" in str(exc)
    else:
        raise AssertionError("missing length predictor should fail")


def test_progressive_active_remaining_uses_qrf_model_len_predictor(tmp_path):
    path = _write_toy_model_len_predictor(tmp_path, "qrf")
    config = SimulatorConfig(
        latency_model_path="/tmp/missing-decode-simulator-model.json",
        model_len_predictor_path=path,
        model_len_predictor_type="qrf",
    )
    simulator = DecodeSimulator(config, strategy=DecodeStrategy.PROGRESSIVE)
    req = Request(
        request_id=11,
        request_type=RequestType.RT,
        prompt_len=128,
        target_output_len=9999,
        arrival_time_s=0.0,
        tokens_generated=64,
    )

    remaining = simulator.saturation_predictor.predict_active_remaining_len(req)

    assert 0 < remaining < 300


def test_progressive_active_remaining_uses_gbdt_model_len_predictor(tmp_path):
    path = _write_toy_model_len_predictor(tmp_path, "gbdt")
    config = SimulatorConfig(
        latency_model_path="/tmp/missing-decode-simulator-model.json",
        model_len_predictor_path=path,
        model_len_predictor_type="gbdt",
    )
    simulator = DecodeSimulator(config, strategy=DecodeStrategy.PROGRESSIVE)
    req = Request(
        request_id=12,
        request_type=RequestType.RT,
        prompt_len=4096,
        target_output_len=9999,
        arrival_time_s=0.0,
        tokens_generated=64,
    )

    remaining = simulator.saturation_predictor.predict_active_remaining_len(req)

    assert remaining > 100


def test_progressive_future_forecasts_use_online_arima_arrival_predictor():
    config = SimulatorConfig(
        latency_model_path="/tmp/missing-decode-simulator-model.json")
    config.saturation.lookahead_window_s = 3.0
    simulator = DecodeSimulator(config, strategy=DecodeStrategy.PROGRESSIVE)
    simulator._reset_runtime_state([], [])
    simulator.current_time_s = 10.0

    for output_len in range(100, 501, 50):
        simulator.saturation_predictor.update_model_len_history(256, output_len)
    for arrival_time_s in [6.0, 6.5, 7.0, 7.5, 8.0, 8.5]:
        simulator.saturation_predictor.observe_rt_arrival_time(arrival_time_s)

    forecasts = simulator._progressive_future_rt_forecasts()
    offsets = [f.arrival_time_s - simulator.current_time_s for f in forecasts]

    assert forecasts
    assert offsets == sorted(offsets)
    assert all(0.0 < offset <= config.saturation.lookahead_window_s
               for offset in offsets)
    assert all(f.prompt_len > 0 and f.remaining_tokens > 0 for f in forecasts)


def test_simulator_time_predictor_respects_decode_full_layer_flag(tmp_path):
    model_path = tmp_path / "time_model.json"
    model_path.write_text(
        """
        {
          "num_layers": 1,
          "use_decode_full_layer": false,
          "decode_attn": {"coef": [1.0, 0.0], "intercept": 0.0},
          "decode_full_layer": {"coef": [100.0, 0.0], "intercept": 0.0},
          "linear": {"width": 32, "coef": {}, "fallback": [0.0, 10.0]}
        }
        """
    )
    predictor = TimePredictor(str(model_path), num_layers=1)

    assert predictor.predict_forward_ms([], [(1, 128)]) == 11.0


def test_simulator_time_predictor_decode_sum_context_uses_query_len(tmp_path):
    model_path = tmp_path / "time_model.json"
    model_path.write_text(
        """
        {
          "num_layers": 1,
          "decode_attn": {"coef": [0.0, 1.0], "intercept": 0.0},
          "linear": {"width": 32, "coef": {}, "fallback": [0.0, 0.0]}
        }
        """
    )
    predictor = TimePredictor(str(model_path), num_layers=1)

    assert predictor.predict_forward_ms([], [(3, 10)]) == 30.0
