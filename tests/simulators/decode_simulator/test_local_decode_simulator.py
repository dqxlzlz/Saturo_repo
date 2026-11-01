from saturo.simulators.decode_simulator.config import SimulatorConfig
from saturo.simulators.decode_simulator.local_simulator import (
    LocalDecodeSimulator,
)
from saturo.simulators.decode_simulator.request import (
    Request,
    RequestState,
    RequestType,
)
from saturo.simulators.decode_simulator.simulator import IterationStats


def _config(tmp_path, *, reserve_blocks=0):
    workload = tmp_path / "be.csv"
    workload.write_text(
        "request_id,prompt_len,output_len,arrival_offset_s\n"
        "be_0,32,2,0\n",
        encoding="utf-8",
    )
    return SimulatorConfig(
        be_workload_csv=str(workload),
        be_prefill_reserve_blocks=reserve_blocks,
        total_gpu_blocks=1024,
        memory_watermark=1.0,
        simulation_duration_s=2.0,
        warmup_s=0.0,
        latency_model_path="/tmp/missing-local-decode-model.json",
    )


def test_local_mode_drops_future_prefill_reserve_and_split_decode_stream(
    tmp_path,
):
    simulator = LocalDecodeSimulator(
        _config(tmp_path, reserve_blocks=999))
    simulator._reset_runtime_state([], [])

    assert simulator.config.be_prefill_reserve_blocks == 0
    assert simulator._saturation_engine.params.be_prefill_reserve_blocks == 0
    assert len(simulator.be_backlog) == 1
    assert list(simulator.be_decode_backlog) == []


def test_local_prefill_completion_joins_same_requests_decode_queue(tmp_path):
    simulator = LocalDecodeSimulator(_config(tmp_path))
    simulator._reset_runtime_state([], [])
    request = simulator.be_backlog.popleft()
    request.state = RequestState.PREFILLING
    request.prefill_start_time = 0.0
    simulator.be_prefilling = request
    assert simulator._try_account_be_blocks(request, 2, 1024)

    simulator._advance_be_prefills([(request, 32)], end_time_s=0.1)

    assert request.state == RequestState.DECODING
    assert request in simulator.be_decoding
    assert request not in simulator.finished_be
    assert simulator._request_blocks[request.request_id] == 2


def test_local_chunk_search_can_fill_target_beyond_split_auto_cap(tmp_path):
    simulator = LocalDecodeSimulator(_config(tmp_path))
    simulator._reset_runtime_state([], [])
    request = Request(
        request_id=123,
        request_type=RequestType.BE,
        prompt_len=10_000,
        target_output_len=1,
        arrival_time_s=0.0,
        state=RequestState.PREFILLING,
    )
    simulator._predict_iteration_latency = (
        lambda _rt, _be, prefills:
        100.0 if prefills[-1][1] <= 8000 else 200.0
    )

    chunk = simulator._find_be_prefill_chunk_by_latency(
        [], [], [], request, 10_000, 150.0, 1024)

    assert chunk == 8000


def test_local_throughput_counts_completed_lifecycles(tmp_path):
    simulator = LocalDecodeSimulator(_config(tmp_path))
    simulator._reset_runtime_state([], [])
    simulator.current_time_s = 2.0
    request = Request(
        request_id=321,
        request_type=RequestType.BE,
        prompt_len=32,
        target_output_len=2,
        arrival_time_s=0.0,
        state=RequestState.FINISHED,
        prefill_tokens_processed=32,
        tokens_generated=2,
        finish_time=1.5,
    )
    simulator.finished_be = [request]
    simulator._be_prefill_completion_times = [0.5]

    result = simulator._compile_results([
        IterationStats(
            iteration_id=0,
            time_s=0.0,
            be_prefill_chunk_size=32,
            num_be_decode=2,
        )
    ])

    assert result.be_prefill_req_per_s == 0.5
    assert result.be_decode_req_per_s == 0.5
    assert result.be_saturation_req_per_s == 0.5

