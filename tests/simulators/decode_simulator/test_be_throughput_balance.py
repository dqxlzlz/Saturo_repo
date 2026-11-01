from saturo.simulators.decode_simulator.config import SimulatorConfig
from saturo.simulators.decode_simulator.simulator import DecodeSimulator


def _make_balancing_simulator(tmp_path) -> DecodeSimulator:
    workload = tmp_path / "be_balance_workload.csv"
    rows = [
        "request_id,prompt_len,output_len,arrival_offset_s",
        *[f"be_{i},100,10,0" for i in range(100)],
    ]
    workload.write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = SimulatorConfig(
        be_workload_csv=str(workload),
        be_chunk_sizes=list(range(16, 101, 16)),
        total_gpu_blocks=2000,
        memory_watermark=1.0,
        latency_model_path="/tmp/missing-decode-simulator-model.json",
    )
    simulator = DecodeSimulator(config)
    simulator._reset_runtime_state([], [])
    simulator.usable_gpu_blocks = 2000

    # L = 10 + 2D + 0.1P ms.  At a 100 ms target, with avg_in=100 and
    # avg_out=10, P and D request-equivalents meet at D=30, P=300.
    simulator._predict_iteration_latency = lambda rt, dec, prefills: (
        10.0
        + 2.0 * len(dec)
        + 0.1 * sum(chunk for _, chunk in prefills)
    )
    return simulator


def test_balancer_releases_excess_resident_decode_for_prefill(tmp_path):
    simulator = _make_balancing_simulator(tmp_path)
    simulator._admit_be_decode_ready(be_budget_blocks=1000)
    assert len(simulator.be_decoding) > 30

    selected = simulator._select_balanced_be_decode_requests(
        rt_decoding=[],
        decode_budget_blocks=1000,
        total_be_budget_blocks=1000,
        target_tbt_ms=100.0,
    )
    prefills = simulator._select_be_prefill_chunks(
        rt_decoding=[],
        be_decoding=selected,
        be_budget_blocks=1000,
        target_tbt_ms=100.0,
    )

    assert len(selected) == 30
    assert len(simulator.be_decoding) == 30
    assert sum(chunk for _, chunk in prefills) == 300
    assert simulator._predict_iteration_latency([], selected, prefills) == 100.0


def test_balancer_uses_decode_capacity_when_prefill_rate_is_still_higher(
    tmp_path,
):
    simulator = _make_balancing_simulator(tmp_path)

    # Seven blocks/request and a 35-block decode budget cap the batch at five.
    selected = simulator._select_balanced_be_decode_requests(
        rt_decoding=[],
        decode_budget_blocks=35,
        total_be_budget_blocks=1000,
        target_tbt_ms=100.0,
    )

    assert len(selected) == 5

