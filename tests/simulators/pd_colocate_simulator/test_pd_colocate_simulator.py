import math

from saturo.simulators.pd_colocate_simulator.config import (
    ModelConfig,
    PDColocateConfig,
    SolverConfig,
    WorkloadStats,
)
from saturo.simulators.pd_colocate_simulator.simulator import (
    PDColocateSimulator,
)


class ConstantLatency:
    """Small deterministic latency stub for solver invariants."""

    def __init__(self, seconds=0.1):
        self.seconds = seconds

    def predict_iter_s(self, *args):
        return self.seconds


def _sim(total_blocks=10_000, balance_be_throughput=True):
    config = PDColocateConfig(
        model=ModelConfig(
            total_gpu_blocks=total_blocks,
            block_size=16,
            memory_watermark=1.0,
        ),
        workload=WorkloadStats(
            rt_prompt_len=100,
            rt_output_len=20,
            be_prompt_len=200,
            be_output_len=50,
        ),
        solver=SolverConfig(
            tol_s=1e-9,
            balance_be_throughput=balance_be_throughput,
            be_balance_search_epsilon=1e-6,
        ),
        chunk_sizes=[64, 128, 256],
    )
    return PDColocateSimulator(config, ConstantLatency())


def test_feasible_point_sustains_rt_and_uses_full_chunk_budget():
    point = _sim().solve(chunk_size=256, rps=2.0)

    assert point.feasible
    assert math.isclose(point.rt_prefill_req_per_s, 2.0)
    assert math.isclose(point.rt_decode_req_per_s, 2.0)
    assert math.isclose(
        point.be_goodput_req_per_s,
        min(point.be_prefill_req_per_s, point.be_decode_req_per_s),
    )
    used_tokens = (
        point.rt_decode_batch
        + point.rt_prefill_tokens
        + point.be_decode_batch
        + point.be_prefill_tokens
    )
    assert math.isclose(used_tokens, point.chunk_size)
    assert point.kv_blocks_used <= point.kv_block_budget


def test_balanced_policy_binary_searches_be_prefill_decode_crossover():
    point = _sim().solve(chunk_size=256, rps=2.0)

    assert point.feasible
    assert point.be_decode_batch < point.be_decode_capacity_batch
    assert math.isclose(
        point.be_prefill_req_per_s,
        point.be_decode_req_per_s,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    # With t=0.1, RT consumes 24 tokens and BE has 232 tokens. The
    # continuous balance is D/50 == (232-D)/200, hence D=46.4.
    assert math.isclose(point.be_decode_batch, 46.4, abs_tol=1e-5)
    assert math.isclose(point.be_prefill_tokens, 185.6, abs_tol=1e-5)


def test_legacy_greedy_policy_can_starve_be_prefill():
    balanced = _sim(balance_be_throughput=True).solve(256, 2.0)
    greedy = _sim(balance_be_throughput=False).solve(256, 2.0)

    assert greedy.feasible
    assert math.isclose(
        greedy.be_decode_batch, greedy.be_decode_capacity_batch)
    assert greedy.be_prefill_tokens == 0.0
    assert greedy.be_goodput_req_per_s == 0.0
    assert balanced.be_goodput_req_per_s > greedy.be_goodput_req_per_s


def test_balanced_policy_uses_decode_capacity_when_kv_limited():
    point = _sim(total_blocks=100).solve(chunk_size=256, rps=2.0)

    assert point.feasible
    assert math.isclose(
        point.be_decode_batch, point.be_decode_capacity_batch,
        rel_tol=1e-6)
    assert point.be_decode_req_per_s < point.be_prefill_req_per_s


def test_rt_chunk_budget_can_make_point_infeasible():
    point = _sim().solve(chunk_size=8, rps=10.0)

    assert not point.feasible
    assert point.infeasible_reason == "rt_exceeds_chunk_budget"


def test_optimizer_selects_maximum_feasible_be_goodput():
    result = _sim().optimize(2.0)
    feasible = [p for p in result.all_points if p.feasible]

    assert result.best is not None
    assert result.best.be_goodput_req_per_s == max(
        p.be_goodput_req_per_s for p in feasible)
