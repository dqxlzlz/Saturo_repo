from __future__ import annotations

from saturo.saturation.decode_saturation_core import DecodeSaturationEngine
from saturo.saturation.types import RequestForecast, SaturationParams


class _AggregatePredictor:

    @staticmethod
    def from_lists(prefill_reqs, decode_reqs) -> float:
        del prefill_reqs
        return (
            2.0
            + 0.25 * len(decode_reqs)
            + 0.001 * sum(context for _, context in decode_reqs)
        )

    @staticmethod
    def from_aggregates(
        *,
        prefill_tokens,
        prefill_sum_ctx,
        decode_tokens,
        decode_sum_ctx,
        prefill_request_count,
        decode_request_count,
    ) -> float:
        del (
            prefill_tokens,
            prefill_sum_ctx,
            decode_tokens,
            prefill_request_count,
        )
        return 2.0 + 0.25 * decode_request_count + 0.001 * decode_sum_ctx


def _forecasts() -> list[RequestForecast]:
    return [
        RequestForecast(
            request_id=index,
            prompt_len=100 + index * 7,
            generated_tokens=index % 5,
            remaining_tokens=20 + index,
            arrival_time_s=(index % 4) * 0.13,
        )
        for index in range(32)
    ]


def test_aggregate_tau_evaluation_matches_tuple_predictor() -> None:
    params = SaturationParams(
        tbt_slo_ms=150.0,
        lookahead_window_s=5.0,
        block_size=16,
        total_gpu_blocks=100_000,
    )
    tuple_engine = DecodeSaturationEngine(
        params, _AggregatePredictor.from_lists)
    aggregate_engine = DecodeSaturationEngine(
        params,
        _AggregatePredictor.from_lists,
        _AggregatePredictor.from_aggregates,
    )

    for tau_ms in (20.0, 75.0, 149.5, 150.0):
        expected = tuple_engine._evaluate_tau_vectorized(
            _forecasts(), tau_ms, now_s=0.0)
        actual = aggregate_engine._evaluate_tau_vectorized(
            _forecasts(), tau_ms, now_s=0.0)
        assert actual == expected


def test_full_slo_is_returned_without_binary_search_when_feasible() -> None:
    params = SaturationParams(
        tbt_slo_ms=150.0,
        lookahead_window_s=5.0,
        search_epsilon_ms=0.5,
        block_size=16,
        total_gpu_blocks=100_000,
    )
    engine = DecodeSaturationEngine(
        params,
        _AggregatePredictor.from_lists,
        _AggregatePredictor.from_aggregates,
    )

    decision = engine.compute_decision(
        active_rt=[],
        future_rt=_forecasts(),
        current_rt_blocks=0,
        current_be_blocks=0,
        now_s=0.0,
    )

    assert decision.search_feasible
    assert decision.target_interval_ms == 150.0
