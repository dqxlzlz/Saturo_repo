#!/usr/bin/env python3
"""CPU-only smoke checks for vllm.core.decode_saturation.

Run from the repository root:

    /root/hybrid_serve_venv/bin/python tests/core/decode_saturation_policy_smoke.py

The script uses tiny fake SequenceGroup/Sequence objects, so it does not start
an engine, allocate KV cache, or touch CUDA.
"""

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional


os.environ["VLLM_DECODE_SATURATION"] = "1"
os.environ["VLLM_DECODE_SATURATION_TBT_MS"] = "100"
os.environ["VLLM_DECODE_SATURATION_LOOKAHEAD_S"] = "0.4"
os.environ["VLLM_DECODE_SATURATION_MIN_INTERVAL_MS"] = "10"
os.environ["VLLM_DECODE_SATURATION_EPSILON_MS"] = "0.25"
os.environ["VLLM_DECODE_SATURATION_RT_BLOCKS"] = "3"
os.environ["VLLM_DECODE_SATURATION_BE_BLOCKS"] = "5"
os.environ["VLLM_DECODE_SATURATION_RT_BURST_BLOCKS"] = "1"

from saturo.saturation import (  # noqa: E402
    DefaultDecodeSaturationStateProvider,
    DecodeSaturationPolicy,
    DecodeSaturationRequestState,
    sampling_offsets,
)


@dataclass
class FakeSamplingParams:
    max_tokens: int


class FakeSeq:

    def __init__(self, prompt_len: int, output_len: int) -> None:
        self.prompt_len = prompt_len
        self.output_len = output_len

    def get_prompt_len(self) -> int:
        return self.prompt_len

    def get_output_len(self) -> int:
        return self.output_len

    def get_len(self) -> int:
        return self.prompt_len + self.output_len

    def get_num_computed_tokens(self) -> int:
        return self.get_len()

    def get_prompt_token_ids(self):
        return tuple(range(self.prompt_len))

    def get_output_token_ids(self):
        return tuple(range(100, 100 + self.output_len))


class FakeSeqGroup:

    def __init__(
        self,
        request_id: str,
        prompt_len: int,
        output_len: int,
        max_tokens: int,
        *,
        prefill_ddl: float = 1.0,
        arrival_time: float = 0.0,
        is_realtime: bool = True,
        is_prefill: bool = False,
    ) -> None:
        self.request_id = request_id
        self.prefill_ddl = prefill_ddl
        self.arrival_time = arrival_time
        self.is_realtime = is_realtime
        self.sampling_params = FakeSamplingParams(max_tokens=max_tokens)
        self._seqs = [FakeSeq(prompt_len, output_len)]
        self._is_prefill = is_prefill

    def get_seqs(self) -> List[FakeSeq]:
        return self._seqs

    def is_prefill(self) -> bool:
        return self._is_prefill


class StaticProvider:

    def __init__(self, future_arrivals: Iterable[DecodeSaturationRequestState]):
        self.future_arrivals = list(future_arrivals)

    def get_remaining_tokens(self, seq_group: FakeSeqGroup) -> Optional[int]:
        generated = max(seq.get_output_len() for seq in seq_group.get_seqs())
        return max(seq_group.sampling_params.max_tokens - generated, 0)

    def get_future_arrivals(
        self,
        now_s: float,
        window_s: float,
    ) -> Iterable[DecodeSaturationRequestState]:
        del now_s, window_s
        return self.future_arrivals


class FakeTimePredictor:

    def predict_forward_ms(self, prefill_reqs, decode_reqs) -> float:
        decode_ctx = sum(ctx for _, ctx in decode_reqs)
        prefill_tokens = sum(tokens for tokens, _ in prefill_reqs)
        return 1.0 + 0.2 * len(decode_reqs) + 0.01 * decode_ctx + (
            0.005 * prefill_tokens)


def assert_close(actual: float, expected: float, eps: float = 1e-9) -> None:
    assert abs(actual - expected) <= eps, (actual, expected)


def make_policy() -> DecodeSaturationPolicy:
    policy = DecodeSaturationPolicy(block_size=16, total_gpu_blocks=10)
    policy.time_predictor = FakeTimePredictor()
    return policy


def test_sampling_offsets() -> None:
    offsets = sampling_offsets(window_s=0.35, tau_s=0.1)
    assert offsets == [0.0, 0.1, 0.2, 0.3, 0.35]
    print("PASS sampling_offsets:", offsets)


def test_predicted_length_snapshots() -> None:
    policy = make_policy()
    now_s = 10.0
    active = [
        DecodeSaturationRequestState(
            request_id="rt_active",
            prompt_len=15,
            generated_tokens=1,
            remaining_tokens=3,
            arrival_time_s=now_s,
        )
    ]
    future = [
        DecodeSaturationRequestState(
            request_id="rt_future",
            prompt_len=10,
            generated_tokens=0,
            remaining_tokens=2,
            arrival_time_s=now_s + 0.15,
        )
    ]
    policy.provider = StaticProvider(future)

    engine = policy._get_engine()
    all_forecasts = active + future
    snapshots = {
        round(offset, 2): lengths
        for offset, lengths in engine._iter_predicted_length_snapshots(
            all_forecasts, tau_ms=100.0, now_s=now_s)
    }
    assert snapshots == {
        0.0: [16],
        0.1: [17],
        0.2: [18, 10],
        0.3: [19, 11],
        0.4: [12],
    }
    peak_blocks = engine._predict_peak_rt_blocks(
        all_forecasts, tau_ms=100.0, now_s=now_s)
    assert peak_blocks == 3
    print("PASS predicted_length_snapshots:", snapshots)
    print("PASS block_rounding_peak:", peak_blocks)


def test_decision_and_be_budget() -> None:
    policy = make_policy()
    now_s = 10.0
    policy.provider = StaticProvider([
        DecodeSaturationRequestState(
            request_id="rt_future",
            prompt_len=10,
            generated_tokens=0,
            remaining_tokens=2,
            arrival_time_s=now_s + 0.15,
        )
    ])
    rt = FakeSeqGroup("rt", prompt_len=15, output_len=1, max_tokens=4)
    be_running = FakeSeqGroup(
        "be_running",
        prompt_len=20,
        output_len=0,
        max_tokens=4,
        prefill_ddl=math.inf,
    )

    decision = policy.make_decision(
        running=[rt, be_running],
        waiting=[],
        pending_decode_admission=[],
        now_s=now_s,
    )
    assert decision.enabled
    assert_close(decision.target_interval_ms, decision.t_rate_ms)
    assert decision.t_rate_ms <= 100.0
    assert decision.predicted_peak_rt_blocks == 3
    assert decision.current_rt_blocks == 1
    assert decision.current_be_blocks == 2
    assert decision.be_block_budget == 5
    assert decision.max_be_prefill_tokens == 48

    small_be = FakeSeqGroup(
        "small_be",
        prompt_len=32,
        output_len=0,
        max_tokens=4,
        prefill_ddl=math.inf,
    )
    large_be = FakeSeqGroup(
        "large_be",
        prompt_len=80,
        output_len=0,
        max_tokens=4,
        prefill_ddl=math.inf,
    )
    assert policy.can_admit_be(small_be, decision, projected_be_blocks=2)
    assert not policy.can_admit_be(large_be, decision, projected_be_blocks=2)
    print("PASS make_decision:", decision)
    print("PASS be_admission_budget")


def test_default_provider_uses_model_len_predictor() -> None:
    class FakeLengthPredictor:

        def __init__(self) -> None:
            self.calls = []

        def predict_remaining(self, **kwargs) -> int:
            self.calls.append(kwargs)
            return 10

    predictor = FakeLengthPredictor()
    provider = DefaultDecodeSaturationStateProvider(load_predictor=False)
    provider.model_len_predictor = predictor
    provider.model_len_predictor_eta = 0.9
    seq_group = FakeSeqGroup(
        "rt",
        prompt_len=8,
        output_len=2,
        max_tokens=5,
    )

    remaining = provider.get_remaining_tokens(seq_group)
    assert remaining == 3
    assert predictor.calls
    call = predictor.calls[0]
    assert call["eta"] == 0.9
    assert call["prompt_len"] == 8
    assert call["generated_len"] == 2
    assert call["prompt_token_ids"] == tuple(range(8))
    assert call["generated_token_ids"] == (100, 101)
    print("PASS default_provider_uses_model_len_predictor")


def test_default_provider_requires_model_len_predictor() -> None:
    provider = DefaultDecodeSaturationStateProvider(
        load_predictor=False,
        require_model_len_predictor=True,
    )
    seq_group = FakeSeqGroup(
        "rt_missing_predictor",
        prompt_len=8,
        output_len=2,
        max_tokens=5,
    )

    try:
        provider.get_remaining_tokens(seq_group)
    except RuntimeError as exc:
        assert "requires a model-len predictor" in str(exc)
    else:
        raise AssertionError("missing length predictor should fail")
    print("PASS default_provider_requires_model_len_predictor")


def test_default_future_arrivals_use_arima() -> None:
    policy = make_policy()
    policy.lookahead_window_s = 2.0
    for arrival_time in (0.0, 0.5, 1.0, 1.5, 2.0):
        policy.observe_rt_arrival(
            FakeSeqGroup(
                f"rt_{arrival_time}",
                prompt_len=12,
                output_len=0,
                max_tokens=7,
            ),
            now_s=arrival_time,
        )

    future = list(policy._future_arrivals(now_s=2.0))

    assert future
    assert [request.arrival_time_s for request in future] == sorted(
        request.arrival_time_s for request in future)
    assert future[0].request_id == "predicted_rt_1"
    assert future[0].prompt_len == 12
    assert future[0].remaining_tokens == 7
    print("PASS default_future_arrivals_use_arima")


def main() -> None:
    test_sampling_offsets()
    test_predicted_length_snapshots()
    test_decision_and_be_budget()
    test_default_provider_uses_model_len_predictor()
    test_default_provider_requires_model_len_predictor()
    test_default_future_arrivals_use_arima()
    print("ALL decode_saturation smoke checks passed.")


if __name__ == "__main__":
    main()
