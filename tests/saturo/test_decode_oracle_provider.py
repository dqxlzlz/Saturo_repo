from __future__ import annotations

import json

from saturo.saturation import OracleDecodeSaturationStateProvider


class _SamplingParams:

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens


class _Seq:

    def __init__(self, generated: int) -> None:
        self.generated = generated

    def get_output_len(self) -> int:
        return self.generated


class _SeqGroup:

    def __init__(self, max_tokens: int, generated: int = 0) -> None:
        self.sampling_params = _SamplingParams(max_tokens)
        self._seqs = [_Seq(generated)]

    def get_seqs(self):
        return self._seqs


def test_oracle_provider_replays_exact_future_and_remaining(tmp_path) -> None:
    schedule = tmp_path / "oracle.json"
    schedule.write_text(json.dumps({
        "lookahead_window_s": 60.0,
        "requests": [
            {
                "arrival_offset_s": 0.0,
                "prompt_len": 100,
                "output_len": 10,
            },
            {
                "arrival_offset_s": 0.5,
                "prompt_len": 200,
                "output_len": 20,
            },
            {
                "arrival_offset_s": 2.0,
                "prompt_len": 300,
                "output_len": 30,
            },
        ],
    }), encoding="utf-8")

    provider = OracleDecodeSaturationStateProvider(str(schedule))
    first = _SeqGroup(max_tokens=10)
    provider.observe_rt_arrival(first, now_s=100.0)

    future = list(provider.get_future_arrivals(
        now_s=100.25, window_s=1.0))
    assert len(future) == 1
    assert future[0].arrival_time_s == 100.5
    assert future[0].prompt_len == 200
    assert future[0].remaining_tokens == 20

    active = _SeqGroup(max_tokens=20, generated=7)
    assert provider.get_remaining_tokens(active) == 13
    provider.observe_rt_arrival(active, now_s=100.5)

    future = list(provider.get_future_arrivals(
        now_s=100.5, window_s=2.0))
    assert len(future) == 1
    assert future[0].arrival_time_s == 102.0
    assert future[0].prompt_len == 300
    assert future[0].remaining_tokens == 30
