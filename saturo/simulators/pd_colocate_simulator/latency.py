"""Latency-model adapter for the PD-aggregated colocation simulator.

Wraps the shared saturo TimePredictor. For the linear iteration-time model,
a group of N decode sequences that all share the same context length ``c`` is
exactly equivalent to a single aggregated entry ``(N, c)`` (decode_n = N,
decode_sum_ctx = N * c), so we pass aggregated tuples to keep the fixed-point
loop cheap regardless of batch size.
"""

from typing import List, Tuple

from saturo.predictors.time_predictor import TimePredictor


class LatencyPredictor:
    def __init__(self, runtime_profile_path: str):
        self.predictor = TimePredictor(runtime_profile_path)
        if not self.predictor.available:
            raise ValueError(
                f"time predictor unavailable for profile: "
                f"{runtime_profile_path}")

    @property
    def total_gpu_blocks(self):
        return self.predictor.total_gpu_blocks

    @property
    def kv_block_size(self):
        return self.predictor.kv_block_size

    def predict_iter_s(
        self,
        rt_decode_n: float,
        rt_decode_ctx: float,
        be_decode_n: float,
        be_decode_ctx: float,
        rt_prefill_tokens: float,
        be_prefill_tokens: float,
    ) -> float:
        """Predict one iteration's latency in seconds.

        Prefill chunks follow the existing simulators' convention: a chunk of
        ``q`` tokens is modeled as a fresh prefill with context ``q`` (cached=0).
        """
        decode_reqs: List[Tuple[float, float]] = []
        if rt_decode_n > 0:
            decode_reqs.append((rt_decode_n, rt_decode_ctx))
        if be_decode_n > 0:
            decode_reqs.append((be_decode_n, be_decode_ctx))

        prefill_reqs: List[Tuple[float, float]] = []
        if rt_prefill_tokens > 0:
            prefill_reqs.append((rt_prefill_tokens, rt_prefill_tokens))
        if be_prefill_tokens > 0:
            prefill_reqs.append((be_prefill_tokens, be_prefill_tokens))

        if not decode_reqs and not prefill_reqs:
            return 0.0
        raw_ms = self.predictor.predict_forward_ms(prefill_reqs, decode_reqs)
        return max(raw_ms, 1e-3) / 1000.0
