"""Latency predictor wrapper for the decode simulator.

Wraps the shared saturo TimePredictor to provide the interface needed by
the simulator: predict iteration latency and find the max prefill chunk
that fits within a time budget. Calibration (latency_scale/offset) and
runtime overhead are owned by the TimePredictor itself (loaded from the
profile), so this wrapper only forwards batch info.
"""

from typing import List, Optional, Tuple

from saturo.predictors.time_predictor import TimePredictor


class LatencyPredictor:
    """Wraps TimePredictor for the decode simulator."""

    def __init__(
        self,
        fitted_model_path: str,
    ):
        self.predictor = TimePredictor(fitted_model_path)

    def predict_iteration_ms(
        self,
        decode_reqs: List[Tuple[int, int]],
        prefill_reqs: Optional[List[Tuple[int, int]]] = None,
    ) -> float:
        """Predict iteration latency in ms.

        Args:
            decode_reqs: [(1, context_len), ...] for each decode request
            prefill_reqs: [(chunk_size, context_len), ...] for prefill chunks
        Returns:
            Predicted latency in ms.
        """
        if prefill_reqs is None:
            prefill_reqs = []
        raw_ms = self.predictor.predict_forward_ms(prefill_reqs, decode_reqs)
        return max(raw_ms, 1.0)

    def predict_iteration_from_aggregates_ms(
        self,
        *,
        prefill_tokens: float,
        prefill_sum_ctx: float,
        decode_tokens: float,
        decode_sum_ctx: float,
        prefill_request_count: int,
        decode_request_count: int,
    ) -> float:
        """Predict from sufficient statistics without rebuilding tuples."""
        raw_ms = self.predictor.predict_forward_from_aggregates_ms(
            prefill_tokens=prefill_tokens,
            prefill_sum_ctx=prefill_sum_ctx,
            decode_tokens=decode_tokens,
            decode_sum_ctx=decode_sum_ctx,
            prefill_request_count=prefill_request_count,
            decode_request_count=decode_request_count,
        )
        return max(raw_ms, 1.0)

    def predict_decode_only_ms(
        self,
        rt_requests,
        be_decode_requests=None,
    ) -> float:
        """Predict decode-only iteration latency.

        Args:
            rt_requests: list of Request objects in decode phase
            be_decode_requests: optional list of BE Request objects in decode
        Returns:
            Predicted latency in ms.
        """
        decode_reqs = [(1, r.current_seq_len) for r in rt_requests]
        if be_decode_requests:
            decode_reqs += [(1, r.current_seq_len) for r in be_decode_requests]
        if not decode_reqs:
            return 1.0
        return self.predict_iteration_ms(decode_reqs)

    def predict_mixed_ms(
        self,
        rt_requests,
        be_decode_requests,
        prefill_chunk_size: int,
        prefill_context_len: int,
    ) -> float:
        """Predict mixed decode + prefill chunk iteration latency.

        Args:
            rt_requests: list of RT Request objects in decode phase
            be_decode_requests: list of BE Request objects in decode phase
            prefill_chunk_size: number of prefill tokens in chunk
            prefill_context_len: cached context length before this chunk
        Returns:
            Predicted latency in ms.
        """
        decode_reqs = [(1, r.current_seq_len) for r in rt_requests]
        decode_reqs += [(1, r.current_seq_len) for r in be_decode_requests]
        prefill_reqs = [(prefill_chunk_size, prefill_context_len + prefill_chunk_size)]
        return self.predict_iteration_ms(decode_reqs, prefill_reqs)

    def find_max_prefill_chunk(
        self,
        time_budget_ms: float,
        rt_requests,
        be_decode_requests,
        prefill_context_len: int,
        candidate_chunk_sizes: List[int],
    ) -> int:
        """Find the largest prefill chunk that fits within the time budget.

        Binary-search style over candidate sizes. Returns 0 if none fits.
        """
        decode_only_ms = self.predict_decode_only_ms(rt_requests, be_decode_requests)
        if decode_only_ms >= time_budget_ms:
            return 0

        candidates = sorted(set(candidate_chunk_sizes))
        best_chunk = 0
        lo, hi = 0, len(candidates) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            chunk_size = candidates[mid]
            mixed_ms = self.predict_mixed_ms(
                rt_requests, be_decode_requests, chunk_size, prefill_context_len
            )
            if mixed_ms <= time_budget_ms:
                best_chunk = chunk_size
                lo = mid + 1
            else:
                hi = mid - 1
        return best_chunk
