"""Latency-model adapter for P-pool mixed BE chunks."""

from typing import List, Tuple

from saturo.predictors.time_predictor import TimePredictor

from .config import BEConfig
from .request import BEChunk


class LatencyPredictor:
    """Small wrapper around vLLM's TimePredictor.

    ``prefill_reqs`` and ``decode_reqs`` use the TimePredictor convention:
    each item is ``(query_len, context_len)``.
    """

    def __init__(self, predictor: TimePredictor):
        self.predictor = predictor

    def predict_be_chunk_s(
        self,
        decode_batch_size: int,
        prefill_chunk_size: int,
        avg_be_decode_context: int,
        prefill_context_len: int = 0,
    ) -> float:
        decode_reqs: List[Tuple[int, int]] = []
        if decode_batch_size > 0:
            decode_reqs = [(1, avg_be_decode_context)] * decode_batch_size

        prefill_reqs: List[Tuple[int, int]] = []
        if prefill_chunk_size > 0:
            context_len = max(prefill_chunk_size, prefill_context_len)
            prefill_reqs = [(prefill_chunk_size, context_len)]

        if not decode_reqs and not prefill_reqs:
            return 0.0
        return max(
            0.0, self.predictor.predict_forward_ms(prefill_reqs, decode_reqs)
        ) / 1000.0

    def build_chunk_for_latency(
        self,
        target_latency_s: float,
        be_config: BEConfig,
    ) -> BEChunk:
        """Build the largest mixed BE chunk fitting ``target_latency_s``."""
        decode_batch = 0
        base_s = self.predict_be_chunk_s(
            decode_batch, 0, be_config.avg_be_decode_context)

        if target_latency_s <= 0.0:
            return BEChunk(
                target_latency_s=target_latency_s,
                latency_s=base_s,
                decode_batch_size=decode_batch,
                prefill_chunk_size=0,
            )

        if base_s > target_latency_s:
            return BEChunk(
                target_latency_s=target_latency_s,
                latency_s=base_s,
                decode_batch_size=decode_batch,
                prefill_chunk_size=0,
            )

        granularity = max(1, int(be_config.prefill_search_granularity))
        left = max(0, int(be_config.min_prefill_chunk))
        right = max(left, int(be_config.max_prefill_chunk))
        left = (left // granularity) * granularity
        right = (right // granularity) * granularity

        best_prefill = 0
        best_latency_s = base_s
        while left <= right:
            mid = ((left + right) // (2 * granularity)) * granularity
            mid = max(0, mid)
            decode_batch = be_config.steady_state_decode_batch(mid)
            latency_s = self.predict_be_chunk_s(
                decode_batch,
                mid,
                be_config.avg_be_decode_context,
                be_config.average_prefill_context(mid),
            )
            if latency_s <= target_latency_s:
                best_prefill = mid
                best_latency_s = latency_s
                left = mid + granularity
            else:
                right = mid - granularity

        return BEChunk(
            target_latency_s=target_latency_s,
            latency_s=best_latency_s,
            decode_batch_size=be_config.steady_state_decode_batch(best_prefill),
            prefill_chunk_size=best_prefill,
        )

    def find_prefill_chunk_for_decode_batch(
        self,
        target_latency_s: float,
        be_config: BEConfig,
        decode_batch: int,
    ) -> Tuple[int, float]:
        """Largest BE prefill chunk fitting the budget for a FIXED decode batch.

        This mirrors the real engine's
        ``Scheduler._find_be_prefill_chunk_by_latency``: the decode batch is
        held constant (it is admitted first), and the BE prefill chunk is the
        largest token count whose predicted mixed-step latency stays under
        ``target_latency_s``.  Returns ``(prefill_chunk, latency_s)``.
        """
        decode_batch = max(0, int(decode_batch))
        base_s = self.predict_be_chunk_s(
            decode_batch, 0, be_config.avg_be_decode_context)
        if target_latency_s <= 0.0 or base_s > target_latency_s:
            # Decode alone already exceeds the budget: no room for prefill.
            return 0, base_s

        granularity = max(1, int(be_config.prefill_search_granularity))
        left = max(0, int(be_config.min_prefill_chunk))
        right = max(left, int(be_config.max_prefill_chunk))
        left = (left // granularity) * granularity
        right = (right // granularity) * granularity

        best_prefill = 0
        best_latency_s = base_s
        while left <= right:
            mid = ((left + right) // (2 * granularity)) * granularity
            mid = max(0, mid)
            latency_s = self.predict_be_chunk_s(
                decode_batch,
                mid,
                be_config.avg_be_decode_context,
                be_config.average_prefill_context(mid),
            )
            if latency_s <= target_latency_s:
                best_prefill = mid
                best_latency_s = latency_s
                left = mid + granularity
            else:
                right = mid - granularity
        return best_prefill, best_latency_s

    def balance_decode_batch(
        self,
        target_latency_s: float,
        be_config: BEConfig,
        max_decode_batch: int | None = None,
    ) -> dict:
        """Pick the BE-decode active-count cap that balances P and D throughput.

        Models the real P-pool engine: within one ``be_chunk_ms`` budget the
        engine admits ``D`` decode requests first, then fills the remaining
        latency with a BE prefill chunk. As ``D`` grows the prefill chunk (and
        prefill throughput) shrinks while decode throughput grows, so
        ``min(prefill_req_s, decode_req_s)`` is unimodal in ``D``.  We binary
        search for the crossover where the two rates meet -- that ``D`` is the
        ``max_be_decode_batch`` that avoids BE-prefill starvation while keeping
        the two BE throughputs comparable.

        Per chunk of measured latency ``L`` the decode side emits ``D`` tokens
        (one per request) and the prefill side emits ``chunk`` tokens, so
        ``decode_req_s = D / (L * avg_out)`` and
        ``prefill_req_s = chunk / (L * avg_in)``.
        """
        d_max = int(be_config.decode_batch_size
                    if max_decode_batch is None else max_decode_batch)
        d_max = max(0, d_max)
        avg_in = max(1, int(be_config.avg_be_length))
        avg_out = max(1, int(be_config.avg_be_output_len))

        def eval_d(d: int) -> dict:
            d = max(0, int(d))
            chunk, latency_s = self.find_prefill_chunk_for_decode_batch(
                target_latency_s, be_config, d)
            if latency_s <= 0.0:
                return {
                    "decode_batch": d, "prefill_chunk": chunk,
                    "latency_s": latency_s,
                    "decode_req_s": 0.0, "prefill_req_s": 0.0,
                }
            return {
                "decode_batch": d, "prefill_chunk": chunk,
                "latency_s": latency_s,
                "decode_req_s": d / (latency_s * avg_out),
                "prefill_req_s": chunk / (latency_s * avg_in),
            }

        if d_max <= 0 or target_latency_s <= 0.0:
            return eval_d(d_max)

        # Binary search the smallest D where decode_req_s >= prefill_req_s
        # (the crossover); the throughput-balancing optimum is at dc-1 or dc.
        lo, hi = 0, d_max
        while lo < hi:
            mid = (lo + hi) // 2
            r = eval_d(mid)
            if r["decode_req_s"] >= r["prefill_req_s"]:
                hi = mid
            else:
                lo = mid + 1
        dc = lo

        def score(r: dict) -> float:
            return min(r["decode_req_s"], r["prefill_req_s"])

        best = None
        for d in (dc - 1, dc, dc + 1):
            if 0 <= d <= d_max:
                r = eval_d(d)
                if best is None or score(r) > score(best):
                    best = r
        return best if best is not None else eval_d(dc)
