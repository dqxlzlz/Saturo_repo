"""D-pool simulator with a local, end-to-end BE request lifecycle.

The regular decode simulator models two independent BE streams:

* prefill-only requests that finish after their prompt is processed; and
* decode-ready requests whose prompt KV is handed to the D-pool.

This variant intentionally removes that split-phase shortcut.  Every BE
request is drawn once from ``be_workload_csv``, is chunk-prefilled locally,
and joins the resident decode queue only after its own prefill completes.
Completed-request throughput is therefore measured directly rather than as
``min(prefill_rate, decode_rate)``.

No future-BE maximum prompt is inspected or reserved.  The current request's
actual KV growth is checked against the current ORACLE/PROGRESSIVE RT budget
for every chunk.  Within that budget the largest latency-feasible chunk is
chosen, so each iteration uses as much of ``target_tbt_ms`` as the current
decode batch, request remainder, and KV budget allow.
"""

from __future__ import annotations

import copy
import math
from typing import List

from .config import SimulatorConfig
from .request import Request, RequestState
from .simulator import (
    DecodeSimulator,
    DecodeStrategy,
    IterationStats,
    SimulationResult,
)


class LocalDecodeSimulator(DecodeSimulator):
    """Decode simulator whose BE requests run local prefill -> decode."""

    def __init__(
        self,
        config: SimulatorConfig,
        strategy: DecodeStrategy = DecodeStrategy.PROGRESSIVE,
    ):
        # Do not mutate a config that may also be used for a split-phase run.
        local_config = copy.deepcopy(config)
        local_config.be_prefill_reserve_blocks = 0
        local_config.balance_be_throughput = False
        super().__init__(local_config, strategy=strategy)
        self._be_prefill_completion_times: List[float] = []

    def _reset_runtime_state(
        self,
        rt_requests: List[Request],
        be_requests: List[Request],
    ):
        super()._reset_runtime_state(rt_requests, be_requests)
        self._be_prefill_completion_times = []

    def _refill_be_decode_backlog(self) -> None:
        """Do not synthesize decode-ready BE work in local mode.

        ``be_decode_backlog`` remains useful for locally-prefilled requests
        that were swapped out during decode; those are inserted by the base
        block-reclamation path and can later resume without re-prefilling.
        """

    def _find_be_prefill_chunk_by_latency(
        self,
        rt_decoding: List[Request],
        be_decoding: List[Request],
        scheduled_prefills: List[tuple[Request, int]],
        prefill_req: Request,
        max_prefill_tokens: int,
        target_tbt_ms: float,
        be_budget_blocks: int,
    ) -> int:
        """Return the largest granularity-aligned chunk fitting target TBT.

        The split simulator's ``auto`` candidates stop at 4096 tokens.  A
        local request can have a much longer prompt, and on a fast/empty step
        a larger chunk may still fit.  Search the entire current remainder,
        using the configured minimum chunk as the granularity and always
        including the exact final remainder.
        """
        if max_prefill_tokens <= 0:
            return 0

        positive_chunks = [
            int(chunk) for chunk in self.config.be_chunk_sizes if chunk > 0
        ]
        granularity = min(positive_chunks, default=16)
        candidate_count = int(
            math.ceil(max_prefill_tokens / max(granularity, 1)))

        best = 0
        lo, hi = 1, candidate_count
        while lo <= hi:
            mid = (lo + hi) // 2
            chunk = min(mid * granularity, max_prefill_tokens)
            target_blocks = self._blocks_for_tokens(
                prefill_req.prefill_tokens_processed + chunk)
            if not self._can_account_be_blocks(
                    prefill_req, target_blocks, be_budget_blocks):
                hi = mid - 1
                continue

            predicted = self._predict_iteration_latency(
                rt_decoding,
                be_decoding,
                scheduled_prefills + [(prefill_req, chunk)],
            )
            if predicted <= target_tbt_ms + 1e-9:
                best = chunk
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _advance_be_prefills(
        self,
        prefill_chunks: List[tuple[Request, int]],
        end_time_s: float,
    ):
        """Move a locally-prefilled request into its own decode phase."""
        for req, chunk in prefill_chunks:
            if chunk <= 0:
                continue
            req.process_prefill_chunk(chunk)
            if req.state != RequestState.DECODING:
                continue

            req.decode_start_time = end_time_s
            if req not in self.be_decoding:
                self.be_decoding.append(req)
            self._be_prefill_completion_times.append(end_time_s)
            if self.be_prefilling is req:
                self.be_prefilling = None

    def _compile_results(
        self,
        stats_list: List[IterationStats],
    ) -> SimulationResult:
        result = super()._compile_results(stats_list)

        if any(stat.time_s >= self.config.warmup_s for stat in stats_list):
            measurement_start_s = self.config.warmup_s
        else:
            measurement_start_s = 0.0
        measured_time_s = max(
            self.current_time_s - measurement_start_s, 1e-9)

        prefill_completions = sum(
            completion_s >= measurement_start_s
            for completion_s in self._be_prefill_completion_times
        )
        lifecycle_completions = sum(
            req.finish_time is not None
            and req.finish_time >= measurement_start_s
            for req in self.finished_be
        )

        # In local mode every decode completion is an end-to-end completion.
        # Keep the component rates for diagnostics, but the reported BE
        # saturation rate is the directly observed lifecycle completion rate.
        result.be_prefill_req_per_s = (
            prefill_completions / measured_time_s)
        result.be_decode_req_per_s = (
            lifecycle_completions / measured_time_s)
        result.be_saturation_req_per_s = result.be_decode_req_per_s
        return result

