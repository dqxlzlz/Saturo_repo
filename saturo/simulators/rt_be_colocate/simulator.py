"""Event-driven simulator for RT-BE colocation on a Prefill node.

Models a single GPU where real-time (RT) prefill requests compete with
best-effort (BE) prefill chunks.  RT requests arrive at a configurable
rate and have strict priority (EDF — earliest deadline first).  BE work
is done in fixed-size chunks whose duration is a tunable parameter.

The core tension: larger BE chunks improve BE throughput but increase
RT tail latency (and thus SLO violations) because RT requests that
arrive mid-chunk must wait for the chunk to finish.

Two scheduling policies are supported:

``greedy`` (default)
    Start a BE chunk whenever no RT request is enqueued, regardless of
    whether an RT arrival is imminent.  Once started the chunk runs to
    completion (non-preemptive).  This maximises BE throughput and is
    the appropriate model for the limitation study.

``conservative``
    Only start a BE chunk when it can finish before the next known RT
    arrival.  Protects RT latency at the cost of BE throughput.

Usage::

    from saturo.simulators.rt_be_colocate.config import ColocateConfig
    from saturo.simulators.rt_be_colocate.simulator import ColocateSimulator

    config = ColocateConfig(
        ttft_slo_s=2.0,
        duration_s=600,
        be_chunk_latency_s=0.050,   # 50 ms chunk
        be_chunk_tokens=4096,
        policy="greedy",
    )
    sim = ColocateSimulator(rt_requests, config)
    result = sim.run()
    print(f"RT violation rate: {result.rt_violation_rate:.4f}")
    print(f"BE throughput:     {result.be_throughput_tok_s:.1f} tok/s")
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Sequence

from .config import ColocateConfig, ColocateResult


@dataclass(frozen=True)
class _RTRequest:
    """Internal RT request with arrival time and runtime."""

    arrival_time_s: float
    runtime_s: float


class ColocateSimulator:
    """Simulate RT-BE colocation on a single Prefill GPU."""

    def __init__(
        self,
        rt_requests: Sequence[_RTRequest],
        config: ColocateConfig,
    ) -> None:
        self._config = config
        # Keep only requests that arrive within the simulation window and
        # sort by arrival time for efficient event-driven processing.
        self._rt: List[_RTRequest] = sorted(
            [r for r in rt_requests if 0.0 <= r.arrival_time_s <= config.duration_s],
            key=lambda r: r.arrival_time_s,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ColocateResult:
        """Execute the simulation and return aggregate metrics."""
        result = ColocateResult(
            config=self._config,
            target_rps=self._estimate_rps(),
            rt_total=len(self._rt),
        )

        chunk_s = self._config.be_chunk_latency_s
        if chunk_s <= 0.0:
            return self._run_no_be(result)

        if self._config.policy == "conservative":
            return self._run_conservative(result, chunk_s)
        return self._run_greedy(result, chunk_s)

    # ------------------------------------------------------------------
    # Conservative policy (RT latency first)
    # ------------------------------------------------------------------

    def _run_conservative(
        self, result: ColocateResult, chunk_s: float
    ) -> ColocateResult:
        """Only start a BE chunk when it fits before the next RT arrival."""
        current_s = 0.0
        rt_idx = 0
        rt_queue: Deque[_RTRequest] = deque()
        ttft_s: List[float] = []
        queue_s_sum = 0.0

        while current_s < self._config.duration_s:
            rt_idx = self._enqueue(current_s, rt_idx, rt_queue)

            # ---- serve RT --------------------------------------------------
            if rt_queue:
                req = rt_queue.popleft()
                start_s = current_s
                q_s = start_s - req.arrival_time_s
                queue_s_sum += q_s
                ttft = q_s + req.runtime_s
                ttft_s.append(ttft)
                if ttft > self._config.ttft_slo_s:
                    result.rt_violations += 1
                result.rt_completed += 1
                result.rt_utilization += req.runtime_s
                current_s = start_s + req.runtime_s
                continue

            # ---- no RT waiting --------------------------------------------
            next_arrival = self._peek(rt_idx)
            if next_arrival is None:
                # No more RT in window — fill with BE chunks
                remaining = self._config.duration_s - current_s
                chunks = int(remaining // chunk_s)
                result.be_chunks_completed += chunks
                result.be_utilization += chunks * chunk_s
                result.idle_time_s += remaining - chunks * chunk_s
                current_s = self._config.duration_s
                break

            gap_s = next_arrival.arrival_time_s - current_s
            if gap_s >= chunk_s:
                # BE chunk fits before next arrival
                result.be_chunks_completed += 1
                result.be_utilization += chunk_s
                current_s += chunk_s
            else:
                # Gap too small for a BE chunk — idle until next arrival
                result.idle_time_s += gap_s
                current_s = next_arrival.arrival_time_s

        return self._finalize(result, ttft_s, queue_s_sum)

    # ------------------------------------------------------------------
    # Greedy policy (maximise BE throughput)
    # ------------------------------------------------------------------

    def _run_greedy(
        self, result: ColocateResult, chunk_s: float
    ) -> ColocateResult:
        """Start a BE chunk whenever RT queue is empty — non-preemptive.

        This is the policy that most clearly demonstrates the RT–BE
        tension: RT requests that arrive mid-chunk wait for the chunk
        to finish, so larger chunks directly increase RT tail latency.
        """
        current_s = 0.0
        rt_idx = 0
        rt_queue: Deque[_RTRequest] = deque()
        ttft_s: List[float] = []
        queue_s_sum = 0.0
        be_finish_s: float | None = None  # when the current BE chunk ends

        while current_s < self._config.duration_s:
            # ---- advance time to next interesting event --------------------
            next_arrival = self._peek(rt_idx)

            if be_finish_s is not None and current_s < be_finish_s:
                # A BE chunk is running.  Jump to the earlier of its
                # completion and the next RT arrival.
                if next_arrival is None:
                    current_s = min(be_finish_s, self._config.duration_s)
                else:
                    current_s = min(be_finish_s, next_arrival.arrival_time_s)

            # ---- enqueue arrivals at current time -------------------------
            rt_idx = self._enqueue(current_s, rt_idx, rt_queue)

            # ---- finish BE chunk if it completed --------------------------
            if be_finish_s is not None and current_s >= be_finish_s:
                result.be_chunks_completed += 1
                result.be_utilization += chunk_s
                be_finish_s = None

            # ---- serve RT (highest priority) ------------------------------
            # Only serve RT when no BE chunk is running (non-preemptive).
            if rt_queue and be_finish_s is None:
                req = rt_queue.popleft()
                start_s = current_s
                q_s = start_s - req.arrival_time_s
                queue_s_sum += q_s
                ttft = q_s + req.runtime_s
                ttft_s.append(ttft)
                if ttft > self._config.ttft_slo_s:
                    result.rt_violations += 1
                result.rt_completed += 1
                result.rt_utilization += req.runtime_s
                current_s = start_s + req.runtime_s
                continue

            # ---- no RT waiting, no BE running → consider BE chunk ---------
            if be_finish_s is not None:
                continue  # BE still running, wait for it

            # No more RT in window
            if next_arrival is None:
                remaining = self._config.duration_s - current_s
                chunks = int(remaining // chunk_s)
                result.be_chunks_completed += chunks
                result.be_utilization += chunks * chunk_s
                result.idle_time_s += remaining - chunks * chunk_s
                current_s = self._config.duration_s
                break

            # Check if a chunk fits in the remaining window
            if current_s + chunk_s > self._config.duration_s:
                result.idle_time_s += self._config.duration_s - current_s
                current_s = self._config.duration_s
                break

            # Greedy: always start a BE chunk when idle
            be_finish_s = current_s + chunk_s
            # Advance to the next event (arrival or chunk end)
            if next_arrival.arrival_time_s > current_s:
                current_s = min(be_finish_s, next_arrival.arrival_time_s)
            else:
                current_s = next_arrival.arrival_time_s

        # If a BE chunk was still running at duration_s, it got cut off
        # (partial chunks don't count).

        return self._finalize(result, ttft_s, queue_s_sum)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enqueue(
        self,
        current_s: float,
        rt_idx: int,
        queue: Deque[_RTRequest],
    ) -> int:
        """Push all RT requests with arrival ≤ *current_s* into *queue*."""
        while rt_idx < len(self._rt):
            if self._rt[rt_idx].arrival_time_s > current_s:
                break
            queue.append(self._rt[rt_idx])
            rt_idx += 1
        return rt_idx

    def _peek(self, rt_idx: int) -> _RTRequest | None:
        """Return the next RT request without consuming it, or None."""
        if rt_idx >= len(self._rt):
            return None
        return self._rt[rt_idx]

    def _estimate_rps(self) -> float:
        """Estimate the achieved RT injection rate."""
        if len(self._rt) < 2:
            return 0.0
        span = self._rt[-1].arrival_time_s - self._rt[0].arrival_time_s
        if span <= 0:
            return 0.0
        return len(self._rt) / span

    def _finalize(
        self,
        result: ColocateResult,
        ttft_s: List[float],
        queue_s_sum: float,
    ) -> ColocateResult:
        """Compute derived metrics from collected raw data."""
        # RT TTFT percentiles
        if ttft_s:
            sorted_ttft = sorted(ttft_s)
            n = len(sorted_ttft)
            result.rt_ttft_ms_p50 = sorted_ttft[n // 2] * 1000.0
            result.rt_ttft_ms_p95 = sorted_ttft[
                min(n - 1, int(math.ceil(0.95 * n)) - 1)
            ] * 1000.0
            result.rt_ttft_ms_p99 = sorted_ttft[
                min(n - 1, int(math.ceil(0.99 * n)) - 1)
            ] * 1000.0
            result.rt_mean_queue_s = queue_s_sum / n if n > 0 else 0.0

        # Violation rate
        result.rt_violation_rate = (
            result.rt_violations / result.rt_total
            if result.rt_total > 0
            else 0.0
        )

        # BE token metrics
        result.be_total_tokens = (
            result.be_chunks_completed * self._config.be_chunk_tokens
        )
        window_s = max(self._config.duration_s, 1e-9)
        result.be_throughput_tok_s = result.be_total_tokens / window_s

        # Normalise utilisation fractions
        result.rt_utilization = result.rt_utilization / window_s
        result.be_utilization = result.be_utilization / window_s

        return result

    # ------------------------------------------------------------------
    # Pure-RT baseline
    # ------------------------------------------------------------------

    def _run_no_be(self, result: ColocateResult) -> ColocateResult:
        """Run with zero BE chunk latency (pure RT, no BE work)."""
        current_s = 0.0
        rt_idx = 0
        rt_queue: Deque[_RTRequest] = deque()
        ttft_s: List[float] = []
        queue_s_sum = 0.0

        while current_s < self._config.duration_s:
            rt_idx = self._enqueue(current_s, rt_idx, rt_queue)

            if rt_queue:
                req = rt_queue.popleft()
                start_s = current_s
                q_s = start_s - req.arrival_time_s
                queue_s_sum += q_s
                ttft = q_s + req.runtime_s
                ttft_s.append(ttft)
                if ttft > self._config.ttft_slo_s:
                    result.rt_violations += 1
                result.rt_completed += 1
                result.rt_utilization += req.runtime_s
                current_s = start_s + req.runtime_s
                continue

            next_arrival = self._peek(rt_idx)
            if next_arrival is None:
                result.idle_time_s += self._config.duration_s - current_s
                current_s = self._config.duration_s
                break
            result.idle_time_s += next_arrival.arrival_time_s - current_s
            current_s = next_arrival.arrival_time_s

        return self._finalize(result, ttft_s, queue_s_sum)
