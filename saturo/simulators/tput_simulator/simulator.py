"""Simulators for BE-only T-pool throughput.

The steady-state path profiles candidate mixed BE batches. The legacy trace
path replays complete Prefill-to-Decode request lifecycles; split-trace mode
mirrors validation with independent Prefill-only and Decode-ready populations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

from saturo.predictors.time_predictor import TimePredictor

from .config import SimulatorConfig


@dataclass(frozen=True)
class TraceRequest:
    prompt_len: int
    output_len: int
    request_id: Optional[str] = None


@dataclass
class _TraceState:
    request: TraceRequest
    prefilled_tokens: int = 0
    decoded_tokens: int = 0
    started: bool = False

    @property
    def prompt_len(self) -> int:
        return max(int(self.request.prompt_len), 0)

    @property
    def output_len(self) -> int:
        return max(int(self.request.output_len), 0)

    @property
    def needs_prefill(self) -> bool:
        return self.prefilled_tokens < self.prompt_len

    @property
    def needs_decode(self) -> bool:
        return (
            self.prefilled_tokens >= self.prompt_len
            and self.decoded_tokens < self.output_len
        )

    @property
    def done(self) -> bool:
        return (
            self.prefilled_tokens >= self.prompt_len
            and self.decoded_tokens >= self.output_len
        )


@dataclass(frozen=True)
class CandidateResult:
    decode_batch_size: int
    chunk_size: int
    prefill_chunk_size: int
    iteration_latency_ms: float
    completed_iterations: int
    measured_time_s: float
    total_prefill_tokens: float
    total_decode_tokens: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    raw_tokens_per_s: float
    prefill_req_per_s: float
    decode_req_per_s: float
    be_saturation_req_per_s: float
    decode_blocks: int
    memory_feasible: bool = True

    def to_dict(self) -> Dict[str, float | int | bool]:
        return {
            "decode_batch_size": self.decode_batch_size,
            "chunk_size": self.chunk_size,
            "prefill_chunk_size": self.prefill_chunk_size,
            "iteration_latency_ms": round(self.iteration_latency_ms, 6),
            "completed_iterations": self.completed_iterations,
            "measured_time_s": round(self.measured_time_s, 6),
            "total_prefill_tokens": round(self.total_prefill_tokens, 6),
            "total_decode_tokens": round(self.total_decode_tokens, 6),
            "prefill_tokens_per_s": round(self.prefill_tokens_per_s, 6),
            "decode_tokens_per_s": round(self.decode_tokens_per_s, 6),
            "raw_tokens_per_s": round(self.raw_tokens_per_s, 6),
            "prefill_req_per_s": round(self.prefill_req_per_s, 9),
            "decode_req_per_s": round(self.decode_req_per_s, 9),
            "be_prefill_req_per_s": round(self.prefill_req_per_s, 9),
            "be_decode_req_per_s": round(self.decode_req_per_s, 9),
            "be_saturation_req_per_s": round(self.be_saturation_req_per_s, 9),
            "decode_blocks": self.decode_blocks,
            "memory_feasible": self.memory_feasible,
        }


@dataclass(frozen=True)
class TraceCandidateResult:
    max_active_sequences: int
    prefill_chunk_size: int
    total_token_budget: int
    completed_requests: int
    total_requests: int
    completed_iterations: int
    measured_time_s: float
    total_prefill_tokens: int
    total_decode_tokens: int
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    raw_tokens_per_s: float
    be_throughput_req_per_s: float
    avg_iteration_latency_ms: float
    p50_iteration_latency_ms: float
    p90_iteration_latency_ms: float
    peak_active_sequences: int
    peak_decode_batch_size: int
    peak_prefill_tokens: int
    peak_kv_blocks: int
    swap_out_count: int
    swap_in_count: int
    completed_all: bool
    memory_feasible: bool = True

    def to_dict(self) -> Dict[str, float | int | bool]:
        return {
            "max_active_sequences": self.max_active_sequences,
            "decode_batch_size": self.peak_decode_batch_size,
            "prefill_chunk_size": self.prefill_chunk_size,
            # Alias used by the real-engine chunk-size selector. In lifecycle
            # trace mode, the total shared budget is the vLLM chunk size.
            "chunk_size": self.total_token_budget,
            "total_token_budget": self.total_token_budget,
            "completed_requests": self.completed_requests,
            "total_requests": self.total_requests,
            "completed_iterations": self.completed_iterations,
            "measured_time_s": round(self.measured_time_s, 6),
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "prefill_tokens_per_s": round(self.prefill_tokens_per_s, 6),
            "decode_tokens_per_s": round(self.decode_tokens_per_s, 6),
            "raw_tokens_per_s": round(self.raw_tokens_per_s, 6),
            "be_throughput_req_per_s": round(
                self.be_throughput_req_per_s, 9),
            "be_saturation_req_per_s": round(
                self.be_throughput_req_per_s, 9),
            "avg_iteration_latency_ms": round(
                self.avg_iteration_latency_ms, 6),
            "p50_iteration_latency_ms": round(
                self.p50_iteration_latency_ms, 6),
            "p90_iteration_latency_ms": round(
                self.p90_iteration_latency_ms, 6),
            "peak_active_sequences": self.peak_active_sequences,
            "peak_decode_batch_size": self.peak_decode_batch_size,
            "peak_prefill_tokens": self.peak_prefill_tokens,
            "peak_kv_blocks": self.peak_kv_blocks,
            "swap_out_count": self.swap_out_count,
            "swap_in_count": self.swap_in_count,
            "completed_all": self.completed_all,
            "memory_feasible": self.memory_feasible,
        }


@dataclass(frozen=True)
class SplitTraceCandidateResult:
    """One T-pool candidate with independent, constantly refilled phases."""

    be_inflight: int
    prefill_inflight: int
    decode_inflight: int
    chunk_size: int
    total_token_budget: int
    completed_prefill_requests: int
    completed_decode_requests: int
    completed_iterations: int
    dummy_handoff_iterations: int
    measured_time_s: float
    total_prefill_tokens: int
    total_decode_tokens: int
    dummy_handoff_tokens: int
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    raw_tokens_per_s: float
    prefill_req_per_s: float
    decode_req_per_s: float
    be_throughput_req_per_s: float
    be_saturation_req_per_s: float
    avg_iteration_latency_ms: float
    p50_iteration_latency_ms: float
    p90_iteration_latency_ms: float
    peak_active_sequences: int
    peak_decode_batch_size: int
    peak_prefill_tokens: int
    peak_decode_kv_blocks: int
    peak_kv_blocks: int
    memory_feasible: bool = True

    def to_dict(self) -> Dict[str, float | int | bool]:
        return {
            "be_inflight": self.be_inflight,
            "prefill_inflight": self.prefill_inflight,
            "decode_inflight": self.decode_inflight,
            "chunk_size": self.chunk_size,
            "total_token_budget": self.total_token_budget,
            "completed_prefill_requests": self.completed_prefill_requests,
            "completed_decode_requests": self.completed_decode_requests,
            "completed_requests": (
                self.completed_prefill_requests
                + self.completed_decode_requests),
            "completed_iterations": self.completed_iterations,
            "dummy_handoff_iterations": self.dummy_handoff_iterations,
            "measured_time_s": round(self.measured_time_s, 6),
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "dummy_handoff_tokens": self.dummy_handoff_tokens,
            "prefill_tokens_per_s": round(self.prefill_tokens_per_s, 6),
            "decode_tokens_per_s": round(self.decode_tokens_per_s, 6),
            "raw_tokens_per_s": round(self.raw_tokens_per_s, 6),
            "prefill_req_per_s": round(self.prefill_req_per_s, 9),
            "decode_req_per_s": round(self.decode_req_per_s, 9),
            "be_prefill_req_per_s": round(self.prefill_req_per_s, 9),
            "be_decode_req_per_s": round(self.decode_req_per_s, 9),
            "be_throughput_req_per_s": round(
                self.be_throughput_req_per_s, 9),
            "be_saturation_req_per_s": round(
                self.be_saturation_req_per_s, 9),
            "avg_iteration_latency_ms": round(
                self.avg_iteration_latency_ms, 6),
            "p50_iteration_latency_ms": round(
                self.p50_iteration_latency_ms, 6),
            "p90_iteration_latency_ms": round(
                self.p90_iteration_latency_ms, 6),
            "peak_active_sequences": self.peak_active_sequences,
            "peak_decode_batch_size": self.peak_decode_batch_size,
            "peak_prefill_tokens": self.peak_prefill_tokens,
            "peak_decode_kv_blocks": self.peak_decode_kv_blocks,
            "peak_kv_blocks": self.peak_kv_blocks,
            "memory_feasible": self.memory_feasible,
        }


@dataclass(frozen=True)
class SimulationResult:
    best: CandidateResult | TraceCandidateResult | SplitTraceCandidateResult
    candidates: Tuple[
        CandidateResult | TraceCandidateResult | SplitTraceCandidateResult,
        ...,
    ] = field(
        default_factory=tuple)
    max_decode_batch_size_by_memory: int = 0
    avg_decode_context_len: int = 0
    mode: str = "steady_state"

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "best": self.best.to_dict(),
            "max_decode_batch_size_by_memory": self.max_decode_batch_size_by_memory,
            "avg_decode_context_len": self.avg_decode_context_len,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class ThroughputPoolSimulator:
    """Profile BE-only steady-state throughput for one T-pool instance."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.predictor = TimePredictor(
            config.latency_model_path,
        )

    def run(self) -> SimulationResult:
        decode_batch_size = self._max_decode_batch_size_by_memory()
        if decode_batch_size <= 0:
            raise ValueError(
                "no feasible T-pool decode batch; increase BE KV blocks or "
                "decrease average decode context length")

        candidates: List[CandidateResult] = []
        for chunk_size in self._steady_state_chunk_size_candidates():
            if int(chunk_size) - decode_batch_size <= 0:
                continue
            candidates.append(
                self._evaluate_candidate(decode_batch_size, chunk_size))

        if not candidates:
            raise ValueError(
                "no feasible T-pool candidates; every chunk_size candidate is "
                "<= decode_batch_size ({}); raise chunk_size candidates or "
                "reduce BE decode blocks".format(decode_batch_size))

        best = max(candidates, key=self._candidate_sort_key)
        return SimulationResult(
            best=best,
            candidates=tuple(candidates),
            max_decode_batch_size_by_memory=decode_batch_size,
            avg_decode_context_len=self.config.be.effective_decode_context_len(),
            mode="steady_state",
        )

    def run_trace(self, trace: List[TraceRequest]) -> SimulationResult:
        if not trace:
            raise ValueError("trace must contain at least one BE request")
        if int(self.config.be.be_inflight) <= 0:
            raise ValueError("trace mode requires positive be_inflight")
        if int(self.config.be.max_num_seqs) <= 0:
            raise ValueError("trace mode requires positive max_num_seqs")

        max_decode_batch = self._max_decode_batch_size_by_memory()
        if int(self.config.be.be_decode_blocks) <= 0:
            raise ValueError("trace mode requires positive BE KV-cache blocks")

        candidates: List[TraceCandidateResult] = []
        # Trace mode uses the configured prefill candidates as total
        # per-iteration token budgets. Decode tokens are scheduled first and
        # consume the same budget; prefill receives only the remainder.
        for total_token_budget in sorted(
                set(self.config.be.prefill_chunk_candidates)):
            total_token_budget = int(total_token_budget)
            if total_token_budget <= 0:
                continue
            candidates.append(
                self._evaluate_trace_candidate(trace, total_token_budget))

        if not candidates:
            raise ValueError(
                "no trace candidates; provide positive total token budgets")

        best = max(candidates, key=self._trace_candidate_sort_key)
        avg_decode_context_len = self._avg_trace_decode_context_len(trace)
        return SimulationResult(
            best=best,
            candidates=tuple(candidates),
            max_decode_batch_size_by_memory=max_decode_batch,
            avg_decode_context_len=avg_decode_context_len,
            mode="trace",
        )

    def run_split_trace(self, trace: List[TraceRequest]) -> SimulationResult:
        """Replay the T-pool benchmark's two independent BE populations.

        Unlike :meth:`run_trace`, a Prefill request never becomes a Decode
        request.  Each completed phase is immediately refilled from its own
        cyclic trace cursor, exactly like ``run_one_config`` in the real
        T-pool benchmark. Decode-ready requests also pay an isolated dummy-KV
        handoff iteration before joining normal mixed batches.
        """
        if not trace:
            raise ValueError("trace must contain at least one BE request")
        if int(self.config.be.be_inflight) <= 0:
            raise ValueError("split trace mode requires positive be_inflight")
        if int(self.config.be.max_num_seqs) <= 0:
            raise ValueError("split trace mode requires positive max_num_seqs")
        if (self.config.be.enable_be_decode
                and int(self.config.be.be_decode_blocks) <= 0):
            raise ValueError(
                "split trace mode requires positive BE decode KV blocks")

        candidates: List[SplitTraceCandidateResult] = []
        for total_token_budget in self._steady_state_chunk_size_candidates():
            candidates.append(self._evaluate_split_trace_candidate(
                trace, total_token_budget))
        if not candidates:
            raise ValueError(
                "no split trace candidates; provide positive chunk sizes")

        best = max(candidates, key=self._split_trace_candidate_sort_key)
        return SimulationResult(
            best=best,
            candidates=tuple(candidates),
            max_decode_batch_size_by_memory=max(
                candidate.peak_decode_batch_size for candidate in candidates),
            avg_decode_context_len=self._avg_trace_decode_context_len(trace),
            mode="split_trace",
        )

    def _evaluate_split_trace_candidate(
        self,
        trace: List[TraceRequest],
        total_token_budget: int,
    ) -> SplitTraceCandidateResult:
        be_inflight = max(int(self.config.be.be_inflight), 1)
        pp_size = self._pp_size()
        # PP "K staggered lanes" model: vLLM spreads the inflight population
        # across pp_size virtual-engine schedulers (min-cost assignment), so
        # one lane only ever batches ~1/pp of the requests. Simulate a single
        # lane with its share (KV blocks in the config are already per-lane,
        # matching Scheduler's total // pp split) and scale the resulting
        # rates by pp_size at the end. Measured lane overlap on TP4xPP2 is
        # near-perfect (K_eff 1.98-2.00), so no extra efficiency factor.
        lane_inflight = max(be_inflight // pp_size, 1)
        enable_decode = bool(self.config.be.enable_be_decode)
        decode_target = round(lane_inflight * 0.5) if enable_decode else 0
        prefill_target = lane_inflight - decode_target
        max_num_seqs = max(int(self.config.be.max_num_seqs), 1)
        decode_block_limit = max(int(self.config.be.be_decode_blocks), 0)
        total_block_capacity = int(self.config.be.total_gpu_blocks)
        if total_block_capacity <= 0:
            total_block_capacity = decode_block_limit
        total_block_capacity = max(total_block_capacity, 1)
        total_token_budget = max(int(total_token_budget), 1)

        prefill_cursor = 0
        decode_cursor = 0

        def new_prefill() -> _TraceState:
            nonlocal prefill_cursor
            request = trace[prefill_cursor % len(trace)]
            prefill_cursor += 1
            return _TraceState(request=request)

        def new_decode() -> _TraceState:
            nonlocal decode_cursor
            request = trace[decode_cursor % len(trace)]
            decode_cursor += 1
            return _TraceState(
                request=request,
                prefilled_tokens=max(int(request.prompt_len), 0),
            )

        prefill_waiting = [new_prefill() for _ in range(prefill_target)]
        prefill_resident: List[_TraceState] = []
        decode_waiting = [new_decode() for _ in range(decode_target)]
        decode_resident: List[_TraceState] = []

        elapsed_s = 0.0
        duration_s = max(float(self.config.duration_s), 0.0)
        warmup_s = max(float(self.config.warmup_s), 0.0)
        completed_prefill = 0
        completed_decode = 0
        measured_iterations = 0
        dummy_handoff_iterations = 0
        total_prefill_tokens = 0
        total_decode_tokens = 0
        dummy_handoff_tokens = 0
        latency_ms_values: List[float] = []
        peak_active = 0
        peak_decode_batch = 0
        peak_prefill_tokens = 0
        peak_decode_blocks = 0
        peak_kv_blocks = 0

        def prefill_blocks(state: _TraceState) -> int:
            if state.prefilled_tokens <= 0:
                return 0
            return self._trace_blocks_for_tokens(state.prefilled_tokens)

        def decode_blocks() -> int:
            return sum(self._trace_footprint(state)
                       for state in decode_resident)

        def all_blocks() -> int:
            return (decode_blocks()
                    + sum(prefill_blocks(state)
                          for state in prefill_resident))

        def record_iteration(
            latency_ms: float,
            step_prefill_tokens: int,
            step_decode_tokens: int,
            step_handoff_tokens: int = 0,
        ) -> bool:
            """Advance time and return whether this step is measured."""
            nonlocal elapsed_s
            latency_s = max(float(latency_ms) / 1000.0, 1e-12)
            elapsed_s += latency_s
            return warmup_s <= elapsed_s < duration_s

        while elapsed_s < duration_s:
            # DummyConnector admission is an isolated pure-prefill scheduler
            # step. It allocates full prompt KV but bypasses model forward.
            admitted: List[_TraceState] = []
            projected_decode_blocks = decode_blocks()
            projected_total_blocks = all_blocks()
            while decode_waiting:
                if (len(decode_resident) + len(prefill_resident)
                        + len(admitted) >= max_num_seqs):
                    break
                state = decode_waiting[0]
                needed = self._trace_footprint(state)
                # Match Scheduler._schedule_p_pool_dummy_decode_handoff: the
                # first decode-ready request may enter by itself even when its
                # prompt is larger than the configured share; the limit gates
                # adding another request to an existing/admissible batch.
                if ((decode_resident or admitted)
                        and projected_decode_blocks + needed
                        > decode_block_limit):
                    break
                if projected_total_blocks + needed > total_block_capacity:
                    break
                admitted.append(decode_waiting.pop(0))
                projected_decode_blocks += needed
                projected_total_blocks += needed

            if admitted:
                decode_resident.extend(admitted)
                handoff_tokens = sum(state.prompt_len for state in admitted)
                latency_ms = max(float(self.config.min_iteration_ms), 1e-9)
                measured = record_iteration(
                    latency_ms, 0, 0, handoff_tokens)
                if measured:
                    measured_iterations += 1
                    dummy_handoff_iterations += 1
                    dummy_handoff_tokens += handoff_tokens
                    latency_ms_values.append(latency_ms)
                peak_active = max(
                    peak_active,
                    len(decode_resident) + len(prefill_resident),
                )
                peak_decode_blocks = max(
                    peak_decode_blocks, decode_blocks())
                peak_kv_blocks = max(peak_kv_blocks, all_blocks())
                continue

            decode_reqs: List[Tuple[int, int]] = []
            step_decode_tokens = 0
            for state in decode_resident:
                if step_decode_tokens >= total_token_budget:
                    break
                if not state.needs_decode:
                    continue
                old_blocks = self._trace_footprint(state)
                new_blocks = self._trace_blocks_for_tokens(
                    state.prompt_len + state.decoded_tokens + 1)
                if all_blocks() + new_blocks - old_blocks > total_block_capacity:
                    continue
                context_len = max(1, state.prompt_len + state.decoded_tokens)
                decode_reqs.append((1, context_len))
                state.decoded_tokens += 1
                step_decode_tokens += 1

            prefill_reqs: List[Tuple[int, int]] = []
            step_prefill_tokens = 0
            remaining_budget = total_token_budget - step_decode_tokens

            # The real chunked-prefill scheduler advances resident partial
            # prefills before admitting more waiting requests.
            while remaining_budget > 0:
                schedulable = next(
                    (state for state in prefill_resident
                     if state.needs_prefill),
                    None,
                )
                if schedulable is None:
                    if not prefill_waiting:
                        break
                    if (len(decode_resident) + len(prefill_resident)
                            >= max_num_seqs):
                        break
                    schedulable = prefill_waiting.pop(0)
                    prefill_resident.append(schedulable)

                current_blocks = prefill_blocks(schedulable)
                free_blocks = max(total_block_capacity - all_blocks(), 0)
                max_position = (
                    (current_blocks + free_blocks)
                    * max(int(self.config.model.block_size), 1)
                )
                tokens_fitting_memory = max(
                    max_position - schedulable.prefilled_tokens, 0)
                chunk = min(
                    schedulable.prompt_len - schedulable.prefilled_tokens,
                    remaining_budget,
                    tokens_fitting_memory,
                )
                if chunk <= 0:
                    break
                context_len = schedulable.prefilled_tokens + chunk
                prefill_reqs.append((chunk, max(context_len, chunk)))
                schedulable.prefilled_tokens += chunk
                remaining_budget -= chunk
                step_prefill_tokens += chunk

            if not prefill_reqs and not decode_reqs:
                break

            latency_ms = self._predict_forward_ms(prefill_reqs, decode_reqs)
            measured = record_iteration(
                latency_ms, step_prefill_tokens, step_decode_tokens)

            finished_prefill = [
                state for state in prefill_resident
                if not state.needs_prefill
            ]
            finished_decode = [
                state for state in decode_resident if state.done
            ]
            prefill_resident = [
                state for state in prefill_resident if state.needs_prefill
            ]
            decode_resident = [
                state for state in decode_resident if not state.done
            ]
            prefill_waiting.extend(
                new_prefill() for _ in finished_prefill)
            decode_waiting.extend(new_decode() for _ in finished_decode)

            if measured:
                measured_iterations += 1
                completed_prefill += len(finished_prefill)
                completed_decode += len(finished_decode)
                total_prefill_tokens += step_prefill_tokens
                total_decode_tokens += step_decode_tokens
                latency_ms_values.append(latency_ms)

            peak_active = max(
                peak_active,
                len(decode_resident) + len(prefill_resident),
            )
            peak_decode_batch = max(
                peak_decode_batch, step_decode_tokens)
            peak_prefill_tokens = max(
                peak_prefill_tokens, step_prefill_tokens)
            peak_decode_blocks = max(peak_decode_blocks, decode_blocks())
            peak_kv_blocks = max(peak_kv_blocks, all_blocks())

        measured_time_s = max(min(elapsed_s, duration_s) - warmup_s, 1e-12)
        # Scale the single simulated lane up to the whole PP engine: pp_size
        # lanes run staggered and (measured) fully overlapped, so system
        # totals and rates are pp_size x the lane values. Peaks and memory
        # feasibility stay per-lane, matching each scheduler's KV budget.
        completed_prefill *= pp_size
        completed_decode *= pp_size
        measured_iterations *= pp_size
        dummy_handoff_iterations *= pp_size
        total_prefill_tokens *= pp_size
        total_decode_tokens *= pp_size
        dummy_handoff_tokens *= pp_size
        prefill_tps = total_prefill_tokens / measured_time_s
        decode_tps = total_decode_tokens / measured_time_s
        avg_input = max(float(self.config.be.avg_input_len), 1.0)
        avg_output = max(float(self.config.be.avg_output_len), 1.0)
        prefill_req_s = prefill_tps / avg_input
        decode_req_s = decode_tps / avg_output
        saturation_req_s = (
            min(prefill_req_s, decode_req_s)
            if decode_target > 0 else prefill_req_s
        )

        return SplitTraceCandidateResult(
            be_inflight=be_inflight,
            prefill_inflight=prefill_target,
            decode_inflight=decode_target,
            chunk_size=total_token_budget,
            total_token_budget=total_token_budget,
            completed_prefill_requests=completed_prefill,
            completed_decode_requests=completed_decode,
            completed_iterations=measured_iterations,
            dummy_handoff_iterations=dummy_handoff_iterations,
            measured_time_s=measured_time_s,
            total_prefill_tokens=total_prefill_tokens,
            total_decode_tokens=total_decode_tokens,
            dummy_handoff_tokens=dummy_handoff_tokens,
            prefill_tokens_per_s=prefill_tps,
            decode_tokens_per_s=decode_tps,
            raw_tokens_per_s=(
                total_prefill_tokens + total_decode_tokens) / measured_time_s,
            prefill_req_per_s=prefill_req_s,
            decode_req_per_s=decode_req_s,
            be_throughput_req_per_s=(
                completed_prefill + completed_decode) / measured_time_s,
            be_saturation_req_per_s=saturation_req_s,
            avg_iteration_latency_ms=self._mean(latency_ms_values),
            p50_iteration_latency_ms=self._percentile(
                latency_ms_values, 0.50),
            p90_iteration_latency_ms=self._percentile(
                latency_ms_values, 0.90),
            peak_active_sequences=peak_active,
            peak_decode_batch_size=peak_decode_batch,
            peak_prefill_tokens=peak_prefill_tokens,
            peak_decode_kv_blocks=peak_decode_blocks,
            peak_kv_blocks=peak_kv_blocks,
            memory_feasible=peak_kv_blocks <= total_block_capacity,
        )

    def _evaluate_candidate(
        self,
        decode_batch_size: int,
        chunk_size: int,
    ) -> CandidateResult:
        # chunk_size is the total per-iteration token budget
        # (max_num_batched_tokens). Each resident decode request occupies one
        # token slot, so the prefill budget is what remains after decode.
        prefill_chunk_size = max(int(chunk_size) - int(decode_batch_size), 0)
        latency_ms = self._predict_iteration_ms(
            decode_batch_size,
            prefill_chunk_size,
        )
        latency_s = max(latency_ms / 1000.0, 1e-12)
        measured_time_s = max(
            float(self.config.duration_s) - float(self.config.warmup_s),
            0.0,
        )
        # One simulated lane; a PP=K engine completes K staggered lane-steps
        # per latency window (see SimulatorConfig.pipeline_parallel_size).
        pp_size = self._pp_size()
        lane_iterations = int(math.floor(measured_time_s / latency_s))
        effective_time_s = max(lane_iterations * latency_s, 1e-12)
        completed_iterations = lane_iterations * pp_size

        total_prefill_tokens = completed_iterations * prefill_chunk_size
        total_decode_tokens = completed_iterations * decode_batch_size
        prefill_tokens_per_s = total_prefill_tokens / effective_time_s
        decode_tokens_per_s = total_decode_tokens / effective_time_s
        raw_tokens_per_s = (
            total_prefill_tokens + total_decode_tokens) / effective_time_s

        avg_input = max(float(self.config.be.avg_input_len), 1.0)
        avg_output = max(float(self.config.be.avg_output_len), 1.0)
        prefill_req_per_s = prefill_tokens_per_s / avg_input
        decode_req_per_s = decode_tokens_per_s / avg_output
        be_saturation_req_per_s = min(prefill_req_per_s, decode_req_per_s)

        return CandidateResult(
            decode_batch_size=decode_batch_size,
            chunk_size=int(chunk_size),
            prefill_chunk_size=prefill_chunk_size,
            iteration_latency_ms=latency_ms,
            completed_iterations=completed_iterations,
            measured_time_s=effective_time_s,
            total_prefill_tokens=total_prefill_tokens,
            total_decode_tokens=total_decode_tokens,
            prefill_tokens_per_s=prefill_tokens_per_s,
            decode_tokens_per_s=decode_tokens_per_s,
            raw_tokens_per_s=raw_tokens_per_s,
            prefill_req_per_s=prefill_req_per_s,
            decode_req_per_s=decode_req_per_s,
            be_saturation_req_per_s=be_saturation_req_per_s,
            decode_blocks=self._decode_blocks(decode_batch_size),
        )

    def _predict_iteration_ms(
        self,
        decode_batch_size: int,
        prefill_chunk_size: int,
    ) -> float:
        decode_context = self.config.be.effective_decode_context_len()
        decode_reqs = [(1, decode_context)] * decode_batch_size
        prefill_reqs = self._build_prefill_reqs(prefill_chunk_size)
        raw_ms = self.predictor.predict_forward_ms(prefill_reqs, decode_reqs)
        raw_ms += self._pp_comm_ms(
            int(decode_batch_size) + int(prefill_chunk_size))
        return max(float(self.config.min_iteration_ms), raw_ms)

    def _steady_state_chunk_size_candidates(self) -> List[int]:
        return sorted({
            int(chunk)
            for chunk in self.config.be.chunk_size_candidates
            if int(chunk) > 0
        })

    def _evaluate_trace_candidate(
        self,
        trace: List[TraceRequest],
        total_token_budget: int,
    ) -> TraceCandidateResult:
        states = [_TraceState(request=req) for req in trace]
        total_token_budget = int(total_token_budget)
        kv_block_capacity = max(int(self.config.be.be_decode_blocks), 0)
        next_request = 0
        resident: List[_TraceState] = []
        swapped: List[_TraceState] = []
        completed = 0
        measured_completed = 0
        elapsed_s = 0.0
        iterations = 0
        total_prefill_tokens = 0
        total_decode_tokens = 0
        used_kv_blocks = 0
        peak_active = 0
        peak_decode = 0
        peak_prefill_tokens = 0
        peak_kv_blocks = 0
        swap_out_count = 0
        swap_in_count = 0
        latency_ms_values: List[float] = []
        duration_limit_s = max(float(self.config.duration_s), 0.0)
        warmup_s = max(float(self.config.warmup_s), 0.0)
        active_sequence_limit = min(
            int(self.config.be.be_inflight),
            int(self.config.be.max_num_seqs),
        )

        while completed < len(states):
            # Reclaim blocks from requests that finished the previous iteration.
            done_resident = [state for state in resident if state.done]
            used_kv_blocks -= sum(
                self._trace_footprint(state) for state in done_resident)
            resident = [state for state in resident if not state.done]

            # Swap previously evicted requests back in (shortest-remaining first).
            for state in sorted(
                list(swapped),
                key=lambda item: (
                    self._trace_remaining_decode_tokens(item),
                    item.prompt_len,
                    item.request.request_id or "",
                ),
            ):
                needed_blocks = self._trace_footprint(state)
                free_blocks = max(
                    0, kv_block_capacity - used_kv_blocks)
                if needed_blocks > free_blocks:
                    continue
                self._trace_remove_state(swapped, state)
                resident.append(state)
                used_kv_blocks += needed_blocks
                swap_in_count += 1

            # Admit new requests, reserving each prompt's FULL KV-block
            # footprint up front. A running vLLM sequence holds its prompt
            # blocks until it finishes, so a request whose prompt cannot fit is
            # not admitted. Reserving here (instead of lazily during prefill)
            # fixes the over-admission bug where an un-prefilled request counted
            # as a single block, letting ~kv_block_capacity requests in at once
            # and starving the prefill budget so nothing ever completed.
            while next_request < len(states):
                if len(resident) + len(swapped) >= active_sequence_limit:
                    break
                state = states[next_request]
                needed_blocks = self._trace_blocks_for_tokens(state.prompt_len)
                if needed_blocks > max(0, kv_block_capacity - used_kv_blocks):
                    break
                state.started = True
                resident.append(state)
                used_kv_blocks += needed_blocks
                next_request += 1

            if not resident:
                if next_request >= len(states) and not swapped:
                    break
                continue

            # Decode scheduling: one decoded token may grow a request's
            # footprint by a block when it crosses a block boundary.
            decode_reqs: List[Tuple[int, int]] = []
            step_decode_tokens = 0
            for state in list(resident):
                if step_decode_tokens >= total_token_budget:
                    break
                if not self._trace_is_resident(state, resident):
                    continue
                if not state.needs_decode:
                    continue
                extra_blocks = (
                    self._trace_blocks_for_tokens(
                        state.prompt_len + state.decoded_tokens + 1)
                    - self._trace_footprint(state)
                )
                while extra_blocks > max(
                        0, kv_block_capacity - used_kv_blocks):
                    victim = self._trace_swap_out_shortest_decode(
                        resident,
                        swapped,
                    )
                    if victim is None:
                        break
                    used_kv_blocks -= self._trace_footprint(victim)
                    swap_out_count += 1
                    if victim is state:
                        break
                if not self._trace_is_resident(state, resident):
                    continue
                if extra_blocks > max(
                        0, kv_block_capacity - used_kv_blocks):
                    continue
                context_len = max(1, state.prompt_len + state.decoded_tokens)
                decode_reqs.append((1, context_len))
                state.decoded_tokens += 1
                used_kv_blocks += extra_blocks
                step_decode_tokens += 1

            # Prefill scheduling: prompt blocks were reserved at admission, so
            # advancing prefill only consumes the shared token budget.
            prefill_reqs: List[Tuple[int, int]] = []
            step_prefill_tokens = 0
            prefill_token_budget = max(
                total_token_budget - step_decode_tokens,
                0,
            )
            for state in list(resident):
                if prefill_token_budget <= 0:
                    break
                if not state.needs_prefill:
                    continue
                chunk = min(
                    state.prompt_len - state.prefilled_tokens,
                    prefill_token_budget,
                )
                if chunk <= 0:
                    continue
                context_len = state.prefilled_tokens + chunk
                prefill_reqs.append((chunk, max(chunk, context_len)))
                state.prefilled_tokens += chunk
                prefill_token_budget -= chunk
                step_prefill_tokens += chunk

            if not prefill_reqs and not decode_reqs:
                break

            latency_ms = self._predict_forward_ms(prefill_reqs, decode_reqs)
            latency_s = max(latency_ms / 1000.0, 1e-12)
            if (
                duration_limit_s > 0
                and elapsed_s >= duration_limit_s
            ):
                break

            elapsed_s += latency_s
            in_measurement = (
                elapsed_s >= warmup_s
                and (
                    duration_limit_s <= 0
                    or elapsed_s < duration_limit_s
                )
            )
            if in_measurement:
                iterations += 1
                latency_ms_values.append(latency_ms)
                total_prefill_tokens += step_prefill_tokens
                total_decode_tokens += step_decode_tokens

            newly_done = sum(1 for state in resident if state.done)
            completed += newly_done
            if in_measurement:
                measured_completed += newly_done

            peak_active = max(peak_active, len(resident))
            peak_decode = max(peak_decode, step_decode_tokens)
            peak_prefill_tokens = max(peak_prefill_tokens, step_prefill_tokens)
            peak_kv_blocks = max(
                peak_kv_blocks,
                used_kv_blocks,
            )
            done_resident = [state for state in resident if state.done]
            used_kv_blocks -= sum(
                self._trace_footprint(state) for state in done_resident)
            resident = [state for state in resident if not state.done]

        measured_time_s = max(
            min(elapsed_s, duration_limit_s) - warmup_s
            if duration_limit_s > 0
            else elapsed_s - warmup_s,
            1e-12,
        )
        completed_requests = measured_completed
        prefill_tps = total_prefill_tokens / measured_time_s
        decode_tps = total_decode_tokens / measured_time_s
        raw_tps = (total_prefill_tokens + total_decode_tokens) / measured_time_s
        req_tps = completed_requests / measured_time_s

        return TraceCandidateResult(
            max_active_sequences=peak_active,
            # Kept for output compatibility. In trace mode this is the
            # configured maximum prefill budget (when there is no decode);
            # peak_prefill_tokens reports the maximum actually consumed.
            prefill_chunk_size=total_token_budget,
            total_token_budget=total_token_budget,
            completed_requests=completed_requests,
            total_requests=len(states),
            completed_iterations=iterations,
            measured_time_s=measured_time_s,
            total_prefill_tokens=total_prefill_tokens,
            total_decode_tokens=total_decode_tokens,
            prefill_tokens_per_s=prefill_tps,
            decode_tokens_per_s=decode_tps,
            raw_tokens_per_s=raw_tps,
            be_throughput_req_per_s=req_tps,
            avg_iteration_latency_ms=self._mean(latency_ms_values),
            p50_iteration_latency_ms=self._percentile(latency_ms_values, 0.50),
            p90_iteration_latency_ms=self._percentile(latency_ms_values, 0.90),
            peak_active_sequences=peak_active,
            peak_decode_batch_size=peak_decode,
            peak_prefill_tokens=peak_prefill_tokens,
            peak_kv_blocks=peak_kv_blocks,
            swap_out_count=swap_out_count,
            swap_in_count=swap_in_count,
            completed_all=completed == len(states),
            memory_feasible=(
                peak_kv_blocks
                <= kv_block_capacity
            ),
        )

    def _pp_size(self) -> int:
        """Pipeline-parallel degree (number of virtual-engine lanes)."""
        return max(int(getattr(self.config, "pipeline_parallel_size", 1)), 1)

    def _pp_comm_ms(self, step_tokens: int) -> float:
        """Non-overlapped cross-stage transfer cost for one iteration.

        The vLLM V0 PP handoff is a blocking ``torch.distributed.send`` of
        the hidden states for every token in the step (plus a blocking gloo
        metadata send), so the cost scales with the step token count and is
        paid once per hop (pp_size - 1 hops).
        """
        pp_size = self._pp_size()
        if pp_size <= 1:
            return 0.0
        per_hop = (
            float(getattr(self.config, "pp_comm_ms_per_token", 0.0))
            * float(max(step_tokens, 0))
            + float(getattr(self.config, "pp_comm_base_ms", 0.0))
        )
        return float(pp_size - 1) * max(per_hop, 0.0)

    def _predict_forward_ms(
        self,
        prefill_reqs: List[Tuple[int, int]],
        decode_reqs: List[Tuple[int, int]],
    ) -> float:
        raw_ms = self.predictor.predict_forward_ms(prefill_reqs, decode_reqs)
        step_tokens = (sum(q for q, _ in prefill_reqs)
                       + sum(q for q, _ in decode_reqs))
        raw_ms += self._pp_comm_ms(step_tokens)
        return max(float(self.config.min_iteration_ms), raw_ms)

    def _build_prefill_reqs(
        self,
        total_prefill_tokens: int,
    ) -> List[Tuple[int, int]]:
        remaining = max(int(total_prefill_tokens), 0)
        avg_input = max(int(self.config.be.avg_input_len), 1)
        reqs: List[Tuple[int, int]] = []
        while remaining > 0:
            chunk = min(remaining, avg_input)
            context_len = self._prefill_context_len(chunk)
            reqs.append((chunk, context_len))
            remaining -= chunk
        return reqs

    def _prefill_context_len(self, chunk_size: int) -> int:
        avg_input = max(int(self.config.be.avg_input_len), 1)
        chunk = max(int(chunk_size), 1)
        policy = self.config.be.prefill_context_policy.lower()
        if policy == "chunk":
            return chunk
        if policy == "full":
            return max(chunk, avg_input)
        if policy != "avg":
            raise ValueError(
                "prefill_context_policy must be one of: avg, chunk, full")
        cached = max(0.0, (avg_input - chunk) / 2.0)
        return max(chunk, int(round(cached + chunk)))

    def _max_decode_batch_size_by_memory(self) -> int:
        """Max concurrent decode requests that fit in available KV-cache blocks."""
        blocks_per_req = self._blocks_per_request()
        if blocks_per_req <= 0:
            return 0
        total_blocks = int(self.config.be.be_decode_blocks)
        return max(0, total_blocks // blocks_per_req)

    def _blocks_per_request(self) -> int:
        """KV-cache blocks occupied by one average decode request."""
        context_len = self.config.be.effective_decode_context_len()
        block_size = max(int(self.config.model.block_size), 1)
        return int(math.ceil(context_len / block_size))

    def _decode_blocks(self, decode_batch_size: int) -> int:
        return int(decode_batch_size) * self._blocks_per_request()

    def _trace_blocks_for_tokens(self, num_tokens: int) -> int:
        block_size = max(int(self.config.model.block_size), 1)
        return int(math.ceil(max(int(num_tokens), 1) / block_size))

    def _trace_footprint(self, state: _TraceState) -> int:
        """KV blocks a resident request holds: full prompt + decoded tokens.

        A started request reserves its whole prompt at admission and grows by
        one block per block-boundary crossing as it decodes.
        """
        return self._trace_blocks_for_tokens(
            state.prompt_len + state.decoded_tokens)

    @staticmethod
    def _trace_remaining_decode_tokens(state: _TraceState) -> int:
        return max(0, state.output_len - state.decoded_tokens)

    def _trace_swap_out_shortest_decode(
        self,
        resident: List[_TraceState],
        swapped: List[_TraceState],
    ) -> Optional[_TraceState]:
        candidates = [state for state in resident if state.needs_decode]
        if not candidates:
            return None
        victim = min(
            candidates,
            key=lambda state: (
                self._trace_remaining_decode_tokens(state),
                state.prompt_len,
                state.request.request_id or "",
            ),
        )
        self._trace_remove_state(resident, victim)
        swapped.append(victim)
        return victim

    @staticmethod
    def _trace_is_resident(
        state: _TraceState,
        resident: List[_TraceState],
    ) -> bool:
        return any(item is state for item in resident)

    @staticmethod
    def _trace_remove_state(
        states: List[_TraceState],
        target: _TraceState,
    ) -> None:
        for idx, state in enumerate(states):
            if state is target:
                del states[idx]
                return

    @staticmethod
    def _avg_trace_decode_context_len(trace: List[TraceRequest]) -> int:
        if not trace:
            return 0
        values = [
            max(1, int(req.prompt_len + req.output_len / 2))
            for req in trace
        ]
        return int(round(sum(values) / len(values)))

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = int(round(max(0.0, min(1.0, q)) * (len(ordered) - 1)))
        return float(ordered[idx])

    @staticmethod
    def _candidate_sort_key(candidate: CandidateResult) -> Tuple[float, float, int]:
        return (
            candidate.be_saturation_req_per_s,
            candidate.raw_tokens_per_s,
            -candidate.prefill_chunk_size,
        )

    @staticmethod
    def _trace_candidate_sort_key(
        candidate: TraceCandidateResult,
    ) -> Tuple[float, int, float, int]:
        return (
            candidate.be_throughput_req_per_s,
            int(candidate.completed_all),
            candidate.raw_tokens_per_s,
            -candidate.prefill_chunk_size,
        )

    @staticmethod
    def _split_trace_candidate_sort_key(
        candidate: SplitTraceCandidateResult,
    ) -> Tuple[float, float, int]:
        return (
            candidate.be_saturation_req_per_s,
            candidate.raw_tokens_per_s,
            -candidate.chunk_size,
        )
