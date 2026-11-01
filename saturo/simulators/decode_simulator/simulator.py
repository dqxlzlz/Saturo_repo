"""Trace-driven CPU simulator for D-pool decode saturation.

This simulator intentionally keeps only the behavior needed for throughput and
RT SLO studies:

* admit arrived RT decode requests while RT/total KV capacity allows;
* choose a target TBT from the RT-only lookahead constraints;
* use remaining BE memory and latency slack for active BE decode and BE prefill;
* advance simulated request state and count BE prefill/decode tokens.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, List, Optional, Set

import numpy as np

from saturo.saturation import (
    DecodeSaturationDecision,
    DecodeSaturationEngine,
    RequestForecast,
    SaturationParams,
)

from .config import SimulatorConfig
from .predictors.latency import LatencyPredictor
from .predictors.memory import DPoolSaturationPredictor
from .request import Request, RequestState, RequestType
from saturo.predictors.time_predictor import TimePredictor

_NATIVE_LATENCY_FORWARD = LatencyPredictor.predict_iteration_ms
_NATIVE_LATENCY_AGGREGATE = LatencyPredictor.predict_iteration_from_aggregates_ms
_NATIVE_TIME_FORWARD = TimePredictor.predict_forward_ms

logger = logging.getLogger(__name__)


class DecodeStrategy(Enum):
    PROGRESSIVE = "progressive"
    ORACLE = "oracle"
    GREEDY = "greedy"


@dataclass
class IterationStats:
    iteration_id: int
    time_s: float

    num_rt_decode: int = 0
    num_rt_waiting: int = 0
    num_be_decode: int = 0
    be_prefill_chunk_size: int = 0

    rt_blocks: int = 0
    be_blocks: int = 0
    predicted_peak_rt_blocks: int = 0
    be_budget_blocks: int = 0

    safe_tbt_ms: float = 0.0
    target_tbt_ms: float = 0.0
    rt_decode_latency_ms: float = 0.0
    actual_latency_ms: float = 0.0

    rt_admission_violations: int = 0
    rt_memory_allocation_failures: int = 0
    rt_memory_overflow: bool = False
    rt_tbt_violations: int = 0
    slo_violated: bool = False
    infeasible: bool = False

    # Host-side simulator profiling. These are wall-clock costs, not modeled
    # GPU latencies. Keeping them on each iteration makes pathological ranges
    # visible directly in the result JSON.
    wall_enqueue_ms: float = 0.0
    wall_admit_ms: float = 0.0
    wall_decision_ms: float = 0.0
    wall_iteration_ms: float = 0.0


@dataclass
class SimulationResult:
    strategy: str = ""
    total_rt_requests: int = 0
    completed_rt_requests: int = 0
    total_be_completed: int = 0
    total_be_prefill_tokens: int = 0
    total_be_decode_tokens: int = 0
    total_iterations: int = 0

    rt_tbt_violations: int = 0
    rt_tbt_total_samples: int = 0
    rt_tbt_violation_rate: float = 0.0
    rt_tbt_p50_ms: float = 0.0
    rt_tbt_p99_ms: float = 0.0
    rt_tbt_max_ms: float = 0.0

    rt_admission_violations: int = 0
    rt_memory_overflow_events: int = 0
    rt_slo_violations_total: int = 0
    infeasible_iterations: int = 0
    be_swap_out_events: int = 0
    be_swap_in_events: int = 0

    be_saturation_req_per_s: float = 0.0
    be_prefill_req_per_s: float = 0.0
    be_decode_req_per_s: float = 0.0
    be_throughput_tokens_per_s: float = 0.0
    be_prefill_throughput_tokens_per_s: float = 0.0
    be_decode_throughput_tokens_per_s: float = 0.0
    avg_saturation_ratio: float = 0.0
    avg_be_prefill_chunk: float = 0.0
    avg_rt_batch_size: float = 0.0
    avg_rt_queue_size: float = 0.0
    avg_be_batch_size: float = 0.0
    avg_iteration_latency_ms: float = 0.0
    avg_safe_tbt_ms: float = 0.0
    avg_target_tbt_ms: float = 0.0
    p50_target_tbt_ms: float = 0.0
    p99_target_tbt_ms: float = 0.0
    peak_rt_blocks: int = 0
    avg_rt_blocks: float = 0.0
    avg_be_blocks: float = 0.0

    wall_run_s: float = 0.0
    wall_enqueue_s: float = 0.0
    wall_admit_s: float = 0.0
    wall_decision_s: float = 0.0
    wall_iteration_s: float = 0.0
    terminated_early: bool = False
    termination_reason: Optional[str] = None
    # True when the run hit the terminal RT KV stall: even with every
    # swappable BE reclaimed, RT demand exceeds the usable KV capacity.
    # The whole (config, RPS) operating point is judged infeasible.
    rps_infeasible: bool = False

    iteration_stats: List[IterationStats] = field(default_factory=list)


@dataclass(frozen=True)
class _BEBalanceEvaluation:
    """Dry-run result for one candidate BE-decode batch size."""

    decode_count: int
    prefill_tokens: int
    predicted_latency_ms: float
    decode_req_per_s: float
    prefill_req_per_s: float
    saturation_req_per_s: float
    feasible: bool


# _RTForecast and DecodeSaturationDecision are now imported from
# saturo.saturation as RequestForecast and DecodeSaturationDecision.


class DecodeSimulator:
    """Single D-pool instance simulator."""

    def __init__(
        self,
        config: SimulatorConfig,
        strategy: DecodeStrategy = DecodeStrategy.PROGRESSIVE,
    ):
        self.config = config
        self.model_config = config.model
        self._be_request_pool = self._load_be_request_pool()
        self.sat_config = config.saturation
        self.slo_config = config.slo
        self.strategy = strategy

        configured_total = getattr(self.config, "total_gpu_blocks", None)
        if configured_total:
            self.total_gpu_blocks = int(configured_total)
        else:
            # Fallback for standalone/debug runs: estimate from a reasonable
            # per-GPU KV budget. The real collector always sets this from the
            # TimePredictor profile.
            self.total_gpu_blocks = 1024 * 20  # ~20k blocks ≈ 40 GB

        # KV-cache watermark: RT+BE allocation never exceeds this. BE
        # saturates all free blocks up to this cap; RT admission swaps BE out
        # when it cannot fit under this cap.
        watermark = float(getattr(self.config, "memory_watermark", 0.9))
        self.usable_gpu_blocks = max(
            int(self.total_gpu_blocks * watermark), 1)

        self.latency_predictor = LatencyPredictor(
            config.latency_model_path,
        )
        self.saturation_predictor = DPoolSaturationPredictor(
            self.model_config, self.sat_config,
            total_gpu_blocks=self.total_gpu_blocks)

        # ── 预测器初始化：优先走 registry（dataset + parquet），兼容旧的裸路径 ──
        if config.predictor_dataset and config.predictor_parquet:
            from saturo.predictors.registry import load_or_train
            predictor = load_or_train(
                dataset_name=config.predictor_dataset,
                parquet_path=config.predictor_parquet,
                model_type=config.predictor_type,
                feature_mode=config.predictor_feature_mode,
                max_samples=config.predictor_max_samples,
                max_steps_per_request=config.predictor_max_steps,
                verbose=True,
            )
            self.saturation_predictor.configure_model_len_predictor(
                None,  # 不走路径，直接注入预测器实例
                config.predictor_type,
                predictor=predictor,
            )
        elif config.model_len_predictor_path:
            self.saturation_predictor.configure_model_len_predictor(
                config.model_len_predictor_path,
                config.model_len_predictor_type,
            )
        elif self.strategy == DecodeStrategy.PROGRESSIVE:
            # Saturo never reads the trace's true remaining output as a
            # substitute for QRF. Explicit oracle/greedy baselines are separate.
            self.saturation_predictor.configure_model_len_predictor(None)
        self.memory_predictor = self.saturation_predictor

        self.rt_waiting: Deque[Request] = deque()
        self.rt_active: List[Request] = []
        self.finished_rt: List[Request] = []

        self.be_backlog: Deque[Request] = deque()
        self.be_decode_backlog: Deque[Request] = deque()
        self.be_prefilling: Optional[Request] = None
        self.be_decoding: List[Request] = []
        self.finished_be: List[Request] = []
        self._all_be_requests: List[Request] = []

        self.current_time_s = 0.0
        self.iteration_count = 0
        self._all_rt_requests: List[Request] = []
        self._next_rt_index = 0

        self._request_blocks: dict[int, int] = {}
        self._rt_allocated_blocks = 0
        self._be_allocated_blocks = 0

        self._rt_last_token_time: dict[int, float] = {}
        self._admission_violation_recorded: Set[int] = set()
        self.rt_tbt_samples: List[float] = []
        self.rt_tbt_violations = 0
        self.rt_tbt_total_samples = 0
        self.rt_admission_violations = 0
        self.rt_memory_overflow_events = 0
        self.be_swap_out_events = 0
        self.be_swap_in_events = 0
        self._iteration_rt_ready_ids: Set[int] = set()
        self._iteration_rt_memory_failures = 0

        self._next_be_request_id = 10_000_000
        self._next_be_decode_request_id = 20_000_000
        self._termination_reason: Optional[str] = None

        # Core saturation engine for PROGRESSIVE/ORACLE strategies
        self._saturation_engine = self._build_saturation_engine()

    def _build_saturation_engine(self) -> DecodeSaturationEngine:
        """Build the shared saturation engine from simulator config."""
        # The engine's capacity ceiling must match the simulator's actual
        # KV-allocation watermark (usable_gpu_blocks), not the raw physical
        # total.  Otherwise the binary search deems a tau feasible when
        # peak blocks fit within the physical total but exceed the usable
        # budget, and the BE block budget is likewise overestimated.
        params = SaturationParams(
            tbt_slo_ms=self.slo_config.tbt_slo_ms,
            lookahead_window_s=self.sat_config.lookahead_window_s,
            min_interval_ms=self.sat_config.min_interval_ms,
            search_epsilon_ms=self.sat_config.search_epsilon_ms,
            block_size=max(int(self.model_config.block_size), 1),
            total_gpu_blocks=self.usable_gpu_blocks,
            be_prefill_reserve_blocks=int(
                getattr(self.config, "be_prefill_reserve_blocks", 0) or 0),
        )
        # Wrap the latency predictor as a TimePredictorFn callback
        def _time_predictor_fn(prefill_reqs, decode_reqs):
            return self.latency_predictor.predict_iteration_ms(
                decode_reqs=decode_reqs, prefill_reqs=prefill_reqs)
        aggregate_time_predictor_fn = (
            self.latency_predictor.predict_iteration_from_aggregates_ms
        )
        return DecodeSaturationEngine(
            params,
            _time_predictor_fn,
            aggregate_time_predictor_fn,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        rt_requests: List[Request],
        be_requests: List[Request],
    ) -> SimulationResult:
        wall_run_start = time.perf_counter()
        self._reset_runtime_state(rt_requests, be_requests)
        stats_list: List[IterationStats] = []

        logger.info(
            "[%s] Start: %d RT, %d BE, duration=%.1fs, "
            "total_gpu_blocks=%d, lookahead=%.1fs",
            self.strategy.value,
            len(self._all_rt_requests),
            len(self.be_backlog),
            self.config.simulation_duration_s,
            self.total_gpu_blocks,
            self.sat_config.lookahead_window_s,
        )

        while self.current_time_s < self.config.simulation_duration_s:
            wall_step_start = time.perf_counter()
            self._enqueue_rt_arrivals()
            wall_enqueue_end = time.perf_counter()

            admission_violations = self._admit_waiting_rt()
            wall_admit_end = time.perf_counter()

            if self._has_terminal_rt_memory_stall():
                self._termination_reason = "rt_kv_capacity_exhausted"
                logger.warning(
                    "[%s] Terminating at simulated t=%.3fs: all %d active "
                    "RT requests require another KV block, no swappable BE "
                    "decode remains (pinned BE-prefill holds %d blocks), and "
                    "the usable %d-block capacity is full. Judging this RPS "
                    "as exceeding the configuration's KV capacity.",
                    self.strategy.value,
                    self.current_time_s,
                    len(self.rt_active),
                    self._be_allocated_blocks,
                    self.usable_gpu_blocks,
                )
                # The system cannot make further progress. Move the clock to
                # the requested horizon so throughput denominators still cover
                # the full experiment, without fabricating thousands of 1 ms
                # empty iterations.
                self.current_time_s = self.config.simulation_duration_s
                break

            decision = self._make_decode_saturation_decision()
            wall_decision_end = time.perf_counter()
            stats = self._run_one_iteration(decision, admission_violations)
            wall_iteration_end = time.perf_counter()
            stats.wall_enqueue_ms = (
                (wall_enqueue_end - wall_step_start) * 1000.0)
            stats.wall_admit_ms = (
                (wall_admit_end - wall_enqueue_end) * 1000.0)
            stats.wall_decision_ms = (
                (wall_decision_end - wall_admit_end) * 1000.0)
            stats.wall_iteration_ms = (
                (wall_iteration_end - wall_decision_end) * 1000.0)
            stats_list.append(stats)

            self.current_time_s += stats.actual_latency_ms / 1000.0
            self.iteration_count += 1

            if self.iteration_count % 2000 == 0:
                logger.info(
                    "[%s] iter=%d t=%.1fs RT=%d wait=%d target_tbt=%.1f "
                    "lat=%.1f BE_dec=%d prefill=%d viol=%d",
                    self.strategy.value,
                    self.iteration_count,
                    self.current_time_s,
                    len(self.rt_active),
                    len(self.rt_waiting),
                    decision.target_interval_ms,
                    stats.actual_latency_ms,
                    stats.num_be_decode,
                    stats.be_prefill_chunk_size,
                    self.rt_admission_violations + self.rt_tbt_violations
                    + self.rt_memory_overflow_events,
                )

        result = self._compile_results(stats_list)
        result.wall_run_s = time.perf_counter() - wall_run_start
        return result

    def _reset_runtime_state(
        self,
        rt_requests: List[Request],
        be_requests: List[Request],
    ):
        self.rt_waiting = deque()
        self.rt_active = []
        self.finished_rt = []
        self.be_backlog = deque()
        self.be_decode_backlog = deque()
        self.be_prefilling = None
        self.be_decoding = []
        self.finished_be = []

        for req in list(rt_requests) + list(be_requests):
            req.state = RequestState.WAITING
            req.tokens_generated = 0
            req.prefill_tokens_processed = 0
            req.prefill_start_time = None
            req.decode_start_time = None
            req.finish_time = None

        self.current_time_s = 0.0
        self.iteration_count = 0
        self._all_rt_requests = sorted(rt_requests, key=lambda r: r.arrival_time_s)
        self._all_be_requests = list(be_requests)
        self._next_rt_index = 0

        self._request_blocks = {}
        self._rt_allocated_blocks = 0
        self._be_allocated_blocks = 0

        self._rt_last_token_time = {}
        self._admission_violation_recorded = set()
        self.rt_tbt_samples = []
        self.rt_tbt_violations = 0
        self.rt_tbt_total_samples = 0
        self.rt_admission_violations = 0
        self.rt_memory_overflow_events = 0
        self.be_swap_out_events = 0
        self.be_swap_in_events = 0
        self._iteration_rt_ready_ids = set()
        self._iteration_rt_memory_failures = 0

        self._next_be_request_id = 10_000_000
        self._next_be_decode_request_id = 20_000_000
        self._termination_reason = None

        # BE work flows through two independent streams that mirror the real
        # benchmark: a decode-ready stream (cheap dummy-KV handoff, admitted in
        # bulk up to the block budget) and a prefill-only stream (chunked
        # prefill using the reserve region). Both cycle the BE template pool so
        # the GPU stays saturated for the whole run.
        self._refill_be_backlog()
        self._refill_be_decode_backlog()

        self.saturation_predictor.seed_from_trace(self._all_rt_requests[:500])

    def _has_terminal_rt_memory_stall(self) -> bool:
        """Return whether the current RT-only state can never make progress.

        A request that failed its next-token allocation remains at the same
        block boundary. If every active request failed, the usable KV cache is
        full, and the BE side has no swappable decode resident left (any
        remaining BE blocks belong to the pinned in-progress prefill, which
        itself needs a free block to advance), repeating the loop cannot
        change any of those conditions. This state means the RPS exceeds the
        KV capacity of this configuration and is judged an SLO failure.
        """
        return bool(
            self.rt_active
            and not self._iteration_rt_ready_ids
            and self._iteration_rt_memory_failures == len(self.rt_active)
            and not self.be_decoding
            and self._free_gpu_blocks() == 0
        )

    # ------------------------------------------------------------------
    # RT arrivals and target TBT
    # ------------------------------------------------------------------

    def _enqueue_rt_arrivals(self) -> int:
        count = 0
        while (
            self._next_rt_index < len(self._all_rt_requests)
            and self._all_rt_requests[self._next_rt_index].arrival_time_s
            <= self.current_time_s
        ):
            req = self._all_rt_requests[self._next_rt_index]
            req.state = RequestState.WAITING
            self.rt_waiting.append(req)
            self.saturation_predictor.observe_rt_arrival_time(req.arrival_time_s)
            self.saturation_predictor.update_prompt_len_distribution(req.prompt_len)
            self._next_rt_index += 1
            count += 1
        return count

    def _admit_waiting_rt(self) -> int:
        """Prepare all RT KV allocations before scheduling the iteration.

        The memory plan covers both newly admitted requests and the block that
        every active request may need after appending its next token.  BE
        victims are reclaimed in one place before any RT state is advanced.
        """
        violations = 0
        self._iteration_rt_ready_ids = set()
        self._iteration_rt_memory_failures = 0

        active_plans = [
            (req, self._blocks_for_tokens(req.current_seq_len + 1))
            for req in self.rt_active
            if not req.is_finished
        ]
        waiting_plans = [
            (req, self._blocks_for_tokens(req.current_seq_len + 1))
            for req in self.rt_waiting
        ]
        required_delta = sum(
            self._additional_blocks(req, target)
            for req, target in active_plans + waiting_plans
        )
        self._reclaim_be_for_free_blocks(required_delta)

        for req, target_blocks in active_plans:
            if self._try_account_rt_blocks(req, target_blocks):
                self._iteration_rt_ready_ids.add(req.request_id)
            else:
                self.rt_memory_overflow_events += 1
                self._iteration_rt_memory_failures += 1

        while waiting_plans:
            req, needed_blocks = waiting_plans.pop(0)
            if self._try_account_rt_blocks(req, needed_blocks):
                queued = self.rt_waiting.popleft()
                assert queued is req
                req.state = RequestState.DECODING
                req.prefill_tokens_processed = req.prompt_len
                req.decode_start_time = self.current_time_s
                self.rt_active.append(req)
                self._iteration_rt_ready_ids.add(req.request_id)
                continue

            # Even after swapping out every BE-decode request the RT request
            # still does not fit (RT-only demand exceeds the usable budget).
            if req.request_id not in self._admission_violation_recorded:
                self._admission_violation_recorded.add(req.request_id)
                self.rt_admission_violations += 1
                violations += 1
            break
        return violations

    def _make_decode_saturation_decision(self) -> DecodeSaturationDecision:
        active = self._active_rt_forecasts()
        future = self._future_rt_forecasts()

        if self.strategy == DecodeStrategy.GREEDY:
            rt_only_now_ms = self._predict_rt_only_tbt_now_ms(active)
            # BE decode KV budget: saturate all free blocks under the usable
            # watermark cap minus what RT currently holds.
            be_block_budget = max(
                self.usable_gpu_blocks - self._rt_allocated_blocks, 0)
            return DecodeSaturationDecision(
                enabled=False,
                t_rate_ms=self.slo_config.tbt_slo_ms,
                target_interval_ms=self.slo_config.tbt_slo_ms,
                rt_only_tbt_ms=rt_only_now_ms,
                predicted_peak_rt_blocks=self._rt_allocated_blocks,
                current_rt_blocks=self._rt_allocated_blocks,
                current_be_blocks=self._be_allocated_blocks,
                be_block_budget=be_block_budget,
                max_be_prefill_tokens=max(
                    be_block_budget - self._be_allocated_blocks, 0
                ) * self.model_config.block_size,
            )

        # PROGRESSIVE / ORACLE: delegate to the shared engine
        pinned_prefill_blocks = (
            self._request_blocks.get(self.be_prefilling.request_id, 0)
            if self.be_prefilling is not None else 0
        )
        return self._saturation_engine.compute_decision(
            active_rt=active,
            future_rt=future,
            current_rt_blocks=self._rt_allocated_blocks,
            current_be_blocks=self._be_allocated_blocks,
            now_s=self.current_time_s,
            unreclaimable_be_blocks=pinned_prefill_blocks,
        )

    def _active_rt_forecasts(self) -> List[RequestForecast]:
        if self.strategy == DecodeStrategy.PROGRESSIVE:
            batch = getattr(self.saturation_predictor,
                            "predict_active_remaining_lens", None)
            remaining_lengths = (list(batch(self.rt_active)) if callable(batch) else [
                self.saturation_predictor.predict_active_remaining_len(req)
                for req in self.rt_active])
            if len(remaining_lengths) != len(self.rt_active):
                raise RuntimeError("Active RT forecast batch has an invalid row count")
        elif self.strategy == DecodeStrategy.ORACLE:
            remaining_lengths = [req.remaining_output for req in self.rt_active]
        else:
            remaining_lengths = [1] * len(self.rt_active)
        forecasts = []
        for req, remaining in zip(self.rt_active, remaining_lengths):
            forecasts.append(RequestForecast(
                request_id=req.request_id,
                prompt_len=req.prompt_len,
                generated_tokens=req.tokens_generated,
                remaining_tokens=max(int(remaining), 0),
                arrival_time_s=self.current_time_s,
            ))
        return forecasts

    def _future_rt_forecasts(self) -> List[RequestForecast]:
        if self.strategy == DecodeStrategy.GREEDY:
            return []
        if self.strategy == DecodeStrategy.ORACLE:
            return self._oracle_future_rt_forecasts()
        return self._progressive_future_rt_forecasts()

    def _oracle_future_rt_forecasts(self) -> List[RequestForecast]:
        end_s = self.current_time_s + self.sat_config.lookahead_window_s
        future = []
        for req in self._all_rt_requests[self._next_rt_index:]:
            if req.arrival_time_s > end_s:
                break
            future.append(RequestForecast(
                request_id=req.request_id,
                prompt_len=req.prompt_len,
                generated_tokens=0,
                remaining_tokens=max(req.target_output_len, 0),
                arrival_time_s=req.arrival_time_s,
            ))
        return future

    def _progressive_future_rt_forecasts(self) -> List[RequestForecast]:
        predicted = self.saturation_predictor.predict_arima_new_arrivals(
            self.current_time_s, self.sat_config.lookahead_window_s)
        return [
            RequestForecast(
                request_id=f"predicted_rt_{i}",
                prompt_len=prompt_len,
                generated_tokens=0,
                remaining_tokens=output_len,
                arrival_time_s=self.current_time_s + offset_s,
            )
            for i, (offset_s, prompt_len, output_len) in enumerate(predicted, 1)
        ]

    def _predict_rt_only_tbt_now_ms(
        self, active: List[RequestForecast]
    ) -> float:
        lengths = [
            req.prompt_len + req.generated_tokens for req in active
            if req.remaining_tokens > 0
        ]
        if not lengths:
            return self.sat_config.min_interval_ms
        return self._predict_decode_lengths_ms(lengths)

    def _predict_decode_lengths_ms(self, lengths: List[int]) -> float:
        return self.latency_predictor.predict_iteration_ms(
            decode_reqs=[(1, int(length)) for length in lengths],
            prefill_reqs=[],
        )

    # ------------------------------------------------------------------
    # Iteration scheduling
    # ------------------------------------------------------------------

    def _run_one_iteration(
        self,
        decision: DecodeSaturationDecision,
        admission_violations: int,
    ) -> IterationStats:
        target_tbt_ms = decision.target_interval_ms
        # BE decode+prefill admission uses the SAME budget the real scheduler
        # applies: floor(usable * fraction) - predicted_peak_rt - prefill_reserve
        # (decision.be_block_budget). Using the predicted RT peak (not the
        # instantaneous RT allocation) keeps headroom for RT growth, so BE does
        # not over-admit and then have to be swapped out when RT ramps up.
        be_budget_blocks = max(int(decision.be_block_budget), 0)
        reserve_blocks = int(
            getattr(self.config, "be_prefill_reserve_blocks", 0) or 0)
        rt_decoding = [
            req for req in self.rt_active
            if req.request_id in self._iteration_rt_ready_ids
        ]
        if self.config.balance_be_throughput:
            # Pick decode concurrency and prefill work jointly.  Resident
            # decodes not selected by the balanced plan are swapped out before
            # prefill is scheduled; merely leaving them idle on GPU would keep
            # their KV blocks pinned and starve the prefill stream.
            be_decoding = self._select_balanced_be_decode_requests(
                rt_decoding,
                be_budget_blocks,
                be_budget_blocks + reserve_blocks,
                target_tbt_ms,
            )
        else:
            # Legacy decode-first behavior, retained for A/B comparisons.
            self._admit_be_decode_ready(be_budget_blocks)
            be_decoding = self._select_be_decode_requests(
                rt_decoding, be_budget_blocks, target_tbt_ms)
        prefill_chunks = self._select_be_prefill_chunks(
            rt_decoding, be_decoding, be_budget_blocks + reserve_blocks,
            target_tbt_ms)

        actual_latency_ms = max(
            self._predict_iteration_latency(rt_decoding, be_decoding,
                                            prefill_chunks),
            1.0,
        )
        end_time_s = self.current_time_s + actual_latency_ms / 1000.0

        rt_tbt_violations = self._advance_rt_decode(rt_decoding, end_time_s)
        self._advance_be_decode(be_decoding, end_time_s)
        self._advance_be_prefills(prefill_chunks, end_time_s)

        rt_decode_latency = (
            self.latency_predictor.predict_decode_only_ms(rt_decoding, [])
            if rt_decoding else 0.0
        )
        stats = IterationStats(
            iteration_id=self.iteration_count,
            time_s=self.current_time_s,
            num_rt_decode=len(rt_decoding),
            num_rt_waiting=len(self.rt_waiting),
            num_be_decode=len(be_decoding),
            be_prefill_chunk_size=sum(chunk for _, chunk in prefill_chunks),
            rt_blocks=self._rt_allocated_blocks,
            be_blocks=self._be_allocated_blocks,
            predicted_peak_rt_blocks=decision.predicted_peak_rt_blocks,
            be_budget_blocks=be_budget_blocks,
            safe_tbt_ms=decision.t_rate_ms,
            target_tbt_ms=target_tbt_ms,
            rt_decode_latency_ms=rt_decode_latency,
            actual_latency_ms=actual_latency_ms,
            rt_admission_violations=admission_violations,
            rt_memory_allocation_failures=(
                self._iteration_rt_memory_failures),
            rt_memory_overflow=(
                self._iteration_rt_memory_failures > 0
                or self._rt_allocated_blocks > self.usable_gpu_blocks),
            rt_tbt_violations=rt_tbt_violations,
            infeasible=(
                actual_latency_ms > target_tbt_ms
                or not decision.search_feasible
            ),
        )
        if self._rt_allocated_blocks > self.usable_gpu_blocks:
            self.rt_memory_overflow_events += 1
        stats.slo_violated = (
            stats.rt_tbt_violations > 0
            or stats.rt_admission_violations > 0
            or stats.rt_memory_overflow
            or stats.infeasible
        )
        return stats

    def _admit_be_decode_ready(self, be_budget_blocks: int) -> None:
        """Bulk-admit decode-ready BE requests up to the decode block budget.

        Mirrors the real machine's decode-ready path: the DummyConnector hands
        off a pre-filled KV cache in one cheap step, so many BE decodes become
        resident at once rather than being fed one serial prefill at a time.
        Requests are drawn from be_decode_backlog (refilled when empty) and
        accounted at current_seq_len + 1 so the immediate next-token decode has
        room without triggering a swap. Admission stops as soon as the budget
        cannot absorb the next request.
        """
        while True:
            if not self.be_decode_backlog:
                self._refill_be_decode_backlog()
            if not self.be_decode_backlog:
                break
            req = self.be_decode_backlog[0]
            target = self._blocks_for_tokens(req.current_seq_len + 1)
            if not self._try_account_be_blocks(req, target, be_budget_blocks):
                break
            self.be_decode_backlog.popleft()
            req.state = RequestState.DECODING
            req.prefill_tokens_processed = req.prompt_len
            req.decode_start_time = self.current_time_s
            if req not in self.be_decoding:
                self.be_decoding.append(req)
            self.be_swap_in_events += 1

    def _select_be_decode_requests(
        self,
        rt_decoding: List[Request],
        be_budget_blocks: int,
        target_tbt_ms: float,
    ) -> List[Request]:
        selected: List[Request] = []
        for req in list(self.be_decoding):
            if req.state != RequestState.DECODING or req.is_finished:
                continue
            predicted = self.latency_predictor.predict_decode_only_ms(
                rt_decoding, selected + [req])
            if predicted > target_tbt_ms:
                break
            selected.append(req)
        return self._prepare_be_decode_blocks(selected, be_budget_blocks)

    def _select_balanced_be_decode_requests(
        self,
        rt_decoding: List[Request],
        decode_budget_blocks: int,
        total_be_budget_blocks: int,
        target_tbt_ms: float,
    ) -> List[Request]:
        """Choose a BE-decode batch that balances BE P/D request rates.

        For a candidate prefix of ``D`` decode-ready requests, dry-run the
        largest BE-prefill token total that fits the same target-TBT and KV
        budgets.  The two independent streams produce request-equivalents at
        rates ``D / (latency * avg_output_len)`` and
        ``P / (latency * avg_prompt_len)``.  Their minimum is the useful BE
        throughput, so we search the P/D crossover and retain the best point
        around it.
        """
        candidates, non_decode_blocks = self._be_decode_candidates(
            decode_budget_blocks, total_be_budget_blocks)
        avg_prompt_len, avg_output_len = self._be_template_avg_lengths()
        cache: dict[int, _BEBalanceEvaluation] = {}

        def evaluate(decode_count: int) -> _BEBalanceEvaluation:
            decode_count = max(0, min(int(decode_count), len(candidates)))
            cached = cache.get(decode_count)
            if cached is not None:
                return cached

            selected = candidates[:decode_count]
            decode_blocks = sum(
                self._blocks_for_tokens(req.current_seq_len + 1)
                for req in selected
            )
            planned_prefills = self._estimate_be_prefill_chunks(
                rt_decoding=rt_decoding,
                be_decoding=selected,
                initial_be_blocks=non_decode_blocks + decode_blocks,
                be_budget_blocks=total_be_budget_blocks,
                target_tbt_ms=target_tbt_ms,
            )
            prefill_tokens = sum(chunk for _, chunk in planned_prefills)
            predicted_ms = self._predict_iteration_latency(
                rt_decoding, selected, planned_prefills)
            feasible = predicted_ms <= target_tbt_ms + 1e-9
            latency_s = max(predicted_ms / 1000.0, 1e-9)
            if feasible:
                decode_rate = (
                    decode_count / (latency_s * avg_output_len))
                prefill_rate = (
                    prefill_tokens / (latency_s * avg_prompt_len))
                saturation_rate = min(decode_rate, prefill_rate)
            else:
                decode_rate = 0.0
                prefill_rate = 0.0
                saturation_rate = 0.0
            result = _BEBalanceEvaluation(
                decode_count=decode_count,
                prefill_tokens=prefill_tokens,
                predicted_latency_ms=predicted_ms,
                decode_req_per_s=decode_rate,
                prefill_req_per_s=prefill_rate,
                saturation_req_per_s=saturation_rate,
                feasible=feasible,
            )
            cache[decode_count] = result
            return result

        # P decreases and D increases as decode_count grows. Find the first
        # point where D >= P, then compare its immediate neighbours because
        # chunk-size granularity makes the crossover discrete.
        lo, hi = 0, len(candidates)
        while lo < hi:
            mid = (lo + hi) // 2
            point = evaluate(mid)
            if (not point.feasible
                    or point.decode_req_per_s >= point.prefill_req_per_s):
                hi = mid
            else:
                lo = mid + 1
        crossover = lo

        points = [
            evaluate(count)
            for count in range(max(0, crossover - 2),
                               min(len(candidates), crossover + 2) + 1)
        ]
        # Include the capacity endpoint for the P>D-at-cap case and zero for
        # degenerate/infeasible workloads.
        points.extend((evaluate(0), evaluate(len(candidates))))
        best = max(
            points,
            key=lambda point: (
                point.saturation_req_per_s,
                -abs(point.decode_req_per_s - point.prefill_req_per_s),
                point.prefill_tokens,
            ),
        )
        selected = candidates[:best.decode_count]
        return self._apply_be_decode_selection(
            selected, total_be_budget_blocks)

    def _be_decode_candidates(
        self,
        decode_budget_blocks: int,
        total_be_budget_blocks: int,
    ) -> tuple[List[Request], int]:
        """Return the largest FIFO decode prefix feasible after rebalancing."""
        if not self.be_decode_backlog:
            self._refill_be_decode_backlog()

        resident_blocks = sum(
            self._request_blocks.get(req.request_id, 0)
            for req in self.be_decoding
        )
        non_decode_blocks = max(
            self._be_allocated_blocks - resident_blocks, 0)
        decode_blocks = 0
        candidates: List[Request] = []
        for req in list(self.be_decoding) + list(self.be_decode_backlog):
            target_blocks = self._blocks_for_tokens(req.current_seq_len + 1)
            next_decode_blocks = decode_blocks + target_blocks
            next_total_be_blocks = non_decode_blocks + next_decode_blocks
            if next_decode_blocks > max(int(decode_budget_blocks), 0):
                break
            if next_total_be_blocks > max(int(total_be_budget_blocks), 0):
                break
            if (self._rt_allocated_blocks + next_total_be_blocks
                    > self.usable_gpu_blocks):
                break
            candidates.append(req)
            decode_blocks = next_decode_blocks
        return candidates, non_decode_blocks

    def _apply_be_decode_selection(
        self,
        selected: List[Request],
        total_be_budget_blocks: int,
    ) -> List[Request]:
        """Make the dry-run decode selection resident and release all others."""
        selected_ids = {req.request_id for req in selected}
        for req in list(self.be_decoding):
            if req.request_id not in selected_ids:
                self._swap_out_specific_be(req)

        admitted: List[Request] = []
        for req in selected:
            was_resident = req.state == RequestState.DECODING
            if not was_resident:
                try:
                    self.be_decode_backlog.remove(req)
                except ValueError:
                    continue
            target_blocks = self._blocks_for_tokens(req.current_seq_len + 1)
            if not self._try_account_be_blocks(
                    req, target_blocks, total_be_budget_blocks):
                if not was_resident:
                    self.be_decode_backlog.appendleft(req)
                break
            if not was_resident:
                req.state = RequestState.DECODING
                req.prefill_tokens_processed = req.prompt_len
                req.decode_start_time = self.current_time_s
                self.be_decoding.append(req)
                self.be_swap_in_events += 1
            admitted.append(req)
        return admitted

    def _estimate_be_prefill_chunks(
        self,
        rt_decoding: List[Request],
        be_decoding: List[Request],
        initial_be_blocks: int,
        be_budget_blocks: int,
        target_tbt_ms: float,
    ) -> List[tuple[Request, int]]:
        """Dry-run the prefill selector without changing queues or KV state."""
        if self.be_prefilling is None and not self.be_backlog:
            self._refill_be_backlog()
        prefill_candidates: List[Request] = []
        if self.be_prefilling is not None:
            prefill_candidates.append(self.be_prefilling)
        prefill_candidates.extend(self.be_backlog)

        selected: List[tuple[Request, int]] = []
        decode_statistics = self._fixed_decode_statistics(rt_decoding, be_decoding)
        allocated_blocks = max(int(initial_be_blocks), 0)
        for req in prefill_candidates:
            max_prefill_tokens = req.remaining_prefill
            if max_prefill_tokens <= 0:
                continue
            candidates = [
                chunk for chunk in sorted(set(self.config.be_chunk_sizes))
                if 0 < chunk <= max_prefill_tokens
            ]
            if max_prefill_tokens not in candidates:
                candidates.append(max_prefill_tokens)
                candidates.sort()

            best = 0
            lo, hi = 0, len(candidates) - 1
            current_blocks = self._request_blocks.get(req.request_id, 0)
            while lo <= hi:
                mid = (lo + hi) // 2
                chunk = candidates[mid]
                target_blocks = self._blocks_for_tokens(
                    req.prefill_tokens_processed + chunk)
                delta = max(target_blocks - current_blocks, 0)
                memory_ok = (
                    allocated_blocks + delta <= be_budget_blocks
                    and self._rt_allocated_blocks + allocated_blocks + delta
                    <= self.usable_gpu_blocks
                )
                if not memory_ok:
                    hi = mid - 1
                    continue
                predicted = self._predict_with_fixed_decode_statistics(
                    rt_decoding,
                    be_decoding,
                    selected + [(req, chunk)],
                    decode_statistics,
                )
                if predicted <= target_tbt_ms:
                    best = chunk
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best <= 0:
                break

            target_blocks = self._blocks_for_tokens(
                req.prefill_tokens_processed + best)
            allocated_blocks += max(target_blocks - current_blocks, 0)
            selected.append((req, best))
            if best < max_prefill_tokens:
                break
        return selected

    def _be_template_avg_lengths(self) -> tuple[float, float]:
        avg_prompt = float(np.mean([
            max(req.prompt_len, 1) for req in self._be_request_pool
        ]))
        avg_output = float(np.mean([
            max(req.target_output_len, 1) for req in self._be_request_pool
        ]))
        return max(avg_prompt, 1.0), max(avg_output, 1.0)

    def _select_be_prefill_chunks(
        self,
        rt_decoding: List[Request],
        be_decoding: List[Request],
        be_budget_blocks: int,
        target_tbt_ms: float,
    ) -> List[tuple[Request, int]]:
        selected: List[tuple[Request, int]] = []
        while self._be_allocated_blocks < be_budget_blocks:
            req = self._get_or_start_be_prefill()
            if req is None:
                break
            chunk = self._find_be_prefill_chunk_by_latency(
                rt_decoding,
                be_decoding,
                selected,
                req,
                req.remaining_prefill,
                target_tbt_ms,
                be_budget_blocks,
            )
            if chunk <= 0:
                break

            target_blocks = self._blocks_for_tokens(
                req.prefill_tokens_processed + chunk)
            if not self._try_account_be_blocks(
                    req, target_blocks, be_budget_blocks):
                break
            selected.append((req, chunk))

            if chunk < req.remaining_prefill:
                break
            self.be_prefilling = None
        return selected

    def _get_or_start_be_prefill(self) -> Optional[Request]:
        if self.be_prefilling is not None:
            return self.be_prefilling
        if not self.be_backlog:
            self._refill_be_backlog()
        if not self.be_backlog:
            return None
        req = self.be_backlog.popleft()
        req.state = RequestState.PREFILLING
        req.prefill_start_time = self.current_time_s
        self.be_prefilling = req
        return req

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
        if max_prefill_tokens <= 0:
            return 0

        candidates = [
            c for c in sorted(set(self.config.be_chunk_sizes))
            if 0 < c <= max_prefill_tokens
        ]
        if max_prefill_tokens not in candidates:
            candidates.append(max_prefill_tokens)
            candidates = sorted(set(candidates))

        best = 0
        decode_statistics = self._fixed_decode_statistics(rt_decoding, be_decoding)
        lo, hi = 0, len(candidates) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            chunk = candidates[mid]
            target_blocks = self._blocks_for_tokens(
                prefill_req.prefill_tokens_processed + chunk)
            if not self._can_account_be_blocks(
                    prefill_req, target_blocks, be_budget_blocks):
                hi = mid - 1
                continue

            predicted = self._predict_with_fixed_decode_statistics(
                rt_decoding,
                be_decoding,
                scheduled_prefills + [(prefill_req, chunk)],
                decode_statistics,
            )
            if predicted <= target_tbt_ms:
                best = chunk
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _fixed_decode_statistics(self, rt_decoding, be_decoding):
        """Reuse decode sums only within one unchanged prefill search.

        Custom latency hooks keep their tuple-based path. Float accumulation
        follows TimePredictor.predict_forward_ms in RT-then-BE order, without
        integer reductions or reassociation at feasibility boundaries.
        """
        predictor = self.latency_predictor
        if (getattr(self._predict_iteration_latency, "__func__", None)
                is not _NATIVE_SIMULATOR_LATENCY
                or type(predictor) is not LatencyPredictor
                or getattr(predictor.predict_iteration_ms, "__func__", None)
                is not _NATIVE_LATENCY_FORWARD
                or getattr(predictor.predict_iteration_from_aggregates_ms,
                           "__func__", None)
                is not _NATIVE_LATENCY_AGGREGATE
                or getattr(predictor.predictor.predict_forward_ms,
                           "__func__", None)
                is not _NATIVE_TIME_FORWARD):
            return None
        decode_tokens = 0.0
        decode_sum_ctx = 0.0
        for requests in (rt_decoding, be_decoding):
            for request in requests:
                decode_tokens += 1.0
                decode_sum_ctx += 1.0 * float(request.current_seq_len)
        return decode_tokens, decode_sum_ctx, len(rt_decoding) + len(be_decoding)

    def _predict_with_fixed_decode_statistics(
        self, rt_decoding, be_decoding, prefill_chunks, decode_statistics,
    ):
        if decode_statistics is None:
            return self._predict_iteration_latency(
                rt_decoding, be_decoding, prefill_chunks)
        decode_tokens, decode_sum_ctx, decode_count = decode_statistics
        prefill_tokens = 0.0
        prefill_sum_ctx = 0.0
        prefill_count = 0
        for request, chunk in prefill_chunks:
            if chunk <= 0:
                continue
            query = float(chunk)
            context = float(request.prefill_tokens_processed + chunk)
            cached = max(0.0, context - query)
            prefill_tokens += query
            prefill_sum_ctx += query * cached + query * (query + 1.0) / 2.0
            prefill_count += 1
        if decode_tokens == 0 and prefill_tokens == 0:
            return 0.0
        return self.latency_predictor.predict_iteration_from_aggregates_ms(
            prefill_tokens=prefill_tokens,
            prefill_sum_ctx=prefill_sum_ctx,
            decode_tokens=decode_tokens,
            decode_sum_ctx=decode_sum_ctx,
            prefill_request_count=prefill_count,
            decode_request_count=decode_count,
        )

    def _predict_iteration_latency(
        self,
        rt_decoding: List[Request],
        be_decoding: List[Request],
        prefill_chunks: List[tuple[Request, int]],
    ) -> float:
        decode_reqs = [(1, req.current_seq_len) for req in rt_decoding]
        decode_reqs += [(1, req.current_seq_len) for req in be_decoding]
        prefill_reqs = [
            (chunk, req.prefill_tokens_processed + chunk)
            for req, chunk in prefill_chunks
            if chunk > 0
        ]
        if not decode_reqs and not prefill_reqs:
            return 0.0
        return self.latency_predictor.predict_iteration_ms(
            decode_reqs=decode_reqs,
            prefill_reqs=prefill_reqs,
        )

    # ------------------------------------------------------------------
    # State advancement
    # ------------------------------------------------------------------

    def _advance_rt_decode(
        self,
        rt_decoding: List[Request],
        end_time_s: float,
    ) -> int:
        violations = 0
        finished: List[Request] = []
        for req in rt_decoding:
            last_time = self._rt_last_token_time.get(req.request_id)
            if last_time is not None:
                tbt_ms = (end_time_s - last_time) * 1000.0
                if end_time_s > self.config.warmup_s:
                    self.rt_tbt_samples.append(tbt_ms)
                    self.rt_tbt_total_samples += 1
                    if tbt_ms > self.slo_config.tbt_slo_ms:
                        self.rt_tbt_violations += 1
                        violations += 1

            req.generate_one_token()
            self._rt_last_token_time[req.request_id] = end_time_s
            if req.is_finished:
                req.finish_time = end_time_s
                finished.append(req)
                self.saturation_predictor.update_model_len_history(
                    req.prompt_len, req.tokens_generated)

        for req in finished:
            if req in self.rt_active:
                self.rt_active.remove(req)
            self.finished_rt.append(req)
            self._release_request_blocks(req)
            self._rt_last_token_time.pop(req.request_id, None)
        return violations

    def _advance_be_decode(
        self,
        be_decoding: List[Request],
        end_time_s: float,
    ):
        finished: List[Request] = []
        for req in be_decoding:
            req.generate_one_token()
            if req.is_finished:
                req.finish_time = end_time_s
                finished.append(req)

        for req in finished:
            if req in self.be_decoding:
                self.be_decoding.remove(req)
            self.finished_be.append(req)
            self._release_request_blocks(req)

    def _advance_be_prefills(
        self,
        prefill_chunks: List[tuple[Request, int]],
        end_time_s: float,
    ):
        for req, chunk in prefill_chunks:
            if chunk <= 0:
                continue
            req.process_prefill_chunk(chunk)
            if req.state == RequestState.DECODING:
                # The prefill-only stream mirrors the real benchmark's
                # phase="prefill" requests (max_tokens=1): once the prompt is
                # fully prefilled the request is done. It does NOT join the
                # resident decode pool -- decode concurrency is driven solely by
                # the decode-ready handoff stream (_admit_be_decode_ready).
                req.state = RequestState.FINISHED
                req.finish_time = end_time_s
                self.finished_be.append(req)
                self._release_request_blocks(req)
                if self.be_prefilling is req:
                    self.be_prefilling = None

    # ------------------------------------------------------------------
    # Block accounting and BE generation
    # ------------------------------------------------------------------

    def _blocks_for_tokens(self, num_tokens: int) -> int:
        block_size = max(int(self.model_config.block_size), 1)
        return max(int(math.ceil(max(int(num_tokens), 1) / block_size)), 1)

    def _can_account_rt_blocks(self, request: Request, target_blocks: int) -> bool:
        delta = self._additional_blocks(request, target_blocks)
        return self._has_total_gpu_blocks(delta)

    def _additional_blocks(self, request: Request, target_blocks: int) -> int:
        current = self._request_blocks.get(request.request_id, 0)
        return max(int(target_blocks) - current, 0)

    def _free_gpu_blocks(self) -> int:
        return max(
            self.usable_gpu_blocks
            - self._rt_allocated_blocks
            - self._be_allocated_blocks,
            0,
        )

    def _reclaim_be_for_free_blocks(
        self,
        required_free_blocks: int,
        exclude_ids: Optional[Set[int]] = None,
    ) -> bool:
        """Swap out shortest BE decodes until the requested space is free."""
        required = max(int(required_free_blocks), 0)
        excluded = exclude_ids or set()
        while self._free_gpu_blocks() < required:
            if not self._swap_out_shortest_be(excluded):
                break
        return self._free_gpu_blocks() >= required

    def _try_account_rt_blocks(self, request: Request, target_blocks: int) -> bool:
        if not self._can_account_rt_blocks(request, target_blocks):
            return False
        self._set_accounted_blocks(request, target_blocks, is_rt=True)
        return True

    def _can_account_be_blocks(
        self,
        request: Request,
        target_blocks: int,
        be_budget_blocks: Optional[int] = None,
    ) -> bool:
        current = self._request_blocks.get(request.request_id, 0)
        delta = max(int(target_blocks) - current, 0)
        if be_budget_blocks is not None:
            limit = max(int(be_budget_blocks), 0)
        else:
            limit = self.usable_gpu_blocks
        return (
            self._be_allocated_blocks + delta <= limit
            and self._has_total_gpu_blocks(delta)
        )

    def _try_account_be_blocks(
        self,
        request: Request,
        target_blocks: int,
        be_budget_blocks: Optional[int] = None,
    ) -> bool:
        if not self._can_account_be_blocks(
                request, target_blocks, be_budget_blocks):
            return False
        self._set_accounted_blocks(request, target_blocks, is_rt=False)
        return True

    def _prepare_be_decode_blocks(
        self,
        selected: List[Request],
        be_budget_blocks: int,
    ) -> List[Request]:
        """Prepare all selected BE next-token blocks before execution.

        Unscheduled BE requests are reclaimed first.  If that is insufficient,
        the shortest selected BE requests are removed from this iteration and
        swapped out until the remaining batch fits both the BE budget and the
        physical watermark.
        """
        selected = list(selected)
        while selected:
            required_delta = sum(
                self._additional_blocks(
                    req, self._blocks_for_tokens(req.current_seq_len + 1))
                for req in selected
            )
            budget_ok = (
                self._be_allocated_blocks + required_delta
                <= max(int(be_budget_blocks), 0)
            )
            physical_ok = self._free_gpu_blocks() >= required_delta
            if budget_ok and physical_ok:
                break

            selected_ids = {req.request_id for req in selected}
            if self._swap_out_shortest_be(selected_ids):
                continue

            victim = min(selected, key=lambda req: req.current_seq_len)
            selected.remove(victim)
            self._swap_out_specific_be(victim)

        prepared: List[Request] = []
        for req in selected:
            target = self._blocks_for_tokens(req.current_seq_len + 1)
            if self._try_account_be_blocks(
                    req, target, be_budget_blocks):
                prepared.append(req)
        return prepared

    def _set_accounted_blocks(self, request: Request, target_blocks: int, is_rt: bool):
        """Update the simulated KV-block accounting for one request."""
        current = self._request_blocks.get(request.request_id, 0)
        delta = max(int(target_blocks) - current, 0)
        if delta <= 0:
            return
        self._request_blocks[request.request_id] = int(target_blocks)
        if is_rt:
            self._rt_allocated_blocks += delta
        else:
            self._be_allocated_blocks += delta

    def _release_request_blocks(self, request: Request):
        blocks = self._request_blocks.pop(request.request_id, 0)
        if request.request_type == RequestType.RT:
            self._rt_allocated_blocks = max(self._rt_allocated_blocks - blocks, 0)
        else:
            self._be_allocated_blocks = max(self._be_allocated_blocks - blocks, 0)

    def _swap_out_shortest_be(self, exclude_ids: Set[int]) -> bool:
        """Swap the shortest GPU-resident BE-decode request out to CPU.

        Only BE-decode requests are eligible victims; the in-progress
        BE-prefill is never swapped out (evicting a partially prefilled
        request would strand its progress and fragment prefill work). D-pool
        does not model recompute: swapping is simulated by reclaiming the
        victim's GPU KV blocks and returning it (with its decode progress
        preserved) to the waiting backlog, to be re-admitted later once blocks
        free up. Returns True if a victim was swapped out, False if none exist.
        """
        candidates = list(self.be_decoding)
        victim: Optional[Request] = None
        for req in candidates:
            if req.request_id in exclude_ids:
                continue
            if req.is_finished:
                continue
            req_key = (
                self._request_blocks.get(req.request_id, 0),
                req.current_seq_len,
            )
            victim_key = (
                self._request_blocks.get(victim.request_id, 0),
                victim.current_seq_len,
            ) if victim is not None else None
            if victim is None or req_key < victim_key:
                victim = req
        if victim is None:
            return False
        self._swap_out_specific_be(victim)
        return True

    def _swap_out_specific_be(self, victim: Request) -> None:
        """Reclaim one BE decode while preserving its decode progress."""
        self._release_request_blocks(victim)
        victim.state = RequestState.WAITING
        victim.decode_start_time = None
        if victim in self.be_decoding:
            self.be_decoding.remove(victim)
        if self.be_prefilling is victim:
            self.be_prefilling = None
        self.be_decode_backlog.appendleft(victim)
        self.be_swap_out_events += 1

    def _has_total_gpu_blocks(self, delta_blocks: int) -> bool:
        return (
            self._rt_allocated_blocks + self._be_allocated_blocks
            + max(delta_blocks, 0)
            <= self.usable_gpu_blocks
        )

    def _load_be_request_pool(self) -> List[Request]:
        """Load BE request length templates from be_workload_csv.

        The CSV shares the online scheduler's --be-workload-csv format
        (columns: request_id, prompt_len, output_len, arrival_offset_s). Only
        prompt_len/output_len are used here as length templates; BE requests
        are synthesized by cycling through this pool.
        """
        path = getattr(self.config, "be_workload_csv", None)
        if not path:
            raise ValueError(
                "[decode_sim] be_workload_csv is required but not configured")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"[decode_sim] BE workload CSV not found: {path}")
        pairs: List[tuple[int, int]] = []
        max_prompt_len = getattr(self.config, "be_max_prompt_len", None)
        skipped_prompt_too_long = 0
        clamped_output_len = 0
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    in_len = int(float(row["prompt_len"]))
                    out_len = int(float(row["output_len"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if in_len <= 0 or out_len <= 0:
                    continue
                if max_prompt_len is not None:
                    # Mirror the real benchmark's load_length_trace: drop rows
                    # whose prompt cannot fit the model context and clamp the
                    # output to the remaining budget.
                    if in_len >= int(max_prompt_len):
                        skipped_prompt_too_long += 1
                        continue
                    max_output = int(max_prompt_len) - in_len
                    if out_len > max_output:
                        out_len = max(1, max_output)
                        clamped_output_len += 1
                pairs.append((in_len, out_len))
        if not pairs:
            raise ValueError(
                f"[decode_sim] BE workload CSV has no usable rows: {path}")
        pool = [
            Request(
                request_id=0,
                request_type=RequestType.BE,
                prompt_len=int(in_len),
                target_output_len=int(out_len),
                arrival_time_s=0.0,
            )
            for in_len, out_len in pairs
        ]
        logger.info(
            "[decode_sim] loaded %d BE request templates from %s "
            "(max_prompt_len=%s skipped_too_long=%d clamped_output=%d)",
            len(pool), path, max_prompt_len,
            skipped_prompt_too_long, clamped_output_len)
        return pool

    def _refill_be_backlog(self) -> None:
        """Cycle through _be_request_pool and refill be_backlog.

        When the backlog runs dry, this copies every template from the master
        pool into fresh Request objects, cycling back to the beginning.
        """
        for template in self._be_request_pool:
            req = Request(
                request_id=self._next_be_request_id,
                request_type=RequestType.BE,
                prompt_len=template.prompt_len,
                target_output_len=template.target_output_len,
                arrival_time_s=self.current_time_s,
            )
            self._next_be_request_id += 1
            self.be_backlog.append(req)
            self._all_be_requests.append(req)

    def _refill_be_decode_backlog(self) -> None:
        """Cycle through _be_request_pool and refill be_decode_backlog.

        Produces fresh decode-ready BE requests (same length templates as the
        prefill stream) that are admitted via a one-step handoff, matching the
        real DummyConnector decode-ready path.
        """
        for template in self._be_request_pool:
            req = Request(
                request_id=self._next_be_decode_request_id,
                request_type=RequestType.BE,
                prompt_len=template.prompt_len,
                target_output_len=template.target_output_len,
                arrival_time_s=self.current_time_s,
            )
            self._next_be_decode_request_id += 1
            self.be_decode_backlog.append(req)
            self._all_be_requests.append(req)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _compile_results(self, stats_list: List[IterationStats]) -> SimulationResult:
        result = SimulationResult()
        result.strategy = self.strategy.value
        result.total_rt_requests = len(self._all_rt_requests)
        result.completed_rt_requests = len(self.finished_rt)
        result.total_be_completed = len(self.finished_be)
        result.total_iterations = len(stats_list)
        result.iteration_stats = stats_list
        result.termination_reason = self._termination_reason
        result.terminated_early = self._termination_reason is not None
        result.rps_infeasible = (
            self._termination_reason == "rt_kv_capacity_exhausted")

        result.total_be_prefill_tokens = sum(s.be_prefill_chunk_size for s in stats_list)
        result.total_be_decode_tokens = sum(s.num_be_decode for s in stats_list)

        throughput_stats = [
            stat for stat in stats_list if stat.time_s >= self.config.warmup_s
        ]
        if throughput_stats:
            prefill_tokens = sum(s.be_prefill_chunk_size for s in throughput_stats)
            decode_tokens = sum(s.num_be_decode for s in throughput_stats)
            measured_time_s = max(
                self.current_time_s - self.config.warmup_s, 1e-9)
        else:
            prefill_tokens = result.total_be_prefill_tokens
            decode_tokens = result.total_be_decode_tokens
            measured_time_s = max(self.current_time_s, 1e-9)
        throughput_tokens = prefill_tokens + decode_tokens
        result.be_throughput_tokens_per_s = throughput_tokens / measured_time_s
        result.be_prefill_throughput_tokens_per_s = (
            prefill_tokens / measured_time_s)
        result.be_decode_throughput_tokens_per_s = (
            decode_tokens / measured_time_s)
        avg_be_prompt_len, avg_be_output_len = self._avg_be_request_lengths()
        result.be_prefill_req_per_s = (
            result.be_prefill_throughput_tokens_per_s / avg_be_prompt_len)
        result.be_decode_req_per_s = (
            result.be_decode_throughput_tokens_per_s / avg_be_output_len)
        result.be_saturation_req_per_s = min(
            result.be_prefill_req_per_s,
            result.be_decode_req_per_s,
        )

        if self.rt_tbt_samples:
            arr = np.array(self.rt_tbt_samples)
            result.rt_tbt_p50_ms = float(np.percentile(arr, 50))
            result.rt_tbt_p99_ms = float(np.percentile(arr, 99))
            result.rt_tbt_max_ms = float(np.max(arr))
        result.rt_tbt_violations = self.rt_tbt_violations
        result.rt_tbt_total_samples = self.rt_tbt_total_samples
        if self.rt_tbt_total_samples > 0:
            result.rt_tbt_violation_rate = (
                self.rt_tbt_violations / self.rt_tbt_total_samples)

        result.rt_admission_violations = self.rt_admission_violations
        result.rt_memory_overflow_events = self.rt_memory_overflow_events
        result.infeasible_iterations = sum(1 for s in stats_list if s.infeasible)
        result.be_swap_out_events = self.be_swap_out_events
        result.be_swap_in_events = self.be_swap_in_events
        result.rt_slo_violations_total = (
            result.rt_tbt_violations
            + result.rt_admission_violations
            + result.rt_memory_overflow_events
        )

        result.wall_enqueue_s = sum(
            s.wall_enqueue_ms for s in stats_list) / 1000.0
        result.wall_admit_s = sum(
            s.wall_admit_ms for s in stats_list) / 1000.0
        result.wall_decision_s = sum(
            s.wall_decision_ms for s in stats_list) / 1000.0
        result.wall_iteration_s = sum(
            s.wall_iteration_ms for s in stats_list) / 1000.0

        if stats_list:
            tbt_slo = max(self.slo_config.tbt_slo_ms, 1e-9)
            result.avg_saturation_ratio = float(
                np.mean([min(s.actual_latency_ms / tbt_slo, 1.0)
                         for s in stats_list]))
            result.avg_be_prefill_chunk = float(
                np.mean([s.be_prefill_chunk_size for s in stats_list]))
            result.avg_rt_batch_size = float(
                np.mean([s.num_rt_decode for s in stats_list]))
            result.avg_rt_queue_size = float(
                np.mean([s.num_rt_waiting for s in stats_list]))
            result.avg_be_batch_size = float(
                np.mean([s.num_be_decode for s in stats_list]))
            result.avg_iteration_latency_ms = float(
                np.mean([s.actual_latency_ms for s in stats_list]))

            safe_tbts = np.array([s.safe_tbt_ms for s in stats_list])
            target_tbts = np.array([s.target_tbt_ms for s in stats_list])
            result.avg_safe_tbt_ms = float(np.mean(safe_tbts))
            result.avg_target_tbt_ms = float(np.mean(target_tbts))
            result.p50_target_tbt_ms = float(np.percentile(target_tbts, 50))
            result.p99_target_tbt_ms = float(np.percentile(target_tbts, 99))
            result.peak_rt_blocks = int(
                max(s.rt_blocks for s in stats_list))
            result.avg_rt_blocks = float(
                np.mean([s.rt_blocks for s in stats_list]))
            result.avg_be_blocks = float(
                np.mean([s.be_blocks for s in stats_list]))

        return result

    def _avg_be_request_lengths(self) -> tuple[float, float]:
        requests = self._all_be_requests
        if requests:
            avg_prompt = float(np.mean([max(req.prompt_len, 1)
                                        for req in requests]))
            avg_output = float(np.mean([max(req.target_output_len, 1)
                                        for req in requests]))
            return max(avg_prompt, 1.0), max(avg_output, 1.0)

        pool = getattr(self, "_be_request_pool", None)
        if pool:
            avg_prompt = float(np.mean([max(r.prompt_len, 1) for r in pool]))
            avg_output = float(np.mean([max(r.target_output_len, 1) for r in pool]))
            return max(avg_prompt, 1.0), max(avg_output, 1.0)
        return 1.0, 1.0


_NATIVE_SIMULATOR_LATENCY = DecodeSimulator._predict_iteration_latency
