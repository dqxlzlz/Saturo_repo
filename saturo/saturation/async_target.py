"""Bounded background D-target optimization over immutable request snapshots."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .types import DecodeSaturationDecision, RequestForecast, SaturationParams


@dataclass(frozen=True)
class RequestFeatures:
    request_id: str
    prompt_len: int
    generated_tokens: int
    arrival_time_s: float
    prompt_token_ids: Optional[tuple[int, ...]] = None
    generated_token_ids: Optional[tuple[int, ...]] = None
    forecast_prompt_len: Optional[int] = None

    def prediction_row(self) -> dict:
        return dict(prompt_len=self.prompt_len,
                    generated_len=self.generated_tokens,
                    prompt_token_ids=self.prompt_token_ids,
                    generated_token_ids=self.generated_token_ids)


@dataclass(frozen=True)
class ArrivalState:
    history: Optional[tuple[tuple[str, Any], ...]]
    intervals: tuple[float, ...]
    last_arrival_time_s: Optional[float]
    min_history_intervals: int
    max_history_intervals: int
    min_interval_s: float
    prompt_len_sum: float
    prompt_len_count: int
    output_len_sum: float
    output_len_count: int


@dataclass(frozen=True)
class TargetSnapshot:
    version: int
    epoch: int
    now_s: float
    requests: tuple[RequestFeatures, ...]
    arrivals: ArrivalState
    current_rt_blocks: int
    current_be_blocks: int
    pinned_prefill_blocks: int
    submitted_perf_s: float


@dataclass(frozen=True)
class TargetResult:
    version: int
    epoch: int
    captured_time_s: float
    started_perf_s: float
    completed_perf_s: float
    decision: Optional[DecodeSaturationDecision]
    timings: tuple[tuple[str, float], ...] = ()
    error_type: Optional[str] = None
    error_message: str = ""


class TargetComputer:
    """Private predictor/model cache, accessed by the optimizer thread only.

    Templates are read-only fitted artifacts. Deep copies, QRF inference,
    future forecasting and target search all happen on the background thread.
    Neither the serving provider nor a native sequence/block manager is used.
    """

    def __init__(self, predictor, time_predictor, params: SaturationParams,
                 eta: float):
        self._predictor_template = predictor
        self._time_template = time_predictor
        self.params = params
        self.eta = float(eta)
        self._predictor = None
        self._time_predictor = None
        self._engine = None
        self.initialized = False

    def __call__(self, snapshot: TargetSnapshot):
        from .decode_saturation_core import (
            DecodeSaturationEngine, DefaultDecodeSaturationStateProvider)
        from saturo.predictors.request_arima_arrival_predictor import (
            ARIMAArrivalPredictor)

        timings = {}
        begin = time.perf_counter()
        if not self.initialized:
            self._predictor = copy.deepcopy(self._predictor_template)
            self._time_predictor = copy.deepcopy(self._time_template)
            predictor = self._time_predictor
            self._engine = DecodeSaturationEngine(
                self.params,
                predictor.predict_forward_ms if predictor is not None else None,
                getattr(predictor, "predict_forward_from_aggregates_ms", None),
                getattr(predictor, "predict_forward_from_aggregates_batch_ms", None))
            self._predictor_template = self._time_template = None
            self.initialized = True
        timings["kernel_init_ms"] = (time.perf_counter() - begin) * 1000

        # Construct a private provider from scalar/tuple history, never copy a
        # live provider whose observations can change on the engine thread.
        provider = DefaultDecodeSaturationStateProvider.__new__(
            DefaultDecodeSaturationStateProvider)
        provider.model_len_predictor = self._predictor
        provider.model_len_predictor_eta = self.eta
        provider.prediction_calls = 0
        state = snapshot.arrivals
        provider._history_statistics = (
            dict(state.history) if state.history is not None else None)
        provider._lookahead_window_s = self.params.lookahead_window_s
        provider._prompt_len_sum = state.prompt_len_sum
        provider._prompt_len_count = state.prompt_len_count
        provider._output_len_sum = state.output_len_sum
        provider._output_len_count = state.output_len_count
        arrival = ARIMAArrivalPredictor(
            min_history_intervals=state.min_history_intervals,
            max_history_intervals=state.max_history_intervals,
            min_interval_s=state.min_interval_s)
        arrival._inter_arrivals_s = list(state.intervals)
        arrival._last_arrival_time_s = state.last_arrival_time_s
        provider._arima_arrival_predictor = arrival

        begin = time.perf_counter()
        rows = [request.prediction_row() for request in snapshot.requests]
        predictions = provider.predict_remaining_feature_rows(rows)
        active = [RequestForecast(
            request.request_id, (request.forecast_prompt_len
                                 if request.forecast_prompt_len is not None
                                 else request.prompt_len), request.generated_tokens,
            remaining, request.arrival_time_s)
            for request, remaining in zip(snapshot.requests, predictions)]
        timings["qrf_prediction_ms"] = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        future = list(provider.get_future_arrivals(
            snapshot.now_s, self.params.lookahead_window_s))
        timings["future_forecast_ms"] = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        decision = self._engine.compute_decision(
            active_rt=active, future_rt=future,
            current_rt_blocks=snapshot.current_rt_blocks,
            current_be_blocks=snapshot.current_be_blocks,
            now_s=snapshot.now_s,
            unreclaimable_be_blocks=snapshot.pinned_prefill_blocks)
        timings["target_search_ms"] = (time.perf_counter() - begin) * 1000
        timings["active_rt_count"] = len(active)
        timings["future_rt_count"] = len(future)
        timings["qrf_prediction_calls"] = provider.prediction_calls
        cache_info = getattr(self._predictor, "remaining_batch_cache_info", None)
        if callable(cache_info):
            for key, value in cache_info().items():
                timings["qrf_cache_"+key] = value
        return decision, tuple(timings.items())


class AsyncTargetOptimizer:
    """Exactly one active computation and one replaceable pending snapshot."""

    def __init__(self):
        self._condition = threading.Condition()
        self._pending = None
        self._active = None
        self._active_started_s = None
        self._result = None
        self._latest = (0, 0)
        self._closed = False
        self._submitted = self._completed = self._discarded = 0
        self._coalesced = 0
        self._prediction_calls = 0
        self._thread = threading.Thread(
            target=self._run, name="saturo-d-target", daemon=True)
        self._thread.start()

    def submit(self, snapshot: TargetSnapshot, computer: Callable) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Async target optimizer is closed")
            key = (snapshot.epoch, snapshot.version)
            if key <= self._latest:
                raise ValueError("Async target versions must increase")
            self._latest = key
            self._result = None
            self._coalesced += int(self._pending is not None)
            self._pending = (snapshot, computer)
            self._submitted += 1
            self._condition.notify_all()

    def invalidate(self, epoch: int, version: int) -> None:
        with self._condition:
            if (epoch, version) <= self._latest:
                raise ValueError("Async target invalidation must advance version")
            self._latest = (epoch, version)
            self._pending = self._result = None
            self._condition.notify_all()

    def poll(self) -> Optional[TargetResult]:
        with self._condition:
            result, self._result = self._result, None
            return result

    def wait(self, timeout_s: float) -> Optional[TargetResult]:
        """Startup/test-only wait; native scheduling uses poll exclusively."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._result is not None,
                timeout=max(float(timeout_s), 0.))
            result, self._result = self._result, None
            return result

    def active_age_s(self) -> float:
        with self._condition:
            return (time.perf_counter() - self._active_started_s
                    if self._active_started_s is not None else 0.)

    def diagnostics(self) -> dict:
        with self._condition:
            return dict(active=int(self._active is not None),
                        pending=int(self._pending is not None),
                        submitted=self._submitted, completed=self._completed,
                        discarded=self._discarded, coalesced=self._coalesced,
                        latest_epoch=self._latest[0],
                        latest_version=self._latest[1], closed=self._closed,
                        qrf_prediction_calls=self._prediction_calls,
                        thread_alive=self._thread.is_alive())

    def close(self, timeout_s: float = 1.) -> bool:
        with self._condition:
            self._closed = True
            self._pending = self._result = None
            self._condition.notify_all()
        self._thread.join(timeout=max(float(timeout_s), 0.))
        return not self._thread.is_alive()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                snapshot, computer = self._pending
                self._pending = None
                self._active = snapshot
                self._active_started_s = started = time.perf_counter()
            cpu_started = time.thread_time()
            decision = None
            timings = ()
            error_type, error_message = None, ""
            try:
                decision, timings = computer(snapshot)
            except Exception as exc:
                error_type, error_message = type(exc).__name__, str(exc)
            completed = time.perf_counter()
            timings = tuple(timings) + (
                ("worker_wall_ms", (completed - started) * 1000),
                ("worker_cpu_ms", (time.thread_time() - cpu_started) * 1000),
                ("queue_wait_ms", (started - snapshot.submitted_perf_s) * 1000))
            result = TargetResult(
                snapshot.version, snapshot.epoch, snapshot.now_s,
                started, completed, decision, timings, error_type, error_message)
            with self._condition:
                self._active = self._active_started_s = None
                self._completed += 1
                self._prediction_calls += int(dict(timings).get(
                    "qrf_prediction_calls", 0))
                if (not self._closed and self._latest
                        == (snapshot.epoch, snapshot.version)):
                    self._result = result
                else:
                    self._discarded += 1
                self._condition.notify_all()
