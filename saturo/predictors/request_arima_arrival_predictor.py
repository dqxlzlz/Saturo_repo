"""Arrival-time predictors for decode saturation planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class ArrivalForecast:
    """Predicted arrivals relative to the current decision time."""

    offsets_s: List[float]
    inter_arrivals_s: List[float]


class ARIMAArrivalPredictor:
    """Robust online predictor over RT inter-arrival times.

    The model observes exact request arrival timestamps, turns them into an
    inter-arrival-time series, and predicts the next intervals in log space.
    A robust local trend is damped toward the historical median so a short or
    bursty history cannot collapse every future interval to the absolute
    minimum. Cumulative predicted intervals become future arrival offsets.

    Forecasts can be conditioned on ``now_s``. This is important for an online
    scheduler: offsets are relative to the decision time, not to the most
    recent observed arrival.
    """

    def __init__(
        self,
        *,
        min_history_intervals: int = 4,
        max_history_intervals: int = 1024,
        min_interval_s: float = 1e-3,
    ) -> None:
        self.min_history_intervals = max(int(min_history_intervals), 2)
        self.max_history_intervals = max(
            int(max_history_intervals), self.min_history_intervals)
        self.min_interval_s = max(float(min_interval_s), 1e-9)
        self._last_arrival_time_s: float | None = None
        self._inter_arrivals_s: List[float] = []

    def observe_arrival(self, arrival_time_s: float) -> None:
        arrival_time_s = float(arrival_time_s)
        if self._last_arrival_time_s is not None:
            delta = arrival_time_s - self._last_arrival_time_s
            if delta > 0:
                self._inter_arrivals_s.append(max(delta, self.min_interval_s))
                if len(self._inter_arrivals_s) > self.max_history_intervals:
                    self._inter_arrivals_s = (
                        self._inter_arrivals_s[-self.max_history_intervals:])
        if (self._last_arrival_time_s is None
                or arrival_time_s > self._last_arrival_time_s):
            self._last_arrival_time_s = arrival_time_s

    def observe_arrivals(self, arrival_times_s: Iterable[float]) -> None:
        for arrival_time_s in sorted(float(t) for t in arrival_times_s):
            self.observe_arrival(arrival_time_s)

    def observe(self, new_arrivals: int, window_s: float) -> None:
        """Compatibility path when only aggregate counts are available.

        Exact timestamps are preferred. If a caller only has a count over a
        window, approximate the missing inter-arrival samples by evenly splitting
        that observed window.
        """

        if new_arrivals <= 0 or window_s <= 0:
            return
        interval = max(float(window_s) / float(new_arrivals), self.min_interval_s)
        self._inter_arrivals_s.extend([interval] * int(new_arrivals))
        if len(self._inter_arrivals_s) > self.max_history_intervals:
            self._inter_arrivals_s = (
                self._inter_arrivals_s[-self.max_history_intervals:])

    def seed_from_arrival_times(self, arrival_times_s: Iterable[float]) -> None:
        self._last_arrival_time_s = None
        self._inter_arrivals_s = []
        self.observe_arrivals(arrival_times_s)

    def predict_arrival_offsets(
        self,
        window_s: float,
        now_s: float | None = None,
    ) -> List[float]:
        return self.forecast(window_s, now_s=now_s).offsets_s

    def forecast(
        self,
        window_s: float,
        now_s: float | None = None,
    ) -> ArrivalForecast:
        if window_s <= 0:
            return ArrivalForecast(offsets_s=[], inter_arrivals_s=[])

        age_s = 0.0
        if now_s is not None and self._last_arrival_time_s is not None:
            age_s = max(float(now_s) - self._last_arrival_time_s, 0.0)

        intervals = self.forecast_inter_arrivals(window_s + age_s)
        offsets: List[float] = []
        elapsed_from_last = 0.0
        for interval in intervals:
            elapsed_from_last += interval
            offset = elapsed_from_last - age_s
            # If a forecasted arrival is already overdue, condition on the
            # fact that it did not occur and advance to the next one.
            if offset <= 0:
                continue
            if offset > window_s:
                break
            offsets.append(offset)
        return ArrivalForecast(offsets_s=offsets, inter_arrivals_s=intervals)

    def forecast_inter_arrivals(self, window_s: float) -> List[float]:
        history = np.asarray(self._inter_arrivals_s, dtype=np.float64)
        if window_s <= 0 or history.size == 0:
            return []

        # Cap the number of recursive steps so a tiny interval forecast cannot
        # spin forever on a long lookahead window.
        lower_bound = max(float(np.percentile(history, 5)), self.min_interval_s)
        max_steps = max(int(math.ceil(window_s / lower_bound)) + 1, 1)

        if history.size < self.min_history_intervals:
            baseline = max(float(np.median(history)), self.min_interval_s)
            return [baseline] * max_steps

        # Inter-arrival intervals are positive and often heavy-tailed. Work in
        # log space and use a median local trend instead of mean(diff(raw)).
        # The old raw drift could be strongly negative after one long initial
        # interval and collapse every forecast to 1 ms.
        log_history = np.log(np.maximum(history, self.min_interval_s))
        log_diffs = np.diff(log_history)
        recent_diffs = log_diffs[-min(log_diffs.size, 8):]
        trend = float(np.median(recent_diffs)) if recent_diffs.size else 0.0
        # Do not extrapolate a more-than-2x step change per future request.
        trend = float(np.clip(trend, -math.log(2.0), math.log(2.0)))

        center_log = float(np.median(log_history))
        last_log = float(log_history[-1])
        lower = max(float(np.percentile(history, 5)), self.min_interval_s)
        upper = max(float(np.percentile(history, 95)), lower)
        # Leave room for a real trend while keeping forecasts tied to observed
        # traffic rather than the absolute 1 ms safety floor.
        lower = max(min(lower, math.exp(center_log) * 0.5), self.min_interval_s)
        upper = max(upper, math.exp(center_log) * 2.0)

        forecast: List[float] = []
        for _ in range(max_steps):
            next_log = last_log + trend
            # Twenty percent median reversion plus trend damping prevents
            # recursive forecasts from diverging over a multi-second window.
            next_log = 0.8 * next_log + 0.2 * center_log
            next_interval = min(max(math.exp(next_log), lower), upper)
            forecast.append(next_interval)
            last_log = math.log(next_interval)
            trend *= 0.5
        return forecast

    @staticmethod
    def _estimate_ar1(diffs: np.ndarray) -> float:
        if diffs.size < 3:
            return 0.0
        x = diffs[:-1] - np.mean(diffs[:-1])
        y = diffs[1:] - np.mean(diffs[1:])
        denom = float(np.dot(x, x))
        if denom <= 1e-12:
            return 0.0
        return float(np.clip(np.dot(x, y) / denom, -0.95, 0.95))

    @property
    def rate_estimate(self) -> float:
        history = np.asarray(self._inter_arrivals_s, dtype=np.float64)
        if history.size == 0:
            return 0.0
        interval = max(float(np.mean(history[-min(history.size, 32):])),
                       self.min_interval_s)
        return 1.0 / interval
