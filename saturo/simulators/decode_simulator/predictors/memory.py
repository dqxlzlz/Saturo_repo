"""Progressive RT memory prediction for D-pool saturation.

This module implements the paper-side estimator used by the Progressive
strategy. Oracle prediction lives in ``simulator.py`` because it needs direct
access to the trace future.
"""

import math
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np

from saturo.predictors.request_arima_arrival_predictor import ARIMAArrivalPredictor
from saturo.predictors.request_length_predictor import (
    RemainingLengthPredictor, TFIDFQRFRemainingLengthPredictor)

_NATIVE_MODEL_REMAINING = (
    RemainingLengthPredictor.predict_remaining,
    TFIDFQRFRemainingLengthPredictor.predict_remaining,
)
from saturo.saturation import (
    sampling_offsets as _core_sampling_offsets,
)

from ..config import ModelConfig, SaturationConfig
from ..request import Request


def sampling_offsets(window_s: float, tau_s: float, max_samples: int) -> List[float]:
    """Backward-compatible wrapper around saturo.saturation.sampling_offsets.

    The *max_samples* parameter is retained for API compatibility but is
    no longer used (the canonical implementation uses tau-step offsets only).
    """
    del max_samples
    return _core_sampling_offsets(window_s, tau_s)

class DPoolSaturationPredictor:
    """Online predictors used by Progressive D-pool saturation."""

    def __init__(self, model_config: ModelConfig, sat_config: SaturationConfig,
                 total_gpu_blocks: int = 0):
        self.model_config = model_config
        self.sat_config = sat_config
        self.total_gpu_blocks = int(total_gpu_blocks)

        self._recent_output_lens: List[int] = []
        self._recent_prompt_lens: List[int] = []
        self._recent_request_lens: List[Tuple[int, int]] = []
        self._output_len_estimate = 512.0
        self._prompt_len_estimate = 1024.0
        self._arima_arrival_predictor = ARIMAArrivalPredictor()
        self._model_len_predictor: Optional[Any] = None
        self._model_len_predictor_type: Optional[str] = None

    # ------------------------------------------------------------------
    # Online updates
    # ------------------------------------------------------------------

    def seed_from_trace(self, requests: Iterable[Request]):
        for req in requests:
            self.update_model_len_history(
                req.prompt_len, req.target_output_len)

    def update_arrival_rate(
        self,
        current_time_s: float,
        new_arrivals: int,
        window_s: float,
    ):
        del current_time_s, new_arrivals, window_s
        # ARIMA arrival prediction is trained from exact RT arrival timestamps
        # via observe_rt_arrival_time(). This method remains for older callers.

    def observe_rt_arrival_time(self, arrival_time_s: float):
        self._arima_arrival_predictor.observe_arrival(arrival_time_s)

    def update_model_len_history(self, prompt_len: int, output_len: int):
        prompt_len = max(int(prompt_len), 1)
        output_len = max(int(output_len), 1)
        self._recent_request_lens.append((prompt_len, output_len))
        if len(self._recent_request_lens) > 2000:
            self._recent_request_lens = self._recent_request_lens[-2000:]
        self.update_prompt_len_distribution(prompt_len)
        self.update_output_len_distribution(output_len)

    def update_output_len_distribution(self, output_len: int):
        self._recent_output_lens.append(max(int(output_len), 1))
        if len(self._recent_output_lens) > 1000:
            self._recent_output_lens = self._recent_output_lens[-1000:]
        self._refresh_length_estimates()

    def update_prompt_len_distribution(self, prompt_len: int):
        self._recent_prompt_lens.append(max(int(prompt_len), 1))
        if len(self._recent_prompt_lens) > 1000:
            self._recent_prompt_lens = self._recent_prompt_lens[-1000:]
        self._refresh_length_estimates()

    def _refresh_length_estimates(self):
        if self._recent_output_lens:
            self._output_len_estimate = float(np.mean(self._recent_output_lens))
        if self._recent_prompt_lens:
            self._prompt_len_estimate = float(np.mean(self._recent_prompt_lens))

    # ------------------------------------------------------------------
    # Q_eta / active RT model length
    # ------------------------------------------------------------------

    def configure_model_len_predictor(
        self,
        predictor_path: Optional[str],
        expected_type: Optional[str] = None,
        *,
        predictor=None,  # 直接注入已实例化的预测器（优先于路径）
    ):
        from saturo.predictors.request_length_predictor import (
            load_length_predictor,
            normalize_length_predictor_type,
        )

        from saturo.predictors.qrf import require_qrf, resolve_qrf_path
        if predictor is not None:
            require_qrf(predictor)
            # 直接使用调用方传入的预测器实例
            model_type = normalize_length_predictor_type(
                getattr(predictor, "model_type", expected_type or "qrf"))
            self._model_len_predictor = predictor
            self._model_len_predictor_type = model_type
            return

        predictor_path = resolve_qrf_path(predictor_path)
        if not predictor_path:
            require_qrf(None)
        predictor = require_qrf(load_length_predictor(predictor_path))
        model_type = normalize_length_predictor_type(
            getattr(predictor, "model_type", ""))
        if expected_type is not None:
            expected = normalize_length_predictor_type(expected_type)
            if model_type != expected:
                raise ValueError(
                    f"length predictor artifact type {model_type!r} does not "
                    f"match expected type {expected!r}")

        self._model_len_predictor = predictor
        self._model_len_predictor_type = model_type

    def predict_active_remaining_len(self, request: Request) -> int:
        from saturo.predictors.qrf import require_qrf
        predictor = require_qrf(self._model_len_predictor)
        predicted = predictor.predict_remaining(
            eta=self.sat_config.model_len_quantile,
            prompt_len=max(int(request.prompt_len), 1),
            generated_len=max(int(request.tokens_generated), 0),
            # Pass actual token IDs when available (WildChat parquet trace).
            # Falls back to length-only features for Azure CSV traces (None).
            prompt_token_ids=request.prompt_token_ids,
            generated_token_ids=request.generated_token_ids,
        )
        return max(int(predicted), 0)

    def predict_active_remaining_lens(self, requests: Iterable[Request]) -> List[int]:
        """Batch the same observed features through the bounded QRF cache.

        Scalar overrides retain their original behavior. Neither request IDs
        nor output targets are prediction inputs; token-aware models receive
        only the prompt and the already-generated prefix, exactly as above.
        """
        requests = list(requests)
        if not requests:
            return []
        if (getattr(self.predict_active_remaining_len, "__func__", None)
                is not _NATIVE_ACTIVE_REMAINING):
            return [self.predict_active_remaining_len(req) for req in requests]
        from saturo.predictors.qrf import require_qrf
        predictor = require_qrf(self._model_len_predictor)
        batch = getattr(predictor, "predict_remaining_batch", None)
        scalar = getattr(predictor, "predict_remaining", None)
        if (not callable(batch)
                or getattr(scalar, "__func__", None) not in _NATIVE_MODEL_REMAINING):
            return [self.predict_active_remaining_len(req) for req in requests]
        rows = [{
            "prompt_len": max(int(req.prompt_len), 1),
            "generated_len": max(int(req.tokens_generated), 0),
            "prompt_token_ids": req.prompt_token_ids,
            "generated_token_ids": req.generated_token_ids,
        } for req in requests]
        predictions = list(batch(rows, eta=self.sat_config.model_len_quantile))
        if len(predictions) != len(requests):
            raise RuntimeError("Remaining-length batch has an invalid row count")
        return [max(int(predicted), 0) for predicted in predictions]

    # ------------------------------------------------------------------
    # ARIMA-predicted new RT arrivals
    # ------------------------------------------------------------------

    def predict_arima_new_arrivals(
        self,
        now_s: float,
        window_s: float,
    ) -> List[Tuple[float, int, int]]:
        arrival_offsets = self._arima_arrival_predictor.predict_arrival_offsets(
            window_s, now_s=now_s)
        if not arrival_offsets:
            return []

        count = len(arrival_offsets)
        prompt_lens = self._predicted_length_sketch(
            self._recent_prompt_lens, count, self._prompt_len_estimate)
        output_lens = self._predicted_length_sketch(
            self._recent_output_lens, count, self._output_len_estimate)
        return [
            (
                float(offset),
                max(1, int(round(prompt_len))),
                max(1, int(round(output_len))),
            )
            for offset, prompt_len, output_len
            in zip(arrival_offsets, prompt_lens, output_lens)
        ]

    def _predicted_length_sketch(
        self,
        samples: List[int],
        count: int,
        fallback: float,
    ) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.float64)
        if not samples:
            return np.full(count, fallback, dtype=np.float64)

        rho = self.sat_config.model_len_quantile
        if count == 1:
            quantiles = np.array([rho], dtype=np.float64)
        else:
            # A predicted burst is a set of requests, not count copies of the
            # same high-percentile request. We assign a deterministic quantile
            # sketch from rho down to the median, and give earlier arrivals the
            # longer lengths because they have more time to become resident.
            quantiles = np.linspace(rho, 0.5, count, dtype=np.float64)
        return np.asarray(np.percentile(samples, quantiles * 100.0), dtype=np.float64)

    @property
    def ewma_arrival_rate(self) -> Optional[float]:
        return self._arima_arrival_predictor.rate_estimate

    @property
    def output_len_estimate(self) -> float:
        return self._output_len_estimate

    @property
    def prompt_len_estimate(self) -> float:
        return self._prompt_len_estimate


_NATIVE_ACTIVE_REMAINING = DPoolSaturationPredictor.predict_active_remaining_len


# Backward-compatible alias for older simulator tests and scripts.
MemoryPredictor = DPoolSaturationPredictor
