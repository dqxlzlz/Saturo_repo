"""Backward-compatible re-export shim.

The canonical implementation has moved to
``saturo.predictors.request_arima_arrival_predictor``.
This module re-exports everything from there so that existing imports continue
to work unchanged.
"""
from __future__ import annotations

from saturo.predictors.request_arima_arrival_predictor import (  # noqa: F401
    ARIMAArrivalPredictor,
    ArrivalForecast,
)

__all__ = [
    "ARIMAArrivalPredictor",
    "ArrivalForecast",
]
