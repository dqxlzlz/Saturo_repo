"""Backward-compatible re-export shim.

The canonical implementation has moved to
``saturo.predictors.time_predictor``.
This module re-exports everything from there so that existing imports continue
to work unchanged.
"""
# ruff: noqa: F401
from __future__ import annotations

from saturo.predictors.time_predictor import (  # noqa: F401
    PiecewiseLinearModel,
    TileQuantizedLinearModel,
    RuntimeOverheadModel,
    TimePredictorProfileIdentity,
    TimePredictor,
    time_predictor_cache_path,
    profile_matches_identity,
    build_or_load_time_predictor,
    STATELESS_GROUP,
    STATEFUL_GROUP,
    TIME_PREDICTOR_PROFILE_VERSION,
    DEFAULT_PROFILE_WARMUP,
    DEFAULT_PROFILE_REPEAT,
    DEFAULT_PROFILE_NUM_LAYERS,
    DEFAULT_PROFILE_LAYER_START,
    DEFAULT_PROFILE_KV_CACHE_FRACTION,
    DEFAULT_LINEAR_PIECEWISE_WIDTH,
    DEFAULT_PROFILE_BLOCK_SIZE,
    PREFILL_REQ_LEN_RANGE,
    PREFILL_NUM_REQ_RANGE,
    PREFILL_MAX_TOKENS,
    DECODE_REQ_LEN_RANGE,
    DECODE_NUM_REQ_RANGE,
)

__all__ = [
    "PiecewiseLinearModel",
    "TileQuantizedLinearModel",
    "RuntimeOverheadModel",
    "TimePredictorProfileIdentity",
    "TimePredictor",
    "time_predictor_cache_path",
    "profile_matches_identity",
    "build_or_load_time_predictor",
    "STATELESS_GROUP",
    "STATEFUL_GROUP",
    "TIME_PREDICTOR_PROFILE_VERSION",
    "DEFAULT_PROFILE_WARMUP",
    "DEFAULT_PROFILE_REPEAT",
    "DEFAULT_PROFILE_NUM_LAYERS",
    "DEFAULT_PROFILE_LAYER_START",
    "DEFAULT_PROFILE_KV_CACHE_FRACTION",
    "DEFAULT_LINEAR_PIECEWISE_WIDTH",
    "DEFAULT_PROFILE_BLOCK_SIZE",
    "PREFILL_REQ_LEN_RANGE",
    "PREFILL_NUM_REQ_RANGE",
    "PREFILL_MAX_TOKENS",
    "DECODE_REQ_LEN_RANGE",
    "DECODE_NUM_REQ_RANGE",
]
