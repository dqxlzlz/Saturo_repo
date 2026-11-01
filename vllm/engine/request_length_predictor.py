"""Backward-compatible re-export shim.

The canonical implementation has moved to
``saturo.predictors.request_length_predictor``.
This module re-exports everything from there so that existing imports continue
to work unchanged.
"""
# ruff: noqa: F401
from __future__ import annotations

from saturo.predictors.request_length_predictor import (  # noqa: F401
    FEATURE_MODE_LEN,
    FEATURE_MODE_TOKEN_IDS,
    SUPPORTED_FEATURE_MODES,
    LENGTH_FEATURE_NAMES,
    LENGTH_PREDICTOR_SCHEMA_VERSION,
    LengthPredictorConfig,
    RemainingLengthPredictor,
    QRFRemainingLengthPredictor,
    GBDTRemainingLengthPredictor,
    build_length_feature_row,
    build_feature_row,
    build_feature_matrix,
    create_length_predictor,
    load_length_predictor,
    normalize_length_predictor_type,
    _hashed_count_features,
    _as_token_tuple,
)

__all__ = [
    "FEATURE_MODE_LEN",
    "FEATURE_MODE_TOKEN_IDS",
    "SUPPORTED_FEATURE_MODES",
    "LENGTH_FEATURE_NAMES",
    "LENGTH_PREDICTOR_SCHEMA_VERSION",
    "LengthPredictorConfig",
    "RemainingLengthPredictor",
    "QRFRemainingLengthPredictor",
    "GBDTRemainingLengthPredictor",
    "build_length_feature_row",
    "build_feature_row",
    "build_feature_matrix",
    "create_length_predictor",
    "load_length_predictor",
    "normalize_length_predictor_type",
    "_hashed_count_features",
    "_as_token_tuple",
]
