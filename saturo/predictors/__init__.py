"""Predictors used by Saturo runtimes and simulators.

Public imports are resolved lazily so optional ML dependencies used by the
request-length predictor do not block the arrival-time or runtime predictors.
"""

from importlib import import_module


_EXPORTS = {
    # request_length_predictor
    "FEATURE_MODE_LEN": "saturo.predictors.request_length_predictor",
    "FEATURE_MODE_TOKEN_IDS": "saturo.predictors.request_length_predictor",
    "FEATURE_MODE_TFIDF": "saturo.predictors.request_length_predictor",
    "GBDTRemainingLengthPredictor":
        "saturo.predictors.request_length_predictor",
    "LengthPredictorConfig": "saturo.predictors.request_length_predictor",
    "QRFRemainingLengthPredictor":
        "saturo.predictors.request_length_predictor",
    "TFIDFQRFRemainingLengthPredictor":
        "saturo.predictors.request_length_predictor",
    "RemainingLengthPredictor": "saturo.predictors.request_length_predictor",
    "build_feature_matrix": "saturo.predictors.request_length_predictor",
    "build_feature_row": "saturo.predictors.request_length_predictor",
    "build_length_feature_row":
        "saturo.predictors.request_length_predictor",
    "create_length_predictor": "saturo.predictors.request_length_predictor",
    "load_length_predictor": "saturo.predictors.request_length_predictor",
    "normalize_length_predictor_type":
        "saturo.predictors.request_length_predictor",
    # registry
    "artifact_path": "saturo.predictors.registry",
    "load_or_train": "saturo.predictors.registry",
    "train_predictor": ("saturo.predictors.registry", "train"),
    # arrival predictor
    "ARIMAArrivalPredictor":
        "saturo.predictors.request_arima_arrival_predictor",
    "ArrivalForecast": "saturo.predictors.request_arima_arrival_predictor",
    # runtime predictor
    "PiecewiseLinearModel": "saturo.predictors.time_predictor",
    "RuntimeOverheadModel": "saturo.predictors.time_predictor",
    "TimePredictorProfileIdentity": "saturo.predictors.time_predictor",
    "TimePredictor": "saturo.predictors.time_predictor",
    "time_predictor_cache_path": "saturo.predictors.time_predictor",
    "profile_matches_identity": "saturo.predictors.time_predictor",
    "build_or_load_time_predictor": "saturo.predictors.time_predictor",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if isinstance(target, tuple):
        module_name, attribute_name = target
    else:
        module_name, attribute_name = target, name
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
