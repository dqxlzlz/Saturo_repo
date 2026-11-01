"""Train the paper's length-only QRF from completed historical requests."""

from pathlib import Path

import numpy as np

from saturo.history import load_trace_rows
from saturo.predictors.request_length_predictor import (
    FEATURE_MODE_LEN, QRFRemainingLengthPredictor, build_feature_matrix)


def train_history_qrf(source: str, output: Path, *, window_s: float = 1800,
                      model: str = "", max_steps: int = 16,
                      n_estimators: int = 32, min_requests: int = 32):
    rows = load_trace_rows(source, window_s=window_s, model=model)
    if max_steps < 2 or len(rows) < max(min_requests, 1):
        raise ValueError("insufficient completed history or max_steps < 2")
    features, targets = [], []
    for row in rows:
        output_len = int(row["output_len"])
        if output_len < 1:
            raise ValueError("history output lengths must be positive")
        for generated in np.unique(np.linspace(
                0, output_len, min(max_steps, output_len + 1), dtype=int)):
            features.append({"prompt_len": int(row["prompt_len"]),
                             "generated_len": int(generated)})
            targets.append(output_len - int(generated))
    predictor = QRFRemainingLengthPredictor(
        n_estimators=n_estimators, min_samples_leaf=2, n_jobs=1,
        feature_mode=FEATURE_MODE_LEN,
        metadata={"history_source": source, "requests": len(rows),
                  "feature_mode": FEATURE_MODE_LEN, "window_s": window_s})
    predictor.fit(build_feature_matrix(
        features, feature_mode=FEATURE_MODE_LEN, hash_bins=128),
        np.asarray(targets, dtype=np.float32))
    predictor.save(output)
    return predictor
