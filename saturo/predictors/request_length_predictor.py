"""Remaining generation length predictors for scheduler-side planning.

The predictors estimate Saturo's Q_eta(p_r, o_r^t): the eta-quantile of the
remaining output length for a request given its prompt and current generated
prefix. Two implementations are provided:

* QRFRemainingLengthPredictor: one random-forest model can serve many eta.
* GBDTRemainingLengthPredictor: one quantile-GBDT model is trained per eta.

Runtime code should usually load an artifact with `load_length_predictor()` and
call `predict_remaining(...)`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


LENGTH_PREDICTOR_SCHEMA_VERSION = 1

FEATURE_MODE_LEN = "len"
FEATURE_MODE_TOKEN_IDS = "token_ids"
FEATURE_MODE_TFIDF = "tfidf"  # JITServe-style: TF-IDF(prompt) + length → predict total output len
SUPPORTED_FEATURE_MODES = (FEATURE_MODE_LEN, FEATURE_MODE_TOKEN_IDS, FEATURE_MODE_TFIDF)

LENGTH_FEATURE_NAMES = (
    "prompt_tokens",
    "log_prompt_tokens",
    "generated_tokens",
    "log_generated_tokens",
    "generated_prompt_ratio",
)


def _as_token_tuple(token_ids: Optional[Sequence[int]]) -> tuple[int, ...]:
    if token_ids is None:
        return ()
    return tuple(int(token_id) for token_id in token_ids)


def _hashed_count_features(token_ids: Sequence[int], hash_bins: int) -> np.ndarray:
    features = np.zeros(hash_bins, dtype=np.float32)
    if not token_ids:
        return features
    for token_id in token_ids:
        features[int(token_id) % hash_bins] += 1.0
    return np.log1p(features)


def build_length_feature_row(
    *,
    prompt_len: int,
    generated_len: int,
) -> list[float]:
    return [
        float(prompt_len),
        math.log1p(float(prompt_len)),
        float(generated_len),
        math.log1p(float(generated_len)),
        float(generated_len) / max(1.0, float(prompt_len)),
    ]


def build_feature_row(
    *,
    feature_mode: str,
    hash_bins: int,
    prompt_len: Optional[int] = None,
    generated_len: Optional[int] = None,
    prompt_token_ids: Optional[Sequence[int]] = None,
    generated_token_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Build the fixed-width feature row used by both predictor types.

    For `feature_mode="len"`, only lengths are used. For
    `feature_mode="token_ids"`, the length features are concatenated with hashed
    bag/count features over prompt token ids and already-generated token ids.
    """

    if feature_mode not in SUPPORTED_FEATURE_MODES:
        raise ValueError(f"unknown feature mode: {feature_mode}")

    prompt_ids = _as_token_tuple(prompt_token_ids)
    generated_ids = _as_token_tuple(generated_token_ids)
    if prompt_len is None:
        prompt_len = len(prompt_ids)
    if generated_len is None:
        generated_len = len(generated_ids)
    if prompt_len is None or generated_len is None:
        raise ValueError("prompt_len and generated_len must be known")

    base = np.asarray(
        build_length_feature_row(
            prompt_len=int(prompt_len),
            generated_len=int(generated_len),
        ),
        dtype=np.float32,
    )
    if feature_mode == FEATURE_MODE_LEN:
        return base

    prompt_hash = _hashed_count_features(prompt_ids, hash_bins)
    generated_hash = _hashed_count_features(generated_ids, hash_bins)
    return np.concatenate([base, prompt_hash, generated_hash]).astype(np.float32)


def build_feature_matrix(
    rows: Iterable[Mapping[str, Any]],
    *,
    feature_mode: str,
    hash_bins: int,
) -> np.ndarray:
    return np.vstack([
        build_feature_row(
            feature_mode=feature_mode,
            hash_bins=hash_bins,
            prompt_len=row.get("prompt_len"),
            generated_len=row.get("generated_len"),
            prompt_token_ids=row.get("prompt_token_ids"),
            generated_token_ids=row.get("generated_token_ids"),
        )
        for row in rows
    ]).astype(np.float32)


@dataclass
class LengthPredictorConfig:
    model_type: str
    feature_mode: str = FEATURE_MODE_LEN
    hash_bins: int = 128

    def __post_init__(self) -> None:
        if self.feature_mode not in SUPPORTED_FEATURE_MODES:
            raise ValueError(f"unknown feature mode: {self.feature_mode}")
        if self.hash_bins <= 0:
            raise ValueError("hash_bins must be positive")

    def feature_names(self) -> list[str]:
        names = list(LENGTH_FEATURE_NAMES)
        if self.feature_mode == FEATURE_MODE_TOKEN_IDS:
            names.extend([f"prompt_token_hash_{i}" for i in range(self.hash_bins)])
            names.extend([f"generated_token_hash_{i}" for i in range(self.hash_bins)])
        elif self.feature_mode == FEATURE_MODE_TFIDF:
            names = ["tfidf_..."] + list(LENGTH_FEATURE_NAMES)  # tfidf dim varies
        return names

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "feature_mode": self.feature_mode,
            "hash_bins": self.hash_bins,
            "feature_names": self.feature_names(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LengthPredictorConfig":
        return cls(
            model_type=str(data["model_type"]),
            feature_mode=str(data.get("feature_mode", FEATURE_MODE_LEN)),
            hash_bins=int(data.get("hash_bins", 128)),
        )


class RemainingLengthPredictor(ABC):
    model_type: str
    _REMAINING_CACHE_MAX_ENTRIES = 1024
    _REMAINING_CACHE_MAX_BYTES = 4 * 1024 * 1024

    def __init__(
        self,
        *,
        feature_mode: str = FEATURE_MODE_LEN,
        hash_bins: int = 128,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = LengthPredictorConfig(
            model_type=self.model_type,
            feature_mode=feature_mode,
            hash_bins=hash_bins,
        )
        self.metadata = metadata or {}
        self._prediction_revision = 0

    def _feature_row(
        self,
        *,
        prompt_len: Optional[int] = None,
        generated_len: Optional[int] = None,
        prompt_token_ids: Optional[Sequence[int]] = None,
        generated_token_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        return build_feature_row(
            feature_mode=self.config.feature_mode,
            hash_bins=self.config.hash_bins,
            prompt_len=prompt_len,
            generated_len=generated_len,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
        )

    def predict_remaining(
        self,
        *,
        eta: float = 0.95,
        prompt_len: Optional[int] = None,
        generated_len: Optional[int] = None,
        prompt_token_ids: Optional[Sequence[int]] = None,
        generated_token_ids: Optional[Sequence[int]] = None,
    ) -> int:
        x = np.asarray([
            self._feature_row(
                prompt_len=prompt_len,
                generated_len=generated_len,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=generated_token_ids,
            )
        ], dtype=np.float32)
        pred = self.predict_quantile(x, eta)[0]
        return max(0, int(math.ceil(float(pred))))

    def _remaining_batch_features(self, rows: Sequence[Mapping[str, Any]]):
        return np.vstack([
            self._feature_row(
                prompt_len=row.get("prompt_len"),
                generated_len=row.get("generated_len"),
                prompt_token_ids=row.get("prompt_token_ids"),
                generated_token_ids=row.get("generated_token_ids"),
            ) for row in rows
        ]).astype(np.float32)

    def _remaining_cache_identity(self):
        inner = getattr(self, "_inner_qrf", self)
        forest = getattr(inner, "rf", None)
        vectorizer = getattr(self, "vectorizer", None)
        estimators = getattr(forest, "estimators_", ())
        values = (self.config.model_type, self.config.feature_mode,
                  self.config.hash_bins,
                  getattr(self, "_prediction_revision", 0),
                  getattr(inner, "_prediction_revision", 0),
                  len(estimators), getattr(self, "_vectorizer_fitted", None))
        # Retain the references, rather than only ids, to prevent id reuse
        # after artifact replacement from making old cache entries valid.
        refs = (forest, estimators, getattr(inner, "leaf_values", None),
                vectorizer, getattr(vectorizer, "vocabulary_", None))
        return values, refs

    def _ensure_remaining_cache(self):
        identity = self._remaining_cache_identity()
        previous = getattr(self, "_remaining_cache_identity_value", None)
        if not hasattr(self, "_remaining_cache_stats"):
            self._remaining_cache_stats = dict(
                hits=0, misses=0, model_calls=0, model_rows=0, invalidations=0)
        changed = (previous is None or previous[0] != identity[0]
                   or any(a is not b for a, b in zip(previous[1], identity[1])))
        if changed:
            if previous is not None:
                self._remaining_cache_stats["invalidations"] += 1
            self._remaining_cache = OrderedDict()
            self._remaining_cache_bytes = 0
            self._remaining_cache_identity_value = identity

    def clear_remaining_prediction_cache(self) -> None:
        """Invalidate cached batch predictions before modifying model weights."""
        self._prediction_revision = getattr(self, "_prediction_revision", 0) + 1
        if hasattr(self, "_remaining_cache"):
            self._remaining_cache.clear()
            self._remaining_cache_bytes = 0
            self._remaining_cache_identity_value = None
            self._remaining_cache_stats["invalidations"] += 1

    def remaining_batch_cache_info(self) -> dict[str, int]:
        self._ensure_remaining_cache()
        return {**self._remaining_cache_stats,
                "entries": len(self._remaining_cache),
                "feature_bytes": self._remaining_cache_bytes}

    def predict_remaining_batch(
        self, rows: Sequence[Mapping[str, Any]], *, eta: float = 0.95,
    ) -> list[int]:
        """Predict unchanged QRF features once, in bounded feature-key storage.

        Keys contain the actual dense or sparse feature values presented to
        the model, so token/TF-IDF features and eta retain their semantics.
        The scalar API stays uncached for callers measuring raw inference.
        """
        if not rows:
            return []
        self._ensure_remaining_cache()
        x = self._remaining_batch_features(rows)
        sparse = hasattr(x, "getrow")
        cache = self._remaining_cache
        results = [None] * len(rows)
        missing = OrderedDict()
        for index in range(len(rows)):
            if sparse:
                row = x.getrow(index)
                payload = (row.indices.tobytes(), row.data.tobytes())
            else:
                payload = (np.ascontiguousarray(x[index]).tobytes(),)
            key = (float(eta), str(x.dtype), x.shape[1], *payload)
            if key in cache:
                results[index] = cache[key]
                cache.move_to_end(key)
                self._remaining_cache_stats["hits"] += 1
            else:
                missing.setdefault(key, []).append(index)
                self._remaining_cache_stats["misses"] += 1
        if missing:
            indices = [positions[0] for positions in missing.values()]
            predictions = self.predict_quantile(x[indices], eta)
            if len(predictions) != len(indices):
                raise RuntimeError("Remaining-length batch has an invalid row count")
            self._remaining_cache_stats["model_calls"] += 1
            self._remaining_cache_stats["model_rows"] += len(indices)
            for (key, positions), prediction in zip(missing.items(), predictions):
                value = max(0, int(math.ceil(float(prediction))))
                for index in positions:
                    results[index] = value
                size = sum(len(part) for part in key[3:])
                if size <= self._REMAINING_CACHE_MAX_BYTES:
                    cache[key] = value
                    self._remaining_cache_bytes += size
                    while (len(cache) > self._REMAINING_CACHE_MAX_ENTRIES
                           or self._remaining_cache_bytes
                           > self._REMAINING_CACHE_MAX_BYTES):
                        evicted, _ = cache.popitem(last=False)
                        self._remaining_cache_bytes -= sum(
                            len(part) for part in evicted[3:])
        return results

    @abstractmethod
    def predict_quantile(self, x: np.ndarray, eta: float) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "RemainingLengthPredictor":
        raise NotImplementedError

    @abstractmethod
    def _model_to_artifact(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": LENGTH_PREDICTOR_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "metadata": self.metadata,
            "model": self._model_to_artifact(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_artifact(), path)


class QRFRemainingLengthPredictor(RemainingLengthPredictor):
    model_type = "qrf"

    def __init__(
        self,
        *,
        n_estimators: int = 64,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
        feature_mode: str = FEATURE_MODE_LEN,
        hash_bins: int = 128,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            feature_mode=feature_mode,
            hash_bins=hash_bins,
            metadata=metadata,
        )
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.leaf_values: list[dict[int, np.ndarray]] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "QRFRemainingLengthPredictor":
        self.clear_remaining_prediction_cache()
        self.rf.fit(x, y)
        train_leaves = self.rf.apply(x)
        self.leaf_values = []
        for tree_idx in range(train_leaves.shape[1]):
            buckets: dict[int, list[float]] = defaultdict(list)
            for leaf_id, target in zip(train_leaves[:, tree_idx], y):
                buckets[int(leaf_id)].append(float(target))
            self.leaf_values.append(
                {leaf: np.asarray(values, dtype=np.float32) for leaf, values in buckets.items()}
            )
        return self

    def predict_quantile(self, x: np.ndarray, eta: float) -> np.ndarray:
        leaves = self.rf.apply(x)
        preds = np.empty(x.shape[0], dtype=np.float32)
        for row_idx in range(x.shape[0]):
            parts = []
            for tree_idx, leaf_id in enumerate(leaves[row_idx]):
                values = self.leaf_values[tree_idx].get(int(leaf_id))
                if values is not None:
                    parts.append(values)
            preds[row_idx] = np.quantile(np.concatenate(parts), eta) if parts else np.nan
        return preds

    def _model_to_artifact(self) -> dict[str, Any]:
        return {
            "rf": self.rf,
            "leaf_values": self.leaf_values,
            "params": {
                "n_estimators": self.n_estimators,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            },
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "QRFRemainingLengthPredictor":
        config = LengthPredictorConfig.from_dict(artifact["config"])
        model_data = artifact["model"]
        params = model_data.get("params", {})
        predictor = cls(
            n_estimators=int(params.get("n_estimators", 64)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=int(params.get("n_jobs", -1)),
            feature_mode=config.feature_mode,
            hash_bins=config.hash_bins,
            metadata=dict(artifact.get("metadata", {})),
        )
        predictor.rf = model_data["rf"]
        predictor.leaf_values = model_data["leaf_values"]
        return predictor


class GBDTRemainingLengthPredictor(RemainingLengthPredictor):
    model_type = "gbdt"

    def __init__(
        self,
        *,
        quantiles: Sequence[float] = (0.95,),
        learning_rate: float = 0.05,
        max_iter: int = 200,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int = 42,
        feature_mode: str = FEATURE_MODE_LEN,
        hash_bins: int = 128,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            feature_mode=feature_mode,
            hash_bins=hash_bins,
            metadata=metadata,
        )
        self.quantiles = tuple(float(q) for q in quantiles)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.models: dict[float, HistGradientBoostingRegressor] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GBDTRemainingLengthPredictor":
        self.clear_remaining_prediction_cache()
        self.models = {}
        for eta in self.quantiles:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=eta,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                random_state=self.random_state,
            )
            model.fit(x, y)
            self.models[eta] = model
        return self

    def _nearest_eta(self, eta: float) -> float:
        if not self.models:
            raise RuntimeError("GBDT predictor has no fitted quantile models")
        for trained_eta in self.models:
            if abs(trained_eta - eta) < 1e-9:
                return trained_eta
        return min(self.models, key=lambda trained_eta: abs(trained_eta - eta))

    def predict_quantile(self, x: np.ndarray, eta: float) -> np.ndarray:
        trained_eta = self._nearest_eta(float(eta))
        return self.models[trained_eta].predict(x)

    def _model_to_artifact(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "params": {
                "quantiles": list(self.quantiles),
                "learning_rate": self.learning_rate,
                "max_iter": self.max_iter,
                "max_leaf_nodes": self.max_leaf_nodes,
                "min_samples_leaf": self.min_samples_leaf,
                "l2_regularization": self.l2_regularization,
                "random_state": self.random_state,
            },
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "GBDTRemainingLengthPredictor":
        config = LengthPredictorConfig.from_dict(artifact["config"])
        model_data = artifact["model"]
        params = model_data.get("params", {})
        predictor = cls(
            quantiles=tuple(float(q) for q in params.get("quantiles", (0.95,))),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_iter=int(params.get("max_iter", 200)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            l2_regularization=float(params.get("l2_regularization", 0.0)),
            random_state=int(params.get("random_state", 42)),
            feature_mode=config.feature_mode,
            hash_bins=config.hash_bins,
            metadata=dict(artifact.get("metadata", {})),
        )
        predictor.models = {
            float(eta): model for eta, model in model_data["models"].items()
        }
        return predictor


class TFIDFQRFRemainingLengthPredictor(RemainingLengthPredictor):
    """TF-IDF + QRF remaining-length predictor.

    X = TF-IDF(prompt_token_ids + generated_token_ids[:t]) concatenated with
        length features (prompt_len, log_prompt_len, gen_len, log_gen_len, ratio)
    y = remaining output tokens (O - t)

    At each decode step t, the model predicts how many tokens remain to be
    generated.  Prompt tokens are converted to pseudo-words ("t<id>") before
    TF-IDF so the vocabulary captures token co-occurrence patterns.

    Runtime usage (from decode-pool)::

        predictor = load_length_predictor(artifact_path)
        remaining = predictor.predict_remaining(
            eta=0.90,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated_ids,
        )
    """
    model_type = "tfidf_qrf"

    def __init__(
        self,
        *,
        tfidf_max_features: int = 50000,
        n_estimators: int = 64,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Skip parent __init__ since we manage config ourselves
        self.config = LengthPredictorConfig(
            model_type=self.model_type,
            feature_mode=FEATURE_MODE_TFIDF,
            hash_bins=128,  # unused, kept for config compat
        )
        self.metadata = metadata or {}
        self.tfidf_max_features = tfidf_max_features
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        self._inner_qrf = QRFRemainingLengthPredictor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            feature_mode=FEATURE_MODE_LEN,  # inner QRF doesn’t need feature building
        )
        self._vectorizer_fitted = False

    @staticmethod
    def _tokens_to_str(token_ids: Sequence[int]) -> str:
        """Convert token id sequence to a pseudo-word string for TF-IDF."""
        return " ".join(f"t{tid}" for tid in token_ids)

    def fit_from_raw(
        self,
        aug_strings: list[str],
        length_features: np.ndarray,
        y_remaining: np.ndarray,
    ) -> "TFIDFQRFRemainingLengthPredictor":
        """Fit from augmented prompt+generated strings + length features.

        Args:
            aug_strings: list of "t<id> t<id> ..." strings
                         (prompt token ids + generated token ids[:t] concatenated)
            length_features: (N, 5) array of length features
            y_remaining: (N,) remaining output length targets (O - t)
        """
        from scipy.sparse import hstack as sp_hstack, csr_matrix

        self.clear_remaining_prediction_cache()

        X_tfidf = self.vectorizer.fit_transform(aug_strings)  # sparse CSR
        X = sp_hstack([X_tfidf, csr_matrix(length_features)], format="csr")
        self._inner_qrf.fit(X, y_remaining)
        self._vectorizer_fitted = True
        self.metadata.setdefault("mean_output_len", float(np.mean(
            np.asarray(y_remaining) + np.asarray(length_features)[:, 2])))
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TFIDFQRFRemainingLengthPredictor":
        """Standard interface: x must already be the combined (tfidf + length) matrix.

        NOTE: vectorizer must have been fitted separately before calling this.
        """
        self.clear_remaining_prediction_cache()
        self._inner_qrf.fit(x, y)
        return self

    def predict_quantile(self, x: np.ndarray, eta: float) -> np.ndarray:
        """Predict eta-quantile of **remaining output length** given combined features."""
        return self._inner_qrf.predict_quantile(x, eta)

    def _remaining_batch_features(self, rows: Sequence[Mapping[str, Any]]):
        from scipy.sparse import hstack as sp_hstack, csr_matrix

        if not self._vectorizer_fitted:
            raise RuntimeError("Content QRF requires a fitted TF-IDF vectorizer")
        features, strings = [], []
        for row in rows:
            prompt_ids = row.get("prompt_token_ids")
            generated_ids = row.get("generated_token_ids")
            if prompt_ids is None or not len(prompt_ids):
                raise ValueError("Content QRF requires actual prompt_token_ids")
            if int(row.get("generated_len") or 0) and generated_ids is None:
                raise ValueError("Content QRF requires actually generated prefix token IDs")
            prompt_len = int(row["prompt_len"]) if row.get("prompt_len") is not None else len(prompt_ids)
            generated_len = (int(row["generated_len"]) if row.get("generated_len") is not None
                             else len(generated_ids or []))
            if prompt_len != len(prompt_ids) or generated_len != len(generated_ids or []):
                raise ValueError("Content QRF lengths must match prompt and observed-prefix token IDs")
            features.append(build_length_feature_row(
                prompt_len=prompt_len, generated_len=generated_len))
            strings.append(self._tokens_to_str(
                list(prompt_ids) + list(generated_ids or []))
                if prompt_ids is not None else "")
        tfidf = self.vectorizer.transform(strings)
        return sp_hstack([tfidf, csr_matrix(
            np.asarray(features, dtype=np.float32))], format="csr")

    def predict_remaining(
        self,
        *,
        eta: float = 0.95,
        prompt_len: Optional[int] = None,
        generated_len: Optional[int] = None,
        prompt_token_ids: Optional[Sequence[int]] = None,
        generated_token_ids: Optional[Sequence[int]] = None,
    ) -> int:
        """Predict remaining tokens via TF-IDF(prompt + generated_so_far)."""
        X = self._remaining_batch_features([dict(
            prompt_len=prompt_len, generated_len=generated_len,
            prompt_token_ids=prompt_token_ids, generated_token_ids=generated_token_ids)])
        predicted_remaining = float(self._inner_qrf.predict_quantile(X, eta)[0])
        return max(0, int(math.ceil(predicted_remaining)))

    def _model_to_artifact(self) -> dict[str, Any]:
        return {
            "vectorizer": self.vectorizer,
            "vectorizer_fitted": self._vectorizer_fitted,
            "inner_qrf": self._inner_qrf._model_to_artifact(),
            "params": {
                "tfidf_max_features": self.tfidf_max_features,
                "n_estimators": self.n_estimators,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            },
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "TFIDFQRFRemainingLengthPredictor":
        model_data = artifact["model"]
        params = model_data.get("params", {})
        predictor = cls(
            tfidf_max_features=int(params.get("tfidf_max_features", 8000)),
            n_estimators=int(params.get("n_estimators", 64)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=int(params.get("n_jobs", -1)),
            metadata=dict(artifact.get("metadata", {})),
        )
        predictor.vectorizer = model_data["vectorizer"]
        predictor._vectorizer_fitted = bool(model_data.get("vectorizer_fitted", False))
        # Restore inner QRF weights
        inner_data = model_data["inner_qrf"]
        predictor._inner_qrf.rf = inner_data["rf"]
        predictor._inner_qrf.leaf_values = inner_data["leaf_values"]
        return predictor


def normalize_length_predictor_type(model_type: str) -> str:
    normalized = model_type.strip().lower().replace("-", "_")
    if normalized in ("gbdt", "gbpt", "gbrt", "gradient_boosting",
                      "gradient_boosted_trees"):
        return GBDTRemainingLengthPredictor.model_type
    if normalized in ("tfidf_qrf", "tfidf", "jitserve"):
        return TFIDFQRFRemainingLengthPredictor.model_type
    return normalized


def create_length_predictor(
    model_type: str,
    *,
    feature_mode: str = FEATURE_MODE_LEN,
    hash_bins: int = 128,
    quantiles: Sequence[float] = (0.95,),
    metadata: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> RemainingLengthPredictor:
    model_type = normalize_length_predictor_type(model_type)
    if model_type == QRFRemainingLengthPredictor.model_type:
        return QRFRemainingLengthPredictor(
            feature_mode=feature_mode,
            hash_bins=hash_bins,
            metadata=metadata,
            n_estimators=int(kwargs.get("n_estimators", kwargs.get("trees", 64))),
            min_samples_leaf=int(kwargs.get("min_samples_leaf", kwargs.get("min_leaf", 20))),
            random_state=int(kwargs.get("random_state", 42)),
            n_jobs=int(kwargs.get("n_jobs", -1)),
        )
    if model_type == GBDTRemainingLengthPredictor.model_type:
        return GBDTRemainingLengthPredictor(
            feature_mode=feature_mode,
            hash_bins=hash_bins,
            metadata=metadata,
            quantiles=quantiles,
            learning_rate=float(kwargs.get("learning_rate", 0.05)),
            max_iter=int(kwargs.get("max_iter", 200)),
            max_leaf_nodes=int(kwargs.get("max_leaf_nodes", 31)),
            min_samples_leaf=int(kwargs.get("min_samples_leaf", kwargs.get("min_leaf", 20))),
            l2_regularization=float(kwargs.get("l2_regularization", 0.0)),
            random_state=int(kwargs.get("random_state", 42)),
        )
    if model_type == TFIDFQRFRemainingLengthPredictor.model_type:
        return TFIDFQRFRemainingLengthPredictor(
            tfidf_max_features=int(kwargs.get("tfidf_max_features", 8000)),
            n_estimators=int(kwargs.get("n_estimators", kwargs.get("trees", 64))),
            min_samples_leaf=int(kwargs.get("min_samples_leaf", kwargs.get("min_leaf", 20))),
            random_state=int(kwargs.get("random_state", 42)),
            n_jobs=int(kwargs.get("n_jobs", -1)),
            metadata=metadata,
        )
    raise ValueError(f"unknown length predictor type: {model_type}")


def load_length_predictor(path: str | Path) -> RemainingLengthPredictor:
    artifact = joblib.load(path)

    # Backward compatibility with the first experimental QRF artifact.
    if "qrf" in artifact and "config" not in artifact:
        predictor = QRFRemainingLengthPredictor(
            feature_mode=FEATURE_MODE_LEN,
            metadata=dict(artifact.get("metadata", {})),
        )
        predictor.rf = artifact["qrf"]["rf"]
        predictor.leaf_values = artifact["qrf"]["leaf_values"]
        return predictor

    config = LengthPredictorConfig.from_dict(artifact["config"])
    model_type = normalize_length_predictor_type(config.model_type)
    if model_type == QRFRemainingLengthPredictor.model_type:
        return QRFRemainingLengthPredictor.from_artifact(artifact)
    if model_type == GBDTRemainingLengthPredictor.model_type:
        artifact = dict(artifact)
        artifact["config"] = {
            **dict(artifact["config"]),
            "model_type": GBDTRemainingLengthPredictor.model_type,
        }
        return GBDTRemainingLengthPredictor.from_artifact(artifact)
    if model_type == TFIDFQRFRemainingLengthPredictor.model_type:
        return TFIDFQRFRemainingLengthPredictor.from_artifact(artifact)
    raise ValueError(f"unknown length predictor type: {config.model_type}")
