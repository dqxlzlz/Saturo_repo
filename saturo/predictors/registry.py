"""Predictor artifact registry.

Manages loading and auto-training of remaining-length predictors.
Artifacts are keyed by (dataset_name, model_type, feature_mode) and stored
under saturo/predictors/artifacts/.

Typical usage (decode-pool integration)
---------------------------------------
    from saturo.predictors import load_or_train

    # First call trains & caches; subsequent calls load from disk
    predictor = load_or_train(
        dataset_name="wildchat_0000",
        parquet_path="data/wildchat/0000.parquet",
        model_type="tfidf_qrf",
    )
    remaining = predictor.predict_remaining(
        eta=0.90,
        prompt_token_ids=request.prompt_token_ids,
        generated_token_ids=request.output_token_ids_so_far,
    )
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from saturo.predictors.request_length_predictor import (
    FEATURE_MODE_LEN,
    FEATURE_MODE_TOKEN_IDS,
    FEATURE_MODE_TFIDF,
    GBDTRemainingLengthPredictor,
    QRFRemainingLengthPredictor,
    TFIDFQRFRemainingLengthPredictor,
    RemainingLengthPredictor,
    _hashed_count_features,
    build_length_feature_row,
    create_length_predictor,
    load_length_predictor,
    normalize_length_predictor_type,
)

# ---------------------------------------------------------------------------
# Artifact directory (relative to this file → project-level canonical store)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def artifact_path(
    dataset_name: str,
    model_type: str = "qrf",
    feature_mode: str = FEATURE_MODE_TOKEN_IDS,
) -> Path:
    """Return the canonical .joblib path for a given (dataset, model, feature) key."""
    mtype = normalize_length_predictor_type(model_type)
    return ARTIFACTS_DIR / f"{dataset_name}__{mtype}__{feature_mode}.joblib"


def list_artifacts() -> list[Path]:
    """Return all .joblib artifact files in the artifacts directory."""
    return sorted(ARTIFACTS_DIR.glob("*.joblib"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_wildchat_records(
    parquet_path: Path,
    max_samples: int = 1000,
    min_output_tokens: int = 5,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Load (prompt_token_ids, output_token_ids) pairs from a WildChat parquet."""
    import tiktoken
    import pyarrow.parquet as pq

    enc = tiktoken.get_encoding("cl100k_base")
    pf = pq.ParquetFile(parquet_path)
    records: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for rg in range(pf.num_row_groups):
        for row in pf.read_row_group(rg, columns=["conversation", "toxic"]).to_pylist():
            if row.get("toxic"):
                continue
            conv = list(row.get("conversation") or [])
            for idx, msg in enumerate(conv):
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                out_text = (msg.get("content") or "").strip()
                if not out_text:
                    continue
                pp = [
                    f"{m.get('role','')}: {(m.get('content') or '').strip()}"
                    for m in conv[:idx]
                    if isinstance(m, dict) and (m.get("content") or "").strip()
                ]
                p_text = "\n".join(pp)
                if not p_text:
                    continue
                p_ids = tuple(enc.encode(p_text))
                o_ids = tuple(enc.encode(out_text))
                if len(o_ids) < min_output_tokens:
                    continue
                records.append((p_ids, o_ids))
                if len(records) >= max_samples:
                    return records
    return records


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def _build_features(
    records: list[tuple[tuple[int, ...], tuple[int, ...]]],
    feature_mode: str,
    hash_bins: int,
    max_steps_per_request: Optional[int],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand records into one row per decode step (hash-bin and len modes).

    Returns (X, y, output_lens_per_row, generated_len_per_row).
    For tfidf mode use `_build_tfidf_features()` instead.
    """
    rng = np.random.default_rng(seed)
    xs, ys, ols, gls = [], [], [], []
    for p_ids, o_ids in records:
        O = len(o_ids)
        if max_steps_per_request is not None and O > max_steps_per_request:
            steps = sorted(rng.choice(O, size=max_steps_per_request, replace=False).tolist())
        else:
            steps = range(O)
        for t in steps:
            base = np.asarray(build_length_feature_row(prompt_len=len(p_ids), generated_len=t), np.float32)
            if feature_mode == FEATURE_MODE_TOKEN_IDS:
                ph = _hashed_count_features(p_ids, hash_bins)
                gh = _hashed_count_features(o_ids[:t], hash_bins)
                row = np.concatenate([base, ph, gh])
            else:
                row = base
            xs.append(row)
            ys.append(float(O - t))
            ols.append(O)
            gls.append(t)
    return (
        np.array(xs, np.float32),
        np.array(ys, np.float32),
        np.array(ols, np.int32),
        np.array(gls, np.int32),
    )


def _build_tfidf_features(
    records: list[tuple[tuple[int, ...], tuple[int, ...]]],
    max_steps_per_request: Optional[int],
    seed: int = 42,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build TF-IDF training data.

    X = TF-IDF(prompt_token_ids + generated_token_ids[:t]) — prompt 与已生成 token 拼接
    y = O - t  — 目标是剩余长度（不是总长）

    Returns (augmented_strings, length_features, y_remaining, output_lens, generated_lens).
    """
    rng = np.random.default_rng(seed)
    aug_strings: list[str] = []
    len_rows, y_remaining, ols, gls = [], [], [], []
    for p_ids, o_ids in records:
        O = len(o_ids)
        if max_steps_per_request is not None and O > max_steps_per_request:
            steps = sorted(rng.choice(O, size=max_steps_per_request, replace=False).tolist())
        else:
            steps = range(O)
        for t in steps:
            # 拼接 prompt + 已生成 token
            aug_ids = p_ids + o_ids[:t]
            aug_strings.append(" ".join(f"t{tid}" for tid in aug_ids))
            len_rows.append(np.asarray(build_length_feature_row(prompt_len=len(p_ids), generated_len=t), np.float32))
            y_remaining.append(float(O - t))
            ols.append(O)
            gls.append(t)
    return (
        aug_strings,
        np.array(len_rows, np.float32),
        np.array(y_remaining, np.float32),
        np.array(ols, np.int32),
        np.array(gls, np.int32),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    parquet_path: str | Path,
    model_type: str = "tfidf_qrf",
    feature_mode: Optional[str] = None,
    max_samples: int = 1000,
    max_steps_per_request: Optional[int] = 200,
    quantiles: Sequence[float] = (0.80, 0.90, 0.95),
    hash_bins: int = 128,
    tfidf_max_features: int = 50000,
    n_estimators: int = 64,
    min_samples_leaf: int = 20,
    seed: int = 42,
    verbose: bool = True,
    records: Optional[list[tuple[tuple[int, ...], tuple[int, ...]]]] = None,
) -> RemainingLengthPredictor:
    """Train a predictor from a WildChat parquet file or pre-loaded records.

    If *records* is provided, uses them directly (skipping parquet loading).
    """
    parquet_path = Path(parquet_path)
    mtype = normalize_length_predictor_type(model_type)
    # Default feature_mode per model type
    if feature_mode is None:
        feature_mode = FEATURE_MODE_TFIDF if mtype == "tfidf_qrf" else FEATURE_MODE_TOKEN_IDS

    if records is None:
        if verbose:
            print(f"  加载数据集: {parquet_path}  (max_samples={max_samples})")
        t0 = time.perf_counter()
        records = _load_wildchat_records(parquet_path, max_samples=max_samples)
        if verbose:
            ol_mean = float(np.mean([len(o) for _, o in records]))
            print(f"  记录: {len(records)} 条  output_len mean={ol_mean:.0f}  耗时={time.perf_counter()-t0:.1f}s")
    else:
        if verbose:
            ol_mean = float(np.mean([len(o) for _, o in records]))
            print(f"  使用预加载记录: {len(records)} 条  output_len mean={ol_mean:.0f}")

    # ── TF-IDF 模式 ──
    if mtype == TFIDFQRFRemainingLengthPredictor.model_type:
        if verbose:
            print(f"  构建 TF-IDF 特征 (max_features={tfidf_max_features}, max_steps={max_steps_per_request})...")
        t0 = time.perf_counter()
        prompt_strings, len_feats, y_remaining, _, _ = _build_tfidf_features(
            records, max_steps_per_request, seed=seed
        )
        if verbose:
            print(f"  数据点: {len(prompt_strings):,}  特征维度(tfidf+len)={tfidf_max_features + 5}  耗时={time.perf_counter()-t0:.1f}s")

        predictor = TFIDFQRFRemainingLengthPredictor(
            tfidf_max_features=tfidf_max_features,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
            metadata={
                "dataset": str(parquet_path),
                "max_samples": max_samples,
                "n_records": len(records),
                "n_rows": len(prompt_strings),
                "tfidf_max_features": tfidf_max_features,
            },
        )
        if verbose:
            print(f"  训练 tfidf_qrf (TF-IDF fit + QRF fit)...")
        t0 = time.perf_counter()
        predictor.fit_from_raw(prompt_strings, len_feats, y_remaining)
        if verbose:
            print(f"  训练完成  耗时={time.perf_counter()-t0:.1f}s")
        return predictor

    # ── Hash-bin 模式（qrf / gbdt）──
    if verbose:
        print(f"  构建特征 ({feature_mode}, hash_bins={hash_bins}, max_steps={max_steps_per_request})...")
    t0 = time.perf_counter()
    X, y, _, _ = _build_features(records, feature_mode, hash_bins, max_steps_per_request, seed=seed)
    if verbose:
        print(f"  数据点: {len(X):,}  特征维度: {X.shape[1]}  耗时={time.perf_counter()-t0:.1f}s")

    predictor = create_length_predictor(
        mtype,
        feature_mode=feature_mode,
        hash_bins=hash_bins,
        quantiles=list(quantiles),
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
        metadata={
            "dataset": str(parquet_path),
            "max_samples": max_samples,
            "n_records": len(records),
            "n_rows": len(X),
            "feature_dim": X.shape[1],
        },
    )

    if verbose:
        print(f"  训练 {mtype}...")
    t0 = time.perf_counter()
    predictor.fit(X, y)
    if verbose:
        print(f"  训练完成  耗时={time.perf_counter()-t0:.1f}s")
    return predictor


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def load_or_train(
    dataset_name: str,
    parquet_path: str | Path,
    model_type: str = "tfidf_qrf",
    feature_mode: Optional[str] = None,
    max_samples: int = 1000,
    max_steps_per_request: Optional[int] = 200,
    quantiles: Sequence[float] = (0.80, 0.90, 0.95),
    hash_bins: int = 128,
    tfidf_max_features: int = 50000,
    n_estimators: int = 64,
    min_samples_leaf: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> RemainingLengthPredictor:
    """Load artifact from disk if it exists; otherwise train and save it.

    The artifact is stored at:
        saturo/predictors/artifacts/{dataset_name}__{model_type}__{feature_mode}.joblib
    """
    path = artifact_path(dataset_name, model_type, feature_mode)
    if path.exists():
        if verbose:
            print(f"[predictor] 加载已有模型: {path}")
        return load_length_predictor(path)

    if verbose:
        print(f"[predictor] 未找到模型，开始训练 ({dataset_name}, {model_type}, {feature_mode})...")
    predictor = train(
        parquet_path=parquet_path,
        model_type=model_type,
        feature_mode=feature_mode,
        max_samples=max_samples,
        max_steps_per_request=max_steps_per_request,
        quantiles=quantiles,
        hash_bins=hash_bins,
        tfidf_max_features=tfidf_max_features,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
        verbose=verbose,
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    predictor.save(path)
    if verbose:
        print(f"[predictor] 模型已保存: {path}")
    return predictor
