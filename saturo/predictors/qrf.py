"""Required QRF artifacts for Saturo RT remaining-length prediction."""

from __future__ import annotations

import os
import json
from pathlib import Path

QRF_MODEL_TYPES = ("qrf", "tfidf_qrf")


def resolve_qrf_path(path=None):
    return (path or os.environ.get("SATURO_HISTORY_LENGTH_PREDICTOR")
            or os.environ.get("VLLM_DECODE_SATURATION_MODEL_LEN_PREDICTOR")
            or os.environ.get("VLLM_REQUEST_MODEL_LEN_PREDICTOR")
            or os.environ.get("VLLM_DECODE_SATURATION_LENGTH_PREDICTOR")
            or os.environ.get("VLLM_REQUEST_LENGTH_PREDICTOR"))


def require_qrf(predictor):
    if predictor is None:
        raise RuntimeError(
            "Decode saturation requires a QRF remaining-length predictor. "
            "Set SATURO_HISTORY_LENGTH_PREDICTOR to a trained QRF artifact.")
    model_type = getattr(predictor, "model_type", None)
    if model_type not in QRF_MODEL_TYPES:
        raise ValueError(f"Decode saturation requires QRF; got {model_type!r}")
    return predictor


def load_required_qrf(path=None):
    path = resolve_qrf_path(path)
    if not path:
        return require_qrf(None)
    from saturo.predictors.request_length_predictor import load_length_predictor
    return require_qrf(load_length_predictor(path))


def uses_content(predictor):
    return getattr(getattr(predictor, "config", None), "feature_mode", "len") in ("tfidf", "token_ids")


def validate_qrf_workload(predictor, args):
    """Keep paper content inference explicit; preserve length-only dummy runs."""
    get = args.get if isinstance(args, dict) else lambda k, d=None: getattr(args, k, d)
    content = uses_content(predictor)
    if content and get("kv_handoff", "dummy") != "real":
        raise ValueError("Content QRF requires --kv-handoff real; dummy D-pool prompts are synthetic. "
                         "Use a length-only artifact for dummy profiling.")
    csv_path = get("rt_workload_csv")
    manifest_path = Path(str(csv_path) + ".manifest.json") if csv_path else None
    if get("kv_handoff", "dummy") != "real" or manifest_path is None or not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") not in ("conv", "code"):
        return
    if not content:
        raise ValueError("Prepared real-content workloads require a content QRF, not feature_mode=len; "
                         "train with train_qrf --feature-mode tfidf")
    saved = {r["name"]: r["sha256"] for r in getattr(predictor, "metadata", {}).get("tokenizer_files", [])}
    expected = {r["name"]: r["sha256"] for r in manifest["tokenizer_files"]}
    names = [name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
             if name in expected]
    if not names or any(saved.get(name) != expected[name] for name in names):
        raise ValueError("Content QRF tokenizer provenance does not match this workload; retrain for this tokenizer")
