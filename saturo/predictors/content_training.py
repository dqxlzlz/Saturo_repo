"""Train TF-IDF/QRF from real prompts and observed/reference response prefixes."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from saturo.experiments.workloads.prepare import sha256_file
from saturo.experiments.setup.prepare_content_history import token_digest
from .request_length_predictor import (
    TFIDFQRFRemainingLengthPredictor, QRFRemainingLengthPredictor,
    build_length_feature_row,
)


def _ids(value, name):
    tokens = json.loads(value)
    if (not isinstance(tokens, list) or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)):
        raise ValueError(f"{name} must contain actual nonempty integer token IDs")
    return tokens


def read_content_history(source, observed_outputs=None):
    csv.field_size_limit(16 * 1024 * 1024)
    observed = None
    if observed_outputs:
        observed = {}
        with Path(observed_outputs).open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if not row.get("finished_s") or not row.get("generated_token_ids"):
                    continue
                tokens = _ids(row["generated_token_ids"], "generated_token_ids")
                if int(row["observed_tokens"]) != len(tokens):
                    raise ValueError("Observed output count does not match actual generated tokens")
                if row["request_id"] in observed:
                    raise ValueError("Duplicate observed request ID")
                observed[row["request_id"]] = tokens
    rows = []
    with Path(source).open(newline="", encoding="utf-8-sig") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            prompt = _ids(row.get("prompt_token_ids", "null"), "prompt_token_ids")
            if len(prompt) != int(row["prompt_len"]):
                raise ValueError("prompt_len does not match actual token IDs")
            if observed is not None:
                output = observed.get(row["request_id"])
                if output is None:
                    continue
            else:
                output = _ids(row.get("output_token_ids", "null"), "output_token_ids")
                if len(output) != int(row["output_len"]):
                    raise ValueError("output_len does not match reference tokens")
            rows.append(dict(prompt_token_ids=prompt, output_token_ids=output,
                             source_id=row.get("source_id") or row.get("request_id") or str(index),
                             prompt_group=token_digest(prompt)))
    return rows


def split_requests(rows, validation_fraction=.2, seed=20261101):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    groups = sorted({r["prompt_group"] for r in rows})
    if len(groups) < 4:
        raise ValueError("At least four distinct prompts are required")
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
    validation = set(groups[:count])
    return ([r for r in rows if r["prompt_group"] not in validation],
            [r for r in rows if r["prompt_group"] in validation])


def prefix_samples(rows, max_steps=8):
    if max_steps < 2:
        raise ValueError("max_steps must be at least 2")
    features, targets = [], []
    for row in rows:
        prompt, output = row["prompt_token_ids"], row["output_token_ids"]
        for t in np.unique(np.linspace(0, len(output) - 1,
                                       min(max_steps, len(output)), dtype=int)):
            features.append(dict(prompt_len=len(prompt), generated_len=int(t),
                                 prompt_token_ids=prompt, generated_token_ids=output[:t]))
            targets.append(len(output) - int(t))
    return features, np.asarray(targets, dtype=np.float32)


def metrics(truth, p50, p95, eta=.95):
    errors = truth - p95
    return {"samples": len(truth), "coverage_95": float(np.mean(truth <= p95)),
            "median_mae_tokens": float(np.mean(np.abs(truth - p50))),
            "pinball_95": float(np.mean(np.maximum(eta * errors, (eta - 1) * errors)))}


def train_content_qrf(source, output, *, observed_outputs=None, max_steps=8,
                      n_estimators=32, min_samples_leaf=8, max_features=2048,
                      validation_fraction=.2, seed=20261101, n_jobs=4):
    output = Path(output)
    if output.exists() or Path(str(output) + ".report.json").exists():
        raise ValueError(f"Refuse to overwrite {output}")
    if n_estimators < 1 or min_samples_leaf < 1 or max_features < 1 or n_jobs < 1:
        raise ValueError("Training sizes and CPU job count must be positive")
    manifest_file = Path(str(source) + ".manifest.json")
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else None
    if manifest and manifest["output"]["sha256"] != sha256_file(source):
        raise ValueError("Training source SHA256 mismatch")
    rows = read_content_history(source, observed_outputs)
    training, validation = split_requests(rows, validation_fraction, seed)
    train_x, train_y = prefix_samples(training, max_steps)
    val_x, val_y = prefix_samples(validation, max_steps)
    metadata = {
        "feature_mode": "tfidf", "training_source_sha256": sha256_file(source),
        "label_source": "model_observed_completions" if observed_outputs else "source_dataset_reference_responses",
        "mean_output_len": float(np.mean([len(r["output_token_ids"]) for r in training])),
        "train_requests": len(training), "validation_requests": len(validation),
        "split": "seeded groups by complete prompt token-ID hash; vectorizer fits training split only",
        "seed": seed, "tokenizer_files": manifest.get("tokenizer_files", []) if manifest else [],
        "online_features": "real prompt IDs + actually generated prefix IDs + five length features",
        "training_only_targets": "reference/observed output length minus prefix length; future output tokens excluded",
    }
    if observed_outputs:
        metadata["observed_outputs_sha256"] = sha256_file(observed_outputs)
    predictor = TFIDFQRFRemainingLengthPredictor(
        tfidf_max_features=max_features, n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=n_jobs, metadata=metadata)
    strings = [predictor._tokens_to_str(r["prompt_token_ids"] + r["generated_token_ids"]) for r in train_x]
    lengths = np.asarray([build_length_feature_row(prompt_len=r["prompt_len"],
                         generated_len=r["generated_len"]) for r in train_x], dtype=np.float32)
    predictor.fit_from_raw(strings, lengths, train_y)
    baseline = QRFRemainingLengthPredictor(n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=n_jobs)
    baseline.fit(lengths, train_y)
    content50 = predictor.predict_remaining_batch(val_x, eta=.5)
    content95 = predictor.predict_remaining_batch(val_x, eta=.95)
    baseline50 = baseline.predict_remaining_batch(val_x, eta=.5)
    baseline95 = baseline.predict_remaining_batch(val_x, eta=.95)
    matrix = predictor._remaining_batch_features(val_x)
    dimension = len(predictor.vectorizer.vocabulary_)
    report = {
        "metadata": metadata, "train_prefix_samples": len(train_x), "validation_prefix_samples": len(val_x),
        "content_features": dimension, "total_features": dimension + 5,
        "validation_rows_with_nonzero_content": int(np.count_nonzero(np.diff(matrix[:, :dimension].tocsr().indptr))),
        "prompt_group_overlap": len({r["prompt_group"] for r in training} & {r["prompt_group"] for r in validation}),
        "content_qrf": metrics(val_y, np.asarray(content50), np.asarray(content95)),
        "length_only_baseline": metrics(val_y, np.asarray(baseline50), np.asarray(baseline95)),
        "parameters": dict(seed=seed, trees=n_estimators, min_leaf=min_samples_leaf,
                           max_steps=max_steps, tfidf_max_features=max_features, validation_fraction=validation_fraction),
    }
    if report["prompt_group_overlap"] or not report["validation_rows_with_nonzero_content"]:
        raise ValueError("Invalid content training split/features")
    output.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(output)
    report["artifact_sha256"] = sha256_file(output)
    Path(str(output) + ".report.json").write_text(json.dumps(report, indent=2) + "\n")
    return predictor, report
