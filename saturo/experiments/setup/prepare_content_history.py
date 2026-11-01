"""Prepare independent real-content QRF training data, excluding evaluation prompts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random

from saturo.experiments.workloads.prepare import (
    log, measure_pairs, raw_pairs, sha256_file, source_files, write_rows,
)


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def prepare(source, dataset, model, exclusions, output, *, count=6000,
            seed=20261101, max_total_tokens=65536):
    output = Path(output)
    if output.exists() or Path(str(output) + ".manifest.json").exists():
        raise ValueError(f"Refuse to overwrite {output}")
    if count < 1 or max_total_tokens < 2:
        raise ValueError("count and max_total_tokens must be positive")
    csv.field_size_limit(16 * 1024 * 1024)
    excluded_text, excluded_tokens = set(), set()
    for path in exclusions:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                excluded_text.add(hashlib.sha256(row["prompt"].encode()).digest())
                excluded_tokens.add(token_digest(json.loads(row["prompt_token_ids"])))
    rng = random.Random(seed)
    selected, seen_text, eligible = [], set(), 0
    capacity = count * 2
    for pair in raw_pairs(source, dataset):
        digest = hashlib.sha256(pair["prompt"].encode()).digest()
        if digest in excluded_text or digest in seen_text:
            continue
        seen_text.add(digest)
        eligible += 1
        if len(selected) < capacity:
            selected.append(pair)
        else:
            index = rng.randrange(eligible)
            if index < capacity:
                selected[index] = pair
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    selected.sort(key=lambda r: r["source_id"])
    tokenized = measure_pairs(selected, tokenizer, retain_ids=True)
    rng.shuffle(tokenized)
    rows, seen_tokens = [], set()
    for row in tokenized:
        digest = token_digest(row["prompt_token_ids"])
        if digest in excluded_tokens or digest in seen_tokens:
            continue
        if row["prompt_len"] + row["output_len"] > max_total_tokens:
            continue
        seen_tokens.add(digest)
        rows.append(dict(row, request_id=f"train_{dataset}_{len(rows):06d}", arrival_offset_s=0))
        if len(rows) == count:
            break
    if len(rows) < count:
        raise ValueError(f"Only {len(rows)} eligible tokenized pairs; requested {count}")
    write_rows(output, rows)
    manifest = {
        "schema_version": 1, "kind": "predictor_training", "count": count, "seed": seed,
        "source_dataset": dataset, "eligible_unique_raw_prompts": eligible,
        "max_total_tokens_filter": max_total_tokens, "prompt_overlap_with_exclusions": 0,
        "exclusion_policy": "exclude raw-text and token-identical evaluation prompts; unique training prompts",
        "label_source": "source_dataset_reference_responses; not model-observed completions",
        "inputs": [{"name": p.name, "sha256": sha256_file(p)} for p in source_files(source)],
        "excluded_workloads": [{"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for p in exclusions],
        "tokenizer_files": [{"name": p.name, "sha256": sha256_file(p)}
                            for p in sorted(Path(model).iterdir()) if p.is_file()],
        "output": {"name": output.name, "sha256": sha256_file(output)},
    }
    Path(str(output) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"Prepared {count} independent {dataset} training requests: {output}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", choices=("sharegpt", "lmsys"), required=True)
    parser.add_argument("--model", required=True, help="Local tokenizer directory")
    parser.add_argument("--exclude-workload", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    args = parser.parse_args()
    os.environ.setdefault("RAYON_NUM_THREADS", "8")
    print(json.dumps(prepare(args.source, args.dataset, args.model, args.exclude_workload,
                             args.output, count=args.count, seed=args.seed,
                             max_total_tokens=args.max_total_tokens), indent=2))


if __name__ == "__main__":
    main()
