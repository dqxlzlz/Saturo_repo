"""Verify prepared workload hashes, actual token IDs, arrivals and BE task balance."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from .prepare import encode_many, log, sha256_file


def validate_workload(filename, tokenizer_dir=None, check_inputs=True):
    path = Path(filename)
    log(f"Verifying {path}")
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if sha256_file(path) != manifest["output"]["sha256"]:
        raise ValueError(f"Output SHA256 mismatch: {path}")
    if check_inputs:
        for entry in manifest["inputs"]:
            if sha256_file(entry["path"]) != entry["sha256"]:
                raise ValueError(f"Input SHA256 mismatch: {entry['path']}")
    tokenizer_dir = Path(tokenizer_dir or manifest["model_tokenizer"])
    for entry in manifest["tokenizer_files"]:
        if sha256_file(tokenizer_dir / entry["name"]) != entry["sha256"]:
            raise ValueError(f"Tokenizer SHA256 mismatch: {entry['name']}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    vocab_size = len(tokenizer)
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    count, previous, seen, sources, tasks, verified = 0, -1.0, set(), set(), Counter(), {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            count += 1
            if row["request_id"] in seen:
                raise ValueError("Duplicate request ID")
            seen.add(row["request_id"])
            encoded = [json.loads(row[key + "_token_ids"]) for key in ("prompt", "output")]
            for key, ids in zip(("prompt", "output"), encoded):
                if (not isinstance(ids, list) or not ids or len(ids) != int(row[key + "_len"])
                        or any(type(t) is not int or not 0 <= t < vocab_size for t in ids)):
                    raise ValueError(f"Invalid {key} IDs: {row['request_id']}")
            pair_key = (row["source_id"], row["prompt"], row["output"])
            if pair_key not in verified:
                actual = encode_many(tokenizer, [row["prompt"], row["output"]])
                verified[pair_key] = actual
            if verified[pair_key] != encoded:
                raise ValueError(f"Token IDs do not encode saved text: {row['request_id']}")
            offset = float(row["arrival_offset_s"])
            if not math.isfinite(offset) or offset < previous:
                raise ValueError("Invalid/nonmonotonic arrivals")
            if count == 1 and offset != 0:
                raise ValueError("First arrival must be zero")
            previous = offset
            sources.add(row["source_id"])
            if manifest["kind"] == "be":
                if offset != 0:
                    raise ValueError("BE must be ready immediately")
                tasks[row["dataset"]] += 1
            else:
                if int(row["azure_prompt_len"]) <= 0 or int(row["azure_output_len"]) <= 0:
                    raise ValueError("Invalid Azure target lengths")
            if count % 2000 == 0:
                log(f"Verified {count} rows in {path.name}")
    if count != manifest["count"] or len(sources) != manifest["unique_payloads"]:
        raise ValueError("Request/payload count mismatch")
    if manifest["kind"] == "be":
        if dict(tasks) != manifest["task_counts"] or len(sources) != count:
            raise ValueError("BE task balance or uniqueness mismatch")
    from saturo.runtime.workload import BE, RT, load_workload_csv
    loaded = load_workload_csv(str(path), BE if manifest["kind"] == "be" else RT)
    if len(loaded) != count or any(r.prompt_token_ids is None for r in loaded):
        raise ValueError("Saturo runtime did not load actual prompt IDs")
    return {"file": str(path), "count": count, "unique_payloads": len(sources),
            "task_counts": dict(tasks), "all_text_token_ids_verified": True,
            "runtime_loader_verified": True, "sha256": manifest["output"]["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--skip-input-hashes", action="store_true", help="For a relocated build without raw sources")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reports = [validate_workload(p, args.tokenizer_dir, not args.skip_input_hashes) for p in args.files]
    result = json.dumps(reports, indent=2)
    if args.report:
        args.report.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
