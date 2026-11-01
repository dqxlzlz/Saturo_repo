"""Build real-content RT/BE CSVs from explicit local dataset snapshots."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import random
import sys


ENCODING = "raw text; add_special_tokens=False; no chat template; no truncation"
BE_TEMPLATES = {
    "govreport": "Summarize the following government report.\n\n{context}\n\nSummary:",
    "multinews": "Write a summary of the following news articles.\n\n{context}\n\nSummary:",
    "qmsum": ("Read the meeting transcript and answer the query with a summary.\n\n"
              "Transcript:\n{context}\n\nQuery: {input}\n\nSummary:"),
}


def log(message):
    print(message, file=sys.stderr, flush=True)


def sha256_file(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(value):
    values = [value] if isinstance(value, (str, Path)) else value
    paths = []
    for value in values:
        p = Path(value)
        if p.is_dir():
            paths.extend(x for x in p.rglob("*")
                         if x.is_file() and x.suffix in (".json", ".jsonl", ".parquet")
                         and ".cache" not in x.parts)
        elif p.is_file():
            paths.append(p)
        else:
            raise ValueError(f"Source does not exist: {p}")
    paths = sorted(set(paths), key=lambda p: p.as_posix())
    if not paths:
        raise ValueError(f"No JSON/JSONL/Parquet files in {value}")
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("Source basenames must be unique for stable source IDs")
    return paths


def records(filename):
    p = Path(filename)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(p).iter_batches(batch_size=512, use_threads=False):
            yield from batch.to_pylist()
    elif p.suffix == ".jsonl":
        with p.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif p.suffix == ".json":
        with p.open(encoding="utf-8") as stream:
            obj = json.load(stream)
        if not isinstance(obj, list):
            raise ValueError("JSON source must contain a list")
        yield from obj
    else:
        raise ValueError(f"Unsupported source type: {p}")


def conversation_pairs(row):
    messages = row.get("conversations", row.get("conversation", []))
    for a, b in zip(messages, messages[1:]):
        if (a.get("from", a.get("role")) in ("human", "user")
                and b.get("from", b.get("role")) in ("gpt", "assistant")):
            yield a.get("value", a.get("content")), b.get("value", b.get("content"))


def summary_pair(row, dataset):
    if "context" in row and "answers" in row:  # LongBench v1 schema.
        expected = {"govreport": "gov_report", "multinews": "multi_news", "qmsum": "qmsum"}
        if row.get("dataset", expected[dataset]) != expected[dataset]:
            raise ValueError(f"Expected LongBench {expected[dataset]}, got {row.get('dataset')}")
        answers = row["answers"]
        return (BE_TEMPLATES[dataset].format(context=row["context"], input=row.get("input", "")),
                answers[0] if isinstance(answers, list) and answers else None)
    if dataset == "govreport":
        return row.get("report"), row.get("summary")
    if dataset == "multinews":
        return row.get("document"), row.get("summary")
    transcript = row.get("meeting_transcripts", row.get("transcript", ""))
    if isinstance(transcript, list):
        transcript = "\n".join(str(x.get("content", x)) if isinstance(x, dict) else str(x)
                               for x in transcript)
    return f"{transcript}\n{row.get('query', '')}", row.get("answer")


def raw_pairs(source, dataset):
    for path in source_files(source):
        log(f"Reading {dataset}: {path.name}")
        for row_id, row in enumerate(records(path)):
            if dataset in ("sharegpt", "lmsys"):
                pairs = conversation_pairs(row)
            elif dataset == "qmsum" and "context" not in row and not row.get("query"):
                pairs = (summary_pair(dict(row, **query), dataset)
                         for query in row.get("specific_query_list", []) + row.get("general_query_list", []))
            else:
                pairs = [summary_pair(row, dataset)]
            for pair_id, (prompt, output) in enumerate(pairs):
                if not all(isinstance(x, str) and x.strip() for x in (prompt, output)):
                    continue
                yield {"prompt": prompt, "output": output,
                       "source_id": f"{dataset}:{path.name}:{row_id}:{pair_id}",
                       "source_file": path.name, "source_row": row_id, "source_pair": pair_id,
                       "original_id": str(row.get("_id", row.get("conversation_id", row.get("id", ""))))}


def sample_candidates(source, dataset, limit, seed):
    """Uniform reservoir over all valid adjacent pairs, across every shard."""
    rng = random.Random(seed)
    selected = []
    seen = 0
    for seen, pair in enumerate(raw_pairs(source, dataset), 1):
        if not limit or len(selected) < limit:
            selected.append(pair)
        else:
            index = rng.randrange(seen)
            if index < limit:
                selected[index] = pair
    if not selected:
        raise ValueError(f"No valid {dataset} pairs")
    selected.sort(key=lambda x: x["source_id"])
    return selected, {"valid_source_pairs": seen, "sampled_pairs": len(selected),
                      "candidate_limit": limit, "sampling": "uniform reservoir over valid pairs; 0=all",
                      "seed": seed}


def encode_many(tokenizer, texts):
    if callable(tokenizer):
        return tokenizer(texts, add_special_tokens=False, truncation=False,
                         padding=False, return_attention_mask=False)["input_ids"]
    return [tokenizer.encode(text, add_special_tokens=False) for text in texts]


def measure_pairs(pairs, tokenizer, batch_size=128, retain_ids=False):
    measured = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        encoded = encode_many(tokenizer, [x[k] for x in batch for k in ("prompt", "output")])
        for row, pi, oi in zip(batch, encoded[::2], encoded[1::2]):
            if not pi or not oi:
                continue
            item = dict(row, prompt_len=len(pi), output_len=len(oi))
            if retain_ids:
                item.update(prompt_token_ids=pi, output_token_ids=oi)
            measured.append(item)
        if start % (batch_size * 50) == 0:
            log(f"Tokenized {min(start + batch_size, len(pairs))}/{len(pairs)} pairs")
    if not measured:
        raise ValueError("No nonempty tokenized pairs")
    return measured


def token_pairs(filename, dataset, tokenizer, limit):
    # Small-source compatibility helper; large builds measure first and encode only selected IDs.
    pairs, _ = sample_candidates(filename, dataset, limit, 0)
    return measure_pairs(pairs, tokenizer, retain_ids=True)


def read_trace(trace, *, timestamp_col="TIMESTAMP", input_col="ContextTokens",
               output_col="GeneratedTokens", limit=0, timestamp_unit="iso", start_row=0):
    """Read only a requested contiguous prefix/window, without loading the full Azure CSV."""
    result = []
    origin = previous = None
    with Path(trace).open(newline="", encoding="utf-8-sig") as stream:
        reader = itertools.islice(csv.DictReader(stream), start_row,
                                  start_row + limit if limit else None)
        for i, row in enumerate(reader, start_row):
            stamp = row[timestamp_col]
            if timestamp_unit == "iso":
                current = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if current.tzinfo is None:
                    raise ValueError(f"Azure row {i}: ISO timestamps must include a timezone")
                if origin is None:
                    origin = current
                offset = (current - origin).total_seconds()
            else:
                current = float(stamp) * (0.001 if timestamp_unit == "ms" else 1)
                if origin is None:
                    origin = current
                offset = current - origin
            n, m = int(row[input_col]), int(row[output_col])
            if not math.isfinite(offset) or n <= 0 or m <= 0:
                raise ValueError(f"Azure row {i}: nonfinite timestamp or nonpositive length")
            if previous is not None and offset < previous:
                raise ValueError(f"Azure row {i}: timestamps are not monotonic")
            previous = offset
            result.append({"arrival_offset_s": offset, "azure_timestamp": stamp, "azure_row": i,
                           "azure_prompt_len": n, "azure_output_len": m})
    if not result or (limit and len(result) != limit):
        raise ValueError(f"Azure trace window has {len(result)} rows; requested {limit}")
    return result


def match_trace(rows, candidates, seed=0):
    import numpy as np
    from scipy.spatial import cKDTree

    groups = defaultdict(list)
    for index, row in enumerate(candidates):
        groups[row["prompt_len"], row["output_len"]].append(index)
    sizes = sorted(groups)
    coordinates = np.log1p(sizes)
    tree = cKDTree(coordinates)
    rng = random.Random(seed)
    matched = []
    for row in rows:
        point = np.log1p([row["azure_prompt_len"], row["azure_output_len"]])
        distance, _ = tree.query(point)
        # A stable tie break between equidistant length bins; sample actual payloads within the bin.
        near = tree.query_ball_point(point, float(distance) + 1e-10)
        index = min(near, key=lambda i: (float(np.sum((coordinates[i] - point) ** 2)), i))
        pair = candidates[rng.choice(groups[sizes[index]])]
        item = dict(pair, **row, length_match_distance=float(np.linalg.norm(coordinates[index] - point)))
        item["prompt_length_ratio"] = pair["prompt_len"] / row["azure_prompt_len"]
        item["output_length_ratio"] = pair["output_len"] / row["azure_output_len"]
        matched.append(item)
    return matched


def match_rt(trace, candidates, *, timestamp_col, input_col, output_col, limit=0,
             timestamp_unit="s", seed=0, start_row=0):
    return match_trace(read_trace(trace, timestamp_col=timestamp_col, input_col=input_col,
                                 output_col=output_col, limit=limit, timestamp_unit=timestamp_unit,
                                 start_row=start_row), candidates, seed)


def materialize_ids(rows, tokenizer, batch_size=128):
    unique = {r["source_id"]: r for r in rows}
    encoded = measure_pairs(list(unique.values()), tokenizer, batch_size, retain_ids=True)
    lookup = {r["source_id"]: r for r in encoded}
    for row in rows:
        pair = lookup[row["source_id"]]
        if (row["prompt_len"], row["output_len"]) != (pair["prompt_len"], pair["output_len"]):
            raise ValueError("Tokenizer lengths changed during materialization")
        row.update(prompt_token_ids=pair["prompt_token_ids"], output_token_ids=pair["output_token_ids"])


def write_rows(filename, rows):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with p.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(v, list) else v for k, v in row.items()})


def distribution(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "mean": float(a.mean())}


def package_versions():
    result = {"python": sys.version.split()[0]}
    for name in ("transformers", "tokenizers", "numpy", "scipy", "pyarrow", "huggingface-hub"):
        result[name] = importlib.metadata.version(name)
    return result


def build_workload(*, kind, model, output, source=None, azure_trace=None, govreport=None,
                   multinews=None, qmsum=None, candidate_limit=100000, num_req=8000,
                   seed=20261101, timestamp_unit="iso", timestamp_col="TIMESTAMP",
                   input_col="ContextTokens", output_col="GeneratedTokens", start_row=0,
                   batch_size=128, tokenizer_origin=None):
    output = Path(output)
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise ValueError(f"Refuse to overwrite existing output/manifest: {output}")
    if kind not in ("conv", "code", "be"):
        raise ValueError(f"Unknown workload kind: {kind}")
    if min(num_req, candidate_limit, start_row) < 0 or batch_size <= 0:
        raise ValueError("Counts must be nonnegative and batch_size positive")
    if kind == "be" and (num_req <= 0 or num_req % 3):
        raise ValueError("BE num_req must be positive and divisible by 3")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    inputs, sampling = [], {}
    if kind != "be":
        files = source_files(source)
        inputs = files + [Path(azure_trace)]
        trace = read_trace(azure_trace, timestamp_col=timestamp_col, input_col=input_col,
                           output_col=output_col, limit=num_req, timestamp_unit=timestamp_unit,
                           start_row=start_row)
        candidates, sampling = sample_candidates(files, "sharegpt" if kind == "conv" else "lmsys",
                                                 candidate_limit, seed)
        candidates = measure_pairs(candidates, tokenizer, batch_size)
        sampling["tokenized_pairs"] = len(candidates)
        rows = match_trace(trace, candidates, seed)
        del candidates
    else:
        rng = random.Random(seed)
        rows = []
        for dataset, path in zip(("govreport", "multinews", "qmsum"), (govreport, multinews, qmsum)):
            files = source_files(path)
            inputs.extend(files)
            pairs, info = sample_candidates(files, dataset, 0, seed)
            pairs = measure_pairs(pairs, tokenizer, batch_size)
            count = num_req // 3
            if len(pairs) < count:
                raise ValueError(f"{dataset} has {len(pairs)} pairs, fewer than {count}; no replacement")
            rows.extend(dict(r, dataset=dataset, arrival_offset_s=0.0) for r in rng.sample(pairs, count))
            sampling[dataset] = info
        rng.shuffle(rows)
    materialize_ids(rows, tokenizer, batch_size)
    for i, row in enumerate(rows):
        row["request_id"] = f"{'conv_sharegpt' if kind == 'conv' else 'code_lmsys' if kind == 'code' else 'be_mix'}_{i:06d}"
    log("Hashing input snapshots")
    manifest = {
        "schema_version": 1, "kind": kind, "count": len(rows), "seed": seed,
        "model_tokenizer": str(Path(model).resolve()), "tokenizer_origin": tokenizer_origin,
        "tokenizer_class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "encoding": ENCODING,
        "tokenizer_files": [{"name": p.name, "sha256": sha256_file(p)}
                            for p in sorted(Path(model).iterdir()) if p.is_file()],
        "inputs": [{"name": p.name, "path": str(p.resolve()), "bytes": p.stat().st_size,
                    "sha256": sha256_file(p)} for p in inputs],
        "sampling": sampling, "versions": package_versions(),
        "builder_sha256": sha256_file(__file__),
        "statistics": {key: distribution([r[key] for r in rows]) for key in ("prompt_len", "output_len")},
        "max_total_tokens": max(r["prompt_len"] + r["output_len"] for r in rows),
        "unique_payloads": len({r["source_id"] for r in rows}),
        "runtime": "prompt_token_ids are replayed; output_token_ids are source references, not teacher-forced generation",
    }
    if kind != "be":
        span = rows[-1]["arrival_offset_s"]
        manifest.update(
            timestamp_unit=timestamp_unit, start_row=start_row,
            matching="nearest Euclidean log1p(prompt_len,output_len) in seeded reservoir; reuse allowed",
            native_span_s=span, native_rps=len(rows) / span if span > 0 else None,
            arrival_scaling="not applied at build time; Saturo scales gaps by native_rps/target_rps at load time",
            length_matching={key: distribution([r[key] for r in rows])
                             for key in ("length_match_distance", "prompt_length_ratio", "output_length_ratio")},
        )
    else:
        manifest.update(task_counts={d: sum(r["dataset"] == d for r in rows)
                                     for d in ("govreport", "multinews", "qmsum")},
                        prompt_templates=BE_TEMPLATES,
                        order="one fixed seed-shuffled order; save and reuse across all tests; no duplication")
    write_rows(output, rows)
    manifest["output"] = {"name": output.name, "sha256": sha256_file(output), "bytes": output.stat().st_size}
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    log(f"Built {output}: {len(rows)} requests")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("conv", "code", "be"), required=True)
    parser.add_argument("--model", required=True, help="Local tokenizer directory; no model weights needed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", nargs="+", help="One or more JSON/JSONL/Parquet files, or a shard directory")
    parser.add_argument("--azure-trace")
    parser.add_argument("--timestamp-col", default="TIMESTAMP")
    parser.add_argument("--timestamp-unit", choices=("s", "ms", "iso"), default="iso")
    parser.add_argument("--input-col", default="ContextTokens")
    parser.add_argument("--output-col", default="GeneratedTokens")
    parser.add_argument("--govreport")
    parser.add_argument("--multinews")
    parser.add_argument("--qmsum")
    parser.add_argument("--candidate-limit", type=int, default=100000, help="Uniform reservoir size for RT; 0=all")
    parser.add_argument("--num-req", type=int, default=8000, help="RT rows; 0=all. BE must be positive/divisible by 3")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    try:
        manifest = build_workload(**vars(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": manifest["output"], "count": manifest["count"]}, indent=2))


if __name__ == "__main__":
    main()
