"""Source-format, matching, reproducibility and runtime-payload integration checks."""
import csv
import hashlib
import json
import math

import pytest

from saturo.experiments.workloads.prepare import (
    build_workload, match_trace, raw_pairs, read_trace, sample_candidates,
    sha256_file, summary_pair,
)
from saturo.experiments.workloads.validate import validate_workload


class ByteTokenizer:
    def __len__(self):
        return 256

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text) for text in texts]}


@pytest.fixture
def tokenizer_dir(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: ByteTokenizer())
    monkeypatch.setenv("RT_LEN_SCALE", "1")
    monkeypatch.setenv("RT_TRIM_TOP_FRAC", "0")
    path = tmp_path / "tokenizer"
    path.mkdir()
    (path / "tokenizer.json").write_text('{"test": "byte tokenizer"}')
    return path


def write_trace(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        writer.writerows(rows)


def test_iso_microseconds_and_bounded_prefix(tmp_path):
    path = tmp_path / "azure.csv"
    write_trace(path, [
        ["2024-05-12 00:00:00.001163+00:00", 7, 3],
        ["2024-05-12 00:00:00.041683+00:00", 8, 4],
        ["2024-05-12 00:00:00.041684+00:00", 9, 5],
        ["intentionally invalid beyond requested prefix", "bad", "bad"],
    ])
    rows = read_trace(path, limit=3)
    assert [r["arrival_offset_s"] for r in rows] == pytest.approx([0, .040520, .040521])
    assert rows[2]["azure_prompt_len"] == 9
    window = read_trace(path, start_row=1, limit=2)
    assert [r["azure_row"] for r in window] == [1, 2]
    assert window[1]["arrival_offset_s"] == pytest.approx(.000001)


@pytest.mark.parametrize("unit,stamps", [("s", [100, 100.5]), ("ms", [100000, 100500])])
def test_numeric_timestamp_units(tmp_path, unit, stamps):
    path = tmp_path / "trace.csv"
    write_trace(path, [[s, 2, 3] for s in stamps])
    assert read_trace(path, timestamp_unit=unit)[1]["arrival_offset_s"] == .5


@pytest.mark.parametrize("rows,error", [
    ([["2024-05-12T00:00:00Z", 0, 2]], "nonpositive"),
    ([["2024-05-12T00:00:01Z", 2, 2], ["2024-05-12T00:00:00Z", 2, 2]], "monotonic"),
    ([["2024-05-12T00:00:00", 2, 2]], "timezone"),
])
def test_bad_trace_is_not_silently_rewritten(tmp_path, rows, error):
    path = tmp_path / "trace.csv"
    write_trace(path, rows)
    with pytest.raises(ValueError, match=error):
        read_trace(path)


def test_all_parquet_shards_and_seeded_reservoir(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    for shard in range(6):
        rows = [{"conversation_id": f"{shard}:{i}", "conversation": [
            {"role": "user", "content": f"prompt {shard}:{i}"},
            {"role": "assistant", "content": "reference answer"},
        ]} for i in range(20)]
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / f"part-{shard}.parquet")
    all_pairs = list(raw_pairs(tmp_path, "lmsys"))
    assert len(all_pairs) == 120
    assert len({r["source_file"] for r in all_pairs}) == 6
    a, info = sample_candidates(tmp_path, "lmsys", 20, 42)
    b, _ = sample_candidates(tmp_path, "lmsys", 20, 42)
    c, _ = sample_candidates(tmp_path, "lmsys", 20, 99)
    assert a == b and a != c
    assert info["valid_source_pairs"] == 120
    assert len({r["source_file"] for r in a}) > 1


def test_match_is_nearest_in_both_length_dimensions():
    candidates = [dict(source_id=str(i), prompt="real", output="answer", prompt_len=n, output_len=m)
                  for i, (n, m) in enumerate([(10, 20), (20, 10), (100, 30), (300, 100), (10, 20)])]
    targets = [dict(azure_prompt_len=n, azure_output_len=m, arrival_offset_s=i)
               for i, (n, m) in enumerate([(19, 11), (16, 14), (150, 30), (2, 200), (10, 20)])]
    actual = match_trace(targets, candidates, seed=17)
    assert actual == match_trace(targets, candidates, seed=17)
    for target, chosen in zip(targets, actual):
        distances = [math.hypot(math.log1p(c["prompt_len"]) - math.log1p(target["azure_prompt_len"]),
                                math.log1p(c["output_len"]) - math.log1p(target["azure_output_len"]))
                     for c in candidates]
        assert chosen["length_match_distance"] == pytest.approx(min(distances))
    assert actual[0]["source_id"] == "1"


def test_longbench_preserves_context_query_and_reference():
    row = {"dataset": "qmsum", "context": "Alice: budget discussion", "input": "What was agreed?",
           "answers": ["Reduce spending."]}
    prompt, output = summary_pair(row, "qmsum")
    assert "Alice: budget discussion" in prompt and "What was agreed?" in prompt
    assert output == "Reduce spending."
    with pytest.raises(ValueError, match="Expected LongBench"):
        summary_pair(row, "govreport")


def be_sources(tmp_path):
    sources = {}
    for dataset, label in [("govreport", "gov_report"), ("multinews", "multi_news"), ("qmsum", "qmsum")]:
        path = tmp_path / f"{label}.jsonl"
        rows = [dict(context=f"真实文档 {label} {i}", input=f"Question {i}?", answers=[f"摘要 {i}"],
                     dataset=label, _id=f"original-{i}") for i in range(200)]
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        sources[dataset] = path
    return sources


def test_full_600_be_rows_reproduce_and_load_actual_ids(tmp_path, tokenizer_dir):
    sources = be_sources(tmp_path)
    paths = [tmp_path / name / "be_mix.csv" for name in ("first", "second")]
    for path in paths:
        manifest = build_workload(kind="be", model=tokenizer_dir, output=path, num_req=600,
                                  seed=20261101, **sources)
        assert manifest["task_counts"] == {"govreport": 200, "multinews": 200, "qmsum": 200}
        assert manifest["unique_payloads"] == 600
        report = validate_workload(path)
        assert report["count"] == 600 and report["runtime_loader_verified"]
    assert sha256_file(paths[0]) == sha256_file(paths[1])
    with pytest.raises(ValueError, match="overwrite"):
        build_workload(kind="be", model=tokenizer_dir, output=paths[0], num_req=600, **sources)
    with pytest.raises(ValueError, match="fewer"):
        build_workload(kind="be", model=tokenizer_dir, output=tmp_path / "too_many.csv", num_req=603, **sources)


def test_real_rt_payload_is_not_padded_to_azure_targets(tmp_path, tokenizer_dir):
    source = tmp_path / "sharegpt.json"
    source.write_text(json.dumps([{"id": "a", "conversations": [
        {"from": "human", "value": "actual prompt"}, {"from": "gpt", "value": "actual answer"}]}]))
    trace = tmp_path / "trace.csv"
    write_trace(trace, [["2024-05-12T00:00:00Z", 100, 50], ["2024-05-12T00:00:01Z", 20, 5]])
    path = tmp_path / "rt.csv"
    manifest = build_workload(kind="conv", model=tokenizer_dir, output=path, source=source,
                              azure_trace=trace, candidate_limit=0, num_req=2)
    assert manifest["unique_payloads"] == 1
    assert validate_workload(path)["count"] == 2
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["azure_prompt_len"] == "100"
    assert int(rows[0]["prompt_len"]) == len("actual prompt")
    assert json.loads(rows[0]["prompt_token_ids"]) == list(b"actual prompt")
    from saturo.runtime.workload import load_rt_workload_csv
    loaded = load_rt_workload_csv(str(path), num_req=2, target_rps=4)
    assert [r.arrival_offset_s for r in loaded] == [0, .5]
    assert loaded[0].prompt_token_ids == list(b"actual prompt")
    path.write_text(path.read_text() + "tampered")
    with pytest.raises(ValueError, match="SHA256"):
        validate_workload(path)


def test_inspection_accepts_long_real_content_csv_fields(tmp_path):
    from saturo.experiments.workloads.inspect_workloads import read_rows
    path = tmp_path / "long.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["prompt", "prompt_len", "output_len"])
        writer.writeheader()
        writer.writerow({"prompt": "x" * 200000, "prompt_len": 200000, "output_len": 1})
    csv.field_size_limit(131072)
    assert len(read_rows(str(path))[0]["prompt"]) == 200000


def test_manual_tokenizer_is_checked_against_pinned_files(tmp_path):
    from saturo.experiments.workloads.build_paper import REQUIRED_TOKENIZER_FILES, verify_tokenizer_files
    spec = {"repo_id": "test/model", "revision": "pinned"}
    expected = {**spec, "files": {}}
    for name in REQUIRED_TOKENIZER_FILES:
        data = (name + "\n").encode()
        (tmp_path / name).write_bytes(data)
        expected["files"][name] = {"bytes": len(data), "git_blob_sha1": hashlib.sha1(
            f"blob {len(data)}\0".encode() + data).hexdigest()}
    verify_tokenizer_files(tmp_path, spec, expected)
    changed = tmp_path / "tokenizer.json"
    changed.write_bytes(b"X" + changed.read_bytes()[1:])
    with pytest.raises(ValueError, match="hash differs"):
        verify_tokenizer_files(tmp_path, spec, expected)
    with pytest.raises(ValueError, match="revision differs"):
        verify_tokenizer_files(tmp_path, {**spec, "revision": "other"}, expected)
