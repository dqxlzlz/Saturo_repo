"""Prepared payload validation, model compatibility and scheduler/worker integration."""
import csv
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from saturo.runtime.workload import (
    BE, RT, TPUT, load_rt_workload_csv, load_workload_csv,
    preflight_workload_manifests, validate_workload_manifest,
)


@pytest.fixture(autouse=True)
def no_reshape(monkeypatch):
    monkeypatch.setenv("RT_LEN_SCALE", "1")
    monkeypatch.setenv("RT_TRIM_TOP_FRAC", "0")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    return path


def row(**changes):
    return dict(request_id="r1", prompt_len=3, output_len=2,
                arrival_offset_s=0, prompt="real prompt", output="reference",
                prompt_token_ids=[2, 4, 6], output_token_ids=[8, 10],
                source_id="sharegpt:source:1:0", azure_prompt_len=100,
                azure_output_len=50) | changes


def manifest(path, model, kind="conv", count=1, maximum=5):
    model.mkdir(exist_ok=True)
    (model / "tokenizer.json").write_text('{"vocab": "example"}')
    (model / "config.json").write_text('{"max_position_embeddings": 16}')
    value = dict(schema_version=1, kind=kind, count=count, max_total_tokens=maximum,
                 vocab_size=256,
                 output={"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                 tokenizer_files=[{"name": "tokenizer.json", "sha256": hashlib.sha256(
                     (model / "tokenizer.json").read_bytes()).hexdigest()}])
    (path.parent / (path.name + ".manifest.json")).write_text(json.dumps(value))
    return value


def test_payload_lengths_are_actual_and_provenance_survives(tmp_path):
    path = write_csv(tmp_path / "rt.csv", [row(prompt="x" * 200000)])
    csv.field_size_limit(131072)
    r = load_workload_csv(path, RT)[0]
    assert r.prompt_token_ids == [2, 4, 6]
    assert (r.prompt_len, r.output_len) == (3, 2)
    assert (r.azure_prompt_len, r.azure_output_len) == (100, 50)
    assert r.source_id == "sharegpt:source:1:0"
    assert not hasattr(r, "prompt") and not hasattr(r, "output_token_ids")


@pytest.mark.parametrize("changes,error", [
    ({"prompt_token_ids": ""}, "prompt_token_ids"),
    ({"prompt_token_ids": [1, 2]}, "prompt_token_ids"),
    ({"prompt_token_ids": [True, 2, 3]}, "prompt_token_ids"),
    ({"prompt_token_ids": [1, -2, 3]}, "prompt_token_ids"),
    ({"prompt_token_ids": [1.5, 2, 3]}, "prompt_token_ids"),
    ({"output_token_ids": [1]}, "output_token_ids"),
    ({"output_token_ids": ""}, "output_token_ids"),
    ({"output_len": 1.5}, "output_len"),
    ({"output_len": 0}, "output_len"),
    ({"arrival_offset_s": "NaN"}, "arrival_offset_s"),
    ({"arrival_offset_s": -1}, "arrival_offset_s"),
])
def test_bad_content_never_silently_becomes_synthetic(tmp_path, changes, error):
    path = write_csv(tmp_path / "bad.csv", [row(**changes)])
    with pytest.raises(ValueError, match=error):
        load_workload_csv(path, RT)


def test_text_without_token_ids_is_explicitly_rejected(tmp_path):
    data = row()
    del data["prompt_token_ids"]
    path = write_csv(tmp_path / "text.csv", [data])
    with pytest.raises(ValueError, match="text prompt requires"):
        load_workload_csv(path, RT)


def test_legacy_lengths_and_reshape_remain_supported(tmp_path, monkeypatch):
    path = write_csv(tmp_path / "legacy.csv", [dict(prompt_len="3.0", output_len=2, arrival_offset_s=1)])
    monkeypatch.setenv("RT_LEN_SCALE", "2")
    r = load_workload_csv(path, RT)[0]
    assert r.prompt_len == 6 and r.prompt_token_ids is None
    real = write_csv(tmp_path / "real.csv", [row()])
    with pytest.raises(ValueError, match="RT_LEN_SCALE cannot"):
        load_workload_csv(real, RT)


def test_arrival_scaling_keeps_the_corresponding_real_ids(tmp_path):
    path = write_csv(tmp_path / "rt.csv", [
        row(request_id="later", arrival_offset_s=20, prompt_token_ids=[10, 11, 12]),
        row(request_id="first", arrival_offset_s=10),
    ])
    rows = load_rt_workload_csv(path, num_req=2, target_rps=4)
    assert [r.request_id for r in rows] == ["first", "later"]
    assert [r.arrival_offset_s for r in rows] == [0, .5]
    assert rows[0].prompt_token_ids == [2, 4, 6]
    assert rows[1].prompt_token_ids == [10, 11, 12]


def test_manifest_checks_hash_tokenizer_and_be_context(tmp_path):
    path = write_csv(tmp_path / "be.csv", [row(dataset="qmsum")])
    model = tmp_path / "matching"
    manifest(path, model, kind="be")
    assert load_workload_csv(path, BE, expected_model=str(model), max_model_len=5)[0].dataset == "qmsum"
    with pytest.raises(ValueError, match="requires 5 total tokens"):
        load_workload_csv(path, BE, expected_model=str(model), max_model_len=4)
    other = tmp_path / "different"
    other.mkdir()
    (other / "tokenizer.json").write_text("different mapping with the same vocabulary size")
    with pytest.raises(ValueError, match="tokenizer does not match"):
        load_workload_csv(path, BE, expected_model=str(other), max_model_len=5)
    with pytest.raises(ValueError, match="kind mismatch"):
        load_workload_csv(path, RT)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_workload_manifest(path, BE)


def test_context_applies_only_to_replayed_rt_prefix(tmp_path):
    path = write_csv(tmp_path / "rt.csv", [
        row(), row(request_id="tail", arrival_offset_s=1,
                   prompt_len=30, prompt_token_ids=[1] * 30),
    ])
    model = tmp_path / "model"
    manifest(path, model, count=2, maximum=32)
    assert len(load_rt_workload_csv(path, num_req=1, target_rps=1,
                                   expected_model=str(model), max_model_len=8)) == 1
    with pytest.raises(ValueError, match="needs 32 total tokens"):
        load_rt_workload_csv(path, num_req=2, target_rps=1,
                             expected_model=str(model), max_model_len=8)


def test_duplicate_ids_are_not_merged(tmp_path):
    path = write_csv(tmp_path / "duplicate.csv", [row(), row()])
    with pytest.raises(ValueError, match="duplicate request_id"):
        load_workload_csv(path, RT)


def test_preflight_rejects_before_any_profiler_or_scheduler(tmp_path, monkeypatch):
    from saturo.cli import global_scheduler
    rt = write_csv(tmp_path / "rt.csv", [row()])
    be = write_csv(tmp_path / "be.csv", [row(dataset="govreport")])
    model = tmp_path / "model"
    manifest(rt, model)
    manifest(be, model, kind="be")
    args = NS(worker=False, model=str(model), rt_workload_csv=str(rt), be_workload_csv=str(be),
              max_model_len=4)
    monkeypatch.setattr(global_scheduler, "parse_args", lambda: args)
    # No QRF/resource options exist: reaching the profiler branch would fail.
    with pytest.raises(ValueError, match="requires 5 total tokens"):
        global_scheduler.main()


def test_be_real_ids_reach_scheduler_and_worker(tmp_path):
    from saturo.runtime.scheduler import SaturoGlobalScheduler
    from saturo.runtime.worker import RayEngineWorker
    rt = write_csv(tmp_path / "rt.csv", [row()])
    be = write_csv(tmp_path / "be.csv", [row(request_id="be1", dataset="qmsum")])
    model = tmp_path / "model"
    manifest(rt, model)
    manifest(be, model, kind="be")
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler.args = NS(model=str(model), rt_workload_csv=str(rt), be_workload_csv=str(be),
                        num_req=1, target_rps=1, window_s=1, max_model_len=16)
    scheduler._build_workload()
    target = NS(role=TPUT, ready=True, worker_id=0, active_attempts={}, pending_commands=[])
    scheduler._live_workers = lambda: [target]
    scheduler._preload_be_to_workers()
    command = target.pending_commands[0]
    assert command["prompt_token_ids"] == [2, 4, 6]
    assert command["max_tokens"] == 2  # Reference length, not Azure target 50.
    assert "output_token_ids" not in command
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker.worker_role = TPUT
    worker._role_args = {"kv_handoff": "real"}
    worker._use_dummy_kv = False
    worker.SamplingParams = lambda **kw: NS(**kw)
    prompt, params, _ = worker._request_inputs(command)
    assert prompt["prompt_token_ids"] == [2, 4, 6] and params.max_tokens == 2
    assert scheduler._request_index["be1"].source_id == "sharegpt:source:1:0"
    write_csv(be, [row()])
    (tmp_path / "be.csv.manifest.json").unlink()
    with pytest.raises(ValueError, match="across RT/BE"):
        scheduler._build_workload()
