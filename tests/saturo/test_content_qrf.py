import csv
import json
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from saturo.predictors.request_length_predictor import (
    TFIDFQRFRemainingLengthPredictor, build_length_feature_row, load_length_predictor,
)
from saturo.predictors.content_training import (
    prefix_samples, read_content_history, split_requests, train_content_qrf,
)
from saturo.predictors.qrf import validate_qrf_workload
from saturo.saturation.decode_saturation_core import DefaultDecodeSaturationStateProvider


@pytest.fixture
def predictor():
    p = TFIDFQRFRemainingLengthPredictor(n_estimators=16, min_samples_leaf=1,
                                        tfidf_max_features=64, n_jobs=1)
    data, y = [], []
    for _ in range(12):
        for prompt, length in [([11, 12, 13, 14], 5), ([21, 22, 23, 24], 45)]:
            for t in range(4):
                data.append(dict(prompt_len=4, generated_len=t,
                                 prompt_token_ids=prompt, generated_token_ids=[31] * t))
                y.append(length - t)
    strings = [p._tokens_to_str(r["prompt_token_ids"] + r["generated_token_ids"]) for r in data]
    lengths = np.asarray([build_length_feature_row(prompt_len=4, generated_len=r["generated_len"])
                          for r in data], dtype=np.float32)
    p.fit_from_raw(strings, lengths, np.asarray(y, dtype=np.float32))
    return p


def test_same_lengths_different_real_contents_change_prediction(predictor):
    a = dict(prompt_len=4, generated_len=0, prompt_token_ids=[11, 12, 13, 14], generated_token_ids=[])
    b = dict(a, prompt_token_ids=[21, 22, 23, 24])
    matrix = predictor._remaining_batch_features([a, b])
    assert (matrix[0] != matrix[1]).nnz > 0
    scalar = [predictor.predict_remaining(**r) for r in (a, b)]
    assert scalar[0] < scalar[1]
    assert predictor.predict_remaining_batch([a, b]) == scalar


@pytest.mark.parametrize("kwargs", [
    dict(prompt_len=4, generated_len=0),
    dict(prompt_token_ids=[], generated_len=0),
    dict(prompt_len=4, generated_len=1, prompt_token_ids=[11, 12, 13, 14]),
    dict(prompt_len=4, generated_len=1, prompt_token_ids=[11, 12, 13, 14], generated_token_ids=[31, 32]),
])
def test_missing_content_or_future_tokens_never_fall_back_to_lengths(predictor, kwargs):
    with pytest.raises(ValueError, match="Content QRF"):
        predictor.predict_remaining(**kwargs)
    with pytest.raises(ValueError, match="Content QRF"):
        predictor.predict_remaining_batch([kwargs])


def test_provider_and_background_optimizer_use_actual_content(predictor):
    from saturo.saturation.async_target import ArrivalState, RequestFeatures, TargetSnapshot, TargetComputer
    from saturo.saturation.types import SaturationParams
    provider = DefaultDecodeSaturationStateProvider(load_predictor=False)
    provider.model_len_predictor = predictor
    def group(tokens):
        seq = NS(get_prompt_len=lambda: 4, get_output_len=lambda: 0,
                 get_prompt_token_ids=lambda: tuple(tokens), get_output_token_ids=lambda: ())
        return NS(get_seqs=lambda: [seq])
    groups = [group([11, 12, 13, 14]), group([21, 22, 23, 24])]
    expected = provider.get_remaining_tokens_batch(groups)
    assert expected == [provider.get_remaining_tokens(g) for g in groups]
    assert expected[0] < expected[1]
    assert provider._avg_output_len() == predictor.metadata["mean_output_len"]
    requests = tuple(RequestFeatures(str(i), 4, 0, 10., tuple(g.get_seqs()[0].get_prompt_token_ids()), ())
                     for i, g in enumerate(groups))
    arrivals = ArrivalState(None, (), None, 4, 1024, .001, 8., 2, 0., 0)
    snapshot = TargetSnapshot(1, 1, 10., requests, arrivals, 2, 0, 0, time.perf_counter())
    computer = TargetComputer(predictor, None, SaturationParams(total_gpu_blocks=1000), .95)
    decision, _ = computer(snapshot)
    assert decision.enabled
    assert computer._predictor.predict_remaining_batch([r.prediction_row() for r in requests]) == expected


def write_history(path, count=20):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["request_id", "source_id", "prompt_len", "output_len",
                                                    "prompt_token_ids", "output_token_ids"])
        writer.writeheader()
        for i in range(count):
            writer.writerow(dict(request_id=f"r{i}", source_id=f"s{i}", prompt_len=3, output_len=5 + i % 2,
                prompt_token_ids=json.dumps([11 if i % 2 else 21, 31, 100 + i]),
                output_token_ids=json.dumps([50] * (5 + i % 2))))


def test_request_group_split_precedes_prefix_expansion(tmp_path):
    path = tmp_path / "history.csv"
    write_history(path)
    rows = read_content_history(path)
    # Repeated source payloads stay in the same side of the split.
    rows.append(dict(rows[0], source_id="duplicate-source"))
    a, b = split_requests(rows)
    assert not {r["prompt_group"] for r in a} & {r["prompt_group"] for r in b}
    assert (a, b) == split_requests(rows)
    samples, targets = prefix_samples(rows, 4)
    for sample, target in zip(samples, targets):
        assert len(sample["generated_token_ids"]) == sample["generated_len"]
        assert target > 0
        assert target + sample["generated_len"] in (5, 6)


def test_train_save_reload_with_actual_content_and_auditable_labels(tmp_path):
    path, output = tmp_path / "history.csv", tmp_path / "content.joblib"
    write_history(path)
    p, report = train_content_qrf(path, output, n_estimators=4, min_samples_leaf=1,
                                 max_features=32, max_steps=3, n_jobs=1)
    assert report["prompt_group_overlap"] == 0
    assert report["validation_rows_with_nonzero_content"] > 0
    assert report["total_features"] > 5
    assert p.metadata["label_source"] == "source_dataset_reference_responses"
    sample = dict(prompt_token_ids=[11, 31, 101], generated_token_ids=[50])
    assert p.predict_remaining(**sample) == load_length_predictor(output).predict_remaining(**sample)


def test_observed_model_outputs_replace_reference_training_labels(tmp_path):
    history = tmp_path / "history.csv"
    write_history(history)
    outputs = tmp_path / "requests.csv"
    outputs.write_text('request_id,finished_s,observed_tokens,generated_token_ids\nr0,1,2,"[88,89]"\n')
    rows = read_content_history(history, outputs)
    assert len(rows) == 1 and rows[0]["output_token_ids"] == [88, 89]


def test_runtime_guard_rejects_length_artifact_for_prepared_real_workloads(tmp_path, predictor):
    path = tmp_path / "rt.csv"
    manifest = {"kind": "conv", "tokenizer_files": [{"name": "tokenizer.json", "sha256": "abc"}]}
    (tmp_path / "rt.csv.manifest.json").write_text(json.dumps(manifest))
    args = NS(kv_handoff="real", rt_workload_csv=str(path))
    length = NS(config=NS(feature_mode="len"))
    with pytest.raises(ValueError, match="content QRF"):
        validate_qrf_workload(length, args)
    with pytest.raises(ValueError, match="provenance"):
        validate_qrf_workload(predictor, args)
    predictor.metadata["tokenizer_files"] = manifest["tokenizer_files"]
    validate_qrf_workload(predictor, args)
    args.kv_handoff = "dummy"
    with pytest.raises(ValueError, match="--kv-handoff real"):
        validate_qrf_workload(predictor, args)
    validate_qrf_workload(length, args)


def test_resource_simulator_csv_retains_real_prompt_and_observed_prefix(tmp_path, predictor):
    from saturo.simulators.decode_simulator.trace_loader import load_rt_trace
    from saturo.simulators.decode_simulator.predictors.memory import DPoolSaturationPredictor
    from saturo.simulators.decode_simulator.config import ModelConfig, SaturationConfig
    path = tmp_path / "rt.csv"
    path.write_text('request_id,prompt_len,output_len,arrival_offset_s,prompt_token_ids,output_token_ids\n'
                    'r,4,3,0,"[11,12,13,14]","[31,32,33]"\n')
    request = load_rt_trace(str(path), duration_s=10)[0]
    assert request.prompt_token_ids == (11, 12, 13, 14)
    request.tokens_generated = 1
    assert request.generated_token_ids == (31,)
    simulator = DPoolSaturationPredictor(ModelConfig(), SaturationConfig())
    simulator.configure_model_len_predictor(None, predictor=predictor)
    assert simulator.predict_active_remaining_len(request) == predictor.predict_remaining(
        prompt_len=4, generated_len=1, prompt_token_ids=[11, 12, 13, 14], generated_token_ids=[31])
    with pytest.raises(ValueError, match="would be truncated"):
        load_rt_trace(str(path), duration_s=10, max_prompt_len=2)
