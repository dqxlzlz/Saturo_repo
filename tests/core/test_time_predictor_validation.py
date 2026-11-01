import json
from types import SimpleNamespace

import numpy as np
import pytest

from saturo.predictors.time_predictor import (
    STATEFUL_GROUP,
    TimePredictor,
    _fit_attention,
    _fit_split_profiles,
    _resolve_num_layers,
    build_or_load_time_predictor,
)


def _payload():
    return {
        "model_type": "per_layer",
        "num_layers": 2,
        "prefill_attn": {"coef": [0.1, 0.001], "intercept": 0},
        "decode_attn": {"coef": [0.2, 0.003], "intercept": 0},
        "linear": {"width": 32, "coef": {}, "fallback": [0, 7]},
    }


def _load(tmp_path, payload, **kwargs):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return TimePredictor(str(path), **kwargs)


def test_legacy_layer_count_fallback(tmp_path):
    payload = _payload()
    del payload["num_layers"]
    predictor = _load(tmp_path, payload, num_layers=80)
    assert predictor.available
    assert predictor.num_layers == 80
    assert predictor.predict_forward_ms([(1, 1)], []) > 0


def test_saved_layer_count_takes_precedence_over_legacy_fallback(tmp_path):
    predictor = _load(tmp_path, _payload(), num_layers=80)
    assert predictor.num_layers == 2


@pytest.mark.parametrize("nested", [False, True])
def test_multimodal_profiles_use_language_model_layer_count(nested):
    text_config = SimpleNamespace(num_hidden_layers=80)
    outer_config = SimpleNamespace(num_hidden_layers=24)
    model_config = SimpleNamespace(hf_config=outer_config)
    if nested:
        outer_config.text_config = text_config
    else:
        model_config.hf_text_config = text_config
    executor = SimpleNamespace(model_config=model_config)
    assert _resolve_num_layers(executor) == 80


def test_full_decode_model_does_not_add_a_zero_token_linear_intercept(tmp_path):
    payload = _payload()
    payload.update(use_decode_full_layer=True,
                   decode_full_layer={"coef": [3, 0], "intercept": 4})
    predictor = _load(tmp_path, payload)
    assert predictor.predict_forward_ms([], [(1, 100)] * 3) == 26


@pytest.mark.parametrize("model_type", ["per_layer", "iteration_level"])
def test_empty_batch_has_no_execution_or_runtime_overhead(tmp_path, model_type):
    payload = _payload()
    payload.update(model_type=model_type, coef=[1, 1, 1, 1], intercept=8,
                   latency_offset_ms=2, runtime_overhead={"constant_ms": 4})
    predictor = _load(tmp_path, payload)
    assert predictor.predict_forward_ms([], []) == 0
    assert predictor.predict_forward_from_aggregates_ms(
        prefill_tokens=0, prefill_sum_ctx=0, decode_tokens=0,
        decode_sum_ctx=0, prefill_request_count=0, decode_request_count=0) == 0


@pytest.mark.parametrize("change, message", [
    ({"num_layers": 0}, "positive num_layers"),
    ({"model_type": "unsupported"}, "unsupported"),
    ({"model_type": "iteration_level", "coef": [1, 2, 3], "intercept": 0},
     "four coefficients"),
    ({"linear": {"width": 0, "fallback": [0, 1]}}, "width"),
    ({"decode_attn": {"coef": [float("nan"), 0], "intercept": 0}}, "finite"),
    ({"runtime_overhead": {"constant_ms": float("inf")}}, "finite"),
    ({"latency_scale": 0}, "latency_scale"),
    ({"linear_tile_override": {"per_layer_values_ms": [float("nan")]}},
     "finite"),
])
def test_invalid_profiles_are_rejected(tmp_path, change, message):
    payload = _payload()
    payload.update(change)
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, payload)


def test_git_lfs_pointer_reports_how_to_retrieve_the_profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("version https://git-lfs.github.com/spec/v1\n"
                    "oid sha256:" + "0" * 64 + "\nsize 100\n")
    with pytest.raises(ValueError, match="git lfs pull"):
        TimePredictor(str(path))


def test_attention_fit_ignores_nonfinite_measurements():
    rows = [
        {"mode": "decode", "n_tokens": n, "sum_context": context,
         "group_per_layer_ms": {STATEFUL_GROUP: {"median": measured}}}
        for n, context, measured in [(1, 1, 6), (2, 4, 17), (3, 9, 34),
                                     (4, 16, float("nan")),
                                     (5, float("inf"), 86)]
    ]
    fitted = _fit_attention(rows, "decode")
    assert fitted["coef"] == pytest.approx([2, 3])
    assert fitted["intercept"] == pytest.approx(1)
    assert np.isfinite(fitted["coef"]).all()


def test_incomplete_profiling_cannot_produce_an_available_model(tmp_path):
    executor = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2)),
        profile_qwen2_module_times=lambda **kwargs: {"rows": []},
    )
    path = tmp_path / "fitted.json"
    predictor = build_or_load_time_predictor(str(path), executor)
    assert not predictor.available
    assert not path.exists()
    assert path.with_suffix(".json.raw_profile.json").exists()


def test_split_fit_requires_a_known_layer_count():
    with pytest.raises(ValueError, match="positive model layer count"):
        _fit_split_profiles([], [], [], num_layers=0, meta={})
