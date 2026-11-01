import json

from saturo.predictors.time_predictor import (
    TimePredictor,
    TimePredictorProfileIdentity,
    profile_matches_identity,
    time_predictor_cache_path,
)


def _identity(tp: int = 1) -> TimePredictorProfileIdentity:
    return TimePredictorProfileIdentity(
        model="/models/qwen2.5-72b",
        dtype="float16",
        quantization="unknown",
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        gpu_name="NVIDIA L20D",
    )


def _profile_payload(identity: TimePredictorProfileIdentity) -> dict:
    return {
        "model_type": "iteration_level",
        "num_layers": 80,
        "profile_identity": identity.to_dict(),
        "profile_key": identity.digest,
        "coef": [1.0, 0.0, 1.0, 0.0],
        "intercept": 0.0,
    }


def test_time_predictor_cache_path_includes_parallel_identity(tmp_path):
    tp1 = time_predictor_cache_path(_identity(tp=1), tmp_path)
    tp4 = time_predictor_cache_path(_identity(tp=4), tmp_path)

    assert tp1 != tp4
    assert "tp1" in tp1.name
    assert "tp4" in tp4.name


def test_time_predictor_default_cache_root_is_runtime_profiles():
    path = time_predictor_cache_path(_identity(tp=1))

    assert "saturo/results/runtime_profiles" in path.as_posix()


def test_profile_identity_match_rejects_wrong_parallel_config(tmp_path):
    profile = tmp_path / "predictor.json"
    profile.write_text(json.dumps(_profile_payload(_identity(tp=4))))

    assert profile_matches_identity(profile, _identity(tp=4))
    assert not profile_matches_identity(profile, _identity(tp=1))


def test_profile_identity_accepts_repo_id_for_local_checkout(tmp_path):
    saved = TimePredictorProfileIdentity(
        model="Qwen/Qwen2.5-72B-Instruct",
        dtype="float16",
        quantization="unknown",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        gpu_name="NVIDIA GB200",
    )
    local = TimePredictorProfileIdentity(
        model="/root/models/Qwen/Qwen2.5-72B-Instruct",
        dtype=saved.dtype,
        quantization=saved.quantization,
        tensor_parallel_size=saved.tensor_parallel_size,
        pipeline_parallel_size=saved.pipeline_parallel_size,
        gpu_name=saved.gpu_name,
    )
    profile = tmp_path / "predictor.json"
    profile.write_text(json.dumps(_profile_payload(saved)))

    assert profile_matches_identity(profile, local)


def test_profile_identity_rejects_different_absolute_paths(tmp_path):
    saved = TimePredictorProfileIdentity(
        model="/models-a/Qwen/Qwen2.5-72B-Instruct",
        dtype="float16",
        quantization="unknown",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        gpu_name="NVIDIA GB200",
    )
    other = TimePredictorProfileIdentity(
        model="/models-b/Qwen/Qwen2.5-72B-Instruct",
        dtype=saved.dtype,
        quantization=saved.quantization,
        tensor_parallel_size=saved.tensor_parallel_size,
        pipeline_parallel_size=saved.pipeline_parallel_size,
        gpu_name=saved.gpu_name,
    )
    profile = tmp_path / "predictor.json"
    profile.write_text(json.dumps(_profile_payload(saved)))

    assert not profile_matches_identity(profile, other)


def test_time_predictor_skips_mismatched_identity(tmp_path):
    profile = tmp_path / "predictor.json"
    profile.write_text(json.dumps(_profile_payload(_identity(tp=4))))

    predictor = TimePredictor(
        str(profile),
        num_layers=80,
        expected_identity=_identity(tp=1),
    )

    assert not predictor.available


def test_time_predictor_adds_runtime_overhead(tmp_path):
    profile = tmp_path / "predictor.json"
    payload = _profile_payload(_identity(tp=1))
    payload["runtime_overhead"] = {
        "constant_ms": 2.0,
        "per_prefill_request_ms": 3.0,
        "per_decode_request_ms": 5.0,
        "per_active_request_ms": 7.0,
        "per_prefill_token_ms": 0.5,
        "per_decode_token_ms": 0.25,
    }
    profile.write_text(json.dumps(payload))

    predictor = TimePredictor(str(profile), num_layers=80)

    # Forward: 4 prefill tokens + 2 decode tokens = 6 ms.
    # Overhead: 2 + 1*3 + 2*5 + 3*7 + 4*0.5 + 2*0.25 = 38.5 ms.
    assert predictor.predict_forward_ms(
        [(4, 4)],
        [(1, 10), (1, 20)],
    ) == 44.5
