import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from saturo.predictors.qrf import load_required_qrf
from saturo.saturation.decode_saturation_core import (
    DefaultDecodeSaturationStateProvider, DecodeSaturationPolicy)


@pytest.fixture(autouse=True)
def clean_predictor_environment(monkeypatch):
    for name in (
            "SATURO_HISTORY_LENGTH_PREDICTOR",
            "VLLM_DECODE_SATURATION_MODEL_LEN_PREDICTOR",
            "VLLM_REQUEST_MODEL_LEN_PREDICTOR",
            "VLLM_DECODE_SATURATION_LENGTH_PREDICTOR",
            "VLLM_REQUEST_LENGTH_PREDICTOR",
            "SATURO_DECODE_ORACLE_SCHEDULE", "VLLM_DECODE_SATURATION"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def qrf_path(tmp_path):
    from saturo.predictors.request_length_predictor import (
        QRFRemainingLengthPredictor, build_feature_matrix)
    rows = [{"prompt_len": 16, "generated_len": i} for i in range(8)]
    features = build_feature_matrix(rows, feature_mode="len", hash_bins=128)
    model = QRFRemainingLengthPredictor(
        n_estimators=2, min_samples_leaf=1, n_jobs=1)
    model.fit(features, np.full(8, 10.0))
    path = tmp_path / "qrf.joblib"
    model.save(path)
    return path


class SamplingWithoutLimit:
    # Source arrival metadata is allowed; generation caps remain forbidden.
    extra_args = {}

    @property
    def max_tokens(self):
        raise AssertionError("max_tokens is not a remaining-length prediction")


class RequestWithoutSamplingLimit:
    sampling_params = SamplingWithoutLimit()

    def get_seqs(self):
        return [SimpleNamespace(get_prompt_len=lambda: 16,
                                get_output_len=lambda: 2)]


def test_enabled_provider_fails_at_construction_without_qrf():
    with pytest.raises(RuntimeError, match="requires a QRF"):
        DefaultDecodeSaturationStateProvider()


def test_legacy_flags_do_not_enable_missing_model_fallback():
    provider = DefaultDecodeSaturationStateProvider(
        load_predictor=False, require_model_len_predictor=False,
        model_len_predictor_strict=False)
    with pytest.raises(RuntimeError, match="requires a QRF"):
        provider.get_remaining_tokens(RequestWithoutSamplingLimit())


def test_real_qrf_predicts_without_reading_sampling_cap(qrf_path):
    provider = DefaultDecodeSaturationStateProvider(predictor_path=str(qrf_path))
    assert provider.get_remaining_tokens(RequestWithoutSamplingLimit()) == 10
    # A cap lower than the QRF output still does not replace/clamp prediction.
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(max_tokens=3),
        get_seqs=RequestWithoutSamplingLimit().get_seqs)
    assert provider.get_remaining_tokens(req) == 10
    assert provider.model_len_predictor_eta == .95


def test_prediction_failure_is_not_silently_replaced():
    def fail(**kwargs):
        raise ValueError("bad prediction")
    provider = DefaultDecodeSaturationStateProvider(
        load_predictor=False, model_len_predictor_strict=False)
    provider.model_len_predictor = SimpleNamespace(
        model_type="qrf", predict_remaining=fail)
    with pytest.raises(ValueError, match="bad prediction"):
        provider.get_remaining_tokens(RequestWithoutSamplingLimit())


def test_non_qrf_artifact_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "saturo.predictors.request_length_predictor.load_length_predictor",
        lambda path: SimpleNamespace(model_type="gbdt"))
    with pytest.raises(ValueError, match="requires QRF"):
        load_required_qrf("gbdt.joblib")


def test_bad_qrf_path_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_required_qrf(tmp_path / "missing.joblib")


def test_enabled_policy_cannot_disable_qrf_with_legacy_flag():
    config = SimpleNamespace(enable_decode_saturation=True,
                             decode_saturation_require_model_len_predictor=False)
    with pytest.raises(RuntimeError, match="requires a QRF"):
        DecodeSaturationPolicy(16, 100, scheduler_config=config)


def test_disabled_policy_does_not_need_a_predictor():
    policy = DecodeSaturationPolicy(16, 100)
    decision = policy.make_decision([object()], [], [], 0)
    assert not decision.enabled


def test_runtime_oracle_environment_cannot_bypass_qrf(monkeypatch):
    monkeypatch.setenv("SATURO_DECODE_ORACLE_SCHEDULE", "oracle.json")
    config = SimpleNamespace(enable_decode_saturation=True)
    with pytest.raises(ValueError, match="requires QRF"):
        DecodeSaturationPolicy(16, 100, scheduler_config=config)


def test_progressive_simulator_has_no_true_remaining_fallback(qrf_path):
    from saturo.simulators.decode_simulator.config import (
        ModelConfig, SaturationConfig)
    from saturo.simulators.decode_simulator.predictors.memory import (
        DPoolSaturationPredictor)
    predictor = DPoolSaturationPredictor(ModelConfig(), SaturationConfig())
    request = SimpleNamespace(prompt_len=16, tokens_generated=2,
                              prompt_token_ids=None, generated_token_ids=None)
    with pytest.raises(RuntimeError, match="requires a QRF"):
        predictor.predict_active_remaining_len(request)
    predictor.configure_model_len_predictor(str(qrf_path))
    assert predictor.predict_active_remaining_len(request) == 10


def test_worker_info_records_actual_qrf_type_path_and_quantile(qrf_path):
    from saturo.runtime.worker import RayEngineWorker
    provider = DefaultDecodeSaturationStateProvider(predictor_path=str(qrf_path))
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker.worker_id, worker.worker_role = 0, "decode"
    scheduler = SimpleNamespace(decode_saturation=SimpleNamespace(provider=provider))
    worker.engine = SimpleNamespace(engine=SimpleNamespace(scheduler=[scheduler]))
    info = asyncio.run(worker.info())
    assert info["remaining_length_predictors"] == [
        {"type": "qrf", "path": str(qrf_path), "quantile": .95, "feature_mode": "len"}]


def test_scheduler_checks_qrf_before_launching_serving_workers():
    from saturo.runtime.scheduler import SaturoGlobalScheduler
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler.args = SimpleNamespace(model="test-model", enable_decode_saturation=True)
    scheduler.worker_plan = [SimpleNamespace(role="decode")]
    with pytest.raises(RuntimeError, match="requires a QRF"):
        scheduler.run()


def test_arrival_does_not_treat_generation_cap_as_historical_output(qrf_path):
    provider = DefaultDecodeSaturationStateProvider(predictor_path=str(qrf_path))
    request = RequestWithoutSamplingLimit()
    provider.observe_rt_arrival(request, now_s=100.0)
    assert provider._output_len_count == 0
    assert provider._avg_output_len() == 10.0  # Cold start uses QRF, no cap.
    provider.observe_rt_finish(request)
    assert provider._output_len_count == 1
    assert provider._avg_output_len() == 2.0  # Observed completed output only.
