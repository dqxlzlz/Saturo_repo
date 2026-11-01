"""Exact predictor/search equivalence and bounded batching for CPU rebuilds."""
import math
from types import SimpleNamespace
import json

import numpy as np
import pytest

from saturo.predictors.request_length_predictor import QRFRemainingLengthPredictor
from saturo.simulators.decode_simulator.config import (
    ModelConfig, SaturationConfig, SimulatorConfig)
from saturo.simulators.decode_simulator.predictors.latency import LatencyPredictor
from saturo.simulators.decode_simulator.predictors.memory import DPoolSaturationPredictor
from saturo.simulators.decode_simulator.request import Request, RequestType
from saturo.simulators.decode_simulator.simulator import DecodeSimulator, DecodeStrategy


def request(index, prompt=16, generated=2, output=100):
    value = Request(index, RequestType.RT, prompt, output, 0,
                    tokens_generated=generated)
    value.prompt_token_ids = tuple(range(prompt))
    value.output_token_ids = tuple(range(1000, 1000 + output))
    return value


@pytest.mark.parametrize('feature_mode', ['len', 'token_ids'])
@pytest.mark.parametrize('eta', [.5, .95, .99])
def test_batched_observed_features_equal_scalar_QRF_and_reuse_exact_rows(feature_mode, eta):
    model = QRFRemainingLengthPredictor(n_estimators=5, min_samples_leaf=1,
                                       n_jobs=1, feature_mode=feature_mode,
                                       hash_bins=8, random_state=7)
    requests = [request(i, 8 + i, i % 5) for i in range(16)]
    features = np.vstack([model._feature_row(prompt_len=r.prompt_len,
        generated_len=r.tokens_generated, prompt_token_ids=r.prompt_token_ids,
        generated_token_ids=r.generated_token_ids) for r in requests])
    model.fit(features, np.arange(16, dtype=np.float32) * 3 + 5)
    predictor = DPoolSaturationPredictor(ModelConfig(),
        SaturationConfig(model_len_quantile=eta))
    predictor.configure_model_len_predictor(None, predictor=model)
    baseline = [predictor.predict_active_remaining_len(r) for r in requests]
    assert predictor.predict_active_remaining_lens(requests) == baseline
    info = model.remaining_batch_cache_info()
    assert info['model_calls'] == 1
    # Changing requested output caps cannot affect observed-prefix forecasts.
    for r in requests:
        r.target_output_len += 10000
    assert predictor.predict_active_remaining_lens(requests) == baseline
    assert model.remaining_batch_cache_info()['model_calls'] == 1
    requests[0].tokens_generated += 1
    assert predictor.predict_active_remaining_lens(requests) == [
        predictor.predict_active_remaining_len(r) for r in requests]


def test_scalar_override_and_legacy_predictor_keep_original_path():
    predictor = DPoolSaturationPredictor(ModelConfig(), SaturationConfig())
    predictor.predict_active_remaining_len = lambda r: r.prompt_len + 3
    values = [request(1), request(2, 32)]
    assert predictor.predict_active_remaining_lens(values) == [19, 35]
    simulator = DecodeSimulator.__new__(DecodeSimulator)
    simulator.strategy = DecodeStrategy.PROGRESSIVE
    simulator.current_time_s = 10
    simulator.rt_active = values
    simulator.saturation_predictor = SimpleNamespace(
        predict_active_remaining_len=lambda r: 7)
    assert [r.remaining_tokens for r in simulator._active_rt_forecasts()] == [7, 7]


def test_empty_batch_does_not_require_QRF_and_bad_row_count_fails():
    predictor = DPoolSaturationPredictor(ModelConfig(), SaturationConfig())
    assert predictor.predict_active_remaining_lens([]) == []
    predictor._model_len_predictor = QRFRemainingLengthPredictor(n_estimators=1)
    predictor._model_len_predictor.predict_remaining_batch = lambda rows, eta: []
    with pytest.raises(RuntimeError, match='row count'):
        predictor.predict_active_remaining_lens([request(1)])


@pytest.mark.parametrize('strategy', [DecodeStrategy.GREEDY, DecodeStrategy.ORACLE])
def test_nonprogressive_forecasts_never_use_QRF(strategy):
    simulator = DecodeSimulator.__new__(DecodeSimulator)
    simulator.strategy = strategy
    simulator.current_time_s = 10
    simulator.rt_active = [request(1, generated=2, output=9)]
    simulator.saturation_predictor = None
    remaining = simulator._active_rt_forecasts()[0].remaining_tokens
    assert remaining == (1 if strategy == DecodeStrategy.GREEDY else 7)


def simulator(tmp_path):
    profile = tmp_path / 'time.json'
    profile.write_text(json.dumps({'time_predictor': {
        'model_type': 'iteration_level', 'coef': [0.02, .0003, .7, .002],
        'intercept': 1.25}}))
    value = DecodeSimulator.__new__(DecodeSimulator)
    value.latency_predictor = LatencyPredictor(str(profile))
    value.config = SimulatorConfig(be_chunk_sizes=[1, 5, 16, 64])
    value.model_config = ModelConfig()
    value._can_account_be_blocks = lambda *args: True
    return value


def test_fixed_decode_stats_are_bitwise_equal_and_recomputed_after_progress(tmp_path):
    value = simulator(tmp_path)
    rt = [request(0, 934, 89), request(1, 17, 4)]
    be = [request(i + 3, 123 + i, i) for i in range(100)]
    prefill = request(900, 256, 0)
    prefill.prefill_tokens_processed = 71
    for chunk in [0, 1, 5, 16, 64, 256]:
        chunks = [(prefill, chunk)]
        before = value._predict_iteration_latency(rt, be, chunks)
        after = value._predict_with_fixed_decode_statistics(
            rt, be, chunks, value._fixed_decode_statistics(rt, be))
        assert float(before).hex() == float(after).hex()
    old = value._fixed_decode_statistics(rt, be)
    rt[0].generate_one_token()
    updated = value._fixed_decode_statistics(rt, be)
    assert old != updated
    assert value._predict_with_fixed_decode_statistics(rt, be, [], updated) == \
        value._predict_iteration_latency(rt, be, [])


def test_prefill_search_same_at_exact_and_nextafter_latency_boundaries(tmp_path):
    value = simulator(tmp_path)
    rt, be = [request(1, 31)], [request(2, 79)]
    prefill = request(900, 64, 0)
    prefill.prefill_tokens_processed = 7
    native = value._fixed_decode_statistics
    for chunk in [1, 5, 16, 64]:
        latency = value._predict_iteration_latency(rt, be, [(prefill, chunk)])
        for target in [math.nextafter(latency, -math.inf), latency,
                       math.nextafter(latency, math.inf)]:
            value._fixed_decode_statistics = lambda *args: None
            baseline = value._find_be_prefill_chunk_by_latency(
                rt, be, [], prefill, 64, target, 100000)
            value._fixed_decode_statistics = native
            optimized = value._find_be_prefill_chunk_by_latency(
                rt, be, [], prefill, 64, target, 100000)
            assert optimized == baseline


def test_custom_time_or_simulator_hooks_use_tuple_fallback(tmp_path):
    value = simulator(tmp_path)
    value.latency_predictor.predictor.predict_forward_ms = lambda *args: 3.25
    assert value._fixed_decode_statistics([request(1)], []) is None
    assert value._predict_with_fixed_decode_statistics([request(1)], [], [], None) == 3.25
    value._predict_iteration_latency = lambda *args: .123
    assert value._fixed_decode_statistics([], []) is None
    assert value._predict_with_fixed_decode_statistics([], [], [], None) == .123


def test_class_level_prediction_override_retains_scalar_semantics(monkeypatch):
    predictor = DPoolSaturationPredictor(ModelConfig(), SaturationConfig())
    monkeypatch.setattr(DPoolSaturationPredictor, 'predict_active_remaining_len',
                        lambda self, req: 37)
    assert predictor.predict_active_remaining_lens([request(1)]) == [37]


def test_class_level_latency_override_retains_tuple_semantics(tmp_path, monkeypatch):
    value = simulator(tmp_path)
    monkeypatch.setattr(DecodeSimulator, '_predict_iteration_latency',
                        lambda self, *args: .321)
    assert value._fixed_decode_statistics([request(1)], []) is None
    assert value._predict_with_fixed_decode_statistics([request(1)], [], [], None) == .321


@pytest.mark.parametrize('instance_override', [False, True])
def test_inherited_QRF_batch_does_not_bypass_scalar_override(instance_override):
    class CustomizedQRF(QRFRemainingLengthPredictor):
        def predict_remaining(self, **kwargs):
            return 91

    model = (QRFRemainingLengthPredictor(n_estimators=1)
             if instance_override else CustomizedQRF(n_estimators=1))
    if instance_override:
        model.predict_remaining = lambda **kwargs: 91
    predictor = DPoolSaturationPredictor(ModelConfig(), SaturationConfig())
    predictor.configure_model_len_predictor(None, predictor=model)
    assert predictor.predict_active_remaining_lens([request(1)]) == [91]


def test_custom_forecast_batch_cannot_silently_omit_active_RT():
    simulator = DecodeSimulator.__new__(DecodeSimulator)
    simulator.strategy = DecodeStrategy.PROGRESSIVE
    simulator.current_time_s = 10
    simulator.rt_active = [request(1)]
    simulator.saturation_predictor = SimpleNamespace(
        predict_active_remaining_lens=lambda requests: [])
    with pytest.raises(RuntimeError, match='row count'):
        simulator._active_rt_forecasts()
