from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from saturo.predictors.request_length_predictor import (
    QRFRemainingLengthPredictor, TFIDFQRFRemainingLengthPredictor,
    build_feature_matrix, build_length_feature_row)
from saturo.saturation.decode_saturation_core import (
    DefaultDecodeSaturationStateProvider, DecodeSaturationPolicy)


def rows(n=32):
    return [dict(prompt_len=16 + i * 3, generated_len=i % 7,
                 prompt_token_ids=[1 + i % 8] * (16 + i * 3),
                 generated_token_ids=[1 + (i + 1) % 8] * (i % 7))
            for i in range(n)]


def trained(mode='len'):
    data = rows()
    target = np.asarray([20 + i * 2 for i in range(len(data))], dtype=np.float32)
    if mode == 'tfidf':
        p = TFIDFQRFRemainingLengthPredictor(
            n_estimators=4, min_samples_leaf=2, n_jobs=1,
            tfidf_max_features=32)
        strings = [p._tokens_to_str(r['prompt_token_ids'] + r['generated_token_ids'])
                   for r in data]
        features = np.asarray([build_length_feature_row(
            prompt_len=r['prompt_len'], generated_len=r['generated_len'])
            for r in data], dtype=np.float32)
        p.fit_from_raw(strings, features, target)
    else:
        p = QRFRemainingLengthPredictor(
            feature_mode=mode, hash_bins=8, n_estimators=4,
            min_samples_leaf=2, n_jobs=1)
        p.fit(build_feature_matrix(data, feature_mode=mode, hash_bins=8), target)
    return p


@pytest.mark.parametrize('mode', ['len', 'token_ids', 'tfidf'])
@pytest.mark.parametrize('eta', [.5, .95, 1.0])
def test_batched_cached_predictions_match_scalar_for_all_features(mode, eta):
    p = trained(mode)
    data = rows(13)
    if mode != 'tfidf':
        data += [dict(prompt_len=23, generated_len=0,
                      prompt_token_ids=None, generated_token_ids=None)]
    expected = [p.predict_remaining(eta=eta, **r) for r in data]
    assert p.predict_remaining_batch(data, eta=eta) == expected
    calls = p.remaining_batch_cache_info()['model_calls']
    assert p.predict_remaining_batch(data[::-1], eta=eta) == expected[::-1]
    assert p.remaining_batch_cache_info()['model_calls'] == calls
    assert p.remaining_batch_cache_info()['hits'] == len(data)


def test_quantiles_configuration_and_refit_cannot_reuse_stale_values():
    p = trained()
    data = rows(3)
    assert p.predict_remaining_batch(data, eta=.5) == [
        p.predict_remaining(eta=.5, **r) for r in data]
    assert p.predict_remaining_batch(data, eta=.95) == [
        p.predict_remaining(eta=.95, **r) for r in data]
    assert p.remaining_batch_cache_info()['model_calls'] == 2
    invalidations = p.remaining_batch_cache_info()['invalidations']
    p.config.hash_bins += 1  # Length features stay the same; config still invalidates.
    p.predict_remaining_batch(data, eta=.95)
    assert p.remaining_batch_cache_info()['invalidations'] == invalidations + 1
    x = build_feature_matrix(rows(), feature_mode='len', hash_bins=8)
    p.fit(x, np.full(len(x), 1000, dtype=np.float32))
    assert p.predict_remaining_batch(data, eta=.95) == [1000] * len(data)


def test_replacing_forest_and_leaf_artifact_invalidates_identity():
    p, other = trained(), trained()
    data = rows(2)
    p.predict_remaining_batch(data)
    x = build_feature_matrix(rows(), feature_mode='len', hash_bins=8)
    other.fit(x, np.full(len(x), 777, dtype=np.float32))
    p.rf, p.leaf_values = other.rf, other.leaf_values
    assert p.predict_remaining_batch(data) == [777, 777]
    assert p.remaining_batch_cache_info()['invalidations'] == 1


@pytest.mark.parametrize('mode', ['token_ids', 'tfidf'])
def test_same_lengths_changed_token_features_are_not_a_cache_hit(mode):
    p = trained(mode)
    data = [rows(1)[0]]
    p.predict_remaining_batch(data)
    changed = [{**data[0], 'prompt_token_ids': [2] * data[0]['prompt_len']}]
    assert p.predict_remaining_batch(changed) == [p.predict_remaining(**changed[0])]
    assert p.remaining_batch_cache_info()['model_calls'] == 2


def test_cache_bounds_and_uncacheable_row_preserve_predictions():
    p = trained()
    p._REMAINING_CACHE_MAX_BYTES = 24  # Each length row has 20 feature bytes.
    data = rows(4)
    expected = [p.predict_remaining(**r) for r in data]
    assert p.predict_remaining_batch(data) == expected
    info = p.remaining_batch_cache_info()
    assert info['feature_bytes'] <= 24 and info['entries'] == 1
    assert p.predict_remaining_batch(data[:1]) == expected[:1]
    assert p.remaining_batch_cache_info()['model_calls'] == 2
    p = trained('token_ids')
    p._REMAINING_CACHE_MAX_BYTES = 8
    for _ in range(2):
        assert p.predict_remaining_batch(data[:1]) == [p.predict_remaining(**data[0])]
        assert p.remaining_batch_cache_info()['entries'] == 0
    assert p.remaining_batch_cache_info()['model_calls'] == 2
    p = trained()
    p._REMAINING_CACHE_MAX_ENTRIES = 2
    p.predict_remaining_batch(data)
    assert p.remaining_batch_cache_info()['entries'] == 2


def test_nonfinite_model_prediction_and_bad_row_count_still_raise():
    p = trained()
    with patch.object(p, 'predict_quantile', return_value=np.asarray([np.nan])):
        with pytest.raises(ValueError):
            p.predict_remaining_batch(rows(1))
    p.clear_remaining_prediction_cache()
    with patch.object(p, 'predict_quantile', return_value=np.asarray([])):
        with pytest.raises(RuntimeError, match='row count'):
            p.predict_remaining_batch(rows(1))


class Seq:
    def __init__(self, prompt, generated):
        self.prompt, self.generated = prompt, generated
    def get_prompt_len(self): return self.prompt
    def get_output_len(self): return self.generated
    def get_prompt_token_ids(self):
        raise AssertionError('length-only models do not use token content')
    def get_output_token_ids(self):
        raise AssertionError('length-only models do not use token content')


def group(i):
    seq = Seq(16 + 3 * (i % 32), i % 7)
    return SimpleNamespace(request_id=str(i), arrival_time=10,
                           get_seqs=lambda: [seq], seq=seq)


def test_provider_batches_misses_and_reuses_unchanged_pending_states():
    provider = DefaultDecodeSaturationStateProvider(load_predictor=False)
    provider.model_len_predictor = p = trained()
    groups = [group(i) for i in range(80)]
    expected = [p.predict_remaining(prompt_len=g.seq.prompt,
                                   generated_len=g.seq.generated) for g in groups]
    assert provider.get_remaining_tokens_batch(groups) == expected
    calls = p.remaining_batch_cache_info()['model_calls']
    for g in groups[:16]: g.seq.generated += 1
    assert provider.get_remaining_tokens_batch(groups) == [p.predict_remaining(
        prompt_len=g.seq.prompt, generated_len=g.seq.generated) for g in groups]
    assert p.remaining_batch_cache_info()['model_calls'] == calls + 1
    assert provider.prediction_calls == 160
    provider.model_len_predictor = other = trained()
    assert provider.get_remaining_tokens_batch(groups) == [other.predict_remaining(
        prompt_len=g.seq.prompt, generated_len=g.seq.generated) for g in groups]
    assert other.remaining_batch_cache_info()['model_calls'] > 0


def test_custom_predictor_uses_scalar_contract_and_errors():
    calls = []
    provider = DefaultDecodeSaturationStateProvider(load_predictor=False)
    provider.model_len_predictor = SimpleNamespace(
        model_type='qrf', predict_remaining=lambda **kwargs: calls.append(kwargs) or 7)
    request = SimpleNamespace(get_seqs=lambda: [SimpleNamespace(
        get_prompt_len=lambda: 16, get_output_len=lambda: 3)])
    assert provider.get_remaining_tokens_batch([request, request]) == [7, 7]
    assert len(calls) == 2 and provider.prediction_calls == 2
    def fail(**kwargs): raise ValueError('actual prediction failed')
    provider.model_len_predictor.predict_remaining = fail
    with pytest.raises(ValueError, match='actual prediction failed'):
        provider.get_remaining_tokens_batch([request])


def policy_with_provider(provider):
    policy = DecodeSaturationPolicy.__new__(DecodeSaturationPolicy)
    policy.enabled, policy.provider = True, provider
    policy.is_rt, policy.is_be = lambda g: True, lambda g: False
    policy._seq_group_blocks = lambda g: 1
    policy._future_arrivals = lambda now: []
    policy._get_engine = lambda: SimpleNamespace(compute_decision=lambda **kwargs: kwargs)
    return policy


def test_policy_batches_running_and_pending_but_keeps_scalar_provider_fallback():
    groups = [group(i) for i in range(3)]
    calls = []
    provider = SimpleNamespace(get_remaining_tokens_batch=lambda gs:
                               calls.append(list(gs)) or [0, 4, 7])
    policy = policy_with_provider(provider)
    result = policy.make_decision(groups[:1], [], groups[1:], 10)
    assert calls == [groups]
    assert [r.remaining_tokens for r in result['active_rt']] == [0, 4, 7]
    scalar_calls = []
    provider = SimpleNamespace(get_remaining_tokens=lambda g:
                               scalar_calls.append(g) or 5)
    policy = policy_with_provider(provider)
    result = policy.make_decision(groups[:1], [], groups[1:], 10)
    assert scalar_calls == groups
    assert [r.remaining_tokens for r in result['active_rt']] == [5, 5, 5]


@pytest.mark.parametrize('predictions', [[1], [None, 3]])
def test_policy_rejects_invalid_custom_batch_without_truncation(predictions):
    provider = SimpleNamespace(get_remaining_tokens_batch=lambda gs: predictions)
    with pytest.raises(RuntimeError):
        policy_with_provider(provider).make_decision([group(0)], [], [group(1)], 10)


def test_opt_in_phase_timing_reports_actual_batch_cache_counters(monkeypatch):
    monkeypatch.setenv('SATURO_STEP_TIMING', '1')
    provider = DefaultDecodeSaturationStateProvider(load_predictor=False)
    provider.model_len_predictor = trained()
    policy = policy_with_provider(provider)
    groups = [group(i) for i in range(3)]
    policy.make_decision(groups[:1], [], groups[1:], 10)
    first = policy.last_phase_timings
    assert first['qrf_model_calls'] == 1 and first['qrf_model_rows'] == 3
    assert first['qrf_cache_entries'] == 3
    assert first['qrf_cache_feature_bytes'] == 60
    assert first['active_state_ms'] >= 0
    policy.make_decision(groups[:1], [], groups[1:], 10)
    assert policy.last_phase_timings['qrf_model_calls'] == 1
    assert policy.last_phase_timings['qrf_hits'] == 3
    assert provider.prediction_calls == 6
