"""Bitwise parity of scalar and batch sufficient-statistics timing predictions."""
from pathlib import Path
import types

import numpy as np
import pytest

from saturo.predictors.time_predictor import TimePredictor, RuntimeOverheadModel
NAMES = ('prefill_tokens', 'prefill_sum_ctx', 'decode_tokens', 'decode_sum_ctx',
         'prefill_request_count', 'decode_request_count')
PROFILE = Path(__file__).resolve().parents[2] / 'output/e2e_validation_20261031/runtime_iteration/root-saturo-models-qwen2.5-0.5b/time_predictor_root-saturo-models-qwen2.5-0.5b_gpu-nvidia-geforce-rtx-4090_tp1_pp1_torch.float16_d39ce41be825e0c8.json'


def assert_parity(predictor, **values):
    arrays = np.broadcast_arrays(*(np.asarray(values[k]) for k in NAMES))
    expected = np.empty(arrays[0].shape, dtype=np.float64)
    with np.errstate(all='ignore'):
        for index in np.ndindex(expected.shape):
            expected[index] = predictor.predict_forward_from_aggregates_ms(
                **{name: arr[index] for name, arr in zip(NAMES, arrays)})
        actual = predictor.predict_forward_from_aggregates_batch_ms(**values)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint64),
                                  expected.view(np.uint64))


def random_values(seed=123, rows=10000):
    rng = np.random.default_rng(seed)
    return dict(prefill_tokens=rng.integers(0, 1025, rows).astype(float),
                prefill_sum_ctx=rng.uniform(0, 1e7, rows),
                decode_tokens=rng.integers(0, 2001, rows).astype(float),
                decode_sum_ctx=rng.uniform(0, 1e7, rows),
                prefill_request_count=rng.uniform(-4, 64, rows),
                decode_request_count=rng.uniform(-4, 2000, rows))


def configured():
    p = TimePredictor('/does-not-exist', warn_missing=False)
    p.model_type = 'iteration_level'
    p.iter_coef = [.01, -1e-6, .02, 1e-5]
    p.iter_intercept = -3.1
    p.latency_scale, p.latency_offset_ms = 1.17, -.3
    p.runtime_overhead = RuntimeOverheadModel(
        constant_ms=-.2, per_query_token_ms=.002,
        query_token_hinge_threshold=19.5,
        per_query_token_above_hinge_ms=-.003,
        per_prefill_request_ms=.011, per_decode_request_ms=.013,
        per_active_request_ms=-.005, per_prefill_token_ms=.007,
        per_decode_token_ms=-.001, decode_request_hinge_threshold=10.5,
        per_decode_request_above_hinge_ms=.004)
    return p


@pytest.mark.skipif(not PROFILE.is_file(), reason='optional measured timing artifact unavailable')
def test_real_profile_matches_all_scalar_bits():
    assert_parity(TimePredictor(PROFILE), **random_values(rows=15000))


@pytest.mark.parametrize('overhead', [False, True])
@pytest.mark.parametrize('coef', [None, [.01, -1e-6, .02, 1e-5],
                                 [-.1, -.2, -.3, -.4]])
def test_all_calibration_overhead_clamp_stages(overhead, coef):
    p = configured()
    p.apply_runtime_overhead, p.iter_coef = overhead, coef
    assert_parity(p, **random_values(rows=3000))


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64])
def test_dtype_promotion_and_numeric_edges(dtype):
    p = configured()
    values = random_values(rows=100)
    values = {name: arr.astype(dtype) for name, arr in values.items()}
    assert_parity(p, **values)


def test_scalar_broadcast_empty_zero_work_and_signed_zero():
    p = configured()
    for decode in [np.array([]), np.array([0., -0., 1., 2., 3.])]:
        assert_parity(p, prefill_tokens=0., prefill_sum_ctx=0.,
                      decode_tokens=decode, decode_sum_ctx=decode * 100,
                      prefill_request_count=0, decode_request_count=decode)
    assert_parity(p, prefill_tokens=0., prefill_sum_ctx=0., decode_tokens=0.,
                  decode_sum_ctx=0., prefill_request_count=0,
                  decode_request_count=0)
    assert_parity(p, prefill_tokens=0., prefill_sum_ctx=float('inf'),
                  decode_tokens=0., decode_sum_ctx=float('nan'),
                  prefill_request_count=float('nan'), decode_request_count=None)


def test_nonfinite_features_and_huge_request_counts_fallback():
    p = configured()
    assert_parity(p, prefill_tokens=np.array([float('inf'), float('nan'), 1.]),
                  prefill_sum_ctx=0., decode_tokens=1., decode_sum_ctx=1.,
                  prefill_request_count=2**80, decode_request_count=2**80)


def test_custom_scalar_override_is_not_bypassed():
    p = configured()
    original = p.predict_forward_from_aggregates_ms
    calls = []
    def override(self, **kw):
        calls.append(kw)
        return original(**kw) + 3.25
    p.predict_forward_from_aggregates_ms = types.MethodType(override, p)
    assert_parity(p, **random_values(rows=11))
    assert len(calls) == 22


def test_custom_calibration_hook_is_not_bypassed():
    p = configured()
    p._calibrate_ms = types.MethodType(lambda self, value: value + 1.23, p)
    assert_parity(p, **random_values(rows=11))


def test_legacy_per_layer_model_keeps_scalar_implementation():
    p = configured()
    p.model_type = 'per_layer'
    p.num_layers = 24
    p.prefill_coef = (.001, 1e-7, .01)
    p.decode_coef = (.003, 2e-7, .02)
    p.linear_model.coef = {0: (.001, .01)}
    p.linear_model.fallback = (.001, .02)
    assert_parity(p, **random_values(rows=111))


def test_disabled_overhead_never_touches_unused_model():
    p = configured()
    p.apply_runtime_overhead = False
    p.runtime_overhead = None
    assert_parity(p, **random_values(rows=111))
