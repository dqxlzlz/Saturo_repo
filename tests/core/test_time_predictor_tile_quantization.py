import json

import pytest

from saturo.predictors.time_predictor import TimePredictor


def test_linear_tile_override_supports_efficiency_drops_and_bounded_range(
        tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "time_predictor": {
            "model_type": "per_layer",
            "num_layers": 2,
            "prefill_attn": {"coef": [0, 0], "intercept": 0},
            "decode_attn": {"coef": [0, 0], "intercept": 0},
            "linear": {
                "width": 32,
                "coef": {},
                "fallback": [0, 10],
            },
            "linear_tile_override": {
                "tile_width": 128,
                "max_tokens": 256,
                "token_upper_bounds": [16, 128, 256],
                "per_layer_values_ms": [3, 2, 1],
            },
        },
    }), encoding="utf-8")

    predictor = TimePredictor(str(profile))

    assert predictor.predict_linear_ms(1) == pytest.approx(3)
    assert predictor.predict_linear_ms(16) == pytest.approx(3)
    assert predictor.predict_linear_ms(17) == pytest.approx(2)
    assert predictor.predict_linear_ms(128) == pytest.approx(2)
    # Larger tiny batches can be faster because GPU utilization improves.
    assert predictor.predict_linear_ms(129) == pytest.approx(1)
    assert predictor.predict_linear_ms(256) == pytest.approx(1)
    # Above the measured range, retain the legacy model's extrapolation.
    assert predictor.predict_linear_ms(257) == pytest.approx(10)
    assert predictor.predict_forward_ms([], [(1, 100)] * 129) == pytest.approx(2)
