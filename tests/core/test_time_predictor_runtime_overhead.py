from saturo.predictors.time_predictor import RuntimeOverheadModel


def test_query_token_correction_is_independent_of_request_mix():
    model = RuntimeOverheadModel(
        constant_ms=-5.677086,
        per_query_token_ms=0.090377,
        query_token_hinge_threshold=384,
        per_query_token_above_hinge_ms=-0.028164,
    )

    decode_only = model.predict_ms(
        prefill_request_count=0,
        decode_request_count=512,
        prefill_tokens=0,
        decode_tokens=512,
    )
    mixed = model.predict_ms(
        prefill_request_count=1,
        decode_request_count=256,
        prefill_tokens=256,
        decode_tokens=256,
    )

    assert decode_only == mixed


def test_query_token_correction_round_trips():
    model = RuntimeOverheadModel()
    model.load_dict({
        "per_query_token_ms": 0.09,
        "query_token_hinge_threshold": 384,
        "per_query_token_above_hinge_ms": -0.03,
    })

    payload = model.to_dict()
    assert payload["per_query_token_ms"] == 0.09
    assert payload["query_token_hinge_threshold"] == 384
    assert payload["per_query_token_above_hinge_ms"] == -0.03


def test_decode_request_hinge_only_applies_above_threshold():
    model = RuntimeOverheadModel(
        per_decode_request_ms=0.05,
        decode_request_hinge_threshold=512,
        per_decode_request_above_hinge_ms=0.10,
    )

    assert model.predict_ms(
        prefill_request_count=0,
        decode_request_count=256,
        prefill_tokens=0,
        decode_tokens=256,
    ) == 12.8
    assert model.predict_ms(
        prefill_request_count=0,
        decode_request_count=768,
        prefill_tokens=0,
        decode_tokens=768,
    ) == 64.0


def test_decode_request_hinge_round_trips():
    model = RuntimeOverheadModel()
    model.load_dict({
        "decode_request_hinge_threshold": 384,
        "per_decode_request_above_hinge_ms": 0.125,
    })

    payload = model.to_dict()
    assert payload["decode_request_hinge_threshold"] == 384
    assert payload["per_decode_request_above_hinge_ms"] == 0.125
