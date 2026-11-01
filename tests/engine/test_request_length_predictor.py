from vllm.engine.request_length_predictor import (
    GBDTRemainingLengthPredictor,
    QRFRemainingLengthPredictor,
    create_length_predictor,
    normalize_length_predictor_type,
)


def test_length_predictor_type_aliases_cover_gbdt_and_qrf():
    assert normalize_length_predictor_type("qrf") == "qrf"
    assert normalize_length_predictor_type("gbdt") == "gbdt"
    assert normalize_length_predictor_type("gbpt") == "gbdt"
    assert normalize_length_predictor_type("gbrt") == "gbdt"
    assert normalize_length_predictor_type("gradient-boosted-trees") == "gbdt"


def test_create_length_predictor_supports_qrf_and_gbdt_aliases():
    assert isinstance(create_length_predictor("qrf"), QRFRemainingLengthPredictor)

    for alias in ("gbdt", "gbpt", "gbrt"):
        predictor = create_length_predictor(alias, quantiles=(0.5, 0.95))
        assert isinstance(predictor, GBDTRemainingLengthPredictor)
        assert predictor.model_type == "gbdt"
