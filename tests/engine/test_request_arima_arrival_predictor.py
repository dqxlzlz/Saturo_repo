import math

from vllm.engine.request_arima_arrival_predictor import ARIMAArrivalPredictor


def test_arima_arrival_predictor_forecasts_inter_arrival_trend():
    predictor = ARIMAArrivalPredictor(min_history_intervals=4)
    predictor.observe_arrivals([0.0, 1.0, 2.2, 3.6, 5.2])

    intervals = predictor.forecast_inter_arrivals(window_s=8.0)

    assert intervals
    assert intervals[0] > 1.6


def test_arima_arrival_predictor_predicts_offsets_from_inter_arrivals():
    predictor = ARIMAArrivalPredictor(min_history_intervals=3)
    predictor.observe_arrivals([0.0, 0.5, 1.0, 1.5, 2.0])

    offsets = predictor.predict_arrival_offsets(window_s=2.0)

    assert offsets == sorted(offsets)
    assert all(0.0 < offset <= 2.0 for offset in offsets)
    assert offsets == [0.5, 1.0, 1.5, 2.0]


def test_arima_arrival_predictor_conditions_offsets_on_current_time():
    predictor = ARIMAArrivalPredictor(min_history_intervals=3)
    predictor.observe_arrivals([0.0, 1.0, 2.0, 3.0])

    offsets = predictor.predict_arrival_offsets(window_s=2.0, now_s=3.4)

    assert len(offsets) == 2
    assert math.isclose(offsets[0], 0.6)
    assert math.isclose(offsets[1], 1.6)


def test_arima_arrival_predictor_does_not_collapse_bursty_history():
    predictor = ARIMAArrivalPredictor(min_history_intervals=4)
    predictor.observe_arrivals([
        0.0, 6.308703, 6.641054, 9.228609,
        11.325332, 12.065100, 12.721301,
    ])

    offsets = predictor.predict_arrival_offsets(
        window_s=5.0, now_s=12.761491)

    assert offsets
    assert offsets[0] > 0.1
    assert len(offsets) < 14
    assert offsets[-1] > 1.0
