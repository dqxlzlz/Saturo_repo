import csv

import pytest

from saturo.runtime.workload import load_rt_workload_csv


def _write_trace(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "request_id",
                "prompt_len",
                "output_len",
                "arrival_offset_s",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_target_rps_preserves_arrival_shape_and_request_pairing(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("RT_TRIM_TOP_FRAC", raising=False)
    monkeypatch.delenv("RT_LEN_SCALE", raising=False)
    trace = tmp_path / "rt.csv"
    _write_trace(trace, [
        {"request_id": "at_10", "prompt_len": 10, "output_len": 1,
         "arrival_offset_s": 10},
        {"request_id": "at_0", "prompt_len": 20, "output_len": 2,
         "arrival_offset_s": 0},
        {"request_id": "at_5", "prompt_len": 30, "output_len": 3,
         "arrival_offset_s": 5},
        {"request_id": "at_15", "prompt_len": 40, "output_len": 4,
         "arrival_offset_s": 15},
    ])

    requests = load_rt_workload_csv(
        str(trace), num_req=3, target_rps=2.0)

    assert [request.request_id for request in requests] == [
        "at_0", "at_5", "at_10"
    ]
    # The selected prefix has native RPS=3/10 and is scaled into 3/2 seconds.
    assert [request.arrival_offset_s for request in requests] == pytest.approx(
        [0.0, 0.75, 1.5]
    )
    assert [request.prompt_len for request in requests] == [20, 30, 10]


def test_fixed_num_requires_enough_source_rows(tmp_path, monkeypatch):
    monkeypatch.delenv("RT_TRIM_TOP_FRAC", raising=False)
    monkeypatch.delenv("RT_LEN_SCALE", raising=False)
    trace = tmp_path / "sparse_prefix.csv"
    _write_trace(trace, [
        {"request_id": str(index), "prompt_len": 10, "output_len": 1,
         "arrival_offset_s": arrival}
        for index, arrival in enumerate([0, 10, 11, 12, 13, 14])
    ])

    with pytest.raises(ValueError, match="num_req=7"):
        load_rt_workload_csv(str(trace), num_req=7, target_rps=1.0)


def test_fixed_num_scales_selected_prefix_to_derived_window(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("RT_TRIM_TOP_FRAC", raising=False)
    monkeypatch.delenv("RT_LEN_SCALE", raising=False)
    trace = tmp_path / "sparse_prefix.csv"
    _write_trace(trace, [
        {"request_id": str(index), "prompt_len": 10, "output_len": 1,
         "arrival_offset_s": arrival}
        for index, arrival in enumerate([0, 10, 11, 12, 13, 14])
    ])

    requests = load_rt_workload_csv(
        str(trace), num_req=3, target_rps=1.0)

    assert [request.request_id for request in requests] == ["0", "1", "2"]
    assert [request.arrival_offset_s for request in requests] == pytest.approx(
        [0.0, 30.0 / 11.0, 3.0])
