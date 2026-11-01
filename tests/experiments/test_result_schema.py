import csv

import pytest

from experiments.normalize_results import normalize_split_e2e
from experiments.result_schema import (
    E2E_SCHEMA,
    load_bundle,
    read_e2e_series,
    write_bundle,
)


def test_e2e_bundle_round_trip_preserves_sparse_metrics(tmp_path):
    metadata = {
        "baseline": "Saturo",
        "model": "Qwen2.5-72B-Instruct",
        "run_id": "test",
    }
    rows = [
        {"total_rt_rps": 1.0, "rt_slo_attainment_pct": 100.0,
         "be_throughput_req_s": 5.0},
        {"total_rt_rps": 2.0, "rt_slo_attainment_pct": 99.0,
         "be_throughput_req_s": None, "slo_feasible": False},
    ]

    json_path, csv_path = write_bundle(
        tmp_path,
        basename="e2e_summary",
        schema_version=E2E_SCHEMA,
        metadata=metadata,
        rows=rows,
    )

    json_metadata, json_rows = load_bundle(
        json_path, expected_schema=E2E_SCHEMA)
    csv_metadata, csv_rows = load_bundle(
        csv_path, expected_schema=E2E_SCHEMA)
    assert json_metadata["model"] == csv_metadata["model"]
    assert json_rows[1]["be_throughput_req_s"] is None
    assert csv_rows[1]["be_throughput_req_s"] is None
    assert csv_rows[1]["slo_feasible"] is False
    assert read_e2e_series(csv_path, "be_throughput_req_s") == [(1.0, 5.0)]


def test_split_e2e_normalization_uses_union_without_interpolation(tmp_path):
    sources = {
        "rt_slo_attainment.csv": [
            (1.0, 100.0),
            (2.0, 90.0),
        ],
        "be_throughput.csv": [
            (1.0, 5.0),
            (3.0, 4.0),
        ],
    }
    for filename, values in sources.items():
        metric = (
            "rt_slo_attainment_pct"
            if filename.startswith("rt_") else "be_throughput_req_s"
        )
        with (tmp_path / filename).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["total_rt_rps", metric])
            writer.writeheader()
            for rps, value in values:
                writer.writerow({"total_rt_rps": rps, metric: value})

    _, rows = normalize_split_e2e(
        tmp_path,
        baseline="Saturo",
        model="Mistral-Large-Instruct-2411",
        run_id="test",
    )

    assert [row["total_rt_rps"] for row in rows] == [1.0, 2.0, 3.0]
    assert rows[1]["be_throughput_req_s"] is None
    assert rows[2]["rt_slo_attainment_pct"] is None


def test_e2e_schema_rejects_invalid_attainment(tmp_path):
    with pytest.raises(ValueError, match="invalid RT-SLO attainment"):
        write_bundle(
            tmp_path,
            basename="bad",
            schema_version=E2E_SCHEMA,
            metadata={
                "baseline": "Saturo",
                "model": "model",
                "run_id": "bad",
            },
            rows=[{
                "total_rt_rps": 1.0,
                "rt_slo_attainment_pct": 101.0,
            }],
        )
