"""Canonical experiment-result schemas for Saturo measurements.

The canonical CSV files are intentionally flat and stable for plotting.  The
matching JSON files carry the same rows plus structured run metadata.  Raw
simulator output remains run-specific and is referenced through the
``source`` column rather than embedded in the plotting schema.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


E2E_SCHEMA = "hybrid_serve.e2e_rps.v1"
P_POOL_SCHEMA = "hybrid_serve.prefill_pool_rps.v1"
D_POOL_SCHEMA = "hybrid_serve.decode_pool_rps.v1"

COMMON_FIELDS = [
    "schema_version",
    "kind",
    "baseline",
    "model",
    "run_id",
]

E2E_FIELDS = COMMON_FIELDS + [
    "total_rt_rps",
    "rt_slo_attainment_pct",
    "be_throughput_req_s",
    "p99_ttft_ms",
    "p99_tbt_ms",
    "slo_feasible",
    "ttft_slo_ms",
    "tbt_slo_ms",
    "num_instances",
    "tp_size",
    "total_gpus",
    "p_instances",
    "d_instances",
    "result_source",
]

P_POOL_FIELDS = COMMON_FIELDS + [
    "rt_rps",
    "rt_requests",
    "rt_slo_percentile",
    "ttft_slo_ms",
    "rt_slo_attainment_pct",
    "rt_slo_attainment_is_lower_bound",
    "rt_slo_met",
    "rps_feasible",
    "ttft_p50_ms",
    "ttft_p80_ms",
    "ttft_p90_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "ttft_p999_ms",
    "ttft_max_ms",
    "be_throughput_req_s",
    "be_throughput_token_s",
    "selected_chunk_tokens",
    "source",
]

D_POOL_FIELDS = COMMON_FIELDS + [
    "rt_rps",
    "rt_requests",
    "tbt_slo_ms",
    "rt_slo_attainment_pct",
    "tbt_slo_attainment_pct",
    "admission_attainment_pct",
    "rt_slo_met",
    "rps_feasible",
    "tbt_p50_ms",
    "tbt_p95_ms",
    "tbt_p99_ms",
    "tbt_max_ms",
    "be_throughput_req_s",
    "be_throughput_token_s",
    "be_completed_req_s",
    "sequence_slots",
    "source",
]

SCHEMA_FIELDS = {
    E2E_SCHEMA: E2E_FIELDS,
    P_POOL_SCHEMA: P_POOL_FIELDS,
    D_POOL_SCHEMA: D_POOL_FIELDS,
}

SCHEMA_KINDS = {
    E2E_SCHEMA: "e2e_rps_sweep",
    P_POOL_SCHEMA: "prefill_pool_rps_sweep",
    D_POOL_SCHEMA: "decode_pool_rps_sweep",
}

BOOL_FIELDS = {
    "slo_feasible",
    "rt_slo_attainment_is_lower_bound",
    "rt_slo_met",
    "rps_feasible",
}
INT_FIELDS = {
    "num_instances",
    "tp_size",
    "total_gpus",
    "p_instances",
    "d_instances",
    "rt_requests",
    "selected_chunk_tokens",
    "sequence_slots",
}
TEXT_FIELDS = {
    *COMMON_FIELDS,
    "result_source",
    "source",
}


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def _parse_csv_value(field: str, value: str) -> Any:
    if value == "":
        return None
    if field in BOOL_FIELDS:
        lower = value.strip().lower()
        if lower not in {"true", "false"}:
            raise ValueError(f"invalid boolean {field}={value!r}")
        return lower == "true"
    if field in INT_FIELDS:
        return int(value)
    if field in TEXT_FIELDS:
        return value
    return float(value)


def _row_with_metadata(
    row: dict[str, Any],
    *,
    schema_version: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "kind": SCHEMA_KINDS[schema_version],
        "baseline": metadata["baseline"],
        "model": metadata["model"],
        "run_id": metadata["run_id"],
        **row,
    }


def validate_rows(
    schema_version: str,
    metadata: dict[str, Any],
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    if schema_version not in SCHEMA_FIELDS:
        raise ValueError(f"unknown result schema: {schema_version}")
    for field in ("baseline", "model", "run_id"):
        if not str(metadata.get(field, "")).strip():
            raise ValueError(f"missing metadata field {field!r}")

    result = list(rows)
    rps_field = "total_rt_rps" if schema_version == E2E_SCHEMA else "rt_rps"
    seen = set()
    for index, row in enumerate(result):
        if rps_field not in row:
            raise ValueError(f"row {index} missing {rps_field!r}")
        rps = float(row[rps_field])
        if rps < 0 or rps in seen:
            raise ValueError(f"invalid/duplicate {rps_field}: {rps}")
        seen.add(rps)
        attainment = row.get("rt_slo_attainment_pct")
        if attainment is not None and not 0.0 <= float(attainment) <= 100.0:
            raise ValueError(f"invalid RT-SLO attainment: {attainment}")
        be = row.get("be_throughput_req_s")
        if be is not None and float(be) < 0:
            raise ValueError(f"negative BE throughput: {be}")
    stable_row_fields = [
        field for field in SCHEMA_FIELDS[schema_version]
        if field not in COMMON_FIELDS
    ]
    stable_rows = [
        {field: row.get(field) for field in stable_row_fields}
        for row in result
    ]
    return sorted(stable_rows, key=lambda row: float(row[rps_field]))


def write_bundle(
    output_dir: Path,
    *,
    basename: str,
    schema_version: str,
    metadata: dict[str, Any],
    rows: Iterable[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write one canonical JSON/CSV pair and return their paths."""
    normalized = validate_rows(schema_version, metadata, rows)
    fields = SCHEMA_FIELDS[schema_version]
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{basename}.json"
    csv_path = output_dir / f"{basename}.csv"
    payload = {
        "schema_version": schema_version,
        "kind": SCHEMA_KINDS[schema_version],
        "metadata": metadata,
        "rows": normalized,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in normalized:
            flattened = _row_with_metadata(
                row,
                schema_version=schema_version,
                metadata=metadata,
            )
            writer.writerow({field: _csv_value(flattened.get(field))
                             for field in fields})
    return json_path, csv_path


def load_bundle(
    path: Path,
    *,
    expected_schema: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a canonical JSON or CSV result bundle."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = str(payload.get("schema_version", ""))
        if expected_schema is not None and schema != expected_schema:
            raise ValueError(f"expected {expected_schema}, got {schema}")
        metadata = dict(payload["metadata"])
        rows = validate_rows(schema, metadata, payload["rows"])
        return metadata, rows

    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"empty canonical CSV: {path}")
    schema = raw_rows[0]["schema_version"]
    if expected_schema is not None and schema != expected_schema:
        raise ValueError(f"expected {expected_schema}, got {schema}")
    metadata = {
        field: raw_rows[0][field]
        for field in ("baseline", "model", "run_id")
    }
    row_fields = [
        field for field in SCHEMA_FIELDS[schema]
        if field not in COMMON_FIELDS
    ]
    rows = [
        {field: _parse_csv_value(field, row[field])
         for field in row_fields}
        for row in raw_rows
    ]
    return metadata, validate_rows(schema, metadata, rows)


def read_e2e_series(path: Path, metric: str) -> list[tuple[float, float]]:
    """Read one non-null metric series from a canonical E2E bundle."""
    if metric not in E2E_FIELDS:
        raise ValueError(f"unknown E2E metric: {metric}")
    _, rows = load_bundle(path, expected_schema=E2E_SCHEMA)
    return [
        (float(row["total_rt_rps"]), float(row[metric]))
        for row in rows
        if row.get(metric) is not None
    ]
