#!/usr/bin/env python3
"""Convert legacy experiment outputs into canonical plotting schemas."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.result_schema import (
    D_POOL_SCHEMA,
    E2E_SCHEMA,
    P_POOL_SCHEMA,
    write_bundle,
)


E2E_METRIC_FILES = {
    "rt_slo_attainment.csv": "rt_slo_attainment_pct",
    "be_throughput.csv": "be_throughput_req_s",
    "p99_ttft.csv": "p99_ttft_ms",
    "p99_tbt.csv": "p99_tbt_ms",
}


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes"}


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def normalize_split_e2e(
    input_dir: Path,
    *,
    baseline: str,
    model: str,
    run_id: str,
    num_instances: int | None = None,
    tp_size: int | None = None,
    ttft_slo_ms: float | None = None,
    tbt_slo_ms: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge the historical four two-column E2E CSVs on their RPS union."""
    by_rps: dict[float, dict[str, Any]] = {}
    sources = []
    for filename, metric in E2E_METRIC_FILES.items():
        path = input_dir / filename
        if not path.is_file():
            continue
        sources.append(str(path))
        with path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                rps = float(item["total_rt_rps"])
                by_rps.setdefault(rps, {"total_rt_rps": rps})[metric] = (
                    float(item[metric])
                )
    if not by_rps:
        raise ValueError(f"no legacy E2E metric CSVs found in {input_dir}")
    total_gpus = (
        num_instances * tp_size
        if num_instances is not None and tp_size is not None else None
    )
    rows = []
    for rps, values in sorted(by_rps.items()):
        rows.append({
            "total_rt_rps": rps,
            "rt_slo_attainment_pct": values.get(
                "rt_slo_attainment_pct"),
            "be_throughput_req_s": values.get("be_throughput_req_s"),
            "p99_ttft_ms": values.get("p99_ttft_ms"),
            "p99_tbt_ms": values.get("p99_tbt_ms"),
            "slo_feasible": None,
            "ttft_slo_ms": ttft_slo_ms,
            "tbt_slo_ms": tbt_slo_ms,
            "num_instances": num_instances,
            "tp_size": tp_size,
            "total_gpus": total_gpus,
            "p_instances": None,
            "d_instances": None,
            "result_source": ";".join(sources),
        })
    metadata = {
        "baseline": baseline,
        "model": model,
        "run_id": run_id,
        "source_format": "legacy_split_metric_csvs",
        "source_files": sources,
        "cluster": {
            "num_instances": num_instances,
            "tp_size": tp_size,
            "total_gpus": total_gpus,
        },
        "slo": {"ttft_ms": ttft_slo_ms, "tbt_ms": tbt_slo_ms},
        "missing_value_semantics": (
            "metric not available at this RPS; never interpolated"
        ),
    }
    return metadata, rows




def normalize_prefill_pool(
    path: Path,
    *,
    baseline: str,
    model: str | None,
    run_id: str,
    csv_slo_percentile: float = 0.999,
    csv_slo_ms: float = 1000.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        if not model:
            raise ValueError("--model is required for legacy Prefill CSV")
        rows = []
        for item in source_rows:
            p999_met = _bool(item.get("p999_met"))
            p90_met = _bool(item.get("p90_met"))
            p80_met = _bool(item.get("p80_met"))
            lower_bound = (
                99.9 if p999_met else 90.0 if p90_met
                else 80.0 if p80_met else 0.0
            )
            rows.append({
                "rt_rps": float(item["rps"]),
                "rt_requests": None,
                "rt_slo_percentile": csv_slo_percentile,
                "ttft_slo_ms": csv_slo_ms,
                "rt_slo_attainment_pct": lower_bound,
                "rt_slo_attainment_is_lower_bound": True,
                "rt_slo_met": p999_met,
                "rps_feasible": (
                    item.get("strongest_ttft_slo_met", "none") != "none"),
                "ttft_p50_ms": _optional_float(item.get("rt_ttft_p50_ms")),
                "ttft_p80_ms": _optional_float(item.get("rt_ttft_p80_ms")),
                "ttft_p90_ms": _optional_float(item.get("rt_ttft_p90_ms")),
                "ttft_p95_ms": _optional_float(item.get("rt_ttft_p95_ms")),
                "ttft_p99_ms": _optional_float(item.get("rt_ttft_p99_ms")),
                "ttft_p999_ms": _optional_float(item.get("rt_ttft_p999_ms")),
                "ttft_max_ms": _optional_float(item.get("rt_ttft_max_ms")),
                "be_throughput_req_s": float(item["be_prefill_req_per_s"]),
                "be_throughput_token_s": float(
                    item["be_prefill_token_per_s"]),
                "selected_chunk_tokens": _optional_int(
                    item.get("selected_chunk_tokens")),
                "source": str(path),
            })
        workload: dict[str, Any] = {}
        operating: dict[str, Any] = {}
        resolved_model = model
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        workload = dict(payload.get("workload", {}))
        operating = dict(payload.get("operating_config", {}))
        source_rows = list(payload["rows"])
        resolved_model = model or str(workload.get("model", ""))
        percentile = float(operating.get("rt_slo_percentile", 0.999))
        slo_ms = 1000.0 * float(operating.get("rt_ttft_slo_s", 1.0))
        rows = []
        for item in source_rows:
            count = int(item.get("num_req", 0))
            violations = int(item.get("rt_slo_violations", 0))
            attainment = (
                100.0 if count <= 0 and violations <= 0
                else 0.0 if count <= 0
                else 100.0 * (1.0 - violations / count)
            )
            percentile_ms = _optional_float(
                item.get("rt_slo_percentile_ttft_ms"))
            rows.append({
                "rt_rps": float(item["rps"]),
                "rt_requests": count,
                "rt_slo_percentile": percentile,
                "ttft_slo_ms": slo_ms,
                "rt_slo_attainment_pct": attainment,
                "rt_slo_attainment_is_lower_bound": False,
                "rt_slo_met": _bool(item.get("rt_slo_met")),
                "rps_feasible": _bool(item.get("rps_feasible"), default=True),
                "ttft_p50_ms": _optional_float(item.get("rt_ttft_p50_ms")),
                "ttft_p80_ms": _optional_float(item.get("rt_ttft_p80_ms")),
                "ttft_p90_ms": _optional_float(item.get("rt_ttft_p90_ms")),
                "ttft_p95_ms": _optional_float(item.get("rt_ttft_p95_ms")),
                "ttft_p99_ms": _optional_float(item.get("rt_ttft_p99_ms")),
                "ttft_p999_ms": _optional_float(
                    item.get("rt_ttft_p999_ms")
                    or (percentile_ms if percentile == 0.999 else None)),
                "ttft_max_ms": _optional_float(item.get("rt_ttft_max_ms")),
                "be_throughput_req_s": float(item["be_prefill_req_per_s"]),
                "be_throughput_token_s": float(
                    item["be_prefill_token_per_s"]),
                "selected_chunk_tokens": _optional_int(
                    item.get("be_prefill_chunk_size")),
                "source": str(path),
            })
    if not resolved_model:
        raise ValueError(f"cannot determine model for {path}")
    metadata = {
        "baseline": baseline,
        "model": resolved_model,
        "run_id": run_id,
        "source_format": "legacy_prefill_csv" if path.suffix == ".csv"
        else "legacy_prefill_json",
        "source_files": [str(path)],
        "workload": workload,
        "operating_config": operating,
    }
    return metadata, rows


def normalize_decode_pool(
    path: Path,
    *,
    baseline: str,
    model: str | None,
    run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    workload = dict(payload.get("workload", {}))
    operating = dict(payload.get("operating_config", {}))
    resolved_model = model or str(workload.get("model", ""))
    if not resolved_model:
        raise ValueError(f"cannot determine model for {path}")
    slo_ms = float(_first(operating, "tbt_slo_ms", "rt_tbt_slo_ms"))
    sequence_slots = _optional_int(operating.get("sequence_slots"))
    rows = []
    for item in payload["rows"]:
        count = int(_first(item, "num_req", "total_rt_requests") or 0)
        failures = int(item.get("rt_admission_failures", 0))
        admission = _optional_float(item.get("rt_admission_attainment"))
        if admission is None:
            admission = 1.0 if count <= 0 else 1.0 - failures / count
        tbt = float(item.get("rt_tbt_attainment", 0.0))
        overall = _optional_float(item.get("rt_slo_attainment"))
        if overall is None:
            overall = min(tbt, admission)
        rows.append({
            "rt_rps": float(item["rps"]),
            "rt_requests": count,
            "tbt_slo_ms": slo_ms,
            "rt_slo_attainment_pct": 100.0 * overall,
            "tbt_slo_attainment_pct": 100.0 * tbt,
            "admission_attainment_pct": 100.0 * admission,
            "rt_slo_met": _bool(item.get("rt_slo_met")),
            "rps_feasible": _bool(item.get("rps_feasible"), default=True),
            "tbt_p50_ms": _optional_float(item.get("rt_tbt_p50_ms")),
            "tbt_p95_ms": _optional_float(item.get("rt_tbt_p95_ms")),
            "tbt_p99_ms": _optional_float(item.get("rt_tbt_p99_ms")),
            "tbt_max_ms": _optional_float(item.get("rt_tbt_max_ms")),
            "be_throughput_req_s": float(item["be_decode_req_per_s"]),
            "be_throughput_token_s": float(_first(
                item,
                "be_decode_token_per_s",
                "be_throughput_tok_per_s",
            ) or 0.0),
            "be_completed_req_s": _optional_float(
                item.get("be_completed_req_per_s")),
            "sequence_slots": sequence_slots,
            "source": str(path),
        })
    metadata = {
        "baseline": baseline,
        "model": resolved_model,
        "run_id": run_id,
        "source_format": "legacy_decode_json",
        "source_files": [str(path)],
        "workload": workload,
        "operating_config": operating,
    }
    return metadata, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--baseline", required=True)
        subparser.add_argument("--model", required=True)
        subparser.add_argument("--run-id", required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)

    e2e = subparsers.add_parser("e2e-split")
    common(e2e)
    e2e.add_argument("--input-dir", type=Path, required=True)
    e2e.add_argument("--num-instances", type=int)
    e2e.add_argument("--tp-size", type=int)
    e2e.add_argument("--ttft-slo-ms", type=float)
    e2e.add_argument("--tbt-slo-ms", type=float)

    prefill = subparsers.add_parser("prefill-pool")
    common(prefill)
    prefill.add_argument("--input", type=Path, required=True)
    prefill.add_argument("--slo-percentile", type=float, default=0.999)
    prefill.add_argument("--ttft-slo-ms", type=float, default=1000.0)

    decode = subparsers.add_parser("decode-pool")
    common(decode)
    decode.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "e2e-split":
        metadata, rows = normalize_split_e2e(
            args.input_dir,
            baseline=args.baseline,
            model=args.model,
            run_id=args.run_id,
            num_instances=args.num_instances,
            tp_size=args.tp_size,
            ttft_slo_ms=args.ttft_slo_ms,
            tbt_slo_ms=args.tbt_slo_ms,
        )
        schema, basename = E2E_SCHEMA, "e2e_summary"
    elif args.command == "prefill-pool":
        metadata, rows = normalize_prefill_pool(
            args.input,
            baseline=args.baseline,
            model=args.model,
            run_id=args.run_id,
            csv_slo_percentile=args.slo_percentile,
            csv_slo_ms=args.ttft_slo_ms,
        )
        schema, basename = P_POOL_SCHEMA, "pool_summary"
    else:
        metadata, rows = normalize_decode_pool(
            args.input,
            baseline=args.baseline,
            model=args.model,
            run_id=args.run_id,
        )
        schema, basename = D_POOL_SCHEMA, "pool_summary"
    json_path, csv_path = write_bundle(
        args.output_dir,
        basename=basename,
        schema_version=schema,
        metadata=metadata,
        rows=rows,
    )
    print(f"[wrote] {json_path}")
    print(f"[wrote] {csv_path}")


if __name__ == "__main__":
    main()
