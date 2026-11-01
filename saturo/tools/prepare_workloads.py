#!/usr/bin/env python3
"""Prepare RT and BE workload CSVs from Azure trace and LongBench datasets."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def _parse_timestamp(s: str) -> datetime:
    s = s.strip()
    if "." in s:
        head, frac = s.rsplit(".", 1)
        return datetime.strptime(f"{head}.{frac[:6]}", "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def prepare_rt_from_azure(
    azure_csv: Path,
    out_csv: Path,
    max_model_len: int,
    num_req: int,
    target_rps: float,
) -> int:
    """Write exactly ``num_req`` chronological Azure requests at target RPS."""
    from saturo.tools.trace_scaler import scale_to_target_rps_fixed_num

    rows = []
    with azure_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = int(float(row.get("ContextTokens") or row.get("prompt_len") or 1))
            output = int(float(row.get("GeneratedTokens") or row.get("output_len") or 1))
            prompt = max(1, min(prompt, max_model_len - 2))
            output = max(1, min(output, max_model_len - prompt))
            rows.append((row.get("TIMESTAMP", ""), prompt, output))

    if len(rows) < num_req:
        raise ValueError(
            f"Azure trace has {len(rows)} requests, but num_req={num_req}")

    # Parse and normalise arrival times
    if rows[0][0]:
        try:
            times = [_parse_timestamp(r[0]) for r in rows]
            base = times[0]
            raw = [(t - base).total_seconds() for t in times]
        except Exception:
            raw = [i for i in range(len(rows))]
    else:
        raw = [i for i in range(len(rows))]

    paired = sorted(zip(raw, rows), key=lambda item: item[0])[:num_req]
    selected_rows = [row for _, row in paired]
    arrivals = scale_to_target_rps_fixed_num(
        [arrival for arrival, _ in paired],
        num_req=num_req,
        target_rps=target_rps,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["request_id", "prompt_len", "output_len", "arrival_offset_s"])
        writer.writeheader()
        for idx, (_, prompt, output) in enumerate(selected_rows):
            arrival = arrivals[idx]
            writer.writerow({
                "request_id": f"rt_{idx}",
                "prompt_len": prompt,
                "output_len": output,
                "arrival_offset_s": round(arrival, 6),
            })
    print(f"[RT] Wrote {num_req} requests to {out_csv}")
    return num_req


def prepare_be_from_longbench(
    longbench_dir: Path,
    out_csv: Path,
    max_model_len: int,
    max_rows: int = 512,
) -> int:
    """Convert LongBench JSONL files to BE workload CSV."""
    entries = []
    for jf in sorted(longbench_dir.glob("*.jsonl")):
        with jf.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # context length in tokens (the 'length' field)
                prompt_len = int(obj.get("length") or 512)
                # output length: estimate from answer length (rough word count)
                answer = obj.get("answers") or []
                if isinstance(answer, list) and answer:
                    answer_text = str(answer[0])
                else:
                    answer_text = str(answer)
                output_len = max(1, min(len(answer_text.split()), 256))
                prompt_len = max(1, min(prompt_len, max_model_len - output_len - 1))
                output_len = max(1, min(output_len, max_model_len - prompt_len))
                entries.append((prompt_len, output_len))
                if len(entries) >= max_rows:
                    break
        if len(entries) >= max_rows:
            break

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["request_id", "prompt_len", "output_len", "arrival_offset_s"])
        writer.writeheader()
        for idx, (prompt, output) in enumerate(entries):
            writer.writerow({
                "request_id": f"be_{idx}",
                "prompt_len": prompt,
                "output_len": output,
                "arrival_offset_s": 0.0,
            })
    print(f"[BE] Wrote {len(entries)} requests to {out_csv}")
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--azure-csv",
                        default="data/datasets/AzureLLMInferenceTrace_conv.csv")
    parser.add_argument("--longbench-dir", default="data/longbench/data")
    parser.add_argument("--out-dir", default="saturo/results/workloads_e2e")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--num-req", type=int, required=True)
    parser.add_argument("--target-rps", type=float, required=True)
    parser.add_argument("--max-be", type=int, default=512)
    args = parser.parse_args()
    if args.num_req <= 0 or args.target_rps <= 0:
        parser.error("--num-req and --target-rps must be positive")

    out_dir = Path(args.out_dir)
    prepare_rt_from_azure(
        azure_csv=Path(args.azure_csv),
        out_csv=out_dir / "rt_workload.csv",
        max_model_len=args.max_model_len,
        num_req=args.num_req,
        target_rps=args.target_rps,
    )
    prepare_be_from_longbench(
        longbench_dir=Path(args.longbench_dir),
        out_csv=out_dir / "be_workload.csv",
        max_model_len=args.max_model_len,
        max_rows=args.max_be,
    )


if __name__ == "__main__":
    main()
