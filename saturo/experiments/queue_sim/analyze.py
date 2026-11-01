#!/usr/bin/env python3
"""统计 prefill-ttft 目录下各 CSV 文件的 TTFT 指标。"""

import csv
import glob
import os
import statistics

import numpy as np


def analyze_csv(path: str) -> dict | None:
    """读取单个 CSV，返回 TTFT 统计指标（毫秒）。"""
    values = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            raw = row.get("ttft_ms") or row.get("ttft_s")
            if raw is None or raw == "":
                continue
            val = float(raw)
            if "ttft_s" in row and not row.get("ttft_ms"):
                val *= 1000.0  # 秒 -> 毫秒
            values.append(val)

    if not values:
        return None

    arr = np.array(values)
    return {
        "count": len(values),
        "average": np.mean(arr),
        "median": np.median(arr),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
        "max": np.max(arr),
    }


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ttft_dir = os.path.normpath(
        os.path.join(script_dir, "..", "results", "ttft"))
    csv_files = sorted(glob.glob(os.path.join(ttft_dir, "*.csv")))

    if not csv_files:
        print("未找到任何 CSV 文件。")
        return

    # 表头
    header = f"{'File':<58} {'N':>5} {'Avg':>8} {'Median':>8} {'P95':>8} {'P99':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))

    for path in csv_files:
        name = os.path.basename(path)
        stats = analyze_csv(path)
        if stats is None:
            print(f"{name:<58} {'N/A':>5}")
            continue
        print(
            f"{name:<58} {stats['count']:>5} "
            f"{stats['average']:>8.1f} {stats['median']:>8.1f} "
            f"{stats['p95']:>8.1f} {stats['p99']:>8.1f} {stats['max']:>8.1f}"
        )


if __name__ == "__main__":
    main()
