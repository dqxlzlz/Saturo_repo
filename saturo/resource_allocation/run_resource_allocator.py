"""CLI for two-stage Saturo resource allocation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from saturo.resource_allocation.profiles import load_profiles  # noqa: E402
from saturo.resource_allocation.profiles import AllocationProfiles  # noqa: E402
from saturo.resource_allocation.solver import (  # noqa: E402
    Allocation,
    ResourceAllocationSolver,
    Workload,
)


def _float_csv(value: str) -> List[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",")
                  if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--target-rps must be a comma-separated list of numbers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "--target-rps must contain positive values")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage Saturo resource allocation solver")
    parser.add_argument(
        "--profiles",
        required=True,
        help="JSON profile table containing mu_y(x) and phi_y(x,w) samples.",
    )
    parser.add_argument(
        "--total-gpus",
        type=int,
        required=True,
        help="Cluster GPU budget G.",
    )

    parser.add_argument(
        "--target-rps",
        type=_float_csv,
        required=True,
        help=(
            "End-to-end RT request RPS. Accepts comma-separated values for "
            "a sweep; W_P = W_D = target RPS."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--config-filter",
        default="",
        help=(
            "Comma-separated parallel config names to include, for example "
            "tp1_pp1,tp2_pp1. Empty means include all configs in the profile."
        ),
    )
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def filter_profiles(
    profiles: AllocationProfiles,
    config_filter: str,
) -> AllocationProfiles:
    allowed = {item.strip() for item in config_filter.split(",") if item.strip()}
    if not allowed:
        return profiles

    missing = allowed - set(profiles.configs.keys())
    if missing:
        raise ValueError(
            "config filter references missing profile configs: "
            + ", ".join(sorted(missing)))

    payload = profiles.to_dict()
    payload["configs"] = {
        name: config
        for name, config in payload["configs"].items()
        if name in allowed
    }
    payload["pools"] = {
        pool: {
            name: profile
            for name, profile in entries.items()
            if name in allowed
        }
        for pool, entries in payload["pools"].items()
    }
    return AllocationProfiles.from_dict(payload)


def allocation_to_row(index: int, allocation: Allocation) -> str:
    return (
        f"{index:>3} "
        f"{allocation.objective:>12.3f} "
        f"{allocation.total_gpus:>4} "
        f"{allocation.p.config_name:>12} {allocation.p.instances:>3} "
        f"{allocation.p.per_instance_rt_workload:>8.3f} "
        f"{allocation.d.config_name:>12} {allocation.d.instances:>3} "
        f"{allocation.d.per_instance_rt_workload:>8.3f} "
        f"{allocation.t.config_name:>12} {allocation.t.instances:>3}"
    )


def print_allocations(allocations: List[Allocation]) -> None:
    if not allocations:
        print("No feasible allocation found.")
        return
    header = (
        f"{'#':>3} {'BE objective':>12} {'GPU':>4} "
        f"{'P config':>12} {'nP':>3} {'wP':>8} "
        f"{'D config':>12} {'nD':>3} {'wD':>8} "
        f"{'T config':>12} {'nT':>3}"
    )
    print(header)
    print("-" * len(header))
    for i, allocation in enumerate(allocations, start=1):
        print(allocation_to_row(i, allocation))


def workload_for_rps(target_rps: float) -> Workload:
    return Workload(p_rps=target_rps, d_rps=target_rps)


def print_sweep_summary(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    print("\nSweep summary")
    header = (
        f"{'RT RPS':>8} {'Objective':>12} {'GPU':>4} "
        f"{'P':>12} {'nP':>3} {'D':>12} {'nD':>3} {'T':>12} {'nT':>3}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        allocation = row["best"]
        if allocation is None:
            print(f"{row['target_rps']:>8.3f} {'infeasible':>12}")
            continue
        print(
            f"{row['target_rps']:>8.3f} "
            f"{allocation.objective:>12.3f} "
            f"{allocation.total_gpus:>4} "
            f"{allocation.p.config_name:>12} {allocation.p.instances:>3} "
            f"{allocation.d.config_name:>12} {allocation.d.instances:>3} "
            f"{allocation.t.config_name:>12} {allocation.t.instances:>3}"
        )


def main() -> None:
    args = parse_args()
    profiles = filter_profiles(load_profiles(args.profiles), args.config_filter)
    solver = ResourceAllocationSolver(profiles)
    rps_values = args.target_rps
    sweep_rows: List[Dict[str, Any]] = []

    for target_rps in rps_values:
        workload = workload_for_rps(target_rps)
        allocations = solver.solve(
            total_gpus=args.total_gpus,
            workload=workload,
            top_k=args.top_k,
        )
        if len(rps_values) > 1:
            print(
                f"\nRT RPS={target_rps:.3f} "
                f"(W_P={workload.p_rps:.3f}, W_D={workload.d_rps:.3f})")
        print_allocations(allocations)
        sweep_rows.append({
            "target_rps": target_rps,
            "workload": workload,
            "best": allocations[0] if allocations else None,
            "allocations": allocations,
        })

    if len(sweep_rows) > 1:
        print_sweep_summary(sweep_rows)

    if args.output:
        payload: Dict[str, Any] = {
            "input": {
                "profiles": args.profiles,
                "total_gpus": args.total_gpus,
                "target_rps": rps_values,
                "top_k": args.top_k,
                "config_filter": args.config_filter,
            },
            "results": [
                {
                    "target_rps": row["target_rps"],
                    "workload": {
                        "p_rps": row["workload"].p_rps,
                        "d_rps": row["workload"].d_rps,
                    },
                    "best": (
                        None if row["best"] is None
                        else row["best"].to_dict()
                    ),
                    "allocations": [
                        item.to_dict() for item in row["allocations"]
                    ],
                }
                for row in sweep_rows
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Output] wrote {args.output}")


if __name__ == "__main__":
    main()
