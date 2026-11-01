import pytest

from saturo.resource_allocation.profiles import AllocationProfiles
from saturo.resource_allocation.solver import (
    ResourceAllocationSolver,
    Workload,
)


EXPECTED_CONFIGS = {
    "tp1_pp1",
    "tp2_pp1",
    "tp4_pp1",
    "tp1_pp2",
    "tp2_pp2",
    "tp4_pp2",
}


def make_profiles() -> AllocationProfiles:
    return AllocationProfiles.from_dict({
        "configs": {
            "small": {"gpus": 2, "tp": 2, "pp": 1},
            "large": {"gpus": 4, "tp": 4, "pp": 1},
        },
        "pools": {
            "P": {
                "small": {
                    "mu": 2.0,
                    "saturation": [
                        {"rt_rps": 0.0, "be_throughput": 100.0},
                        {"rt_rps": 1.0, "be_throughput": 80.0},
                        {"rt_rps": 2.0, "be_throughput": 50.0},
                    ],
                },
                "large": {
                    "mu": 4.0,
                    "saturation": [
                        {"rt_rps": 0.0, "be_throughput": 160.0},
                        {"rt_rps": 2.0, "be_throughput": 110.0},
                        {"rt_rps": 4.0, "be_throughput": 30.0},
                    ],
                },
            },
            "D": {
                "small": {
                    "mu": 2.0,
                    "saturation": [
                        {"rt_rps": 0.0, "be_throughput": 90.0},
                        {"rt_rps": 1.0, "be_throughput": 60.0},
                        {"rt_rps": 2.0, "be_throughput": 10.0},
                    ],
                },
                "large": {
                    "mu": 3.0,
                    "saturation": [
                        {"rt_rps": 0.0, "be_throughput": 170.0},
                        {"rt_rps": 1.5, "be_throughput": 120.0},
                        {"rt_rps": 3.0, "be_throughput": 40.0},
                    ],
                },
            },
            "T": {
                "small": {
                    "saturation": [
                        {"rt_rps": 0.0, "be_throughput": 130.0},
                    ],
                },
                "large": {
                    "saturation": [
                        {"rt_rps": 0.0, "be_throughput": 240.0},
                    ],
                },
            },
        },
    })


def test_brute_force_solver_returns_feasible_best_allocation():
    solver = ResourceAllocationSolver(make_profiles())
    results = solver.solve(
        total_gpus=8,
        workload=Workload(p_rps=2.0, d_rps=2.0),
        top_k=5,
    )

    assert results
    best = results[0]
    assert best.total_gpus <= 8
    assert best.p.instances >= 1
    assert best.d.instances >= 1
    assert best.t.instances >= 0
    assert best.p.total_rt_capacity >= 2.0
    assert best.d.total_rt_capacity >= 2.0
    assert best.objective == pytest.approx(
        best.p.total_be_throughput
        + best.d.total_be_throughput
        + best.t.total_be_throughput
    )
    assert results == sorted(
        results,
        key=lambda item: (item.objective, -item.total_gpus),
        reverse=True,
    )


def test_phi_uses_linear_interpolation_without_high_load_extrapolation():
    profiles = make_profiles()
    p_small = profiles.profile("P", "small")

    assert p_small.phi(0.5) == pytest.approx(90.0)
    assert p_small.phi(3.0) is None


def test_solver_reports_no_solution_when_capacity_is_insufficient():
    solver = ResourceAllocationSolver(make_profiles())
    results = solver.solve(
        total_gpus=4,
        workload=Workload(p_rps=4.0, d_rps=4.0),
        top_k=1,
    )

    assert results == []


def test_example_profile_covers_candidate_grid():
    profiles = AllocationProfiles.from_dict(__import__("json").load(open(
        "experiments/resource_allocation/example_profiles.json",
        encoding="utf-8",
    )))

    assert set(profiles.configs) == EXPECTED_CONFIGS
    for pool in ("P", "D", "T"):
        assert set(profiles.pools[pool]) == EXPECTED_CONFIGS
