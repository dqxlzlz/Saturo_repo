import pytest

from saturo.resource_allocation import AllocationProfiles, ResourceAllocationSolver
from saturo.resource_allocation import Workload


def test_objective_adds_complete_be_throughput_from_all_pools():
    profiles = AllocationProfiles.from_dict({
        "configs": {
            "tp1": {
                "gpus": 1,
                "tp": 1,
                "pp": 1,
            },
        },
        "pools": {
            "P": {
                "tp1": {
                    "mu": 1.0,
                    "saturation": [{
                        "rt_rps": 0.1,
                        "be_throughput": 1000.0,
                    }],
                },
            },
            "D": {
                "tp1": {
                    "mu": 1.0,
                    "saturation": [{
                        "rt_rps": 0.1,
                        "be_throughput": 10.0,
                    }],
                },
            },
            "T": {
                "tp1": {
                    "saturation": [{
                        "rt_rps": 0.0,
                        "be_throughput": 100.0,
                    }],
                },
            },
        },
    })

    best = ResourceAllocationSolver(profiles).solve(
        total_gpus=8,
        workload=Workload(p_rps=0.1, d_rps=0.1),
    )[0]

    assert best.objective == pytest.approx(
        best.p.total_be_throughput +
        best.d.total_be_throughput +
        best.t.total_be_throughput)
    # P has the highest complete-request BE throughput per GPU, so after
    # reserving the required D instance the additive objective assigns every
    # remaining GPU to P.
    assert best.p.instances == 7
    assert best.d.instances == 1
    assert best.t.instances == 0
