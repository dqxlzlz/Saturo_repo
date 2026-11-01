"""Compare the budget solver against the previous exhaustive search."""

from itertools import product
import random

import pytest

from saturo.resource_allocation import (
    Allocation, AllocationProfiles, ResourceAllocationSolver, Workload,
)


def brute_force(profiles, total_gpus, workload, top_k):
    """Frozen legacy enumeration; use no budget-table/pruning helpers."""
    solver = ResourceAllocationSolver(profiles)
    candidates = []
    configs = [list(profiles.config_names_for_pool(pool))
               for pool in ("P", "D", "T")]
    for p_config, d_config, t_config in product(*configs):
        p_gpu = profiles.config(p_config).gpus
        d_gpu = profiles.config(d_config).gpus
        t_gpu = profiles.config(t_config).gpus
        for n_p in range(1, total_gpus // p_gpu + 1):
            if n_p * p_gpu >= total_gpus:
                continue
            p = solver._build_rt_pool_allocation(
                pool="P", config_name=p_config, instances=n_p,
                rt_workload=workload.p_rps)
            if p is None:
                continue
            for n_d in range(1, total_gpus // d_gpu + 1):
                used = n_p * p_gpu + n_d * d_gpu
                if used > total_gpus:
                    continue
                d = solver._build_rt_pool_allocation(
                    pool="D", config_name=d_config, instances=n_d,
                    rt_workload=workload.d_rps)
                if d is None:
                    continue
                t = solver._build_t_pool_allocation(
                    config_name=t_config,
                    instances=(total_gpus - used) // t_gpu)
                if t is not None:
                    candidates.append(Allocation(p, d, t))
    candidates.sort(key=lambda item: (item.objective, -item.total_gpus),
                    reverse=True)
    return candidates[:top_k]


def random_profiles(seed):
    rng = random.Random(seed)
    names = ["config_%d" % i for i in range(rng.randint(1, 5))]
    rng.shuffle(names)
    configs = {name: {"gpus": rng.randint(1, 5)} for name in names}
    pools = {}
    for pool in ("P", "D", "T"):
        order = names[:]
        rng.shuffle(order)
        entries = {}
        for name in order:
            if rng.random() < 0.12:
                continue
            mu = rng.choice([None, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
            rates = [0.0] if pool == "T" else [0.0, 0.5, 2.0, 4.0]
            entries[name] = {
                "mu": mu,
                "saturation": [
                    {"rt_rps": rate,
                     "be_throughput": rng.choice([0, 0, 1, 4, 16, 64])}
                    for rate in rates
                ],
            }
        pools[pool] = entries
    return AllocationProfiles.from_dict({"configs": configs, "pools": pools})


def constant_profiles(gpus=(1,), throughputs=(1, 1, 100), mu=1):
    names = ["config_%d" % i for i in range(len(gpus))]
    return AllocationProfiles.from_dict({
        "configs": {name: {"gpus": size}
                    for name, size in zip(names, gpus)},
        "pools": {
            pool: {
                name: {
                    "mu": mu,
                    "saturation": [
                        {"rt_rps": rate, "be_throughput": throughput}
                        for rate in ([0] if pool == "T" else [0, 8])
                    ],
                }
                for name in reversed(names)
            }
            for pool, throughput in zip(("P", "D", "T"), throughputs)
        },
    })


def assert_matches(profiles, total_gpus, workload, top_k):
    expected = brute_force(profiles, total_gpus, workload, top_k)
    actual = ResourceAllocationSolver(profiles).solve(
        total_gpus=total_gpus, workload=workload, top_k=top_k)
    # Verify the entire ranked plan, including configuration order on ties.
    assert [a.to_dict() for a in actual] == [a.to_dict() for a in expected]
    for allocation in actual:
        assert allocation.total_gpus <= total_gpus
        assert allocation.p.instances >= 1
        assert allocation.d.instances >= 1
        assert allocation.t.instances >= 0
        assert allocation.p.total_rt_capacity + 1e-9 >= workload.p_rps
        assert allocation.d.total_rt_capacity + 1e-9 >= workload.d_rps
    assert len({(a.p.config_name, a.p.instances, a.d.config_name,
                 a.d.instances, a.t.config_name, a.t.instances)
                for a in actual}) == len(actual)


@pytest.mark.parametrize("seed", range(120))
def test_random_profiles_match_brute_force(seed):
    rng = random.Random(seed + 10000)
    profiles = random_profiles(seed)
    total_gpus = rng.randint(1, 24)
    workload = Workload(rng.choice([0, 0.25, 1, 3, 8, 32]),
                        rng.choice([0, 0.5, 2, 4, 16]))
    for top_k in (1, 3, 10):
        assert_matches(profiles, total_gpus, workload, top_k)


@pytest.mark.parametrize("top_k", [1, 3, 10, 1000])
@pytest.mark.parametrize("gpus", [(1,), (2, 3), (2, 4, 5)])
@pytest.mark.parametrize("throughputs", [(0, 0, 0), (1, 1, 0), (0, 0, 100)])
def test_ties_unused_gpus_and_zero_throughput(gpus, throughputs, top_k):
    assert_matches(constant_profiles(gpus, throughputs), 11,
                   Workload(1, 1), top_k)


def test_infeasible_rt_pools_cannot_be_replaced_by_t_only_plan():
    profiles = constant_profiles()
    assert ResourceAllocationSolver(profiles).solve(
        total_gpus=4, workload=Workload(3, 3)) == []


def test_zero_be_throughput_is_feasible_and_t_can_be_empty():
    profiles = constant_profiles(throughputs=(0, 0, 100))
    allocation = ResourceAllocationSolver(profiles).solve(
        total_gpus=2, workload=Workload(1, 1))[0]
    assert (allocation.p.instances, allocation.d.instances,
            allocation.t.instances, allocation.objective) == (1, 1, 0, 0)


@pytest.mark.parametrize("p_rps,d_rps", [(0, 0), (0, 1), (1, 0)])
def test_zero_rt_load_keeps_legacy_minimum_pool_sizes(p_rps, d_rps):
    assert_matches(constant_profiles(), 7, Workload(p_rps, d_rps), 100)


@pytest.mark.parametrize("total_gpus,top_k", [(0, 1), (-1, 1), (4, 0)])
def test_invalid_budgets_and_top_k(total_gpus, top_k):
    with pytest.raises(ValueError):
        ResourceAllocationSolver(constant_profiles()).solve(
            total_gpus=total_gpus, workload=Workload(1, 1), top_k=top_k)


def test_profile_evaluations_scale_with_configs_times_gpu_budget(monkeypatch):
    from saturo.resource_allocation.profiles import PoolProfile

    evaluations = 0
    original = PoolProfile.phi

    def counted(self, rt_rps):
        nonlocal evaluations
        evaluations += 1
        return original(self, rt_rps)

    monkeypatch.setattr(PoolProfile, "phi", counted)
    profiles = constant_profiles((1, 2, 3, 4), mu=8)
    ResourceAllocationSolver(profiles).solve(
        total_gpus=128, workload=Workload(1, 1))
    assert evaluations <= 3 * len(profiles.configs) * (128 + 1)
