"""Two-stage GPU-budget resource allocation solver for Saturo."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush, nsmallest
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from saturo.resource_allocation.profiles import (
    POOL_D,
    POOL_P,
    POOL_T,
    AllocationProfiles,
)


@dataclass(frozen=True)
class Workload:
    """Incoming RT workload assigned to the P and D pools."""

    p_rps: float
    d_rps: float

    def __post_init__(self) -> None:
        if self.p_rps < 0 or self.d_rps < 0:
            raise ValueError("RT workloads must be non-negative")


@dataclass(frozen=True)
class PoolAllocation:
    pool: str
    config_name: str
    instances: int
    gpus_per_instance: int
    rt_workload: float
    per_instance_rt_workload: float
    rt_capacity_per_instance: Optional[float]
    be_throughput_per_instance: float

    @property
    def total_gpus(self) -> int:
        return self.instances * self.gpus_per_instance

    @property
    def total_be_throughput(self) -> float:
        return self.instances * self.be_throughput_per_instance

    @property
    def total_rt_capacity(self) -> Optional[float]:
        if self.rt_capacity_per_instance is None:
            return None
        return self.instances * self.rt_capacity_per_instance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool": self.pool,
            "config": self.config_name,
            "instances": self.instances,
            "gpus_per_instance": self.gpus_per_instance,
            "total_gpus": self.total_gpus,
            "rt_workload": self.rt_workload,
            "per_instance_rt_workload": self.per_instance_rt_workload,
            "rt_capacity_per_instance": self.rt_capacity_per_instance,
            "total_rt_capacity": self.total_rt_capacity,
            "be_throughput_per_instance": self.be_throughput_per_instance,
            "total_be_throughput": self.total_be_throughput,
        }


@dataclass(frozen=True)
class Allocation:
    p: PoolAllocation
    d: PoolAllocation
    t: PoolAllocation

    @property
    def objective(self) -> float:
        # Each pool profile reports complete, end-to-end BE request
        # throughput. P- and D-pool BE work is not modeled as two stages of
        # the same request pipeline, so their capacities are additive.
        return (
            self.p.total_be_throughput
            + self.d.total_be_throughput
            + self.t.total_be_throughput
        )

    @property
    def total_gpus(self) -> int:
        return self.p.total_gpus + self.d.total_gpus + self.t.total_gpus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "total_gpus": self.total_gpus,
            "p": self.p.to_dict(),
            "d": self.d.to_dict(),
            "t": self.t.to_dict(),
        }


class ResourceAllocationSolver:
    """Optimize each pool first, then combine their GPU budgets.

    For top_k=1, preprocessing takes O(|C| G) pool evaluations and the
    budget search takes O(G^2), excluding the cost of profile interpolation.
    Keeping up to k choices per budget also supports the existing top-k API.
    """

    def __init__(self, profiles: AllocationProfiles):
        self.profiles = profiles

    def solve(
        self,
        *,
        total_gpus: int,
        workload: Workload,
        top_k: int = 1,
    ) -> List[Allocation]:
        if total_gpus <= 0:
            raise ValueError("total_gpus must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        configs = {
            pool: list(self.profiles.config_names_for_pool(pool))
            for pool in (POOL_P, POOL_D, POOL_T)
        }
        if any(not names for names in configs.values()):
            return []

        config_order = {
            pool: {name: i for i, name in enumerate(names)}
            for pool, names in configs.items()
        }

        def allocation_key(item: Allocation) -> Tuple:
            # Reproduce the old stable ordering after objective/GPU ties:
            # product(P configs, D configs, T configs), then n_P and n_D.
            return (
                -item.objective, item.total_gpus,
                config_order[POOL_P][item.p.config_name],
                config_order[POOL_D][item.d.config_name],
                config_order[POOL_T][item.t.config_name],
                item.p.instances, item.d.instances,
            )

        # Stage 1: retain E_y(q) and its (config, count) choices per pool.
        # Empty lists represent infeasibility, not zero BE throughput.
        tables = {}
        for pool, rt_workload in (
            (POOL_P, workload.p_rps), (POOL_D, workload.d_rps), (POOL_T, 0.0)
        ):
            tables[pool] = self._build_budget_table(
                pool=pool, config_names=configs[pool], total_gpus=total_gpus,
                rt_workload=rt_workload, top_k=top_k,
                config_order=config_order[pool],
            )

        def candidates() -> Iterator[Allocation]:
            # Stage 2: enumerate P/D GPU usage and give T the remaining budget.
            for q_p in range(1, total_gpus):
                p_choices = tables[POOL_P][q_p]
                if not p_choices:
                    continue
                for q_d in range(1, total_gpus - q_p + 1):
                    d_choices = tables[POOL_D][q_d]
                    t_choices = tables[POOL_T][total_gpus - q_p - q_d]
                    if not d_choices or not t_choices:
                        continue
                    yield from self._best_combinations(
                        p_choices, d_choices, t_choices, top_k, allocation_key)

        return nsmallest(top_k, candidates(), key=allocation_key)

    def _build_budget_table(
        self, *, pool: str, config_names: List[str], total_gpus: int,
        rt_workload: float, top_k: int, config_order: Dict[str, int],
    ) -> List[List[PoolAllocation]]:
        """Build exact-usage P/D tables and an at-most-budget T table.

        Enumerating exact P/D usages in stage 2 makes their prefix-max pass
        unnecessary. It also preserves legacy top-k results and tie-breaking.
        T fills its budget with floor(q/g(x)) instances for each config, as
        before, including when its BE throughput is zero. Thus unused GPUs,
        indivisible configurations and empty T pools retain their semantics.
        """
        table: List[List[PoolAllocation]] = [
            [] for _ in range(total_gpus + 1)
        ]

        def offer(q: int, choice: Optional[PoolAllocation]) -> None:
            if choice is None:
                return
            choices = table[q]
            choices.append(choice)
            choices.sort(key=lambda item: (
                -item.total_be_throughput, item.total_gpus,
                config_order[item.config_name], item.instances,
            ))
            del choices[top_k:]

        for config_name in config_names:
            gpus = self.profiles.config(config_name).gpus
            if pool == POOL_T:
                for q in range(total_gpus + 1):
                    offer(q, self._build_t_pool_allocation(
                        config_name=config_name, instances=q // gpus))
            else:
                for n in range(1, total_gpus // gpus + 1):
                    offer(n * gpus, self._build_rt_pool_allocation(
                        pool=pool, config_name=config_name, instances=n,
                        rt_workload=rt_workload))
        return table

    @staticmethod
    def _best_combinations(
        p_choices: List[PoolAllocation], d_choices: List[PoolAllocation],
        t_choices: List[PoolAllocation], top_k: int,
        key: Callable[[Allocation], Tuple],
    ) -> Iterator[Allocation]:
        """Yield k best combinations without enumerating a k^3 product."""
        if top_k == 1:
            yield Allocation(p_choices[0], d_choices[0], t_choices[0])
            return

        pools = (p_choices, d_choices, t_choices)
        first = Allocation(p_choices[0], d_choices[0], t_choices[0])
        heap = [(key(first), (0, 0, 0), first)]
        visited = {(0, 0, 0)}
        for _ in range(top_k):
            if not heap:
                break
            _, indices, allocation = heappop(heap)
            yield allocation
            for axis in range(3):
                neighbor = list(indices)
                neighbor[axis] += 1
                next_indices = tuple(neighbor)
                if (neighbor[axis] >= len(pools[axis])
                        or next_indices in visited):
                    continue
                visited.add(next_indices)
                choice = Allocation(*(
                    pool[i] for pool, i in zip(pools, next_indices)
                ))
                heappush(heap, (key(choice), next_indices, choice))

    def _build_rt_pool_allocation(
        self,
        *,
        pool: str,
        config_name: str,
        instances: int,
        rt_workload: float,
    ) -> Optional[PoolAllocation]:
        profile = self.profiles.profile(pool, config_name)
        config = self.profiles.config(config_name)
        if instances <= 0:
            return None
        if profile.mu is None:
            return None
        if instances * profile.mu + 1e-9 < rt_workload:
            return None

        per_instance_rt = rt_workload / instances
        phi = profile.phi(per_instance_rt)
        if phi is None:
            return None
        return PoolAllocation(
            pool=pool,
            config_name=config_name,
            instances=instances,
            gpus_per_instance=config.gpus,
            rt_workload=rt_workload,
            per_instance_rt_workload=per_instance_rt,
            rt_capacity_per_instance=profile.mu,
            be_throughput_per_instance=phi,
        )

    def _build_t_pool_allocation(
        self,
        *,
        config_name: str,
        instances: int,
    ) -> Optional[PoolAllocation]:
        profile = self.profiles.profile(POOL_T, config_name)
        config = self.profiles.config(config_name)
        phi = profile.phi(0.0)
        if phi is None:
            return None
        return PoolAllocation(
            pool=POOL_T,
            config_name=config_name,
            instances=instances,
            gpus_per_instance=config.gpus,
            rt_workload=0.0,
            per_instance_rt_workload=0.0,
            rt_capacity_per_instance=None,
            be_throughput_per_instance=phi,
        )
