"""Demand-based baseline from the paper: minimum RT P/D GPUs, rest to T."""

import math
from heapq import nsmallest

from saturo.resource_allocation.solver import Allocation, ResourceAllocationSolver


class DemandBasedSolver(ResourceAllocationSolver):
    def solve(self, *, total_gpus, workload, top_k=1):
        if total_gpus <= 0 or top_k <= 0:
            raise ValueError("total_gpus and top_k must be positive")
        candidates = []
        for p_name in self.profiles.config_names_for_pool("P"):
            p_profile = self.profiles.profile("P", p_name)
            if not p_profile.mu or p_profile.mu <= 0:
                continue
            n_p = max(1, math.ceil(workload.p_rps / p_profile.mu))
            p = self._build_rt_pool_allocation(
                pool="P", config_name=p_name, instances=n_p,
                rt_workload=workload.p_rps)
            if p is None:
                continue
            for d_name in self.profiles.config_names_for_pool("D"):
                d_profile = self.profiles.profile("D", d_name)
                if not d_profile.mu or d_profile.mu <= 0:
                    continue
                n_d = max(1, math.ceil(workload.d_rps / d_profile.mu))
                d = self._build_rt_pool_allocation(
                    pool="D", config_name=d_name, instances=n_d,
                    rt_workload=workload.d_rps)
                if d is None:
                    continue
                remaining = total_gpus - p.total_gpus - d.total_gpus
                if remaining < 0:
                    continue
                for t_name in self.profiles.config_names_for_pool("T"):
                    gpus = self.profiles.config(t_name).gpus
                    t = self._build_t_pool_allocation(
                        config_name=t_name, instances=remaining // gpus)
                    if t is not None:
                        candidates.append(Allocation(p, d, t))
        # Extra P/D instances are never bought for BE gain. Throughput breaks
        # ties only among allocations using the same minimum RT GPU budget.
        return nsmallest(top_k, candidates, key=lambda a: (
            a.p.total_gpus + a.d.total_gpus, -a.objective,
            a.total_gpus, a.p.config_name, a.d.config_name, a.t.config_name))
