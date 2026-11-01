"""Small resource-aware worker planner for Saturo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from saturo.history import (configured_history_source, is_redis_history,
                            read_redis_history)

from saturo.resource_allocation import (
    Allocation,
    AllocationProfiles,
    ResourceAllocationSolver,
    Workload,
    load_profiles,
)


POOL_TO_ROLE = {
    "P": "prefill",
    "D": "decode",
    "T": "throughput",
}


def repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "vllm").exists() and (parent / "saturo").exists():
            return parent
    return path.parents[1]


def parse_config_names(payload: str) -> set[str]:
    return {item.strip() for item in payload.split(",") if item.strip()}


def _merge_profile_dicts(
    base: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Merge two allocation-profile dicts, with patch entries taking priority."""
    merged_configs: dict[str, Any] = dict(base.get("configs", {}))
    merged_configs.update(patch.get("configs", {}))

    all_pools = (
        set(base.get("pools", {}).keys()) |
        set(patch.get("pools", {}).keys())
    )
    merged_pools: dict[str, Any] = {}
    for pool in all_pools:
        pool_entries: dict[str, Any] = dict(
            base.get("pools", {}).get(pool, {}))
        pool_entries.update(patch.get("pools", {}).get(pool, {}))
        merged_pools[pool] = pool_entries

    return {"configs": merged_configs, "pools": merged_pools}


class SaturoResourcePlanner:
    """Resolve P/D/T resources from an allocation profile and total GPU count."""

    def __init__(
        self,
        args: argparse.Namespace,
        placement_factory: Callable[..., Any],
    ):
        self.args = args
        self.placement_factory = placement_factory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_worker_plan(self) -> list[Any]:
        total_gpus = self._total_gpus()
        profile_path = self._profile_path()
        profiles = self._load_or_collect_profile(profile_path)
        profiles = self._filter_profiles(profiles)
        allocation = self._solve(profiles, total_gpus=total_gpus)
        placements = self._allocation_to_placements(allocation, profiles)
        self._write_plan_summary(profile_path, allocation, placements, total_gpus)
        return placements

    def solve_staged_profile(self, profile_path: str | Path, *,
                             total_gpus: int, target_rps: float
                             ) -> dict[str, Any]:
        """Solve an already rebuilt table without changing this planner.

        This path deliberately cannot collect GPU timings, resolve Ray GPU
        availability, overwrite the active table, or change live arguments.
        The returned JSON can be staged and transactionally applied later.
        """
        if total_gpus <= 0 or target_rps <= 0:
            raise ValueError("positive GPU budget and observed arrival rate required")
        gpu_ids = [v.strip() for v in str(getattr(self.args, "gpu_ids", "")).split(",") if v.strip()]
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("duplicate GPU ids in staged placement budget")
        args = argparse.Namespace(**vars(self.args))
        args.target_rps = float(target_rps)
        staged = SaturoResourcePlanner(args, self.placement_factory)
        profiles = staged._filter_profiles(load_profiles(profile_path))
        allocation = staged._solve(profiles, total_gpus=int(total_gpus))
        placements = staged._allocation_to_placements(allocation, profiles)
        rows = []
        index = 0
        for pool in (allocation.p, allocation.d, allocation.t):
            for _ in range(pool.instances):
                item = placements[index]
                index += 1
                rows.append({
                    "role": item.role, "config_name": pool.config_name,
                    "tensor_parallel_size": int(item.tensor_parallel_size),
                    "pipeline_parallel_size": int(item.pipeline_parallel_size),
                    "expert_parallel_size": int(getattr(item, "expert_parallel_size", 0)),
                    "gpu_id": str(getattr(item, "gpu_id", "")),
                })
        return {"profile_path": str(Path(profile_path).resolve()),
                "allocation_policy": getattr(args, "resource_allocation_policy", "saturation_aware"),
                "target_rps": float(target_rps), "total_gpus": int(total_gpus),
                "allocation": allocation.to_dict(), "placements": rows}

    def _total_gpus(self) -> int:
        """Return total GPU count: explicit arg or auto-detected from Ray."""
        explicit = int(getattr(self.args, "total_gpus", 0) or 0)
        if explicit > 0:
            return explicit
        import ray
        if not ray.is_initialized():
            address = getattr(self.args, "ray_address", "") or ""
            if address in ("", "local"):
                ray.init(address="local", include_dashboard=False)
            else:
                ray.init(address=address)
        total = int(ray.available_resources().get("GPU", 0))
        if total <= 0:
            raise RuntimeError(
                "No GPUs reported by Ray. Pass --total-gpus explicitly or "
                "ensure the Ray cluster has GPU resources registered.")
        return total

    def _profile_path(self) -> Path:
        explicit = self.args.resource_profile.strip()
        if explicit:
            return Path(explicit).expanduser()
        return (Path("saturo/results/resource_allocation/generated") /
                self._default_profile_key() / "allocation_profiles.json")

    def _load_or_collect_profile(self, profile_path: Path) -> AllocationProfiles:
        requested = parse_config_names(self.args.resource_config_filter)
        existing = load_profiles(profile_path) if profile_path.exists() else None
        source = configured_history_source(
            getattr(self.args, "rt_workload_csv", "") or "")
        if is_redis_history(source):
            snapshot = read_redis_history(
                source, window_s=float(os.environ.get(
                    "SATURO_HISTORY_WINDOW_S", "1800")),
                model=str(self.args.model))
            self.args._history_snapshot = snapshot
            self.args._history_metadata = snapshot.metadata()
            cached_history = (json.loads(profile_path.read_text()).get(
                "history", {}) if profile_path.exists() else {})
            if not snapshot.records and existing is not None:
                print("[planner] Redis history empty; keeping existing "
                      "allocation profile for cold start", flush=True)
            elif cached_history != snapshot.metadata():
                # BE throughput depends on the history, unlike the offline
                # GPU timing model. Do not reuse a table for an older trace.
                existing = None
        existing_names = set(existing.configs.keys()) if existing else set()
        missing = requested - existing_names

        if not missing and existing is not None:
            return existing

        # 只采集缺失的配置（文件不存在时 missing==requested，相当于全量采集）
        import tempfile
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            suffix=".json", dir=profile_path.parent)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            self._collect_real_allocation_profile(
                tmp_path, configs_override=missing or None)
            new_profiles = load_profiles(tmp_path)
            base = existing.to_dict() if existing else {}
            merged = _merge_profile_dicts(base, new_profiles.to_dict())
            if is_redis_history(source):
                merged["history"] = self.args._history_metadata
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(json.dumps(merged, indent=2))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return load_profiles(profile_path)

    def _collect_real_allocation_profile(
        self,
        profile_path: Path,
        configs_override: set[str] | None = None,
    ) -> None:
        """Measure GPU step timings, then model historical P/D/T capacity.

        A local engine measures module timing for each parallel configuration.
        Simulators use the fitted timing model and historical lengths to build
        mu/phi tables; these are modeled capacities, not measured e2e RPS.
        """

        history_source = configured_history_source(
            getattr(self.args, "rt_workload_csv", "") or "")
        if not history_source.strip():
            raise ValueError(
                "missing --rt-workload-csv. To collect a missing resource "
                "profile, Saturo must replay a real history workload on each "
                "requested config first.")
        requested_configs = (
            configs_override
            if configs_override is not None
            else parse_config_names(self.args.resource_config_filter)
        )
        if not requested_configs:
            raise ValueError(
                "missing --resource-config-filter. It is used as the config "
                "set to profile when allocation_profiles.json does not exist.")
        from saturo.profiling.real_profile_collector import (
            collect_real_allocation_profile,
        )

        collect_real_allocation_profile(
            self.args,
            profile_path,
            sorted(requested_configs),
        )

    def _default_profile_key(self) -> str:
        digest_input = "|".join([
            str(self.args.model),
            str(self.args.num_req),
            str(self.args.target_rps),
            str(self.args.rt_ttft_slo_ms),
            str(self.args.rt_tbt_slo_ms),
            str(getattr(self.args, "resource_decode_strategies", "progressive")),
            str(getattr(self.args, "gpu_memory_utilization", "")),
            str(getattr(self.args, "max_model_len", "")),
            str(getattr(self.args, "load_format", "")),
            str(getattr(self.args, "dtype", "")),
            self._qrf_fingerprint(),
        ])
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
        model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(self.args.model).name)
        return f"{model_name or 'model'}_{digest}"

    def _qrf_fingerprint(self) -> str:
        from saturo.predictors.qrf import resolve_qrf_path
        path = resolve_qrf_path(getattr(self.args, "model_len_predictor", None))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else ""

    def _filter_profiles(self, profiles: AllocationProfiles) -> AllocationProfiles:
        allowed = parse_config_names(self.args.resource_config_filter)
        payload = profiles.to_dict()
        if not allowed:
            allowed = set(payload["configs"])

        payload["configs"] = {
            name: config
            for name, config in payload["configs"].items()
            if name in allowed
        }
        payload["pools"] = {
            pool: {
                name: profile
                for name, profile in entries.items()
                if name in payload["configs"]
            }
            for pool, entries in payload["pools"].items()
        }
        for pool, flag in (("P", "enable_prefill_saturation"),
                           ("D", "enable_decode_saturation")):
            if not getattr(self.args, flag, True):
                for profile in payload["pools"].get(pool, {}).values():
                    for point in profile["saturation"]:
                        point["be_throughput"] = 0.0
        return AllocationProfiles.from_dict(payload)

    def _solve(self, profiles: AllocationProfiles, total_gpus: int) -> Allocation:
        solver = ResourceAllocationSolver
        if getattr(self.args, "resource_allocation_policy", "saturation_aware") == "demand_based":
            from saturo.resource_allocation.demand import DemandBasedSolver
            solver = DemandBasedSolver
        allocations = solver(profiles).solve(
            total_gpus=total_gpus,
            workload=Workload(
                p_rps=self.args.target_rps,
                d_rps=self.args.target_rps,
            ),
            top_k=1,
        )
        if not allocations:
            raise RuntimeError("resource solver found no feasible allocation")
        return allocations[0]

    def _allocation_to_placements(
        self,
        allocation: Allocation,
        profiles: AllocationProfiles,
    ) -> list[Any]:
        placements: list[Any] = []
        gpu_groups = self._placement_gpu_groups(allocation)
        group_idx = 0
        for pool_alloc in (allocation.p, allocation.d, allocation.t):
            config = profiles.config(pool_alloc.config_name).to_dict()
            role = POOL_TO_ROLE[pool_alloc.pool]
            for _ in range(pool_alloc.instances):
                gpu_id = gpu_groups[group_idx] if gpu_groups else ""
                group_idx += 1
                placements.append(self.placement_factory(
                    role,
                    int(config.get("tp", pool_alloc.gpus_per_instance)),
                    int(config.get("pp", 1)),
                    int(config.get("ep", 0)),
                    gpu_id=gpu_id,
                ))
        return placements

    def _placement_gpu_groups(self, allocation: Allocation) -> list[str]:
        gpu_ids = [
            item.strip()
            for item in str(getattr(self.args, "gpu_ids", "")).split(",")
            if item.strip()
        ]
        if not gpu_ids:
            return []
        if len(gpu_ids) < allocation.total_gpus:
            raise ValueError(
                f"--gpu-ids provides {len(gpu_ids)} GPUs, but allocation "
                f"requires {allocation.total_gpus}")

        groups: list[str] = []
        cursor = 0
        for pool_alloc in (allocation.p, allocation.d, allocation.t):
            for _ in range(pool_alloc.instances):
                width = pool_alloc.gpus_per_instance
                group = gpu_ids[cursor:cursor + width]
                if len(group) != width:
                    raise ValueError("insufficient --gpu-ids for allocation")
                groups.append(",".join(group))
                cursor += width
        return groups

    def _write_plan_summary(
        self,
        profile_path: Path,
        allocation: Allocation,
        placements: Iterable[Any],
        total_gpus: int,
    ) -> None:
        payload = {
            "profile_path": str(profile_path),
            "allocation_policy": getattr(self.args, "resource_allocation_policy", "saturation_aware"),
            "target_rps": self.args.target_rps,
            "total_gpus": total_gpus,
            "allocation": allocation.to_dict(),
            "placements": [
                {
                    "role": item.role,
                    "config_name": getattr(item, "config_name", ""),
                    "tensor_parallel_size": getattr(
                        item, "tensor_parallel_size", 1),
                    "pipeline_parallel_size": getattr(
                        item, "pipeline_parallel_size", 1),
                    "gpu_id": getattr(item, "gpu_id", ""),
                }
                for item in placements
            ],
        }
        (self.output_dir / "resource_worker_plan.json").write_text(
            json.dumps(payload, indent=2))
