"""Profile tables consumed by the resource allocation solver.

The solver is intentionally decoupled from the P/D saturation simulators. Each
profile entry records the measured or simulated capacity of one pool instance
under one parallel configuration. Resource allocation then becomes a pure
integer optimization over these profile tables.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union


POOL_P = "P"
POOL_D = "D"
POOL_T = "T"
POOLS = (POOL_P, POOL_D, POOL_T)


@dataclass(frozen=True)
class ParallelConfig:
    """One feasible instance-level parallelism configuration."""

    name: str
    gpus: int
    tp: Optional[int] = None
    pp: Optional[int] = None
    metadata: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_dict(cls, name: str, payload: Mapping[str, Any]) -> "ParallelConfig":
        gpus = int(payload.get("gpus", payload.get("gpu", 0)))
        if gpus <= 0:
            raise ValueError(f"config {name!r} must use at least one GPU")
        metadata = {
            k: v for k, v in payload.items()
            if k not in {"gpus", "gpu", "tp", "pp", "name"}
        }
        return cls(
            name=name,
            gpus=gpus,
            tp=_optional_int(payload.get("tp")),
            pp=_optional_int(payload.get("pp")),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"name": self.name, "gpus": self.gpus}
        if self.tp is not None:
            result["tp"] = self.tp
        if self.pp is not None:
            result["pp"] = self.pp
        if self.metadata:
            result.update(dict(self.metadata))
        return result


@dataclass(frozen=True)
class SaturationPoint:
    """One measured phi_y(x, w) sample."""

    rt_rps: float
    be_throughput: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SaturationPoint":
        rt_rps = float(payload.get("rt_rps", payload.get("workload", 0.0)))
        be_throughput = float(payload.get(
            "be_throughput",
            payload.get("be_tput", payload.get("phi", 0.0)),
        ))
        if rt_rps < 0:
            raise ValueError("rt_rps must be non-negative")
        if be_throughput < 0:
            raise ValueError("be throughput must be non-negative")
        return cls(rt_rps=rt_rps, be_throughput=be_throughput)

    def to_dict(self) -> Dict[str, float]:
        return {
            "rt_rps": self.rt_rps,
            "be_throughput": self.be_throughput,
        }


@dataclass(frozen=True)
class PoolProfile:
    """Profile data for a single pool and parallel config."""

    pool: str
    config_name: str
    mu: Optional[float]
    saturation: Sequence[SaturationPoint]
    be_chunk_latency_ms: Optional[float] = None

    @classmethod
    def from_dict(
        cls,
        pool: str,
        config_name: str,
        payload: Mapping[str, Any],
    ) -> "PoolProfile":
        mu_value = payload.get("mu", payload.get("rt_capacity"))
        mu = None if mu_value is None else float(mu_value)
        if mu is not None and mu < 0:
            raise ValueError(f"mu for {pool}/{config_name} must be non-negative")

        points = [
            SaturationPoint.from_dict(item)
            for item in payload.get("saturation", payload.get("phi", []))
        ]
        if not points:
            raise ValueError(f"{pool}/{config_name} has no saturation samples")

        be_chunk_latency_ms = payload.get("be_chunk_latency_ms")
        if be_chunk_latency_ms is not None:
            be_chunk_latency_ms = float(be_chunk_latency_ms)

        return cls(
            pool=pool,
            config_name=config_name,
            mu=mu,
            saturation=tuple(sorted(points, key=lambda p: p.rt_rps)),
            be_chunk_latency_ms=be_chunk_latency_ms,
        )

    def phi(self, rt_rps: float) -> Optional[float]:
        """Return phi_y(x, w) using conservative linear interpolation.

        If ``rt_rps`` is below the smallest sampled RT workload, the first
        sample is reused. This is conservative when BE saturation decreases as
        RT load grows. If it is above the largest sample, no value is returned:
        the profile has not demonstrated feasibility at that load.
        """

        if rt_rps < 0:
            return None
        points = list(self.saturation)
        if not points:
            return None
        if rt_rps <= points[0].rt_rps:
            return points[0].be_throughput
        if rt_rps > points[-1].rt_rps:
            return None

        for left, right in zip(points, points[1:]):
            if left.rt_rps <= rt_rps <= right.rt_rps:
                width = right.rt_rps - left.rt_rps
                if width <= 0:
                    return min(left.be_throughput, right.be_throughput)
                ratio = (rt_rps - left.rt_rps) / width
                return (
                    left.be_throughput
                    + ratio * (right.be_throughput - left.be_throughput)
                )
        return None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "saturation": [point.to_dict() for point in self.saturation],
        }
        if self.mu is not None:
            result["mu"] = self.mu
        if self.be_chunk_latency_ms is not None:
            result["be_chunk_latency_ms"] = self.be_chunk_latency_ms
        return result


@dataclass(frozen=True)
class AllocationProfiles:
    """Complete profile table for all pools."""

    configs: Mapping[str, ParallelConfig]
    pools: Mapping[str, Mapping[str, PoolProfile]]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AllocationProfiles":
        configs_payload = payload.get("configs", {})
        configs: Dict[str, ParallelConfig] = {}
        if isinstance(configs_payload, Mapping):
            for name, config_payload in configs_payload.items():
                configs[name] = ParallelConfig.from_dict(name, config_payload)
        else:
            for item in configs_payload:
                name = str(item["name"])
                configs[name] = ParallelConfig.from_dict(name, item)

        pools_payload = payload.get("pools", {})
        pools: Dict[str, Dict[str, PoolProfile]] = {}
        for pool, entries in pools_payload.items():
            pool_name = str(pool).upper()
            if pool_name not in POOLS:
                raise ValueError(f"unknown pool {pool!r}")
            pools[pool_name] = {}
            for config_name, entry_payload in entries.items():
                if config_name not in configs:
                    raise ValueError(
                        f"{pool_name}/{config_name} references missing config")
                pools[pool_name][config_name] = PoolProfile.from_dict(
                    pool_name, config_name, entry_payload)

        for pool_name in POOLS:
            pools.setdefault(pool_name, {})
        return cls(configs=configs, pools=pools)

    def profile(self, pool: str, config_name: str) -> PoolProfile:
        return self.pools[pool][config_name]

    def config(self, config_name: str) -> ParallelConfig:
        return self.configs[config_name]

    def config_names_for_pool(self, pool: str) -> Iterable[str]:
        return self.pools.get(pool, {}).keys()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configs": {
                name: config.to_dict()
                for name, config in sorted(self.configs.items())
            },
            "pools": {
                pool: {
                    name: profile.to_dict()
                    for name, profile in sorted(entries.items())
                }
                for pool, entries in sorted(self.pools.items())
            },
        }


def load_profiles(path: Union[str, Path]) -> AllocationProfiles:
    with open(path, "r", encoding="utf-8") as f:
        return AllocationProfiles.from_dict(json.load(f))


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)
