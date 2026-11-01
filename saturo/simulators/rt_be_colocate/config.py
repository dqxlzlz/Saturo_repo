"""Configuration dataclasses for the RT-BE colocation simulator."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ColocateConfig:
    """Parameters for one simulation run.

    Attributes:
        ttft_slo_s: TTFT SLO threshold in seconds.  RT requests whose
            queueing + runtime exceeds this count as violations.
        duration_s: Length of the simulation window in seconds.
        be_chunk_latency_s: How long one BE chunk takes on the GPU.
            Larger values model bigger BE prefill batches.
        be_chunk_tokens: How many prefill tokens a single BE chunk
            processes.  Used to compute BE token throughput.
        policy: Scheduling policy for BE chunks.
            ``"greedy"`` — Start a BE chunk whenever the RT queue is
            empty, even if an RT arrival is imminent (non-preemptive).
            This maximises BE throughput at the cost of RT tail
            latency and is the recommended setting for studying the
            RT–BE tension.
            ``"conservative"`` — Only start a BE chunk when it can
            complete before the next known RT arrival.  This protects
            RT latency but under-utilises the GPU at low RPS.
    """

    ttft_slo_s: float = 2.0
    duration_s: float = 600.0
    be_chunk_latency_s: float = 0.050
    be_chunk_tokens: int = 4096
    policy: str = "greedy"

    def __post_init__(self) -> None:
        if self.policy not in ("greedy", "conservative"):
            raise ValueError(
                f"Unknown policy {self.policy!r}; expected 'greedy' or "
                "'conservative'")
        if self.be_chunk_latency_s < 0:
            raise ValueError("be_chunk_latency_s must be >= 0")
        if self.be_chunk_tokens < 1:
            raise ValueError("be_chunk_tokens must be >= 1")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be > 0")


@dataclass
class ColocateResult:
    """Aggregate metrics from one simulation run.

    Attributes:
        config: The config that produced this result (for bookkeeping).
        target_rps: Actual RT injection rate achieved.
        rt_total: Number of RT requests whose arrival fell in the window.
        rt_completed: Number of RT requests served.
        rt_violations: Number of RT requests whose TTFT exceeded the SLO.
        rt_violation_rate: ``rt_violations / rt_total`` (0 if none).
        rt_ttft_ms_p50: Median TTFT in milliseconds.
        rt_ttft_ms_p95: 95th-percentile TTFT in milliseconds.
        rt_ttft_ms_p99: 99th-percentile TTFT in milliseconds.
        rt_mean_queue_s: Mean RT queueing delay in seconds.
        be_chunks_completed: Total BE chunks completed.
        be_total_tokens: Total BE prefill tokens processed.
        be_throughput_tok_s: BE token throughput (tokens / wall second).
        be_utilization: Fraction of wall time spent on BE work.
        rt_utilization: Fraction of wall time spent on RT work.
        idle_time_s: Cumulative idle time (neither RT nor BE).
        preemptions: Not used by the colocate simulator (always 0).
    """

    config: ColocateConfig = field(repr=False)
    target_rps: float = 0.0
    rt_total: int = 0
    rt_completed: int = 0
    rt_violations: int = 0
    rt_violation_rate: float = 0.0
    rt_ttft_ms_p50: float = 0.0
    rt_ttft_ms_p95: float = 0.0
    rt_ttft_ms_p99: float = 0.0
    rt_mean_queue_s: float = 0.0
    be_chunks_completed: int = 0
    be_total_tokens: int = 0
    be_throughput_tok_s: float = 0.0
    be_utilization: float = 0.0
    rt_utilization: float = 0.0
    idle_time_s: float = 0.0
    preemptions: int = 0

    def to_dict(self) -> dict:
        return {
            "target_rps": round(self.target_rps, 4),
            "be_chunk_latency_ms": round(self.config.be_chunk_latency_s * 1000, 2),
            "be_chunk_tokens": self.config.be_chunk_tokens,
            "policy": self.config.policy,
            "ttft_slo_ms": round(self.config.ttft_slo_s * 1000, 2),
            "duration_s": self.config.duration_s,
            "rt_total": self.rt_total,
            "rt_completed": self.rt_completed,
            "rt_violations": self.rt_violations,
            "rt_violation_rate": round(self.rt_violation_rate, 6),
            "rt_ttft_ms_p50": round(self.rt_ttft_ms_p50, 3),
            "rt_ttft_ms_p95": round(self.rt_ttft_ms_p95, 3),
            "rt_ttft_ms_p99": round(self.rt_ttft_ms_p99, 3),
            "rt_mean_queue_s": round(self.rt_mean_queue_s, 6),
            "be_chunks_completed": self.be_chunks_completed,
            "be_total_tokens": self.be_total_tokens,
            "be_throughput_tok_s": round(self.be_throughput_tok_s, 1),
            "be_utilization": round(self.be_utilization, 6),
            "rt_utilization": round(self.rt_utilization, 6),
            "idle_time_s": round(self.idle_time_s, 6),
        }
