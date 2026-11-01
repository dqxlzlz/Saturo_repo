"""Request and result data models for P-pool simulation."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class RTRequest:
    request_id: int
    arrival_time_s: float
    runtime_s: float
    prompt_len: int = 0
    output_len: int = 1


@dataclass(frozen=True)
class BEChunk:
    target_latency_s: float
    latency_s: float
    decode_batch_size: int
    prefill_chunk_size: int

    @property
    def useful_prefill_tokens(self) -> int:
        return self.prefill_chunk_size

    @property
    def useful_decode_tokens(self) -> int:
        return self.decode_batch_size


@dataclass
class SimulationResult:
    chunk: BEChunk
    duration_s: float
    total_rt_requests: int = 0
    completed_rt_requests: int = 0
    rt_slo_violations: int = 0
    rt_slacks_s: List[float] = field(default_factory=list)
    preemptions: int = 0
    completed_be_chunks: int = 0
    wasted_be_time_s: float = 0.0
    useful_be_time_s: float = 0.0
    idle_time_s: float = 0.0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    prefill_req_per_s: float = 0.0
    decode_req_per_s: float = 0.0
    saturation_req_per_s: float = 0.0
    be_token_per_s: float = 0.0

    @property
    def min_slack_s(self) -> float:
        return min(self.rt_slacks_s) if self.rt_slacks_s else 0.0

    @property
    def rt_violation_rate(self) -> float:
        """Fraction of RT requests that violated the TTFT SLO."""
        if self.total_rt_requests <= 0:
            return 0.0
        return self.rt_slo_violations / self.total_rt_requests

    @property
    def preempt_ratio(self) -> float:
        """Fraction of launched BE chunks that were preempted by RT."""
        launched = self.completed_be_chunks + self.preemptions
        if launched <= 0:
            return 0.0
        return self.preemptions / launched

    @property
    def wasted_time_ratio(self) -> float:
        """Fraction of non-RT time thrown away by preempted BE chunks."""
        total = self.useful_be_time_s + self.wasted_be_time_s + self.idle_time_s
        if total <= 0.0:
            return 0.0
        return self.wasted_be_time_s / total
