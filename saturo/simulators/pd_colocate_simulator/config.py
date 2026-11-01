"""Configuration dataclasses for the PD-aggregated colocation simulator."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """KV-cache capacity for the target parallel config.

    total_gpu_blocks / block_size are normally read from the runtime profile
    (``kv_cache_block`` section). memory_watermark caps the fraction of blocks
    usable for RT + BE KV cache.
    """
    total_gpu_blocks: int = 5268
    block_size: int = 16
    memory_watermark: float = 0.9

    @property
    def kv_block_budget(self) -> int:
        return max(1, int(self.total_gpu_blocks * self.memory_watermark))


@dataclass
class WorkloadStats:
    """Average request shape for RT and BE, in tokens."""
    rt_prompt_len: float = 1157.0
    rt_output_len: float = 258.0
    be_prompt_len: float = 8000.0
    be_output_len: float = 256.0


@dataclass
class SolverConfig:
    """Fixed-point solver knobs."""
    max_iters: int = 400
    tol_s: float = 1e-5
    damping: float = 0.5
    # Balance BE prefill/decode request throughput inside every fixed-point
    # iteration. Disabling restores the legacy decode-first greedy policy.
    balance_be_throughput: bool = True
    # Stop tolerance for the continuous BE-decode batch-size binary search.
    be_balance_search_epsilon: float = 1e-3
    # Iteration latency is deemed divergent (RT over capacity) once it exceeds
    # this multiple of the decode-only-with-tiny-batch latency.
    divergence_factor: float = 200.0


@dataclass
class PDColocateConfig:
    """Top-level configuration for one chunk-size sweep at a target RPS."""
    runtime_profile: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    workload: WorkloadStats = field(default_factory=WorkloadStats)
    solver: SolverConfig = field(default_factory=SolverConfig)

    # Candidate per-iteration token budgets (max_num_batched_tokens) to sweep.
    # The experiment asks for a roughly 64-token search granularity.
    chunk_sizes: List[int] = field(
        default_factory=lambda: list(range(64, 8192 + 1, 64)))

    # Input CSVs (used only to compute WorkloadStats when provided).
    rt_workload_csv: Optional[str] = None
    be_workload_csv: Optional[str] = None
