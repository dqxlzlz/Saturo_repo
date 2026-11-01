"""Configuration objects for the T-pool BE-only throughput simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture parameters for per-GPU KV-cache estimation."""

    block_size: int = 16


@dataclass
class BEConfig:
    avg_input_len: int = 8000
    avg_output_len: int = 100
    avg_decode_context_len: Optional[int] = None
    # Physical KV capacity and the smaller admission limit used only by the
    # independent decode-ready BE population.  ``total_gpu_blocks`` falls back
    # to ``be_decode_blocks`` for older callers.
    total_gpu_blocks: int = 0
    be_decode_blocks: int = 0
    be_inflight: int = 128
    max_num_seqs: int = 256
    enable_be_decode: bool = True
    chunk_size_candidates: List[int] = field(
        default_factory=lambda: list(range(128, 24577, 128)))
    # Historical name: trace mode interprets these as total per-iteration
    # token budgets shared by decode and prefill.
    prefill_chunk_candidates: List[int] = field(
        default_factory=lambda: [128, 256, 512, 1024, 2048, 4096, 8192,
                                 16384, 24576])
    prefill_context_policy: str = "avg"

    def effective_decode_context_len(self) -> int:
        if self.avg_decode_context_len is not None:
            return max(1, int(self.avg_decode_context_len))
        return max(1, int(self.avg_input_len + self.avg_output_len / 2))


@dataclass
class SimulatorConfig:
    duration_s: float = 300.0
    warmup_s: float = 0.0
    latency_model_path: str = (
        "/root/hybrid_serve_proj/benchmarks/latency_model/results/"
        "qwen2_module_tp4/qwen2_modular_latency_model_tp4.json"
    )
    min_iteration_ms: float = 1.0
    model: ModelConfig = field(default_factory=ModelConfig)
    be: BEConfig = field(default_factory=BEConfig)
    # ------------------------------------------------------------------
    # Pipeline parallelism ("K staggered lanes" model).
    #
    # A PP=K engine runs K virtual engines: vLLM spreads requests across K
    # schedulers (min-cost assignment) and each scheduler owns
    # total_gpu_blocks // K KV blocks. Measured on Mistral-123B TP4xPP2
    # (2x4 GB200, chunk 1024/2048/4096) the lanes overlap almost perfectly
    # (K_eff = step_rate x step_latency = 1.98-2.00), so the simulator
    # models ONE lane with a 1/K share of the inflight population and
    # per-lane KV blocks, then scales system rates by K.
    #
    # NOTE: ``be.total_gpu_blocks``/``be.be_decode_blocks`` are therefore
    # interpreted as PER-LANE (per virtual engine) budgets, matching both
    # vLLM's per-scheduler split and the tp4_pp2 profile convention of
    # storing total_gpu_blocks // pp_size.
    pipeline_parallel_size: int = 1
    # Per-hop PP communication overhead added to every iteration latency:
    #   (K - 1) * max(0, pp_comm_ms_per_token * step_tokens + pp_comm_base_ms)
    # The blocking send_tensor_dict transfers hidden states for every token
    # in the step, so the cost grows with the token budget. Defaults were
    # least-squares fit on real-vs-predicted per-step gaps of Mistral-123B
    # TP4xPP2 cross-machine runs (chunks 1024/2048/4096; residual ~5%).
    pp_comm_ms_per_token: float = 0.0487
    pp_comm_base_ms: float = -30.5
