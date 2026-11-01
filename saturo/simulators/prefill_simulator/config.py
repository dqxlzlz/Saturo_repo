"""Configuration objects for the P-pool prefill simulator."""

from dataclasses import dataclass, field


@dataclass
class SLOConfig:
    ttft_slo_s: float = 2.0
    # If the pure-RT (BE-free) TTFT-SLO violation rate already exceeds this
    # fraction, the RPS point is beyond RT capacity: BE saturation is
    # undefined, so the optimizer short-circuits to saturation=0 without
    # searching. Range [0, 1]; 0.01 means "tolerate up to 1% RT violations".
    max_rt_violation_rate: float = 0.01


@dataclass
class BEConfig:
    free_gpu_blocks: int = 7200
    block_size: int = 16
    memory_watermark: float = 0.9
    avg_be_length: int = 8000
    avg_be_output_len: int = 100
    avg_be_decode_context: int = 8000
    max_prefill_chunk: int = 8192
    min_prefill_chunk: int = 0
    prefill_search_granularity: int = 1

    def average_prefill_context(self, prefill_chunk_size: int) -> int:
        """Estimate total context for a steady-state partial-prefill chunk.

        A long BE prompt is processed over many chunks.  Treating every chunk
        as the first chunk (context == query length) substantially
        underestimates attention time.  For a uniformly observed chunk within
        an average-length prompt, the expected end context is approximately
        half way between the first and final chunk.
        """
        chunk = max(0, int(prefill_chunk_size))
        if chunk == 0:
            return 0
        return min(
            max(chunk, int(self.avg_be_length)),
            max(chunk, int(round((self.avg_be_length + chunk) / 2))),
        )

    def steady_state_decode_batch(self, prefill_chunk_size: int) -> int:
        """Decode tokens naturally available per chunk in steady state.

        ``decode_batch_size`` is a KV-cache capacity ceiling, not a batch that
        can be sustained unconditionally.  Prefill completions replenish the
        decode population, so the steady-state batch is flow-balanced using
        the average input/output token ratio.
        """
        capacity = self.decode_batch_size
        if capacity <= 0 or prefill_chunk_size <= 0:
            return 0
        supplied = (
            float(prefill_chunk_size)
            * max(1, int(self.avg_be_output_len))
            / max(1, int(self.avg_be_length))
        )
        return min(capacity, max(1, int(round(supplied))))

    @property
    def decode_batch_size(self) -> int:
        # KV-cache-limited BE decode concurrency: how many BE decode requests
        # fit in the profiled GPU blocks. Each resident request holds its full
        # context (avg input + generated output) in the KV cache, so the
        # per-request footprint is avg_be_length + avg_be_output_len tokens.
        available_tokens = (
            self.free_gpu_blocks * self.block_size * self.memory_watermark
        )
        per_req_tokens = max(
            1, self.avg_be_length + self.avg_be_output_len)
        return max(0, int(available_tokens // per_req_tokens))


@dataclass
class SimulatorConfig:
    duration_s: float = 600.0
    # Fixed-count replay: ignore the duration window and run until every RT
    # request in the (fixed) trace subset has been served; throughput is then
    # normalized by the actual makespan instead of duration_s.
    run_to_completion: bool = False
    slo: SLOConfig = field(default_factory=SLOConfig)
    be: BEConfig = field(default_factory=BEConfig)
