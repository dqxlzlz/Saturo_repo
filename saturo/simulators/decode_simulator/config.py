"""Configuration for the decode simulator."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture parameters for KV cache memory estimation."""
    block_size: int = 16  # vLLM block size



@dataclass
class SLOConfig:
    """SLO targets."""
    tbt_slo_ms: float = 100.0        # per-token TBT SLO in ms
    ttft_slo_ms: float = 2000.0      # time-to-first-token SLO in ms


@dataclass
class SaturationConfig:
    """Saturation policy parameters."""
    lookahead_window_s: float = 5.0        # W: look-ahead window in seconds
    model_len_quantile: float = 0.95          # eta for Q_eta model-len prediction
    min_interval_ms: float = 5.0           # smallest candidate inter-token interval
    search_epsilon_ms: float = 0.5         # binary-search stop threshold
    memory_prediction_samples: int = 16    # look-ahead peak sampling density


@dataclass
class SimulatorConfig:
    """Top-level simulator configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    slo: SLOConfig = field(default_factory=SLOConfig)
    saturation: SaturationConfig = field(default_factory=SaturationConfig)

    # RT trace
    rt_trace_path: str = (
        "/root/hybrid_serve_proj/data/datasets/AzureLLMInferenceTrace_conv.csv"
    )

    # BE workload generation — lengths are loaded from be_workload_csv.
    be_chunk_sizes: List[int] = field(default_factory=lambda: list(range(16, 4097, 16)))

    # Best-Effort workload CSV (shared with the online scheduler's
    # --be-workload-csv). Columns: request_id, prompt_len, output_len,
    # arrival_offset_s. Only prompt_len/output_len are used here as length
    # templates. All rows are loaded into the waiting backlog at once and
    # cycled to keep the GPU saturated.
    be_workload_csv: Optional[str] = None

    # Balance the two independent BE streams on every decode iteration.
    # When enabled, the scheduler chooses the BE-decode batch whose predicted
    # request throughput is closest to the maximum feasible BE-prefill request
    # throughput under the same target-TBT and KV budgets.  Disabling this
    # restores the legacy decode-first greedy policy.
    balance_be_throughput: bool = True

    # Maximum model length applied to BE templates, mirroring the real
    # benchmark: BE prompts with prompt_len >= be_max_prompt_len are skipped and
    # output_len is clamped to (be_max_prompt_len - prompt_len). None disables
    # filtering (keeps raw CSV lengths). Matching real here is essential: real
    # runs with max_model_len=4096 drop the long-prompt rows, so the surviving
    # short prompts are what actually fill the decode pool.
    be_max_prompt_len: Optional[int] = None

    # GPU config
    # Total GPU KV-cache blocks for this parallel config. When set (loaded
    # from the TimePredictor profile), it overrides the rt+be budget-derived
    # total in the simulator.
    total_gpu_blocks: Optional[int] = None
    # Fraction of total_gpu_blocks usable for RT+BE allocation (KV cache
    # watermark). BE saturates all free blocks up to this cap; it also gates
    # the RT-admission swap threshold.
    memory_watermark: float = 0.9
    # KV blocks reserved for the longest upcoming BE-prefill input, subtracted
    # from the BE-decode budget so a prefill chunk can always be admitted.
    # Mirrors the real scheduler's SATURO_DECODE_BE_PREFILL_RESERVE_BLOCKS.
    be_prefill_reserve_blocks: int = 0

    # Simulation duration
    simulation_duration_s: float = 300.0   # simulate 5 minutes of traffic
    warmup_s: float = 10.0                 # warmup period to ignore

    # Latency model
    latency_model_path: str = (
        "/root/hybrid_serve_proj/benchmarks/latency_model/results/"
        "qwen2_module_tp4/qwen2_modular_latency_model_tp4.json"
    )

    # Remaining-output predictor for Progressive active RT requests.
    #
    # Option A（推荐）: 通过 dataset + parquet 自动 load-or-train。
    #   - predictor_dataset: 模型文件名中的标识，如 "wildchat_0000"
    #   - predictor_parquet: 对应的 WildChat parquet 路径（找不到模型时用于训练）
    #   - predictor_type / predictor_feature_mode: 选择模型类型
    # Option B（兼容）: 直接指定 .joblib 路径（旧行为保留）。
    predictor_dataset: Optional[str] = None
    predictor_parquet: Optional[str] = None
    predictor_type: str = "qrf"
    predictor_feature_mode: str = "token_ids"
    predictor_max_samples: int = 1000
    predictor_max_steps: int = 200
    # 保留旧字段供显式路径覆盖
    model_len_predictor_path: Optional[str] = None
    model_len_predictor_type: Optional[str] = None  # "qrf" or "gbdt" when set
