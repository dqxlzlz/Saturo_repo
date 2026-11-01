# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared data structures for the D-pool decode saturation algorithm.

These types are used by both the runtime policy (vLLM scheduler integration)
and the offline trace-driven simulator.
"""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class RequestForecast:
    """Unified RT request forecast state.

    Replaces the former ``DecodeSaturationRequestState`` (runtime) and
    ``_RTForecast`` (simulator) with a single shared definition.
    """

    request_id: Union[int, str]
    prompt_len: int
    generated_tokens: int
    remaining_tokens: int
    arrival_time_s: float
    is_rt: bool = True


@dataclass(frozen=True)
class DecodeSaturationDecision:
    """Result of the saturation decision algorithm.

    Shared between the runtime scheduler and the simulator.
    """

    enabled: bool
    t_rate_ms: float
    target_interval_ms: float
    rt_only_tbt_ms: float
    predicted_peak_rt_blocks: int
    current_rt_blocks: int
    current_be_blocks: int
    be_block_budget: int
    max_be_prefill_tokens: int
    search_feasible: bool = True
    target_version: int = 0
    target_epoch: int = 0
    target_snapshot_time_s: float = 0.0
    target_status: str = "sync"


@dataclass(frozen=True)
class SaturationParams:
    """Configuration parameters for the core saturation engine.

    Decoupled from any vLLM config object so the engine can be used
    standalone by the simulator or other callers.
    """

    tbt_slo_ms: float = 100.0
    lookahead_window_s: float = 5.0
    min_interval_ms: float = 1.0
    search_epsilon_ms: float = 0.5
    block_size: int = 16
    total_gpu_blocks: int = 0
    # Watermark fraction of physical KV blocks the BE-decode budget may fill.
    decode_block_fraction: float = 1.0
    # KV blocks reserved for the longest upcoming BE-prefill input so a prefill
    # chunk can always be admitted even when decode has filled the cache.
    be_prefill_reserve_blocks: int = 0
