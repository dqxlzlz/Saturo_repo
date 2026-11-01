# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Saturo D-pool decode saturation module.

Public API for the shared saturation algorithm used by both the vLLM runtime
scheduler and the offline trace-driven simulator.
"""

from .types import (  # noqa: F401
    DecodeSaturationDecision,
    RequestForecast,
    SaturationParams,
)
from .decode_saturation_core import (  # noqa: F401
    DecodeSaturationEngine,
    DecodeSaturationPolicy,
    DecodeSaturationStateProvider,
    DefaultDecodeSaturationStateProvider,
    OracleDecodeSaturationStateProvider,
    TimePredictorFn,
    blocks_for_tokens,
    forecast_length_after_elapsed,
    sampling_offsets,
)

# Backward-compatible alias: the old name used by tests and vllm scheduler.
DecodeSaturationRequestState = RequestForecast

__all__ = [
    "DecodeSaturationDecision",
    "DecodeSaturationEngine",
    "DecodeSaturationPolicy",
    "DecodeSaturationRequestState",
    "DecodeSaturationStateProvider",
    "DefaultDecodeSaturationStateProvider",
    "OracleDecodeSaturationStateProvider",
    "RequestForecast",
    "SaturationParams",
    "TimePredictorFn",
    "blocks_for_tokens",
    "forecast_length_after_elapsed",
    "sampling_offsets",
]
