"""Request metadata helpers used by Saturo pool schedulers.

Saturo distinguishes request QoS (RT vs BE) from pool role (P, D, T) and from
the current execution phase (prefill vs decode).  RT requests are identified by
the ``is_realtime`` attribute on the sampling params / sequence group.
"""

from __future__ import annotations

import enum
from typing import Any


class PoolRole(str, enum.Enum):
    PREFILL = "P"
    DECODE = "D"
    THROUGHPUT = "T"


class RequestQoS(str, enum.Enum):
    REAL_TIME = "rt"
    BEST_EFFORT = "be"


def get_prefill_deadline(obj: Any) -> float:
    """Return the request's prefill deadline or infinity for BE requests."""
    return float(getattr(obj, "prefill_ddl", float("inf")))


def is_rt_request(obj: Any) -> bool:
    """True when the request should be protected by RT SLO logic."""
    return bool(getattr(obj, "is_realtime", False))


def is_be_request(obj: Any) -> bool:
    return not is_rt_request(obj)


def request_qos(obj: Any) -> RequestQoS:
    return RequestQoS.REAL_TIME if is_rt_request(obj) else RequestQoS.BEST_EFFORT


def is_decode_phase(seq_group: Any) -> bool:
    """True for post-prefill sequence groups (already in decode state)."""
    return not seq_group.is_prefill()


def is_prefill_phase(seq_group: Any) -> bool:
    return not is_decode_phase(seq_group)
