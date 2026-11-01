"""RT-BE colocation simulator for Prefill-node limitation studies.

Simulates the tension between RT prefill latency and BE prefill
throughput when they share a single GPU.  Sweeps over BE chunk size
and RT RPS to quantify the trade-off.
"""

from .config import ColocateConfig, ColocateResult  # noqa: F401
from .simulator import ColocateSimulator  # noqa: F401
