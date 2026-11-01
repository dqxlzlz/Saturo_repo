"""Workload statistics loader for the PD-aggregated colocation simulator.

Reads the shared Saturo workload CSV schema (columns: request_id, prompt_len,
output_len, arrival_offset_s) and computes mean prompt/output lengths. The RT
statistics always come from the same fixed chronological prefix used by the
event simulators.
"""

import csv
from typing import List, Optional, Tuple

from .config import WorkloadStats


def _read_len_pairs(path: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                prompt = int(float(row["prompt_len"]))
                output = int(float(row["output_len"]))
            except (KeyError, TypeError, ValueError):
                continue
            if prompt > 0 and output > 0:
                pairs.append((prompt, output))
    if not pairs:
        raise ValueError(f"no usable prompt_len/output_len rows in {path}")
    return pairs


def _mean(values: List[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_workload_stats(
    rt_csv: str,
    be_csv: Optional[str],
    base: WorkloadStats,
    *,
    num_req: int,
    target_rps: float,
) -> WorkloadStats:
    """Return stats for exactly ``num_req`` RT rows and the optional BE CSV."""
    from saturo.runtime.workload import load_rt_workload_csv

    stats = WorkloadStats(
        rt_prompt_len=base.rt_prompt_len,
        rt_output_len=base.rt_output_len,
        be_prompt_len=base.be_prompt_len,
        be_output_len=base.be_output_len,
    )
    rt_rows = load_rt_workload_csv(
        rt_csv,
        num_req=num_req,
        target_rps=target_rps,
    )
    stats.rt_prompt_len = _mean([row.prompt_len for row in rt_rows])
    stats.rt_output_len = _mean([row.output_len for row in rt_rows])
    if be_csv:
        pairs = _read_len_pairs(be_csv)
        stats.be_prompt_len = _mean([p for p, _ in pairs])
        stats.be_output_len = _mean([o for _, o in pairs])
    return stats
