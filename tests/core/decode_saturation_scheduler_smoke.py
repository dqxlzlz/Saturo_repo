#!/usr/bin/env python3
"""CPU-only smoke checks for decode-saturation scheduler behavior.

Run from the repository root:

    /root/hybrid_serve_venv/bin/python tests/core/decode_saturation_scheduler_smoke.py

The script instantiates the scheduler with tiny fake requests and fake cache
sizes. It exercises the decode-saturation scheduling path without starting an
engine, loading a model, or launching GPU kernels.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional


os.environ["VLLM_DECODE_SATURATION"] = "1"
os.environ["VLLM_DECODE_SATURATION_TBT_MS"] = "100"
os.environ["VLLM_DECODE_SATURATION_BE_BLOCKS"] = "8"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.core.utils import (  # noqa: E402
    create_dummy_prompt,
    get_sequence_groups,
    schedule_and_update_computed_tokens,
)
from vllm.config import CacheConfig, SchedulerConfig  # noqa: E402
from vllm.core.interfaces import AllocStatus  # noqa: E402
from vllm.core.scheduler import Scheduler  # noqa: E402


def make_scheduler(
    *,
    block_size: int = 4,
    be_blocks: int = 8,
    max_num_batched_tokens: int = 16,
    max_num_seqs: int = 4,
    max_model_len: int = 64,
    gpu_blocks: int = 32,
    max_num_partial_prefills: Optional[int] = None,
    max_long_partial_prefills: Optional[int] = None,
) -> Scheduler:
    os.environ["VLLM_DECODE_SATURATION_BE_BLOCKS"] = str(be_blocks)
    kwargs = {}
    if max_num_partial_prefills is not None:
        kwargs["max_num_partial_prefills"] = max_num_partial_prefills
    if max_long_partial_prefills is not None:
        kwargs["max_long_partial_prefills"] = max_long_partial_prefills

    scheduler_config = SchedulerConfig(
        "generate",
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        **kwargs,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = gpu_blocks
    cache_config.num_gpu_blocks = gpu_blocks
    return Scheduler(scheduler_config, cache_config, None)


def make_decode_request(
    request_id: str,
    *,
    rt: bool,
    block_size: int = 4,
):
    _, seq_group = create_dummy_prompt(
        request_id, block_size, block_size=block_size)
    if rt:
        seq_group.is_realtime = True
    return seq_group


def test_rt_decode_before_be_decode() -> None:
    scheduler = make_scheduler(be_blocks=1, max_num_seqs=1)
    be = make_decode_request("1", rt=False)
    rt = make_decode_request("2", rt=True)
    scheduler.waiting_be_decode_admission.append(be)
    scheduler.waiting_rt_decode_admission.append(rt)

    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [rt]
    print("PASS rt_decode_before_be_decode")


def test_be_decode_admission_respects_block_budget() -> None:
    scheduler = make_scheduler(be_blocks=1)
    be_a = make_decode_request("1", rt=False)
    be_b = make_decode_request("2", rt=False)
    scheduler.waiting_be_decode_admission.append(be_a)
    scheduler.waiting_be_decode_admission.append(be_b)

    assert list(scheduler.running) == []
    assert list(scheduler.waiting_be_decode_admission) == [be_a, be_b]

    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [be_a]
    assert list(scheduler.waiting_be_decode_admission) == [be_b]
    print("PASS be_decode_admission_respects_block_budget")


def test_be_decode_admission_respects_target_tbt() -> None:
    class FakeTimePredictor:

        @property
        def available(self):
            return True

        def predict_forward_ms(self, prefill_reqs, decode_reqs):
            del prefill_reqs
            return 40.0 * len(decode_reqs)

    scheduler = make_scheduler(be_blocks=8, max_num_seqs=3)
    scheduler.time_predictor = FakeTimePredictor()

    rt = make_decode_request("1", rt=True)
    be_a = make_decode_request("2", rt=False)
    be_b = make_decode_request("3", rt=False)
    scheduler.waiting_rt_decode_admission.append(rt)
    scheduler.waiting_be_decode_admission.append(be_a)
    scheduler.waiting_be_decode_admission.append(be_b)

    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [rt, be_a]
    assert list(scheduler.waiting_be_decode_admission) == [be_b]
    print("PASS be_decode_admission_respects_target_tbt")


def test_rt_decode_admission_is_fcfs() -> None:
    scheduler = make_scheduler(be_blocks=8, max_num_seqs=1)
    rt_a = make_decode_request("1", rt=True)
    rt_b = make_decode_request("2", rt=True)

    original_can_allocate = scheduler.block_manager.can_allocate
    admission_open = False

    def can_allocate(seq_group):
        if seq_group in (rt_a, rt_b) and not admission_open:
            return AllocStatus.LATER
        return original_can_allocate(seq_group)

    scheduler.block_manager.can_allocate = can_allocate
    scheduler.waiting_rt_decode_admission.append(rt_a)
    scheduler.waiting_rt_decode_admission.append(rt_b)
    assert list(scheduler.waiting_rt_decode_admission) == [rt_a, rt_b]

    admission_open = True
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [rt_a]
    assert list(scheduler.waiting_rt_decode_admission) == [rt_b]
    print("PASS rt_decode_admission_is_fcfs")


def test_be_prefill_only_and_iterative_slack_fill() -> None:
    scheduler = make_scheduler(
        be_blocks=8,
        max_num_batched_tokens=16,
        max_num_seqs=4,
        max_model_len=16,
        gpu_blocks=16,
        max_num_partial_prefills=2,
        max_long_partial_prefills=2,
    )
    _, rt_prefill = create_dummy_prompt("1", 4, block_size=4)
    _, be_a = create_dummy_prompt("2", 4, block_size=4)
    _, be_b = create_dummy_prompt("3", 4, block_size=4)
    rt_prefill.prefill_ddl = time.time() + 1.0
    be_a.prefill_ddl = float("inf")
    be_b.prefill_ddl = float("inf")

    scheduler.add_seq_group(rt_prefill)
    scheduler.add_seq_group(be_a)
    scheduler.add_seq_group(be_b)

    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [be_a, be_b]
    assert out.num_prefill_groups == 2
    assert list(scheduler.waiting) == [rt_prefill]
    print("PASS be_prefill_only_and_iterative_slack_fill")


def test_partial_be_prefill_continues_from_running() -> None:
    scheduler = make_scheduler(
        be_blocks=16,
        max_num_batched_tokens=4,
        max_num_seqs=2,
        max_model_len=64,
        gpu_blocks=32,
        max_num_partial_prefills=2,
        max_long_partial_prefills=2,
    )
    _, be = create_dummy_prompt("1", 12, block_size=4, max_tokens=8)
    be.prefill_ddl = float("inf")
    scheduler.add_seq_group(be)

    _, first = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(first) == [be]
    assert first.num_prefill_groups == 1
    assert first.scheduled_seq_groups[0].token_chunk_size == 4
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [be]
    assert be.is_prefill()
    assert be.get_num_uncomputed_tokens() == 8

    _, second = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(second) == [be]
    assert second.num_prefill_groups == 1
    assert second.scheduled_seq_groups[0].token_chunk_size == 4
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [be]
    assert be.is_prefill()
    assert be.get_num_uncomputed_tokens() == 4
    print("PASS partial_be_prefill_continues_from_running")


def main() -> None:
    test_rt_decode_before_be_decode()
    test_be_decode_admission_respects_block_budget()
    test_be_decode_admission_respects_target_tbt()
    test_rt_decode_admission_is_fcfs()
    test_be_prefill_only_and_iterative_slack_fill()
    test_partial_be_prefill_continues_from_running()
    print("ALL decode_saturation scheduler smoke checks passed.")


if __name__ == "__main__":
    main()
