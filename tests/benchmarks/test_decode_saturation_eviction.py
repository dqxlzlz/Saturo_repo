"""Unit tests for decode-saturation native CPU KV swapping in the vLLM scheduler.

These tests exercise ``_prepare_decode_saturation_blocks``,
``_smallest_running_be``, ``_restore_decode_saturation_be``, and related
helpers directly with minimal mocking so we verify scheduler-level
correctness beyond the decode-simulator model.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceStatus, SequenceStage


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
TOTAL_GPU_BLOCKS = 100


def _make_seq_group(
    *,
    request_id: str,
    prompt_len: int,
    output_len: int,
    is_be: bool = False,
    status: int = SequenceStatus.RUNNING,
) -> MagicMock:
    """Create a minimal mock ``SequenceGroup`` with one ``Sequence``."""
    seq = MagicMock()
    seq.seq_id = hash(request_id) & 0x7FFFFFFF
    seq.status = status
    seq.data.get_len.return_value = prompt_len + output_len
    seq.data.stage = SequenceStage.DECODE if status == SequenceStatus.RUNNING else SequenceStage.PREFILL
    seq.get_len.return_value = prompt_len + output_len
    seq.get_token_ids.return_value = [1] * (prompt_len + output_len)
    seq.get_num_new_tokens.return_value = 0 if status == SequenceStatus.RUNNING else prompt_len
    # Make MagicMock's default __lt__ work for PartialPrefillMetadata
    seq.__lt__ = lambda self, other: id(self) < id(other)

    sg = MagicMock()
    sg.request_id = request_id
    sg.get_seqs.return_value = [seq]
    sg.first_seq = seq
    sg.is_be = is_be
    sg.get_max_num_running_seqs.return_value = 1
    sg.seqs_dict = {seq.seq_id: seq}
    sg.__hash__ = lambda self: hash(self.request_id)
    sg.__eq__ = lambda self, other: self.request_id == other.request_id
    return sg


def _make_block_manager(
    *,
    total_blocks: int = TOTAL_GPU_BLOCKS,
    free_blocks: int = 80,
    watermark_blocks: int = 5,
    block_size: int = BLOCK_SIZE,
) -> MagicMock:
    bm = MagicMock()
    bm.watermark_blocks = watermark_blocks
    bm.block_size = block_size
    bm.get_num_free_gpu_blocks.return_value = free_blocks
    bm.get_block_table.return_value = [0, 1, 2]  # 3 blocks per running seq
    bm.get_num_blocks_touched_by_append_slots.return_value = 1
    bm.get_num_required_blocks.return_value = 4  # 4 blocks for admission
    bm.can_allocate.return_value = AllocStatus.OK
    bm.can_append_slots.return_value = True
    bm.can_swap_out.return_value = True
    bm.can_swap_in.return_value = AllocStatus.OK
    bm.swap_out.return_value = [(0, 100)]
    bm.swap_in.return_value = [(100, 10)]
    return bm


def _make_decode_saturation(is_be_fn=None, seq_group_blocks_fn=None) -> MagicMock:
    ds = MagicMock()
    ds.is_be = is_be_fn or (lambda sg: sg.is_be)
    ds.is_rt = lambda sg: not sg.is_be
    ds.seq_group_blocks = seq_group_blocks_fn or (lambda sg: 3)
    ds.provider = MagicMock()
    ds.provider.get_remaining_tokens.return_value = 5
    ds.provider.get_future_arrivals.return_value = ()
    ds.make_decision.return_value = MagicMock(
        be_block_budget=50,
        max_be_prefill_tokens=50 * BLOCK_SIZE,
        target_interval_ms=100.0,
    )
    ds.total_gpu_blocks = TOTAL_GPU_BLOCKS
    ds.enabled = True
    ds.policy = MagicMock()
    ds.policy.tbt_slo_ms = 100.0
    ds.policy.total_gpu_blocks = TOTAL_GPU_BLOCKS
    ds.policy.block_size = BLOCK_SIZE
    return ds


def _make_scheduler(
    *,
    running: list[MagicMock] | None = None,
    waiting: list[MagicMock] | None = None,
    waiting_rt_decode_admission: list[MagicMock] | None = None,
    waiting_be_decode_admission: list[MagicMock] | None = None,
    swapped: Deque[MagicMock] | None = None,
    block_manager: MagicMock | None = None,
    decode_saturation: MagicMock | None = None,
) -> Scheduler:
    """Build a scheduler with just enough state for the eviction methods."""
    sched = Scheduler.__new__(Scheduler)
    sched.running = deque(running or [])
    sched.waiting = deque(waiting or [])
    sched.waiting_rt_decode_admission = list(waiting_rt_decode_admission or [])
    sched.waiting_be_decode_admission = list(waiting_be_decode_admission or [])
    sched.swapped = swapped or deque()
    sched.block_manager = block_manager or _make_block_manager()
    sched.decode_saturation = decode_saturation or _make_decode_saturation()
    sched.cache_config = MagicMock(block_size=BLOCK_SIZE)
    sched.scheduler_config = MagicMock()
    sched.scheduler_config.max_num_batched_tokens = 16384
    sched.scheduler_config.max_num_seqs = 1024
    sched.scheduler_config.max_num_partial_prefills = 8
    sched.scheduler_config.max_long_partial_prefills = 4
    sched.scheduler_config.long_prefill_token_threshold = 16384
    sched.scheduler_config.is_multi_step = False
    sched.time_predictor = None
    sched.free_seq = MagicMock()
    sched._free_seq_group_cross_attn_blocks = MagicMock()
    sched._get_num_lookahead_slots = MagicMock(return_value=2)
    sched._schedule_default = MagicMock(return_value="default_scheduled")
    sched._admit_decode_admission_queues = MagicMock(return_value=0)
    sched._schedule_decode_for_saturation = MagicMock(return_value=None)
    sched._schedule_be_prefill_chunks_for_saturation = MagicMock()
    sched._order_finishing_prefills_first = lambda x: x
    sched.chunked_prefill_enabled = False
    return sched


def test_rt_with_direct_decode_marker_enters_decode_admission() -> None:
    """A phase-isolated RT request skips synthetic prefill without an oracle."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.decode_saturation = MagicMock(uses_oracle=False)
    scheduler.waiting = deque()
    scheduler.waiting_rt_decode_admission = deque()
    scheduler.waiting_be_decode_admission = deque()

    seq = MagicMock()
    seq.get_prompt_len.return_value = 123
    seq.get_last_token_id.return_value = 7
    seq_group = MagicMock()
    seq_group.sampling_params.extra_args = {
        "saturo_direct_decode": True,
    }
    seq_group.get_seqs.return_value = [seq]

    with (
        patch("vllm.core.scheduler.is_rt_request", return_value=True),
        patch("vllm.core.scheduler.is_be_request", return_value=False),
    ):
        Scheduler.add_seq_group(scheduler, seq_group)

    seq.data.update_num_computed_tokens.assert_called_once_with(123)
    seq.append_token_id.assert_called_once()
    assert list(scheduler.waiting_rt_decode_admission) == [seq_group]
    assert not scheduler.waiting


def test_be_with_direct_decode_marker_enters_decode_admission() -> None:
    """A phase-isolated BE request skips the synthetic hand-off iteration."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.decode_saturation = MagicMock(uses_oracle=False, enabled=True)
    scheduler.waiting = deque()
    scheduler.waiting_rt_decode_admission = deque()
    scheduler.waiting_be_decode_admission = deque()

    seq = MagicMock()
    seq.get_prompt_len.return_value = 456
    seq.get_last_token_id.return_value = 9
    seq_group = MagicMock()
    seq_group.sampling_params.extra_args = {
        "saturo_direct_decode": True,
    }
    seq_group.get_seqs.return_value = [seq]

    with (
        patch("vllm.core.scheduler.is_rt_request", return_value=False),
        patch("vllm.core.scheduler.is_be_request", return_value=True),
    ):
        Scheduler.add_seq_group(scheduler, seq_group)

    seq.data.update_num_computed_tokens.assert_called_once_with(456)
    seq.append_token_id.assert_called_once()
    assert list(scheduler.waiting_be_decode_admission) == [seq_group]
    assert not scheduler.waiting


# ======================================================================
# _smallest_running_be
# ======================================================================

def test_smallest_running_be_returns_none_when_no_be_running():
    rt = _make_seq_group(request_id="rt_0", prompt_len=100, output_len=50, is_be=False)
    sched = _make_scheduler(running=[rt])
    result = Scheduler._smallest_running_be(sched)
    assert result is None


def test_smallest_running_be_picks_fewest_blocks():
    be_short = _make_seq_group(request_id="be_short", prompt_len=10, output_len=10, is_be=True)
    be_long = _make_seq_group(request_id="be_long", prompt_len=100, output_len=50, is_be=True)

    def get_block_table(seq):
        if seq.seq_id == hash("be_short") & 0x7FFFFFFF:
            return [0]  # 1 block
        return [0, 1, 2, 3, 4]  # 5 blocks

    bm = _make_block_manager()
    bm.get_block_table = get_block_table
    sched = _make_scheduler(running=[be_long, be_short], block_manager=bm)
    result = Scheduler._smallest_running_be(sched)
    assert result is not None
    assert result.request_id == "be_short"


# ======================================================================
# _decode_saturation_required_free_blocks
# ======================================================================

def test_required_free_blocks_counts_running_decodes():
    """Every running decode group contributes append-touch blocks."""
    rt = _make_seq_group(request_id="rt_0", prompt_len=100, output_len=50, is_be=False)
    sched = _make_scheduler(running=[rt])
    # is_decode_phase(rt) == True:
    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        required = Scheduler._decode_saturation_required_free_blocks(sched)
    # 1 running decode * 1 append block = 1
    assert required == 1


def test_required_free_blocks_adds_watermark_on_rt_admission():
    rt = _make_seq_group(request_id="rt_new", prompt_len=50, output_len=20, is_be=False)
    bm = _make_block_manager(free_blocks=10, watermark_blocks=5)
    sched = _make_scheduler(
        running=[],
        waiting_rt_decode_admission=[rt],
        block_manager=bm,
    )
    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        required = Scheduler._decode_saturation_required_free_blocks(sched)
    # 4 blocks required + 5 watermark = 9
    assert required == 9


def test_required_free_blocks_skips_never_allocable():
    rt = _make_seq_group(request_id="rt_huge", prompt_len=999999, output_len=100, is_be=False)
    bm = _make_block_manager(free_blocks=10)
    bm.can_swap_in.return_value = AllocStatus.NEVER
    bm.can_allocate.return_value = AllocStatus.NEVER  # actual enum
    sched = _make_scheduler(
        running=[],
        waiting_rt_decode_admission=[rt],
        block_manager=bm,
    )
    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        required = Scheduler._decode_saturation_required_free_blocks(sched)
    # NEVER-allocable is skipped, no watermark added
    assert required == 0


# ======================================================================
# _prepare_decode_saturation_blocks (the core eviction loop)
# ======================================================================

def test_eviction_stops_when_enough_free_blocks():
    """When free_blocks >= required, no eviction happens."""
    rt = _make_seq_group(request_id="rt_0", prompt_len=100, output_len=50, is_be=False)
    be = _make_seq_group(request_id="be_0", prompt_len=50, output_len=30, is_be=True)
    bm = _make_block_manager(free_blocks=100, watermark_blocks=5)
    sched = _make_scheduler(running=[rt, be], block_manager=bm)

    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        num_swapped = Scheduler._prepare_decode_saturation_blocks(sched)

    assert num_swapped == 0
    assert be in sched.running  # BE is NOT evicted


def test_eviction_frees_be_sequence_slot_for_waiting_rt():
    """RT admission evicts BE even when KV blocks are plentiful."""
    be_a = _make_seq_group(
        request_id="be_a", prompt_len=50, output_len=30, is_be=True)
    be_b = _make_seq_group(
        request_id="be_b", prompt_len=50, output_len=30, is_be=True)
    rt = _make_seq_group(
        request_id="rt_new", prompt_len=100, output_len=50, is_be=False)
    bm = _make_block_manager(free_blocks=100, watermark_blocks=0)
    sched = _make_scheduler(
        running=[be_a, be_b],
        waiting_rt_decode_admission=[rt],
        block_manager=bm,
    )
    sched.scheduler_config.max_num_seqs = 2

    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        num_swapped = Scheduler._prepare_decode_saturation_blocks(sched)

    assert num_swapped == 1
    assert len(sched.running) == 1
    assert len(sched.swapped) == 1
    assert rt in sched.waiting_rt_decode_admission
    sched.block_manager.swap_out.assert_called_once()


def test_eviction_removes_shortest_be_when_blocks_are_tight():
    """Skewed demand forces eviction of the smallest BE group."""
    be_short = _make_seq_group(request_id="be_short", prompt_len=10, output_len=10, is_be=True)
    be_long = _make_seq_group(request_id="be_long", prompt_len=100, output_len=50, is_be=True)
    rt = _make_seq_group(request_id="rt_0", prompt_len=100, output_len=50, is_be=False)

    bm = _make_block_manager(free_blocks=3, watermark_blocks=5)
    bm.get_num_blocks_touched_by_append_slots.return_value = 2  # each running decode needs 2
    bm.get_num_required_blocks.return_value = 4

    # Block table sizes: be_short=1 block, be_long=5 blocks
    def get_block_table(seq):
        if seq.seq_id == hash("be_short") & 0x7FFFFFFF:
            return [0]
        return [0, 1, 2, 3, 4]

    bm.get_block_table = get_block_table

    sched = _make_scheduler(
        running=[be_long, be_short, rt],
        block_manager=bm,
    )

    # Simulate free blocks increasing after each eviction (be_short frees 1 block)
    free_blocks_val = [3]

    def free_side_effect(seq):
        n = len(get_block_table(seq))
        free_blocks_val[0] += n
        bm.get_num_free_gpu_blocks.return_value = free_blocks_val[0]

    bm.swap_out.side_effect = lambda g: (free_side_effect(g.get_seqs()[0]) or [(0, 100)])

    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        num_swapped = Scheduler._prepare_decode_saturation_blocks(sched)

    assert num_swapped == 1
    assert be_short not in sched.running
    assert be_long in sched.running
    assert rt in sched.running
    # Verify the evicted BE is in swapped queue
    assert be_short in sched.swapped
    # Verify free_seq was called
    sched.block_manager.swap_out.assert_called_once()


def test_eviction_evicts_multiple_be_when_one_is_not_enough():
    """If evicting one BE doesn't free enough, evict more."""
    be_1 = _make_seq_group(request_id="be_1", prompt_len=20, output_len=20, is_be=True)
    be_2 = _make_seq_group(request_id="be_2", prompt_len=20, output_len=20, is_be=True)
    rt = _make_seq_group(request_id="rt_0", prompt_len=100, output_len=50, is_be=False)

    bm = _make_block_manager(free_blocks=2, watermark_blocks=5)
    # Each running decode needs 4 append blocks, rt needs 5 admission blocks + 5 watermark = 10
    bm.get_num_blocks_touched_by_append_slots.return_value = 4
    bm.get_num_required_blocks.return_value = 5
    bm.get_block_table.return_value = [0, 1]  # 2 blocks each

    sched = _make_scheduler(
        running=[be_1, be_2, rt],
        waiting_rt_decode_admission=[],  # no new RT admissions
        block_manager=bm,
    )

    def free_side_effect(seq):
        # After each eviction, free blocks increase by 2 (BE blocks)
        current_free = bm.get_num_free_gpu_blocks.return_value
        bm.get_num_free_gpu_blocks.return_value = current_free + 2

    bm.swap_out.side_effect = lambda g: (free_side_effect(g.get_seqs()[0]) or [(0, 100)])

    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        num_swapped = Scheduler._prepare_decode_saturation_blocks(sched)

    # 3 running decode * 4 = 12 required. Initial free=2.
    # Free 2 blocks (evict be_1): free=4, required recalculated: 2 running * 4 = 8
    # Still 4 < 8, evict be_2: free=6, required: 1 running * 4 = 4
    # 6 >= 4, stop.
    assert num_swapped == 2
    assert be_1 not in sched.running
    assert be_2 not in sched.running
    assert rt in sched.running


def test_eviction_stops_when_no_more_be():
    """When all BEs are evicted but still not enough, loop breaks cleanly."""
    be = _make_seq_group(request_id="be_0", prompt_len=20, output_len=20, is_be=True)
    rt = _make_seq_group(request_id="rt_0", prompt_len=100, output_len=50, is_be=False)

    bm = _make_block_manager(free_blocks=0, watermark_blocks=5)
    bm.get_num_blocks_touched_by_append_slots.return_value = 10
    bm.get_block_table.return_value = [0, 1]

    sched = _make_scheduler(running=[be, rt], block_manager=bm)
    bm.swap_out.side_effect = lambda g: [(0, 100)]

    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        num_swapped = Scheduler._prepare_decode_saturation_blocks(sched)

    assert num_swapped == 1
    assert be not in sched.running
    sched.block_manager.swap_out.assert_called_once()


# ======================================================================
# _can_restore_decode_saturation_be
# ======================================================================

def test_can_restore_returns_true_when_enough_blocks():
    be = _make_seq_group(request_id="be_0", prompt_len=50, output_len=30, is_be=True,
                         status=SequenceStatus.SWAPPED)
    bm = _make_block_manager(free_blocks=50)
    bm.can_allocate.return_value = AllocStatus.OK
    sched = _make_scheduler(block_manager=bm)

    assert Scheduler._can_restore_decode_saturation_be(sched, be) is True
    # Status should have been restored after the check
    for seq in be.get_seqs.return_value:
        assert seq.status == SequenceStatus.SWAPPED


def test_can_restore_returns_false_when_not_enough_blocks():
    be = _make_seq_group(request_id="be_0", prompt_len=50, output_len=30, is_be=True,
                         status=SequenceStatus.SWAPPED)
    bm = _make_block_manager(free_blocks=1)
    bm.can_swap_in.return_value = AllocStatus.NEVER
    bm.can_allocate.return_value = AllocStatus.NEVER
    sched = _make_scheduler(block_manager=bm)

    assert Scheduler._can_restore_decode_saturation_be(sched, be) is False
    for seq in be.get_seqs.return_value:
        assert seq.status == SequenceStatus.SWAPPED


# ======================================================================
# _restore_decode_saturation_be
# ======================================================================

def test_restore_re_admits_be_within_budget():
    be = _make_seq_group(request_id="be_0", prompt_len=50, output_len=30, is_be=True,
                         status=SequenceStatus.SWAPPED)
    bm = _make_block_manager(free_blocks=50)
    bm.can_allocate.return_value = AllocStatus.OK
    ds = _make_decode_saturation(seq_group_blocks_fn=lambda _: 3)
    sched = _make_scheduler(
        running=[],
        swapped=deque([be]),
        block_manager=bm,
        decode_saturation=ds,
    )

    decision = MagicMock(be_block_budget=50)
    new_be_blocks = Scheduler._restore_decode_saturation_be(sched, decision, projected_be_blocks=0)

    assert new_be_blocks == 3  # projected_be_blocks increased by group_blocks
    assert be in sched.running
    assert len(sched.swapped) == 0
    bm.swap_in.assert_called_once_with(be)


def test_restore_does_not_exceed_sequence_capacity():
    be = _make_seq_group(
        request_id="be_swapped",
        prompt_len=50,
        output_len=30,
        is_be=True,
        status=SequenceStatus.SWAPPED,
    )
    rt = _make_seq_group(
        request_id="rt_running", prompt_len=100, output_len=50, is_be=False)
    sched = _make_scheduler(running=[rt], swapped=deque([be]))
    sched.scheduler_config.max_num_seqs = 1

    projected = Scheduler._restore_decode_saturation_be(
        sched, MagicMock(be_block_budget=50), projected_be_blocks=0)

    assert projected == 0
    assert list(sched.running) == [rt]
    assert list(sched.swapped) == [be]
    sched.block_manager.swap_in.assert_not_called()


def test_restore_counts_temporarily_scheduled_running_groups():
    be = _make_seq_group(
        request_id="be_swapped_reserved",
        prompt_len=50,
        output_len=30,
        is_be=True,
        status=SequenceStatus.SWAPPED,
    )
    sched = _make_scheduler(swapped=deque([be]))
    sched.scheduler_config.max_num_seqs = 1

    projected = Scheduler._restore_decode_saturation_be(
        sched,
        MagicMock(be_block_budget=50),
        projected_be_blocks=0,
        reserved_running_seqs=1,
    )

    assert projected == 0
    assert not sched.running
    assert list(sched.swapped) == [be]
    sched.block_manager.swap_in.assert_not_called()


def test_restore_stops_when_be_block_budget_exhausted():
    be = _make_seq_group(request_id="be_0", prompt_len=50, output_len=30, is_be=True)
    for seq in be.get_seqs.return_value:
        seq.status = SequenceStatus.SWAPPED

    bm = _make_block_manager(free_blocks=50)
    bm.can_allocate.return_value = True
    ds = _make_decode_saturation(seq_group_blocks_fn=lambda _: 10)
    sched = _make_scheduler(
        running=[],
        swapped=deque([be]),
        block_manager=bm,
        decode_saturation=ds,
    )

    # Budget already consumed: 45 projected + 10 for this BE = 55 > 50 budget
    decision = MagicMock(be_block_budget=50)
    new_be_blocks = Scheduler._restore_decode_saturation_be(sched, decision, projected_be_blocks=45)

    assert new_be_blocks == 45  # unchanged, no restoration
    assert be not in sched.running
    assert be in sched.swapped


def test_restore_stops_at_non_be_group():
    be = _make_seq_group(request_id="be_0", prompt_len=50, output_len=30, is_be=True)
    rt_swapped = _make_seq_group(request_id="rt_swapped", prompt_len=20, output_len=10, is_be=False)

    for seq in be.get_seqs.return_value:
        seq.status = SequenceStatus.SWAPPED
    for seq in rt_swapped.get_seqs.return_value:
        seq.status = SequenceStatus.SWAPPED

    bm = _make_block_manager(free_blocks=50)
    bm.can_allocate.return_value = True
    sched = _make_scheduler(
        running=[],
        swapped=deque([rt_swapped, be]),
        block_manager=bm,
    )

    decision = MagicMock(be_block_budget=50)
    new_be_blocks = Scheduler._restore_decode_saturation_be(sched, decision, projected_be_blocks=0)

    assert new_be_blocks == 0  # nothing restored
    assert rt_swapped in sched.swapped  # still there
    assert be in sched.swapped  # blocked by RT at head


# ======================================================================
# Integration: full _schedule_decode_saturation flow
# ======================================================================

def test_schedule_decode_saturation_reports_preempted_count():
    """preempted count in SchedulerOutputs matches num_swapped_out."""
    rt = _make_seq_group(request_id="rt_0", prompt_len=100, output_len=50, is_be=False)
    be = _make_seq_group(request_id="be_0", prompt_len=50, output_len=30, is_be=True)

    bm = _make_block_manager(free_blocks=0, watermark_blocks=5)
    bm.get_num_blocks_touched_by_append_slots.return_value = 10
    bm.get_block_table.return_value = [0, 1, 2]

    ds = _make_decode_saturation()
    ds.provider.get_remaining_tokens.return_value = 10
    ds.provider.get_future_arrivals.return_value = ()
    ds.make_decision.return_value = MagicMock(
        be_block_budget=10,
        max_be_prefill_tokens=10 * BLOCK_SIZE,
        target_interval_ms=100.0,
    )

    sched = _make_scheduler(
        running=[rt, be],
        block_manager=bm,
        decode_saturation=ds,
    )

    def free_side_effect(seq):
        bm.get_num_free_gpu_blocks.return_value = bm.get_num_free_gpu_blocks.return_value + 3

    bm.swap_out.side_effect = lambda g: (free_side_effect(g.get_seqs()[0]) or [(0, 100)])

    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        result = Scheduler._schedule_decode_saturation(sched)

    assert result.preempted == 1  # one BE was evicted
    assert result.blocks_to_swap_out == [(0, 100)]
    assert not result.is_empty()
    assert be not in sched.running


def test_restore_is_skipped_when_eviction_happened_this_step():
    """When num_swapped_out > 0, _restore_decode_saturation_be is NOT called."""
    rt = _make_seq_group(request_id="rt_0", prompt_len=100, output_len=50, is_be=False)
    be = _make_seq_group(request_id="be_0", prompt_len=50, output_len=30, is_be=True)
    be_swapped = _make_seq_group(request_id="be_prev", prompt_len=20, output_len=10, is_be=True)
    for seq in be_swapped.get_seqs.return_value:
        seq.status = SequenceStatus.SWAPPED

    bm = _make_block_manager(free_blocks=0, watermark_blocks=5)
    bm.get_num_blocks_touched_by_append_slots.return_value = 10
    bm.get_block_table.return_value = [0, 1, 2]

    ds = _make_decode_saturation()
    ds.provider.get_remaining_tokens.return_value = 10
    ds.provider.get_future_arrivals.return_value = ()
    ds.make_decision.return_value = MagicMock(
        be_block_budget=50,
        max_be_prefill_tokens=50 * BLOCK_SIZE,
        target_interval_ms=100.0,
    )

    sched = _make_scheduler(
        running=[rt, be],
        swapped=deque([be_swapped]),
        block_manager=bm,
        decode_saturation=ds,
    )

    def free_side_effect(seq):
        bm.get_num_free_gpu_blocks.return_value = bm.get_num_free_gpu_blocks.return_value + 3

    bm.swap_out.side_effect = lambda g: (free_side_effect(g.get_seqs()[0]) or [(0, 100)])

    # Track whether restore was called
    original_restore = Scheduler._restore_decode_saturation_be
    restore_calls = []
    def restore_spy(s, d, pbb, reserved_running_seqs=0, blocks_to_swap_in=None):
        restore_calls.append(1)
        return original_restore(
            s, d, pbb, reserved_running_seqs=reserved_running_seqs,
            blocks_to_swap_in=blocks_to_swap_in)

    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        with patch.object(Scheduler, "_restore_decode_saturation_be", restore_spy):
            result = Scheduler._schedule_decode_saturation(sched)

    assert result.preempted == 1
    assert len(restore_calls) == 0  # skipped because eviction happened


def test_restore_is_called_when_no_eviction_this_step():
    """When num_swapped_out == 0, _restore_decode_saturation_be IS called."""
    rt = _make_seq_group(request_id="rt_0", prompt_len=100, output_len=50, is_be=False)
    be_swapped = _make_seq_group(request_id="be_prev", prompt_len=20, output_len=10, is_be=True)
    for seq in be_swapped.get_seqs.return_value:
        seq.status = SequenceStatus.SWAPPED

    bm = _make_block_manager(free_blocks=100, watermark_blocks=5)
    bm.get_num_blocks_touched_by_append_slots.return_value = 1
    bm.can_allocate.return_value = True

    ds = _make_decode_saturation()
    ds.provider.get_remaining_tokens.return_value = 10
    ds.provider.get_future_arrivals.return_value = ()
    ds.make_decision.return_value = MagicMock(
        be_block_budget=50,
        max_be_prefill_tokens=50 * BLOCK_SIZE,
        target_interval_ms=100.0,
    )

    sched = _make_scheduler(
        running=[rt],
        swapped=deque([be_swapped]),
        block_manager=bm,
        decode_saturation=ds,
    )

    restore_calls = []
    original_restore = Scheduler._restore_decode_saturation_be
    def restore_spy(s, d, pbb, reserved_running_seqs=0, blocks_to_swap_in=None):
        restore_calls.append(1)
        return original_restore(
            s, d, pbb, reserved_running_seqs=reserved_running_seqs,
            blocks_to_swap_in=blocks_to_swap_in)

    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        with patch.object(Scheduler, "_restore_decode_saturation_be", restore_spy):
            result = Scheduler._schedule_decode_saturation(sched)

    assert result.preempted == 0
    assert len(restore_calls) == 1


# ======================================================================
# Edge-case: re-entrant restore after prior-step eviction
# ======================================================================

def test_be_restored_on_next_step_after_eviction_pressure_relieved():
    """End-to-end: evict a BE (step 1), then restore it (step 2) when RT
    load drops and budget is available."""

    # --- Setup ---
    rt = _make_seq_group(request_id="rt_0", prompt_len=100, output_len=50, is_be=False)
    be = _make_seq_group(request_id="be_lucky", prompt_len=20, output_len=10, is_be=True)

    bm = _make_block_manager(free_blocks=2, watermark_blocks=5)
    bm.get_num_blocks_touched_by_append_slots.return_value = 8
    bm.get_num_required_blocks.return_value = 4
    bm.get_block_table.return_value = [0, 1, 2]

    ds = _make_decode_saturation()
    ds.provider.get_remaining_tokens.return_value = 10
    ds.provider.get_future_arrivals.return_value = ()
    ds.make_decision.return_value = MagicMock(
        be_block_budget=50,
        max_be_prefill_tokens=50 * BLOCK_SIZE,
        target_interval_ms=100.0,
    )

    sched = _make_scheduler(
        running=[rt, be],
        block_manager=bm,
        decode_saturation=ds,
    )

    # --- Step 1: tight blocks → evict BE ---
    def free_side_effect_step1(seq):
        bm.get_num_free_gpu_blocks.return_value = bm.get_num_free_gpu_blocks.return_value + 3

    bm.swap_out.side_effect = lambda g: (free_side_effect_step1(g.get_seqs()[0]) or [(0, 100)])

    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        result1 = Scheduler._schedule_decode_saturation(sched)

    assert result1.preempted == 1
    assert be not in sched.running
    assert be in sched.swapped

    # --- Step 2: blocks are plentiful, no RT in waiting → restore BE ---
    bm.get_num_free_gpu_blocks.return_value = 80
    bm.get_num_blocks_touched_by_append_slots.return_value = 1
    sched.free_seq.reset_mock()

    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        result2 = Scheduler._schedule_decode_saturation(sched)

    assert result2.preempted == 0  # no eviction this step
    assert be in sched.running  # restored!


def test_cpu_swap_exhaustion_keeps_original_gpu_kv():
    be = _make_seq_group(request_id="be_full_cpu", prompt_len=32,
                        output_len=8, is_be=True)
    bm = _make_block_manager(free_blocks=0)
    bm.can_swap_out.return_value = False
    scheduler = _make_scheduler(running=[be], block_manager=bm)
    mappings = []
    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        assert scheduler._prepare_decode_saturation_blocks(mappings) == 0
    assert be in scheduler.running
    assert mappings == []
    bm.swap_out.assert_not_called()
    scheduler.free_seq.assert_not_called()


def test_early_reservation_reclaims_be_before_rt_arrives():
    be = _make_seq_group(request_id="be_early", prompt_len=32,
                        output_len=8, is_be=True)
    bm = _make_block_manager(free_blocks=3, watermark_blocks=0)
    bm.get_block_table.return_value = list(range(6))
    def swap(group):
        bm.get_num_free_gpu_blocks.return_value += 6
        return [(i, 100+i) for i in range(6)]
    bm.swap_out.side_effect = swap
    scheduler = _make_scheduler(running=[be], block_manager=bm)
    scheduler.reserve_decode_request("rt:p1", 32)
    assert scheduler._decode_reserved_blocks() == 3
    mappings = []
    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        assert scheduler._prepare_decode_saturation_blocks(mappings) == 1
    assert mappings == [(i, 100+i) for i in range(6)]
    assert not scheduler.waiting_rt_decode_admission
    assert be in scheduler.swapped
    scheduler.free_seq.assert_not_called()


def test_restore_emits_cpu_to_gpu_mapping_and_preserves_claims():
    be = _make_seq_group(request_id="be_restore", prompt_len=32,
                        output_len=8, is_be=True, status=SequenceStatus.SWAPPED)
    bm = _make_block_manager(free_blocks=5, watermark_blocks=0)
    scheduler = _make_scheduler(swapped=deque([be]), block_manager=bm)
    scheduler.reserve_decode_request("rt:p1", 32)
    mappings = []
    with patch("vllm.core.scheduler.is_decode_phase", return_value=True):
        assert scheduler._restore_decode_saturation_be(
            MagicMock(be_block_budget=50), 0, blocks_to_swap_in=mappings) == 0
        assert be in scheduler.swapped
        scheduler.release_decode_reservation("rt:p1")
        assert scheduler._restore_decode_saturation_be(
            MagicMock(be_block_budget=50), 0, blocks_to_swap_in=mappings) == 3
    assert mappings == [(100, 10)]
    bm.allocate.assert_not_called()
    bm.swap_in.assert_called_once_with(be)


def test_direct_rt_admission_consumes_reservation_exactly_once():
    scheduler = _make_scheduler()
    scheduler.reserve_decode_request("rt:p1", 32)
    seq = MagicMock()
    seq.get_prompt_len.return_value = 32
    seq.get_last_token_id.return_value = 1
    group = MagicMock()
    group.get_seqs.return_value = [seq]
    group.sampling_params.extra_args = {
        "saturo_direct_decode": True,
        "saturo_decode_reservation_id": "rt:p1"}
    with patch("vllm.core.scheduler.is_rt_request", return_value=True), \
         patch("vllm.core.scheduler.is_be_request", return_value=False):
        scheduler.add_seq_group(group)
    assert scheduler._decode_reserved_blocks() == 0
    assert group in scheduler.waiting_rt_decode_admission
    scheduler.release_decode_reservation("rt:p1")
    assert scheduler._decode_reserved_slots() == 0
