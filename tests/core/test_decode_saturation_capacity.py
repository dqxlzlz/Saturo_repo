"""Resident KV slots must remain reserved even when a group skips a step."""
from collections import deque
from types import SimpleNamespace

import pytest

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import (PartialPrefillMetadata, Scheduler,
                                 SchedulerRunningOutputs,
                                 SchedulerSwappedInOutputs, SchedulingBudget)

from .utils import (append_new_token, create_dummy_prompt, create_seq_group,
                    get_sequence_groups, schedule_and_update_computed_tokens)


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setattr(
        "saturo.predictors.qrf.load_required_qrf",
        lambda *args, **kwargs: SimpleNamespace(
            predict_remaining=lambda **kwargs: 8))
    monkeypatch.setenv("VLLM_DECODE_SATURATION", "1")
    monkeypatch.setenv("VLLM_DECODE_SATURATION_TBT_MS", "100")
    monkeypatch.setenv("VLLM_DECODE_SATURATION_BE_BLOCKS", "128")
    config = SchedulerConfig(
        "generate", max_num_batched_tokens=8, max_num_seqs=2,
        max_model_len=32, enable_chunked_prefill=True,
        max_num_partial_prefills=2, max_long_partial_prefills=2)
    cache = CacheConfig(4, 1.0, 1, "auto")
    cache.num_gpu_blocks = cache.num_cpu_blocks = 128
    return Scheduler(config, cache, None)


def populate(scheduler):
    groups = []
    for i in range(2):
        _, group = create_dummy_prompt(str(i), 4, block_size=4)
        group.prefill_ddl = float("inf")
        scheduler.add_seq_group(group)
        groups.append(group)
    _, output = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(output) == groups
    append_new_token(output, 1)
    _, waiting = create_dummy_prompt("2", 4, block_size=4)
    waiting.prefill_ddl = float("inf")
    scheduler.add_seq_group(waiting)
    return groups, waiting


@pytest.mark.parametrize("deferred_count", [1, 2])
def test_deferred_decode_keeps_resident_slot(scheduler, monkeypatch,
                                           deferred_count):
    groups, waiting = populate(scheduler)
    original = scheduler._can_append_slots
    monkeypatch.setattr(scheduler, "_can_append_slots", lambda group, *args:
                        False if group in groups[:deferred_count]
                        else original(group, *args))
    _, output = schedule_and_update_computed_tokens(scheduler)
    assert waiting in scheduler.waiting
    assert waiting not in get_sequence_groups(output)
    assert set(scheduler.running) == set(groups)
    # Once a resident finishes, a queued BE request must still make progress.
    scheduler.abort_seq_group(groups[0].request_id)
    monkeypatch.setattr(scheduler, "_can_append_slots", original)
    append_new_token(output, 1)
    _, output = schedule_and_update_computed_tokens(scheduler)
    assert waiting in get_sequence_groups(output)


def test_restored_be_keeps_slot_before_its_next_decode_step(scheduler):
    groups, waiting = populate(scheduler)
    restored = groups[1]
    scheduler._swap_out(restored, [])
    scheduler.running.remove(restored)
    scheduler.swapped.append(restored)
    _, output = schedule_and_update_computed_tokens(scheduler)
    assert output.blocks_to_swap_in
    assert restored in scheduler.running
    assert waiting in scheduler.waiting
    assert waiting not in get_sequence_groups(output)
    assert len(scheduler.running) == 2
    # The restored group is eligible on the next iteration, not stranded.
    append_new_token(output, 1)
    _, output = schedule_and_update_computed_tokens(scheduler)
    assert restored in get_sequence_groups(output)


@pytest.mark.parametrize("pool", ["rt", "be"])
@pytest.mark.parametrize("resident_seqs,candidate_seqs,admitted", [
    (1, 1, True), (2, 1, False), (1, 2, False), (0, 2, True)])
def test_decode_admission_counts_sequences_not_groups(
        scheduler, monkeypatch, pool, resident_seqs, candidate_seqs, admitted):
    if resident_seqs:
        resident = create_seq_group(seq_prompt_len=4,
                                    seq_output_lens=(1,) * resident_seqs)
        scheduler.running = deque([resident])
    candidate = create_seq_group(seq_prompt_len=4, request_id="2",
                                 seq_output_lens=(1,) * candidate_seqs)
    queue = (scheduler.waiting_rt_decode_admission if pool == "rt"
             else scheduler.waiting_be_decode_admission)
    queue.append(candidate)
    monkeypatch.setattr(scheduler.block_manager, "can_allocate",
                        lambda group: AllocStatus.OK)
    monkeypatch.setattr(scheduler, "_allocate_and_set_running",
                        lambda group: None)
    monkeypatch.setattr(scheduler, "_can_admit_be_decode_by_latency",
                        lambda *args: True)
    monkeypatch.setattr(scheduler.decode_saturation, "can_admit_be",
                        lambda *args: True)
    monkeypatch.setattr(scheduler.decode_saturation, "seq_group_blocks",
                        lambda *args: 1)
    if pool == "rt":
        scheduler._admit_rt_decode_admission_queue()
    else:
        scheduler._admit_be_decode_admission_queue(
            SimpleNamespace(current_be_blocks=0))
        queue = scheduler.waiting_be_decode_admission
    assert (candidate in scheduler.running) is admitted
    assert (candidate in queue) is not admitted


@pytest.mark.parametrize("reserve", [False, True])
def test_new_be_respects_last_free_slot_and_early_rt_claim(scheduler, reserve):
    groups, waiting = populate(scheduler)
    scheduler.abort_seq_group(groups[0].request_id)
    if reserve:
        scheduler.reserve_decode_request("early-rt", 4)
    _, output = schedule_and_update_computed_tokens(scheduler)
    assert (waiting in get_sequence_groups(output)) is not reserve
    assert sum(g.get_max_num_running_seqs() for g in scheduler.running) == (
        1 if reserve else 2)


def test_partial_be_continues_at_full_resident_capacity(scheduler):
    groups = []
    for i in range(2):
        _, group = create_dummy_prompt(str(i), 12, block_size=4)
        group.prefill_ddl = float("inf")
        scheduler._allocate_and_set_running(group)
        scheduler.running.append(group)
        groups.append(group)
    _, output = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(output) == [groups[0]]
    assert 0 < output.num_batched_tokens < 12
    assert len(scheduler.running) == 2


def test_scheduled_resident_is_not_counted_twice(scheduler):
    _, resident = create_dummy_prompt("0", 4, block_size=4)
    scheduler._allocate_and_set_running(resident)
    scheduler.running.append(resident)
    _, waiting = create_dummy_prompt("1", 4, block_size=4)
    waiting.prefill_ddl = float("inf")
    scheduler.add_seq_group(waiting)
    budget = SchedulingBudget(token_budget=8, max_num_seqs=2)
    budget.add_num_seqs(resident.request_id, 1)
    output = scheduler._schedule_be_prefill_chunks_for_saturation(
        SimpleNamespace(target_interval_ms=100), 4, budget, set(),
        SchedulerRunningOutputs.create_empty(),
        SchedulerSwappedInOutputs.create_empty(),
        PartialPrefillMetadata.from_queues(
            running=scheduler.running, waiting=scheduler.waiting,
            scheduler_config=scheduler.scheduler_config), deque())
    assert [s.seq_group for s in output.seq_groups] == [waiting]


@pytest.mark.parametrize("decode_enabled", [False, True])
def test_resident_limit_does_not_change_unreserved_p_pool_policy(
        scheduler, decode_enabled):
    groups, waiting = populate(scheduler)
    scheduler.decode_saturation.enabled = decode_enabled
    # P-pool scheduling can defer resident decodes while filling this step's token
    # budget. D additionally needs a strict resident cap for early RT claims.
    scheduler.running.clear()
    budget = SchedulingBudget(token_budget=8, max_num_seqs=2)
    output = scheduler._schedule_be_prefill_chunks_for_saturation(
        SimpleNamespace(target_interval_ms=100), 4, budget, set(),
        SchedulerRunningOutputs.create_empty(),
        SchedulerSwappedInOutputs.create_empty(),
        PartialPrefillMetadata.from_queues(
            running=deque(groups), waiting=scheduler.waiting,
            scheduler_config=scheduler.scheduler_config), deque(),
        deferred_running=groups)
    assert ([s.seq_group for s in output.seq_groups] == [waiting]) is (
        not decode_enabled)
    assert budget.num_curr_seqs <= 2
