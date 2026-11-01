from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import (Scheduler, SchedulerRunningOutputs,
                                 SchedulerSwappedInOutputs)
from vllm.sequence import Logprob
from tests.core.utils import create_dummy_prompt
from saturo.saturation.types import DecodeSaturationDecision


class Latency:
    def __init__(self, costs):
        self.costs = costs

    def predict_forward_ms(self, prefill, decode):
        return self.costs[len(decode)] + 3 * sum(q for q, _ in prefill)


def native_scheduler(target=10., feasible=True, costs=(5., 5., 200., 300.)):
    config = SchedulerConfig('generate', max_num_batched_tokens=64,
        max_num_seqs=4, max_model_len=64, enable_chunked_prefill=True,
        enable_decode_saturation=False)
    cache = CacheConfig(16, .9, 1, 'auto')
    cache.num_gpu_blocks = cache.num_cpu_blocks = 32
    scheduler = Scheduler(config, cache, None)
    scheduler.time_predictor = Latency(costs)
    scheduler.decode_saturation.enabled = True
    result = DecodeSaturationDecision(True, target, target, 5., 2, 1, 1,
        30 if feasible else 0, 0, feasible,
        target_status='async_ready' if feasible else 'pending')
    scheduler.decode_saturation.make_decision = lambda *args: result
    return scheduler


def add_running(scheduler, identifier, rt):
    seq, group = create_dummy_prompt(str(identifier), 16, block_size=16)
    group.is_realtime = rt
    scheduler._allocate_and_set_running(group)
    seq.data.update_num_computed_tokens(16)
    seq.append_token_id(1, {1: Logprob(logprob=0.)})
    scheduler.running.append(group)
    return group


def ids(output):
    return [scheduled.seq_group.request_id for scheduled in output.scheduled_seq_groups]


def test_actual_native_whole_caller_rejects_existing_slow_be_and_keeps_rt():
    scheduler = native_scheduler()
    be = add_running(scheduler, 1, False)
    rt = add_running(scheduler, 0, True)
    output = scheduler._schedule_decode_saturation()
    assert ids(output) == ['0']
    assert be in scheduler.running and rt in scheduler.running
    assert scheduler.time_predictor.predict_forward_ms(
        *scheduler._predictor_reqs_from_scheduled(output.scheduled_seq_groups)) <= 10


def test_resident_be_is_selected_incrementally_against_final_rt_batch():
    scheduler = native_scheduler(costs=(0., 5., 8., 12.))
    add_running(scheduler, 0, True)
    add_running(scheduler, 1, False)
    deferred = add_running(scheduler, 2, False)
    output = scheduler._schedule_decode_saturation()
    assert ids(output) == ['0', '1'] and deferred in scheduler.running


def test_pending_async_target_disables_resident_be_even_when_model_is_fast():
    scheduler = native_scheduler(feasible=False, costs=(0., .1, .2, .3))
    add_running(scheduler, 0, True)
    be = add_running(scheduler, 1, False)
    assert ids(scheduler._schedule_decode_saturation()) == ['0']
    assert be in scheduler.running


def test_rt_batch_is_never_removed_by_be_latency_gate():
    scheduler = native_scheduler()
    add_running(scheduler, 0, True)
    add_running(scheduler, 1, True)
    assert ids(scheduler._schedule_decode_saturation()) == ['0', '1']


def test_new_rt_arrival_rechecks_previously_executable_be():
    scheduler = native_scheduler()
    be = add_running(scheduler, 1, False)
    assert ids(scheduler._schedule_decode_saturation()) == ['1']
    add_running(scheduler, 0, True)
    assert ids(scheduler._schedule_decode_saturation()) == ['0']
    assert be in scheduler.running


def test_d_fixed_prefill_cap_does_not_bypass_tbt_and_p_fixed_mode_is_retained():
    scheduler = native_scheduler()
    scheduler.scheduler_config.p_pool_be_chunk_size = 32
    _, candidate = create_dummy_prompt('1', 64, block_size=16)
    running = SchedulerRunningOutputs.create_empty()
    swapped = SchedulerSwappedInOutputs.create_empty()
    decision = DecodeSaturationDecision(True, 10., 10., 0., 0, 0, 0, 32, 64)
    assert scheduler._find_be_prefill_chunk_by_latency(
        decision, 64, running, swapped, [], candidate) == 1
    scheduler.decode_saturation.enabled = False
    assert scheduler._find_be_prefill_chunk_by_latency(
        decision, 64, running, swapped, [], candidate) == 32
