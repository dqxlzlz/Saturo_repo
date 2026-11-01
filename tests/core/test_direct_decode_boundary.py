"""Native CPU reproducer for a context-preserving direct-decode handoff."""
from collections import deque
from types import SimpleNamespace
import time

import pytest

from saturo.request_metadata import is_be_request, is_rt_request
from saturo.runtime.workload import SaturoRequest, RT
from tests.core.utils import create_dummy_prompt
from vllm import SamplingParams
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.sequence import Logprob, SequenceData, SequenceGroup, SequenceStatus






def direct_group(prompt_len, remaining, *, is_rt=True):
    context = list(range(prompt_len))
    seq, group = create_dummy_prompt('952', prompt_len, block_size=16,
                                     prompt_tokens=context,
                                     max_tokens=remaining + 1)
    group.sampling_params = SamplingParams(
        max_tokens=remaining + 1, ignore_eos=True, is_realtime=is_rt,
        detokenize=False, extra_args={'saturo_direct_decode': True})
    group.is_realtime = is_rt
    group.prefill_ddl = time.time() + 1 if is_rt else float('inf')
    return context, seq, group


def scheduler():
    return SimpleNamespace(
        decode_saturation=SimpleNamespace(enabled=True,
            observe_rt_arrival=lambda *args: None),
        waiting=deque(), waiting_rt_decode_admission=deque(),
        waiting_be_decode_admission=deque(),
        release_decode_reservation=lambda *args: None)


def run_to_native_stop(seq, params, limit):
    checker = StopChecker(limit, lambda _: None)
    while not seq.is_finished():
        seq.append_token_id(17, {17: Logprob(logprob=0.)})
        checker.maybe_stop_sequence(seq, 0, params)
        assert seq.get_len() <= limit
    return seq.get_output_len() - 1  # Existing worker hides the input seed.




@pytest.mark.parametrize('observed', [0, 1, 7, 89])
@pytest.mark.parametrize('is_rt', [False, True])
def test_exact_context_boundary_completes_requested_tokens(observed, is_rt):
    req = SaturoRequest('952', RT, 934, 90, 0, observed_tokens=observed)
    context, seq, group = direct_group(req.decode_prompt_len,
                                       req.remaining_tokens, is_rt=is_rt)
    state = scheduler()
    Scheduler.add_seq_group(state, group)
    assert seq.data.get_token_ids() == context  # No input dropped/duplicated.
    assert seq.get_len() == req.decode_prompt_len
    assert seq.get_prompt_len() + seq.get_output_len() == len(context)
    assert seq.data.get_num_computed_tokens() == len(context) - 1
    assert not seq.is_prefill() and seq.get_num_new_tokens() == 1
    queue = (state.waiting_rt_decode_admission if is_rt
             else state.waiting_be_decode_admission)
    assert list(queue) == [group]
    delivered = run_to_native_stop(seq, group.sampling_params, 1024)
    assert delivered == req.remaining_tokens
    assert observed + delivered == 90
    assert seq.get_len() == 1024


@pytest.mark.parametrize('context,remaining,limit', [
    (1, 1, 2), (1, 8, 9), (2, 6, 8), (15, 1, 16),
    (16, 16, 32), (1023, 1, 1024),
])
def test_small_context_and_block_boundaries_keep_length(context, remaining, limit):
    ids, seq, group = direct_group(context, remaining)
    Scheduler.add_seq_group(scheduler(), group)
    assert seq.data.get_token_ids() == ids
    assert seq.get_len() == context
    assert seq.get_num_new_tokens() == 1
    assert run_to_native_stop(seq, group.sampling_params, limit) == remaining
    assert seq.get_len() == limit


def test_native_context_limit_still_stops_overlong_work():
    _, seq, group = direct_group(935, 90)
    Scheduler.add_seq_group(scheduler(), group)
    assert run_to_native_stop(seq, group.sampling_params, 1024) == 89
    assert seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED
