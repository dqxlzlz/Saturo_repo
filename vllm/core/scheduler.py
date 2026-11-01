# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Deque, Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from saturo.saturation import DecodeSaturationPolicy
from saturo.saturation.reservations import DecodeReservations
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from saturo.request_metadata import (
    get_prefill_deadline,
    is_be_request,
    is_decode_phase,
    is_rt_request,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupBase, SequenceGroupMetadata,
                           SequenceGroupMetadataDelta, SequenceStage, Logprob,
                           SequenceStatus)
from vllm.utils import Device, PyObjectCache

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """

    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """

    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    # Number of cached tokens in the batch.
    _num_cached_tokens: int = 0
    # Number of actual non-cached tokens in the batch.
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        # We allow num_new_tokens to be 0 when the entire sequence has
        # been cached.
        assert num_new_tokens >= 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self,
                               req_id: str,
                               num_batched_tokens: int,
                               num_cached_tokens: int = 0):
        if req_id in self._request_ids_num_batched_tokens:
            return
        assert num_cached_tokens >= 0
        assert num_batched_tokens >= 0

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens
        self._num_cached_tokens += num_cached_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs

    @property
    def num_cached_tokens(self):
        return self._num_cached_tokens


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass(frozen=True)
class _PrefillSaturationChunkTarget:
    target_interval_ms: float


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""

    # Scheduled sequence groups.
    scheduled_seq_groups: GenericSequence[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        assert 0 <= self.num_prefill_groups <= len(self.scheduled_seq_groups)

        def key_fn(group: ScheduledSequenceGroup):
            key = (group.seq_group.lora_int_id, group.seq_group.request_id)
            if 0 < self.num_prefill_groups < len(self.scheduled_seq_groups):
                # Sort sequence groups so that all prefills come before all
                # decodes as required by chunked prefill.
                return (not group.seq_group.is_prefill(), *key)
            return key

        self.scheduled_seq_groups = sorted(self.scheduled_seq_groups,
                                           key=key_fn)

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {
            g.seq_group.prompt_adapter_request
            for g in self.scheduled_seq_groups
            if g.seq_group.prompt_adapter_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """

    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    # Optimization for fast-access to seq_group lists
    decode_seq_groups_list: List[SequenceGroup]
    prefill_seq_groups_list: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            decode_seq_groups_list=[],
            prefill_seq_groups_list=[],
        )


@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """

    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible sequence groups.
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            infeasible_seq_groups=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """

    # Selected sequences for prefill.
    seq_groups: List[ScheduledSequenceGroup]
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )


def seq_group_metadata_builder():
    return SequenceGroupMetadata(request_id="",
                                 is_prompt=False,
                                 seq_data={},
                                 sampling_params=None,
                                 block_tables={})


def scheduler_running_outputs_builder():
    return SchedulerRunningOutputs(decode_seq_groups=[],
                                   prefill_seq_groups=[],
                                   preempted=[],
                                   swapped_out=[],
                                   blocks_to_swap_out=[],
                                   blocks_to_copy=[],
                                   num_lookahead_slots=0,
                                   prefill_seq_groups_list=[],
                                   decode_seq_groups_list=[])


def scheduled_seq_group_builder():
    return ScheduledSequenceGroup(SequenceGroup.__new__(SequenceGroup),
                                  token_chunk_size=0)
    # return ScheduledSequenceGroup(seq_group=None, token_chunk_size=0)


@dataclass
class PartialPrefillMetadata:
    """Holds information about the partial prefills that are currently running
    during a single iteration of the Scheduler.
    When chunked prefill is enabled, we allow a certain number of seqs to be
    partially prefilled during each iteration. Having multiple partial prefills
    in flight allows us to minimize TTFT and avoid decode starvation in cases
    where a single sequence group with a very large prompt blocks the queue for
    too many iterations.
    The number of long prefill requests is limited so that smaller
    requests may jump the queue in front of them and get to the decode
    phase faster.
    """

    # A minimum bound on the total number of prefills to be scheduled during
    # this iteration
    schedulable_prefills: int

    # The number of long prefill requests currently running
    long_prefills: int

    scheduler_config: SchedulerConfig

    def can_schedule(self, seq_group: SequenceGroup) -> bool:
        """When concurrent partial prefills are enabled,
        we limit the number of long requests and only accept
        shorter requests from the queue while running them
        concurrently"""
        return not (seq_group.first_seq.get_num_new_tokens()
                    > self.scheduler_config.long_prefill_token_threshold
                    and self.long_prefills
                    >= self.scheduler_config.max_long_partial_prefills
                    and self.scheduler_config.max_num_partial_prefills > 1)

    def maybe_increment_partial_prefills(self,
                                         seq_group: SequenceGroup) -> None:
        # When a new prefill is scheduled, we need to know if it is a
        # long request
        if (seq_group.first_seq.get_num_new_tokens()
                > self.scheduler_config.long_prefill_token_threshold):
            self.long_prefills += 1

    @classmethod
    def from_queues(
        cls,
        running: Deque[SequenceGroup],
        waiting: Deque[SequenceGroup],
        scheduler_config: SchedulerConfig,
    ) -> "PartialPrefillMetadata":
        """Create a PartialPrefillMetadata object from the current state of
        the scheduler's queues.
        This accounts for the currently running prefill requests, and peeks into
        the waiting queue to see if there are more prefills to potentially be
        scheduled during this iteration."""
        prefills = 0
        long_prefills = 0

        waiting_long_prefills = 0

        for sg in running:
            if sg.first_seq.data.stage == SequenceStage.PREFILL:
                prefills += 1
                if (sg.first_seq.get_num_new_tokens()
                        > scheduler_config.long_prefill_token_threshold):
                    long_prefills += 1

        for sg in waiting:
            # Don't bother looping through the rest of the queue if we know
            # there are already at
            # least max_partial_prefills requests to fill
            if prefills >= scheduler_config.max_num_partial_prefills:
                break

            # Don't count long requests from the waiting queue if we aren't
            # going to schedule them anyway
            if (sg.first_seq.get_num_new_tokens()
                    > scheduler_config.long_prefill_token_threshold):
                if (long_prefills + waiting_long_prefills
                        >= scheduler_config.max_long_partial_prefills):
                    continue
                waiting_long_prefills += 1
            prefills += 1

        # NB: long_prefills and waiting_long_prefills are tracked separately.
        # We don't account for the waiting requests here because we need to use
        # this metadata to track how many have actually been scheduled.
        return PartialPrefillMetadata(
            schedulable_prefills=min(
                prefills, scheduler_config.max_num_partial_prefills),
            long_prefills=long_prefills,
            scheduler_config=scheduler_config,
        )


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "selfattn"
        if (self.scheduler_config.runner_type == "pooling"
                or self.cache_config.is_attention_free):
            version = "placeholder"

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)

        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
        )
        self.decode_saturation = DecodeSaturationPolicy(
            block_size=self.cache_config.block_size,
            total_gpu_blocks=num_gpu_blocks or 0,
            scheduler_config=self.scheduler_config,
        )
        self.time_predictor = None

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Decode admission backlogs used by decode-saturation mode. These
        # requests have completed prefill elsewhere and can enter the regular
        # running queue once the saturation policy admits them. Keep RT and BE
        # separate so RT admission remains FCFS while BE can be throttled.
        self.waiting_rt_decode_admission: Deque[SequenceGroup] = deque()
        self.waiting_be_decode_admission: Deque[SequenceGroup] = deque()
        self._decode_reservations = DecodeReservations()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()
        # Sequence groups finished requests ids since last step iteration.
        # It lets the model know that any state associated with these requests
        # can and must be released after the current step.
        # This is used to evict the finished requests from the Mamba cache.
        self._finished_requests_ids: List[str] = list()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        # preemption mode, RECOMPUTE or SWAP
        self.user_specified_preemption_mode = scheduler_config.preemption_mode

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0

        # Used to cache python objects
        self._seq_group_metadata_cache: List[PyObjectCache] = []
        self._scheduler_running_outputs_cache: List[PyObjectCache] = []
        self._scheduled_seq_group_cache: List[PyObjectCache] = []

        # For async output processing, we need to swap cache buffers between
        # iterations. I.e. since the output processing is lagged one step,
        # we cannot reuse the cached objects immediately when the schedule()
        # is called again, but only when schedule() is called the second time.
        self.output_proc_callback = output_proc_callback
        self.use_async_output_proc = self.output_proc_callback is not None
        self.num_cache_iters = 2 if self.use_async_output_proc else 1

        self.cache_id = 0
        for i in range(self.num_cache_iters):
            self._seq_group_metadata_cache.append(
                PyObjectCache(seq_group_metadata_builder))
            self._scheduler_running_outputs_cache.append(
                PyObjectCache(scheduler_running_outputs_builder))
            self._scheduled_seq_group_cache.append(
                PyObjectCache(scheduled_seq_group_builder))

        # For async postprocessor, the extra decode run cannot be done
        # when the request reaches max_model_len. In this case, the request
        # will be stopped during schedule() call and added to this stop list
        # for processing and deallocation by the free_finished_seq_groups()
        self._async_stopped: List[SequenceGroup] = []

        # List with the chunk sizes to hand out to each sequence depending
        # on how many partial prefills are running. This is slightly faster than
        # running an integer division every time a prefill is scheduled.
        # This splits the budget evenly among all prefills.
        self.partial_prefill_budget_lookup_list = [0] * (
            self.scheduler_config.max_num_partial_prefills + 1)
        self.partial_prefill_budget_lookup_list[0] = (
            scheduler_config.max_num_batched_tokens)
        for i in range(1, self.scheduler_config.max_num_partial_prefills + 1):
            self.partial_prefill_budget_lookup_list[i] = (
                scheduler_config.max_num_batched_tokens // i)

    @property
    def next_cache_id(self):
        return (self.cache_id + 1) % self.num_cache_iters

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # D-pool decode-ready requests stay out of the normal prefill queue.
        # Direct injection marks the externally computed prompt complete and
        # routes the request through capacity-aware decode admission.
        self.decode_saturation.observe_rt_arrival(seq_group, time.time())
        extra_args = getattr(seq_group.sampling_params, "extra_args", None)
        direct_decode = bool(
            extra_args and extra_args.get("saturo_direct_decode", False))
        legacy_dummy_decode = bool(
            extra_args and extra_args.get("saturo_dummy_kv_decode", False))
        if self.decode_saturation.enabled and legacy_dummy_decode:
            raise ValueError(
                "D-pool decode-only requests must use "
                "saturo_direct_decode; DummyConnector hand-off was removed")
        is_rt = is_rt_request(seq_group)
        is_be = is_be_request(seq_group)
        direct_rt = is_rt and direct_decode
        direct_be = is_be and direct_decode
        if direct_rt or direct_be:
            # A phase-isolated D-pool replay models the post-transfer boundary:
            # prompt KV has already been produced by P-pool. Mark the prompt
            # computed and let the dedicated admission queue allocate its local
            # block table without consuming a synthetic hand-off/prefill step.
            # The benchmark does not validate token correctness, so the local
            # blocks may contain dummy KV just like DummyConnector's handoff.
            for seq in seq_group.get_seqs():
                # The supplied context already includes the latest P-pool
                # output (or last prompt token). Reclassify that token as
                # the hidden decode input; appending an additional copy would
                # consume one extra native context position and truncate
                # otherwise valid prompt+output == max_model_len requests.
                context_token_ids = seq.get_prompt_token_ids()
                seed_token_id = context_token_ids[-1]
                seq.data = SequenceData.from_seqs(context_token_ids[:-1])
                seq.data.update_num_computed_tokens(
                    len(context_token_ids) - 1)
                seq.append_token_id(
                    seed_token_id,
                    {seed_token_id: Logprob(logprob=0.0)},
                )
            if direct_rt:
                self.waiting_rt_decode_admission.append(seq_group)
                # Convert the early claim into the admission queue's real
                # block demand without an unreserved gap or double counting.
                self.release_decode_reservation(
                    (extra_args or {}).get("saturo_decode_reservation_id", ""))
            else:
                self.waiting_be_decode_admission.append(seq_group)
        elif is_be and legacy_dummy_decode:
            # P/T-pool validation still supports request-scoped dummy-KV
            # populations. D-pool workers no longer emit this legacy marker.
            self.waiting_be_decode_admission.append(seq_group)
        else:
            self.waiting.append(seq_group)

    def abort_scheduled_be_batch(
        self,
        scheduled_seq_groups: Iterable["ScheduledSequenceGroup"],
        is_be_group: Optional[Callable[[SequenceGroup], bool]] = None,
    ) -> int:
        """Rollback scheduled BE groups after a mid-forward interruption.

        The scheduler has already moved these groups to RUNNING and may have
        allocated/appended KV slots before model execution starts. If execution
        is aborted at a layer boundary, no model output should be processed, so
        scheduler-side state must be restored explicitly. The predicate keeps
        this helper usable for both P-pool and future D-pool saturation code.
        """
        if is_be_group is None:
            is_be_group = is_be_request

        aborted: List[SequenceGroup] = []
        decode_ready_aborted: List[SequenceGroup] = []
        seen: Set[str] = set()

        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            if not is_be_group(seq_group):
                continue
            if seq_group.request_id in seen:
                continue
            seen.add(seq_group.request_id)

            self.running = deque(
                sg for sg in self.running if sg is not seq_group)
            self.swapped = deque(
                sg for sg in self.swapped if sg is not seq_group)

            swapped_seqs = seq_group.get_seqs(status=SequenceStatus.SWAPPED)
            if swapped_seqs:
                for seq in swapped_seqs:
                    seq.status = SequenceStatus.RUNNING

            if seq_group.get_seqs(status=SequenceStatus.RUNNING):
                self._preempt_by_recompute(seq_group)
                extra_args = getattr(
                    seq_group.sampling_params, "extra_args", None) or {}
                if extra_args.get("saturo_dummy_kv_decode", False):
                    decode_ready_aborted.append(seq_group)
                else:
                    aborted.append(seq_group)

        if aborted:
            self.waiting.extend(aborted)
        if decode_ready_aborted:
            # Recompute preemption releases the synthetic prompt KV.  These
            # requests must therefore repeat the DummyConnector hand-off;
            # putting them in the ordinary waiting queue would silently turn
            # the Decode-ready population into real Prefill work after every
            # layer-wise interruption.
            self.waiting_be_decode_admission.extendleft(
                reversed(decode_ready_aborted))
        return len(aborted) + len(decode_ready_aborted)

    def abort_p_pool_scheduled_be(
        self,
        scheduled_seq_groups: Iterable["ScheduledSequenceGroup"],
    ) -> int:
        return self.abort_scheduled_be_batch(scheduled_seq_groups)

    def _is_rt_seq_group(self, seq_group: SequenceGroup) -> bool:
        return is_rt_request(seq_group)

    def _is_be_seq_group(self, seq_group: SequenceGroup) -> bool:
        return is_be_request(seq_group)

    def _should_defer_be_for_p_pool(
        self,
        seq_groups: Iterable[SequenceGroup],
    ) -> bool:
        return (
            getattr(self.scheduler_config, "enable_p_pool_saturation", False)
            and any(self._is_rt_seq_group(seq_group)
                    for seq_group in seq_groups)
        )

    def _defer_be_from_running_queue(
        self,
        running_queue: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        """Temporarily move BE running groups aside while RT is pending.

        P-pool preemption and D-pool saturation both treat infinite-deadline
        groups as BE. Keeping the predicate in one place avoids splitting the
        same RT/BE classification across the main scheduler loops.
        """
        skipped_be: Deque[SequenceGroup] = deque()
        rt_running_queue: Deque[SequenceGroup] = deque()
        while running_queue:
            seq_group = running_queue.popleft()
            if self._is_be_seq_group(seq_group):
                skipped_be.append(seq_group)
            else:
                rt_running_queue.append(seq_group)
        running_queue.extend(rt_running_queue)
        return skipped_be

    def _schedule_fixed_budget_rt_prefill_priority(self) -> SchedulerOutputs:
        """Fixed-budget mixed P/D policy with RT-prefill priority.

        Vanilla chunked prefill schedules every already-running partial
        prefill before looking at newly waiting requests.  Consequently, a
        long BE partial prefill can repeatedly consume the whole token budget
        while a newly arrived RT prefill waits.  For the fixed-budget
        limitation experiment, promote one waiting RT prefill to RUNNING and
        order the running queue as:

            all decode -> RT partial prefill -> BE partial prefill

        The normal chunked-prefill implementation then performs allocation,
        KV checks, preemption bookkeeping, and output construction.  The
        global token budget is never enlarged; BE prefill can still use any
        tokens left after RT prefill.
        """
        has_running_rt_prefill = any(
            self._is_rt_seq_group(group) and not is_decode_phase(group)
            for group in self.running)

        if not has_running_rt_prefill:
            waiting_rt = next((
                group for group in self.waiting
                if self._is_rt_seq_group(group)
                and not is_decode_phase(group)
            ), None)
            if (waiting_rt is not None
                    and len(self.running)
                    < self.scheduler_config.max_num_seqs
                    and self.block_manager.can_allocate(waiting_rt)
                    == AllocStatus.OK):
                self.waiting.remove(waiting_rt)
                self._allocate_and_set_running(waiting_rt)
                self.running.append(waiting_rt)

        decode_groups: Deque[SequenceGroup] = deque()
        rt_prefills: Deque[SequenceGroup] = deque()
        be_prefills: Deque[SequenceGroup] = deque()
        while self.running:
            group = self.running.popleft()
            if is_decode_phase(group):
                decode_groups.append(group)
            elif self._is_rt_seq_group(group):
                rt_prefills.append(group)
            else:
                be_prefills.append(group)
        self.running.extend(decode_groups)
        self.running.extend(rt_prefills)
        self.running.extend(be_prefills)
        return self._schedule_chunked_prefill()

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the running queue.
        # Only for testing purposes.
        self.running.append(seq_group)

    def _add_seq_group_to_swapped(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the swapped queue.
        # Only for testing purposes.
        self.swapped.append(seq_group)

    def abort_seq_group(
        self,
        request_id: Union[str, Iterable[str]],
        seq_id_to_seq_group: Optional[Dict[str, SequenceGroupBase]] = None,
    ) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
            seq_id_to_seq_group: helper for groups with n>1
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        seq_id_to_seq_group = seq_id_to_seq_group or {}
        for state_queue in [
                self.waiting, self.waiting_rt_decode_admission,
                self.waiting_be_decode_admission, self.running, self.swapped
        ]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                # When n>1, seq_group.request_id looks like
                # foo_parallel_sample_0, while request_ids is just foo, and we
                # should resolve it as real_request_id to match.
                if seq_group.request_id in seq_id_to_seq_group:
                    real_request_id = seq_id_to_seq_group[
                        seq_group.request_id].group_id
                else:
                    real_request_id = seq_group.request_id
                if real_request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    # We can't remove real_request_id in request_ids here,
                    # because there may be other seq groups sharing the same
                    # real_request_id
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                # Remove the aborted request from the Mamba cache.
                self._finished_requests_ids.append(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)
                if aborted_group.request_id in seq_id_to_seq_group:
                    del seq_id_to_seq_group[aborted_group.request_id]

                self._free_seq_group_cross_attn_blocks(aborted_group)

    def _free_seq_group_cross_attn_blocks(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        """
        Free a sequence group from a cross-attention block table.
        Has no effect on decoder-only models.
        """
        if seq_group.is_encoder_decoder():
            self.block_manager.free_cross(seq_group)

    def has_unfinished_seqs(self) -> bool:
        return (len(self.waiting) != 0
                or len(self.waiting_rt_decode_admission) != 0
                or len(self.waiting_be_decode_admission) != 0
                or len(self.running) != 0 or len(self.swapped) != 0)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_manager.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        return self.block_manager.reset_prefix_cache(device)

    def get_num_unfinished_seq_groups(self) -> int:
        return (len(self.waiting) + len(self.waiting_rt_decode_admission)
                + len(self.waiting_be_decode_admission) + len(self.running)
                + len(self.swapped))

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """Flushes the list of request ids of previously finished seq_groups."""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerRunningOutputs:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
            partial_prefill_metadata: information about the partial prefills
            that are currently running

        Returns:
            SchedulerRunningOutputs.
        """
        ret: SchedulerRunningOutputs = self._scheduler_running_outputs_cache[
            self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()

        ret.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)

        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[
            ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        running_queue = self.running

        if self.decode_saturation.enabled:
            self.decode_saturation.order_running_queue(running_queue)

        # P-pool: when RT requests are pending, do not schedule BE groups that
        # are already in RUNNING. Their KV/progress is preserved and they are
        # restored to the running queue after this scheduling pass.
        skipped_be: Deque[SequenceGroup] = deque()
        if self._should_defer_be_for_p_pool(self.waiting):
            skipped_be = self._defer_be_from_running_queue(running_queue)

        assert len(self._async_stopped) == 0
        while running_queue:
            seq_group = running_queue[0]
            # We discard the cached tokens info here because we don't need it
            # for running sequence:
            #   1. If a sequence is running with chunked prefill, the cached
            #      tokens info was already used for the first prefill.
            #   2. If a sequence is running with non-chunked prefill, then
            #      there it's a decoding sequence, and the cached tokens info is
            #      irrelevant.
            num_uncached_new_tokens, _ = \
                self._get_num_new_uncached_and_cached_tokens(
                seq_group,
                SequenceStatus.RUNNING,
                enable_chunking,
                budget,
                partial_prefill_metadata,
            )

            num_running_tokens = num_uncached_new_tokens
            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # With async postprocessor, an extra decode run is done
            # to process the final tokens. The check below avoids this extra
            # decode run when the model max len is reached, in order to avoid
            # a memory overflow.
            if (self.use_async_output_proc and seq_group.seqs[0].get_len()
                    > self.scheduler_config.max_model_len):
                self._async_stopped.append(seq_group)
                continue

            # NOTE(woosuk): Preemption happens only when there is no available
            # slot to keep all the sequence groups in the RUNNING state.
            while not self._can_append_slots(seq_group, enable_chunking):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    # Preempt the lowest-priority sequence group.
                    victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # With async postprocessor, before preempting a sequence
                # we need to ensure it has no pending async postprocessor
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(
                        request_id=victim_seq_group.request_id)

                    # It may be that the async pending "victim_seq_group"
                    # becomes finished, in which case we simply free it.
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                # Do preemption
                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                is_prefill = seq_group.is_prefill()

                scheduled_seq_group: ScheduledSequenceGroup = (
                    self._scheduled_seq_group_cache[
                        self.cache_id].get_object())
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        if skipped_be:
            running_queue.extend(skipped_be)

        return ret

    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # Logically evicted seq groups (from decode-saturation
            # _prepare_decode_saturation_blocks) have no block table entry
            # because free_seq() was called.  Skip them here; the saturation
            # path restores them through _restore_decode_saturation_be.
            if self.decode_saturation.enabled:
                any_logical = any(
                    seq.seq_id not in self.block_manager.block_tables
                    for seq in seq_group.get_seqs(
                        status=SequenceStatus.SWAPPED))
                if any_logical:
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group,
                self._get_num_lookahead_slots(is_prefill, enable_chunking))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id,
                )
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.SWAPPED, enable_chunking,
                    budget))

            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.SWAPPED)
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(
                        seq_group,
                        token_chunk_size=num_new_tokens_uncached +
                        num_new_tokens_cached,
                    ))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False, enable_chunking=enable_chunking),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if (self.scheduler_config.chunked_prefill_enabled
                and not self.scheduler_config.is_multi_step):
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(
                self.scheduler_config.max_model_len,
                self.scheduler_config.max_num_batched_tokens,
            )

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if seq_group.lora_request and seq_group.lora_request.long_lora_max_len:
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _get_priority(self,
                      seq_group: SequenceGroup) -> Tuple[Optional[int], float]:
        """Get the priority of the sequence group.
        Highest preference to user-defined priority, followed by arrival time.
        Args:
            seq_group: The sequence group input.
        Returns:
            The priority of the sequence group.
        """
        return seq_group.priority, seq_group.arrival_time

    def _schedule_priority_preemption(
        self,
        budget: SchedulingBudget,
    ) -> int:
        """Sorts waiting and running queue. Also, force preempt requests
        from the running queue if their priority is lower.
        Priority-based preemption is used with the priority policy.
        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
        Returns:
            A count of priority-based preemptions.
        """

        waiting_queue = self.waiting

        running_queue = deque(sorted(self.running, key=self._get_priority))

        blocks_to_swap_out: List[Tuple[int, int]] = []
        force_preemption_count = 0

        if waiting_queue:
            seq_group = waiting_queue.popleft()
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, _ = \
                self._get_num_new_uncached_and_cached_tokens(
                seq_group, SequenceStatus.WAITING, False, budget)

            # Only preempt if priority inversion exists
            while running_queue and self._get_priority(
                    running_queue[-1]) > self._get_priority(seq_group):
                # Only preempt if waiting sequence cannot be allocated
                can_allocate = self.block_manager.can_allocate(seq_group)
                if (num_new_tokens_uncached > 0
                        and can_allocate == AllocStatus.OK
                        and budget.can_schedule(
                            num_new_tokens=num_new_tokens_uncached,
                            num_new_seqs=num_new_seqs,
                        )):
                    break

                # Adjust budget to remove the victim sequence group
                vseq_group = running_queue.pop()
                num_running_tokens_uncached, _ = (
                    self._get_num_new_uncached_and_cached_tokens(
                        vseq_group, SequenceStatus.RUNNING, False, budget))
                budget.subtract_num_batched_tokens(
                    vseq_group.request_id, num_running_tokens_uncached)
                num_running_seqs = vseq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(vseq_group.request_id,
                                         num_running_seqs)

                # Preempt out the victim sequence group
                self._preempt(vseq_group, blocks_to_swap_out)
                waiting_queue.appendleft(vseq_group)
                force_preemption_count += 1
            # Put the sequence back into the waiting queue
            waiting_queue.appendleft(seq_group)

            self.remove_seq_from_computed_blocks_tracker(
                seq_group, SequenceStatus.WAITING)

        waiting_queue = deque(sorted(waiting_queue, key=self._get_priority))

        self.waiting = waiting_queue
        self.running = running_queue
        return force_preemption_count

    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
            partial_prefill_metadata: information about the partial prefills
                that are currently running

        Returns:
            SchedulerPrefillOutputs.
        """
        if budget.remaining_token_budget() == 0:
            # Do nothing: Can't add any more prefill anyway
            return SchedulerPrefillOutputs(
                seq_groups=[],
                ignored_seq_groups=[],
                num_lookahead_slots=self._get_num_lookahead_slots(
                    is_prefill=True, enable_chunking=enable_chunking),
            )
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []
        using_prompt_embeds: bool = False

        waiting_queue = self.waiting

        self.waiting = deque(
            sorted(self.waiting, key=get_prefill_deadline))

        waiting_queue = self.waiting
        
        if (self.scheduler_config.serial_execution and waiting_queue
                and is_rt_request(waiting_queue[0])):
            budget.token_budget = budget.num_batched_tokens + waiting_queue[0].seqs[0].get_prompt_len()

        p_pool_skip_be = self._should_defer_be_for_p_pool(waiting_queue)

        
        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            if p_pool_skip_be and self._is_be_seq_group(seq_group):
                break

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            if (partial_prefill_metadata is not None
                    and not partial_prefill_metadata.can_schedule(seq_group)):
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group,
                    SequenceStatus.WAITING,
                    enable_chunking,
                    budget,
                    partial_prefill_metadata=partial_prefill_metadata,
                ))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.FINISHED_IGNORED)
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.WAITING)
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens,
                    num_lookahead_slots,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.FINISHED_IGNORED)
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # We cannot mix sequence groups that use prompt embeds and
            # those that do not.
            if len(seq_groups) == 0:
                using_prompt_embeds = seq_group.uses_prompt_embeds()
            if using_prompt_embeds != seq_group.uses_prompt_embeds():
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.WAITING)
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    self.remove_seq_from_computed_blocks_tracker(
                        seq_group, SequenceStatus.WAITING)
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens
                    >= self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.WAITING)
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.WAITING)
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if partial_prefill_metadata is not None:
                partial_prefill_metadata.maybe_increment_partial_prefills(
                    seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking,
                )

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking),
        )

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.

        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = (set(
            seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None)

        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            prefills = self._schedule_prefills(budget,
                                               curr_loras,
                                               enable_chunking=False)

        if len(prefills.seq_groups
               ) == 0 and self.scheduler_config.policy == "priority":
            self._schedule_priority_preemption(budget)

        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            running_scheduled = self._schedule_running(budget,
                                                       curr_loras,
                                                       enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if (len(running_scheduled.preempted) +
                    len(running_scheduled.swapped_out) == 0):
                swapped_in = \
                    self._schedule_swapped(budget, curr_loras)

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])

        self.running.extend(running_scheduled.decode_seq_groups_list)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = len(running_scheduled.preempted) + len(
            running_scheduled.swapped_out)

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        # Merge lists
        num_prefill_groups = len(prefills.seq_groups)
        ignored_seq_groups_for_embeds = list[SequenceGroup]()
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
            ignored_seq_groups_for_embeds.clear()
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
            if len(scheduled_seq_groups) > 0:
                using_prompt_embeds = scheduled_seq_groups[
                    0].seq_group.uses_prompt_embeds()
                ignored_seq_groups_for_embeds.clear()
                indices_ignored = list[int]()
                for i, schedule_seq_group in enumerate(scheduled_seq_groups):
                    if using_prompt_embeds !=\
                        schedule_seq_group.seq_group.uses_prompt_embeds():
                        ignored_seq_groups_for_embeds.append(
                            schedule_seq_group.seq_group)
                        indices_ignored.append(i)
                if len(ignored_seq_groups_for_embeds) > 0:
                    scheduled_seq_groups = [
                        group for i, group in enumerate(scheduled_seq_groups)
                        if i not in indices_ignored
                    ]
            else:
                ignored_seq_groups_for_embeds.clear()

        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(ignored_seq_groups_for_embeds)
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )

    def _schedule_chunked_prefill(self) -> SchedulerOutputs:
        """Schedule queued requests.

        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to be blocked
        by prefill requests.
        """
        # T-pool validation may preload an independent decode-ready BE
        # population. DummyConnector needs one isolated hand-off iteration to
        # allocate its prompt KV before those requests join normal decode. P/D
        # saturation policies perform the same hand-off in their own schedulers.
        if (self.waiting_be_decode_admission
                and self._is_dummy_kv_decode(
                    self.waiting_be_decode_admission[0])):
            active_decode = [
                seq_group for seq_group in self.running
                if self._is_dummy_kv_decode(seq_group)
            ]
            candidate_decode = active_decode + [
                self.waiting_be_decode_admission[0]
            ]
            if (not active_decode
                    or self._p_pool_dummy_decode_within_limits(
                        candidate_decode)):
                return self._schedule_p_pool_dummy_decode_handoff()

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        prefills = SchedulerPrefillOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # Create partial prefill metadata
        partial_prefill_metadata = PartialPrefillMetadata.from_queues(
            running=self.running,
            waiting=self.waiting,
            scheduler_config=self.scheduler_config,
        )

        # Decoding should be always scheduled first by fcfs.
        running_scheduled = self._schedule_running(
            budget,
            curr_loras,
            enable_chunking=True,
            partial_prefill_metadata=partial_prefill_metadata,
        )

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            swapped_in = self._schedule_swapped(budget, curr_loras)

        prefills = self._schedule_prefills(
            budget,
            curr_loras,
            enable_chunking=True,
            partial_prefill_metadata=partial_prefill_metadata,
        )

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)

        # Update new running requests.
        # By default, vLLM scheduler prioritizes prefills.
        # Once chunked prefill is enabled,
        # the policy is changed to prioritize decode requests.
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        # Because multiple prefills may be running concurrently, we need to
        # make sure that prefills which are scheduled to finish are listed
        # before those that won't. This is so that on the next scheduling
        # iteration when they have transitioned to the decode stage, they are
        # properly prioritized over sequences that are still in the prefill
        # stage.
        self.running.extend(
            self._order_finishing_prefills_first(
                running_scheduled.prefill_seq_groups))
        self.running.extend([s.seq_group for s in prefills.seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        # Put prefills first due to Attention backend ordering assumption.
        scheduled_seq_groups = (prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups)
        num_prefill_groups = (len(prefills.seq_groups) +
                              len(swapped_in.prefill_seq_groups) +
                              len(running_scheduled.prefill_seq_groups))
        # If all prompts, then we set num_lookahead_slots to 0
        # this allows us to go through the `no_spec` path in
        # `spec_decode_worker.py`
        all_prefills = len(scheduled_seq_groups) == num_prefill_groups
        num_lookahead_slots = (0 if
                               (all_prefills
                                and not self.scheduler_config.is_multi_step)
                               else running_scheduled.num_lookahead_slots)
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

    def _has_rt_work_for_p_pool(self) -> bool:
        rt_queues = (
            self.waiting,
            self.running,
            self.swapped,
            self.waiting_rt_decode_admission,
        )
        return any(
            self._is_rt_seq_group(seq_group)
            for queue in rt_queues
            for seq_group in queue)

    def _schedule_p_pool_rt_only(self) -> SchedulerOutputs:
        """Schedule RT work only while P-pool has pending/active RT requests.

        Under P-pool saturation RT prefill runs to completion in a single
        step (no chunking). The token budget is raised to max_model_len and
        the partial-prefill slot cap is bypassed, so max_num_batched_tokens
        only bounds BE chunk size, never RT prefill.
        """
        rt_token_budget = max(
            int(self.scheduler_config.max_num_batched_tokens),
            int(self.scheduler_config.max_model_len),
        )
        budget = SchedulingBudget(
            token_budget=rt_token_budget,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        # None => RT prefills are not split by the per-slot chunk budget
        # (max_num_batched_tokens // num_partial_prefills), so a full RT
        # prompt is prefilled in one step.
        partial_prefill_metadata = None

        running_queue = self.running
        rt_running_queue: Deque[SequenceGroup] = deque()
        deferred_non_rt: Deque[SequenceGroup] = deque()
        while running_queue:
            seq_group = running_queue.popleft()
            if self._is_rt_seq_group(seq_group):
                rt_running_queue.append(seq_group)
            else:
                deferred_non_rt.append(seq_group)

        self.running = rt_running_queue
        running_scheduled = self._schedule_running(
            budget,
            curr_loras,
            enable_chunking=True,
            partial_prefill_metadata=partial_prefill_metadata,
        )
        waiting_queue = self.waiting
        rt_waiting_queue: Deque[SequenceGroup] = deque()
        deferred_waiting: Deque[SequenceGroup] = deque()
        while waiting_queue:
            seq_group = waiting_queue.popleft()
            if self._is_rt_seq_group(seq_group):
                rt_waiting_queue.append(seq_group)
            else:
                deferred_waiting.append(seq_group)
        self.waiting = rt_waiting_queue
        prefills = self._schedule_prefills(
            budget,
            curr_loras,
            enable_chunking=True,
            partial_prefill_metadata=partial_prefill_metadata,
        )
        self.waiting.extend(deferred_waiting)

        assert budget.num_batched_tokens <= rt_token_budget
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        self.waiting.extendleft(running_scheduled.preempted)
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            self._order_finishing_prefills_first(
                running_scheduled.prefill_seq_groups))
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(deferred_non_rt)
        self.swapped.extend(running_scheduled.swapped_out)

        scheduled_seq_groups = (
            prefills.seq_groups +
            running_scheduled.prefill_seq_groups +
            running_scheduled.decode_seq_groups)
        num_prefill_groups = (len(prefills.seq_groups) +
                              len(running_scheduled.prefill_seq_groups))
        all_prefills = len(scheduled_seq_groups) == num_prefill_groups
        num_lookahead_slots = (0 if
                               (all_prefills
                                and not self.scheduler_config.is_multi_step)
                               else running_scheduled.num_lookahead_slots)
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=[],
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

    @staticmethod
    def _is_dummy_kv_decode(seq_group: SequenceGroup) -> bool:
        return bool((getattr(seq_group.sampling_params, "extra_args", None)
                     or {}).get("saturo_dummy_kv_decode", False))

    @staticmethod
    def _is_direct_decode(seq_group: SequenceGroup) -> bool:
        return bool((getattr(seq_group.sampling_params, "extra_args", None)
                     or {}).get("saturo_direct_decode", False))

    def _p_pool_dummy_decode_within_limits(
        self,
        seq_groups: List[SequenceGroup],
    ) -> bool:
        # BE-decode is bounded ONLY by the KV blocks left after reserving space
        # for the peak RT prefill and peak BE prefill. That reservation is
        # computed at dataset load time and passed here as an absolute block
        # count (SATURO_BE_DECODE_BLOCK_LIMIT). Keep the P-pool-specific name as
        # a compatibility fallback for existing launchers. When neither is set,
        # decode may use every GPU block. The BE-chunk latency budget is never
        # divided between prefill and decode -- every admitted decode request
        # runs each step.
        total_blocks = int(self.cache_config.num_gpu_blocks or 0)
        limit_env = (os.environ.get("SATURO_BE_DECODE_BLOCK_LIMIT", "")
                     or os.environ.get(
                         "SATURO_P_POOL_BE_DECODE_BLOCK_LIMIT", ""))
        if limit_env:
            block_limit = max(1, int(float(limit_env)))
        else:
            block_limit = max(1, total_blocks)
        if total_blocks:
            block_limit = min(block_limit, total_blocks)
        used_blocks = sum(
            self.decode_saturation.seq_group_blocks(seq_group)
            for seq_group in seq_groups)
        if used_blocks > block_limit:
            return False

        # BE-decode active-count cap. Without it decode fills greedily to the
        # block limit, consumes the whole be_chunk_ms budget, and starves BE
        # prefill. The simulator derives the balancing cap; prefer the
        # SchedulerConfig field, fall back to the env var (same layered pattern
        # as the block limit above). 0 / unset means unlimited.
        batch_cap = int(getattr(
            self.scheduler_config, "p_pool_max_be_decode_batch", 0) or 0)
        if batch_cap <= 0:
            batch_env = os.environ.get("SATURO_P_POOL_MAX_BE_DECODE_BATCH", "")
            if batch_env:
                batch_cap = int(float(batch_env))
        if batch_cap > 0 and len(seq_groups) > batch_cap:
            return False
        return True

    def _schedule_p_pool_dummy_decode_handoff(self) -> SchedulerOutputs:
        """Restore a pure batch of P-pool BE decode-ready requests.

        These requests have prompt KV supplied by DummyConnector.  They are
        deliberately scheduled in a separate, full-prompt iteration: mixing
        them with ordinary prefills would make the connector bypass real
        prefill work, while mixing them with decode tokens would leave the
        connector without decode hidden states.
        """
        token_budget = max(
            int(self.scheduler_config.max_num_batched_tokens),
            int(self.scheduler_config.max_model_len),
        )
        budget = SchedulingBudget(
            token_budget=token_budget,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        active_decode = [
            seq_group for seq_group in self.running
            if self._is_dummy_kv_decode(seq_group)
        ]

        # Only hand off the FIFO prefix that fits the decode KV share. The
        # rest remains queued, leaving the other share available for real BE
        # prefill allocation instead of letting long decode prompts consume
        # the entire cache before the first prefill can start.
        queued = self.waiting_be_decode_admission
        admissible: Deque[SequenceGroup] = deque()
        while queued:
            seq_group = queued[0]
            candidate_decode = active_decode + list(admissible) + [seq_group]
            if ((active_decode or admissible)
                    and not self._p_pool_dummy_decode_within_limits(
                        candidate_decode)):
                break
            admissible.append(queued.popleft())

        normal_waiting = self.waiting
        self.waiting = admissible
        prefills = self._schedule_prefills(
            budget,
            curr_loras,
            enable_chunking=False,
            partial_prefill_metadata=None,
        )
        self.waiting_be_decode_admission.extendleft(reversed(self.waiting))
        self.waiting = normal_waiting

        self.running.extend([s.seq_group for s in prefills.seq_groups])
        return SchedulerOutputs(
            scheduled_seq_groups=prefills.seq_groups,
            num_prefill_groups=len(prefills.seq_groups),
            num_batched_tokens=(budget.num_batched_tokens
                                + budget.num_cached_tokens),
            blocks_to_swap_in=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=0,
            running_queue_size=len(self.running),
            preempted=0,
        )





    def _schedule_prefill_saturation(self) -> SchedulerOutputs:
        """P-pool policy: RT-only when RT exists, otherwise launch BE chunk."""
        if self._has_rt_work_for_p_pool():
            return self._schedule_p_pool_rt_only()

        if self.waiting_be_decode_admission:
            active_decode = [
                seq_group for seq_group in self.running
                if self._is_dummy_kv_decode(seq_group)
            ]
            candidate_decode = active_decode + [
                self.waiting_be_decode_admission[0]
            ]
            if (not active_decode
                    or self._p_pool_dummy_decode_within_limits(
                        candidate_decode)):
                return self._schedule_p_pool_dummy_decode_handoff()

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        partial_prefill_metadata = PartialPrefillMetadata.from_queues(
            running=self.running,
            waiting=self.waiting,
            scheduler_config=self.scheduler_config,
        )
        running_scheduled = SchedulerRunningOutputs.create_empty()
        running_scheduled.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=True)

        running_queue = self.running
        self.running = deque()
        running_be_prefills: Deque[SequenceGroup] = deque()
        deferred_non_be: Deque[SequenceGroup] = deque()

        while running_queue:
            seq_group = running_queue.popleft()
            if not self._is_be_seq_group(seq_group):
                deferred_non_be.append(seq_group)
                continue
            if not is_decode_phase(seq_group):
                running_be_prefills.append(seq_group)
                continue

            scheduled = self._schedule_decode_for_saturation(
                seq_group,
                budget,
                curr_loras,
                running_scheduled.blocks_to_copy,
            )
            if scheduled is None:
                self.running.append(seq_group)
                continue
            running_scheduled.decode_seq_groups.append(scheduled)
            running_scheduled.decode_seq_groups_list.append(seq_group)

        target_ms = max(
            float(getattr(self.scheduler_config, "p_pool_be_chunk_ms",
                          100.0)), 0.0)
        chunk_size = int(getattr(
            self.scheduler_config, "p_pool_be_chunk_size", 0) or 0)
        free_block_tokens = (
            max(self.block_manager.get_num_free_gpu_blocks(), 0)
            * self.cache_config.block_size)
        max_be_prefill_tokens = 0
        if target_ms > 0 or chunk_size > 0:
            max_be_prefill_tokens = min(
                max(budget.remaining_token_budget(), 0), free_block_tokens)
            if chunk_size > 0:
                # Fixed-size mode: chunk_size is the per-STEP BE prefill
                # token cap (plain chunked-prefill semantics), not a
                # per-request slice. Without this cap the multi-request loop
                # below would fill the whole max_num_batched_tokens budget.
                max_be_prefill_tokens = min(max_be_prefill_tokens, chunk_size)
        target = _PrefillSaturationChunkTarget(
            target_interval_ms=target_ms)
        prefills = self._schedule_be_prefill_chunks_for_saturation(
            target,
            max_be_prefill_tokens,
            budget,
            curr_loras,
            running_scheduled,
            SchedulerSwappedInOutputs.create_empty(),
            partial_prefill_metadata,
            running_be_prefills,
            deferred_running=deferred_non_be,
        )

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        self.running.extend(deferred_non_be)
        self.running.extend(running_be_prefills)
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend([s.seq_group for s in prefills.seq_groups])

        scheduled_seq_groups = (
            prefills.seq_groups + running_scheduled.decode_seq_groups)
        num_prefill_groups = len(prefills.seq_groups)
        all_prefills = len(scheduled_seq_groups) == num_prefill_groups
        num_lookahead_slots = (0 if
                               (all_prefills
                                and not self.scheduler_config.is_multi_step)
                               else running_scheduled.num_lookahead_slots)
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=[],
            blocks_to_swap_out=[],
            blocks_to_copy=running_scheduled.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=0,
        )

    def _ignore_never_allocatable_decode_request(
        self, seq_group: SequenceGroup
    ) -> None:
        logger.warning(
            "Prefilled decode request %s is too long for the KV cache.",
            seq_group.request_id,
        )
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.FINISHED_IGNORED

    def _has_decode_admission_capacity(
        self, seq_group: SequenceGroup, reserved_slots: int = 0,
    ) -> bool:
        running_seqs = sum(group.get_max_num_running_seqs()
                           for group in self.running)
        return (running_seqs + reserved_slots
                + seq_group.get_max_num_running_seqs()
                <= self.scheduler_config.max_num_seqs)

    def _admit_unconstrained_decode_admission_queue(
        self,
        queue: Deque[SequenceGroup],
    ) -> bool:
        while queue:
            seq_group = queue[0]
            if not self._has_decode_admission_capacity(seq_group):
                return False
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.NEVER:
                queue.popleft()
                self._ignore_never_allocatable_decode_request(seq_group)
                continue
            if can_allocate == AllocStatus.LATER:
                return False

            queue.popleft()
            self._allocate_and_set_running(seq_group)
            self.running.append(seq_group)
        return True

    def _admit_rt_decode_admission_queue(self) -> bool:
        return self._admit_unconstrained_decode_admission_queue(
            self.waiting_rt_decode_admission)

    def _admit_be_decode_admission_queue(self, decision) -> int:
        if not self.waiting_be_decode_admission:
            return decision.current_be_blocks

        queued = self.waiting_be_decode_admission
        self.waiting_be_decode_admission = deque()
        projected_be_blocks = decision.current_be_blocks
        while queued:
            seq_group = queued.popleft()
            if not self._has_decode_admission_capacity(
                    seq_group, self._decode_reserved_slots()):
                self.waiting_be_decode_admission.append(seq_group)
                break
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.NEVER:
                self._ignore_never_allocatable_decode_request(seq_group)
                continue
            if can_allocate == AllocStatus.LATER:
                self.waiting_be_decode_admission.append(seq_group)
                continue
            if not self.decode_saturation.can_admit_be(
                    seq_group, decision, projected_be_blocks):
                self.waiting_be_decode_admission.append(seq_group)
                continue
            if not self._can_admit_be_decode_by_latency(seq_group, decision):
                self.waiting_be_decode_admission.append(seq_group)
                continue

            self._allocate_and_set_running(seq_group)
            self.running.append(seq_group)
            projected_be_blocks += self.decode_saturation.seq_group_blocks(
                seq_group)

        self.waiting_be_decode_admission.extend(queued)
        return projected_be_blocks

    def _admit_decode_admission_queues(self, decision) -> int:
        rt_fully_admitted = self._admit_rt_decode_admission_queue()
        if not rt_fully_admitted:
            return decision.current_be_blocks
        return self._admit_be_decode_admission_queue(decision)

    def _can_admit_be_decode_by_latency(
        self,
        seq_group: SequenceGroup,
        decision,
    ) -> bool:
        if self.time_predictor is None:
            return True
        candidate = [
            sg for sg in self.running
            if is_decode_phase(sg)
        ] + [seq_group]
        predicted_ms = self.decode_saturation.predict_decode_batch_ms(candidate)
        return predicted_ms <= decision.target_interval_ms

    def _predictor_reqs_from_scheduled(self, scheduled_seq_groups):
        prefill_reqs = []
        decode_reqs = []
        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            query_len = scheduled_seq_group.token_chunk_size or 1
            for seq in seq_group.get_seqs():
                computed_tokens = seq.get_num_computed_tokens()
                if not is_decode_phase(seq_group):
                    prefill_reqs.append(
                        (query_len, computed_tokens + query_len))
                else:
                    decode_reqs.append((query_len, computed_tokens))
        return prefill_reqs, decode_reqs

    def _find_be_prefill_chunk_by_latency(
        self,
        decision,
        max_prefill_tokens: int,
        running_scheduled: SchedulerRunningOutputs,
        swapped_in: SchedulerSwappedInOutputs,
        scheduled_be_prefills: List[ScheduledSequenceGroup],
        seq_group: SequenceGroup,
    ) -> int:
        if max_prefill_tokens <= 0:
            return 0
        # Fixed-size mode: when p_pool_be_chunk_size > 0 it overrides the
        # latency (ms) target -- the step runs one plain chunked-prefill
        # slice of at most chunk_size tokens, skipping the TimePredictor
        # binary search. The caller also caps the per-step BE prefill total
        # at chunk_size and stops after the first scheduled chunk.
        chunk_size = int(getattr(
            self.scheduler_config, "p_pool_be_chunk_size", 0) or 0)
        if chunk_size > 0 and not self.decode_saturation.enabled:
            return min(max_prefill_tokens, chunk_size)
        scheduled = (running_scheduled.prefill_seq_groups +
                     swapped_in.prefill_seq_groups +
                     scheduled_be_prefills +
                     running_scheduled.decode_seq_groups +
                     swapped_in.decode_seq_groups)
        prefill_reqs, decode_reqs = self._predictor_reqs_from_scheduled(
            scheduled)
        if self.time_predictor is None:
            prefill_ms_per_token = float(getattr(
                self.scheduler_config,
                "p_pool_fallback_prefill_ms_per_token",
                0.02,
            ))
            decode_ms = float(getattr(
                self.scheduler_config, "p_pool_fallback_decode_ms", 1.0))
            base_ms = (
                sum(query_len for query_len, _ in prefill_reqs)
                * prefill_ms_per_token
                + sum(max(1, query_len) for query_len, _ in decode_reqs)
                * decode_ms
            )
            if base_ms >= decision.target_interval_ms:
                return 0
            if prefill_ms_per_token <= 0:
                return max_prefill_tokens
            available_tokens = int(
                (decision.target_interval_ms - base_ms)
                / prefill_ms_per_token)
            return min(max_prefill_tokens, max(available_tokens, 0))

        base_ms = self.time_predictor.predict_forward_ms(
            prefill_reqs, decode_reqs)
        if base_ms >= decision.target_interval_ms:
            return 0

        computed_context = max(
            seq.get_num_computed_tokens() for seq in seq_group.get_seqs())
        hi = min(max_prefill_tokens,
                 self.scheduler_config.max_num_batched_tokens)
        lo = 0
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_prefills = list(prefill_reqs)
            if mid > 0:
                candidate_prefills.append((mid, computed_context + mid))
            predicted_ms = self.time_predictor.predict_forward_ms(
                candidate_prefills, decode_reqs)
            if predicted_ms <= decision.target_interval_ms:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _schedule_decode_for_saturation(
        self,
        seq_group: SequenceGroup,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        blocks_to_copy: List[Tuple[int, int]],
    ) -> Optional[ScheduledSequenceGroup]:
        num_new_tokens, _ = self._get_num_new_uncached_and_cached_tokens(
            seq_group,
            SequenceStatus.RUNNING,
            True,
            budget,
            None,
        )
        if num_new_tokens == 0:
            return None
        num_new_seqs = seq_group.get_max_num_running_seqs()
        if not budget.can_schedule(
                num_new_tokens=num_new_tokens,
                num_new_seqs=num_new_seqs,
        ):
            return None
        if not self._can_append_slots(seq_group, True):
            return None

        self._append_slots(seq_group, blocks_to_copy, True)
        scheduled = ScheduledSequenceGroup(seq_group=seq_group,
                                           token_chunk_size=1)
        budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
        budget.add_num_seqs(seq_group.request_id, num_new_seqs)
        if curr_loras is not None and seq_group.lora_int_id > 0:
            curr_loras.add(seq_group.lora_int_id)
        return scheduled

    def _can_execute_running_be_decode(self, decision, selected, seq_group) -> bool:
        """Enforce the current target on the actual RT+resident BE batch.

        Initial admission alone is insufficient: RT arrivals, context growth
        and asynchronous target invalidation can all reduce safe BE capacity.
        RT is scheduled first and is never removed by this BE-only gate.
        """
        if not self.decode_saturation.enabled:
            return True
        if not getattr(decision, "search_feasible", True):
            return False
        candidate = list(selected) + [ScheduledSequenceGroup(
            seq_group=seq_group, token_chunk_size=1)]
        prefills, decodes = self._predictor_reqs_from_scheduled(candidate)
        if self.time_predictor is not None:
            predicted = self.time_predictor.predict_forward_ms(prefills, decodes)
        else:
            # Preserve the explicit synchronous embedding fallback. Main
            # async-D mode requires a fitted predictor at initialization.
            predicted = sum(max(query, 1) for query, _ in decodes) * float(
                getattr(self.scheduler_config, "p_pool_fallback_decode_ms", 1.))
        return predicted <= decision.target_interval_ms

    def _pop_next_be_prefill_candidate(
        self,
        running_be_prefills: Deque[SequenceGroup],
    ) -> Tuple[Optional[SequenceGroup], bool]:
        if running_be_prefills:
            return running_be_prefills.popleft(), True

        kept: Deque[SequenceGroup] = deque()
        candidate: Optional[SequenceGroup] = None
        while self.waiting:
            seq_group = self.waiting.popleft()
            if candidate is None and self.decode_saturation.is_be(seq_group):
                candidate = seq_group
                break
            kept.append(seq_group)
        kept.extend(self.waiting)
        self.waiting = kept
        return candidate, False

    def _schedule_selected_be_prefill_chunk(
        self,
        seq_group: SequenceGroup,
        chunk_size: int,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        partial_prefill_metadata: PartialPrefillMetadata,
        using_prompt_embeds: Optional[bool],
        is_running: bool,
        blocks_to_copy: List[Tuple[int, int]],
    ) -> Tuple[Optional[ScheduledSequenceGroup], List[SequenceGroup], bool]:
        if chunk_size <= 0:
            return None, [], False

        status = SequenceStatus.RUNNING if is_running else SequenceStatus.WAITING
        seqs = seq_group.get_seqs(status=status)
        assert len(seqs) == 1, (
            "Prefill sequence group should have one active prompt sequence.")
        if (not is_running
                and not partial_prefill_metadata.can_schedule(seq_group)):
            return None, [], False

        original_token_budget = budget.token_budget
        budget.token_budget = min(original_token_budget,
                                  budget.num_batched_tokens + chunk_size)
        ignored_seq_groups: List[SequenceGroup] = []
        try:
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group,
                    status,
                    True,
                    budget,
                    partial_prefill_metadata=partial_prefill_metadata,
                ))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long and exceeds "
                    "limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                for seq in seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.FINISHED_IGNORED)
                ignored_seq_groups.append(seq_group)
                return None, ignored_seq_groups, True

            num_lookahead_slots = 0
            if self.scheduler_config.is_multi_step:
                num_lookahead_slots = self._get_num_lookahead_slots(True, True)

            if not is_running:
                can_allocate = self.block_manager.can_allocate(
                    seq_group, num_lookahead_slots=num_lookahead_slots)
                if can_allocate == AllocStatus.LATER:
                    self.remove_seq_from_computed_blocks_tracker(
                        seq_group, SequenceStatus.WAITING)
                    return None, ignored_seq_groups, False
                if can_allocate == AllocStatus.NEVER:
                    logger.warning(
                        "Input prompt (%d tokens) + lookahead slots (%d) is "
                        "too long and exceeds the capacity of block_manager",
                        num_new_tokens,
                        num_lookahead_slots,
                    )
                    for seq in seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    self.remove_seq_from_computed_blocks_tracker(
                        seq_group, SequenceStatus.FINISHED_IGNORED)
                    ignored_seq_groups.append(seq_group)
                    return None, ignored_seq_groups, True

            if (using_prompt_embeds is not None
                    and using_prompt_embeds != seq_group.uses_prompt_embeds()):
                self.remove_seq_from_computed_blocks_tracker(seq_group, status)
                return None, ignored_seq_groups, False

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    self.remove_seq_from_computed_blocks_tracker(
                        seq_group, status)
                    return None, ignored_seq_groups, False

            if (budget.num_batched_tokens
                    >= self.scheduler_config.max_num_batched_tokens):
                self.remove_seq_from_computed_blocks_tracker(seq_group, status)
                return None, ignored_seq_groups, False

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                self.remove_seq_from_computed_blocks_tracker(seq_group, status)
                return None, ignored_seq_groups, False

            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            if is_running:
                self._append_slots(seq_group, blocks_to_copy, True)
            else:
                self._allocate_and_set_running(seq_group)
                partial_prefill_metadata.maybe_increment_partial_prefills(
                    seq_group)
                if self.scheduler_config.is_multi_step:
                    blocks_to_copy = []
                    self._append_slots(seq_group, blocks_to_copy, True)
                    assert not blocks_to_copy
                else:
                    seq_group.init_multi_step_from_lookahead_slots(
                        num_lookahead_slots,
                        num_scheduler_steps=self.scheduler_config.
                        num_scheduler_steps,
                        is_multi_step=self.scheduler_config.is_multi_step,
                        enable_chunking=True,
                    )

            scheduled = ScheduledSequenceGroup(
                seq_group=seq_group, token_chunk_size=num_new_tokens)
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)
            self.prev_prompt = True
            return scheduled, ignored_seq_groups, True
        finally:
            budget.token_budget = original_token_budget

    def _schedule_be_prefill_chunks_for_saturation(
        self,
        decision,
        max_prefill_tokens: int,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        running_scheduled: SchedulerRunningOutputs,
        swapped_in: SchedulerSwappedInOutputs,
        partial_prefill_metadata: PartialPrefillMetadata,
        running_be_prefills: Deque[SequenceGroup],
        deferred_running: Iterable[SequenceGroup] = (),
    ) -> SchedulerPrefillOutputs:
        prefills = SchedulerPrefillOutputs(
            seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=True),
        )
        using_prompt_embeds: Optional[bool] = None
        fixed_chunk_size = int(getattr(
            self.scheduler_config, "p_pool_be_chunk_size", 0) or 0)

        while max_prefill_tokens > 0:
            candidate, is_running = self._pop_next_be_prefill_candidate(
                running_be_prefills)
            if candidate is None:
                break

            if (not is_running
                    and (self.decode_saturation.enabled
                         or self._decode_reserved_slots())):
                # P-pool scheduling retains its per-step budget without RT claims.
                # D also needs a strict cap on resident KV sequence slots.
                # A skipped decode or a just-restored group still owns GPU
                # KV, even though it is absent from this step's budget.
                # Count those residents once, including callers' deferred
                # queues, before allocating another BE prompt. Continuing an
                # already-resident prefill does not require a new slot.
                counted = budget._request_ids_num_curr_seqs
                resident_seqs = budget.num_curr_seqs
                for groups in (self.running, running_be_prefills,
                               deferred_running):
                    resident_seqs += sum(
                        group.get_max_num_running_seqs() for group in groups
                        if group.request_id not in counted)
                if (resident_seqs + self._decode_reserved_slots()
                        + candidate.get_max_num_running_seqs()
                        > self.scheduler_config.max_num_seqs):
                    self.waiting.appendleft(candidate)
                    break
            remaining_tokens = candidate.get_num_uncomputed_tokens()
            token_capacity = max(
                self.scheduler_config.max_num_batched_tokens
                - budget.num_batched_tokens, 0)
            max_chunk = min(max_prefill_tokens, remaining_tokens, token_capacity)
            max_chunk = self._find_be_prefill_chunk_by_latency(
                decision, max_chunk, running_scheduled, swapped_in,
                prefills.seq_groups, candidate)
            if max_chunk <= 0:
                if is_running:
                    running_be_prefills.appendleft(candidate)
                else:
                    self.waiting.appendleft(candidate)
                break
            if using_prompt_embeds is None:
                using_prompt_embeds = candidate.uses_prompt_embeds()

            scheduled, ignored, consumed_candidate = (
                self._schedule_selected_be_prefill_chunk(
                    candidate, max_chunk, budget, curr_loras,
                    partial_prefill_metadata, using_prompt_embeds,
                    is_running, running_scheduled.blocks_to_copy))
            prefills.ignored_seq_groups.extend(ignored)
            if scheduled is None:
                if not consumed_candidate:
                    if is_running:
                        running_be_prefills.appendleft(candidate)
                    else:
                        self.waiting.appendleft(candidate)
                    break
                continue

            prefills.seq_groups.append(scheduled)
            scheduled_tokens = scheduled.token_chunk_size or 0
            max_prefill_tokens = max(max_prefill_tokens - scheduled_tokens, 0)
            if fixed_chunk_size > 0:
                # Fixed-size mode runs exactly one BE chunk per step (the
                # local BE simulator's plain chunked-prefill semantics): a
                # request-completing tail chunk is NOT topped up with the
                # next request's tokens.
                break
            if scheduled_tokens < remaining_tokens:
                break

        return prefills

    def reserve_decode_request(self, reservation_id, prompt_tokens):
        if "_decode_reservations" not in self.__dict__:
            self._decode_reservations = DecodeReservations()
        return self._decode_reservations.reserve(
            reservation_id, prompt_tokens, self.cache_config.block_size,
            self._get_num_lookahead_slots(False, True),
            self.decode_saturation.total_gpu_blocks)

    def release_decode_reservation(self, reservation_id):
        claims = self.__dict__.get("_decode_reservations")
        if claims is not None:
            claims.release(reservation_id)

    def _decode_reserved_blocks(self):
        claims = self.__dict__.get("_decode_reservations")
        if claims is None:
            return 0
        return claims.blocks(self.cache_config.block_size,
                             self._get_num_lookahead_slots(False, True))

    def _decode_reserved_slots(self):
        claims = self.__dict__.get("_decode_reservations")
        return len(claims) if claims is not None else 0

    def _decode_saturation_required_free_blocks(self) -> int:
        """Return RT admission/append plus BE decode-append block demand."""
        lookahead = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=True)
        required = self._decode_reserved_blocks()
        has_rt_admission = bool(required)

        for seq_group in self.running:
            if not is_decode_phase(seq_group):
                continue
            required += (
                self.block_manager.get_num_blocks_touched_by_append_slots(
                    seq_group, lookahead))

        for seq_group in self.waiting_rt_decode_admission:
            if self.block_manager.can_allocate(seq_group) == AllocStatus.NEVER:
                continue
            required += self.block_manager.get_num_required_blocks(seq_group)
            has_rt_admission = True

        # can_allocate() requires the watermark to remain free after the last
        # admission. Append-only iterations do not need this extra reservation.
        if has_rt_admission:
            required += int(getattr(self.block_manager, "watermark_blocks", 0))
        return required

    def _smallest_running_be(self) -> Optional[SequenceGroup]:
        candidates = [
            seq_group for seq_group in self.running
            if self.decode_saturation.is_be(seq_group)
        ]
        if not candidates:
            return None

        def resident_blocks(seq_group: SequenceGroup) -> int:
            return sum(
                len(self.block_manager.get_block_table(seq))
                for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING)
            )

        return min(
            candidates,
            key=lambda seq_group: (
                resident_blocks(seq_group),
                seq_group.request_id,
            ),
        )

    def _decode_saturation_lacks_rt_sequence_capacity(self) -> bool:
        """Return whether queued RT decodes need BE sequence slots.

        KV capacity and sequence capacity are independent.  A BE population
        can occupy ``max_num_seqs`` while leaving enough free KV blocks for a
        new RT request.  In that case block-pressure eviction alone leaves the
        RT stuck in the direct-admission queue until a BE finishes naturally.
        """
        max_num_seqs = int(self.scheduler_config.max_num_seqs)
        if max_num_seqs <= 0:
            return False
        admissible_rt_seqs = self._decode_reserved_slots()
        for seq_group in self.waiting_rt_decode_admission:
            if self.block_manager.can_allocate(seq_group) == AllocStatus.NEVER:
                continue
            admissible_rt_seqs += seq_group.get_max_num_running_seqs()
        required_rt_slots = min(admissible_rt_seqs, max_num_seqs)
        running_seqs = sum(
            seq_group.get_max_num_running_seqs()
            for seq_group in self.running)
        return running_seqs + required_rt_slots > max_num_seqs

    def _prepare_decode_saturation_blocks(
        self, blocks_to_swap_out=None,
    ) -> int:
        """Swap BE KV to CPU for RT admission, append and early reservations."""
        if blocks_to_swap_out is None:
            blocks_to_swap_out = []
        swapped_groups = 0
        while (
            self.block_manager.get_num_free_gpu_blocks()
            < self._decode_saturation_required_free_blocks()
            or self._decode_saturation_lacks_rt_sequence_capacity()
        ):
            victim = self._smallest_running_be()
            if victim is None:
                break
            if not self.block_manager.can_swap_out(victim):
                if not self.__dict__.get("_decode_cpu_swap_exhausted", False):
                    logger.warning("CPU KV swap space exhausted; retaining BE KV "
                                   "and deferring RT capacity reclamation")
                self._decode_cpu_swap_exhausted = True
                break
            self._decode_cpu_swap_exhausted = False
            # Native swap updates block ownership while retaining KV on CPU.
            # The returned mapping must reach execute_worker before any new
            # model writes reuse these GPU blocks on the execution stream.
            self._swap_out(victim, blocks_to_swap_out)
            self.running.remove(victim)
            self.swapped.appendleft(victim)
            swapped_groups += 1
        return swapped_groups

    def _can_restore_decode_saturation_be(self, seq_group) -> bool:
        lookahead = self._get_num_lookahead_slots(False, True)
        if self.block_manager.can_swap_in(seq_group, lookahead) != AllocStatus.OK:
            return False
        # Restoring BE must leave every early RT claim and next-token append
        # available. Include a conservative next-token margin for this group.
        blocks = self.decode_saturation.seq_group_blocks(seq_group)
        append_margin = seq_group.get_max_num_running_seqs() * (
            (1 + lookahead + self.cache_config.block_size - 1)
            // self.cache_config.block_size)
        return (self.block_manager.get_num_free_gpu_blocks()
                >= blocks + append_margin
                + self._decode_saturation_required_free_blocks())

    def _restore_decode_saturation_be(
        self,
        decision,
        projected_be_blocks: int,
        reserved_running_seqs: int = 0,
        blocks_to_swap_in=None,
    ) -> int:
        """Restore BE KV from CPU using native swap-in mappings."""
        if blocks_to_swap_in is None:
            blocks_to_swap_in = []
        reserved_running_seqs += self._decode_reserved_slots()
        while self.swapped:
            seq_group = self.swapped[0]
            if not self.decode_saturation.is_be(seq_group):
                break
            running_seqs = reserved_running_seqs + sum(
                group.get_max_num_running_seqs()
                for group in self.running)
            group_seqs = seq_group.get_max_num_running_seqs()
            if (running_seqs + group_seqs
                    > self.scheduler_config.max_num_seqs):
                break
            group_blocks = self.decode_saturation.seq_group_blocks(seq_group)
            if projected_be_blocks + group_blocks > decision.be_block_budget:
                break
            if not self._can_restore_decode_saturation_be(seq_group):
                break
            self.swapped.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self.running.append(seq_group)
            projected_be_blocks += group_blocks
        return projected_be_blocks

    def _schedule_decode_saturation(self) -> SchedulerOutputs:
        blocks_to_swap_out = []
        blocks_to_swap_in = []
        num_swapped_out = self._prepare_decode_saturation_blocks(
            blocks_to_swap_out)

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()
        self.decode_saturation.time_predictor = self.time_predictor
        decision = self.decode_saturation.make_decision(
            list(self.running),
            list(self.waiting),
            list(self.waiting_rt_decode_admission)
            + list(self.waiting_be_decode_admission),
            time.time(),
        )
        reserved_blocks = self._decode_reserved_blocks()
        if reserved_blocks:
            budget_blocks = max(decision.be_block_budget - reserved_blocks, 0)
            decision = replace(
                decision, be_block_budget=budget_blocks,
                max_be_prefill_tokens=max(
                    budget_blocks - decision.current_be_blocks, 0)
                * self.cache_config.block_size)
        projected_be_blocks = self._admit_decode_admission_queues(decision)

        partial_prefill_metadata = PartialPrefillMetadata.from_queues(
            running=self.running,
            waiting=self.waiting,
            scheduler_config=self.scheduler_config,
        )
        running_scheduled = SchedulerRunningOutputs.create_empty()
        running_scheduled.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=True)
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        running_queue = self.running
        self.decode_saturation.order_running_queue(running_queue)
        self.running = deque()
        running_be_prefills: Deque[SequenceGroup] = deque()

        while running_queue:
            seq_group = running_queue.popleft()
            is_decode = is_decode_phase(seq_group)
            if (self.decode_saturation.is_be(seq_group)
                    and not is_decode):
                running_be_prefills.append(seq_group)
                continue
            if not is_decode:
                self.running.append(seq_group)
                continue

            if (self.decode_saturation.is_be(seq_group)
                    and not self._can_execute_running_be_decode(
                        decision, running_scheduled.decode_seq_groups, seq_group)):
                self.running.append(seq_group)
                continue

            scheduled = self._schedule_decode_for_saturation(
                seq_group,
                budget,
                curr_loras,
                running_scheduled.blocks_to_copy,
            )
            if scheduled is None:
                self.running.append(seq_group)
                continue
            running_scheduled.decode_seq_groups.append(scheduled)
            running_scheduled.decode_seq_groups_list.append(seq_group)

        # Do not immediately re-admit a group evicted by this decision cycle.
        # It remains at the queue head and is considered first on the next step.
        if num_swapped_out == 0:
            reserved_running_seqs = sum(
                seq_group.get_max_num_running_seqs()
                for seq_group in running_be_prefills
            ) + sum(
                seq_group.get_max_num_running_seqs()
                for seq_group in running_scheduled.decode_seq_groups_list
            )
            projected_be_blocks = self._restore_decode_saturation_be(
                decision,
                projected_be_blocks,
                reserved_running_seqs=reserved_running_seqs,
                blocks_to_swap_in=blocks_to_swap_in,
            )

        max_be_prefill_tokens = max(
            decision.be_block_budget - projected_be_blocks, 0
        ) * self.cache_config.block_size
        prefills = self._schedule_be_prefill_chunks_for_saturation(
            decision,
            max_be_prefill_tokens,
            budget,
            curr_loras,
            running_scheduled,
            swapped_in,
            partial_prefill_metadata,
            running_be_prefills,
        )

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        self.running.extend(running_be_prefills)
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        assert sum(
            seq_group.get_max_num_running_seqs()
            for seq_group in self.running
        ) <= self.scheduler_config.max_num_seqs

        scheduled_seq_groups = (prefills.seq_groups +
                                running_scheduled.decode_seq_groups)
        num_prefill_groups = len(prefills.seq_groups)
        all_prefills = len(scheduled_seq_groups) == num_prefill_groups
        num_lookahead_slots = (0 if
                               (all_prefills
                                and not self.scheduler_config.is_multi_step)
                               else running_scheduled.num_lookahead_slots)
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=num_swapped_out,
        )

    def _order_finishing_prefills_first(
        self, scheduled_prefill_seqs: List[ScheduledSequenceGroup]
    ) -> List[SequenceGroup]:
        """Returns a list of prefilling SequenceGroups where sequences that are
        scheduled to finish prefilling are listed first"""
        finishing = [
            s.seq_group for s in scheduled_prefill_seqs
            if s.seq_group.get_num_uncomputed_tokens() == s.token_chunk_size
        ]
        not_finishing = [
            s.seq_group for s in scheduled_prefill_seqs
            if s.seq_group.get_num_uncomputed_tokens() != s.token_chunk_size
        ]
        return finishing + not_finishing

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if getattr(self.scheduler_config, "enable_p_pool_saturation", False):
            return self._schedule_prefill_saturation()
        if getattr(
                self.scheduler_config,
                "enable_fixed_budget_rt_prefill_priority",
                False):
            return self._schedule_fixed_budget_rt_prefill_priority()
        if self.decode_saturation.enabled:
            return self._schedule_decode_saturation()
        # Phase-isolated D-pool baselines may inject already-prefilled RT
        # requests directly even when BE-filling decode saturation is off.
        # Admit those requests before entering the ordinary vLLM scheduler;
        # otherwise they remain forever in the dedicated admission queue,
        # which used to be drained only by _schedule_decode_saturation().
        if self.waiting_rt_decode_admission:
            rt_fully_admitted = self._admit_rt_decode_admission_queue()
        else:
            rt_fully_admitted = True
        waiting_be_decode = getattr(
            self, "waiting_be_decode_admission", ())
        if (rt_fully_admitted and waiting_be_decode
                and self._is_direct_decode(waiting_be_decode[0])):
            self._admit_unconstrained_decode_admission_queue(
                self.waiting_be_decode_admission)
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup,
                          enable_chunking: bool) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        is_prefill = seq_group.is_prefill()
        num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        if is_prefill and num_lookahead_slots > 0:
            # Appending prefill slots only happens multi-step and
            # chunked-prefill are enabled together.
            assert self.scheduler_config.is_multi_step and enable_chunking

        return self.block_manager.can_append_slots(
            seq_group=seq_group, num_lookahead_slots=num_lookahead_slots)

    def _allow_async_output_proc(self, seq_group: SequenceGroup) -> bool:
        # async_output_proc is allowed only when we have a single sequence
        # in the sequence group
        no_single_seq = seq_group.sampling_params is None or (
            seq_group.sampling_params.n == 1)
        return no_single_seq

    def schedule(
            self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()
        
        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # print(f"num of schedued prefill seq: {scheduler_outputs.num_prefill_groups}")

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            seq_group_metadata = self._seq_group_metadata_cache[
                self.cache_id].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            if seq_group.is_encoder_decoder():
                # Encoder associated with SequenceGroup
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                # Block table for cross-attention
                # Also managed at SequenceGroup level
                cross_block_table = self.block_manager.get_cross_block_table(
                    seq_group)
            else:
                encoder_seq_data = None
                cross_block_table = None

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + num_computed_tokens
                        < seqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    encoder_seq_data=encoder_seq_data,
                    cross_block_table=cross_block_table,
                    state=seq_group.state,
                    token_type_ids=seq_group.token_type_ids,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=(seq_group.multi_modal_data
                                      if scheduler_outputs.num_prefill_groups
                                      > 0 else None),
                    multi_modal_placeholders=(
                        seq_group.multi_modal_placeholders
                        if scheduler_outputs.num_prefill_groups > 0 else None),
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                )
            else:
                # When SPMD mode is enabled, we only send delta data except for
                # the first request to reduce serialization cost.
                seq_data_delta = {}
                for id, data in seq_data.items():
                    seq_data_delta[id] = data.get_delta_and_reset()
                seq_group_metadata = SequenceGroupMetadataDelta(
                    seq_data_delta,
                    seq_group.request_id,
                    block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=token_chunk_size,
                    computed_block_nums=common_computed_block_nums,
                )
            seq_group_metadata_list.append(seq_group_metadata)

            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(
                    seq_group)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group,
                scheduled_seq_group.token_chunk_size)

        self._seq_group_metadata_cache[self.next_cache_id].reset()

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time

        # Move to next cache (if exists)
        self.cache_id = self.next_cache_id

        # block_table_len = []
        # for seq in seq_group_metadata_list:
        #     for seq, block_table in seq.block_tables.items():
        #         block_table_len.append(len(block_table))
        # # block_table_len = [len(seq.block_tables.items()) for seq in seq_group_metadata_list]
        # print(f"block_table_len_:{block_table_len}")
        # print(f"token info: {[seq.token_chunk_size for seq in seq_group_metadata_list]} | req_ids: {[seq.request_id for seq in seq_group_metadata_list]}")

        # Return results
        return (seq_group_metadata_list, scheduler_outputs,
                allow_async_output_proc)

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def remove_seq_from_computed_blocks_tracker(
            self, seq_group: SequenceGroup,
            status: Optional[SequenceStatus]) -> None:
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            self._remove_seq_from_computed_blocks_tracker(seq)

    def _remove_seq_from_computed_blocks_tracker(self, seq: Sequence) -> None:
        """
        Free a sequence computed blocks tracker _seq_id_to_blocks_hashes
        and _seq_id_to_num_tokens_computed.
        """
        self.block_manager.remove_seq_from_computed_blocks_tracker(seq)

    def _free_finished_seqs(self, seq_group: SequenceGroup) -> None:
        """Free finished seqs in a sequence group."""
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                self.free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        if seq_group.is_finished():
            self.decode_saturation.observe_rt_finish(seq_group)
            # Free cross-attention block table, if it exists
            self._free_seq_group_cross_attn_blocks(seq_group)

            # Add the finished requests to the finished requests list.
            # This list will be used to update the Mamba cache in the
            # next step.
            self._finished_requests_ids.append(seq_group.request_id)

        # Free finished seqs
        self._free_finished_seqs(seq_group)

    def free_finished_seq_groups(self) -> None:
        remaining: Deque[SequenceGroup] = deque()
        for seq_group in self.running:
            self._free_finished_seq_group(seq_group)
            if not seq_group.is_finished():
                remaining.append(seq_group)

        self.running = remaining

        # Handle async stopped sequence groups
        # (ones that reached max model len)
        if self._async_stopped:
            for seq_group in self._async_stopped:
                self._free_seq_group_cross_attn_blocks(seq_group)
                self._finished_requests_ids.append(seq_group.request_id)

                # Free finished seqs
                self._free_finished_seqs(seq_group)

            self._async_stopped.clear()

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: List[Tuple[int, int]],
        enable_chunking: bool = False,
    ) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
            enable_chunking (bool): True if chunked prefill is enabled.
        """
        is_prefill: bool = seq_group.is_prefill()
        num_lookahead_slots: int = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        seq_group.init_multi_step_from_lookahead_slots(
            num_lookahead_slots,
            num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
            is_multi_step=self.scheduler_config.is_multi_step,
            enable_chunking=enable_chunking,
        )

        seq_status: Optional[SequenceStatus] = SequenceStatus.RUNNING
        if self.scheduler_config.is_multi_step and enable_chunking:
            # In multi-step chunked-prefill any sequence type can have
            # slots appended.
            seq_status = None

        for seq in seq_group.get_seqs(status=seq_status):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            if len(cows) > 0:
                blocks_to_copy.extend(cows)

    def _preempt(self, seq_group: SequenceGroup,
                 blocks_to_swap_out: List[Tuple[int, int]]) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if self.user_specified_preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        elif self.user_specified_preemption_mode == "swap":
            preemption_mode = PreemptionMode.SWAP
        else:
            preemption_mode = PreemptionMode.RECOMPUTE

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d",
                seq_group.request_id,
                preemption_mode,
                self.num_cumulative_preemption + 1,
            )
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()
        self._free_seq_group_cross_attn_blocks(seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = ((now - earliest_arrival_time)
                            > (self.scheduler_config.delay_factor *
                               self.last_prompt_latency) or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool,
                                 enable_chunking: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.

        When chunking is enabled with multi-step, we allocate lookahead slots
        for the prefills for when the prefills turn into decodes in the first
        step.
        """
        if is_prefill:
            if self.scheduler_config.is_multi_step and enable_chunking:
                # num_lookahead_slots was introduced in the context of decodes,
                # in Speculative Decoding.
                # When the num_scheduler_steps is 8, say, then the
                # num_lookahead_slots is 7. Meaning, we are doing a 1-step of
                # decode anyways and we wish to do 7 more.
                #
                # "lookaheads" for prefills, is introduced in support for
                # Chunked-Prefill in Multi-Step.
                return self.scheduler_config.num_lookahead_slots + 1
            else:
                return 0

        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_uncached_and_cached_tokens(
        self,
        seq_group: SequenceGroup,
        status: SequenceStatus,
        enable_chunking: bool,
        budget: SchedulingBudget,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> Tuple[int, int]:
        """
        Returns the number of new uncached and cached tokens to schedule for a
        given sequence group that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.

        Returns (0, 0) if the new token cannot be computed due to token budget.

        The cached tokens's blocks are already computed, and the attention
        backend will reuse the cached blocks rather than recomputing them. So
        the scheduler could schedule these cached tokens "for free".

        Args:
            seq_group: The sequence group to get the number of new tokens to
                schedule.
            status: The status of the sequences to get the number of new tokens
                to schedule.
            enable_chunking: Whether to chunk the number of tokens to compute.
            budget: The budget to chunk the number of tokens to compute.
            partial_prefill_metadata: information about the partial prefills
                that are currently running


        Returns:
            A tuple of two ints. The first int is the number of new uncached
            tokens to schedule. The second int is the number of cached tokens.
            If no more new tokens can be scheduled, returns (0, 0).
        """
        num_cached_new_tokens = 0
        num_uncached_new_tokens = 0

        seqs = seq_group.get_seqs(status=status)
        # Compute the number of new uncached and cached tokens for
        # each sequence.
        for seq in seqs:
            if not seq.is_prefill():
                # Decode sequences should always just have 1 uncached token
                # TODO(rickyx): Actually is this still correct for multi-step?
                num_uncached_new_tokens += 1
                continue

            num_computed_tokens_seq = seq.get_num_computed_tokens()
            all_num_new_tokens_seq = seq.get_len() - num_computed_tokens_seq
            if not self.cache_config.enable_prefix_caching:
                # If prefix caching is not enabled, all new tokens are uncached.
                num_uncached_new_tokens += all_num_new_tokens_seq
                continue

            # NOTE: the cache token might be currently in a block that's in an
            # evictor meaning that it's not yet allocated. However, we don't
            # exclude such tokens in the cache count because it will be
            # guaranteed to be allocated later if the sequence can be allocated.
            num_cached_tokens_seq = self.block_manager.get_num_cached_tokens(
                seq)

            # Sanity check.
            if num_cached_tokens_seq < num_computed_tokens_seq:
                # This should only happen with chunked prefill, and
                # the seq is still in prefill. The `num_cached_tokens_seq`
                # is the value we calculated on scheduling the first prefill.
                # For subsequent continuous prefill steps, we cached the
                # number of cache tokens for the sequence so the cached token
                # count could be less than the number of computed tokens.
                # See comments on `ComputedBlocksTracker` for more details.
                assert (
                    seq.is_prefill() and seq.status == SequenceStatus.RUNNING
                    and self.scheduler_config.chunked_prefill_enabled
                ), ("Number of cached tokens should not be less than the "
                    "number of computed tokens for a sequence that's still "
                    f"in prefill. But there are {num_cached_tokens_seq} cached "
                    f"tokens and {num_computed_tokens_seq} computed tokens "
                    f"for sequence {seq.seq_id}.")

            num_cached_new_tokens_seq = max(
                0, num_cached_tokens_seq - num_computed_tokens_seq)
            num_uncached_new_tokens_seq = (all_num_new_tokens_seq -
                                           num_cached_new_tokens_seq)

            num_uncached_new_tokens += num_uncached_new_tokens_seq
            num_cached_new_tokens += num_cached_new_tokens_seq

        if num_uncached_new_tokens == 0 and num_cached_new_tokens > 0:
            # For a fully cached hit sequence, we actually need to recompute the
            # last token. So we need at least 1 uncached token to schedule.
            # See ModelRunner._compute_for_prefix_cache_hit for more details.
            num_uncached_new_tokens = 1
            num_cached_new_tokens -= 1

        if enable_chunking and len(seqs) == 1:
            # Chunk if a running request cannot fit in the given budget.
            # If number of seq > 1, it means it is doing beam search
            # in a decode phase. Do not chunk.
            num_uncached_new_tokens = self._chunk_new_tokens_to_schedule(
                self.scheduler_config,
                self.cache_config,
                budget,
                self._get_prompt_limit(seq_group),
                num_uncached_new_tokens,
                self.partial_prefill_budget_lookup_list,
                partial_prefill_metadata,
            )

        return num_uncached_new_tokens, num_cached_new_tokens

    @staticmethod
    def _chunk_new_tokens_to_schedule(
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        budget: SchedulingBudget,
        prompt_limit: int,
        num_new_tokens: int,
        partial_prefill_budget_lookup_list: List[int],
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> int:
        """
        Chunks the number of new tokens to schedule based on the budget when
        chunked prefill is enabled.

        Args:
            scheduler_config: The scheduler config.
            cache_config: The cache config.
            budget: The budget to chunk the number of tokens to compute.
            prompt_limit: The maximum number of tokens allowed in a prompt.
            num_new_tokens: The number of new tokens to schedule.

        Returns:
            The number of new tokens to schedule after chunking.
        """
        remaining_token_budget = budget.remaining_token_budget()
        if scheduler_config.is_multi_step:
            # The current multi-step + chunked prefill capability does
            # not actually support chunking prompts.
            #
            # Therefore, `num_new_tokens` is computed in the same fashion
            # for both multi-step+chunked-prefill &
            # multi-step+chunked-prefill+APC
            #
            # Prompts with more tokens than the current remaining budget
            # are postponed to future scheduler steps
            if num_new_tokens > prompt_limit:
                # If the seq_group is in prompt-stage, pass the
                # num_new_tokens as-is so the caller can ignore
                # the sequence.
                return num_new_tokens

            return 0 if num_new_tokens > \
                remaining_token_budget else num_new_tokens

        # Get the number of tokens to allocate to this prefill slot
        prefill_slot_budget = (
            remaining_token_budget if partial_prefill_metadata is None else
            partial_prefill_budget_lookup_list[
                partial_prefill_metadata.schedulable_prefills])

        if cache_config.enable_prefix_caching:
            # When prefix caching is enabled and we're partially prefilling
            # a sequence, we always allocate a number of new tokens that is
            # divisible by the block size to avoid partial block matching.
            block_size = cache_config.block_size
            # Don't exceed either the total budget or slot budget.
            # Take min of those and get the next lowest multiple of the
            # block size:
            remaining_token_budget = (
                min(remaining_token_budget, prefill_slot_budget) //
                block_size) * block_size
            # NB: In the case where num_new_tokens < budget, we are
            # finishing prefill for this sequence, so we do not need to
            # allocate a full block.

        num_new_tokens = min(num_new_tokens, remaining_token_budget,
                             prefill_slot_budget)

        return num_new_tokens
