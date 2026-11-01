# SPDX-License-Identifier: Apache-2.0
"""
DummyConnector: A no-op KV connector for experimental PD disaggregation.

When configured as the KV connector for a decode-only pool, this connector
skips actual KV cache transfer and returns dummy hidden states so that the
model forward pass is bypassed during the "prefill" step.  The request then
transitions immediately into decode state with whatever garbage data is
already in the allocated KV blocks.

This is useful for scheduling/timing experiments where the correctness of
generated text is not required but we need realistic block allocation and
decode-phase timing.
"""
from typing import TYPE_CHECKING, Union

import os

import torch

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class DummyConnector(KVConnectorBase):
    """No-op KV connector that bypasses model forward on prefill."""

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):
        self.hidden_size: int = config.model_config.get_hidden_size()
        self.dtype = config.model_config.dtype
        logger.info(
            "DummyConnector initialized (hidden_size=%d, dtype=%s). "
            "Prefill forward will be bypassed with dummy hidden states.",
            self.hidden_size, self.dtype)

    def close(self) -> None:
        pass

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        # D-pool never sends KV; this is a no-op.
        pass

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        """Return dummy hidden states and bypass model forward.

        The returned hidden states are zeros of shape
        (num_prefill_tokens, hidden_size).  The sampler will produce an
        arbitrary token, after which the scheduler moves the request into
        decode state.  Subsequent decode steps use paged attention against
        the (garbage) KV blocks already allocated by the block manager.
        """
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        device = model_input.input_tokens.device

        # A non-last PP stage must still participate in the pipeline handoff.
        # The next stage also uses DummyConnector and ignores these values, so
        # create the model-specific intermediate structure without running any
        # layers.  Returning a plain tensor here violates WorkerBase's PP
        # contract and aborts before the first real T-pool iteration.
        if not get_pp_group().is_last_rank:
            intermediate_tensors = (
                model_executable.make_empty_intermediate_tensors(
                    batch_size=num_prefill_tokens,
                    dtype=self.dtype,
                    device=device,
                ))
            return intermediate_tensors, True, model_input

        # Create dummy hidden states - zeros will produce arbitrary logits
        # but for timing experiments correctness doesn't matter.
        if os.environ.get(
                "SATURO_DUMMY_CONNECTOR_BROADCAST_HIDDEN", "0") == "1":
            # The values are all identical and LogitsProcessor selects only
            # the final prompt token of each sequence before the LM head.
            # An expanded one-row view therefore has the same semantics while
            # avoiding a transient O(num_prefill_tokens * hidden_size)
            # allocation in large controlled decode-batch experiments.
            hidden_states = torch.zeros(
                1,
                self.hidden_size,
                dtype=self.dtype,
                device=device,
            ).expand(num_prefill_tokens, -1)
        else:
            hidden_states = torch.zeros(
                num_prefill_tokens,
                self.hidden_size,
                dtype=self.dtype,
                device=device,
            )

        logger.debug(
            "DummyConnector: bypassing prefill forward for %d tokens",
            num_prefill_tokens)
        _dbg_kv_file = os.environ.get("SATURO_DEBUG_KV_RECV_FILE", "")
        if _dbg_kv_file:
            with open(_dbg_kv_file, "a") as _f:
                _f.write(
                    "DUMMY_CONNECTOR bypassed prefill forward for %d tokens "
                    "(no real prefill computed)\n" % num_prefill_tokens)

        return hidden_states, True, model_input
