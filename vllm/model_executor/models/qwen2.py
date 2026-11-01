# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from collections.abc import Iterable
import asyncio
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .adapters import as_seq_cls_model
from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

ModuleEvents = Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]]


def _event_start(module_events: Optional[ModuleEvents], name: str):
    if module_events is None:
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return name, start, end


def _event_end(module_events: Optional[ModuleEvents], record) -> None:
    if module_events is None or record is None:
        return
    name, start, end = record
    end.record()
    module_events.setdefault(name, []).append((start, end))


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self,
                x,
                module_events: Optional[ModuleEvents] = None):
        event = _event_start(module_events, "mlp.gate_up_proj")
        gate_up, _ = self.gate_up_proj(x)
        _event_end(module_events, event)

        event = _event_start(module_events, "mlp.act_fn")
        x = self.act_fn(gate_up)
        _event_end(module_events, event)

        event = _event_start(module_events, "mlp.down_proj")
        x, _ = self.down_proj(x)
        _event_end(module_events, event)
        return x


class Qwen2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[tuple] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            } if dual_chunk_attention_config else {})

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        module_events: Optional[ModuleEvents] = None,
    ) -> torch.Tensor:
        event = _event_start(module_events, "self_attn.qkv_proj")
        qkv, _ = self.qkv_proj(hidden_states)
        _event_end(module_events, event)

        event = _event_start(module_events, "self_attn.qkv_split")
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        _event_end(module_events, event)

        event = _event_start(module_events, "self_attn.rotary_emb")
        q, k = self.rotary_emb(positions, q, k)
        _event_end(module_events, event)

        event = _event_start(module_events, "self_attn.attn")
        attn_output = self.attn(q, k, v)
        _event_end(module_events, event)

        event = _event_start(module_events, "self_attn.o_proj")
        output, _ = self.o_proj(attn_output)
        _event_end(module_events, event)
        return output

    def forward_with_time_record(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        if attn_events is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        attn_output = self.attn(q, k, v)

        if attn_events is not None:
            end_event.record()
            attn_events.append((start_event, end_event))

        output, _ = self.o_proj(attn_output)
        return output

    def forward_no_attn_only(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        output, _ = self.o_proj(q)
        return output

    def forward_attn_only(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        if events is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        attn_output = self.attn(q, k, v)
        if events is not None:
            end_event.record()
            events.append((start_event, end_event))

        output, _ = self.o_proj(attn_output)
        return output
                

class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        module_events: Optional[ModuleEvents] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # print(f"execute layer!!!")
        event = _event_start(module_events, "input_layernorm")
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        _event_end(module_events, event)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            module_events=module_events,
        )

        # Fully Connected
        event = _event_start(module_events, "post_attention_layernorm")
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        _event_end(module_events, event)

        hidden_states = self.mlp(hidden_states, module_events=module_events)
        return hidden_states, residual

    def forward_with_time_record(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        layer_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
        attn_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_events is not None:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn.forward_with_time_record(
            positions=positions,
            hidden_states=hidden_states,
            attn_events=attn_events,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if layer_events is not None:
            end_evt.record()
            layer_events.append((start_evt, end_evt))

        return hidden_states, residual

    def forward_linear_only(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if events is not None:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn.forward_no_attn_only(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if events is not None:
            end_evt.record()
            events.append((start_evt, end_evt))

        return hidden_states, residual
    
    def forward_attn_only(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn.forward_attn_only(
            hidden_states=hidden_states,
            positions=positions,
            events=events,
        )
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen2Model(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                ))

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(config=config,
                                              cache_config=cache_config,
                                              quant_config=quant_config,
                                              prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.enable_p_pool_preemption = (
            getattr(vllm_config.scheduler_config,
                    "enable_p_pool_saturation", False))
        self.preemption_event = (
            threading.Event() if self.enable_p_pool_preemption else None)
        self.preempted = False

    def _should_preempt(self, device: torch.device) -> bool:
        if not self.enable_p_pool_preemption or self.preemption_event is None:
            return False

        preempt_requested = self.preemption_event.is_set()
        if get_tensor_model_parallel_world_size() > 1:
            flag_tensor = torch.tensor([1.0 if preempt_requested else 0.0],
                                       dtype=torch.float32,
                                       device=device)
            # This collective is also the TP readiness barrier: no rank exits
            # until every rank reaches the same layer boundary.
            flag_tensor = get_tp_group().all_reduce(flag_tensor)
            preempt_requested = flag_tensor.item() > 0

        if preempt_requested:
            self.preempted = True
            self.preemption_event.clear()
            return True
        return False

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        module_events: Optional[ModuleEvents] = None,
        profile_num_layers: Optional[int] = None,
        profile_layer_start: int = 0,
    ) -> Optional[Union[torch.Tensor, IntermediateTensors]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        layers = self.layers[self.start_layer:self.end_layer]
        if profile_num_layers is not None:
            start = max(0, int(profile_layer_start))
            end = start + max(0, int(profile_num_layers))
            layers = layers[start:end]
        for layer in layers:
            if self._should_preempt(hidden_states.device):
                return None

            # # # 使用event来记录attn开销时间
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)        
            # start_event.record()
            
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                module_events=module_events,
            )

            # end_event.record()
            # torch.cuda.synchronize()
            # elapsed_time_ms = start_event.elapsed_time(end_event)
            # print(f"shape of Hidden state:{hidden_states.shape} | Layer time: {elapsed_time_ms} ms")


        if profile_num_layers is not None:
            return hidden_states
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    async def forward_async(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        module_events: Optional[ModuleEvents] = None,
        profile_num_layers: Optional[int] = None,
        profile_layer_start: int = 0,
    ) -> Optional[Union[torch.Tensor, IntermediateTensors]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        layers = self.layers[self.start_layer:self.end_layer]
        if profile_num_layers is not None:
            start = max(0, int(profile_layer_start))
            end = start + max(0, int(profile_num_layers))
            layers = layers[start:end]
        for layer in layers:
            await asyncio.sleep(0)
            sleep_ms = float(os.environ.get(
                "VLLM_P_POOL_PREEMPT_LAYER_SLEEP_MS", "0") or 0)
            if sleep_ms > 0:
                await asyncio.sleep(sleep_ms / 1000.0)
            if self._should_preempt(hidden_states.device):
                return None

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                module_events=module_events,
            )

        if profile_num_layers is not None:
            return hidden_states
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_with_time_record(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        layer_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
        attn_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer.forward_with_time_record(
                positions,
                hidden_states,
                residual,
                layer_events=layer_events,
                attn_events=attn_events,
            )
            # hidden_states, residual = layer.forward_linear_only(
            #     positions,
            #     hidden_states,
            #     residual,
            #     events=layer_events,
            # )


        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_attn_only(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        layer_to_run: int = 5,
        events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> Optional[Union[torch.Tensor, IntermediateTensors]]:

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.start_layer+layer_to_run]:
            hidden_states, residual = layer.forward_attn_only(
                positions,
                hidden_states,
                residual,
                events=events,
            )

        return hidden_states

    def forward_linear_only(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        layer_to_run: int = 5,
        events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Linear-only measures only MLP part; needs embedding + attn first to get hidden_states
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.start_layer+layer_to_run]:
            hidden_states, residual = layer.forward_linear_only(
                positions,
                hidden_states,
                residual,
                events=events,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        module_events: Optional[ModuleEvents] = None,
        profile_num_layers: Optional[int] = None,
        profile_layer_start: int = 0,
    ) -> Optional[Union[torch.Tensor, IntermediateTensors]]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, module_events=module_events,
                                   profile_num_layers=profile_num_layers,
                                   profile_layer_start=profile_layer_start)
        return hidden_states

    async def forward_async(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        module_events: Optional[ModuleEvents] = None,
        profile_num_layers: Optional[int] = None,
        profile_layer_start: int = 0,
    ) -> Optional[Union[torch.Tensor, IntermediateTensors]]:
        return await self.model.forward_async(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            module_events=module_events,
            profile_num_layers=profile_num_layers,
            profile_layer_start=profile_layer_start)

    def forward_with_time_record(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        layer_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
        attn_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if layer_events is None:
            layer_events = []
        if attn_events is None:
            attn_events = []

        hidden_states = self.model.forward_with_time_record(
            input_ids, positions, intermediate_tensors, inputs_embeds,
            layer_events=layer_events, attn_events=attn_events
        )

        # Optional per-step dump of the recorded timings. The serving path
        # otherwise discards the events; SATURO_FORWARD_TIME_DUMP=<path>
        # appends one JSON line per forward with the layer/attn/linear split
        # (linear = whole-layer minus attention submodule). Costs one
        # cuda.synchronize per step, so enable it only for diagnostics.
        dump_path = os.environ.get("SATURO_FORWARD_TIME_DUMP", "")
        if dump_path and layer_events:
            import json as _json
            torch.cuda.synchronize()
            layer_ms = [s.elapsed_time(e) for s, e in layer_events]
            attn_ms = [s.elapsed_time(e) for s, e in attn_events]
            record = {
                "n_tokens": int(input_ids.numel()),
                "num_layers": len(layer_ms),
                "layer_total_ms": float(sum(layer_ms)),
                "attn_total_ms": float(sum(attn_ms)),
                "linear_total_ms": float(sum(layer_ms) - sum(attn_ms)),
                "ts": time.time(),
            }
            try:
                with open(f"{dump_path}.{os.getpid()}", "a",
                          encoding="utf-8") as f:
                    f.write(_json.dumps(record) + "\n")
            except OSError:
                pass
        return hidden_states

    def forward_attn_only(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        num_layer_to_run: int = 5,
        events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model.forward_attn_only(
            input_ids, positions, intermediate_tensors, inputs_embeds,
            num_layer_to_run, events=events
        )
        return hidden_states

    def forward_linear_only(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        num_layer_to_run: int = 5,
        events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model.forward_linear_only(
            input_ids, positions, intermediate_tensors, inputs_embeds, num_layer_to_run, events=events
        )
        return hidden_states


    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


Qwen2ForSequenceClassification = as_seq_cls_model(Qwen2ForCausalLM)
