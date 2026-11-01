import asyncio
import threading

import torch
from torch import nn

from vllm.model_executor.models import mistral3 as mistral3_module
from vllm.model_executor.models.mistral3 import (
    Mistral3ForConditionalGeneration, Mistral3MultiModalProjector)
from vllm.worker.worker_base import LocalOrDistributedWorkerBase


class _DecoderProbe(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.preemption_event = None
        self.preempted = False
        self.calls = []

    def forward(self, input_ids, positions, intermediate_tensors, **kwargs):
        self.calls.append(
            ("forward", input_ids, positions, intermediate_tensors, kwargs))
        return "forward-output"

    async def forward_async(self, input_ids, positions, intermediate_tensors,
                            **kwargs):
        self.calls.append(("forward_async", input_ids, positions,
                           intermediate_tensors, kwargs))
        return "async-output"

    def forward_with_time_record(self, input_ids, positions,
                                 intermediate_tensors, inputs_embeds,
                                 **kwargs):
        self.calls.append(("forward_with_time_record", input_ids, positions,
                           intermediate_tensors, {
                               "inputs_embeds": inputs_embeds,
                               **kwargs,
                           }))
        return "timing-output"

    def forward_attn_only(self, input_ids, positions, intermediate_tensors,
                          inputs_embeds, **kwargs):
        self.calls.append(
            ("forward_attn_only", input_ids, positions, intermediate_tensors, {
                "inputs_embeds": inputs_embeds,
                **kwargs,
            }))
        return "attn-output"

    def forward_linear_only(self, input_ids, positions, intermediate_tensors,
                            inputs_embeds, **kwargs):
        self.calls.append(("forward_linear_only", input_ids, positions,
                           intermediate_tensors, {
                               "inputs_embeds": inputs_embeds,
                               **kwargs,
                           }))
        return "linear-output"


class _LanguageModelProbe(nn.Module):

    def __init__(self, decoder: _DecoderProbe) -> None:
        super().__init__()
        self.model = decoder


class _PatchMergerProbe(nn.Module):

    def forward(self, image_features, image_sizes):
        return image_features


class _ParallelLinearProbe(nn.Module):

    def forward(self, hidden_states):
        return hidden_states, None


def _make_model() -> tuple[Mistral3ForConditionalGeneration, _DecoderProbe]:
    model = Mistral3ForConditionalGeneration.__new__(
        Mistral3ForConditionalGeneration)
    nn.Module.__init__(model)
    decoder = _DecoderProbe()
    model.language_model = _LanguageModelProbe(decoder)
    return model, decoder


def test_mistral3_projector_records_module_events(monkeypatch) -> None:
    event_names = []

    def event_start(module_events, name):
        assert module_events is not None
        event_names.append(name)
        return name

    monkeypatch.setattr(mistral3_module, "_event_start", event_start)
    monkeypatch.setattr(mistral3_module, "_event_end",
                        lambda module_events, record: None)

    projector = Mistral3MultiModalProjector.__new__(
        Mistral3MultiModalProjector)
    nn.Module.__init__(projector)
    projector.norm = nn.Identity()
    projector.patch_merger = _PatchMergerProbe()
    projector.linear_1 = _ParallelLinearProbe()
    projector.act = nn.Identity()
    projector.linear_2 = _ParallelLinearProbe()

    output = projector(
        torch.ones(2, 4),
        torch.tensor([[1, 2]]),
        module_events={},
    )

    assert output.shape == (2, 4)
    assert event_names == [
        "multi_modal_projector.norm",
        "multi_modal_projector.patch_merger",
        "multi_modal_projector.linear_1",
        "multi_modal_projector.act",
        "multi_modal_projector.linear_2",
    ]


def test_mistral3_exposes_decoder_to_worker_preemption() -> None:
    model, decoder = _make_model()
    worker = LocalOrDistributedWorkerBase.__new__(LocalOrDistributedWorkerBase)
    worker.get_model = lambda: model

    assert model.model is decoder
    assert LocalOrDistributedWorkerBase.set_p_pool_preemption(worker, True)
    assert isinstance(decoder.preemption_event, threading.Event)
    assert decoder.preemption_event.is_set()

    assert LocalOrDistributedWorkerBase.set_p_pool_preemption(worker, False)
    assert not decoder.preemption_event.is_set()


def test_mistral3_forwards_timing_and_profile_controls() -> None:
    model, decoder = _make_model()
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    intermediate_tensors = object()
    module_events = {}

    output = model.forward(
        input_ids,
        positions,
        intermediate_tensors=intermediate_tensors,
        module_events=module_events,
        profile_num_layers=3,
        profile_layer_start=2,
    )

    assert output == "forward-output"
    name, _, _, actual_intermediate_tensors, kwargs = decoder.calls[-1]
    assert name == "forward"
    assert actual_intermediate_tensors is intermediate_tensors
    assert kwargs["module_events"] is module_events
    assert kwargs["profile_num_layers"] == 3
    assert kwargs["profile_layer_start"] == 2

    timing_output = model.forward_with_time_record(
        input_ids,
        positions,
        intermediate_tensors=intermediate_tensors,
    )
    assert timing_output == "timing-output"

    attn_output = model.forward_attn_only(
        input_ids,
        positions,
        intermediate_tensors=intermediate_tensors,
        num_layer_to_run=4,
    )
    assert attn_output == "attn-output"
    assert decoder.calls[-1][-1]["layer_to_run"] == 4

    linear_output = model.forward_linear_only(
        input_ids,
        positions,
        intermediate_tensors=intermediate_tensors,
        num_layer_to_run=6,
    )
    assert linear_output == "linear-output"
    assert decoder.calls[-1][-1]["layer_to_run"] == 6


def test_mistral3_forwards_async_execution_controls() -> None:

    async def run_test() -> None:
        model, decoder = _make_model()
        input_ids = torch.tensor([1, 2])
        positions = torch.tensor([0, 1])
        intermediate_tensors = object()
        module_events = {}

        output = await model.forward_async(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            module_events=module_events,
            profile_num_layers=5,
            profile_layer_start=1,
        )

        assert output == "async-output"
        name, _, _, actual_intermediate_tensors, kwargs = decoder.calls[-1]
        assert name == "forward_async"
        assert actual_intermediate_tensors is intermediate_tensors
        assert kwargs["module_events"] is module_events
        assert kwargs["profile_num_layers"] == 5
        assert kwargs["profile_layer_start"] == 1

    asyncio.run(run_test())
