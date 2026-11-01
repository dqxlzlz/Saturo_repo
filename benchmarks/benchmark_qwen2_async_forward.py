# SPDX-License-Identifier: Apache-2.0
"""Compare Qwen2 sync forward, async yield, and TP signal sync overhead.

The benchmark keeps the run scoped to the Qwen2 model file by
monkey-patching Qwen2ForCausalLM.forward_with_time_record, which this branch's
model runner already calls, back to either Qwen2ForCausalLM.forward or
Qwen2ForCausalLM.async_forward. This avoids measuring the CUDA-event timing
path in forward_with_time_record.
"""

import argparse
import asyncio
import dataclasses
import json
import os
import time
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

import vllm.envs as envs
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser


def patch_qwen2_forward_with_sync() -> None:
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

    def sync_forward_without_time_record(
        self: Qwen2ForCausalLM,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        layer_events=None,
        attn_events=None,
    ):
        return self.forward(input_ids, positions, intermediate_tensors,
                            inputs_embeds)

    Qwen2ForCausalLM.forward_with_time_record = sync_forward_without_time_record


def patch_qwen2_forward_with_async() -> None:
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

    loop = asyncio.new_event_loop()

    def async_forward_with_time_record(
        self: Qwen2ForCausalLM,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        layer_events=None,
        attn_events=None,
    ):
        return loop.run_until_complete(
            self.async_forward(input_ids, positions, intermediate_tensors,
                               inputs_embeds))

    Qwen2ForCausalLM.forward_with_time_record = async_forward_with_time_record


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={k: results[k] for k in ["avg_latency", "percentiles"]},
    )
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    use_async_forward = args.qwen2_forward_mode in ("async", "async_signal")
    use_signal_sync = args.qwen2_forward_mode in ("signal", "async_signal")

    if use_signal_sync:
        args.enable_p_pool_saturation = True

    if use_async_forward:
        patch_qwen2_forward_with_async()
    else:
        patch_qwen2_forward_with_sync()

    print(args)

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    assert llm.llm_engine.model_config.max_model_len >= (
        args.input_len + args.output_len
    ), (
        "Please ensure that max_model_len is greater than"
        " the sum of input_len and output_len."
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(
        10000, size=(args.batch_size, args.input_len)
    )
    dummy_prompts: list[PromptType] = [
        {"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()
    ]

    def llm_generate():
        if not args.use_beam_search:
            llm.generate(dummy_prompts, sampling_params=sampling_params,
                         use_tqdm=False)
        else:
            llm.beam_search(
                dummy_prompts,
                BeamSearchParams(
                    beam_width=args.n,
                    max_tokens=args.output_len,
                    ignore_eos=True,
                ),
            )

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            llm.start_profile()
            llm_generate()
            llm.stop_profile()
        else:
            start_time = time.perf_counter()
            llm_generate()
            end_time = time.perf_counter()
            return end_time - start_time

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    if args.profile:
        profile_dir = envs.VLLM_TORCH_PROFILER_DIR
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        return

    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f"Mode: {args.qwen2_forward_mode}")
    print(f"Avg latency: {np.mean(latencies)} seconds")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} seconds")

    if args.output_json:
        results = {
            "mode": args.qwen2_forward_mode,
            "use_async_forward": use_async_forward,
            "use_signal_sync": use_signal_sync,
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark Qwen2 sync forward vs async_forward."
    )
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-iters-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--disable-detokenize", action="store_true")
    parser.add_argument(
        "--qwen2-forward-mode",
        choices=("sync", "signal", "async", "async_signal"),
        default="sync",
        help=("sync: baseline; signal: layer-boundary TP signal all-reduce; "
              "async: async_forward yield only; async_signal: both."),
    )

    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False)
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    if args.profile and not envs.VLLM_TORCH_PROFILER_DIR:
        raise OSError(
            "The environment variable 'VLLM_TORCH_PROFILER_DIR' is not set. "
            "Please set it to a valid path to use torch profiler."
        )
    main(args)
