"""Benchmark for Limitation1_2 HOL-Blocking, decreasing chunked-size"""
import argparse
import asyncio
import csv
import math
import random
import time
from typing import Dict, List, Optional, Tuple
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.global_experiment_config import settings


# AZURE_TRACE_CSV = "/data/liaojianxiong/dataset/AzurePublicDataset/data/AzureLLMInferenceTrace_code.csv"
AZURE_TRACE_CSV = "/data/liaojianxiong/hybrid_service/limitation_experi/benchmarks/limitation_experi/AzureLLMInferenceTrace_conv_1week_p0.92.csv"
# AZURE_TRACE_CSV = "/root/vllm_43_build/AzureLLMInferenceTrace_conv_1week_p0.92.csv"



def sample_requests(
    be_input_len: int,
    be_output_len: int,
    fix_rt_request_len: bool = False,
    rt_input_len: int = 1024,
    rt_output_len: int = 1,
    auto_fill_be_req: bool = False,
    max_chunked_size: int = 64,
    num_be_normal:int = 0,
    num_be_prefill_only: int = 0,
    num_be_decode_only: int = 0,
    num_rt_prefill_only: int = 0,
    num_rt_decode_only: int = 0,
    num_rt_decode_in_batch:int = 128,
    csv_trace_path:str = None
):
    """
    生成请求列表：前 num_rt 个 RT + 后 num_be 个 BE。

    - BE: input_len = be_input_len, output_len = be_output_len, is_rt=False
    - RT:
        - fix_rt_request_len=True: input_len = rt_input_len, output_len = rt_output_len
        - 否则: 从 AZURE_TRACE_CSV 读取 ContextTokens 作为 input_len，output_len = rt_output_len
    - auto_fill_be_req=True: 根据 max_chunked_size 和 num_rt 反推 num_be（会覆盖入参 num_be）
    """

    if auto_fill_be_req:
        be_prefill_chunk_size = max_chunked_size - num_rt_decode_in_batch
        if be_prefill_chunk_size <= 0:
            raise ValueError(
                f"Invalid be_prefill_chunk_size={be_prefill_chunk_size}. "
                f"max_chunked_size({max_chunked_size})"
            )
        print(f"{rt_output_len, be_prefill_chunk_size, be_input_len}")
        num_be = math.ceil(rt_output_len * be_prefill_chunk_size / be_input_len) * 2 +1
        # print(f"num be: {num_be}")
        if num_be <= 0:
            raise ValueError(f"Invalid iterations from be_input_len={be_input_len}.")

    dataset = []

    if num_rt_prefill_only + num_rt_decode_only > 0:
        if fix_rt_request_len:
            rt_prompt = {"prompt_token_ids": [0] * rt_input_len}
            for i in range(num_rt_decode_only):
                dataset.append((rt_prompt, rt_output_len, True, True))
            for i in range(num_rt_prefill_only):
                dataset.append((rt_prompt, 1, False, True))
        else:
            # 从 trace 读取 RT 数据集
            # 目前在测试的时候只能有RT prefill/decode 不能既有RT-Prefill又有RT-Decode
            with open(csv_trace_path, mode="r", encoding="utf-8") as f:
                idx = 0
                reader = csv.DictReader(f)
                for row in reader:
                    if idx>=num_rt_decode_only:
                        break
                    input_len = int(row["ContextTokens"])
                    rt_output_len = int(row["GeneratedTokens"])
                    dataset.append(({"prompt_token_ids": [0] * input_len}, rt_output_len, True, True))
                    idx+=1
            
            with open(csv_trace_path, mode="r", encoding="utf-8") as f:
                idx = 0
                reader = csv.DictReader(f)
                for row in reader:
                    if idx>=num_rt_prefill_only:
                        break
                    input_len = int(row["ContextTokens"])
                    dataset.append(({"prompt_token_ids": [0] * input_len}, 1, False, True))
                    idx+=1


    for i in range(num_be_decode_only):
        dataset.append(({"prompt_token_ids": [0] * be_input_len}, be_output_len, True, False))

    be_prompt = {"prompt_token_ids": [0] * be_input_len}
    for _ in range(num_be_prefill_only):
        dataset.append((be_prompt, 1, False, False))
    
    for _ in range(num_be_normal):
        dataset.append((be_prompt, be_output_len, False, False))

    return dataset


def _copy_trace_and_append_forward_time(
    in_csv_path: str,
    out_csv_path: str,
    # rt_forward_times: List[Tuple[int,float]],
    rt_forward_times: List[float],
) -> None:
    """复制一份原始 trace CSV，并新增 forward_time 列。
    - 对前 len(rt_forward_times) 行写入对应值
    - 剩余行填空（如果你只跑了前 num_rt 行，这是最安全的做法）
    """
    with open(in_csv_path, "r", encoding="utf-8", newline="") as fin, \
         open(out_csv_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or missing header: {in_csv_path}")

        fieldnames = list(reader.fieldnames)
        if "forward_time" not in fieldnames:
            fieldnames.append("forward_time")

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            if i < len(rt_forward_times):
                # 保留更多精度，避免你后续统计 P90/P99 时误差放大
                row["forward_time"] = f"{rt_forward_times[i]:.9f}"
            else:
                row["forward_time"] = ""
            writer.writerow(row)

def run_vllm(
    requests: List[Tuple[Dict[str, List[int]], int, bool, bool]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    seed: int,
    n: int,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    gpu_memory_utilization: float = 0.9,
    num_rt: int = 0,
    prefill_slo: float = 2,
) -> float:
    """
    返回: (wall_time_seconds, ttft_list)
    ttft_list: 以每个 request 的“提交到拿到第一个 yield”的时间差估算 TTFT（秒）
    """

    async def _run_async():
        from vllm import SamplingParams, AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext

        # 1) 初始化 AsyncLLMEngine
        engine_args = AsyncEngineArgs(
            model=model,
            tokenizer=tokenizer,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            seed=seed,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            kv_cache_dtype=kv_cache_dtype,
            device=device,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            load_format="dummy",
            scheduling_policy="priority",
        )

        engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.LLM_CLASS,
        )

        ttft_list: List[float] = []

        async def _one_request(req_idx: int, prompt, sp: SamplingParams, priority: int) -> float:
            submit_t = time.perf_counter()
            first_yield_t = None

            async for _out in engine.generate(
                prompt=prompt,
                sampling_params=sp,
                request_id=str(req_idx),
                priority=priority,
            ):
                if first_yield_t is None:
                    first_yield_t = time.perf_counter()

            if first_yield_t is None:
                return float("nan")
            return first_yield_t - submit_t

        # 2) 构造 prompts + sampling_params + priority
        tasks = []
        for req_idx, (prompt, output_len, decode_only, is_rt_request) in enumerate(requests):
            sp = SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
                decode_only=decode_only,
                prefill_ddl=(prefill_slo + time.time()) if is_rt_request else float("inf"),
            )
            priority = 0 if decode_only else 1
            tasks.append(asyncio.create_task(_one_request(req_idx, prompt, sp, priority)))

        # 3) 并发提交，让 engine 自己 continuous batching
        start = time.perf_counter()
        ttft_list = await asyncio.gather(*tasks)
        end = time.perf_counter()

        # 4) 清理（很重要，不然后台 loop/executor 可能持有资源）
        engine.shutdown_background_loop()  # 文档说明需要在 cleanup 调用 :contentReference[oaicite:2]{index=2}

        return (end - start, ttft_list)

    # 兼容：同步环境直接 asyncio.run
    # 如果你在已有 event loop（比如 Jupyter/uvloop）里调用，这里会报错；
    # 那种情况建议把 _run_async 暴露出去用 `await` 调用。
    return asyncio.run(_run_async())


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    requests = sample_requests(
        args.be_input_len,
        args.be_output_len,
        args.fix_rt_request,
        args.rt_input_len,
        args.rt_output_len,
        args.auto_fill_be_req,
        max_chunked_size=args.max_num_batched_tokens,
        num_be_normal=args.num_be_normal,
        num_be_prefill_only=args.num_be_prefill_only,
        num_rt_prefill_only=args.num_rt_prefill_only,
        num_be_decode_only=args.num_be_decode_only,
        num_rt_decode_only=args.num_rt_decode_only,
        num_rt_decode_in_batch=args.num_rt_decode_in_batch,
        csv_trace_path=args.input_csv_path,
    )
    elapsed_time, ttft_list = run_vllm(
        requests, args.model, args.tokenizer, args.quantization,
        args.tensor_parallel_size, args.pipeline_parallel_size, args.seed, args.n,
        args.trust_remote_code, args.dtype, args.max_model_len,
        args.enforce_eager, args.kv_cache_dtype, args.device,
        args.enable_prefix_caching, args.enable_chunked_prefill,
        args.max_num_batched_tokens, args.max_num_seqs, args.gpu_memory_utilization,
        num_rt=args.num_rt_prefill_only + args.num_rt_decode_only,
    )

    # num_iteration = args.rt_output_len
    iteration = args.num_be_prefill_only * args.be_input_len // args.max_num_batched_tokens
    # print(f"elapsed_time:{elapsed_time} | average iteration latency: {elapsed_time / iteration * 1000} ms | Tput: {args.num_be_prefill_only * args.be_input_len / elapsed_time} token/s")
    print(f"elapsed_time:{elapsed_time}")
    print(f"qps: {args.num_be_normal / elapsed_time} req/s")

    # if args.output_csv_path:
    #     data_list = []
    #     with open(args.output_csv_path, mode="a", encoding="utf-8") as f:
    #         reader = csv.reader(f)
    #         # 1. 跳过表头 ["ContextTokens", "ForwardTime"]
    #         header = next(reader) 
            
    #         # 2. 逐行读取并转换类型
    #         for row in reader:
    #             # row 现在的样子可能是 ['128', '0.0523'] (都是字符串)
    #             if row: # 防止空行
    #                 context_len = int(row[0])      # 转为整数
    #                 forward_time = float(row[1])   # 转为浮点数
    #                 data_list.append(forward_time)
    #                 # data_list.append((context_len, forward_time))

    # if args.output_csv_path:
    #     _copy_trace_and_append_forward_time(
    #         in_csv_path=args.input_csv_path,
    #         out_csv_path=args.output_csv_path,
    #         rt_forward_times=data_list,
    #     )
    #     print(f"[OK] wrote trace with forward_time: {args.output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--be-input-len", type=int, default=None)
    parser.add_argument("--be-output-len", type=int, default=None)
    
    parser.add_argument("--rt-input-len", type=int, default=None)
    parser.add_argument("--rt-output-len", type=int, default=None)
    
    parser.add_argument("--num-be-normal", type=int, default=0)
    parser.add_argument("--num-be-prefill-only", type=int, default=0)
    parser.add_argument("--num-be-decode-only", type=int, default=0)

    parser.add_argument("--num-rt-normal", type=int, default=0)
    parser.add_argument("--num-rt-prefill-only", type=int, default=0)
    parser.add_argument("--num-rt-decode-only", type=int, default=0)

    parser.add_argument("--num-rt-decode-in-batch", type=int, default=128)

    parser.add_argument("--fix-rt-request", action="store_true", default=None)
    parser.add_argument("--auto-fill-be-req", action="store_true", default=None)

    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--quantization", "-q", choices=[*QUANTIZATION_METHODS, None], default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1)
    parser.add_argument("--n", type=int, default=1, help="Number of generated sequences per prompt.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "half", "float16", "bfloat16", "float", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--kv-cache-dtype", type=str,
                        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"], default="auto")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    
    parser.add_argument("--input-csv-path", type=str, default=None)
    parser.add_argument("--output-csv-path", type=str, default=None)
    
    
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.hf_max_batch_size is not None:
        raise ValueError("HF max batch size is only for HF backend.")

    main(args)
