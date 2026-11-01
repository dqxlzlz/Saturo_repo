"""Benchmark for Limitation1_2 HOL-Blocking, decreasing chunked-size"""
import argparse
import csv
import math
import random
import time
from typing import List, Optional, Tuple
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.benchmark_util import sample_requests
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.global_experiment_config import settings


# AZURE_TRACE_CSV = "/data/liaojianxiong/dataset/AzurePublicDataset/data/AzureLLMInferenceTrace_code.csv"
AZURE_TRACE_CSV = "/data/liaojianxiong/hybrid_service/limitation_experi/benchmarks/limitation_experi/AzureLLMInferenceTrace_conv_1week_p0.92.csv"
# AZURE_TRACE_CSV = "/root/vllm_43_build/AzureLLMInferenceTrace_conv_1week_p0.92.csv"

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
    requests: List[Tuple[str, int, int]],
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
    from vllm import LLM, SamplingParams
    llm = LLM(
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
        load_format="dummy",
        max_num_seqs=max_num_seqs,
        scheduling_policy="priority"
    )
    request_id = 0
    prompts = []
    sampling_params = []
    priority_list = []
    for prompt_ids, output_len, decode_only, is_rt_request in requests:
    
        prompts.append(prompt_ids)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
                decode_only=decode_only,
                prefill_ddl=prefill_slo + time.time() if is_rt_request else float("inf"),
            )
        )
        if decode_only:
            priority_list.append(0)
        else:
            priority_list.append(1)
        request_id+=1

    start = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=True, priority=priority_list)
    end = time.perf_counter()

    save_append_csv = True
    # if save_append_csv:
    #     ttft_list = llm.get_rt_ttft_list()
    ttft_list = []
    return end - start, ttft_list



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
        # num_rt_normal=args.num_rt_normal,
        num_be_decode_only=args.num_be_decode_only,
        num_rt_decode_only=args.num_rt_decode_only,
        num_rt_decode_in_batch=args.num_rt_decode_in_batch,
        csv_trace_path=args.input_csv_path,
        rt_first=False,
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
    print(f"elapsed_time:{elapsed_time} | average iteration latency: {elapsed_time / iteration * 1000} ms | Tput: {args.num_be_prefill_only * args.be_input_len / elapsed_time} token/s")
    print(f"elapsed_time:{elapsed_time}")
    print(f"qps: {args.num_be_normal / elapsed_time} req/s")

    # if args.output_csv_path:
        
    #     data_list = []

    #     with open(args.output_csv_path, mode="r", encoding="utf-8") as f:
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

    parser.add_argument("--fix-rt-request", action="store_true", default=None)

    parser.add_argument("--num-rt-decode-in-batch", type=int, default=128)
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
