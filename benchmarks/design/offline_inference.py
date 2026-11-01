# benchmark_vllm.py
import argparse
import asyncio
import csv
import time
from typing import List, Dict

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.usage.usage_lib import UsageContext

from vllm.engine.time_predictor import TimePredictor


def load_trace(csv_path: str, num_requests: int = None) -> List[Dict]:
    """读取预先生成好的请求数据集"""
    requests,id = [],0
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if num_requests is not None and id >= num_requests:
                break
            requests.append({
                "req_id": id,
                "req_type": row["RequestType"],
                "prompt_len": int(row["ContextTokens"]),
                "output_len": int(row["GeneratedTokens"])
            })
            id += 1
    return requests

async def run_vllm_async(requests: List[Dict], args: argparse.Namespace) -> tuple:
    # 1. 初始化 AsyncLLMEngine
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        load_format="dummy",
        scheduling_policy="priority",
        serial_execution=args.serial_execution,
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=UsageContext.LLM_CLASS,
    )

    results = []

    # 2. 定义单个请求的处理逻辑
    async def process_request(req: Dict):
        # 构造 dummy prompt
        prompt_token_ids = [0] * req["prompt_len"]
        
        # 判断请求优先级 (根据你之前的逻辑，如果是 DECODE 优先，或者 RT 优先，可以在这里设)
        priority = 0 if "decode" in req["req_type"] else 1
        
        sp = SamplingParams(
            temperature=1.0,
            ignore_eos=True,
            max_tokens=req["output_len"],
            prefill_ddl=time.time()+args.prefill_slo if req["req_type"] == "RT_prefill" else float("inf")
        )

        submit_time = time.perf_counter()
        first_token_time = None
        end_time = None

        async for _out in engine.generate(
            prompt={"prompt_token_ids": prompt_token_ids},
            sampling_params=sp,
            request_id=str(req["req_id"]),
            priority=priority,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
        
        end_time = time.perf_counter()

        # 记录指标
        ttft = (first_token_time - submit_time) if first_token_time else 0
        e2e_latency = end_time - submit_time
        
        results.append({
            "req_id": req["req_id"],
            "req_type": req["req_type"],
            "prompt_len": req["prompt_len"],
            "output_len": req["output_len"],
            "ttft": ttft,
            "e2e_latency": e2e_latency
        })

    # 3. 执行所有请求 (根据参数决定是串行还是并发)
    start_time = time.perf_counter()
    if args.serial_execution:
        for req in requests:
            await process_request(req)
    else:
        tasks = [asyncio.create_task(process_request(req)) for req in requests]
        await asyncio.gather(*tasks)
    total_elapsed_time = time.perf_counter() - start_time

    # # 4. 清理后台进程
    # engine.shutdown_background_loop()

    return results, total_elapsed_time

def main():
    parser = argparse.ArgumentParser(description="Clean Benchmark script for vLLM")
    
    # 路径参数
    parser.add_argument("--input-trace", type=str, required=True, help="Path to input trace CSV")
    parser.add_argument("--output-metrics", type=str, required=True, help="Path to output metrics CSV")
    
    # vLLM 引擎参数 (只保留跟性能直接相关的)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument("-pp", "--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--prefill-slo", type=float, default=1.0)
    parser.add_argument("--serial-execution", action="store_true", help="Whether to execute requests serially (for collection forward time of each request of RT prefill)")
    parser.add_argument("--num-requests", type=int, default=None, help="Number of requests to read from trace")

    args = parser.parse_args()

    # 1. 加载数据
    requests = load_trace(args.input_trace, args.num_requests)
    print(f"🚀 Loaded {len(requests)} requests from {args.input_trace}")

    # 2. 运行 vLLM
    print(f"🔥 Starting vLLM Engine and running benchmark...")
    results, total_time = asyncio.run(run_vllm_async(requests, args))

    # 3. 统计宏观数据
    total_input_tokens = sum(r["prompt_len"] for r in results)
    total_output_tokens = sum(r["output_len"] for r in results)
    overall_throughput = (total_input_tokens + total_output_tokens) / total_time
    
    print("\n" + "="*40)
    print(f"🎉 Benchmark Completed!")
    print(f"⏱️  Total Elapsed Time: {total_time:.4f} s")
    print(f"🚀 Throughput: {overall_throughput:.2f} tokens/s")
    print(f"📊 QPS: {len(requests)/total_time:.2f} req/s")
    print("="*40 + "\n")

    # 4. 写入详细指标到输出 CSV
    with open(args.output_metrics, mode="w", encoding="utf-8", newline="") as f:
        fieldnames = ["req_id", "req_type", "prompt_len", "output_len", "ttft", "e2e_latency"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Metrics saved to {args.output_metrics}")

if __name__ == "__main__":
    main()