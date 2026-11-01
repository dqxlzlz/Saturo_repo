# generate_trace.py
import csv
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark trace dataset with random lengths")
    parser.add_argument("--output-csv", type=str, required=True)
    
    # BE (Background) requests
    # 使用 nargs=2 接收两个参数作为范围的 [最小值, 最大值]
    parser.add_argument("--be-input-range", type=int, nargs=2, default=[1408, 1409], help="BE请求的输入长度范围 (min max)")
    parser.add_argument("--be-output-range", type=int, nargs=2, default=[200, 400], help="BE请求的输出长度范围 (min max)")
    parser.add_argument("--num-be-prefill-only", type=int, default=0)
    parser.add_argument("--num-be-decode-only", type=int, default=0)
    parser.add_argument("--num-be-normal", type=int, default=0)

    # RT (Real-Time) requests
    parser.add_argument("--rt-input-range", type=int, nargs=2, default=[1204, 2049], help="RT请求的输入长度范围 (min max)")
    parser.add_argument("--rt-output-range", type=int, nargs=2, default=[50, 150], help="RT请求的输出长度范围 (min max)")
    parser.add_argument("--num-rt-prefill-only", type=int, default=0)
    parser.add_argument("--num-rt-decode-only", type=int, default=100)
    parser.add_argument("--num-rt-normal", type=int, default=0)

    args = parser.parse_args()

    dataset = []
    req_id = 0

    # 将参数从固定数字改为长度范围 [min_len, max_len]
    def add_requests(req_type, prompt_range, output_range, count):
        nonlocal req_id
        for _ in range(count):
            # 针对每条请求，在给定的范围内随机生成独立长度
            prompt_len = random.randint(prompt_range[0], prompt_range[1])
            output_len = random.randint(output_range[0], output_range[1])
            
            # 为了防止 prompt_len 为 0 导致 vllm 报错，decode_only 至少给 1 个 token
            p_len = max(1, prompt_len) 
            #
            dataset.append({
                "req_id": req_id,
                "RequestType": req_type,
                "ContextTokens": p_len,
                "GeneratedTokens": output_len
            })
            req_id += 1

    # ================= 组装数据集 =================
    # 1. BE 请求
    # Prefill-only: 输入范围随机，输出写死为 [1, 1] 以保证固定生成 1 个 token
    add_requests("BE_PREFILL", args.be_input_range, [1, 1], args.num_be_prefill_only)
    # Decode-only: 输入写死为 [1, 1]，输出范围随机
    add_requests("BE_decode", args.be_input_range, args.be_output_range, args.num_be_decode_only)
    # Normal: 输入和输出都在指定的范围内随机   
    add_requests("BE_NORMAL", args.be_input_range, args.be_output_range, args.num_be_normal)

    # 2. RT 请求
    add_requests("RT_prefill", args.rt_input_range, [1, 1], args.num_rt_prefill_only)
    add_requests("RT_decode", args.rt_input_range, args.rt_output_range, args.num_rt_decode_only)
    add_requests("RT_NORMAL", args.rt_input_range, args.rt_output_range, args.num_rt_normal)

    # 可选：打乱请求顺序，模拟真实并发
    random.shuffle(dataset)

    # 写入 CSV
    with open(args.output_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["req_id", "RequestType", "ContextTokens", "GeneratedTokens"])
        writer.writeheader()
        writer.writerows(dataset)

    print(f"✅ 成功生成数据集，共 {len(dataset)} 条请求，已保存至 {args.output_csv}")

if __name__ == "__main__":
    main()