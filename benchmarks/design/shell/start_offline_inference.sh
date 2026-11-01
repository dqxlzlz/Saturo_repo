#!/bin/bash
# run_experiments.sh

# ================= 自动推导项目根目录 =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ================= 配置区 =================
# 模型路径可通过环境变量 MODEL_PATH 覆盖
model_name="${MODEL_PATH:-/root/models/Qwen/Qwen2.5-0.5B-Instruct}"
base_out_dir="${PROJECT_ROOT}"

# 中间文件与输出文件
# trace_csv_path="${base_out_dir}/generate_trace.csv"
trace_csv_path="${base_out_dir}/test_trace_for_predictor.csv"
metrics_csv_path="${base_out_dir}/metrics_output.csv"

# 请求数量参数
num_be_prefill_only=0
num_be_decode_only=0
num_be_normal=0
num_rt_prefill_only=100
num_rt_decode_only=0
num_rt_normal=0

# 分布式与引擎参数
TP_size=1
PP_size=1
max_num_seq=64
batch_sizes=(8192)

for size in "${batch_sizes[@]}"; do
    echo "----------------------------------------"
    echo "🚀 Running experiment with --max-num-batched-token $size"
    
    # 动态命名输出文件，区分不同的 batch_size
    current_metrics_csv="${base_out_dir}/metrics_bs${size}_tp${TP_size}_pp${PP_size}.csv"

    FORWARD_TIME_OUTPUT_PATH="$metrics_csv_path" VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=7 python ./benchmarks/design/offline_inference.py \
        --input-trace $trace_csv_path \
        --output-metrics $current_metrics_csv \
        --model $model_name \
        -tp $TP_size \
        -pp $PP_size \
        --max-num-batched-token $size \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs $max_num_seq \
        --max-model-len 8192 \
        --enable-chunked-prefill \
        --enforce-eager

    echo "✅ Completed run for batch size $size"
done

echo "🎉 All experiments completed!"
