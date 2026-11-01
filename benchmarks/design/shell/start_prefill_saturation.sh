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
# trace_csv_path="${base_out_dir}/test_trace_for_predictor.csv"
metrics_csv_path="${base_out_dir}/metrics_output.csv"

# 数据集路径可通过环境变量 DATASET_DIR 覆盖
DATASET_DIR="${DATASET_DIR:-${base_out_dir}/data/datasets}"
dataset_path="${DATASET_DIR}/prefill_dataset.csv"
# dataset_path="${DATASET_DIR}/decode_dataset.csv"
# dataset_path="${DATASET_DIR}/test_trace_for_predictor.csv"

# 请求数量参数
num_be_prefill_only=0
num_be_decode_only=0
num_be_normal=0
num_rt_prefill_only=500
num_rt_decode_only=0
num_rt_normal=0

# 分布式与引擎参数
TP_size=1
PP_size=1
max_num_seq=64
batch_sizes=(8192)

max_batched_token=4096
max_num_seqs=256
api_server_port=9200
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
TOTAL_TIME=600 # second
RPS=3
total_requests=$((RPS * TOTAL_TIME))
MAX_NUM_BATCH_TOKEN=4096


# todo1: 收集600s的trace的运行数据
# 运行动态的chunk-size（chunk大小为每次运行时请求的长度）

# 确保输出目录存在
mkdir -p "${base_out_dir}/experiment_result"

current_metrics_csv="${base_out_dir}/experiment_result/metrics_prefill_saturation${size}_tp${TP_size}_pp${PP_size}.csv"

CUDA_VISIBLE_DEVICES=6 VLLM_USE_V1=0 python ./benchmarks/design/offline_inference.py \
    --input-trace $dataset_path \
    --output-metrics $current_metrics_csv \
    --model $model_name \
    -tp $TP_size \
    -pp $PP_size \
    --num-request $num_rt_prefill_only \
    --max-num-batched-token $MAX_NUM_BATCH_TOKEN \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs $max_num_seq \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --enforce-eager \
    --serial-execution 


# todo2: 运行simulation-driven来获取 BE-chunk-size

# todo3: 运行simulation-driven来获取 BE-Saturation
