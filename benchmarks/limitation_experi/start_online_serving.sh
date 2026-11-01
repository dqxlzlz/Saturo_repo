#!/bin/bash
# run_experiments.sh

be_input_len=6000
be_output_len=300
rt_input_len=1300
rt_output_len=100

num_be_prefill_only=30
num_be_decode_only=0
num_rt_prefill_only=0
num_rt_decode_only=0

time=60

prefill_slo=1.5
rps_list=(1)
# max_batched_token_list=(384)
max_batched_token=1536
model_name=/data/liaojianxiong/hf_config/Qwen2.5-14B
# model_name=/root/huggingface_config/Qwen2.5-7B

dataset_path="/data/liaojianxiong/hybrid_service/hybrid_serve_proj/data/datasets/AzureLLMInferenceTrace_conv_2024(p92).csv"
# dataset_path="/root/vllm_92_build/vllm/data/datasets/azure_conv_forward_time_tp_1(v1).csv"
base_dir="/data/liaojianxiong/hybrid_service/hybrid_serve_proj/data/outcome"
# free_gpu_block_output_path="/data/liaojianxiong/hybrid_service/hybrid_serve_proj/data/outcome/rps_()"
# free_gpu_block_output_path="/root/vllm_92_build/vllm/data/outcome/free_gpu_block.csv"

TP_size=1
PP_size=1

base_out_dir="/data/liaojianxiong/hybrid_service/hybrid_serve_proj/data/outcome"
out_csv_path="${base_out_dir}/Azure_conv_tp${TP_size}_pp${PP_size}.csv"

num_iteration_time_record_free_gpu_mem=100

api_server_port=9200
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}

# 定义清理函数，用于处理 Ctrl+C 等异常退出
cleanup() {
    echo "Caught signal, stopping everything..."
    trap - INT TERM 
    # 杀掉当前记录的所有子进程
    if [ ${#PIDS[@]} -ne 0 ]; then
        kill "${PIDS[@]}" 2>/dev/null
    fi
    # 强制清理端口
    local server_pid=$(lsof -t -i:$api_server_port)
    [ ! -z "$server_pid" ] && kill -9 $server_pid 2>/dev/null
    
    kill -9 -- -$$ 2>/dev/null
    exit 0
}

# 等待服务器就绪的函数
wait_for_server() {
    local port=$1
    local start_time=$(date +%s)
    echo "Waiting for server on port $port..."
    while true; do
        if curl -s "localhost:${port}/health" > /dev/null; then
            echo "Server on port $port is ready."
            return 0
        fi
        if (( $(date +%s) - start_time >= TIMEOUT_SECONDS )); then
            return 1
        fi
        sleep 2
    done
}

trap cleanup INT USR1 TERM

for rps in "${rps_list[@]}"; do
    echo "🚀 Starting new experiment: RPS = $rps"
    if (( max_batched_token < 256 )); then
        max_num_seqs=128
    else
        max_num_seqs=256
    fi

    free_gpu_block_output_path="${base_dir}/rps_${rps}"

    # 1. 启动 api_server
    FREE_GPU_BLOCK_PATH=$free_gpu_block_output_path \
    VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=4,5,6,7 FORWARD_TIME_OUTPUT_PATH=$out_csv_path python benchmarks/limitation_experi/api_server.py \
    --disable-log-requests \
    --model $model_name \
    --max-num-batched-token $max_batched_token \
    --load-format dummy \
    --no-enable-prefix-caching \
    --enforce-eager \
    --max-model-len 6500 \
    --enable-chunked-prefill \
    --port $api_server_port \
    --gpu-memory-utilization 0.95 \
    -tp $TP_size \
    -pp $PP_size \
    --max-num-seqs $max_num_seqs &
    current_server_pid=$!
    PIDS=($current_server_pid) # 记录当前进程 PID 供 trap 使用

    # 2. 等待就绪
    if ! wait_for_server $api_server_port; then
        echo "❌ Fail to start server on port $api_server_port"
        kill -9 $current_server_pid 2>/dev/null
        continue # 尝试下一个 RPS
    fi
    
    # 3. 计算并运行实验
    # num_rt=$(python3 -c "print(int($time * $rps))")
    # num_rt=10
    FREE_GPU_BLOCK_PATH=$free_gpu_block_output_path \
    FORWARD_TIME_OUTPUT_PATH=$out_csv_path \
    ITERATION_TO_WRITE=$num_iteration_time_record_free_gpu_mem \
    python benchmarks/limitation_experi/online_inference.py \
        --model $model_name \
        --be-input-len $be_input_len \
        --be-output-len $be_output_len \
        --rt-input-len $rt_input_len \
        --rt-output-len $rt_output_len \
        --num-be-normal 0 \
        --num-be-prefill-only $num_be_prefill_only \
        --num-be-decode-only $num_be_decode_only \
        --num-rt-normal 0 \
        --num-rt-prefill-only $num_rt_prefill_only \
        --num-rt-decode-only $num_rt_decode_only \
        --request-rate $rps \
        --port $api_server_port \
        --max-num-batched-tokens 6500 \
        --input-csv-path $dataset_path \
        --endpoint /generate

    echo "✅ Finished experiment for RPS: $rps"

    # 4. 关闭服务器并清理，为下一次循环做准备
    echo "Stopping server for next iteration..."
    kill $current_server_pid
    
    # 等待进程真正退出，防止显存未释放
    wait $current_server_pid 2>/dev/null
    
    # 额外保险：如果进程还没死，强制杀掉并清理端口
    sleep 2
    local leftover_pid=$(lsof -t -i:$api_server_port)
    if [ ! -z "$leftover_pid" ]; then
        kill -9 $leftover_pid 2>/dev/null
    fi
    
    PIDS=() # 清空 PID 列表
    echo "Server stopped. Sleeping 5s before next run..."
    sleep 5
done

echo "All experiments completed!"



# --prefill-slo $prefill_slo
# --num-rt $num_rt \
# --rt-input-len $rt_input_len \
# --rt-output-len 1 \