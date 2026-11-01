# check if 30001 already use, and stop it
# 定义要检测的端口
PORT=30001

echo "正在检测端口 $PORT ..."

# 获取占用该端口的 PID
# 1. netstat -tunlp: 列出所有端口
# 2. grep ":$PORT ": 匹配端口号 (注意加空格防止匹配到 300011 这种)
# 3. awk '{print $NF}': 取出最后一列 (通常是 PID/进程名，例如 12345/python)
# 4. cut -d'/' -f1: 以 / 为分隔符，取第一部分 (即 PID)
PID=$(netstat -tunlp 2>/dev/null | grep ":$PORT " | awk '{print $NF}' | cut -d'/' -f1)

# 判断 PID 是否为空
if [ -n "$PID" ]; then
    echo "发现端口 $PORT 被进程 ID [$PID] 占用。"
    
    # 杀掉进程
    kill -9 $PID
    
    # 再次检查确认
    if [ $? -eq 0 ]; then
        echo "✅ 成功终止进程 $PID"
    else
        echo "❌ 终止进程失败，请检查权限"
    fi
else
    echo "🟢 端口 $PORT 当前空闲，没有进程占用。"
fi

# Configuration - can be overridden via environment variables
# MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
MODEL=${MODEL:-/root/huggingface_config/Qwen2.5-7B}
# MODEL=${MODEL:-/root/huggingface_config/Qwen2.5-0.5B}

TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30001}

# Default 1P1D configuration (1 Prefill + 1 Decode with TP=2)
# PREFILL_GPUS=${PREFILL_GPUS:-0}
# DECODE_GPUS=${DECODE_GPUS:-1,2,3}
# PREFILL_PORTS=${PREFILL_PORTS:-20003}
# DECODE_PORTS=${DECODE_PORTS:-20005,20007,20009} 
PREFILL_GPUS=${PREFILL_GPUS:-"5;6"} 
DECODE_GPUS=${DECODE_GPUS:-"0;7"}   
PREFILL_PORTS=${PREFILL_PORTS:-"20003;20013"}
DECODE_PORTS=${DECODE_PORTS:-"20005;20015"}


echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Proxy Port: $PROXY_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=($1)
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        echo "Example: export HF_TOKEN=your_token_here"
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    # Check if the number of GPUs are >=2 via nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    kill -- -$$            # negative PID  ==  "this whole process-group"
    wait                   # reap children so we don't leave zombies
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 1
  done
}

main() {
    check_required_files asymmetric_p2p_nccl_xpyd.py
    check_hf_token
    check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - prefill*.log: Prefill server logs"
    echo "  - decode*.log: Decode server logs"
    echo "  - proxy.log: Proxy server log"

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo ""
    echo "Starting proxy server on port $PROXY_PORT..."
    python3 asymmetric_p2p_nccl_xpyd.py &
    PIDS+=($!)

    # Parse GPU and port arrays
    IFS=';' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=';' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=';' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=';' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    PREFILL_TP_LIST=(1 1)  # 第1个实例TP=2, 第2个TP=2, 第3个TP=4
    PREFILL_PP_LIST=(1 1)

    DECODE_TP_LIST=(1 1)
    DECODE_PP_LIST=(1 1)    
    
    NUM_PREFILL=${#PREFILL_TP_LIST[@]}
    PREFILL_STRATEGY_MAP=""
    for j in "${!PREFILL_TP_LIST[@]}"; do
        tp=${PREFILL_TP_LIST[$j]}
        pp=${PREFILL_PP_LIST[$j]}
        # 构建 JSON 子项
        ITEM="\"$j\":{\"tp_size\":$tp,\"pp_size\":$pp}"
        # 拼接并处理逗号
        if [ -z "$PREFILL_STRATEGY_MAP" ]; then
            PREFILL_STRATEGY_MAP="$ITEM"
        else
            PREFILL_STRATEGY_MAP="$PREFILL_STRATEGY_MAP,$ITEM"
        fi
    done

    DECODE_STRATEGY_MAP=""
    for j in "${!DECODE_TP_LIST[@]}"; do
        global_idx=$((NUM_PREFILL + j)) # 编号偏移
        tp=${DECODE_TP_LIST[$j]}
        pp=${DECODE_PP_LIST[$j]}
        ITEM="\"$global_idx\":{\"tp_size\":$tp,\"pp_size\":$pp}"
        if [ -z "$DECODE_STRATEGY_MAP" ]; then
            DECODE_STRATEGY_MAP="$ITEM"
        else
            DECODE_STRATEGY_MAP="$DECODE_STRATEGY_MAP,$ITEM"
        fi
    done

    PREFILL_ASYMMETRIC_CONFIG="{\"send_target_worker_strategy\":{$DECODE_STRATEGY_MAP}, \"recv_target_worker_strategy\":{}}"
    DECODE_ASYMMETRIC_CONFIG="{\"send_target_worker_strategy\":{}, \"recv_target_worker_strategy\":{$PREFILL_STRATEGY_MAP}}"

    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((21001 + i))
        local tp=${PREFILL_TP_LIST[$i]}
        local pp=${PREFILL_PP_LIST[$i]}
        
        echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_V1=1 vllm serve $MODEL \
        --load-format dummy \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $port \
        --pipeline-parallel-size $pp \
        --tensor-parallel-size $tp \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-log-request \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
        --asymmetric-kv-config "$PREFILL_ASYMMETRIC_CONFIG" > prefill$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # =============================================================================
    # Launch Decode Servers (Y Decoders)
    # =============================================================================
    echo ""
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i))
        local tp=${DECODE_TP_LIST[$i]}
        local pp=${DECODE_PP_LIST[$i]}
        
        echo "  Decode server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=$gpu_id vllm serve $MODEL \
        --load-format dummy \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $port \
        --pipeline-parallel-size $pp \
        --tensor-parallel-size $tp \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-log-request \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
        --asymmetric-kv-config "$DECODE_ASYMMETRIC_CONFIG" > decode$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port"
            cleanup
            exit 1
        fi
    done

    echo ""
    echo "All servers are up. Starting benchmark..."

    # =============================================================================
    # Run Benchmark
    # =============================================================================
    cd ../../../benchmarks/
    python3 benchmark_serving.py --port 10001 --seed $(date +%s) \
        --model $MODEL \
        --dataset-name random --random-input-len 7500 --random-output-len 200 \
        --num-prompts 200 --burstiness 100 --request-rate 2 | tee benchmark.log

    echo "Benchmarking done. Cleaning up..."

    cleanup
}

main