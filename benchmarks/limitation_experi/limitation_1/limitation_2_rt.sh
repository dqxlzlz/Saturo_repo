#!/bin/bash
# run_experiments.sh
be_input_len=6000
# be_output_len=300
be_output_len=50

batch_sizes=(128)
# for ((bs=1280; bs<=4096; bs+=128)); do
#   batch_sizes+=("$bs")
# done
max_num_seq=128
output_csv_path="/data/liaojianxiong/hybrid_service/hybrid_serve_proj/data/outcome/azure_conv_forward_time.csv"

model_name=/data/liaojianxiong/hf_config/Qwen2.5-14B
# model_name=/root/huggingface_config/Qwen2.5-7B

num_be_prefill_only=100
num_be_decode_only=40
num_rt_prefill_only=0
num_rt_decode_only=50
total_requests=$((num_be_prefill_only+num_be_decode_only+num_rt_prefill_only+num_rt_decode_only))

# FORWARD_TIME_OUTPUT_PATH=$output_csv_path

# 为每个 batch size 运行实验
for size in "${batch_sizes[@]}"; do
    echo "🚀 Running experiment with --max-num-batched-token $size"
    CUDA_VISIBLE_DEVICES=6,7 python ./benchmarks/limitation_experi/offline_inference.py \
        --model $model_name \
        --be-input-len $be_input_len \
        --be-output-len $be_output_len \
        --rt-input-len 1300 \
        --rt-output-len 50 \
        --num-be-normal 0 \
        --fix-rt-request \
        --num-be-prefill-only $num_be_prefill_only \
        --num-be-decode-only $num_be_decode_only \
        --num-rt-normal 0 \
        --num-rt-prefill-only $num_rt_prefill_only \
        --num-rt-decode-only $num_rt_decode_only \
        --enforce-eager \
        -pp 2 \
        --max-num-batched-token $size \
        --gpu-memory-utilization 0.95 \
        --enable-chunked-prefill \
        --max-num-seqs $max_num_seq \
        --input-csv-path "/data/liaojianxiong/hybrid_service/hybrid_serve_proj/data/datasets/AzureLLMInferenceTrace_conv_2024(p92).csv" \
        --output-csv-path $output_csv_path \
        --max-model-len 8192
    echo "✅ Completed run for batch size $size"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed!"


# --output-csv-path $output_csv_path \
# FORWARD_TIME_OUTPUT_PATH=$output_csv_path
# TOTAL_REQUEST=$total_requests 
# VLLM_USE_V1=0 