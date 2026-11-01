#!/bin/bash
# run_experiments.sh
be_input_len=1452
be_output_len=300
rt_input_len=1300
rt_output_len=100

batch_sizes=(1452)
# for ((bs=3072; bs<=4096; bs+=512)); do
#   batch_sizes+=("$bs")
# done
max_num_seq=64
output_csv_path="/data/liaojianxiong/hybrid_service/hybrid_serve_proj/data/outcome/azure_conv_forward_time.csv"

# model_name=/data/liaojianxiong/hf_config/Qwen2.5-14B
model_name=/root/huggingface_config/Qwen2.5-7B

num_be_prefill_only=10
# num_be_decode_only=50
num_be_decode_only=0
num_rt_prefill_only=0
num_rt_decode_only=0
total_requests=$((num_be_prefill_only+num_be_decode_only+num_rt_prefill_only+num_rt_decode_only))

TP_size=1
PP_size=1

# base_out_dir="/data/liaojianxiong/hybrid_service/hybrid_serve_proj/data/outcome"
base_out_dir="/root/vllm_92_build/vllm"
input_csv_path="${base_out_dir}/AzureLLMInferenceTrace_conv_2024(p92).csv"
out_csv_path="${base_out_dir}/Azure_conv_tp${TP_size}_pp${PP_size}.csv"

# FORWARD_TIME_OUTPUT_PATH=$out_csv_path
# --fix-rt-request \

# 为每个 batch size 运行实验
for size in "${batch_sizes[@]}"; do
    echo "🚀 Running experiment with --max-num-batched-token $size"
    VLLM_USE_V1=0 TOTAL_REQUEST=$total_requests CUDA_VISIBLE_DEVICES=7 FORWARD_TIME_OUTPUT_PATH=$out_csv_path python ./benchmarks/limitation_experi/offline_inference_v0_async.py \
        --model $model_name \
        --be-input-len $be_input_len \
        --be-output-len $be_output_len \
        --rt-input-len $rt_input_len \
        --rt-output-len $rt_output_len \
        --num-be-normal 0 \
        --fix-rt-request \
        --num-be-prefill-only $num_be_prefill_only \
        --num-be-decode-only $num_be_decode_only \
        --num-rt-normal 0 \
        --num-rt-prefill-only $num_rt_prefill_only \
        --num-rt-decode-only $num_rt_decode_only \
        --enforce-eager \
        -tp $TP_size \
        -pp $PP_size \
        --max-num-batched-token $size \
        --gpu-memory-utilization 0.9 \
        --enable-chunked-prefill \
        --max-num-seqs $max_num_seq \
        --input-csv-path $input_csv_path \
        --output-csv-path $out_csv_path \
        --max-model-len 8192
    echo "✅ Completed run for batch size $size"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed!"


# --output-csv-path $output_csv_path \
# FORWARD_TIME_OUTPUT_PATH=$output_csv_path
# --fix-rt-request \

