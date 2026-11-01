#!/bin/bash
# run_experiments.sh
be_input_len=6000
be_output_len=300

batch_sizes=(512)
for ((bs=1152; bs<=4096; bs+=256)); do
  batch_sizes+=("$bs")
done
max_num_seq=128

model_name=/data/liaojianxiong/hf_config/Qwen2.5-14B
# model_name=/root/huggingface_config/Qwen2.5-7B

# 为每个 batch size 运行实验
for size in "${batch_sizes[@]}"; do
    echo "🚀 Running experiment with --max-num-batched-token $size"
    CUDA_VISIBLE_DEVICES=6 python ./benchmarks/limitation_experi/offline_inference.py \
        --model $model_name \
        --be-input-len $be_input_len \
        --be-output-len $be_output_len \
        --rt-input-len 1300 \
        --rt-output-len 1 \
        --fix-rt-request \
        --num-be-normal 20 \
        --num-be-prefill-only 0 \
        --num-be-decode-only 0 \
        --num-rt-prefill-only 0 \
        --num-rt-decode-only 0 \
        --enforce-eager \
        --max-num-batched-token $size \
        --gpu-memory-utilization 0.9 \
        --enable-chunked-prefill \
        --max-num-seqs $max_num_seq \
        --output-csv-path ./outcome/data/AzureLLMInferenceTrace_conv_with_forward_time.csv \
        --max-model-len 8192
    echo "✅ Completed run for batch size $size"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed!"

# -pp 2 \
# VLLM_DUMMY_PP_COMM=1 
# --num-be-prefill-only 30 \
# VLLM_USE_V1=0 
