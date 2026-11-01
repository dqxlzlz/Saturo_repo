import csv

input_file = "/root/vllm_92_build/vllm/data/datasets/azure_conv_forward_time_tp_1(v1).csv"
output_file = "/root/vllm_92_build/vllm/data/datasets/prefill_dataset.csv"

# 打开输入文件进行读取，打开输出文件进行写入
with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    
    # 使用 DictReader 和 DictWriter 可以方便地按列名操作
    reader = csv.DictReader(infile)
    
    # 指定输出文件需要保留的列名
    fieldnames = ['TIMESTAMP', 'ContextTokens', 'GeneratedTokens', 'RequestType']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    # 写入表头
    writer.writeheader()
    
    # 逐行读取，只提取需要的两列并写入新文件
    for row in reader:
        writer.writerow({
            'TIMESTAMP': row['TIMESTAMP'],
            'ContextTokens': row['ContextTokens'],
            'GeneratedTokens': 1,
            'RequestType': "RT_prefill"
        })

print("处理完成！结果已保存到 output.csv")