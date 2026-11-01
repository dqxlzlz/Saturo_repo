# import json
# import matplotlib.pyplot as plt

# # 读取 JSON 文件
# with open('profile_bench_results.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 提取 linear_ms 数据
# linear_ms = data["linear_ms"]

# # 提取 num_token 和 ms_median
# num_tokens = [item["num_token"] for item in linear_ms]
# ms_medians = [item["ms_median"] for item in linear_ms]

# # 绘制散点图
# plt.figure()
# plt.scatter(num_tokens, ms_medians, s=10)  # s 控制点大小，设小一点

# # 标签和标题
# plt.xlabel("num_token")
# plt.ylabel("ms_median")
# plt.title("num_token vs ms_median")

# # 可选：网格
# plt.grid(True, linestyle='--', alpha=0.5)

# plt.show()
# plt.savefig('num_token_vs_ms_median.png')

import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# ====== 配置 ======
input_file = 'profile_bench_results.json'
output_dir = 'plots'
bucket_size = 512

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取 JSON 文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

linear_ms = data["linear_ms"]

# ====== 分组 ======
buckets = defaultdict(list)

for item in linear_ms:
    num_token = item["num_token"]
    ms_median = item["ms_median"]

    bucket_id = num_token // bucket_size
    buckets[bucket_id].append((num_token, ms_median))

# ====== 画图 ======
for bucket_id, values in buckets.items():
    nums = [v[0] for v in values]
    meds = [v[1] for v in values]

    plt.figure()

    plt.scatter(nums, meds, s=5, alpha=0.7)

    start = bucket_id * bucket_size
    end = start + bucket_size

    plt.xlabel("num_token")
    plt.ylabel("ms_median")
    plt.title(f"{start} ~ {end} num_token")

    plt.grid(True, linestyle='--', alpha=0.5)

    # 保存
    filename = os.path.join(output_dir, f"plot_{start}_{end}.png")
    plt.savefig(filename, dpi=150)

    plt.close()

print(f"已保存 {len(buckets)} 张图到 {output_dir}/")