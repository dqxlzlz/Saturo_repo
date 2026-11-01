import pandas as pd
import matplotlib.pyplot as plt

# 1) 读 CSV（示例：如果你是文件就用 pd.read_csv("xxx.csv")）
from io import StringIO

df = pd.read_csv("data/outcome/azure_conv_forward_time_tp_1(v1).csv")

# 2) 解析时间戳（带 +00:00 的会自动按 UTC 解析）
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)

# 3) 按“自然秒”分桶
df["sec_ts"] = df["TIMESTAMP"].dt.floor("s")

# 4) 统计从起始秒开始的 240 秒（空秒补 0）
start_sec = df["sec_ts"].min()
full_secs = pd.date_range(start_sec, periods=120, freq="s", tz="UTC")

req_per_sec = (
    df[df["sec_ts"].between(full_secs[0], full_secs[-1])]
      .groupby("sec_ts")
      .size()
      .reindex(full_secs, fill_value=0)
)

# 5) 画图
plt.figure(figsize=(12, 4))
plt.plot(req_per_sec.index, req_per_sec.values)
plt.title("Requests per Second (first 240 seconds)")
plt.xlabel("Time (UTC)")
plt.ylabel("Req/s")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("arrival_rate_plot.png")
plt.show()
