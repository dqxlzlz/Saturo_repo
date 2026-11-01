import csv
import os

class GlobalConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            # 在这里集中读取环境变量
            cls._instance.total_request = int(os.getenv("TOTAL_REQUEST", "0"))
            cls._instance.write_forward_time_csv = os.getenv("FORWARD_TIME_OUTPUT_PATH", None)
            cls._instance.free_gpu_block_path = os.getenv("FREE_GPU_BLOCK_PATH", None)
            cls._instance.num_iteration_to_write = os.getenv("ITERATION_TO_WRITE", None)

            # 如果设置了输出路径：存在则覆盖重写，并写入表头；不存在则创建并写表头
            if cls._instance.write_forward_time_csv:
                csv_path = cls._instance.write_forward_time_csv
                os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

                with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["len_", "average_TBT"])
        return cls._instance

# 创建一个全局实例供外部引用
settings = GlobalConfig()

