import csv
import math


# def sample_requests(
#     be_input_len: int,
#     be_output_len: int,
#     fix_rt_request_len: bool = False,
#     rt_input_len: int = 1024,
#     rt_output_len: int = 1,
#     auto_fill_be_req: bool = False,
#     max_chunked_size: int = 64,
#     num_be_normal:int = 0,
#     num_be_prefill_only: int = 0,
#     num_be_decode_only: int = 0,
#     num_rt_prefill_only: int = 0,
#     num_rt_decode_only: int = 0,
#     num_rt_decode_in_batch:int = 128,
#     csv_trace_path:str = None,
#     rt_first:bool = True,
# ):
#     """
#     生成请求列表：前 num_rt 个 RT + 后 num_be 个 BE。

#     - BE: input_len = be_input_len, output_len = be_output_len, is_rt=False
#     - RT:
#         - fix_rt_request_len=True: input_len = rt_input_len, output_len = rt_output_len
#         - 否则: 从 AZURE_TRACE_CSV 读取 ContextTokens 作为 input_len，output_len = rt_output_len
#     - auto_fill_be_req=True: 根据 max_chunked_size 和 num_rt 反推 num_be（会覆盖入参 num_be）
#     """

#     if auto_fill_be_req:
#         be_prefill_chunk_size = max_chunked_size - num_rt_decode_in_batch
#         if be_prefill_chunk_size <= 0:
#             raise ValueError(
#                 f"Invalid be_prefill_chunk_size={be_prefill_chunk_size}. "
#                 f"max_chunked_size({max_chunked_size})"
#             )
#         print(f"{rt_output_len, be_prefill_chunk_size, be_input_len}")
#         num_be = math.ceil(rt_output_len * be_prefill_chunk_size / be_input_len) * 2 +1
#         # print(f"num be: {num_be}")
#         if num_be <= 0:
#             raise ValueError(f"Invalid iterations from be_input_len={be_input_len}.")

#     dataset = []

#     if num_rt_prefill_only + num_rt_decode_only > 0:
#         if fix_rt_request_len:
#             rt_prompt = {"prompt_token_ids": [0] * rt_input_len}
#             for i in range(num_rt_decode_only):
#                 dataset.append((rt_prompt, rt_output_len, True, True))
#             for i in range(num_rt_prefill_only):
#                 dataset.append((rt_prompt, rt_output_len, False, True))
#         else:
#             # 从 trace 读取 RT 数据集
#             # 目前在测试的时候只能有RT prefill/decode 不能既有RT-Prefill又有RT-Decode
#             with open(csv_trace_path, mode="r", encoding="utf-8") as f:
#                 idx = 0
#                 reader = csv.DictReader(f)
#                 for row in reader:
#                     if idx>=num_rt_decode_only:
#                         break
#                     input_len = int(row["ContextTokens"])
#                     rt_output_len = int(row["GeneratedTokens"])
#                     dataset.append(({"prompt_token_ids": [0] * input_len}, rt_output_len, True, True))
#                     idx+=1
            
#             with open(csv_trace_path, mode="r", encoding="utf-8") as f:
#                 idx = 0
#                 reader = csv.DictReader(f)
#                 for row in reader:
#                     if idx>=num_rt_prefill_only:
#                         break
#                     input_len = int(row["ContextTokens"])
#                     dataset.append(({"prompt_token_ids": [0] * input_len}, 1, False, True))
#                     idx+=1


#     for i in range(num_be_decode_only):
#         if i<num_be_decode_only // 2:
#             dataset.insert(num_be_decode_only // 2, ({"prompt_token_ids": [0] * be_input_len}, be_output_len, True, False))
#         else:
#             dataset.append(({"prompt_token_ids": [0] * be_input_len}, be_output_len, True, False))

#     be_prompt = {"prompt_token_ids": [0] * be_input_len}
#     for _ in range(num_be_prefill_only):
#         dataset.append((be_prompt, 1, False, False))
    
#     for _ in range(num_be_normal):
#         dataset.append((be_prompt, be_output_len, False, False))

#     return dataset


def sample_requests(
    be_input_len: int,
    be_output_len: int,
    fix_rt_request_len: bool = False,
    rt_input_len: int = 1024,
    rt_output_len: int = 1,
    auto_fill_be_req: bool = False,
    max_chunked_size: int = 64,
    num_be_normal: int = 0,
    num_be_prefill_only: int = 0,
    num_be_decode_only: int = 0,
    num_rt_prefill_only: int = 0,
    num_rt_decode_only: int = 0,
    num_rt_decode_in_batch: int = 128,
    csv_trace_path: str = None,
    rt_first: bool = True,   # True: RT 在前；False: BE 在前
):
    """
    生成请求列表：按 rt_first 决定顺序

    - rt_first=True : RT 请求在前，BE 请求在后
    - rt_first=False: BE 请求在前，RT 请求在后

    - BE: input_len = be_input_len, output_len = be_output_len, is_rt=False
    - RT:
        - fix_rt_request_len=True: input_len = rt_input_len, output_len = rt_output_len
        - 否则: 从 csv_trace_path 读取 ContextTokens 作为 input_len，output_len 用 trace 的 GeneratedTokens 或固定为 1
    - auto_fill_be_req=True: 根据 max_chunked_size 和 num_rt_decode_in_batch 反推 num_be（会覆盖相关入参）
    """

    if auto_fill_be_req:
        be_prefill_chunk_size = max_chunked_size - num_rt_decode_in_batch
        if be_prefill_chunk_size <= 0:
            raise ValueError(
                f"Invalid be_prefill_chunk_size={be_prefill_chunk_size}. "
                f"max_chunked_size({max_chunked_size})"
            )
        print(f"{rt_output_len, be_prefill_chunk_size, be_input_len}")
        num_be = math.ceil(rt_output_len * be_prefill_chunk_size / be_input_len) * 2 + 1
        if num_be <= 0:
            raise ValueError(f"Invalid iterations from be_input_len={be_input_len}.")
        # 你原逻辑里 num_be 会覆盖，但你实际后面是用 num_be_* 分项数量
        # 如果你希望 auto_fill_be_req 直接覆盖 num_be_normal/num_be_prefill_only/num_be_decode_only，
        # 需要你定义覆盖策略；这里先不擅自改动。

    rt_dataset = []
    be_dataset = []

    # ---------- 生成 RT ----------
    if num_rt_prefill_only + num_rt_decode_only > 0:
        if fix_rt_request_len:
            rt_prompt = {"prompt_token_ids": [0] * rt_input_len}
            for _ in range(num_rt_decode_only):
                rt_dataset.append((rt_prompt, rt_output_len, True, True))   # decode_only=True, is_rt=True
            for _ in range(num_rt_prefill_only):
                rt_dataset.append((rt_prompt, rt_output_len, False, True))  # prefill_only, is_rt=True
        else:
            if not csv_trace_path:
                raise ValueError("csv_trace_path must be provided when fix_rt_request_len=False.")

            # RT decode_only from trace
            with open(csv_trace_path, mode="r", encoding="utf-8") as f:
                idx = 0
                reader = csv.DictReader(f)
                for row in reader:
                    if idx >= num_rt_decode_only:
                        break
                    input_len = int(row["ContextTokens"])
                    out_len = int(row["GeneratedTokens"])
                    rt_dataset.append(({"prompt_token_ids": [0] * input_len}, out_len, True, True))
                    idx += 1

            # RT prefill_only from trace
            with open(csv_trace_path, mode="r", encoding="utf-8") as f:
                idx = 0
                reader = csv.DictReader(f)
                for row in reader:
                    if idx >= num_rt_prefill_only:
                        break
                    input_len = int(row["ContextTokens"])
                    rt_dataset.append(({"prompt_token_ids": [0] * input_len}, 1, False, True))
                    idx += 1

    for i in range(num_be_decode_only):
        item = ({"prompt_token_ids": [0] * be_input_len}, be_output_len, True, False)
        if i < num_be_decode_only // 2:
            be_dataset.insert(num_be_decode_only // 2, item)
        else:
            be_dataset.append(item)

    be_prompt = {"prompt_token_ids": [0] * be_input_len}

    for _ in range(num_be_prefill_only):
        be_dataset.append((be_prompt, 1, False, False))

    for _ in range(num_be_normal):
        be_dataset.append((be_prompt, be_output_len, False, False))

    if rt_first:
        dataset = rt_dataset + be_dataset
    else:
        dataset = be_dataset + rt_dataset

    return dataset
