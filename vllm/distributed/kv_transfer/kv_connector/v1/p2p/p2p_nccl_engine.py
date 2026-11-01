# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import re
import threading
import time
import typing
from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

import msgpack
import torch
from vllm.distributed.parallel_state import get_world_group
import zmq

from vllm.config import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary, buffer_type, cudaStream_t, ncclComm_t, ncclDataTypeEnum)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (  # noqa: E501
    TensorMemoryPool)
from vllm.utils import current_stream, get_ip

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32


@contextmanager
def set_p2p_nccl_context(num_channels: str):
    original_values: dict[str, Any] = {}
    env_vars = [
        'NCCL_MAX_NCHANNELS',
        'NCCL_MIN_NCHANNELS',
        'NCCL_CUMEM_ENABLE',
        'NCCL_BUFFSIZE',
        'NCCL_PROTO',  # LL,LL128,SIMPLE
        'NCCL_ALGO',  # RING,TREE
    ]

    for var in env_vars:
        original_values[var] = os.environ.get(var)

    logger.info("set_p2p_nccl_context, original_values: %s", original_values)

    try:
        os.environ['NCCL_MAX_NCHANNELS'] = num_channels
        os.environ['NCCL_MIN_NCHANNELS'] = num_channels
        os.environ['NCCL_CUMEM_ENABLE'] = '1'
        yield
    finally:
        for var in env_vars:
            if original_values[var] is not None:
                os.environ[var] = original_values[var]
            else:
                os.environ.pop(var, None)


class P2pNcclEngine:

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 hostname: str = "",
                 port_offset: int = 0,
                 library_path: Optional[str] = None,
                 vllm_config = None) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.nccl = NCCLLibrary(library_path)

        if not hostname:
            hostname = get_ip()
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        # Each card corresponds to a ZMQ address.
        self.zmq_address = f"{self._hostname}:{self._port}"

        # The `http_port` must be consistent with the port of OpenAI.
        self.http_address = (
            f"{self._hostname}:"
            f"{self.config.kv_connector_extra_config['http_port']}")

        # If `proxy_ip` or `proxy_port` is `""`,
        # then the ping thread will not be enabled.
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        mem_pool_size_gb = self.config.get_from_extra_config(
            "mem_pool_size_gb", DEFAULT_MEM_POOL_SIZE_GB)
        self.pool = TensorMemoryPool(max_block_size=int(mem_pool_size_gb) *
                                     1024**3)  # GB

        # The sending type includes tree mutually exclusive options:
        # PUT, GET, PUT_ASYNC.
        self.send_type = self.config.get_from_extra_config("send_type", "PUT")
        if self.send_type == "GET":
            # tensor_id: torch.Tensor
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            # PUT or PUT_ASYNC
            # tensor_id: torch.Tensor
            self.send_queue: deque[list[Any]] = deque()
            self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(target=self._send_async,
                                                     daemon=True)
                self._send_thread.start()

        # tensor_id: torch.Tensor/(addr, dtype, shape)
        self.recv_store: dict[str, Any] = {}
        self.recv_store_asymmetric: dict[str: dict[int, Any]] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.socks: dict[str, Any] = {}  # remote_address: client socket
        self.comms: dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)

        self.nccl_num_channels = self.config.get_from_extra_config(
            "nccl_num_channels", "8")

        self._listener_thread = threading.Thread(
            target=self._listen_for_requests, daemon=True)
        self._listener_thread.start()

        if vllm_config:
            self.asymmetric_config = vllm_config.asymmetric_kv_config

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self._ping,
                                                 daemon=True)
            self._ping_thread.start()

        logger.info(
            "💯P2pNcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f, nccl_num_channels:%s", self.rank, self.local_rank,
            self.http_address, self.zmq_address, self.proxy_address,
            self.send_type, self.buffer_size_threshold, self.nccl_num_channels)


    def create_asymmetric_connect(self):
        """
        Docstring for create_asymmetric_connect
        
        :param self: Description
        """


    def _create_connect(self, remote_address: typing.Optional[str] = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            if remote_address in self.comms:
                logger.info("👋comm exists, remote_address:%s, comms:%s",
                            remote_address, self.comms)
                return sock, self.comms[remote_address]

            unique_id = self.nccl.ncclGetUniqueId()
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            sock.send(msgpack.dumps(data))

            # with torch.cuda.device(self.device):
            #     rank = 0
            #     with set_p2p_nccl_context(self.nccl_num_channels):
            #         comm: ncclComm_t = self.nccl.ncclCommInitRank(
            #             2, unique_id, rank)
            #     self.comms[remote_address] = (comm, rank)
            #     logger.info("🤝ncclCommInitRank Success, %s👉%s, MyRank: %s",
            #                 self.zmq_address, remote_address, rank)
                
        # return self.socks[remote_address], self.comms[remote_address]
        return self.socks[remote_address], 

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: typing.Optional[str] = None,
    ) -> bool:
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True
        else:
            if self.send_type == "PUT":
                return self._send_sync(tensor_id, tensor, remote_address)
            elif self.send_type == "PUT_ASYNC":
                with self.send_queue_cv:
                    self.send_queue.append([tensor_id, remote_address, tensor])
                    self.send_queue_cv.notify()
            else:  # GET
                with self.send_store_cv:
                    tensor_size = tensor.element_size() * tensor.numel()
                    while (self.buffer_size + tensor_size
                           > self.buffer_size_threshold):
                        oldest_tenser_id = next(iter(self.send_store))
                        oldest_tenser = self.send_store.pop(oldest_tenser_id)
                        oldest_tenser_size = oldest_tenser.element_size(
                        ) * oldest_tenser.numel()
                        self.buffer_size -= oldest_tenser_size
                        logger.info(
                            "⛔[GET]Send to %s, tensor_id:%s, tensor_size:%d,"
                            " buffer_size:%d, oldest_tenser_size:%d, rank:%d",
                            remote_address, tensor_id, tensor_size,
                            self.buffer_size, oldest_tenser_size, self.rank)

                    self.send_store[tensor_id] = tensor
                    self.buffer_size += tensor_size
                    logger.debug(
                        "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d, "
                        "shape:%s, rank:%d, buffer_size:%d(%.2f%%)",
                        remote_address, tensor_id, tensor_size, tensor.shape,
                        self.rank, self.buffer_size,
                        self.buffer_size / self.buffer_size_threshold * 100)

        return True

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ) -> torch.Tensor:
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            # start_time = time.time()
            # with self.recv_store_cv:
            #     while tensor_id not in self.recv_store:
            #         self.recv_store_cv.wait()
            #     tensor = self.recv_store[tensor_id]
            
            # if tensor is not None:
            #     if isinstance(tensor, tuple):
            #         addr, dtype, shape = tensor
            #         tensor = self.pool.load_tensor(addr, dtype, shape,
            #                                        self.device)
            #     else:
            #         self.buffer_size -= (tensor.element_size() *
            #                              tensor.numel())
            # else:
            #     duration = time.time() - start_time
            #     logger.warning(
            #         "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, "
            #         "rank:%d", remote_address, tensor_id, duration * 1000,
            #         self.rank)
            # return tensor

            start_time = time.time()
            with self.recv_store_cv:
                # 循环等待，直到该 tensor_id 对应的所有分块都到齐并完成聚合
                while True:
                    is_finished, aggregated_tensor = self.judge_tensor_finish_recv(tensor_id)
                    if is_finished:
                        tensor = aggregated_tensor
                        break
                    # 如果还没到齐，继续等待 notify
                    self.recv_store_cv.wait()

                if tensor is not None:
                    # 统计缓冲区大小变化 (根据聚合后的完整 Tensor 统计)
                    # 注意：如果你的 chunks 是在 GPU 上的，这里主要用于流量或显存监控
                    self.buffer_size -= (tensor.element_size() * tensor.numel())
                else:
                    # 这种情况理论上在 judge_tensor_finish_recv 成功时不应发生
                    duration = time.time() - start_time
                    logger.warning(
                        "🔴[Asymmetric PUT] Recv failed or empty, tensor_id:%s, duration:%.3fms, "
                        "rank:%d", tensor_id, duration * 1000, self.rank)
                
                return tensor


        # GET
        if remote_address is None:
            return None

        if remote_address not in self.socks:
            self._create_connect(remote_address)

        sock = self.socks[remote_address]
        # comm, rank = self.comms[remote_address]
        rank = 0

        data = {"cmd": "GET", "tensor_id": tensor_id}
        sock.send(msgpack.dumps(data))

        message = sock.recv()
        data = msgpack.loads(message)
        if data["ret"] != 0:
            logger.warning("🔴[GET]Recv From %s, tensor_id: %s, ret: %d",
                           remote_address, tensor_id, data["ret"])
            return None

        tensor = torch.empty(data["shape"],
                             dtype=getattr(torch, data["dtype"]),
                             device=self.device)

        # self._recv(comm, tensor, rank ^ 1, self.recv_stream)

        return tensor

    def judge_tensor_finish_recv(self, tensor_id:str, ):
        request_id, layer = tensor_id.split("#")
        _, _, instance_id = self.parse_request_id_asymmetric(request_id=tensor_id, is_prefill=False)
        
        rank = get_world_group().rank   
        target_prefill_workers = self.asymmetric_config.recv_plan[instance_id][rank]
        expected_worker_ids = set(target_prefill_workers.keys())

        if tensor_id not in self.recv_store_asymmetric:
            return False, None

        # 3. 检查收到的 worker 数量是否等于预期的数量
        received_workers = set(self.recv_store_asymmetric[tensor_id].keys())
        
        if expected_worker_ids.issubset(received_workers):
            # 数据已到齐，进行聚合
            return True, self.aggregate_received_tensor(tensor_id, instance_id, rank)
        
        return False, None

    
    def aggregate_received_tensor(self, tensor_id: str, instance_id: int, rank: int):
        # 1. 获取本地布局，确定最终 Tensor 的形状
        local_layout = self.asymmetric_config.self_strategy.get_worker_layout(rank)
        # num_layers = local_layout["layer_range"][1] - local_layout["layer_range"][0]
        num_heads = local_layout["head_range"][1] - local_layout["head_range"][0]
        head_size = self.asymmetric_config.self_strategy.num_head_size
        
        # 获取其中一个分块来推断非切分维度的大小（如 seq_len, head_dim）
        sample_chunk = next(iter(self.recv_store_asymmetric[tensor_id].values()))
        
        # Tensor 形状是 [kv, seq_len, hidden_state] 或 
        # 根据你的 slice 定义，前两维是 k/v 和 head
        # full_shape = (2, num_heads) + sample_chunk.shape[2:]
        full_shape = (2, sample_chunk.shape[1], num_heads*head_size)
        
        # 2. 创建目标全零或空 Tensor (在 GPU 上)
        aggregated_tensor = torch.empty(
            full_shape, 
            dtype=sample_chunk.dtype, 
            device=sample_chunk.device
        )

        # 3. 根据 Plan 填入数据
        plans = self.asymmetric_config.recv_plan[instance_id][rank]
        # received_chunks here is: (addr, tensor.dtype, tensor.shape)
        received_chunks = self.recv_store_asymmetric[tensor_id]

        for remote_worker_id, plan in plans.items():
            chunk = received_chunks[remote_worker_id]
            if chunk is not None:
 
                if isinstance(chunk, tuple):
                    addr, dtype, shape = chunk
                    chunk_tensor = self.pool.load_tensor(addr, dtype, shape,
                                                    self.device)
                else:
                    chunk_tensor = chunk
                    self.buffer_size -= (chunk_tensor.element_size() *
                                            chunk_tensor.numel())

                # 使用初始化时计算好的 slice
                # l_slice = slice(plan.recv_local_layer_slice)
                start_head = plan.recv_local_head_slice[0]
                end_head = plan.recv_local_head_slice[1]
                h_slice = slice(start_head*head_size, end_head*head_size)
                # 填充
                aggregated_tensor[..., h_slice] = chunk_tensor
            
            else:
                logger.warning(
                    "🔴[Asymmetric PUT] Recv failed or empty, tensor_id:%s, rank:%d", tensor_id, self.rank)

        # 4. 清理内存：数据已经聚合，删除存储里的碎片
        del self.recv_store_asymmetric[tensor_id]
        
        return aggregated_tensor


    @staticmethod
    def parse_request_id_asymmetric(request_id: str, is_prefill=True) -> tuple[str, int, int]:
        """
        Parses the request_id to extract the target node's IP, Port, and Instance ID.
        
        Args:
            request_id (str): The complex request ID string.
            is_prefill (bool): 
                If True, we are the Prefill node looking for Decode node info (addr + dinstance).
                If False, we are the Decode node looking for Prefill node info (addr + pinstance).
        
        Returns:
            tuple[str, int, int]: (ip, port, instance_id)
        """
        
        if is_prefill:
            addr_pattern = r"___decode_addr_(.*?):(\d+)"
            instance_pattern = r"___dinstance_(\d+)"
        else:
            addr_pattern = r"___prefill_addr_(.*?):(\d+)"
            instance_pattern = r"___pinstance_(\d+)"

        addr_match = re.search(addr_pattern, request_id)
        inst_match = re.search(instance_pattern, request_id)

        if addr_match and inst_match:
            ip = addr_match.group(1)
            port = int(addr_match.group(2))
            instance_id = int(inst_match.group(1))

            return ip, port, instance_id

        missing = []
        if not addr_match: missing.append("Address/Port")
        if not inst_match: missing.append("Instance ID")
        
        raise ValueError(
            f"Request id parse failed. Missing: {', '.join(missing)}. "
            f"Context(is_prefill={is_prefill}), ID: {request_id}"
        )

    # region
    # def _listen_for_requests(self):
    #     while True:
    #         socks = dict(self.poller.poll())
    #         print(f"socks_: {socks}")
    #         if self.router_socket in socks:
    #             remote_address, message = self.router_socket.recv_multipart()
    #             print(f"remote_address in recive: {remote_address}")
    #             data = msgpack.loads(message)
    #             if data["cmd"] == "NEW":
    #                 unique_id = self.nccl.unique_id_from_bytes(
    #                     bytes(data["unique_id"]))
    #                 print(f"create p2p worker {unique_id}")
    #                 with torch.cuda.device(self.device):
    #                     rank = 1
    #                     with set_p2p_nccl_context(self.nccl_num_channels):
    #                         comm: ncclComm_t = self.nccl.ncclCommInitRank(
    #                             2, unique_id, rank)
    #                     self.comms[remote_address.decode()] = (comm, rank)
    #                     logger.info(
    #                         "🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
    #                         self.zmq_address, remote_address.decode(), rank)
    #                     print(f"success init P2P")
                    
    #             elif data["cmd"] == "PUT":
    #                 tensor_id = data["tensor_id"]
    #                 try:
    #                     with torch.cuda.stream(self.recv_stream):
    #                         tensor = torch.empty(data["shape"],
    #                                              dtype=getattr(
    #                                                  torch, data["dtype"]),
    #                                              device=self.device)
    #                     self.router_socket.send_multipart(
    #                         [remote_address, b"0"])
    #                     comm, rank = self.comms[remote_address.decode()]
    #                     self._recv(comm, tensor, rank ^ 1, self.recv_stream)
    #                     tensor_size = tensor.element_size() * tensor.numel()
    #                     if (self.buffer_size + tensor_size
    #                             > self.buffer_size_threshold):
    #                         # Store Tensor in memory pool
    #                         addr = self.pool.store_tensor(tensor)
    #                         tensor = (addr, tensor.dtype, tensor.shape)
    #                         logger.warning(
    #                             "🔴[PUT]Recv Tensor, Out Of Threshold, "
    #                             "%s👈%s, data:%s, addr:%d", self.zmq_address,
    #                             remote_address.decode(), data, addr)
    #                     else:
    #                         self.buffer_size += tensor_size

    #                 except torch.cuda.OutOfMemoryError:
    #                     self.router_socket.send_multipart(
    #                         [remote_address, b"1"])
    #                     tensor = None
    #                     logger.warning(
    #                         "🔴[PUT]Recv Tensor, Out Of Memory, %s👈%s, "
    #                         "data:%s", self.zmq_address,
    #                         remote_address.decode(), data)

    #                 with self.recv_store_cv:
    #                     self.recv_store[tensor_id] = tensor
    #                     self._have_received_tensor_id(tensor_id)
    #                     self.recv_store_cv.notify()

    #             elif data["cmd"] == "GET":
    #                 tensor_id = data["tensor_id"]
    #                 with self.send_store_cv:
    #                     tensor = self.send_store.pop(tensor_id, None)
    #                     if tensor is not None:
    #                         data = {
    #                             "ret": 0,
    #                             "shape": tensor.shape,
    #                             "dtype":
    #                             str(tensor.dtype).replace("torch.", "")
    #                         }
    #                         # LRU
    #                         self.send_store[tensor_id] = tensor
    #                         self._have_sent_tensor_id(tensor_id)
    #                     else:
    #                         data = {"ret": 1}

    #                 self.router_socket.send_multipart(
    #                     [remote_address, msgpack.dumps(data)])

    #                 if data["ret"] == 0:
    #                     comm, rank = self.comms[remote_address.decode()]
    #                     self._send(comm, tensor.to(self.device), rank ^ 1,
    #                                self.send_stream)
    #             else:
    #                 logger.warning(
    #                     "🚧Unexpected, Received message from %s, data:%s",
    #                     remote_address, data)
    # endregion 

    def _listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket in socks:
                remote_address, message = self.router_socket.recv_multipart()
                data = msgpack.loads(message)
                
                if data["cmd"] == "NEW":
                    pass
                    # (保持原有逻辑或按需处理)
                    
                elif data["cmd"] == "PUT":
                    tensor_id = data["tensor_id"]
                    try:
                        with torch.cuda.stream(self.recv_stream):
                            # 使用 randn 模拟有数据的 Tensor，或者用 empty 追求极致分配速度
                            tensor = torch.randn(data["shape"],
                                                 dtype=getattr(torch, data["dtype"]),
                                                 device=self.device)
                        
                        # 发送 ACK，保持 ZeroMQ 的协议握手完成
                        self.router_socket.send_multipart([remote_address, b"0"])
                        
                        # 模拟一下同步（可选），确保流操作被记录
                        # self.recv_stream.synchronize() 

                        # 内存统计逻辑保持不变，用于测试显存压力
                        tensor_size = tensor.element_size() * tensor.numel()
                        if (self.buffer_size + tensor_size > self.buffer_size_threshold):
                            # Store Tensor in memory pool (Offload simulation)
                            addr = self.pool.store_tensor(tensor)
                            tensor = (addr, tensor.dtype, tensor.shape)
                            logger.warning(
                                "🔴[PUT]Mock Recv, Out Of Threshold, %s👈%s, addr:%d", 
                                self.zmq_address, remote_address.decode(), addr)
                        else:
                            self.buffer_size += tensor_size

                    except torch.cuda.OutOfMemoryError:
                        self.router_socket.send_multipart([remote_address, b"1"])
                        tensor = None
                        logger.warning("🔴[PUT]Mock Recv, Out Of Memory")

                    # 存入 recv_store，通知等待者
                    # with self.recv_store_cv:
                    #     self.recv_store[tensor_id] = tensor
                    #     self._have_received_tensor_id(tensor_id)
                    #     self.recv_store_cv.notify()
                    with self.recv_store_cv:
                        tensor_id, worker_id = tensor_id.split("|")
                        if tensor_id not in self.recv_store_asymmetric:
                            self.recv_store_asymmetric[tensor_id] = {}

                        self.recv_store_asymmetric[tensor_id][int(worker_id)] = tensor
                        self._have_received_tensor_id(tensor_id)
                        self.recv_store_cv.notify()


                elif data["cmd"] == "GET":
                    tensor_id = data["tensor_id"]
                    with self.send_store_cv:
                        tensor = self.send_store.pop(tensor_id, None)
                        if tensor is not None:
                            data = {
                                "ret": 0,
                                "shape": tensor.shape,
                                "dtype": str(tensor.dtype).replace("torch.", "")
                            }
                            # LRU
                            self.send_store[tensor_id] = tensor
                            self._have_sent_tensor_id(tensor_id)
                        else:
                            data = {"ret": 1}

                    # 发送元数据给请求方
                    self.router_socket.send_multipart(
                        [remote_address, msgpack.dumps(data)])

                    if data["ret"] == 0:
                        pass
                        # comm, rank = self.comms[remote_address.decode()]
                        # self._send(comm, tensor.to(self.device), rank ^ 1, self.send_stream) <--- 注释掉实际传输
                else:
                    logger.warning(
                        "🚧Unexpected, Received message from %s, data:%s",
                        remote_address, data)


    def _have_sent_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split('#')[0]
        if request_id not in self.send_request_id_to_tensor_ids:
            self.send_request_id_to_tensor_ids[request_id] = set()
        self.send_request_id_to_tensor_ids[request_id].add(tensor_id)

    def _have_received_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split('#')[0]
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)

    def _asymmetric_received_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split('#')[0]
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)

    def _send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                tensor_id, remote_address, tensor = self.send_queue.popleft()
                if not self.send_queue:
                    self.send_queue_cv.notify()
            self._send_sync(tensor_id, tensor, remote_address)

    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.send_queue_cv:
                while self.send_queue:
                    self.send_queue_cv.wait()
            duration = time.time() - start_time
            logger.debug(
                "🚧[PUT_ASYNC]It took %.3fms to wait for the send_queue"
                " to be empty, rank:%d", duration * 1000, self.rank)

    def _send_sync(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: typing.Optional[str] = None,
    ) -> bool:
        if remote_address is None:
            return False
        if remote_address not in self.socks:
            self._create_connect(remote_address)

        sock = self.socks[remote_address]
        # comm, rank = self.comms[remote_address]
        rank = 0

        data = {
            "cmd": "PUT",
            "tensor_id": tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", "")
        }
        sock.send(msgpack.dumps(data))

        response = sock.recv()
        if response != b"0":
            logger.error(
                "🔴Send Tensor, Peer Out Of Memory/Threshold, %s 👉 %s, "
                "MyRank:%s, data:%s, tensor:%s, size:%fGB, response:%s",
                self.zmq_address, remote_address, rank, data, tensor.shape,
                tensor.element_size() * tensor.numel() / 1024**3,
                response.decode())
            return False

        # self._send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)

        if self.send_type == "PUT_ASYNC":
            self._have_sent_tensor_id(tensor_id)

        return True

    def get_finished(
        self, finished_req_ids: set[str], forward_context: "ForwardContext"
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        # Clear the buffer upon request completion.
        for request_id in finished_req_ids:
            for layer_name in forward_context.no_compile_layers:
                tensor_id = request_id + "#" + layer_name
                if tensor_id in self.recv_store:
                    with self.recv_store_cv:
                        tensor = self.recv_store.pop(tensor_id, None)
                        self.send_request_id_to_tensor_ids.pop(
                            request_id, None)
                        self.recv_request_id_to_tensor_ids.pop(
                            request_id, None)
                    addr = 0
                    if isinstance(tensor, tuple):
                        addr, _, _ = tensor
                        self.pool.free(addr)

        # TODO:Retrieve requests that have already sent the KV cache.
        finished_sending: set[str] = set()

        # TODO:Retrieve requests that have already received the KV cache.
        finished_recving: set[str] = set()

        return finished_sending or None, finished_recving or None

    def _ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)

    def _send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            self.nccl.ncclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), dst,
                               comm, cudaStream_t(stream.cuda_stream))
        stream.synchronize()

    def _recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            self.nccl.ncclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), src,
                               comm, cudaStream_t(stream.cuda_stream))
        stream.synchronize()

    def close(self) -> None:
        self._listener_thread.join()
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
