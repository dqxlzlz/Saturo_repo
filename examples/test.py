import zmq
import time
import multiprocessing
import os
import threading

# 地址配置
ADDR = "tcp://127.0.0.1:5678"

def receiver_task():
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(ADDR)
    print(f"✅ [Receiver] Bound to {ADDR}")

    while True:
        try:
            identity, msg = socket.recv_multipart()
            sender_id = identity.decode()
            print(f"📥 [Receiver] Received request from {sender_id}")
        except:
            break

        if msg == b"NEW":
            # ==========================================================
            # ⏱️ 修改点 1：将阻塞时间改为 1 秒
            # 模拟 NCCL Init 稍微有点慢，但不是特别慢
            # ==========================================================
            print(f"⏳ [Receiver] Blocking Init for {sender_id} (Simulating NCCL)...")
            time.sleep(1.0) 
            
            socket.send_multipart([identity, b"ACK"])
            print(f"✅ [Receiver] Init Done for {sender_id}")

def sender_task(rank):
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    identity = f"Sender-{rank}"
    socket.setsockopt_string(zmq.IDENTITY, identity)
    socket.connect(ADDR)

    print(f"📤 [{identity}] Sending NEW...")
    socket.send(b"NEW")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    
    # ==========================================================
    # ⏱️ 修改点 2：超时时间设为 1.5 秒
    # 逻辑：
    # 单个请求耗时 1.0s -> 没问题
    # 两个请求排队耗时 2.0s -> 第二个会超时！
    # ==========================================================
    socks = dict(poller.poll(timeout=1500))

    if socket in socks:
        msg = socket.recv()
        print(f"🎉 [{identity}] Connection Success!")
    else:
        print(f"❌ [{identity}] NCCL ERROR: Internal Error / Timeout! (Waited too long)")

if __name__ == "__main__":
    # --- 测试场景 1：单个 Sender (模拟 TP=1) ---
    print("\n========= TEST 1: Single Sender (Should SUCCEED) =========")
    p_recv = multiprocessing.Process(target=receiver_task)
    p_recv.daemon = True
    p_recv.start()
    time.sleep(0.5)

    p_sender1 = multiprocessing.Process(target=sender_task, args=(0,))
    p_sender1.start()
    p_sender1.join()
    
    p_recv.terminate() # 清理环境
    p_recv.join()
    time.sleep(1) # 等端口释放

    # --- 测试场景 2：并发 Sender (模拟 TP=2) ---
    print("\n========= TEST 2: Dual Senders (Sender-1 Should FAIL) =========")
    p_recv = multiprocessing.Process(target=receiver_task)
    p_recv.daemon = True
    p_recv.start()
    time.sleep(0.5)

    p_s0 = multiprocessing.Process(target=sender_task, args=(0,))
    p_s1 = multiprocessing.Process(target=sender_task, args=(1,))
    
    # 同时启动
    p_s0.start()
    p_s1.start()
    
    p_s0.join()
    p_s1.join()
    p_recv.terminate()