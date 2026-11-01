#!/usr/bin/env python3
"""Sparse baseline-only offline profiling with durable per-point local ACKs.

Each configuration gets one warmup and three timed worker executions. A record
must be fsynced by the local receiver before another configuration starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.profile_complete_runtime import build_request, blocks_required

PREFIX = 'PROFILE_SYNC_V1 '


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as fh:
        json.dump(value, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)


def transfer(kind, name, value):
    print(PREFIX + json.dumps(dict(kind=kind, name=name, value=value)), flush=True)
    if sys.stdin.readline().strip() != 'ACK ' + name:
        raise RuntimeError('Local durable-sync acknowledgement missing: ' + name)


def baseline_once(wrapper, config, repeat):
    import torch
    worker = wrapper.worker
    os.environ['VLLM_COLLECT_REAL_MODULE_TIMES'] = '0'
    # The first call warms this exact batch; input construction is untimed.
    samples = []
    for measured in [False] + [True] * repeat:
        request = build_request(config, worker.cache_config.block_size)
        torch.cuda.synchronize()
        if measured:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            t0 = time.perf_counter()
        outputs = worker.execute_model(request)
        if measured:
            end.record()
        torch.cuda.synchronize()
        if measured:
            wall_ms = (time.perf_counter() - t0) * 1000
            cuda_ms = start.elapsed_time(end)
            samples.append(dict(worker_cuda_ms=cuda_ms, worker_wall_ms=wall_ms))
        if worker.is_driver_worker:
            assert outputs and all(o is not None for o in outputs)
    return dict(rank=worker.rank, samples=samples,
                worker_cuda_ms=statistics.median(s['worker_cuda_ms'] for s in samples),
                worker_wall_ms=statistics.median(s['worker_wall_ms'] for s in samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--repeat', type=int, default=3, choices=[3, 4, 5])
    parser.add_argument('--tp', type=int, default=4, choices=[1, 2, 4])
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    assert plan['repeat'] == args.repeat
    assert plan.get('tp', 4) == args.tp
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for key, value in dict(VLLM_USE_V1='0', VLLM_TARGET_DEVICE='cuda',
                          VLLM_ATTENTION_BACKEND='FLASH_ATTN',
                          VLLM_WORKER_MULTIPROC_METHOD='spawn',
                          SATURO_SKIP_TIME_PREDICTOR_PROFILE='1',
                          VLLM_COLLECT_REAL_MODULE_TIMES='0',
                          TORCHDYNAMO_DISABLE='1', TORCH_COMPILE_DISABLE='1',
                          OMP_NUM_THREADS='1').items():
        os.environ[key] = value
    import torch
    from vllm import LLM
    from vllm.config import CompilationConfig
    sources = ['benchmarks/profile_quick_runtime.py',
               'benchmarks/profile_complete_runtime.py',
               'vllm/worker/worker_base.py', 'vllm/worker/model_runner.py',
               'vllm/model_executor/models/qwen2.py', 'vllm/model_executor/models/llama.py']
    assert torch.cuda.device_count() == args.tp, 'Visible device count must match TP'
    manifest = dict(schema='quick_baseline_v3', model=args.model, tp=args.tp, pp=1,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    devices=[dict(name=torch.cuda.get_device_name(i),
                                  total_memory=torch.cuda.get_device_properties(i).total_memory)
                             for i in range(args.tp)],
                    dtype='bfloat16', load_format='dummy', eager=True,
                    custom_ops=['none'], cuda_graph=False, custom_all_reduce=False,
                    max_model_len=32768, max_num_seqs=512, max_num_batched_tokens=12000,
                    gpu_memory_utilization=0.9, warmup=1, repeat=args.repeat,
                    torch_version=torch.__version__,
                    source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                    plan_sha256=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
                    measurement='Full worker CUDA interval and wall time; no module events; excludes scheduler.')
    for source, expected in plan['expected_core_sha256'].items():
        assert manifest['source_sha256'][source] == expected, source
    if (out / 'manifest.json').exists():
        assert json.loads((out / 'manifest.json').read_text()) == manifest, 'Resume mismatch'
    atomic_json(out / 'manifest.json', manifest)
    transfer('file', 'manifest.json', manifest)
    transfer('file', 'plan.json', plan)
    completed = {}
    raw = out / 'raw.jsonl'
    if raw.exists():
        for line in raw.read_text().splitlines():
            row = json.loads(line)
            completed[row['config']['id']] = row
            transfer('record', row['config']['id'], row)
    status = dict(state='INITIALIZING', completed=len(completed), planned=len(plan['new_configs']))

    def save_status():
        status['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        atomic_json(out / 'status.json', status)
        transfer('file', 'status.json', status)

    save_status()
    llm = None
    try:
        llm = LLM(model=args.model, tensor_parallel_size=args.tp, pipeline_parallel_size=1,
                  dtype='bfloat16', load_format='dummy', enforce_eager=True, seed=0,
                  trust_remote_code=True, skip_tokenizer_init=True,
                  gpu_memory_utilization=.90, max_model_len=32768, max_num_seqs=512,
                  max_num_batched_tokens=12000, enable_chunked_prefill=True,
                  disable_custom_all_reduce=True, disable_log_stats=True,
                  compilation_config=CompilationConfig(level=0, custom_ops=['none'],
                                                        use_cudagraph=False, use_inductor=False))
        executor = llm.llm_engine.model_executor
        cache = llm.llm_engine.cache_config
        capacity = int(cache.num_gpu_blocks * .90)
        skipped = [dict(config=c, reason='Actual KV block budget',
                        required_blocks=blocks_required(c, cache.block_size), available_blocks=capacity)
                   for c in plan['new_configs']
                   if c['id'] not in completed and blocks_required(c, cache.block_size) > capacity]
        skip_ids = {s['config']['id'] for s in skipped}
        atomic_json(out / 'skipped.json', skipped)
        transfer('file', 'skipped.json', skipped)
        status.update(state='RUNNING', skipped_count=len(skipped), capacity_blocks=capacity,
                      valid=len(plan['new_configs']) - len(skipped))
        save_status()
        with raw.open('a') as fh:
            for config in plan['new_configs']:
                if config['id'] in completed or config['id'] in skip_ids:
                    continue
                ranks = executor.collective_rpc(baseline_once, args=(config, args.repeat))
                assert sorted(r['rank'] for r in ranks) == list(range(args.tp))
                row = dict(config=config, ranks=ranks, warmup=1, repeat=args.repeat,
                           measured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                fh.write(json.dumps(row, separators=(',', ':')) + '\n')
                fh.flush()
                os.fsync(fh.fileno())
                transfer('record', config['id'], row)
                completed[config['id']] = row
                status.update(completed=len(completed), last_id=config['id'])
                save_status()
        status['state'] = 'COMPLETED'
        save_status()
    except BaseException as exc:
        status.update(state='FAILED', error=repr(exc))
        atomic_json(out / 'status.json', status)
        raise
    finally:
        if llm is not None:
            llm.llm_engine.model_executor.shutdown()


if __name__ == '__main__':
    main()
