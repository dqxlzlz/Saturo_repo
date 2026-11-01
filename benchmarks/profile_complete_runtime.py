#!/usr/bin/env python3
"""Profile complete offline worker execution and its model components.

Baseline: complete worker.execute_model, with only two outer CUDA events.
Detailed: identical full-model execution with existing module events plus
prepare_input wall time and forward/logits/sampler wall/CUDA ranges.
All TP workers participate. No scheduler/request-arrival time is synthesized.
Results are append-only, checkpointed, and version/config checked on resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = 1


def make_configs(grid='dense'):
    configs = []
    def add(family, pf, dc, pf_final=True):
        # pf entries are (query tokens, cached context); dc are cached lengths.
        payload = dict(family=family, prefill=pf, decode=dc, prefill_final=pf_final)
        payload['id'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        payload['n_tokens'] = sum(q for q,c in pf)+len(dc)
        payload['n_requests'] = len(pf)+len(dc)
        configs.append(payload)
    if grid == 'pilot':
        for n in [32,1024,8192]: add('linear', [(n,0)], [])
        add('prefill', [(512,0)]*8, [])
        add('decode', [], [2048]*32)
        add('decode', [], [4096]*128)
        add('mixed', [(980,1024)], [4096]*44,False)
        add('mixed', [(3968,4096)], [8192]*128,False)
        return configs
    if grid == 'dense':
        tokens=range(2,8193,2)
        lengths=range(32,4097,32)
        batches=range(1,129)
        decode_lengths=range(32,4097,32)
        decode_batches=range(16,513,16)
    else:
        tokens=sorted(set(range(32,8193,32))|{1,2,4,8,16})
        lengths=[32,64,128,256,512,768,1024,1536,2048,3072,4096]
        batches=[1,2,4,8,16,32,64,128]
        decode_lengths=[32,128,256,512,1024,2048,4096,8192,16384]
        decode_batches=[1,2,4,8,16,32,48,64,96,128,192,256,384,512]
    for n in tokens: add('linear',[(n,0)],[])
    for length in lengths:
        for batch in batches:
            if length*batch<=12000: add('prefill',[(length,0)]*batch,[])
    for length in decode_lengths:
        for batch in decode_batches: add('decode',[],[length]*batch)
    # Mixed/chunked held-out grid: absent from the component fitting families.
    for budget in [256,512,768,1024,1536,2048,4096]:
        for decode_batch in [8,32,64,128]:
            for cached in [0,1024,4096]:
                for decode_context in [512,2048,8192]:
                    if decode_batch < budget:
                        add('mixed',[(budget-decode_batch,cached)],
                            [decode_context]*decode_batch,False)
    return configs


def blocks_required(config, block_size=16):
    # Non-final prefill includes one uncomputed future token in metadata;
    # only the currently scheduled positions need allocated KV blocks.
    return sum(math.ceil((q+c)/block_size) for q,c in config['prefill']) + sum(
        math.ceil((c+1)/block_size) for c in config['decode'])


def stats(values):
    values=[float(x) for x in values]
    ordered=sorted(values)
    lo=max(0,(len(ordered)-3)//2)
    return dict(each=values,mean=statistics.mean(values),median=statistics.median(values),
                mid3=statistics.mean(ordered[lo:lo+3]),min=min(values),max=max(values))


def build_request(config, block_size):
    from vllm import SamplingParams
    from vllm.sequence import ExecuteModelRequest,SequenceData,SequenceGroupMetadata
    cursor=0
    groups=[]
    sampling=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,detokenize=False)
    def blocks(length):
        nonlocal cursor
        count=math.ceil(length/block_size)
        ids=list(range(cursor,cursor+count));cursor+=count
        return {0:ids}
    for i,(query,cached) in enumerate(config['prefill']):
        final=config['prefill_final']
        prompt=[1]*(query+cached+(0 if final else 1))
        sequence=SequenceData.from_seqs(prompt)
        if cached: sequence.update_num_computed_tokens(cached)
        groups.append(SequenceGroupMetadata(
            request_id=f'complete-prefill-{i}',is_prompt=True,seq_data={0:sequence},
            sampling_params=sampling,block_tables=blocks(query+cached),
            do_sample=final,token_chunk_size=query))
    for i,cached in enumerate(config['decode']):
        sequence=SequenceData.from_seqs([1]*cached,output_token_ids=[2])
        sequence.update_num_computed_tokens(cached)
        groups.append(SequenceGroupMetadata(
            request_id=f'complete-decode-{i}',is_prompt=False,seq_data={0:sequence},
            sampling_params=sampling,block_tables=blocks(cached+1),
            do_sample=True,token_chunk_size=1))
    return ExecuteModelRequest(seq_group_metadata_list=groups,virtual_engine=0,num_steps=1,
                               blocks_to_swap_in=[],blocks_to_swap_out=[],blocks_to_copy=[])


def profile_worker_batch(wrapper, configs, warmup, repeat, detailed_repeat):
    import functools
    import torch
    worker=wrapper.worker
    runner=worker.model_runner
    result=[]
    old_flag=os.environ.get('VLLM_COLLECT_REAL_MODULE_TIMES')
    originals=[]
    state={'enabled':False,'wall':{},'events':{}}

    def wrap(obj,name,label,gpu=False):
        original=getattr(obj,name)
        @functools.wraps(original)
        def timed(*args,**kwargs):
            if not state['enabled']: return original(*args,**kwargs)
            if gpu:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
            t0=time.perf_counter()
            value=original(*args,**kwargs)
            state['wall'][label]=(time.perf_counter()-t0)*1000
            if gpu:
                end.record();state['events'][label]=(start,end)
            return value
        originals.append((obj,name,original))
        setattr(obj,name,timed)

    wrap(worker,'prepare_input','prepare_input')
    wrap(runner.model,'forward','model_forward',True)
    wrap(runner.model,'compute_logits','logits',True)
    wrap(runner.sampler,'forward','sampling',True)

    def once(config,detailed):
        request=build_request(config,worker.cache_config.block_size)
        state.update(enabled=detailed,wall={},events={})
        torch.cuda.synchronize()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();t0=time.perf_counter()
        outputs=worker.execute_model(request)
        end.record();torch.cuda.synchronize()
        row=dict(worker_wall_ms=(time.perf_counter()-t0)*1000,
                 worker_cuda_ms=start.elapsed_time(end))
        if detailed:
            row.update({name+'_wall_ms':value for name,value in state['wall'].items()})
            row.update({name+'_cuda_ms':s.elapsed_time(e) for name,(s,e) in state['events'].items()})
            modules=next((getattr(o,'module_profile_times',None) for o in (outputs or [])
                          if getattr(o,'module_profile_times',None)),None)
            if modules: row.update(modules)
        if worker.is_driver_worker:
            assert outputs and all(o is not None for o in outputs),'No sampler output'
            if detailed:
                assert row.get('actual_linear_ms',0)>0,'Missing module timing'
                assert 'sampling_wall_ms' in row,'Sampling path was not exercised'
                assert 'logits_cuda_ms' in row and 'model_forward_cuda_ms' in row
        return row
    try:
        for config in configs:
            record={'id':config['id'],'rank':worker.rank}
            # Alternate ordering by config ID to limit systematic ordering bias.
            order=[False,True] if int(config['id'][-1],16)%2==0 else [True,False]
            for detailed in order:
                os.environ['VLLM_COLLECT_REAL_MODULE_TIMES']='1' if detailed else '0'
                for _ in range(warmup): once(config,detailed)
                samples=[once(config,detailed) for _ in range(detailed_repeat if detailed else repeat)]
                metrics={key:stats([r[key] for r in samples]) for key in samples[0]}
                if not worker.is_driver_worker:
                    metrics={k:v for k,v in metrics.items() if k in ('worker_wall_ms','worker_cuda_ms')}
                record['detailed' if detailed else 'baseline']=metrics
            result.append(record)
    finally:
        for obj,name,original in originals: setattr(obj,name,original)
        if old_flag is None:os.environ.pop('VLLM_COLLECT_REAL_MODULE_TIMES',None)
        else:os.environ['VLLM_COLLECT_REAL_MODULE_TIMES']=old_flag
    return result


def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2))
    temporary.replace(path)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--grid',choices=['pilot','standard','dense'],default='dense')
    p.add_argument('--tensor-parallel-size',type=int,default=4)
    p.add_argument('--dtype',default='bfloat16')
    p.add_argument('--load-format',default='dummy')
    p.add_argument('--gpu-memory-utilization',type=float,default=.90)
    p.add_argument('--max-model-len',type=int,default=32768)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--detailed-repeat',type=int,default=3)
    p.add_argument('--chunk-configs',type=int,default=8)
    p.add_argument('--plan-only',action='store_true')
    return p.parse_args()


def main():
    args=parse_args()
    configs=make_configs(args.grid)
    if args.plan_only:
        print(json.dumps(dict(total=len(configs),families={f:sum(c['family']==f for c in configs)
            for f in ['linear','prefill','decode','mixed']}),indent=2));return
    for key,value in dict(VLLM_USE_V1='0',VLLM_TARGET_DEVICE='cuda',
                         VLLM_ATTENTION_BACKEND='FLASH_ATTN',VLLM_WORKER_MULTIPROC_METHOD='spawn',
                         SATURO_SKIP_TIME_PREDICTOR_PROFILE='1',VLLM_COLLECT_REAL_MODULE_TIMES='0',
                         TORCHDYNAMO_DISABLE='1',TORCH_COMPILE_DISABLE='1').items():
        os.environ[key]=value
    import torch
    from vllm import LLM
    from vllm.config import CompilationConfig
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    configuration=dict(vars(args),schema=SCHEMA,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest_path=out/'manifest.json'
    source_files=['vllm/worker/worker_base.py','vllm/worker/model_runner.py',
                  'vllm/model_executor/models/qwen2.py','vllm/model_executor/models/llama.py']
    current_source={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files}
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        assert old['configuration']==configuration,'Resume configuration/source mismatch'
        assert old['source_sha256']==current_source,'Worker/model source changed since initial run'
    else:
        manifest=dict(configuration=configuration,started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      model_config=json.loads((Path(args.model)/'config.json').read_text()),
                      source_sha256=current_source,
                      torch_version=torch.__version__,
                      gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv'],text=True),
                      measurement='Full worker execution; baseline excludes module instrumentation; detailed is a separate run.')
        atomic_json(manifest_path,manifest)
    raw=out/'raw.jsonl'
    completed={}
    if raw.exists():
        # A torn final append can safely be discarded; interior corruption fails.
        lines=raw.read_bytes().splitlines(keepends=True);valid=[]
        for i,line in enumerate(lines):
            try:r=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
                break
            completed[r['config']['id']]=r;valid.append(line if line.endswith(b'\n') else line+b'\n')
        raw.write_bytes(b''.join(valid))
    start=time.perf_counter();initial=len(completed)
    llm=None
    status={'state':'INITIALIZING','planned':len(configs),'completed':len(completed)}
    atomic_json(out/'status.json',status)
    try:
        llm=LLM(model=args.model,tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=1,dtype=args.dtype,load_format=args.load_format,
                enforce_eager=True,seed=0,trust_remote_code=True,skip_tokenizer_init=True,
                gpu_memory_utilization=args.gpu_memory_utilization,max_model_len=args.max_model_len,
                max_num_seqs=512,max_num_batched_tokens=12000,enable_chunked_prefill=True,
                disable_custom_all_reduce=True,disable_log_stats=True,
                compilation_config=CompilationConfig(level=0,custom_ops=['none'],use_cudagraph=False,use_inductor=False))
        executor=llm.llm_engine.model_executor
        cache=llm.llm_engine.cache_config
        capacity=int(cache.num_gpu_blocks*.90)
        skipped=[dict(config=c,reason='KV block budget',required_blocks=blocks_required(c,cache.block_size),available_blocks=capacity)
                 for c in configs if blocks_required(c,cache.block_size)>capacity]
        skip_ids={r['config']['id'] for r in skipped}
        valid=[c for c in configs if c['id'] not in skip_ids]
        atomic_json(out/'grid.json',dict(all_configs=configs,valid=len(valid),skipped=skipped,
                                      block_size=cache.block_size,num_gpu_blocks=cache.num_gpu_blocks))
        remaining=[c for c in valid if c['id'] not in completed]
        with raw.open('a') as fh:
            for offset in range(0,len(remaining),args.chunk_configs):
                batch=remaining[offset:offset+args.chunk_configs]
                status.update(state='RUNNING',current_family=batch[0]['family'],current_id=batch[0]['id'],
                              valid=len(valid),skipped=len(skipped),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                atomic_json(out/'status.json',status)
                ranks=executor._run_workers(profile_worker_batch,batch,args.warmup,args.repeat,args.detailed_repeat)
                for idx,config in enumerate(batch):
                    row=dict(config=config,driver=ranks[0][idx],ranks=[r[idx] for r in ranks[1:]])
                    fh.write(json.dumps(row,separators=(',',':'))+'\n');fh.flush();os.fsync(fh.fileno())
                    completed[config['id']]=row
                elapsed=time.perf_counter()-start;new=len(completed)-initial
                eta=elapsed/max(new,1)*(len(valid)-len(completed))
                status.update(completed=len(completed),elapsed_s=elapsed,eta_s=eta)
                atomic_json(out/'status.json',status)
                print(f"PROGRESS {len(completed)}/{len(valid)} family={batch[-1]['family']} elapsed={elapsed:.1f}s eta={eta/60:.1f}min",flush=True)
        status.update(state='COMPLETED',completed=len(completed),finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        atomic_json(out/'status.json',status)
    except BaseException as exc:
        status.update(state='FAILED',error=repr(exc));atomic_json(out/'status.json',status);raise
    finally:
        if llm is not None:llm.llm_engine.model_executor.shutdown()


if __name__=='__main__':main()
