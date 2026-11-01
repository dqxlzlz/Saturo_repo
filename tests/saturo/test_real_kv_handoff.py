"""Real tensor handoff, TP/PP resharding, fail-closed import and dummy compatibility."""
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from saturo.runtime.kv_handoff import (build_transfer_metadata, new_handoff,
    prompt_digest, real_handoff, transfer_directory)
from saturo.runtime.workload import SaturoRequest, load_workload_csv
from vllm.distributed.kv_transfer.kv_connector.saturo_shared_storage_connector import (
    SaturoSharedStorageConnector, _head_range)


def config(root, tp=1, pp=1, rank=0, heads=4):
    hf = NS(num_key_value_heads=heads, num_attention_heads=4,
            hidden_size=16, head_dim=4, sliding_window=None)
    hf.to_dict = lambda: {'heads': heads, 'layers': 4}
    model = NS(hf_text_config=hf, model='same-model', dtype=torch.float32,
               is_deepseek_mla=False,
               get_layers_start_end_indices=lambda p: ((rank//tp)*4//pp, ((rank//tp)+1)*4//pp))
    return NS(kv_transfer_config=NS(kv_connector_extra_config={
        'directory': str(root), 'namespace': 'run', 'timeout_s': .02}),
        model_config=model, parallel_config=NS(tensor_parallel_size=tp,
            pipeline_parallel_size=pp, rank=rank),
        cache_config=NS(block_size=4,cache_dtype='auto'), lora_config=None)


def canonical_cache():
    # Each tensor element identifies K/V, layer, token, global head and channel.
    return torch.arange(2*4*7*4*4,dtype=torch.float32).reshape(2,4,7,4,4)


def item(op='send', source_tp=1, source_pp=1, blocks=(2,0)):
    tokens = list(range(7))
    return dict(operation=op,key='request1',prompt_len=7,prompt_digest=prompt_digest(tokens),
        prompt_digest_observed=prompt_digest(tokens),source_tp=source_tp,source_pp=source_pp,
        request_id='rt:p1' if op=='send' else 'rt:d1',seq_id=1,block_table=list(blocks))


def model_input(record):
    return NS(saturo_kv_transfers=[record], virtual_engine=0, finished_requests_ids=[])


@pytest.mark.parametrize('source_tp,target_tp,source_pp,target_pp,heads',[
    (1,1,1,1,4),(1,2,1,1,4),(2,1,1,1,4),(2,4,2,1,4),
    (1,2,1,2,4),(2,1,2,2,4),(4,1,1,1,2),(1,4,1,1,2)])
def test_real_tensors_reshard_and_import_to_different_physical_blocks(
        tmp_path,source_tp,target_tp,source_pp,target_pp,heads):
    full=canonical_cache()[:,:,:,:heads]
    for rank in range(source_tp*source_pp):
        connector=SaturoSharedStorageConnector(rank,0,config(tmp_path,source_tp,source_pp,rank,heads))
        first,last=connector.layer_start,connector.layer_end
        h0,h1=_head_range(heads,source_tp,rank%source_tp)
        caches=[torch.full((2,4,4,h1-h0,4),float('nan')) for _ in range(first,last)]
        for i,layer in enumerate(range(first,last)):
            for kv in range(2):
                slots=torch.tensor([8,9,10,11,0,1,2])
                caches[i][kv].flatten(0,1).index_copy_(0,slots,full[kv,layer,:,h0:h1])
        connector.send_kv_caches_and_hidden_states(None,model_input(item(
            source_tp=source_tp,source_pp=source_pp)),caches,None)
        # Destroy the producer caches: receiver must load the exported tensors.
        for cache in caches:cache.fill_(float('nan'))
    for rank in range(target_tp*target_pp):
        connector=SaturoSharedStorageConnector(rank,0,config(tmp_path,target_tp,target_pp,rank,heads))
        first,last=connector.layer_start,connector.layer_end
        h0,h1=_head_range(heads,target_tp,rank%target_tp)
        caches=[torch.full((2,4,4,h1-h0,4),-99.) for _ in range(first,last)]
        incoming=item('recv',source_tp,source_pp,blocks=(1,3))
        result=connector.recv_kv_caches_and_hidden_states(None,model_input(incoming),caches)
        assert result[1] is False  # A real decode forward still runs.
        slots=torch.tensor([4,5,6,7,12,13,14])
        for i,layer in enumerate(range(first,last)):
            for kv in range(2):
                assert torch.equal(caches[i][kv].flatten(0,1)[slots],full[kv,layer,:,h0:h1])
                assert bool((caches[i][kv,0]==-99).all())
        # Reusing a target must not roll back KV written by subsequent decodes.
        for cache in caches:cache.fill_(123.)
        connector.recv_kv_caches_and_hidden_states(None,model_input(incoming),caches)
        assert bool((caches[0]==123).all())


def test_missing_or_wrong_kv_never_falls_back_to_dummy(tmp_path):
    connector=SaturoSharedStorageConnector(0,0,config(tmp_path))
    caches=[torch.zeros(2,4,4,4,4) for _ in range(4)]
    with pytest.raises(TimeoutError,match='no dummy fallback'):
        connector.recv_kv_caches_and_hidden_states(None,model_input(item('recv')),caches)
    wrong=item('recv');wrong['prompt_digest_observed']='wrong'
    with pytest.raises(ValueError,match='prompt token IDs'):
        connector.recv_kv_caches_and_hidden_states(None,model_input(wrong),caches)
    for namespace,key in [('..','request'),('run','../request')]:
        with pytest.raises(ValueError):transfer_directory(tmp_path,namespace,key)


def test_final_prefill_chunk_exports_full_block_table_and_ignores_local_be():
    record=item()
    params=NS(extra_args={'saturo_kv_handoff':record})
    data=NS(get_num_computed_tokens=lambda:3,get_prompt_token_ids=lambda:tuple(range(7)))
    group=NS(sampling_params=params,seq_data={5:data},is_prompt=True,token_chunk_size=2,
             request_id='rt:p',block_tables={5:[2,0]})
    assert build_transfer_metadata([group])==[]
    group.token_chunk_size=4
    be=NS(sampling_params=NS(extra_args={}))
    result=build_transfer_metadata([group,be])
    assert len(result)==1 and result[0]['block_table']==[2,0]


def test_tokenized_workload_and_request_mode_compatibility(tmp_path):
    path=tmp_path/'workload.csv'
    path.write_text('request_id,prompt_len,output_len,arrival_offset_s,prompt_token_ids\nr,2,4,0,"[11, 12]"\n')
    assert load_workload_csv(str(path),'rt')[0].prompt_token_ids==[11,12]
    assert not real_handoff(NS())
    assert real_handoff({'kv_handoff':'real'})


def test_worker_uses_real_p_token_and_preserves_dummy_direct_decode():
    from saturo.runtime.worker import RayEngineWorker
    worker=RayEngineWorker.__new__(RayEngineWorker)
    worker.worker_role='decode';worker._use_dummy_kv=False
    worker.rt_prefill_deadline_s=1
    worker.SamplingParams=lambda **kwargs:NS(**kwargs)
    command=dict(kind='rt',phase='decode',prompt_len=3,max_tokens=4,
                 prompt_token_ids=[11,12,99],kv_handoff=item(),request_id='r')
    worker._role_args={'kv_handoff':'real'}
    prompt,params,_=worker._request_inputs(command)
    assert prompt['prompt_token_ids']==[11,12,99]
    assert params.extra_args['saturo_direct_decode'] is True
    assert params.extra_args['saturo_kv_handoff']['operation']=='recv'
    worker._role_args={'kv_handoff':'dummy'}
    prompt,params,_=worker._request_inputs(command)
    assert prompt['prompt_token_ids']==[1,1,1]
    assert 'saturo_kv_handoff' not in params.extra_args


def test_global_handoff_carries_sampled_context_and_retry_reprefills(tmp_path):
    from saturo.runtime.scheduler import SaturoGlobalScheduler
    scheduler=SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler.args=NS(kv_handoff='real',kv_transfer_dir=str(tmp_path),rt_ttft_slo_ms=1000)
    scheduler._kv_transfer_namespace='run'
    scheduler._elapsed=lambda:0
    scheduler._reserve_decode_for_prefill=lambda r:True
    scheduler._release_decode_reservation=lambda r,**k:None
    scheduler._has_terminal_failure=lambda:False
    worker=NS(worker_id=0,retired=False,ready=True,role='prefill',active_attempts={},
              pending_commands=[],tensor_parallel_size=2,pipeline_parallel_size=1)
    req=SaturoRequest('r','rt',2,5,0,prompt_token_ids=[11,12])
    assert scheduler._assign(worker,req,'prefill')
    outgoing=worker.pending_commands[-1]
    assert outgoing['kv_handoff']['source_tp']==2
    req.generated_token_ids=[99];req.observed_tokens=1;req.active_attempt_id=None
    scheduler._assign(worker,req,'decode')
    outgoing=worker.pending_commands[-1]
    assert outgoing['prompt_token_ids']==[11,12,99]
    assert outgoing['max_tokens']==4
    req.active_attempt_id=None
    scheduler.pending_prefill=deque();scheduler.pending_decode=deque()
    scheduler.role_switcher=NS();scheduler._requeue(req,'decode')
    assert list(scheduler.pending_prefill)==[req]
    assert not scheduler.pending_decode


def test_cli_forwards_real_mode_and_requires_shared_path(tmp_path):
    from saturo.cli.endtoend_experiment import parse_args, build_command
    common=['--model','model','--num-req','3','--target-rps','1',
            '--resource-config-filter','tp1_pp1','--rt-workload-csv','rt.csv',
            '--be-workload-csv','be.csv']
    assert parse_args(common).kv_handoff=='dummy'
    with pytest.raises(SystemExit):parse_args(common+['--kv-handoff','real'])
    with pytest.raises(SystemExit):parse_args(common+['--kv-handoff','real',
        '--kv-transfer-dir',str(tmp_path),'--load-format','dummy'])
    args=parse_args(common+['--kv-handoff','real','--kv-transfer-dir',str(tmp_path)])
    cmd=build_command(args)
    assert cmd[cmd.index('--kv-handoff')+1]=='real'
    assert cmd[cmd.index('--kv-transfer-dir')+1]==str(tmp_path)


def test_real_transfer_metadata_survives_tp_broadcast():
    from vllm.worker.model_runner import ModelInputForGPU, ModelInputForGPUWithSamplingMetadata
    for cls in (ModelInputForGPU,ModelInputForGPUWithSamplingMetadata):
        original=cls(saturo_kv_transfers=[item()])
        # Attention and sampling metadata are absent in this metadata-only check.
        broadcast=original.as_broadcastable_tensor_dict()
        assert broadcast['saturo_kv_transfers']==[item()]
        restored=cls.from_broadcasted_tensor_dict(broadcast)
        assert restored.saturo_kv_transfers==original.saturo_kv_transfers
