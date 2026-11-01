#!/usr/bin/env python3
"""Finite per-host queue; results are durably received after every configuration."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/results/offline_runtime_profile'
parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True, choices=['saturo_1', 'saturo_2'])
args = parser.parse_args()
model = 'qwen2.5-72b' if args.host == 'saturo_1' else 'mistral-123b'
config = 'Qwen2.5-72B' if args.host == 'saturo_1' else 'Mistral-Large-Instruct-2411'
parallelisms = [1, 2, 4] if args.host == 'saturo_1' else [2, 4]
state_path = BASE / f'step64_queue_{args.host}.json'
state = dict(host=args.host, model=model, state='RUNNING', jobs=[])


def save():
    state['updated_at'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    temp = state_path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(state_path)


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


save()
try:
    run(['scp', str(ROOT/'benchmarks/profile_quick_runtime.py'), args.host+':/home/admin/hybrid_serve_proj/benchmarks/'])
    for tp in parallelisms:
        relative = Path('experiments/results/offline_runtime_profile') / model / f'l20a_tp{tp}_bf16_complete_20260914/step64_repeat3'
        local = ROOT / relative
        remote = '/home/admin/hybrid_serve_proj/' + str(relative)
        job = dict(tp=tp, state='STARTING', local_dir=str(local))
        state['jobs'].append(job)
        save()
        run(['ssh', args.host, 'mkdir -p ' + shlex.quote(remote)])
        run(['scp', str(local/'plan.json'), args.host+':'+remote+'/'])
        with (local/'receiver.log').open('a') as log:
            job['state']='RUNNING'
            save()
            run([sys.executable, '-u', str(ROOT/'benchmarks/receive_quick_runtime.py'),
                 '--host', args.host, '--model', '/home/admin/hybrid_serve_proj/data/hf_configs/'+config,
                 '--directory', str(relative), '--repeat', '3', '--tp', str(tp)],
                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        job['state']='FINALIZING'
        save()
        with (local/'summary.log').open('w') as log:
            run([sys.executable, str(ROOT/'benchmarks/summarize_quick_runtime.py'), str(local)],
                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        sources = ['benchmarks/profile_quick_runtime.py','benchmarks/profile_complete_runtime.py',
                   'vllm/worker/worker_base.py','vllm/worker/model_runner.py',
                   'vllm/model_executor/models/qwen2.py','vllm/model_executor/models/llama.py',
                   'vllm/executor/uniproc_executor.py','vllm/executor/mp_distributed_executor.py',
                   'vllm/executor/executor_base.py']
        run(['ssh', args.host, 'cd /home/admin/hybrid_serve_proj && tar -czf ' +
             shlex.quote(remote+'/source_snapshot.tar.gz')+' '+ ' '.join(map(shlex.quote,sources))])
        names=['raw.jsonl','skipped.json','plan.json','source_snapshot.tar.gz']
        hashes = subprocess.check_output(['ssh',args.host,'sha256sum '+
                   ' '.join(shlex.quote(remote+'/'+n) for n in names)],text=True)
        run(['scp','-Cr',args.host+':'+remote,str(local.parent)+'/'])
        checked={}
        for line in hashes.splitlines():
            digest, path = line.split(maxsplit=1)
            name=Path(path.strip()).name
            assert hashlib.sha256((local/name).read_bytes()).hexdigest()==digest,name
            checked[name]=digest
        manifest=json.loads((local/'manifest.json').read_text())
        assert manifest['tp']==tp and manifest['plan_sha256']==checked['plan.json']
        with tarfile.open(local/'source_snapshot.tar.gz') as archive:
            for path,digest in manifest['source_sha256'].items():
                assert hashlib.sha256(archive.extractfile(path).read()).hexdigest()==digest,path
        validation=json.loads((local/'validation.json').read_text())
        records=json.loads((local/'baseline_results.json').read_text())['measurements']
        (local/'high_jitter_points.json').write_text(json.dumps([r for r in records if r['baseline_cv_pct']>10],indent=2)+'\n')
        (local/'sync_verification.json').write_text(json.dumps(dict(remote_hashes=checked,
                    source_archive_verified=True,verified_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds')),indent=2)+'\n')
        job.update(state='COMPLETED',validation=validation)
        save()
        print(args.host,model,'TP'+str(tp),'COMPLETED',validation['covered'],'covered',flush=True)
    state['state']='COMPLETED'
    save()
except BaseException as exc:
    state.update(state='FAILED',error=repr(exc))
    if state['jobs'] and state['jobs'][-1]['state']!='COMPLETED':
        state['jobs'][-1]['state']='FAILED'
    save()
    raise
