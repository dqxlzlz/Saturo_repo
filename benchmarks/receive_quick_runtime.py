#!/usr/bin/env python3
"""Run the remote collector and acknowledge every record only after local fsync."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

PREFIX = 'PROFILE_SYNC_V1 '
parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--directory', required=True)
parser.add_argument('--repeat', type=int, default=3, choices=[3, 4, 5])
parser.add_argument('--tp', type=int, default=4, choices=[1, 2, 4])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = (root / args.directory).resolve()
out.mkdir(parents=True, exist_ok=True)
remote = '/home/admin/hybrid_serve_proj/' + args.directory
command = ('cd /home/admin/hybrid_serve_proj && '
           'export CUDA_VISIBLE_DEVICES=' + ','.join(str(i) for i in range(args.tp)) +
           ' OMP_NUM_THREADS=1 PYTHONPATH=/home/admin/hybrid_serve_proj && '
           'export LD_LIBRARY_PATH=/home/admin/hybrid_serve_proj/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-} && '
           'exec .venv/bin/python3 -u benchmarks/profile_quick_runtime.py --model ' + shlex.quote(args.model) +
           ' --plan ' + shlex.quote(remote + '/plan.json') + ' --output-dir ' + shlex.quote(remote) +
           ' --repeat ' + str(args.repeat) + ' --tp ' + str(args.tp))
known = {}
raw = out / 'raw.jsonl'
if raw.exists():
    for line in raw.read_text().splitlines():
        row = json.loads(line)
        known[row['config']['id']] = row
with (out / 'remote_stderr.log').open('a') as errors, (out / 'remote_stdout.log').open('a') as logs, raw.open('a') as data:
    proc = subprocess.Popen(['ssh', '-T', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', args.host, command],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                            text=True, bufsize=1)
    try:
        for line in proc.stdout:
            if not line.startswith(PREFIX):
                logs.write(line)
                logs.flush()
                continue
            msg = json.loads(line[len(PREFIX):])
            name, value = msg['name'], msg['value']
            if msg['kind'] == 'record':
                assert value['config']['id'] == name
                assert sorted(r['rank'] for r in value['ranks']) == list(range(args.tp))
                assert all(math.isfinite(r[k]) and r[k] > 0 for r in value['ranks'] for k in ['worker_cuda_ms', 'worker_wall_ms'])
                assert value['repeat'] == args.repeat
                assert all(len(r['samples']) == args.repeat for r in value['ranks'])
                assert all(math.isfinite(s[k]) and s[k] > 0 for r in value['ranks'] for s in r['samples'] for k in ['worker_cuda_ms', 'worker_wall_ms'])
                if name in known:
                    assert known[name] == value
                else:
                    data.write(json.dumps(value, separators=(',', ':')) + '\n')
                    data.flush()
                    os.fsync(data.fileno())
                    known[name] = value
                print(args.host, 'SYNCED', len(known), value['config']['family'], name, flush=True)
            else:
                assert msg['kind'] == 'file' and name in ['manifest.json', 'plan.json', 'status.json', 'skipped.json']
                temporary = out / (name + '.tmp')
                with temporary.open('w') as fh:
                    json.dump(value, fh, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(temporary, out / name)
                if name == 'status.json':
                    print(args.host, value['state'], value['completed'], '/', value['planned'], flush=True)
            proc.stdin.write('ACK ' + name + '\n')
            proc.stdin.flush()
        code = proc.wait()
        status_path = out / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if code or status.get('state') != 'COMPLETED':
            raise RuntimeError(f'Remote process exited {code}; last status {status}')
    except BaseException:
        proc.stdin.close()
        proc.terminate()
        raise
print(args.host, 'DONE', len(known), 'records safely stored locally', flush=True)
