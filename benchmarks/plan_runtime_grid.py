#!/usr/bin/env python3
"""Plan a finer token grid, reusing >=3-repeat results at the same model/TP."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('original_profiler', ROOT / 'benchmarks/profile_complete_runtime.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['qwen2.5-72b', 'mistral-123b'])
parser.add_argument('--tp', required=True, type=int, choices=[1, 2, 4])
parser.add_argument('--stride', type=int, default=64, choices=[64])
args = parser.parse_args()
parent = ROOT / 'experiments/results/offline_runtime_profile' / args.model / f'l20a_tp{args.tp}_bf16_complete_20260914'
prior_plan = json.loads((parent / 'step256_repeat3/plan.json').read_text())
prior_status = json.loads((parent / 'step256_repeat3/status.json').read_text())
assert prior_status['state'] == 'COMPLETED'
capacity = prior_status['capacity_blocks']
grid = []


def add(family, prefill, decode, final=True):
    c = dict(family=family, prefill=prefill, decode=decode, prefill_final=final)
    c['id'] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]
    c.update(n_tokens=sum(q for q, cache in prefill) + len(decode), n_requests=len(prefill) + len(decode))
    assert all(q > 0 for q, cache in prefill)
    grid.append(c)


def shape(c):
    return json.dumps([c['prefill'], c['decode'], c['prefill_final']], sort_keys=True)


for n in range(args.stride, 8193, args.stride):
    add('linear', [(n, 0)], [])
for n in range(args.stride, 4097, args.stride):
    for b in [2, 4, 8, 16, 32, 64, 128]:
        if n * b <= 12000:
            add('prefill', [(n, 0)] * b, [])
for context in range(args.stride, 8193, args.stride):
    for b in [1, 8, 32, 64, 128, 256, 512]:
        add('decode', [], [context] * b)
for budget in range(args.stride, 4097, args.stride):
    for b in [8, 32, 128]:
        if b >= budget:
            continue
        for cached in [0, 1024]:
            for context in [2048, 8192]:
                add('mixed', [(budget - b, cached)], [context] * b, False)
assert len({shape(c) for c in grid}) == len(grid)
available = {}
source_hashes = {}
for phase in ['dense', 'pilot', 'quick_repeat3', 'step256_repeat3']:
    raw = parent / phase / 'raw.jsonl'
    if not raw.exists():
        continue
    source_hashes[str(raw)] = hashlib.sha256(raw.read_bytes()).hexdigest()
    for line in raw.read_text().splitlines():
        r = json.loads(line)
        if 'driver' in r:
            baseline = r['driver']['baseline']
            repeats = len(baseline['worker_cuda_ms']['each'])
        else:
            repeats = r['repeat']
            if repeats < 3:
                continue
            driver = next(x for x in r['ranks'] if x['rank'] == 0)
            assert len(driver['samples']) == repeats
            baseline = {k:old.stats([s[k] for s in driver['samples']]) for k in ['worker_cuda_ms', 'worker_wall_ms']}
        if repeats >= 3:
            available.setdefault(shape(r['config']), dict(source=str(raw), source_id=r['config']['id'],
                                                          baseline=baseline, repeat=repeats))
new, reused, skipped = [], [], []
for c in grid:
    if shape(c) in available:
        reused.append(dict(config=c, **available[shape(c)]))
    elif old.blocks_required(c, 16) > capacity:
        skipped.append(dict(config=c, reason='Previously measured KV block budget',
                            required_blocks=old.blocks_required(c, 16), available_blocks=capacity))
    else:
        new.append(c)
out = parent / f'step{args.stride}_repeat3'
out.mkdir(exist_ok=True)
assert not (out / 'manifest.json').exists(), 'Existing run: resume using its original immutable plan'
plan = dict(schema='step64_completion_v1', model=args.model, tp=args.tp, planned=len(grid),
            new_configs=new, reused=reused, skipped=skipped, expected_core_sha256=prior_plan['expected_core_sha256'],
            warmup=1, repeat=3, module_timing=False, prior_raw_sha256=source_hashes,
            sampling=dict(single_prefill_tokens=[64,8192,64], batched_prefill_tokens=[64,4096,64],
                          decode_context=[64,8192,64], mixed_token_budget=[64,4096,64],
                          mixed_prefill_context=[0,1024], mixed_decode_context=[2048,8192]))
(out / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
print(args.model, 'TP', args.tp, 'planned',len(grid), 'new',len(new),'reused',len(reused),'skipped',len(skipped),
      'new families',dict(collections.Counter(c['family'] for c in new)))
