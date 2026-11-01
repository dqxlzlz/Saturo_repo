#!/usr/bin/env python3
"""Build the requested 256-token grid, reusing only >=3-repeat measurements."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('original_profiler', ROOT / 'benchmarks/profile_complete_runtime.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
base = ROOT / 'experiments/results/offline_runtime_profile'


def make(family, prefill, decode, final=True):
    row = dict(family=family, prefill=prefill, decode=decode, prefill_final=final)
    row['id'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]
    row.update(n_tokens=sum(q for q, c in prefill) + len(decode), n_requests=len(prefill) + len(decode))
    return row


def shape(c):
    return json.dumps([c['prefill'], c['decode'], c['prefill_final']], sort_keys=True)


grid = []
for n in range(256, 8193, 256):
    grid.append(make('linear', [(n, 0)], []))
for q in range(256, 4097, 256):
    for b in [2, 4, 8, 16, 32, 64, 128]:
        if q * b <= 12000:
            grid.append(make('prefill', [(q, 0)] * b, []))
for context in range(256, 8193, 256):
    for b in [1, 8, 32, 64, 128, 256, 512]:
        grid.append(make('decode', [], [context] * b))
for budget in range(256, 4097, 256):
    for b in [8, 32, 128]:
        for cached in [0, 1024]:
            for context in [2048, 8192]:
                grid.append(make('mixed', [(budget - b, cached)], [context] * b, False))
assert len({shape(c) for c in grid}) == len(grid)

for model in ['qwen2.5-72b', 'mistral-123b']:
    parent = base / model / 'l20a_tp4_bf16_complete_20260914'
    available = {}
    for phase in ['dense', 'pilot', 'quick_repeat3', 'step256_repeat3']:
        raw = parent / phase / 'raw.jsonl'
        if not raw.exists():
            continue
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            if 'driver' in r:
                baseline = r['driver']['baseline']
                repeats = len(baseline['worker_cuda_ms']['each'])
            else:
                repeats = r['repeat']
                if repeats < 3:
                    continue
                driver = next(rank for rank in r['ranks'] if rank['rank'] == 0)
                assert len(driver['samples']) == repeats
                baseline = {k: old.stats([s[k] for s in driver['samples']])
                            for k in ['worker_cuda_ms', 'worker_wall_ms']}
            if repeats >= 3:
                available.setdefault(shape(r['config']), dict(source=str(raw), source_id=r['config']['id'],
                                                              baseline=baseline, repeat=repeats))
    original = json.loads((parent / 'dense/grid.json').read_text())
    capacity = int(original['num_gpu_blocks'] * .9)
    reused, new, skipped = [], [], []
    for c in grid:
        if shape(c) in available:
            reused.append(dict(config=c, **available[shape(c)]))
        elif old.blocks_required(c, original['block_size']) > capacity:
            skipped.append(dict(config=c, reason='KV block budget',
                                required_blocks=old.blocks_required(c, original['block_size']), available_blocks=capacity))
        else:
            new.append(c)
    manifest = json.loads((parent / 'dense/manifest.json').read_text())
    plan = dict(schema='step256_completion_v1', model=model, planned=len(grid),
                new_configs=new, reused=reused, skipped=skipped,
                expected_core_sha256=manifest['source_sha256'], warmup=1, repeat=3, module_timing=False,
                sampling=dict(single_prefill_tokens=[256, 8192, 256], batched_prefill_tokens=[256, 4096, 256],
                              decode_context=[256, 8192, 256], mixed_token_budget=[256, 4096, 256],
                              mixed_prefill_context=[0, 1024], mixed_decode_context=[2048, 8192]))
    out = parent / 'step256_repeat3'
    out.mkdir(exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Existing run: use its original immutable plan to resume')
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(model, 'planned', len(grid), 'reuse', len(reused), 'new', len(new), 'skipped', len(skipped),
          'new families', dict(collections.Counter(c['family'] for c in new)))
