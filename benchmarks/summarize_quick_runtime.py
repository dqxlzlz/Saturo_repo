#!/usr/bin/env python3
"""Validate sparse completion and export baseline results with source labels."""
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
out = args.directory
plan = json.loads((out / 'plan.json').read_text())
manifest = json.loads((out / 'manifest.json').read_text())
tp = manifest['tp']
assert plan.get('tp', 4) == tp
status = json.loads((out / 'status.json').read_text())
assert status['state'] == 'COMPLETED', status
records = [json.loads(line) for line in (out / 'raw.jsonl').read_text().splitlines()]
ids = [r['config']['id'] for r in records]
assert len(ids) == len(set(ids))
actual_skips = json.loads((out / 'skipped.json').read_text()) if (out / 'skipped.json').exists() else status.get('skipped', [])
assert set(ids) == {c['id'] for c in plan['new_configs']} - {s['config']['id'] for s in actual_skips}
rows = []


def add(config, cuda, wall, source, repeats, source_id, samples):
    assert math.isfinite(cuda) and cuda > 0 and math.isfinite(wall) and wall > 0
    assert len(samples) == repeats and all(math.isfinite(s) and s > 0 for s in samples)
    rows.append(dict(id=config['id'], family=config['family'], total_tokens=config['n_tokens'],
                     request_count=config['n_requests'],
                     prefill_query_per_request=config['prefill'][0][0] if config['prefill'] else 0,
                     prefill_cached_context=config['prefill'][0][1] if config['prefill'] else 0,
                     decode_requests=len(config['decode']),
                     decode_context=config['decode'][0] if config['decode'] else 0,
                     baseline_worker_cuda_ms=cuda, baseline_worker_wall_ms=wall,
                     baseline_cv_pct=statistics.pstdev(samples) / statistics.mean(samples) * 100,
                     baseline_min_ms=min(samples), baseline_max_ms=max(samples),
                     timed_repetitions=repeats, source=source, source_config_id=source_id))


for entry in plan['reused']:
    baseline = entry['baseline']
    add(entry['config'], baseline['worker_cuda_ms']['median'], baseline['worker_wall_ms']['median'],
        entry['source'], entry['repeat'], entry['source_id'], baseline['worker_cuda_ms']['each'])
for entry in records:
    assert entry['repeat'] == plan['repeat'] and entry['warmup'] == 1
    assert sorted(r['rank'] for r in entry['ranks']) == list(range(tp))
    assert all(math.isfinite(r[k]) and r[k] > 0 for r in entry['ranks'] for k in ['worker_cuda_ms', 'worker_wall_ms'])
    driver = next(r for r in entry['ranks'] if r['rank'] == 0)
    for rank in entry['ranks']:
        assert len(rank['samples']) == entry['repeat']
        for field in ['worker_cuda_ms', 'worker_wall_ms']:
            values = [s[field] for s in rank['samples']]
            assert all(math.isfinite(v) and v > 0 for v in values)
            assert rank[field] == statistics.median(values)
    add(entry['config'], driver['worker_cuda_ms'], driver['worker_wall_ms'],
        str(out / 'raw.jsonl'), entry['repeat'], entry['config']['id'], [s['worker_cuda_ms'] for s in driver['samples']])
rows.sort(key=lambda r: (r['family'], r['total_tokens'], r['request_count'], r['prefill_cached_context'], r['decode_context']))
skips = plan['skipped'] + actual_skips
assert len(rows) + len(skips) == plan['planned']
assert len({r['id'] for r in rows}) == len(rows)
with (out / 'baseline_results.csv').open('w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
validation = dict(state='COMPLETED', sparse_grid_complete=True, planned=plan['planned'],
                  covered=len(rows), new_measurements=len(records), reused=len(plan['reused']),
                  skipped=len(skips), families=dict(collections.Counter(r['family'] for r in rows)),
                  verified_tp_ranks=tp, raw_sha256=hashlib.sha256((out / 'raw.jsonl').read_bytes()).hexdigest(),
                  new_timed_repetitions=plan['repeat'],
                  median_cv_pct=statistics.median(r['baseline_cv_pct'] for r in rows),
                  points_cv_above_10pct=sum(r['baseline_cv_pct'] > 10 for r in rows),
                  notes=['New points have one warmup and multiple measured executions; medians and all samples are retained.',
                         'Reused points use the median of at least three existing baseline measurements; counts are recorded per row.',
                         'Completion refers to the new sparse plan, not the abandoned dense grid.',
                         'CUDA intervals include possible host submission gaps. No online accuracy claim.'])
(out / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
(out / 'baseline_results.json').write_text(json.dumps(dict(validation=validation, measurements=rows, skipped=skips), indent=2) + '\n')
print(json.dumps(validation, indent=2))
