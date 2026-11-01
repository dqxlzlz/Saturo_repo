#!/usr/bin/env python3
"""Compare identical validated Qwen batches at TP1, TP2, and TP4."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'experiments/results/offline_runtime_profile/qwen2.5-72b'
data = {}
for tp in [1, 2, 4]:
    directory = root / f'l20a_tp{tp}_bf16_complete_20260914/step256_repeat3'
    result = json.loads((directory / 'baseline_results.json').read_text())
    assert result['validation']['state'] == 'COMPLETED'
    assert result['validation']['verified_tp_ranks'] == tp
    data[tp] = {r['id']:r for r in result['measurements']}
fields = ['family', 'total_tokens', 'request_count', 'prefill_query_per_request',
          'prefill_cached_context', 'decode_requests', 'decode_context']
rows = []
for cid, one in data[1].items():
    row = dict(id=cid, **{k:one[k] for k in fields})
    for tp in [1, 2, 4]:
        other = data[tp][cid]
        assert all(other[k] == one[k] for k in fields)
        row[f'tp{tp}_cuda_ms'] = other['baseline_worker_cuda_ms']
        row[f'tp{tp}_cv_pct'] = other['baseline_cv_pct']
        row[f'tp{tp}_repetitions'] = other['timed_repetitions']
    row['tp1_over_tp2'] = row['tp1_cuda_ms'] / row['tp2_cuda_ms']
    row['tp1_over_tp4'] = row['tp1_cuda_ms'] / row['tp4_cuda_ms']
    row['high_jitter'] = any(row[f'tp{tp}_cv_pct'] > 10 for tp in [1, 2, 4])
    rows.append(row)
out = root / 'l20a_tp1_bf16_complete_20260914/step256_repeat3'
with (out / 'tp1_vs_tp2_vs_tp4.csv').open('w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(out / 'tp1_vs_tp2_vs_tp4.json').write_text(json.dumps(dict(matched=len(rows), measurements=rows), indent=2)+'\n')
print('Matched configurations:', len(rows))
for r in rows:
    if (r['family'] == 'linear' and r['total_tokens'] in [1024, 4096, 8192]) or (
        r['family'] == 'decode' and r['request_count'] == 32 and r['decode_context'] == 2048) or (
        r['family'] == 'mixed' and r['total_tokens'] == 1024 and r['decode_requests'] == 32
        and r['prefill_cached_context'] == 1024 and r['decode_context'] == 2048):
        print(json.dumps(r))
