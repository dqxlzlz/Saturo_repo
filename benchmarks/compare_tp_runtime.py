#!/usr/bin/env python3
"""Join validated TP2 and TP4 results by exact batch ID, retaining jitter data."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('model_directory', type=Path)
args = parser.parse_args()
root = args.model_directory
tp2dir = root / 'l20a_tp2_bf16_complete_20260914/step256_repeat3'
tp4dir = root / 'l20a_tp4_bf16_complete_20260914/step256_repeat3'
tp2 = json.loads((tp2dir / 'baseline_results.json').read_text())
tp4 = json.loads((tp4dir / 'baseline_results.json').read_text())
assert tp2['validation']['verified_tp_ranks'] == 2
assert tp4['validation']['verified_tp_ranks'] == 4
assert tp2['validation']['state'] == tp4['validation']['state'] == 'COMPLETED'
by_id = {r['id']: r for r in tp4['measurements']}
shape_fields = ['family', 'total_tokens', 'request_count', 'prefill_query_per_request',
                'prefill_cached_context', 'decode_requests', 'decode_context']
rows = []
for a in tp2['measurements']:
    assert a['id'] in by_id, a['id']
    b = by_id[a['id']]
    assert all(a[k] == b[k] for k in shape_fields)
    row = dict(id=a['id'], **{k:a[k] for k in shape_fields})
    row.update(tp2_cuda_ms=a['baseline_worker_cuda_ms'], tp4_cuda_ms=b['baseline_worker_cuda_ms'],
               tp2_over_tp4=a['baseline_worker_cuda_ms'] / b['baseline_worker_cuda_ms'],
               tp2_cv_pct=a['baseline_cv_pct'], tp4_cv_pct=b['baseline_cv_pct'],
               tp2_repetitions=a['timed_repetitions'], tp4_repetitions=b['timed_repetitions'],
               high_jitter=a['baseline_cv_pct'] > 10 or b['baseline_cv_pct'] > 10)
    rows.append(row)
with (tp2dir / 'tp2_vs_tp4.csv').open('w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(tp2dir / 'tp2_vs_tp4.json').write_text(json.dumps(dict(matched=len(rows), measurements=rows), indent=2)+'\n')
print('Matched configurations:', len(rows))
for r in rows:
    if (r['family'] == 'linear' and r['total_tokens'] in [1024,4096,8192]) or (
        r['family'] == 'decode' and r['request_count']==32 and r['decode_context']==2048) or (
        r['family']=='mixed' and r['total_tokens']==1024 and r['decode_requests']==32
        and r['prefill_cached_context']==1024 and r['decode_context']==2048):
        print(json.dumps(r))
