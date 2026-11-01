#!/usr/bin/env python3
"""Export all step64 configurations across TP settings, preserving missing capacity cases."""
import csv
import argparse
import json
from pathlib import Path

base=Path(__file__).resolve().parents[1]/'experiments/results/offline_runtime_profile'
parser=argparse.ArgumentParser()
parser.add_argument('--model',choices=['qwen2.5-72b','mistral-123b'])
args=parser.parse_args()
fields=['family','total_tokens','request_count','prefill_query_per_request',
        'prefill_cached_context','decode_requests','decode_context']
for model,tps in [('qwen2.5-72b',[1,2,4]),('mistral-123b',[2,4])]:
    if args.model and model!=args.model:
        continue
    data={}
    for tp in tps:
        p=base/model/f'l20a_tp{tp}_bf16_complete_20260914/step64_repeat3/baseline_results.json'
        r=json.loads(p.read_text())
        assert r['validation']['state']=='COMPLETED' and r['validation']['verified_tp_ranks']==tp
        data[tp]={x['id']:x for x in r['measurements']}
    reference=data[4]
    assert all(set(rows)<=set(reference) for rows in data.values())
    combined=[]
    for cid,config in reference.items():
        row=dict(id=cid,**{f:config[f] for f in fields})
        for tp in tps:
            point=data[tp].get(cid)
            if point:
                assert all(point[f]==config[f] for f in fields)
            row[f'tp{tp}_cuda_ms']=point['baseline_worker_cuda_ms'] if point else None
            row[f'tp{tp}_cv_pct']=point['baseline_cv_pct'] if point else None
            row[f'tp{tp}_repetitions']=point['timed_repetitions'] if point else None
        row['high_jitter']=any(row[f'tp{tp}_cv_pct'] is not None and row[f'tp{tp}_cv_pct']>10 for tp in tps)
        combined.append(row)
    out=base/model
    with (out/'tp_comparison_step64.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
    (out/'tp_comparison_step64.json').write_text(json.dumps(dict(model=model,parallelisms=tps,
        counts={tp:len(rows) for tp,rows in data.items()},notes=['Missing values indicate configurations outside that TP capacity.',
        'Same shape across TP; elapsed worker CUDA intervals, not exclusive kernel busy time.'],measurements=combined),indent=2)+'\n')
    print(model,{tp:len(rows) for tp,rows in data.items()},'comparison exported')
