#!/usr/bin/env python3
"""Read local per-host queues and acknowledged collection progress."""
import json
from pathlib import Path

base=Path(__file__).resolve().parents[1]/'experiments/results/offline_runtime_profile'
total=0
for host in ['saturo_1','saturo_2']:
    q=json.loads((base/f'step64_queue_{host}.json').read_text())
    parts=[]
    for job in q['jobs']:
        p=Path(job['local_dir'])
        if not (p/'status.json').exists():
            parts.append(f"TP{job['tp']} {job['state']}")
            continue
        s=json.loads((p/'status.json').read_text())
        plan=json.loads((p/'plan.json').read_text())
        current=next((c['family'] for c in plan['new_configs'] if c['id']==s.get('last_id')), 'initializing')
        total+=s['completed']
        parts.append(f"TP{job['tp']} {s['completed']}/{s.get('valid',s['planned'])} {s['state']} {current}")
    print(host,q['state'],'; '.join(parts))
    if q.get('error'):
        print('ERROR',q['error'])
print('Total newly received:',total,'/ 5218')
