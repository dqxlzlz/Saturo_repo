#!/usr/bin/env python3
"""Validate/export complete-runtime profiles; fit separate baseline models.

Does not replace legacy time_predictor caches: detailed component timings are
instrumented and must not be added to baseline CPU timings as a correction.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np


def features(c):
    pf=sum(q for q,cache in c['prefill']); dec=len(c['decode'])
    work=sum(q*cache+q*(q+1)/2 for q,cache in c['prefill'])
    ctx=sum(c['decode']); n=pf+dec
    sampled=dec+(len(c['prefill']) if c['prefill_final'] else 0)
    return [1,n,n*n,work,dec,ctx,c['n_requests'],sampled]


def errors(y,pred):
    delta=np.asarray(pred)-np.asarray(y)
    relative=np.abs(delta)/np.maximum(np.asarray(y),1e-9)*100
    return dict(count=len(y),mae_ms=float(np.abs(delta).mean()),
                mape_pct=float(relative.mean()),p95_absolute_pct=float(np.percentile(relative,95)),
                bias_ms=float(delta.mean()))


def baseline_models(records):
    result={}
    for family in ['linear','prefill','decode','mixed']:
        selected=[r for r in records if r['config']['family']==family]
        if len(selected)<10:continue
        rng=np.random.default_rng(42)
        indices=rng.permutation(len(selected));cut=max(2,int(len(indices)*.8))
        train=[selected[i] for i in indices[:cut]]
        test=[selected[i] for i in indices[cut:]]
        target=lambda rs:np.array([r['driver']['baseline']['worker_cuda_ms']['median'] for r in rs])
        if family=='linear':
            table=sorted((r['config']['n_tokens'],r['driver']['baseline']['worker_cuda_ms']['median']) for r in train)
            prediction=np.interp([r['config']['n_tokens'] for r in test],[t[0] for t in table],[t[1] for t in table])
            full=sorted((r['config']['n_tokens'],r['driver']['baseline']['worker_cuda_ms']['median']) for r in selected)
            result[family]=dict(type='linear_interpolation',target='baseline_worker_cuda_ms',
                                token_values=[t[0] for t in full],ms_values=[t[1] for t in full],
                                validation=errors(target(test),prediction))
        else:
            def fit(rs):
                x=np.array([features(r['config']) for r in rs],dtype=float)
                scale=np.maximum(np.abs(x).max(axis=0),1)
                coef=np.linalg.lstsq(x/scale,target(rs),rcond=None)[0]/scale
                return coef
            coef=fit(train)
            prediction=np.maximum(0,np.array([features(r['config']) for r in test])@coef)
            result[family]=dict(type='least_squares',target='baseline_worker_cuda_ms',
                features=['intercept','total_tokens','total_tokens_squared','prefill_causal_context_work',
                          'decode_tokens','decode_context_sum','request_count','sampled_rows'],
                coefficients=fit(selected).tolist(),validation=errors(target(test),prediction),
                warning='Empirical fit within measured domain; inspect validation error before use.')
    return result


def cross_family_repeats(records):
    groups={}
    for r in records:
        c=r['config']
        shape={k:c[k] for k in ['prefill','decode','prefill_final']}
        key=json.dumps(shape,sort_keys=True)
        groups.setdefault(key,[]).append(dict(id=c['id'],family=c['family'],
            baseline_worker_cuda_ms=r['driver']['baseline']['worker_cuda_ms']['median']))
    comparisons=[]
    for shape,points in groups.items():
        if len({p['family'] for p in points})<2:continue
        times=[p['baseline_worker_cuda_ms'] for p in points]
        comparisons.append(dict(shape=json.loads(shape),points=points,
            relative_span_pct=(max(times)-min(times))/min(times)*100))
    return comparisons


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();root=args.directory
    status=json.loads((root/'status.json').read_text())
    if not args.allow_partial:assert status['state']=='COMPLETED',status
    manifest=json.loads((root/'manifest.json').read_text())
    grid=json.loads((root/'grid.json').read_text())
    records=[json.loads(line) for line in (root/'raw.jsonl').read_text().splitlines()]
    ids=[r['config']['id'] for r in records]
    assert len(ids)==len(set(ids))
    planned={r['id'] for r in grid['all_configs']}
    skipped={r['config']['id'] for r in grid['skipped']}
    if not args.allow_partial:assert set(ids)==planned-skipped,(len(ids),len(planned-skipped))
    rows=[]
    for record in records:
        c=record['config'];b=record['driver']['baseline'];d=record['driver']['detailed']
        assert record['driver']['rank']==0
        ranks=[record['driver']]+record['ranks']
        expected_ranks=manifest['configuration']['tensor_parallel_size']
        assert sorted(r['rank'] for r in ranks)==list(range(expected_ranks)),c['id']
        for rank in ranks:
            assert rank['id']==c['id'],c['id']
            for variant,repeat_key in [('baseline','repeat'),('detailed','detailed_repeat')]:
                for name,summary in rank[variant].items():
                    assert len(summary['each'])==manifest['configuration'][repeat_key],(c['id'],name)
                    assert all(np.isfinite(summary['each'])) and min(summary['each'])>=0,(c['id'],name)
        assert d['actual_linear_ms']['median']>0
        assert all(k in d for k in ['model_forward_cuda_ms','logits_cuda_ms','sampling_cuda_ms','prepare_input_wall_ms'])
        row=dict(id=c['id'],family=c['family'],total_tokens=c['n_tokens'],request_count=c['n_requests'],
                 prefill_tokens=sum(q for q,cache in c['prefill']),decode_tokens=len(c['decode']),
                 prefill_query_per_request=c['prefill'][0][0] if c['prefill'] else 0,
                 prefill_cached_context=c['prefill'][0][1] if c['prefill'] else 0,
                 decode_context=c['decode'][0] if c['decode'] else 0,
                 baseline_worker_cuda_ms=b['worker_cuda_ms']['median'],
                 baseline_worker_wall_ms=b['worker_wall_ms']['median'],
                 baseline_slowest_rank_cuda_ms=max(r['baseline']['worker_cuda_ms']['median'] for r in ranks),
                 detailed_worker_cuda_ms=d['worker_cuda_ms']['median'])
        row.update({k:v['median'] for k,v in d.items() if k not in ['worker_cuda_ms','worker_wall_ms']})
        row['detailed_worker_wall_ms']=d['worker_wall_ms']['median']
        row['instrumentation_delta_ms']=row['detailed_worker_cuda_ms']-row['baseline_worker_cuda_ms']
        row['baseline_p95_ms']=float(np.percentile(b['worker_cuda_ms']['each'],95))
        row['baseline_cv_pct']=float(np.std(b['worker_cuda_ms']['each'])/np.mean(b['worker_cuda_ms']['each'])*100)
        rows.append(row)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'complete_profile.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(rows)
    families={f:sum(r['family']==f for r in rows) for f in ['linear','prefill','decode','mixed']}
    repeated=cross_family_repeats(records)
    spans=[r['relative_span_pct'] for r in repeated]
    validation=dict(state=status['state'],rows=len(rows),planned=len(planned),skipped=len(skipped),
                    verified_tp_ranks=manifest['configuration']['tensor_parallel_size'],
                    families=families,median_baseline_cv_pct=statistics.median(r['baseline_cv_pct'] for r in rows),
                    high_jitter_points=sum(r['baseline_cv_pct']>10 for r in rows),
                    cross_family_repeated_shapes=len(repeated),
                    cross_family_span_median_pct=float(np.median(spans)) if spans else None,
                    cross_family_span_p95_pct=float(np.percentile(spans,95)) if spans else None,
                    cross_family_span_over_10pct=sum(x>10 for x in spans),
                    grid_complete=set(ids)==planned-skipped)
    (root/'validation.json').write_text(json.dumps(validation,indent=2))
    (root/'complete_runtime_profile.json').write_text(json.dumps(dict(
        schema='complete_worker_runtime_v1',manifest=manifest,validation=validation,
        fitted_baseline_models=baseline_models(records),measurements=rows,
        cross_family_repeats=repeated,
        notes=['Baseline and detailed measurements are separate executions.',
               'CUDA events measure elapsed stream intervals, not exclusive GPU kernel active time.',
               'prepare_input wall time includes CPU preparation, broadcasts and possible waits.',
               'Do not add CPU wall durations to overlapping CUDA ranges.',
               'Full worker timing excludes scheduler, admission and output dispatch.',
               'Module metrics sum all actual model layers; no three-layer extrapolation.']),indent=2))

    if args.no_plots:
        print(json.dumps(validation,indent=2))
        return
    try:
        import matplotlib
    except ModuleNotFoundError:
        print(json.dumps(dict(validation,plots='not generated: matplotlib unavailable'),indent=2))
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    linear=sorted([r for r in rows if r['family']=='linear'],key=lambda r:r['total_tokens'])
    if linear:
        x=[r['total_tokens'] for r in linear]
        for key,label in [('baseline_worker_cuda_ms','Complete worker, baseline'),
                          ('detailed_worker_cuda_ms','Complete worker, instrumented'),
                          ('model_forward_cuda_ms','Forward, instrumented')]:
            axes[0,0].plot(x,[r[key] for r in linear],label=label,lw=1)
        axes[0,0].legend(frameon=False,fontsize=8)
        axes[0,0].set(xlabel='Tokens (one prefill request)',ylabel='Milliseconds',title='Dense token sweep: full execution')
        for key,label in [('actual_linear_ms','Non-attention'),('actual_combined_attn_ms','Attention'),
                          ('logits_cuda_ms','Logits'),('sampling_cuda_ms','Sampling')]:
            axes[0,1].plot(x,[r[key] for r in linear],label=label,lw=1)
        axes[0,1].legend(frameon=False,fontsize=8)
        axes[0,1].set(xlabel='Tokens',ylabel='Instrumented CUDA interval (ms)',title='Full-model component timing')
    decode=[r for r in rows if r['family']=='decode']
    if decode:
        contexts=sorted({r['decode_context'] for r in decode})
        chosen=sorted(set(contexts[i] for i in np.linspace(0,len(contexts)-1,min(5,len(contexts)),dtype=int)))
        for ctx in chosen:
            selected=sorted([r for r in decode if r['decode_context']==ctx],key=lambda r:r['decode_tokens'])
            axes[1,0].plot([r['decode_tokens'] for r in selected],[r['baseline_worker_cuda_ms'] for r in selected],label=f'Context {ctx}')
        axes[1,0].legend(frameon=False,fontsize=8)
        axes[1,0].set(xlabel='Decode batch size',ylabel='Baseline complete worker (ms)',title='Decode: batch and context sensitivity')
    mixed=[r for r in rows if r['family']=='mixed' and r['prefill_cached_context']==1024 and r['decode_context']==2048]
    for batch in sorted({r['decode_tokens'] for r in mixed}):
        selected=sorted([r for r in mixed if r['decode_tokens']==batch],key=lambda r:r['total_tokens'])
        axes[1,1].plot([r['total_tokens'] for r in selected],[r['baseline_worker_cuda_ms'] for r in selected],marker='.',label=f'{batch} decode requests')
    if mixed:axes[1,1].legend(frameon=False,fontsize=8)
    axes[1,1].set(xlabel='Total token budget',ylabel='Baseline complete worker (ms)',title='Mixed: prefill cache 1024, decode context 2048')
    name=Path(manifest['configuration']['model']).name
    layers=manifest['model_config']['num_hidden_layers']
    fig.suptitle(f'{name} | TP4 L20A BF16 | {layers} actual layers\n{len(rows):,} measured configurations | baseline and instrumented execution',fontweight='bold')
    fig.savefig(root/'complete_profile.png',dpi=160)
    fig.savefig(root/'complete_profile.pdf')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
