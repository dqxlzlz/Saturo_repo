#!/usr/bin/env python3
"""Join the two completed TP4 profiles and write a concise experiment report."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    args = parser.parse_args()
    names = ['qwen2.5-72b', 'mistral-123b']
    profiles = {}
    for name in names:
        path = args.base / name / 'l20a_tp4_bf16_complete_20260914/dense/complete_runtime_profile.json'
        profiles[name] = json.loads(path.read_text())
        validation = profiles[name]['validation']
        assert validation['state'] == 'COMPLETED' and validation['grid_complete'], (name, validation)
    indexed = {name: {r['id']: r for r in p['measurements']} for name, p in profiles.items()}
    qwen, mistral = (indexed[name] for name in names)
    shape_keys = ['id', 'family', 'total_tokens', 'request_count', 'prefill_tokens',
                  'decode_tokens', 'prefill_query_per_request', 'prefill_cached_context', 'decode_context']
    metrics = ['baseline_worker_cuda_ms', 'baseline_worker_wall_ms', 'detailed_worker_cuda_ms',
               'model_forward_cuda_ms', 'actual_linear_ms', 'actual_combined_attn_ms',
               'prepare_input_wall_ms', 'logits_cuda_ms', 'sampling_cuda_ms',
               'instrumentation_delta_ms', 'baseline_cv_pct']
    pairs = []
    for ident, q in qwen.items():
        if ident not in mistral:
            continue
        m = mistral[ident]
        assert all(q[k] == m[k] for k in shape_keys), ident
        row = {k: q[k] for k in shape_keys}
        for prefix, measured in [('qwen', q), ('mistral', m)]:
            row.update({prefix + '_' + k: measured[k] for k in metrics})
        row['mistral_over_qwen_baseline_ratio'] = m['baseline_worker_cuda_ms'] / q['baseline_worker_cuda_ms']
        pairs.append(row)
    assert pairs
    with (args.base / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)

    lines = ['# TP4 完整 offline worker profile', '',
             'Qwen2.5-72B 在 saturo_1、Mistral-123B 在 saturo_2，各使用 4 张 L20A。'
             '两组均为 BF16、dummy weights、eager、PP1，关闭 CUDA Graph、Inductor 和 custom all-reduce，'
             'custom_ops=["none"]。Qwen 测量全部 80 层，Mistral 测量全部 88 层。', '',
             '每点基线测量预热 2 次、重复 5 次；分项插桩测量预热 2 次、重复 3 次。'
             '下表时间均为对应重复样本的中位数，单位 ms。', '',
             '| 模型 | 有效测量点 | KV 预算外配置 | 重复计时 CV 中位数 | CV > 10% 点数 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, p in profiles.items():
        v = p['validation']
        lines.append(f"| {name} | {v['rows']} | {v['skipped']} | {v['median_baseline_cv_pct']:.3f}% | {v['high_jitter_points']} |")
    lines += ['', f'两模型共有 {len(pairs)} 个完全相同的配置，逐点对照保存在 comparison.csv。'
              '预算外配置及原因保留在各模型的 grid.json 中；它们不计为已测量点。', '',
              '## 同配置完整 worker 基线', '',
              '| 配置 | Qwen2.5-72B | Mistral-123B |', '| --- | ---: | ---: |']
    selections = [
        ('单请求 prefill 1024 tokens', dict(family='linear', total_tokens=1024)),
        ('单请求 prefill 4096 tokens', dict(family='linear', total_tokens=4096)),
        ('单请求 prefill 8192 tokens', dict(family='linear', total_tokens=8192)),
        ('Prefill：8 请求 × 512 tokens', dict(family='prefill', request_count=8, prefill_query_per_request=512)),
        ('Prefill：32 请求 × 128 tokens', dict(family='prefill', request_count=32, prefill_query_per_request=128)),
        ('Decode：B=32，context=2048', dict(family='decode', decode_tokens=32, decode_context=2048)),
        ('Decode：B=128，context=4096', dict(family='decode', decode_tokens=128, decode_context=4096)),
        ('Mixed：总 tokens=1024，decode B=32，prefill cache=1024，decode context=2048',
         dict(family='mixed', total_tokens=1024, decode_tokens=32, prefill_cached_context=1024, decode_context=2048))]
    for label, criteria in selections:
        selected = [r for r in pairs if all(r[k] == value for k, value in criteria.items())]
        assert len(selected) == 1, (label, len(selected))
        r = selected[0]
        lines.append(f"| {label} | {r['qwen_baseline_worker_cuda_ms']:.2f} | {r['mistral_baseline_worker_cuda_ms']:.2f} |")
    lines += ['', '## 单请求 1024-token 的分项插桩测量', '',
              '| 模型 | 插桩总时间 | Forward | 非 attention | Attention | CPU 输入准备 | Logits | Sampling |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    keys = ['detailed_worker_cuda_ms', 'model_forward_cuda_ms', 'actual_linear_ms',
            'actual_combined_attn_ms', 'prepare_input_wall_ms', 'logits_cuda_ms', 'sampling_cuda_ms']
    for name, rows in indexed.items():
        r = next(r for r in rows.values() if r['family'] == 'linear' and r['total_tokens'] == 1024)
        lines.append('| ' + name + ' | ' + ' | '.join(f'{r[k]:.2f}' for k in keys) + ' |')
    lines += ['', '## 计时口径', '',
              '完整 worker 路径包含输入准备、广播、forward、logits、sampling 和 worker 返回。'
              '它不包含在线 scheduler、请求到达/接纳、输出分发和 detokenization。', '',
              'CUDA events 测量流上的 elapsed interval，可能包含 CPU 提交间隙和等待，不能当成独占 GPU kernel busy time。'
              'CPU 输入准备使用 wall time，其余上表分项使用 CUDA events。CPU 与 GPU 区间有重叠，不能相加。'
              '分项插桩与基线是两套独立执行，分项时间也不能直接补加到基线上。', '',
              '输入使用合成 token 和 KV block table；这些结果验证执行耗时，不验证生成质量。'
              '不同模型运行在不同主机上，comparison.csv 中的比值是本次环境下的实测比较。', '',
              '同样的总 token 数可以对应不同的请求数与上下文分布，attention 工作量和采样行数也会不同。'
              '与在线执行对齐时，需要同时匹配这些维度，不能只匹配 chunk-size。', '',
              '## 拟合与复测一致性', '',
              '各模型 JSON 同时保存基线经验拟合和固定随机种子的 80/20 留出验证。'
              '这是测量域内的验证，不代表在线流量验证或测量域外精度。', '',
              '| 模型 | 配置类型 | 留出 MAPE | 留出 P95 绝对相对误差 |', '| --- | --- | ---: | ---: |']
    for name, p in profiles.items():
        for family, fitted in p['fitted_baseline_models'].items():
            e = fitted['validation']
            lines.append(f"| {name} | {family} | {e['mape_pct']:.2f}% | {e['p95_absolute_pct']:.2f}% |")
    for name, p in profiles.items():
        v = p['validation']
        lines += ['', f"{name}：跨扫描阶段重复形状 {v['cross_family_repeated_shapes']} 个，"
                  f"相对跨度中位数 {v['cross_family_span_median_pct']:.2f}%，"
                  f"P95 {v['cross_family_span_p95_pct']:.2f}%，"
                  f"超过 10% 的形状 {v['cross_family_span_over_10pct']} 个。"]
    lines += ['', '原始重复数据在各模型 dense/raw.jsonl；配置、源代码快照、版本、'
              'CSV、完整 JSON 和图表位于同一 dense 目录。旧版 TimePredictor 缓存未被替换。', '']
    (args.base / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'common_configurations': len(pairs), 'report': str(args.base / 'report.md')}, indent=2))


if __name__ == '__main__':
    main()
