"""Explicit local BE P/D/T performance probes; preview before --execute."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from saturo.experiments.runner import repo_path


def build_command(case, pool, budget=None):
    c = case['config']
    settings = case.get('study_parameters', {}).get('pool', {})
    out = Path(case['directory'])
    name = {'p': 'benchmark_p_pool_saturation.py', 'd': 'benchmark_d_pool_saturation.py',
            't': 'benchmark_t_pool_saturation.py'}[pool]
    cmd = [sys.executable, str(Path(__file__).parent / name)]
    def arg(key, value):
        if value is not None:
            cmd.extend(['--' + key.replace('_', '-'), str(value)])
    for key in ('model', 'runtime_profile', 'dtype', 'gpu_memory_utilization', 'max_model_len',
                'max_num_seqs', 'max_num_batched_tokens', 'max_wall_s', 'be_workload_csv'):
        if key == 'runtime_profile' and pool == 't':
            continue
        value = c.get(key)
        if key == 'runtime_profile':
            value = case.get('study_parameters', {}).get('pool_runtime_profile') or value
        if key in ('model', 'runtime_profile', 'be_workload_csv') and value:
            value = repo_path(value)
        arg(key, value)
    arg('load_format', 'auto')
    for key in ('tensor_parallel_size', 'pipeline_parallel_size', 'be_inflight'):
        arg(key, settings.get(key, 1 if key != 'be_inflight' else 128))
    if pool != 'd':
        arg('gpu_ids', c.get('gpu_ids', ''))
    if pool in ('p', 'd'):
        arg('rt_workload_csv', repo_path(c['rt_workload_csv']))
        arg('num_req', c['num_req'])
        arg('target_rps', c['target_rps'])
    if pool == 'p':
        arg('ttft_slo_ms', c.get('rt_ttft_slo_ms', 1000))
        arg('be_request_mode', 'normal')
        arg('chunk_ms_source', 'fixed')
        arg('p_pool_be_chunk_ms', budget if budget is not None else '<optimized-from-independent-history>')
        arg('output_json', out / 'pool_results.json')
    elif pool == 'd':
        arg('request_mode', 'local')
        arg('saturation_strategy', 'progressive')
        arg('rt_tbt_slo_ms', c.get('rt_tbt_slo_ms', 150))
        arg('ray_address', c.get('ray_address', 'local'))
        cmd += ['--no-run-simulator']
        arg('output_dir', out / 'pool')
    else:
        arg('request_mode', 'vllm')
        arg('chunk_size_source', 'sweep')
        arg('chunk_sizes', settings.get('chunk_sizes', '256,512,1024'))
        arg('duration_s', settings.get('duration_s', 400))
        arg('output_json', out / 'pool_results.json')
    return cmd


def main(pool=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, help='case JSON emitted by experiments/run.py')
    if pool is None:
        p.add_argument('--pool', choices=('p', 'd', 't'), required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--print-command', action='store_true')
    args = p.parse_args()
    pool = pool or args.pool
    if args.execute and args.print_command:
        p.error('--execute and --print-command are mutually exclusive')
    case = json.loads(repo_path(args.case).read_text())
    if not args.execute:
        print(json.dumps({'command': build_command(case, pool), 'status': 'preview',
                          'scope': 'single-pool performance probe; request payload and paper metric validation still required'}, indent=2))
        return
    c = case['config']
    for key in ('model', 'be_workload_csv', 'runtime_profile'):
        if pool == 't' and key == 'runtime_profile':
            continue
        if not c.get(key) or not repo_path(c[key]).exists():
            p.error(f'Missing {key}: {c.get(key)}')
    env = os.environ.copy()
    for key, value in c.get('env', {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    if c.get('gpu_ids'):
        env['CUDA_VISIBLE_DEVICES'] = c['gpu_ids']
    env['VLLM_USE_V1'] = '0'
    budget = None
    out = repo_path(case['directory']); out.mkdir(parents=True, exist_ok=True)
    if pool == 'p':
        from saturo.experiments.overhead.measure import p_budget
        result = p_budget(case)
        (out / 'history_budget.json').write_text(json.dumps(result, indent=2))
        if result['rt_infeasible']:
            p.error('History optimizer reports RT infeasible; refusing a fixed budget fallback')
        budget = result['best']['be_chunk_latency_ms']
        if budget <= 0:
            p.error('History optimizer has no positive BE budget')
    if pool == 'd':
        from saturo.predictors.qrf import load_required_qrf
        from saturo.experiments.overhead.measure import runtime_file
        qrf_path = repo_path(c['model_len_predictor'])
        load_required_qrf(str(qrf_path))
        runtime_file(case.get('study_parameters', {}).get('pool_runtime_profile') or c['runtime_profile'])
        env['SATURO_HISTORY_LENGTH_PREDICTOR'] = str(qrf_path)
        env['VLLM_DECODE_SATURATION_MODEL_LEN_ETA'] = str(case.get('study_parameters', {}).get(
            'quantile', c.get('resource_decode_model_len_quantile', .95)))
    command = build_command(case, pool, budget)
    (out / 'pool_command.json').write_text(json.dumps(command, indent=2))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
