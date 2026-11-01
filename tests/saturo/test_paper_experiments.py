import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from saturo.experiments.catalog import STUDIES
from saturo.experiments.runner import (ROOT, execute_cases, generate_cases,
                                       load_config, prepare_case)
from saturo.resource_allocation.demand import DemandBasedSolver
from saturo.resource_allocation.planner import SaturoResourcePlanner
from saturo.resource_allocation.profiles import AllocationProfiles
from saturo.resource_allocation.solver import ResourceAllocationSolver, Workload
from saturo.runtime.pool_controls import pool_accepts_be
from saturo.runtime.workload import BE, DECODE, PREFILL, TPUT, SaturoRequest
from saturo.runtime.scheduler import SaturoGlobalScheduler
from saturo.runtime.worker import WorkerState


@pytest.fixture
def config():
    return load_config('saturo/experiments/e2e/paper.example.json', 'paper')


def profiles():
    return AllocationProfiles.from_dict({
        'configs': {'tp1': {'gpus': 1, 'tp': 1, 'pp': 1}},
        'pools': {
            'P': {'tp1': {'mu': 1, 'saturation': [{'rt_rps': .1, 'be_throughput': 1000}]}},
            'D': {'tp1': {'mu': 1, 'saturation': [{'rt_rps': .1, 'be_throughput': 10}]}},
            'T': {'tp1': {'saturation': [{'rt_rps': 0, 'be_throughput': 100}]}}}})


def test_demand_baseline_reserves_minimum_rt_resources():
    p = profiles(); w = Workload(.1, .1)
    saturo = ResourceAllocationSolver(p).solve(total_gpus=8, workload=w)[0]
    demand = DemandBasedSolver(p).solve(total_gpus=8, workload=w)[0]
    assert (saturo.p.instances, saturo.d.instances, saturo.t.instances) == (7, 1, 0)
    assert (demand.p.instances, demand.d.instances, demand.t.instances) == (1, 1, 6)


def test_disabled_pool_mask_preserves_rt_capacity_and_source_table():
    source = profiles(); original = copy.deepcopy(source.to_dict())
    planner = SaturoResourcePlanner.__new__(SaturoResourcePlanner)
    planner.args = SimpleNamespace(resource_config_filter='', enable_prefill_saturation=False,
                                   enable_decode_saturation=True)
    masked = planner._filter_profiles(source)
    assert source.to_dict() == original
    assert masked.profile('P', 'tp1').mu == 1
    assert masked.profile('P', 'tp1').saturation[0].be_throughput == 0
    assert masked.profile('D', 'tp1').saturation[0].be_throughput == 10


@pytest.mark.parametrize('p,d,expected', [(False, False, [False, False, True]),
                                           (False, True, [False, True, True]),
                                           (True, False, [True, False, True])])
def test_pool_eligibility(p, d, expected):
    opts = {'enable_prefill_saturation': p, 'enable_decode_saturation': d}
    assert [pool_accepts_be(role, opts) for role in (PREFILL, DECODE, TPUT)] == expected


def test_disabled_pool_does_not_receive_be_preload():
    s = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    s.args = SimpleNamespace(enable_prefill_saturation=False, enable_decode_saturation=False)
    def worker(i, role):
        return WorkerState(worker_id=i, role=role, host='local', config_name='tp1',
                           tensor_parallel_size=1, pipeline_parallel_size=1,
                           stderr_file=None, ready=True)
    s.workers = {i: worker(i, role) for i, role in enumerate((PREFILL, DECODE, TPUT))}
    s.all_be = [SaturoRequest('be_0', BE, 32, 8, 0)]
    s._preload_be_to_workers()
    assert not s.workers[0].pending_commands and not s.workers[1].pending_commands
    assert s.workers[2].pending_commands[0]['kind'] == BE


def test_figure11_preserves_both_panels_and_disables_be(config):
    cases = generate_cases('fig11_saturation', config)
    assert len(cases) == 24
    assert all(not c['config']['adaptive_role_switching'] for c in cases)
    low = [c for c in cases if c['config']['target_rps'] == 12]
    high = [c for c in cases if c['config']['target_rps'] == 32]
    assert all(c['study_parameters']['panel_a'] and not c['study_parameters']['panel_b'] for c in low)
    assert all(c['study_parameters']['panel_b'] and not c['study_parameters']['panel_a'] for c in high)
    off = next(c for c in cases if c['variant'] == 'without_saturation')
    assert not off['config']['enable_prefill_saturation'] and not off['config']['enable_decode_saturation']


def test_figure12_keeps_both_saturation_modules(config):
    cases = generate_cases('fig12_allocation', config)
    assert len(cases) == 14
    assert all(c['config']['enable_prefill_saturation'] and c['config']['enable_decode_saturation'] for c in cases)
    assert {c['config']['resource_allocation_policy'] for c in cases} == {'saturation_aware', 'demand_based'}


def test_slo_scaling_changes_both_slos(config):
    cases = generate_cases('fig9_slo_scale', config, methods=['saturo'])
    assert {(c['config']['rt_ttft_slo_ms'], c['config']['rt_tbt_slo_ms']) for c in cases} >= {(1300, 195), (750, 112.5)}


def test_quantile_merges_env_and_expands_case_keys(config):
    config['models']['mistral']['env'] = {'FROM_MODEL': 'ok'}
    config['workloads']['conv']['env'] = {'FROM_WORKLOAD': 'ok'}
    case = generate_cases('fig14_quantile', config)[0]
    env = case['config']['env']
    assert env['FROM_MODEL'] == env['FROM_WORKLOAD'] == 'ok'
    assert env['VLLM_DECODE_SATURATION_MODEL_LEN_ETA'] == '0.5'
    assert env['SATURO_TRACE_KEY'].endswith(case['id'])
    assert '{model}' not in env['SATURO_HISTORY_KEY']


def test_delta_is_not_substituted_by_lookahead(config):
    cases = generate_cases('fig14_update_interval', config)
    assert [c['study_parameters']['update_interval_s'] for c in cases] == [5, 10, 20, 40, 80, 160]
    assert [c['config']['saturation_update_interval_s'] for c in cases] == [5, 10, 20, 40, 80, 160]
    assert all(c['config']['decode_target_async'] for c in cases)
    assert all(c['config']['decode_saturation_lookahead_s'] == c['config']['saturation_update_interval_s'] for c in cases)
    # Remaining profile/memory/continuous-BE/metric gaps still block paper claims.
    assert all(prepare_case(c, config, '/tmp/unused')['blockers'] for c in cases)
    smoke = generate_cases('fig14_update_interval', config, mode='smoke')
    assert all(prepare_case(c, config, '/tmp/unused')['command'] for c in smoke)
    assert all(not prepare_case(c, config, '/tmp/unused')['blockers'] for c in smoke)


def test_cpu_benchmark_keeps_existing_allocation_table(config):
    config['models']['mistral']['resource_profile'] = '/existing/table.json'
    case = generate_cases('fig13_allocator_overhead', config)[0]
    assert prepare_case(case, config, '/tmp/unused')['config']['resource_profile'] == '/existing/table.json'


def test_paper_backend_gaps_stop_before_creating_run_dirs(config, tmp_path):
    case = prepare_case(generate_cases('fig6_e2e', config, methods=['saturo'])[0], config, tmp_path)
    with pytest.raises(ValueError, match='before starting any case'):
        execute_cases([case])
    assert not list(tmp_path.iterdir())


def test_unknown_model_does_not_silently_use_base_config(config):
    with pytest.raises(ValueError, match='Unknown model'):
        generate_cases('fig6_e2e', config, models=['typo'])


def test_metric_joint_attainment_rejects_missing_tbt():
    from saturo.experiments.analysis.summarize import summarize
    rt = {'kind': 'rt', 'arrival_offset_s': '0', 'first_token_s': '.1', 'finished_s': '1'}
    result = summarize([rt], 0, 10, 1000, 150)
    assert result['rt_joint_attainment'] is None
    rt['tbt_ms'] = '200'
    assert summarize([rt], 0, 10, 1000, 150)['rt_joint_attainment'] == 0
    rt['tbt_ms'] = '100'
    assert summarize([rt], 0, 10, 1000, 150)['rt_joint_attainment'] == 1


def test_unfinished_rt_counts_as_failure():
    from saturo.experiments.analysis.summarize import summarize
    rows = [{'kind': 'rt', 'arrival_offset_s': '0', 'first_token_s': '.1', 'finished_s': '1', 'tbt_ms': '100'},
            {'kind': 'rt', 'arrival_offset_s': '1', 'first_token_s': '', 'finished_s': ''}]
    assert summarize(rows, 0, 10, 1000, 150)['rt_joint_attainment'] == .5


def test_history_window_refuses_short_trace(tmp_path):
    from saturo.experiments.overhead.measure import window_rows
    p = tmp_path / 'history.csv'
    p.write_text('prompt_len,output_len,arrival_offset_s\n10,5,0\n20,8,5\n')
    with pytest.raises(ValueError, match='shorter'):
        window_rows(p, 600)


def test_be_data_uses_real_pairs_without_placeholder_lengths(tmp_path):
    from saturo.experiments.workloads.prepare import token_pairs
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(c) for c in text]
    source = tmp_path / 'sharegpt.json'
    source.write_text(json.dumps([{'conversations': [
        {'from': 'human', 'value': 'hello'}, {'from': 'gpt', 'value': 'world!'}]}]))
    pair = token_pairs(source, 'sharegpt', Tokenizer(), 0)[0]
    assert pair['prompt_len'] == 5 and pair['output_len'] == 6
    assert pair['prompt_token_ids'] == [104, 101, 108, 108, 111]


def test_duplicate_cases_stop_before_overwriting_results(tmp_path):
    with pytest.raises(ValueError, match='Duplicate case IDs'):
        execute_cases([{'id': 'same'}, {'id': 'same'}])
    assert not list(tmp_path.iterdir())


def test_integer_and_float_rps_have_same_case_identity(config):
    a = generate_cases('fig6_e2e', config, methods=['saturo'], rps=[20])[0]
    b = generate_cases('fig6_e2e', config, methods=['saturo'], rps=[20.0])[0]
    assert a['id'] == b['id']


@pytest.mark.parametrize('observed,target,coverage,tbt,first,expected', [
    (1, 1, 'True', '', '.1', 1),
    (3, 3, 'False', '20', '.1', None),
    (3, 4, 'True', '20', '.1', 0),
    (3, 3, 'True', '20', '', None),
])
def test_joint_metrics_respect_coverage_and_completion(
        observed, target, coverage, tbt, first, expected):
    from saturo.experiments.analysis.summarize import summarize
    row = dict(kind='rt', arrival_offset_s='0', first_token_s=first,
               finished_s='1', observed_tokens=str(observed),
               output_len=str(target), tbt_complete=coverage, tbt_ms=tbt)
    result = summarize([row], 0, 10, 1000, 150)
    assert result['rt_joint_attainment'] == expected
    if coverage == 'False':
        assert result['tbt_p99_ms'] is None


def test_analysis_reads_recorded_tbt_definition(tmp_path, monkeypatch):
    from saturo.experiments.analysis.summarize import main
    from saturo.runtime.workload import REQUEST_TBT_DEFINITION
    path = tmp_path / 'requests.csv'
    path.write_text(
        'kind,arrival_offset_s,first_token_s,finished_s,observed_tokens,output_len,tbt_complete,tbt_ms,tbt_definition\n'
        f'rt,0,.1,.2,1,1,True,,{REQUEST_TBT_DEFINITION}\n')
    output = tmp_path / 'metrics'
    monkeypatch.setattr('sys.argv', ['summarize', '--requests', str(path),
        '--duration-s', '10', '--output-dir', str(output)])
    main()
    result = json.loads((output / 'paper_metrics.json').read_text())
    assert result['rt_joint_attainment'] == 1
    assert result['tbt_definition'] == REQUEST_TBT_DEFINITION
    monkeypatch.setattr('sys.argv', ['summarize', '--requests', str(path),
        '--duration-s', '10', '--output-dir', str(output),
        '--tbt-definition', 'per-request mean'])
    with pytest.raises(SystemExit):
        main()
