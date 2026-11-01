import json
import math
from types import SimpleNamespace

import pytest

from saturo.simulators.prefill_simulator import optimizer as optimizer_module
from saturo.simulators.prefill_simulator.config import SimulatorConfig, SLOConfig
from saturo.simulators.prefill_simulator.optimizer import ChunkLatencyOptimizer
from saturo.simulators.prefill_simulator.request import RTRequest, SimulationResult
from saturo.simulators.prefill_simulator.run_prefill_simulator import (
    optimization_to_dict, print_summary, run_prefill_profile)


REQUESTS = [RTRequest(0, 0.0, 0.2), RTRequest(1, 0.5, 0.1)]


class BudgetShapes:
    def __init__(self):
        self.budgets = []

    def predict_be_chunk_s(self, *args):
        # The initial budget must come from slack, even below this diagnostic.
        return 1.2

    def balance_decode_batch(self, budget, *args, **kwargs):
        self.budgets.append(budget)
        return dict(latency_s=budget, decode_batch=1, prefill_chunk=100)


def scripted_optimizer(monkeypatch, outcomes, *, tolerance=0.0):
    predictor = BudgetShapes()
    visited = []

    class Replay:
        def __init__(self, **kwargs):
            pass

        def run(self, chunk):
            index = len(visited)
            visited.append(chunk.target_latency_s)
            throughput, slacks = outcomes[index]
            return SimulationResult(
                chunk=chunk, duration_s=10.0,
                total_rt_requests=len(slacks),
                rt_slo_violations=sum(s < 0 for s in slacks),
                rt_slacks_s=slacks, saturation_req_per_s=throughput)

    monkeypatch.setattr(optimizer_module, 'PrefillSimulator', Replay)
    config = SimulatorConfig(slo=SLOConfig(
        ttft_slo_s=1.0, max_rt_violation_rate=tolerance))
    return ChunkLatencyOptimizer(config, predictor), predictor, visited


def test_budget_uses_each_replays_minimum_positive_slack_and_keeps_best(monkeypatch):
    optimizer, predictor, visited = scripted_optimizer(monkeypatch, [
        (1.0, [0.0, 0.07]),
        (2.0, [0.01, 0.02]),
        (1.5, [0.3, 0.4]),
    ])
    result = optimizer.optimize(REQUESTS)

    assert visited == pytest.approx([0.8, 0.87, 0.88])
    assert predictor.budgets == visited
    assert result.best_result.chunk.target_latency_s == pytest.approx(0.87)
    assert result.initial_budget_s == pytest.approx(0.8)
    assert result.evaluated_candidates == 3
    assert result.stop_reason == 'throughput_not_improved'
    assert result.budget_history_s == pytest.approx([0.8, 0.87, 0.88])


def test_slack_steps_are_not_capped_by_original_maximum_slack(monkeypatch):
    optimizer, _, visited = scripted_optimizer(monkeypatch, [
        (1.0, [1.0, 2.0]), (0.5, [0.0, 0.0])])
    optimizer.optimize(REQUESTS)
    assert visited == pytest.approx([0.8, 1.8])


def test_throughput_plateau_stops_without_evaluating_another_budget(monkeypatch):
    optimizer, _, visited = scripted_optimizer(monkeypatch, [
        (1.0, [0.1, 0.2]), (1.0, [0.2, 0.3])])
    result = optimizer.optimize(REQUESTS)
    assert len(visited) == 2
    assert result.best_result is result.all_results[0]
    assert result.stop_reason == 'throughput_not_improved'


def test_no_positive_remaining_slack_stops(monkeypatch):
    optimizer, _, visited = scripted_optimizer(monkeypatch, [(1.0, [0.0, 0.0])])
    result = optimizer.optimize(REQUESTS)
    assert len(visited) == 1
    assert result.stop_reason == 'no_positive_slack'


def test_tolerated_negative_initial_slack_starts_with_idle_chunk(monkeypatch):
    optimizer, predictor, visited = scripted_optimizer(monkeypatch, [
        (0.0, [-0.1, 0.05]), (1.0, [0.0, 0.0])], tolerance=1.0)
    result = optimizer.optimize([RTRequest(0, 0.0, 1.1), RTRequest(1, 2.0, 0.1)])
    assert visited == pytest.approx([0.0, 0.05])
    assert predictor.budgets == pytest.approx([0.05])
    assert result.all_results[0].chunk.decode_batch_size == 0
    assert result.all_results[0].chunk.latency_s == 0


def test_overloaded_rt_trace_still_disables_be(monkeypatch):
    optimizer, predictor, visited = scripted_optimizer(monkeypatch, [])
    result = optimizer.optimize([RTRequest(0, 0.0, 1.1)])
    assert result.rt_infeasible
    assert result.stop_reason == 'rt_infeasible'
    assert result.best_result.saturation_req_per_s == 0
    assert not predictor.budgets and not visited


def test_empty_history_uses_single_fallback_without_slack_loop(monkeypatch):
    optimizer, _, visited = scripted_optimizer(monkeypatch, [(1.0, [])])
    result = optimizer.optimize([])
    assert visited == [1.2]
    assert result.stop_reason == 'no_positive_slack'


def test_float_increment_that_cannot_advance_budget_stops(monkeypatch):
    optimizer, _, visited = scripted_optimizer(monkeypatch, [
        (1.0, [math.nextafter(0.0, 1.0), 0.1])])
    result = optimizer.optimize(REQUESTS)
    assert len(visited) == 1
    assert result.stop_reason == 'no_budget_progress'


def test_iteration_limit_is_separate_from_saved_candidate_limit(monkeypatch):
    optimizer, _, visited = scripted_optimizer(monkeypatch, [
        (1.0, [0.1, 0.2]), (2.0, [0.1, 0.2])])
    result = optimizer.optimize(REQUESTS, max_iterations=2, max_candidates=1)
    assert len(visited) == result.evaluated_candidates == 2
    assert len(result.all_results) == 1
    assert result.best_result is result.all_results[0]
    assert result.stop_reason == 'iteration_limit'
    assert optimization_to_dict(result)['search']['budgets_ms'] == pytest.approx([800, 900])


def test_legacy_grid_argument_is_ignored_with_warning(monkeypatch):
    optimizer, _, visited = scripted_optimizer(monkeypatch, [
        (1.0, [0.07, 0.1]), (0.5, [0.0, 0.0])])
    with pytest.warns(DeprecationWarning, match='deprecated and ignored'):
        result = optimizer.optimize(REQUESTS, budget_granularity_ms=9999)
    assert visited == pytest.approx([0.8, 0.87])
    assert optimization_to_dict(result)['search']['method'] == 'slack'


@pytest.mark.parametrize('slack', [float('nan'), float('inf')])
def test_nonfinite_simulation_feedback_is_rejected(monkeypatch, slack):
    optimizer, _, _ = scripted_optimizer(monkeypatch, [(1.0, [slack, 0.1])])
    with pytest.raises(ValueError, match='must be finite'):
        optimizer.optimize(REQUESTS)


def test_nonfinite_history_is_rejected(monkeypatch):
    optimizer, _, _ = scripted_optimizer(monkeypatch, [])
    with pytest.raises(ValueError, match='must be finite'):
        optimizer.optimize([RTRequest(0, 0.0, float('nan'))])


@pytest.mark.parametrize('limit', [0, -1])
def test_nonpositive_iteration_limit_is_rejected(monkeypatch, limit):
    optimizer, _, _ = scripted_optimizer(monkeypatch, [])
    with pytest.raises(ValueError, match='must be positive'):
        optimizer.optimize(REQUESTS, max_iterations=limit)


def test_profile_pipeline_runs_real_predictor_and_replay_with_slack_metadata(tmp_path, capsys):
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({
        'provenance': 'Synthetic timing coefficients for a CPU integration test.',
        'time_predictor': {'model_type': 'iteration_level',
                           'coef': [0.01, 0.0, 0.1, 0.0], 'intercept': 0.1},
        'kv_cache_block': {'total_gpu_blocks': 1024, 'block_size': 16},
    }))
    be_trace = tmp_path / 'be.csv'
    be_trace.write_text('prompt_len,output_len\n128,16\n128,16\n')
    payload = run_prefill_profile(
        rt_trace='unused.csv', runtime_profile=str(profile),
        be_trace=str(be_trace), num_req=3, target_rps=2.0,
        ttft_slo=0.05, max_prefill_chunk=256,
        rt_rows=[{'prompt_len': 16, 'output_len': 1, 'arrival_offset_s': i * 0.5}
                 for i in range(3)],
    )
    assert payload['search']['method'] == 'slack'
    assert payload['search']['initial_budget_ms'] == pytest.approx(49.74)
    assert payload['candidates'][0]['target_latency_ms'] == pytest.approx(49.74)
    assert payload['sweep']['budget_granularity_ms'] is None
    assert payload['search']['evaluated_candidates'] >= 2
    assert payload['search']['stop_reason'] == 'throughput_not_improved'
    assert payload['best']['saturation_req_per_s'] > 0
    print_summary(payload, print_all=True)
    text = capsys.readouterr().out
    assert 'Slack-driven budget search' in text
    assert 'grid step' not in text


def test_online_history_budget_uses_the_shared_slack_search(tmp_path, monkeypatch):
    from saturo.history import HistorySnapshot
    from saturo.runtime.history import RuntimeHistoryController
    from saturo.simulators.prefill_simulator import run_prefill_simulator as entry

    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({
        'provenance': 'Synthetic timing coefficients for a CPU integration test.',
        'time_predictor': {'model_type': 'iteration_level',
                           'coef': [0.01, 0.0, 0.1, 0.0], 'intercept': 0.1},
        'kv_cache_block': {'total_gpu_blocks': 1024, 'block_size': 16},
    }))
    be_trace = tmp_path / 'be.csv'
    be_trace.write_text('prompt_len,output_len\n128,16\n128,16\n')
    snapshot = HistorySnapshot('unit', 'unit-model', 0.0, 1.5, [
        dict(run_id='unit', request_id=str(i), arrival_time_s=i * 0.5,
             prompt_len=16, output_len=1) for i in range(3)])
    controller = RuntimeHistoryController.__new__(RuntimeHistoryController)
    controller.source = 'redis-history:unit'
    controller.window_s = 1.5
    controller.args = SimpleNamespace(
        runtime_profile=str(profile), be_workload_csv=str(be_trace),
        rt_ttft_slo_ms=50.0, rt_max_violation_rate=0.0,
        resource_memory_watermark=0.9, resource_max_prefill_chunk=256,
        resource_prefill_search_granularity=1,
        resource_profile_max_candidates=128,
        be_input_drop_frac=0.0, be_output_keep_frac=None,
        be_output_max_len=None,
    )
    searches = []
    original = entry.run_prefill_profile

    def observed(**kwargs):
        payload = original(**kwargs)
        searches.append(payload['search'])
        return payload

    monkeypatch.setattr(entry, 'run_prefill_profile', observed)
    budget = controller._optimize_p_budget(snapshot, dict(
        runtime_profile=str(profile), max_model_len=256,
        num_gpu_blocks=1024), pool_size=2)
    assert budget['chunk_ms'] > 0
    assert searches[0]['method'] == 'slack'
    assert searches[0]['initial_budget_ms'] == pytest.approx(49.74)
