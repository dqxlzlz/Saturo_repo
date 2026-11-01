import csv
import json
from pathlib import Path

import pytest

from saturo.experiments.e2e.export_timeline import collect


def fixture_run(tmp_path, tokens=None):
    run = tmp_path / 'run'
    run.mkdir()
    summary = dict(status='completed', window_s=2, elapsed_s=2.1,
                   rt_tbt_slo_ms=100, rt_ttft_slo_ms=2000,
                   rt_requests=1, be_requests=1, rt_generated_tokens=3,
                   be_generated_tokens=2, rt_completed=1, be_completed=1)
    (run / 'summary.json').write_text(json.dumps(summary))
    rows = [dict(request_id='rt', kind='rt', output_len=3, observed_tokens=3,
                 arrival_offset_s=0, first_token_s=.11, finished_s=.75),
            dict(request_id='be', kind='be', output_len=2, observed_tokens=2,
                 arrival_offset_s=0, first_token_s=.2, finished_s=.8)]
    with (run / 'requests.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tokens = tokens if tokens is not None else [
        ('rt', 'prefill', 1, 1, .1), ('be', 'throughput', 1, 1, .2),
        ('be', 'throughput', 2, 1, .4), ('rt', 'decode', 2, 1, .5),
        ('rt', 'decode', 3, 1, .7)]
    records = [dict(event='token', request_id=rid, phase=phase,
                    global_token_count=count, delta=delta,
                    scheduler_elapsed_s=t, scheduler_received_perf_s=100 + t)
               for rid, phase, count, delta, t in tokens]
    records.append(dict(event='token', request_id='__warmup__:0', phase='decode',
                        global_token_count=1, delta=1))
    for rid, phase, count, t in [('rt', 'prefill', 1, .1),
                                 ('rt', 'decode', 3, .75),
                                 ('be', 'throughput', 2, .79)]:
        records.append(dict(event='finished', request_id=rid, phase=phase,
                            global_token_count=count, scheduler_elapsed_s=t,
                            scheduler_received_perf_s=100 + t))
    records.append(dict(event='finished', request_id='__warmup__:0', global_token_count=1))
    (run / 'perf_events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    return run


def test_handoff_gap_and_all_workload_events_are_preserved(tmp_path):
    run = fixture_run(tmp_path)
    rows, meta = collect(run, tmp_path / 'out', bin_s=.5, clock_mode='mixed')
    assert rows[0]['rt_ttft_client_p99_ms'] == 110
    assert rows[1]['rt_tbt_receipt_p99_ms'] == pytest.approx(400)
    assert rows[1]['rt_p_to_d_handoff_samples'] == 1
    assert rows[1]['be_completed_req_s'] == 2
    assert rows[3]['rt_tbt_receipt_p99_ms'] is None
    exported = list(csv.DictReader((tmp_path / 'out/token_events.csv').open()))
    assert len(exported) == 5
    assert meta['counts']['rt_advancing_tokens'] == 3
    assert meta['counts']['be_advancing_tokens'] == 2
    assert meta['token_log_coverage_matches_requests']
    assert meta['source_coverage_verified']
    assert 'canonical' in meta['tbt_clock_limit']


def test_refuses_to_pad_a_short_run_to_400_seconds(tmp_path):
    with pytest.raises(ValueError, match='no duration padding'):
        collect(fixture_run(tmp_path), tmp_path / 'out', duration_s=400)


def test_missing_perf_log_is_an_error(tmp_path):
    run = fixture_run(tmp_path)
    (run / 'perf_events.jsonl').unlink()
    with pytest.raises(ValueError, match='SATURO_STEP_TIMING'):
        collect(run, tmp_path / 'out')


def test_coalesced_tokens_are_counted_without_invented_internal_zeros(tmp_path):
    run = fixture_run(tmp_path, [('rt', 'prefill', 1, 1, .1),
        ('be', 'throughput', 2, 2, .4), ('rt', 'decode', 3, 2, .5)])
    _, meta = collect(run, tmp_path / 'out')
    assert meta['diagnostic_rt_tbt_all_run_samples'] == 1
    assert meta['diagnostic_rt_tbt_all_run_max_ms'] == pytest.approx(400)
    assert meta['counts']['rt_advancing_tokens'] == 3
    assert not meta['tbt_internal_interval_coverage_complete']


def test_missing_progress_is_not_hidden_by_a_later_total_token_count(tmp_path):
    run = fixture_run(tmp_path, [('rt', 'prefill', 1, 1, .1),
        ('be', 'throughput', 2, 2, .4), ('rt', 'decode', 3, 1, .5)])
    with pytest.raises(ValueError, match='missing progress tokens'):
        collect(run, tmp_path / 'out')


def test_duplicate_events_remain_in_export_but_not_latency_samples(tmp_path):
    run = fixture_run(tmp_path)
    event = dict(event='token', request_id='rt', phase='decode',
                 global_token_count=3, delta=1, scheduler_elapsed_s=.8,
                 scheduler_received_perf_s=100.8)
    with (run / 'perf_events.jsonl').open('a') as f:
        f.write(json.dumps(event) + '\n')
    _, meta = collect(run, tmp_path / 'out')
    assert meta['diagnostic_rt_tbt_all_run_samples'] == 2
    assert meta['counts']['rt_duplicate_or_stale_events'] == 1
    assert len(list(csv.DictReader((tmp_path / 'out/token_events.csv').open()))) == 6


def test_exact_final_endpoint_is_included_without_adding_a_fake_bin(tmp_path):
    run = fixture_run(tmp_path)
    with (run / 'requests.csv').open() as f:
        rows = list(csv.DictReader(f))
    rows[1]['finished_s'] = 2
    with (run / 'requests.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result, _ = collect(run, tmp_path / 'out', bin_s=.5, clock_mode='mixed')
    assert len(result) == 4 and result[-1]['be_completed'] == 1


def test_unified_receipt_clock_deduplicates_finishes_and_excludes_warmups(tmp_path):
    run = fixture_run(tmp_path)
    path = run / 'perf_events.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        if event.get('event') == 'finished' and event.get('request_id') == 'be':
            event.update(scheduler_elapsed_s=.49, scheduler_received_perf_s=100.49)
    events.append(dict(event='finished', request_id='be', phase='throughput',
                       global_token_count=2, scheduler_elapsed_s=.9,
                       scheduler_received_perf_s=100.9))
    path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    result, meta = collect(run, tmp_path / 'out', bin_s=.5)
    assert result[0]['be_completed'] == 1 and result[1]['be_completed'] == 0
    assert result[0]['rt_ttft_client_p99_ms'] == pytest.approx(100)
    assert meta['clock_mode'] == 'diagnostic'
    assert meta['clock_validation']['csv_rt_ttft_p95_ms'] == 110
    assert meta['clock_validation']['diagnostic_rt_ttft_p95_ms'] == pytest.approx(100)
    assert meta['clock_validation']['diagnostic_completed_requests'] == 2
    assert meta['counts']['be_duplicate_finished_events'] == 1
    assert meta['counts']['warmup_finished_events'] == 1
    finishes = list(csv.DictReader((tmp_path / 'out/finished_events.csv').open()))
    assert sum(r['status'] == 'partial_target' for r in finishes) == 1


def test_missing_finished_events_cannot_fake_diagnostic_be_throughput(tmp_path):
    run = fixture_run(tmp_path)
    path = run / 'perf_events.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(''.join(json.dumps(e) + '\n' for e in events
                           if not (e.get('event') == 'finished' and e.get('request_id') == 'be')))
    with pytest.raises(ValueError, match='Missing full-target finished'):
        collect(run, tmp_path / 'out')


def test_truncated_csv_is_rejected_against_independent_summary_totals(tmp_path):
    run = fixture_run(tmp_path)
    path = run / 'requests.csv'
    rows = list(csv.DictReader(path.open()))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerow(rows[0])
    with pytest.raises(ValueError, match='CSV/summary source totals mismatch'):
        collect(run, tmp_path / 'out')
    _, meta = collect(run, tmp_path / 'inspection', allow_incomplete=True)
    assert not meta['source_coverage_verified']
    assert meta['source_totals_validation']['summary_mismatches']['be_requests'] == dict(csv=0, summary=1)
    assert meta['source_totals_validation']['unexpected_workload_event_ids']['be'] == 3


def test_unknown_perf_request_cannot_be_mislabeled_as_warmup(tmp_path):
    run = fixture_run(tmp_path)
    with (run / 'perf_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event='token', request_id='missing_workload_row')) + '\n')
    with pytest.raises(ValueError, match='Unexpected non-CSV workload event IDs'):
        collect(run, tmp_path / 'out')


@pytest.mark.parametrize('field,value', [('rt_generated_tokens', 4), ('be_completed', 0),
                                        ('rt_requests', None)])
def test_csv_token_completion_and_request_totals_must_match_summary(tmp_path, field, value):
    run = fixture_run(tmp_path)
    path = run / 'summary.json'
    summary = json.loads(path.read_text()); summary[field] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='CSV/summary source totals mismatch'):
        collect(run, tmp_path / 'out')


def test_two_short_rt_rows_with_matching_partial_finished_events_stay_incomplete(tmp_path):
    run = fixture_run(tmp_path)
    path = run / 'requests.csv'
    rows = list(csv.DictReader(path.open()))
    rt = rows[0]
    rt.update(output_len=90, observed_tokens=89, finished_s='')
    rows.append(dict(rt, request_id='rt_second'))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    path = run / 'summary.json'
    summary = json.loads(path.read_text())
    summary.update(status='incomplete', stop_reason='max_wall_s', rt_requests=2,
                   rt_generated_tokens=178, rt_completed=0)
    path.write_text(json.dumps(summary))
    events = []
    for rid in ('rt', 'rt_second'):
        for count in range(1, 90):
            t = .1 + count / 100
            events.append(dict(event='token', request_id=rid,
                phase='prefill' if count == 1 else 'decode', global_token_count=count,
                delta=1, scheduler_elapsed_s=t, scheduler_received_perf_s=100 + t))
        events.append(dict(event='finished', request_id=rid, phase='decode',
            global_token_count=89, scheduler_elapsed_s=.99, scheduler_received_perf_s=100.99))
    events.extend([dict(event='token', request_id='be', phase='throughput',
        global_token_count=c, delta=1, scheduler_elapsed_s=t, scheduler_received_perf_s=100 + t)
        for c, t in [(1, .2), (2, .4)]])
    events.append(dict(event='finished', request_id='be', phase='throughput',
        global_token_count=2, scheduler_elapsed_s=.79, scheduler_received_perf_s=100.79))
    (run / 'perf_events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    with pytest.raises(ValueError, match='Run is incomplete'):
        collect(run, tmp_path / 'out')
    _, meta = collect(run, tmp_path / 'inspection', allow_incomplete=True)
    assert not meta['source_coverage_verified']
    assert meta['token_log_coverage_matches_requests']
    assert meta['clock_validation']['csv_completed_requests'] == 1
    assert meta['clock_validation']['diagnostic_completed_requests'] == 1
    assert meta['source_totals_validation']['csv']['rt_completed'] == 0
    finishes = list(csv.DictReader((tmp_path / 'inspection/finished_events.csv').open()))
    assert [r['status'] for r in finishes].count('partial_target') == 2
