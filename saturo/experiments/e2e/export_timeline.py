"""Export measured Saturo timelines; no simulated or fabricated baseline data."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


def percentile(values, q=.99):
    values = sorted(values)
    return values[math.ceil(q * len(values)) - 1] if values else None


def collect(run_dir, output_dir, *, duration_s=None, bin_s=5., start_s=0.,
            allow_incomplete=False, clock_mode='diagnostic'):
    if clock_mode not in ('diagnostic', 'mixed'):
        raise ValueError('clock_mode must be diagnostic or mixed')
    run, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    summary = json.loads((run / 'summary.json').read_text())
    window = float(summary['window_s'])
    duration = window - start_s if duration_s is None else float(duration_s)
    if (not all(math.isfinite(v) for v in (window, duration, bin_s, start_s))
            or min(duration, bin_s) <= 0 or start_s < 0
            or start_s + duration > window + 1e-9):
        raise ValueError('Requested plot must fit the actual sending window; no duration padding')
    if summary.get('status') != 'completed' and not allow_incomplete:
        raise ValueError('Run is incomplete; use --allow-incomplete to label and inspect it')
    if float(summary['elapsed_s']) + 1e-9 < start_s + duration:
        raise ValueError('Run ended before the requested timeline window')
    source = run / 'perf_events.jsonl'
    if not source.is_file():
        raise ValueError('Missing perf_events.jsonl; collect with SATURO_STEP_TIMING=1')
    request_file = run / 'requests.csv'
    rows = list(csv.DictReader(request_file.open()))
    requests = {r['request_id']: r for r in rows}
    if len(requests) != len(rows):
        raise ValueError('Duplicate request IDs in requests.csv')
    if any(row['kind'] not in ('rt', 'be') for row in rows):
        raise ValueError('Unknown request kind in requests.csv')
    csv_completed = {rid for rid, row in requests.items() if row.get('finished_s')
                     and int(row.get('observed_tokens') or 0) >= int(row['output_len'])}
    csv_totals = {}
    for kind in ('rt', 'be'):
        selected = [row for row in rows if row['kind'] == kind]
        csv_totals[kind + '_requests'] = len(selected)
        csv_totals[kind + '_generated_tokens'] = sum(
            int(row.get('observed_tokens') or 0) for row in selected)
        csv_totals[kind + '_completed'] = sum(
            row['request_id'] in csv_completed for row in selected)
    summary_mismatches = {
        key: dict(csv=value, summary=summary.get(key))
        for key, value in csv_totals.items() if summary.get(key) != value}
    if summary_mismatches and not allow_incomplete:
        raise ValueError(f'CSV/summary source totals mismatch: {summary_mismatches}')
    n = math.ceil(duration / bin_s)
    end = start_s + duration
    bins = [dict(start_s=start_s + i * bin_s,
                 end_s=min(end, start_s + (i + 1) * bin_s),
                 be_completed=0, rt_arrivals=0, ttft=[], tbt=[], handoffs=0)
            for i in range(n)]

    def index(t):
        # Include the final trace arrival exactly at the right endpoint.
        if t < start_s or t > end:
            return None
        return min(int((t - start_s) // bin_s), n - 1)

    request_metrics = []
    csv_ttfts = []
    for row in rows:
        entry = dict(row)
        entry['ttft_client_ms'] = None
        if row['kind'] == 'rt':
            at = index(float(row['arrival_offset_s']))
            if at is not None:
                bins[at]['rt_arrivals'] += 1
            if row.get('first_token_s'):
                ttft = (float(row['first_token_s']) - float(row['arrival_offset_s'])) * 1000
                entry['ttft_client_ms'] = ttft
                csv_ttfts.append(ttft)
                if at is not None and clock_mode == 'mixed':
                    bins[at]['ttft'].append(ttft)
        elif (row['kind'] == 'be' and row.get('finished_s')
              and int(row.get('observed_tokens') or 0) >= int(row['output_len'])):
            at = index(float(row['finished_s']))
            if at is not None and clock_mode == 'mixed':
                bins[at]['be_completed'] += 1
        entry['first_token_diagnostic_receipt_s'] = None
        entry['ttft_diagnostic_receipt_ms'] = None
        entry['finished_diagnostic_receipt_s'] = None
        request_metrics.append(entry)
    metrics_by_id = {row['request_id']: row for row in request_metrics}

    out.mkdir(parents=True, exist_ok=True)
    token_fields = ['source_line', 'request_id', 'kind', 'attempt_id', 'phase',
                    'worker_id', 'worker_role', 'worker_time_s',
                    'scheduler_elapsed_s', 'scheduler_received_perf_s',
                    'global_token_count', 'delta', 'advancing_tokens',
                    'tbt_receipt_ms', 'p_to_d_handoff', 'status']
    state, first_diagnostic, finished_diagnostic = {}, {}, {}
    counts = collections.Counter()
    unexpected_ids = collections.Counter()
    raw_hash = hashlib.sha256()
    gap_count, gap_max = 0, None
    origin_min, origin_max = None, None
    origin_anchor = None
    finish_fields = ['source_line', 'request_id', 'kind', 'phase', 'worker_id',
                     'global_token_count', 'diagnostic_receipt_s', 'status']
    with (out / 'token_events.csv').open('w', newline='') as dest, \
            (out / 'finished_events.csv').open('w', newline='') as finish_dest, \
            source.open('rb') as handle:
        writer = csv.DictWriter(dest, fieldnames=token_fields)
        writer.writeheader()
        finish_writer = csv.DictWriter(finish_dest, fieldnames=finish_fields)
        finish_writer.writeheader()
        for line_no, raw in enumerate(handle, 1):
            raw_hash.update(raw)
            event = json.loads(raw)
            counts['source_events'] += 1
            event_type = event.get('event')
            if event_type not in ('token', 'finished'):
                continue
            rid = str(event.get('request_id', ''))
            if rid not in requests:
                if rid.startswith('__warmup__'):
                    counts['warmup_' + event_type + '_events'] += 1
                else:
                    counts['unexpected_' + event_type + '_events'] += 1
                    unexpected_ids[rid] += 1
                continue
            row = requests[rid]
            kind = row['kind']
            received = float(event['scheduler_received_perf_s'])
            elapsed = float(event['scheduler_elapsed_s'])
            if not math.isfinite(received) or not math.isfinite(elapsed):
                raise ValueError('Nonfinite diagnostic receipt time')
            origin = received - elapsed
            origin_min = origin if origin_min is None else min(origin_min, origin)
            origin_max = origin if origin_max is None else max(origin_max, origin)
            if origin_anchor is None:
                origin_anchor = origin
            diagnostic_elapsed = received - origin_anchor
            if event_type == 'finished':
                full_target = int(event.get('global_token_count', 0)) >= int(row['output_len'])
                status = 'partial_target'
                if full_target:
                    status = 'duplicate_full_target' if rid in finished_diagnostic else 'first_full_target'
                    if rid in finished_diagnostic:
                        counts[kind + '_duplicate_finished_events'] += 1
                    else:
                        finished_diagnostic[rid] = diagnostic_elapsed
                        metrics_by_id[rid]['finished_diagnostic_receipt_s'] = diagnostic_elapsed
                        if kind == 'be' and clock_mode == 'diagnostic':
                            at = index(diagnostic_elapsed)
                            if at is not None:
                                bins[at]['be_completed'] += 1
                finish_writer.writerow(dict(source_line=line_no, request_id=rid,
                    kind=kind, phase=event.get('phase'), worker_id=event.get('worker_id'),
                    global_token_count=event.get('global_token_count'),
                    diagnostic_receipt_s=diagnostic_elapsed, status=status))
                continue
            current = min(int(event['global_token_count']), int(row['output_len']))
            previous = state.get(rid)
            before = previous['tokens'] if previous else 0
            advance = max(current - before, 0)
            gap, handoff = None, False
            status = 'advance' if advance else 'duplicate_or_stale'
            if advance:
                counts[kind + '_advancing_tokens'] += advance
                counts[kind + '_unobserved_internal_intervals'] += max(advance - 1, 0)
                counts[kind + '_missing_progress_tokens'] += max(
                    advance - int(event.get('delta', advance)), 0)
                if kind == 'rt' and rid not in first_diagnostic:
                    first_diagnostic[rid] = diagnostic_elapsed
                    ttft = (diagnostic_elapsed - float(row['arrival_offset_s'])) * 1000
                    metrics_by_id[rid]['first_token_diagnostic_receipt_s'] = diagnostic_elapsed
                    metrics_by_id[rid]['ttft_diagnostic_receipt_ms'] = ttft
                    at = index(float(row['arrival_offset_s']))
                    if at is not None and clock_mode == 'diagnostic':
                        bins[at]['ttft'].append(ttft)
                if previous is not None and kind == 'rt':
                    gap = (received - previous['received']) * 1000
                    if gap < 0:
                        raise ValueError('Nonmonotonic diagnostic receipt clock')
                    handoff = previous['phase'] == 'prefill' and event.get('phase') == 'decode'
                    gap_count += 1
                    gap_max = gap if gap_max is None else max(gap_max, gap)
                    at = index(diagnostic_elapsed)
                    if at is not None:
                        bins[at]['tbt'].append(gap)
                        bins[at]['handoffs'] += int(handoff)
                    else:
                        counts['rt_gap_samples_outside_plot'] += 1
                state[rid] = dict(tokens=current, received=received, phase=event.get('phase'))
            else:
                counts[kind + '_duplicate_or_stale_events'] += 1
            counts[kind + '_token_events'] += 1
            writer.writerow(dict(source_line=line_no, request_id=rid, kind=kind,
                attempt_id=event.get('attempt_id'), phase=event.get('phase'),
                worker_id=event.get('worker_id'), worker_role=event.get('worker_role'),
                worker_time_s=event.get('time_s'), scheduler_elapsed_s=elapsed,
                scheduler_received_perf_s=received, global_token_count=event['global_token_count'],
                delta=event.get('delta'), advancing_tokens=advance,
                tbt_receipt_ms=gap, p_to_d_handoff=handoff, status=status))

    missing = {rid: dict(recorded=state.get(rid, {}).get('tokens', 0),
                        requested_observed=int(row.get('observed_tokens') or 0))
               for rid, row in requests.items()
               if state.get(rid, {}).get('tokens', 0) != int(row.get('observed_tokens') or 0)}
    missing_progress = counts['rt_missing_progress_tokens'] + counts['be_missing_progress_tokens']
    if (missing or missing_progress) and not allow_incomplete:
        raise ValueError(f'Token log coverage mismatch: {len(missing)} requests, '
                         f'{missing_progress} missing progress tokens')
    if unexpected_ids and not allow_incomplete:
        raise ValueError(f'Unexpected non-CSV workload event IDs: {dict(unexpected_ids)}')
    missing_finishes = sorted(csv_completed - set(finished_diagnostic))
    extra_finishes = sorted(set(finished_diagnostic) - csv_completed)
    if (missing_finishes or extra_finishes) and not allow_incomplete:
        raise ValueError(f'Missing full-target finished events for {len(missing_finishes)} requests; '
                         f'{len(extra_finishes)} finished requests absent from CSV completion set')
    source_issues = []
    if summary.get('status') != 'completed':
        source_issues.append('Run status is not completed')
    if summary_mismatches:
        source_issues.append('CSV request/token/completion totals do not match summary')
    if unexpected_ids:
        source_issues.append('Unexpected workload IDs occur outside requests.csv')
    if missing or missing_progress:
        source_issues.append('Token progress does not match requests.csv')
    if missing_finishes or extra_finishes:
        source_issues.append('Full-target finished IDs do not match CSV completion set')
    timeline = []
    tbt_slo = float(summary.get('rt_tbt_slo_ms', 100))
    for row in bins:
        width = row['end_s'] - row['start_s']
        timeline.append(dict(start_s=row['start_s'], end_s=row['end_s'],
            be_completed=row['be_completed'], be_completed_req_s=row['be_completed'] / width,
            rt_arrivals=row['rt_arrivals'], rt_ttft_samples=len(row['ttft']),
            rt_ttft_client_p99_ms=percentile(row['ttft']),
            rt_ttft_client_max_ms=max(row['ttft'], default=None),
            rt_tbt_samples=len(row['tbt']), rt_tbt_receipt_p99_ms=percentile(row['tbt']),
            rt_tbt_receipt_max_ms=max(row['tbt'], default=None),
            rt_p_to_d_handoff_samples=row['handoffs'],
            rt_tbt_over_slo_samples=sum(v > tbt_slo for v in row['tbt'])))
    for name, data in [('timeline.csv', timeline), ('request_metrics.csv', request_metrics)]:
        with (out / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]) if data else ['request_id'])
            writer.writeheader()
            writer.writerows(data)
    diagnostic_ttfts = [row['ttft_diagnostic_receipt_ms'] for row in request_metrics
                       if row['ttft_diagnostic_receipt_ms'] is not None]
    meta = dict(method='Saturo', source_run=str(run), run_status=summary.get('status'),
        clock_mode=clock_mode,
        actual_sending_window_s=window, plot_start_s=start_s, plot_duration_s=duration,
        bin_s=bin_s, percentile_definition='nearest rank ceil(.99*n)',
        endpoint_definition='half-open bins; final right endpoint included',
        be_definition=('first full-target diagnostic finished receipt' if clock_mode == 'diagnostic'
                       else 'canonical CSV finished_s') + ' counts / bin duration; finite preloaded backlog',
        ttft_definition=('first advancing diagnostic RT receipt' if clock_mode == 'diagnostic'
                         else 'canonical CSV first_token_s') + ' minus scheduled arrival; binned by arrival',
        tbt_definition='diagnostic drain-receipt gaps per advancing RT output event; binned by second output; includes P-to-D and attempt transitions',
        tbt_clock_limit='scheduler_received_perf_s precedes perf JSON writing and canonical _on_token; these are diagnostic receipt measurements, not canonical summary samples',
        coalesced_output_definition='all token counts retained; unknown internal intervals are counted, never filled with synthetic zeros',
        empty_latency_bins='null/blank, never zero or interpolated',
        workload_requests=len(rows), rt_requests=sum(r['kind'] == 'rt' for r in rows),
        be_requests=sum(r['kind'] == 'be' for r in rows),
        source_coverage_verified=not source_issues,
        source_completeness_issues=source_issues,
        source_totals_validation=dict(csv=csv_totals, summary_mismatches=summary_mismatches,
                                     unexpected_workload_event_ids=dict(unexpected_ids)),
        token_log_coverage_matches_requests=not missing and not missing_progress,
        token_log_coverage_mismatch=missing,
        tbt_internal_interval_coverage_complete=not counts['rt_unobserved_internal_intervals'],
        counts=dict(counts), diagnostic_rt_tbt_all_run_max_ms=gap_max,
        diagnostic_rt_tbt_all_run_samples=gap_count,
        canonical_summary=summary,
        receipt_origin_range_s=[origin_min, origin_max],
        receipt_origin_anchor_s=origin_anchor,
        clock_validation=dict(csv_completed_requests=len(csv_completed),
            diagnostic_completed_requests=len(finished_diagnostic),
            csv_be_completed=sum(requests[rid]['kind'] == 'be' for rid in csv_completed),
            diagnostic_be_completed=sum(requests[rid]['kind'] == 'be' for rid in finished_diagnostic),
            missing_finished_request_ids=missing_finishes,
            extra_finished_request_ids=extra_finishes,
            csv_rt_ttft_samples=len(csv_ttfts), diagnostic_rt_ttft_samples=len(diagnostic_ttfts),
            csv_rt_ttft_p95_ms=percentile(csv_ttfts, .95),
            diagnostic_rt_ttft_p95_ms=percentile(diagnostic_ttfts, .95)),
        postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={'perf_events.jsonl': raw_hash.hexdigest(),
                       'requests.csv': hashlib.sha256(request_file.read_bytes()).hexdigest(),
                       'summary.json': hashlib.sha256((run / 'summary.json').read_bytes()).hexdigest()})
    (out / 'timeline_metadata.json').write_text(json.dumps(meta, indent=2))
    return timeline, meta


def plot(timeline, meta, output_dir, *, slo_lines=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout='constrained')
    edges = [r['start_s'] for r in timeline] + [timeline[-1]['end_s']]
    mid = [(r['start_s'] + r['end_s']) / 2 for r in timeline]
    color = '#6d4c9a'
    axes[0].stairs([r['be_completed_req_s'] for r in timeline], edges, color=color, label='Saturo')
    for axis, column, label in [
            (axes[1], 'rt_ttft_client_p99_ms', 'TTFT p99 (ms)'),
            (axes[2], 'rt_tbt_receipt_p99_ms', 'TBT p99 (ms)')]:
        axis.plot(mid, [np.nan if r[column] is None else r[column] for r in timeline],
                  color=color, linewidth=1.5)
        axis.set_ylabel(label)
    axes[0].set_ylabel('BE completed\nrequests/s')
    for axis in axes:
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.25)
    if slo_lines:
        axes[1].axhline(meta['canonical_summary']['rt_ttft_slo_ms'], ls='--', color='#777', lw=1)
        axes[2].axhline(meta['canonical_summary']['rt_tbt_slo_ms'], ls='--', color='#777', lw=1)
    axes[0].legend(loc='upper right')
    axes[-1].set_xlabel('Elapsed time (s)')
    axes[-1].set_xlim(meta['plot_start_s'], meta['plot_start_s'] + meta['plot_duration_s'])
    fig.suptitle(f"Saturo: measured {meta['plot_duration_s']:g} s timeline ({meta['bin_s']:g} s bins)")
    clock_note = ('All panels: diagnostic receipt clock before serialization, including P→D.'
                  if meta['clock_mode'] == 'diagnostic' else
                  'BE/TTFT: canonical CSV. TBT: diagnostic receipt clock, including P→D.')
    coverage_note = ('' if meta['tbt_internal_interval_coverage_complete'] else
                     f" Unobserved co-emitted RT intervals: {meta['counts']['rt_unobserved_internal_intervals']}.")
    if not meta['source_coverage_verified']:
        coverage_note += '\nUNVERIFIED/INCOMPLETE SOURCE: ' + '; '.join(meta['source_completeness_issues'])
    axes[-1].text(0, -.26,
        clock_note + '\n' +
        f"Finite BE backlog: {meta['be_requests']} requests. Empty latency bins remain gaps." + coverage_note,
        transform=axes[-1].transAxes, fontsize=8, va='top')
    out = Path(output_dir)
    fig.savefig(out / 'saturo_timeline.png', dpi=180)
    fig.savefig(out / 'saturo_timeline.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--duration-s', type=float)
    parser.add_argument('--start-s', type=float, default=0)
    parser.add_argument('--bin-s', type=float, default=5)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--clock-mode', choices=('diagnostic', 'mixed'), default='diagnostic',
                        help='diagnostic: one receipt clock for all panels; mixed: CSV BE/TTFT, receipt TBT')
    parser.add_argument('--slo-lines', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    timeline, meta = collect(args.run_dir, args.output_dir, duration_s=args.duration_s,
        start_s=args.start_s, bin_s=args.bin_s, allow_incomplete=args.allow_incomplete,
        clock_mode=args.clock_mode)
    if not args.no_plot:
        plot(timeline, meta, args.output_dir, slo_lines=args.slo_lines)
    print(json.dumps({key: meta[key] for key in ('method', 'source_run',
        'actual_sending_window_s', 'plot_duration_s', 'token_log_coverage_matches_requests',
        'source_coverage_verified', 'source_completeness_issues',
        'counts', 'diagnostic_rt_tbt_all_run_max_ms')}, indent=2))


if __name__ == '__main__':
    main()
