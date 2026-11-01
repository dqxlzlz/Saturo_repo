"""Paper metrics from request records; missing per-request TBT never becomes a percentile proxy."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def number(row, key):
    value = row.get(key)
    if value in ('', None, 'None'):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {key}')
    return value


def flag(row, key):
    value = row.get(key)
    if value in ('', None, 'None'):
        return None
    value = str(value).lower()
    if value not in ('true', 'false', '1', '0'):
        raise ValueError(f'Invalid boolean {key}')
    return value in ('true', '1')


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)] if values else None


def summarize(rows, start, duration, ttft_slo, tbt_slo):
    end = start + duration
    rt = [r for r in rows if r.get('kind') == 'rt' and start <= float(r['arrival_offset_s']) < end]
    be = [r for r in rows if r.get('kind') == 'be' and number(r, 'finished_s') is not None
          and start <= number(r, 'finished_s') < end]
    ttfts, tbts, missing_tbt, missing_ttft = [], [], 0, 0
    satisfied = 0
    for row in rt:
        ttft = number(row, 'ttft_ms')
        if ttft is None and number(row, 'first_token_s') is not None:
            ttft = (number(row, 'first_token_s') - float(row['arrival_offset_s'])) * 1000
        tbt = number(row, 'tbt_ms')
        observed = number(row, 'observed_tokens')
        target = number(row, 'output_len')
        complete = number(row, 'finished_s') is not None
        if target is not None:
            complete = complete and observed is not None and observed >= target
        coverage = flag(row, 'tbt_complete')
        if coverage is False:
            tbt = None
        single_token = coverage is True and observed == 1
        if complete and tbt is None and not single_token:
            missing_tbt += 1
        if complete and ttft is None:
            missing_ttft += 1
        if ttft is not None:
            ttfts.append(ttft)
        if tbt is not None:
            tbts.append(tbt)
        satisfied += bool(complete and ttft is not None and ttft <= ttft_slo
                          and (single_token or (tbt is not None and tbt <= tbt_slo)))
    return {'rt_requests': len(rt), 'be_completed': len(be), 'be_throughput_rps': len(be) / duration,
            'rt_joint_attainment': satisfied / len(rt) if rt and not missing_tbt and not missing_ttft else None,
            'status': 'missing_per_request_tbt' if missing_tbt else 'missing_per_request_ttft' if missing_ttft else 'measured' if rt else 'no_rt',
            'missing_completed_rt_tbt': missing_tbt,
            'missing_completed_rt_ttft': missing_ttft,
            'ttft_p99_ms': quantile(ttfts, .99), 'tbt_p99_ms': quantile(tbts, .99)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', required=True)
    p.add_argument('--duration-s', type=float, required=True)
    p.add_argument('--start-s', type=float, default=0)
    p.add_argument('--ttft-slo-ms', type=float, default=1000)
    p.add_argument('--tbt-slo-ms', type=float, default=150)
    p.add_argument('--tbt-definition', default='', help='Required when CSV has tbt_ms: record e.g. per-request mean or max inter-token latency')
    p.add_argument('--bin-s', type=float, default=10)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    if args.duration_s <= 0 or args.bin_s <= 0:
        p.error('Durations must be positive')
    with Path(args.requests).open(newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    definitions = {row['tbt_definition'] for row in rows if row.get('tbt_definition')}
    if len(definitions) > 1:
        p.error('Mixed TBT definitions are not comparable')
    recorded_definition = next(iter(definitions), '')
    if args.tbt_definition and recorded_definition and args.tbt_definition != recorded_definition:
        p.error('--tbt-definition conflicts with the recorded definition')
    args.tbt_definition = args.tbt_definition or recorded_definition
    if 'tbt_ms' in fields and not args.tbt_definition:
        p.error('--tbt-definition is required for comparability; do not infer TBT semantics')
    result = summarize(rows, args.start_s, args.duration_s, args.ttft_slo_ms, args.tbt_slo_ms)
    result.update({'source': str(Path(args.requests).resolve()), 'duration_s': args.duration_s,
                   'start_s': args.start_s, 'ttft_slo_ms': args.ttft_slo_ms,
                   'tbt_slo_ms': args.tbt_slo_ms, 'tbt_definition': args.tbt_definition or None})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / 'paper_metrics.json').write_text(json.dumps(result, indent=2))
    timeline = []
    for i in range(math.ceil(args.duration_s / args.bin_s)):
        offset = i * args.bin_s
        window = min(args.bin_s, args.duration_s - offset)
        timeline.append(dict(start_s=args.start_s + offset, duration_s=window,
                             **summarize(rows, args.start_s + offset, window,
                                         args.ttft_slo_ms, args.tbt_slo_ms)))
    with (out / 'timeline.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(timeline[0])); writer.writeheader(); writer.writerows(timeline)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
