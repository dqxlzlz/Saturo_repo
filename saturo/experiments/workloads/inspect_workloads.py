"""Plot CDFs and arrival rates from prepared real CSVs; never generate requests."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from saturo.experiments.runner import repo_path


def read_rows(value):
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    with repo_path(value).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty workload: ' + str(value))
    for row in rows:
        for key in ('prompt_len', 'output_len'):
            if int(row[key]) <= 0:
                raise ValueError('Lengths must be positive: ' + str(value))
    return rows


def inspect(case, render=True):
    cfg = case['config']
    result, arrays = {}, {}
    for kind, key in [('rt', 'rt_workload_csv'), ('be', 'be_workload_csv')]:
        rows = read_rows(cfg[key])
        entry = {'source': str(repo_path(cfg[key])), 'count': len(rows)}
        for col in ('prompt_len', 'output_len'):
            values = sorted(int(r[col]) for r in rows)
            arrays[kind, col] = values
            entry[col] = {'min': values[0], 'max': values[-1],
                          'mean': sum(values) / len(values)}
        if kind == 'rt':
            offsets = [float(r['arrival_offset_s']) for r in rows]
            bins = {}
            origin = min(offsets)
            for offset in offsets:
                i = int(offset - origin)
                bins[i] = bins.get(i, 0) + 1
            entry['arrival_1s_bins'] = [bins.get(i, 0) for i in range(max(bins) + 1)]
            entry['arrival_origin_s'] = min(offsets)
        result[kind] = entry
    out = repo_path(case['directory']); out.mkdir(parents=True, exist_ok=True)
    (out / 'workload_statistics.json').write_text(json.dumps(result, indent=2))
    if render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for (kind, column), values in arrays.items():
            ax = axes[0 if column == 'prompt_len' else 1]
            ax.step(values, [(i + 1) / len(values) for i in range(len(values))],
                    label=kind, where='post')
            ax.set(title=column, xlabel='Tokens', ylabel='CDF')
            ax.legend()
        fig.tight_layout(); fig.savefig(out / 'length_cdf.pdf'); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 2.5))
        rates = result['rt']['arrival_1s_bins']
        ax.step(range(len(rates)), rates, where='post')
        ax.set(xlabel='Time since first arrival (s)', ylabel='RT requests/s')
        fig.tight_layout(); fig.savefig(out / 'arrival_rate.pdf'); plt.close(fig)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    print(json.dumps(inspect(json.loads(repo_path(args.case).read_text()), not args.no_plots), indent=2))


if __name__ == '__main__':
    main()
