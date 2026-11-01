"""Train content-aware QRF from tokenized history, or explicit legacy length-only QRF."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', required=True, help='CSV file or redis-history:<key>')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='')
    p.add_argument('--window-s', type=float, default=1800)
    p.add_argument('--trees', type=int, default=32)
    p.add_argument('--min-requests', type=int, default=32)
    p.add_argument('--feature-mode', choices=('tfidf', 'len'), default='tfidf')
    p.add_argument('--observed-outputs', help='Completed requests.csv; join actual generated IDs to input prompts')
    p.add_argument('--max-steps', type=int, default=8)
    p.add_argument('--max-features', type=int, default=2048)
    p.add_argument('--min-leaf', type=int, default=8)
    p.add_argument('--validation-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    if args.output.exists():
        p.error('Refuse to overwrite existing QRF: ' + str(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.feature_mode == 'tfidf':
        from saturo.predictors.content_training import train_content_qrf
        _, report = train_content_qrf(
            args.history, args.output, observed_outputs=args.observed_outputs,
            max_steps=args.max_steps, max_features=args.max_features,
            n_estimators=args.trees, min_samples_leaf=args.min_leaf,
            validation_fraction=args.validation_fraction, seed=args.seed, n_jobs=args.jobs)
        import json
        print(json.dumps(report, indent=2))
    else:
        if args.observed_outputs:
            p.error('--observed-outputs is for content training')
        from saturo.predictors.history_training import train_history_qrf
        train_history_qrf(args.history, args.output, window_s=args.window_s,
                          model=args.model, n_estimators=args.trees, min_requests=args.min_requests)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
