"""Collect missing real timing profiles and build an allocation plan, without serving traffic."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='One concrete flat e2e config, e.g. generated case/config.json')
    p.add_argument('--execute', action='store_true', help='May initialize GPUs for missing profiles')
    args = p.parse_args()
    command = [sys.executable, str(ROOT / 'saturo/run_saturo_endtoend_experiment.py'),
               '--config', args.config, '--dry-run-worker-plan']
    if not args.execute:
        command.append('--print-command')
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
