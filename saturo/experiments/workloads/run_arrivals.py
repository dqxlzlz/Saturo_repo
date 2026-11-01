#!/usr/bin/env python3
"""Paper study fig7_arrivals; defaults to plan-only, use --execute explicitly."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from saturo.experiments.runner import main
if __name__ == "__main__":
    main(default_study='fig7_arrivals')
