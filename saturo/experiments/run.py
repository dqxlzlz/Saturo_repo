#!/usr/bin/env python3
"""Unified paper experiment planner and launcher."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from saturo.experiments.runner import main
if __name__ == "__main__":
    main()
