"""CPU allocator entry; pass --profiles/--total-gpus/--target-rps to the existing solver CLI."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'saturo/resource_allocation/run_resource_allocator.py'), run_name='__main__')
