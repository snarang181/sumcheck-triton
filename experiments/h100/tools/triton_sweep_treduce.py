"""Drop-in for benchmarks/field_sweep/triton_sweep.py with the tl.reduce block reduction.

Same arguments and CSV columns; the backend column gets a "-treduce" suffix. Defaults
TRITON_CACHE_DIR to .triton-cache-h100-treduce so the variant's kernels stay separate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".triton-cache-h100-treduce"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import treduce  # noqa: E402

treduce.install()

if __name__ == "__main__":
    raise SystemExit(treduce.ts.main())
