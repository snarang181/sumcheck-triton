"""Drop-in for benchmarks/field_sweep/triton_sweep.py with wide-load table reads (vecload.py).

Same arguments and CSV columns; the backend column gets a "-vecload" suffix. Defaults
TRITON_CACHE_DIR to .triton-cache-h100-vecload so the variant's kernels stay separate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".triton-cache-h100-vecload"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vecload  # noqa: E402

vecload.install()

if __name__ == "__main__":
    raise SystemExit(vecload.ts.main())
