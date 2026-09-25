"""Drop-in for benchmarks/field_sweep/triton_sweep.py with the int64-offset fix installed.

Same arguments and CSV columns; see overflow_fix.py. Uses the sweep's Triton cache (the
eval kernels are unchanged; only the encode and fold kernels differ).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import overflow_fix  # noqa: E402

overflow_fix.install()

if __name__ == "__main__":
    raise SystemExit(overflow_fix.ts.main())
