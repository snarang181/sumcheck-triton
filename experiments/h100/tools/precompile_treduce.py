"""Compile one tl.reduce-variant eval kernel without running it (see treduce.py, warm_cache.py).

python experiments/h100/tools/precompile_treduce.py WORKLOAD BIT_WIDTH POINT_IDX
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import treduce  # noqa: E402
import warm_cache  # noqa: E402

treduce.install()

if __name__ == "__main__":
    raise SystemExit(warm_cache.main())
