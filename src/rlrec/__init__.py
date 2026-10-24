"""rlrec: RL fine-tuning of a GPT-2 next-item recommender against a learned reward model (Yambda-50M).

Importing the package caps the process so the machine stays usable next to the task 1 jobs: 3 worker
threads pinned to logical CPUs 0-2, below-normal priority (owner: the whole PC at ~35% CPU, task 1 holds
11-15). Import rlrec before polars / torch.
Override with RLREC_CORES (e.g. "0-15") or disable with RLREC_NO_LIMIT=1.
"""
import os

if not os.environ.get("RLREC_NO_LIMIT"):
    _lo, _hi = map(int, os.environ.get("RLREC_CORES", "0-2").split("-"))
    _n = str(_hi - _lo + 1)
    for _v in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, _n)
    try:
        import psutil

        _p = psutil.Process()
        _p.cpu_affinity(list(range(_lo, _hi + 1)))
        _p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # limits are a courtesy, never a failure
        pass
