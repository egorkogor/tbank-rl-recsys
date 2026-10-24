"""Build artifacts/seq/ from events.parquet (see rlrec.data)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlrec.data import build  # noqa: E402

t0 = time.time()
build()
print(f"done in {time.time() - t0:.0f} s")
