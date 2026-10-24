"""External judge of a different family: task 1's LightGBM B0 scores our recommended top-10.

The B0 model (task 1, artifacts/runs/b0/model_val.txt) was trained on logged algorithmic impressions of
days 273-286 with day-start snapshot features, so it can judge our val lists (days 287-293) at the same
snapshot. Task 1's code and files are only read: features come from tbrec.features.build, the model
from its file; nothing is written there.

Caveat (task 1): the booster saw only tracks the production recommender showed; on tracks it never
showed (tail, cold) it is out of distribution. Use it as a ranking signal across our specs, not as a
calibrated probability.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

TASK1 = Path(os.environ.get("TBANK_TASK1", "D:/PdM_Projects/tbank-recsys-autofeatures"))
os.environ.setdefault("TBREC_CORES", os.environ.get("RLREC_CORES", "0-2"))  # our CPU budget, not task 1's cores
sys.path.insert(0, str(TASK1 / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401  (caps threads before polars)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from tbrec.data import load_events, load_items  # noqa: E402
from tbrec.features import build  # noqa: E402
from tbrec.model import safe_names  # noqa: E402
from tbrec.specs import B0, BASE  # noqa: E402

from rlrec.config import RUNS, SEED  # noqa: E402
from rlrec.data import Seq  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--model", default="b0")
    ap.add_argument("--users", type=int, default=1000, help="val users sampled (all their days)")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    seq = Seq()
    d = RUNS / "eval" / "val"
    first = np.load(d / f"{a.specs[0].replace(':', '__')}.npz")
    uds = first["uds"]
    users = np.unique(seq.ud["u"][uds])
    pick = np.random.default_rng(SEED).choice(users, size=min(a.users, len(users)), replace=False)
    keep = np.isin(seq.ud["u"][uds], pick)
    booster = lgb.Booster(model_file=str(TASK1 / "artifacts" / "runs" / a.model / "model_val.txt"))
    want = booster.feature_name()
    ev = load_events()
    album = ev.filter(pl.col("kind") == 0).group_by("item").agg(pl.col("album").first())
    items_t1 = load_items()
    uid_of = seq.uid  # user row -> Yambda uid
    out = {}
    for spec in a.specs:
        z = np.load(d / f"{spec.replace(':', '__')}.npz")
        assert (z["uds"] == uds).all(), spec
        top = z["topk"][keep, :a.k]
        u_rows = seq.ud["u"][uds[keep]]
        days = seq.ud["day"][uds[keep]]
        rows = pl.DataFrame({
            "row": np.arange(top.size),
            "uid": np.repeat(uid_of[u_rows], a.k).astype(np.uint32),
            "item": seq.items["item_id"][top.ravel()].astype(np.uint32),
            "artist": seq.items["artist"][top.ravel()].astype(np.uint32),
            "day": np.repeat(days, a.k).astype(np.int16),
        }).join(album, on="item", how="left").with_columns(pl.col("album").fill_null(0).cast(pl.UInt32),
                                                            pl.lit(0, pl.Int32).alias("t"), pl.lit(0).alias("y"))
        x = build(rows, ev, BASE + B0, with_rhythm=True, log=lambda s: None)
        x = x.join(items_t1, on="item", how="left").sort("row")
        cols = {safe_names([c])[0]: c for c in x.columns}
        missing = [f for f in want if f not in cols]
        if missing:
            sys.exit(f"features missing for the booster: {missing[:5]}")
        X = x.select([pl.col(cols[f]).cast(pl.Float32) for f in want]).to_numpy()
        s = booster.predict(X, num_iteration=booster.best_iteration or None).reshape(-1, a.k)
        out[spec] = {"booster_mean_top10": float(s.mean()), "per_userday": s.mean(1).round(5).tolist()}
        print(f"{spec}: booster judge {s.mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    meta = {"model": a.model, "users": int(len(pick)), "userdays": int(keep.sum()), "k": a.k,
            "uds": uds[keep].tolist()}
    (d / f"booster_judge_{a.model}.json").write_text(json.dumps({"meta": meta, **out}))


if __name__ == "__main__":
    main()
