"""Where does the reward gain come from? Oaxaca-style split of the judge-reward change of the top-10.

For the recommended items of a spec and of the base (SFT), every item falls into a stratum:
repeat / new for the user x track-length bucket x train-popularity tercile. With shares w_s and mean
judge Listen+ r_s per stratum:

  delta = sum_s (w_s' - w_s) * r_s          composition: moving to strata that score higher anyway
        + sum_s w_s' * (r_s' - r_s)         within: better items inside the same stratum

The same split is also done along each axis alone (repeat, length, popularity).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rlrec  # noqa: E402,F401
import numpy as np  # noqa: E402

from rlrec.config import RUNS  # noqa: E402
from rlrec.data import Seq  # noqa: E402
from rlrec.metrics import history  # noqa: E402

LEN_EDGES = [150, 250]


def item_strata(seq, uds: np.ndarray, top: np.ndarray) -> dict[str, np.ndarray]:
    rep = np.zeros(top.shape, np.int64)
    for r, i in enumerate(uds):
        rep[r] = np.isin(top[r], history(seq, i))
    tlen = np.searchsorted(LEN_EDGES, seq.items["track_len"][top])
    pop = seq.items["log_pop"]
    terc = np.quantile(pop[2:], [1 / 3, 2 / 3])
    popb = np.searchsorted(terc, pop[top])
    return {"repeat": rep, "length": tlen, "popularity": popb, "all": rep * 9 + tlen * 3 + popb}


def split(s0: np.ndarray, r0: np.ndarray, s1: np.ndarray, r1: np.ndarray) -> dict:
    keys = np.union1d(np.unique(s0), np.unique(s1))
    comp = within = 0.0
    for k in keys:
        m0, m1 = s0 == k, s1 == k
        w0, w1 = m0.mean(), m1.mean()
        rb0 = r0[m0].mean() if m0.any() else (r1[m1].mean() if m1.any() else 0.0)
        rb1 = r1[m1].mean() if m1.any() else rb0
        comp += (w1 - w0) * rb0
        within += w1 * (rb1 - rb0)
    return {"delta": float(r1.mean() - r0.mean()), "composition": float(comp), "within": float(within)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--stage", default="val")
    ap.add_argument("--base", default="sft")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    seq = Seq()
    d = RUNS / "eval" / a.stage
    base = np.load(d / f"{a.base.replace(':', '__')}.npz")
    uds = base["uds"]
    t0 = base["topk"][:, :a.k]
    st0 = item_strata(seq, uds, t0)
    r0 = base["lp_judge_items"].astype(np.float64)[:, :a.k].ravel()
    out = {}
    for spec in a.specs:
        z = np.load(d / f"{spec.replace(':', '__')}.npz")
        assert (z["uds"] == uds).all(), spec
        t1 = z["topk"][:, :a.k]
        st1 = item_strata(seq, uds, t1)
        r1 = z["lp_judge_items"].astype(np.float64)[:, :a.k].ravel()
        out[spec] = {ax: split(st0[ax].ravel(), r0, st1[ax].ravel(), r1) for ax in st0}
        out[spec]["shares"] = {"repeat": [float(st0["repeat"].mean()), float(st1["repeat"].mean())],
                               "short_le150": [float((st0["length"] == 0).mean()), float((st1["length"] == 0).mean())],
                               "top_pop_tercile": [float((st0["popularity"] == 2).mean()),
                                                   float((st1["popularity"] == 2).mean())]}
        a_ = out[spec]["all"]
        print(f"{spec}: delta {a_['delta']:+.4f} = composition {a_['composition']:+.4f} + within {a_['within']:+.4f}; "
              + "; ".join(f"{ax} comp {out[spec][ax]['composition']:+.4f}" for ax in ("repeat", "length", "popularity")),
              flush=True)
    (d / "decompose.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
