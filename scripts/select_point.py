"""The selection rule fixed in docs/metrics_hypotheses_RU.md (section 3), applied to the val evaluation.

Candidates: every RL checkpoint and pi*_beta. Objective (owner, 24.10): largest gain of the judges'
balanced minutes B of the user's week with the top-10 added as a session (snapshot judges m4-m6 if
present, else m1-m3); plain session minutes are reported next to it. Constraints against SFT:
  1. NDCG@10: verdict not "хуже" and delta >= -2 % of SFT;
  2. Recall@100 on tracks new to the user: delta >= -10 % of SFT;
  3. judges' P(skip) of the top-10: at most +0.005 over SFT;
  4. share of user-days whose top-10 holds the most frequent track: at most 2 x SFT;
  5. judges' P(Listen+) of the top-10 (completion, owner 24.10): at most 0.005 below SFT.
If nothing passes, the recommendation is SFT.

--len-guard g (post-OOT, docs/post_oot_diagnostics_RU.md): a sixth constraint, the personal length profile
len_dev_user_log at most (1 + g) x SFT; the length-penalised pi* (pistarlen:) join the candidates and the
result goes to selection_lenguard.json. Without the flag the rule is the frozen one (selection.json).
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
from rlrec.metrics import user_bootstrap, verdict  # noqa: E402
from compare import REL_THRESHOLD, load  # noqa: E402


def judges(z: dict, kind: str) -> np.ndarray:
    rw = z[f"rw_{kind}"]
    return rw[4:].mean(0) if rw.shape[0] > 4 else rw[1:4].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="val")
    ap.add_argument("--len-guard", type=float, default=0.0)
    a = ap.parse_args()
    seq = Seq()
    ev = load(a.stage)
    base = ev["sft"]
    b = base["npz"]
    users = seq.ud["u"][b["uds"]]
    b_ndcg = float(np.nanmean(b["rel_ndcg@10"]))
    b_new = float(np.nanmean(b["rel_recall@100_new"]))
    b_skip = float(judges(b, "skip").mean())
    b_lp = float(judges(b, "lp").mean())
    b_top = base["json"]["behaviour"]["top_item_share"]
    b_len = base["json"]["behaviour"]["len_dev_user_log"]
    prefixes = ("rl:", "pistar:", "pistarlen:") if a.len_guard else ("rl:", "pistar:")
    rows = []
    for spec, d in ev.items():
        if not spec.startswith(prefixes):
            continue
        z, j = d["npz"], d["json"]
        ci_n = user_bootstrap(users, z["rel_ndcg@10"], b["rel_ndcg@10"])
        ci_m = user_bootstrap(users, judges(z, "balanced"), judges(b, "balanced"))
        ci_min = user_bootstrap(users, judges(z, "minutes"), judges(b, "minutes"))
        d_new = float(np.nanmean(z["rel_recall@100_new"])) - b_new
        d_skip = float(judges(z, "skip").mean()) - b_skip
        d_lp = float(judges(z, "lp").mean()) - b_lp
        top = j["behaviour"]["top_item_share"]
        checks = {
            "ndcg": verdict(ci_n, REL_THRESHOLD * b_ndcg) != "хуже" and ci_n["mean"] >= -REL_THRESHOLD * b_ndcg,
            "new_tracks": d_new >= -0.10 * b_new,
            "skip": d_skip <= 0.005,
            "concentration": top <= 2 * b_top,
            "completion": d_lp >= -0.005,
        }
        if a.len_guard:
            checks["length_profile"] = j["behaviour"]["len_dev_user_log"] <= (1 + a.len_guard) * b_len
        rows.append({"spec": spec, "d_balanced": ci_m, "d_minutes": ci_min, "d_ndcg": ci_n, "d_ndcg_rel": ci_n["mean"] / b_ndcg,
                     "d_recall100_new_rel": d_new / b_new if b_new else 0.0, "d_skip": d_skip, "d_lp": d_lp,
                     "top_item_share": top, "len_dev_user_log": j["behaviour"]["len_dev_user_log"], "checks": checks, "passes": all(checks.values())})
    rows.sort(key=lambda r: -r["d_balanced"]["mean"])
    ok = [r for r in rows if r["passes"]]
    choice = ok[0]["spec"] if ok else "sft"
    out = {"choice": choice, "n_candidates": len(rows), "n_pass": len(ok), "rows": rows}
    name = "selection_lenguard.json" if a.len_guard else "selection.json"
    (RUNS / "eval" / a.stage / name).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"candidates {len(rows)}, pass {len(ok)}, choice: {choice}")
    for r in rows[:15]:
        print(f"{r['spec']:28s} d_B {r['d_balanced']['mean']:+.3f} [{r['d_balanced']['lo']:+.3f},{r['d_balanced']['hi']:+.3f}]"
              f" d_min {r['d_minutes']['mean']:+.3f}"
              f" ndcg {r['d_ndcg_rel']:+.1%} new {r['d_recall100_new_rel']:+.1%} skip {r['d_skip']:+.4f}"
              f" top {r['top_item_share']:.3f} {'PASS' if r['passes'] else 'fail ' + ','.join(k for k, v in r['checks'].items() if not v)}")


if __name__ == "__main__":
    main()
