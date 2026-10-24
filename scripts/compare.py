"""Tables and the Pareto plot from artifacts/runs/eval/<stage>/*.npz.

Paired user bootstrap against the base spec for NDCG@10, Recall@100, NDCG@10 on new tracks and the
judge reward of the chosen kind; decision words as in task 1 (thresholds fixed before OOT).
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

# Decision thresholds, fixed 24.10 before any OOT run: relevance relative to the base, reward absolute.
REL_THRESHOLD = 0.02
REWARD_THRESHOLD = 0.005


def load(stage: str) -> dict:
    out = {}
    for f in sorted((RUNS / "eval" / stage).glob("*.npz")):
        if f.stem == "multistep":
            continue
        z = np.load(f)
        out[f.stem.replace("__", ":")] = {"npz": {k: z[k] for k in z.files},
                                          "json": json.loads(f.with_suffix(".json").read_text())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="val")
    ap.add_argument("--base", default="sft")
    ap.add_argument("--kind", default="lp", help="reward kind for the judge column")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    seq = Seq()
    ev = load(a.stage)
    base = ev[a.base]["npz"]
    users = seq.ud["u"][base["uds"]]
    rows = []
    for spec, d in ev.items():
        if a.only and spec not in a.only:
            continue
        z, j = d["npz"], d["json"]
        assert (z["uds"] == base["uds"]).all(), spec
        row = {"spec": spec}
        for m in ("ndcg@10", "recall@100", "ndcg@10_new"):
            ci = user_bootstrap(users, z[f"rel_{m}"], base[f"rel_{m}"])
            b_mean = float(np.nanmean(base[f"rel_{m}"]))
            row[m] = float(np.nanmean(z[f"rel_{m}"]))
            row[f"d_{m}"] = ci
            row[f"v_{m}"] = verdict(ci, REL_THRESHOLD * b_mean) if spec != a.base else ""
        rj, rj0 = z[f"rw_{a.kind}"][1:].mean(0), base[f"rw_{a.kind}"][1:].mean(0)
        ci = user_bootstrap(users, rj, rj0)
        row["r_judge"] = float(rj.mean())
        row["r_train"] = float(z[f"rw_{a.kind}"][0].mean())
        row["d_r_judge"] = ci
        row["v_r_judge"] = verdict(ci, REWARD_THRESHOLD) if spec != a.base else ""
        row["gap_train_judge"] = row["r_train"] - row["r_judge"]
        row["lp_std"] = float(z["lp_std"].mean())
        row.update({k: j["behaviour"][k] for k in ("coverage@10", "top_item_share", "jaccard_pairs",
                                                   "track_len_mean", "repeat_share", "new_artist_share")})
        rows.append(row)
    out = RUNS / "eval" / a.stage / f"compare_{a.kind}.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    hdr = ("spec", "ndcg@10", "v", "recall@100", "ndcg@10_new", f"r_judge[{a.kind}]", "v", "train-judge",
           "top_item", "len", "repeat", "new_art")
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        lines.append("| " + " | ".join([r["spec"], f"{r['ndcg@10']:.4f}", r["v_ndcg@10"], f"{r['recall@100']:.4f}",
                                        f"{r['ndcg@10_new']:.4f}", f"{r['r_judge']:.4f}", r["v_r_judge"],
                                        f"{r['gap_train_judge']:+.4f}", f"{r['top_item_share']:.2f}",
                                        f"{r['track_len_mean']:.0f}", f"{r['repeat_share']:.2f}",
                                        f"{r['new_artist_share']:.2f}"]) + " |")
    md = "\n".join(lines)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        fam = {}
        for r in rows:
            f = r["spec"].split(":")[0] if ":" in r["spec"] else r["spec"]
            key = f if f in ("sft", "mostpop", "personal") else f + ":" + r["spec"].split(":")[1].split("_")[0]
            fam.setdefault(key, []).append(r)
        for key, rs in fam.items():
            rs = sorted(rs, key=lambda r: r["r_judge"])
            ax.plot([r["r_judge"] for r in rs], [r["ndcg@10"] for r in rs], "o-" if len(rs) > 1 else "o", label=key)
        ax.set_xlabel(f"judge reward of top-10 ({a.kind})")
        ax.set_ylabel("NDCG@10")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out.with_name(f"pareto_{a.kind}.png"), dpi=130)
    except Exception as e:  # the plot is a convenience
        print("plot failed:", e)


if __name__ == "__main__":
    main()
