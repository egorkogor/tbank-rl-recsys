"""Full evaluation of recommenders on every val (or OOT) user-day under one protocol.

Specs:
  sft                      the SFT policy (with the repeat channel)
  ckpt:<file>              any policy checkpoint in CKPT (e.g. the ablation without the repeat channel)
  copyscale:<f>            SFT with the repeat-channel weights times f (no-RL baseline tuned on val)
  rl:<run>:<step>          checkpoint of an RL run (artifacts/runs/rl/<run>/step<step>.pt)
  pistar:<reward>:<beta>   closed-form optimum of the T = 1 objective on the SFT top-1000:
                           score = log pi_ref + r_train / beta (beta = 0: reward only)
  pistarlen:<reward>:<beta>:<lam>
                           the same with a penalty for leaving the user's length profile (post-OOT, 25.10):
                           score - lam * |log L - log L_u|, L_u from the tracks heard before the day
  mostpop                 most listened tracks of the 7 days before the day
  personal                 the user's most Listen+'d tracks of the 30 days before the day, then mostpop

Per spec: top-100, relevance per user-day, behaviour, and the mean RM reward of the top-10 for every
reward kind under the training RM (member 0) and the judges (members 1-3).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rlrec  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rlrec.config import ART, CKPT, DAY, DEVICE, N_SPECIAL, OOT_DAYS, RUNS, TRAIN_END, VAL_DAYS  # noqa: E402
from rlrec.data import Seq  # noqa: E402
from rlrec.evaluate import contexts, recommend, reward_of_lists  # noqa: E402
from rlrec.metrics import behaviour, relevance, user_bootstrap  # noqa: E402
from rlrec.model import limit_gpu, load_policy, make_batch  # noqa: E402
from rlrec.rl import REWARDS, Scorer  # noqa: E402
from rlrec.rm import load_rm  # noqa: E402
from rlrec.protocol import admit_oot  # noqa: E402


def user_loglen(seq, ctx) -> np.ndarray:
    """Mean log length of the distinct tracks heard before the day: the personal length profile of
    metrics.behaviour (len_dev_user_log); NaN without in-vocabulary history."""
    loglen = np.log(np.clip(seq.items["track_len"], 10, 3600))
    out = np.full(len(ctx), np.nan, np.float32)
    for r, (s, e) in enumerate(ctx):
        h = np.unique(np.asarray(seq.tok[s:e]))
        h = h[h >= N_SPECIAL]
        if len(h):
            out[r] = loglen[h].mean()
    return out


@torch.no_grad()
def pistar_topk(ref, rm, scorer, seq, uds, kind, beta, dev, n_cand=1000, k=100, bs=64, lam=0.0):
    ctx = contexts(seq, uds)
    out = np.zeros((len(uds), k), np.int64)
    loglen = torch.log(scorer.tlen.clamp(10, 3600))
    for i in range(0, len(uds), bs):
        c = ctx[i:i + bs]
        b = make_batch(seq, c, 256, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logp = F.log_softmax(ref.last_logits(b), -1)
            h = rm.hidden(b)[:, -1]
        lp_top, cand = logp.topk(n_cand, 1)
        side = scorer.side(c, cand.cpu().numpy())
        if kind == "balanced":
            scorer.set_week(c)
        r = scorer.reward(scorer.probs(rm, h, cand, side), cand, kind, side).float()
        score = r + 1e-6 * lp_top if beta == 0 else lp_top + r / beta
        if lam:
            lu = torch.from_numpy(user_loglen(seq, c)).to(dev)
            score = score - lam * torch.nan_to_num((loglen[cand] - lu[:, None]).abs())
        order = score.topk(k, 1).indices
        out[i:i + bs] = cand.gather(1, order).cpu().numpy()
    return out


def mostpop_topk(seq, uds, k=100, personal=False):
    day, tok, pr = np.asarray(seq.day), np.asarray(seq.tok), np.asarray(seq.pr)
    t_ok = tok >= N_SPECIAL
    pop = {}
    for d in np.unique(seq.ud["day"][uds]):
        m = (day >= d - 7) & (day < d) & t_ok
        pop[d] = np.argsort(-np.bincount(tok[m], minlength=seq.n_tok), kind="stable")[:k]
    out = np.zeros((len(uds), k), np.int64)
    for r, i in enumerate(uds):
        d = seq.ud["day"][i]
        if not personal:
            out[r] = pop[d]
            continue
        u, e = seq.ud["u"][i], seq.ud["ctx_end"][i]
        s = seq.ptr[u]
        sl = slice(s, e)
        m = (day[sl] >= d - 30) & (pr[sl] >= 50) & t_ok[sl]
        h = tok[sl][m]
        if len(h):
            ut, ct = np.unique(h, return_counts=True)
            own = ut[np.argsort(-ct, kind="stable")][:k]
        else:
            own = np.zeros(0, np.int64)
        fill = pop[d][~np.isin(pop[d], own)]
        out[r] = np.concatenate([own, fill])[:k]
    return out


def topk_for(spec, seq, uds, dev, ref, rm0, scorer):
    if spec == "sft":
        return recommend(ref, seq, uds, dev)
    if spec in ("mostpop", "personal"):
        return mostpop_topk(seq, uds, personal=spec == "personal")
    kind, *rest = spec.split(":")
    if kind == "copyscale":
        # no-RL baseline for the day-level metric: SFT with its repeat-channel weights multiplied by f
        pol = load_policy(CKPT / "sft.pt", dev)
        with torch.no_grad():
            pol.copy_w.mul_(float(rest[0]))
        return recommend(pol, seq, uds, dev)
    if kind == "ckpt":
        return recommend(load_policy(CKPT / rest[0], dev), seq, uds, dev)
    if kind == "rl":
        run, step = rest
        return recommend(load_policy(RUNS / "rl" / run / f"step{step}.pt", dev), seq, uds, dev)
    if kind == "pistar":
        rw, beta = rest
        return pistar_topk(ref, rm0, scorer, seq, uds, rw, float(beta), dev)
    if kind == "pistarlen":
        rw, beta, lam = rest
        return pistar_topk(ref, rm0, scorer, seq, uds, rw, float(beta), dev, lam=float(lam))
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--stage", default="val", choices=["val", "oot"])
    ap.add_argument("--subset", type=int, default=0, help="evaluate on the first n user-days only (debug)")
    a = ap.parse_args()
    if a.stage == "oot" and not (ART / "freeze.json").exists():
        sys.exit("OOT needs artifacts/freeze.json")
    if a.stage == "oot":
        if a.subset:
            sys.exit("OOT protocol requires the full frozen evaluation set")
        a.specs = admit_oot(ART, RUNS / "eval" / "oot", a.specs)
        if not a.specs:
            print("All requested OOT results already exist; no model was loaded")
            return
    days = VAL_DAYS if a.stage == "val" else OOT_DAYS
    dev = torch.device(DEVICE)
    limit_gpu(dev)
    seq = Seq()
    uds = seq.userdays(days)
    if a.subset:
        uds = uds[:a.subset]
    ref = load_policy(CKPT / "sft.pt", dev).eval()
    # m0 trains the policy; judges: m1-m3 (window-trained, like m0) and, if trained, m4-m6 (snapshot-trained)
    members = [m for m in range(7) if (CKPT / f"rm_m{m}.pt").exists()]
    assert members[:4] == [0, 1, 2, 3], members
    rms = [load_rm(CKPT / f"rm_m{m}.pt", dev).eval() for m in members]
    snap = [i for i, m in enumerate(members) if m >= 4]
    scorer = Scorer(seq, TRAIN_END, dev)
    outdir = RUNS / "eval" / a.stage
    outdir.mkdir(parents=True, exist_ok=True)
    users = seq.ud["u"][uds]
    for spec in a.specs:
        fn = spec.replace(":", "__")
        if a.stage == "oot" and (outdir / f"{fn}.json").exists():
            print(f"{spec}: OOT result exists, not overwritten")
            continue
        topk = topk_for(spec, seq, uds, dev, ref, rms[0], scorer)
        rel = relevance(seq, uds, topk)
        rw = reward_of_lists(rms, scorer, seq, uds, topk[:, :10], dev, kinds=REWARDS)
        beh = behaviour(seq, uds, topk)
        lp_items = rw.pop("lp_items")
        summ = {"spec": spec, "n_userdays": int(len(uds)),
                "relevance": {m: user_bootstrap(users, v, n_boot=200) for m, v in rel.items()},
                "r_train": {k: float(v[0].mean()) for k, v in rw.items()},
                "r_judge": {k: float(v[1:4].mean()) for k, v in rw.items()},
                "r_judge_snap": {k: float(v[snap].mean()) for k, v in rw.items()} if snap else None,
                "members": members,
                "ensemble_std_lp": float(lp_items[:4].std(0).mean()),
                "behaviour": beh}
        np.savez_compressed(outdir / f"{fn}.npz", uds=uds, topk=topk.astype(np.int32),
                            **{f"rel_{k}": v for k, v in rel.items()},
                            **{f"rw_{k}": v for k, v in rw.items()}, lp_std=lp_items[:4].std(0).mean(1),
                            lp_train_items=lp_items[0].astype(np.float16),
                            lp_judge_items=lp_items[1:4].mean(0).astype(np.float16),
                            lp_snap_items=(lp_items[snap].mean(0) if snap else lp_items[1:4].mean(0)).astype(np.float16))
        (outdir / f"{fn}.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
        print(spec, json.dumps({"ndcg@10": summ["relevance"]["ndcg@10"]["mean"],
                                "recall@100": summ["relevance"]["recall@100"]["mean"],
                                "r_judge_lp": summ["r_judge"]["lp"], "r_train_lp": summ["r_train"]["lp"],
                                "top_item_share": beh["top_item_share"], "track_len": beh["track_len_mean"],
                                "repeat": beh["repeat_share"]}), flush=True)


if __name__ == "__main__":
    main()
