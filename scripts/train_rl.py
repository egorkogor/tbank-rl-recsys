"""RL fine-tuning of the SFT policy against RM member 0.

States are day-start snapshots of RL_DAYS (273-286): contexts only, no labels. Every `eval_every`
steps a light check on the fixed 1,000-user val subset: NDCG@10, Recall@100, train-RM reward of the
top-10, KL to the reference, concentration and track length. Checkpoints are kept for the Pareto front.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rlrec  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rlrec.config import CKPT, DEVICE, RL_DAYS, RUNS, SEED, TRAIN_END  # noqa: E402
from rlrec.data import Seq  # noqa: E402
from rlrec.evaluate import contexts, recommend, reward_of_lists  # noqa: E402
from rlrec.metrics import relevance  # noqa: E402
from rlrec.model import limit_gpu, load_policy, make_batch, save  # noqa: E402
from rlrec.rl import REWARDS, Scorer, exact_kl, step_seq, step_t1  # noqa: E402
from rlrec.rm import load_rm  # noqa: E402
from train_sft import val_subset  # noqa: E402


@torch.no_grad()
def light_eval(policy, ref, rm, scorer, seq, vsub, kind, dev) -> dict:
    topk = recommend(policy, seq, vsub, dev)
    rel = relevance(seq, vsub, topk)
    rw = reward_of_lists([rm], scorer, seq, vsub, topk[:, :10], dev, kinds=(kind, "lp"))
    kls = []
    ctx = contexts(seq, vsub)
    for i in range(0, len(vsub), 256):
        b = make_batch(seq, ctx[i:i + 256], 256, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = F.log_softmax(policy.last_logits(b), -1)
            lr = F.log_softmax(ref.last_logits(b), -1)
        kls.append(exact_kl(lp, lr).cpu().numpy())
    top10 = topk[:, :10]
    most = np.bincount(top10.ravel()).argmax()
    return {"ndcg@10": float(np.nanmean(rel["ndcg@10"])), "recall@100": float(np.nanmean(rel["recall@100"])),
            "ndcg@10_new": float(np.nanmean(rel["ndcg@10_new"])), f"r_train_{kind}": float(rw[kind][0].mean()),
            "r_train_lp": float(rw["lp"][0].mean()), "kl": float(np.concatenate(kls).mean()),
            "top_item_share": float((top10 == most).any(1).mean()), "unique_top10": int(len(np.unique(top10))),
            "track_len": float(seq.items["track_len"][top10].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", required=True, choices=REWARDS)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--T", type=int, default=1)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--rm", default="rm_m0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--vsub-users", type=int, default=1000)
    ap.add_argument("--seq-chunk", type=int, default=4, help="states per gradient-accumulation chunk at T > 1")
    a = ap.parse_args()

    name = a.tag or f"{a.reward}_b{a.beta:g}_T{a.T}"
    out = RUNS / "rl" / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = torch.device(DEVICE)
    limit_gpu(dev)
    seq = Seq()
    policy = load_policy(CKPT / "sft.pt", dev)
    ref = load_policy(CKPT / "sft.pt", dev)
    rm = load_rm(CKPT / f"{a.rm}.pt", dev)
    for m in (policy, ref, rm):
        m.eval()  # no dropout: sampling and update see the same distribution
    for p in list(ref.parameters()) + list(rm.parameters()):
        p.requires_grad_(False)
    scorer = Scorer(seq, TRAIN_END, dev)
    states = seq.userdays(RL_DAYS, need_targets=False)
    ctx = contexts(seq, states)
    vsub = val_subset(seq, a.vsub_users, SEED)
    opt = torch.optim.Adam(policy.parameters(), lr=a.lr)
    log = [{"step": 0, **light_eval(policy, ref, rm, scorer, seq, vsub, a.reward, dev)}]
    print(json.dumps(log[-1]), flush=True)
    t0, run = time.time(), []
    for step in range(1, a.steps + 1):
        idx = rng.choice(len(states), a.bs, replace=False)
        c = [ctx[i] for i in idx]
        b = make_batch(seq, c, a.len, dev)
        opt.zero_grad(set_to_none=True)
        if a.T == 1:
            res = step_t1(policy, ref, rm, scorer, b, c, a.reward, a.beta, a.G)
            res["loss"].backward()
        else:
            # T > 1: rollouts of G x T tokens per state do not fit at once -> gradient accumulation over
            # chunks of whole states (each state's G samples stay together, so the RLOO baseline is unchanged)
            parts = []
            chunks = [list(range(i, min(i + a.seq_chunk, a.bs))) for i in range(0, a.bs, a.seq_chunk)]
            for ch in chunks:
                bc = {k: v[ch] for k, v in b.items()}
                r_ = step_seq(policy, ref, rm, scorer, bc, [c[i] for i in ch], a.reward, a.beta, a.G, a.T)
                (r_["loss"] * len(ch) / a.bs).backward()
                parts.append({k: float(v) * len(ch) / a.bs for k, v in r_.items()})
            res = {k: sum(p_[k] for p_ in parts) for k in parts[0]}
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        run.append({k: float(v) for k, v in res.items()})
        if step % 20 == 0:
            m = {k: np.mean([r[k] for r in run]) for k in run[0]}
            print(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in m.items()) + f" {time.time() - t0:.0f}s",
                  flush=True)
            run = []
        if step % a.eval_every == 0 or step == a.steps:
            log.append({"step": step, "sec": round(time.time() - t0),
                        **light_eval(policy, ref, rm, scorer, seq, vsub, a.reward, dev)})
            print(json.dumps(log[-1]), flush=True)
            save(policy, out / f"step{step}.pt", step=step, args=vars(a))
            (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
