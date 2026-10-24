"""Five steps ahead: how recommendations and reward estimates behave as the context drifts from reality.

For each policy and val user-day (subset), step k = 1..T, three context modes:
  open     generated tracks appended with outcome UNKNOWN (the policy's own rollout, greedy top-1)
  imputed  generated tracks appended with the outcome the training RM predicts for them
  oracle   the real first k-1 listens of the day with their real outcomes (teacher forcing)
  open_ban as open, but tracks generated earlier in the chain cannot be recommended again
Per step: NDCG@10 and hit@10 against the day's Listen+ set, train / judge reward and ensemble spread of the
top-1 track, repeats inside the rollout and same-artist chains.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rlrec  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rlrec.config import CKPT, DEVICE, LISTEN_PLUS, PARTIAL, RUNS, SKIP, TRAIN_END, VAL_DAYS  # noqa: E402
from rlrec.data import Seq  # noqa: E402
from rlrec.evaluate import contexts  # noqa: E402
from rlrec.model import limit_gpu, append, load_policy, make_batch  # noqa: E402
from rlrec.rl import Scorer  # noqa: E402
from rlrec.rm import load_rm  # noqa: E402


def ndcg10(top: np.ndarray, tg: np.ndarray, n_t: int) -> float:
    w = 1 / np.log2(np.arange(2, 12))
    return float((np.isin(top, tg) * w).sum() / w[:min(n_t, 10)].sum()) if n_t else np.nan


@torch.no_grad()
def run(policy, rms, scorer, seq, uds, mode, T, dev, bs=128):
    ctx = contexts(seq, uds)
    art = seq.items["artist"]
    res = {k: np.full((len(uds), T), np.nan) for k in ("ndcg10", "hit10", "r_train", "r_judge", "lp_std",
                                                         "repeat", "same_artist")}
    for i in range(0, len(uds), bs):
        c = ctx[i:i + bs]
        ids = uds[i:i + bs]
        b = make_batch(seq, c, 256, dev)
        prev = []
        cc = c
        for t in range(T):
            if mode == "oracle" and t > 0:
                # real context: history + the day's first t listens
                cc = [(s, e + t) for s, e in c]
                b = make_batch(seq, cc, 256 + t, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = policy.last_logits(b)
            if mode == "open_ban" and prev:
                # a generated track is not a listen: forbid re-recommending tracks generated earlier in the chain
                z = z.scatter(1, torch.from_numpy(np.stack(prev, 1)).to(z.device), float("-inf"))
            top = z.topk(10, 1).indices
            a = top[:, 0]
            top_np = top.cpu().numpy()
            a_np = a.cpu().numpy()
            # reward of the chosen track under the current context, every RM member
            side = scorer.side(cc, a_np[:, None])
            ps = []
            for rm in rms:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = rm.hidden(b)[:, -1]
                ps.append(scorer.probs(rm, h, a[:, None], side)[:, 0].float())
            lp = torch.stack([p[:, 1] for p in ps])
            res["r_train"][i:i + bs, t] = lp[0].cpu().numpy()
            res["r_judge"][i:i + bs, t] = lp[1:].mean(0).cpu().numpy()
            res["lp_std"][i:i + bs, t] = lp.std(0).cpu().numpy()
            for r, ud in enumerate(ids):
                tg = seq.targets(ud)
                res["ndcg10"][i + r, t] = ndcg10(top_np[r], tg, int(seq.ud["n_tgt"][ud]))
                res["hit10"][i + r, t] = float(np.isin(top_np[r], tg).any())
            res["repeat"][i:i + bs, t] = [float(x in p) for x, p in zip(a_np, zip(*prev))] if prev else 0.0
            res["same_artist"][i:i + bs, t] = (art[a_np] == art[prev[-1]]).astype(float) if prev else 0.0
            prev.append(a_np)
            if mode in ("open", "open_ban"):
                b = append(b, a)
            elif mode == "imputed":
                p0 = ps[0]
                oc = torch.where(p0[:, 1] >= 0.5, LISTEN_PLUS, torch.where(p0[:, 2] >= 0.5, SKIP, PARTIAL))
                b = append(b, a, oc.long())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="sft or rl:<run>:<step>")
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--modes", default="open,imputed,oracle", help="also: open_ban (no repeats inside the chain)")
    a = ap.parse_args()
    dev = torch.device(DEVICE)
    limit_gpu(dev)
    seq = Seq()
    uds = seq.userdays(VAL_DAYS)
    uds = np.sort(np.random.default_rng(0).choice(uds, size=min(a.n, len(uds)), replace=False))
    # oracle needs T-1 real listens of the day after the context
    nxt = seq.ud["ctx_end"][uds] + a.T
    uidx = seq.user_index()
    day = np.asarray(seq.day)
    ok = (uidx[np.minimum(nxt, len(uidx) - 1)] == seq.ud["u"][uds]) & (day[np.minimum(nxt, len(day) - 1)] == seq.ud["day"][uds])
    uds = uds[ok]
    rms = [load_rm(CKPT / f"rm_m{m}.pt", dev).eval() for m in range(4)]
    scorer = Scorer(seq, TRAIN_END, dev)
    out = {}
    for spec in a.specs:
        if spec == "sft":
            pol = load_policy(CKPT / "sft.pt", dev).eval()
        else:
            _, r, s = spec.split(":")
            pol = load_policy(RUNS / "rl" / r / f"step{s}.pt", dev).eval()
        out[spec] = {}
        for mode in a.modes.split(","):
            res = run(pol, rms, scorer, seq, uds, mode, a.T, dev)
            out[spec][mode] = {k: np.nanmean(v, 0).round(5).tolist() for k, v in res.items()}
            print(spec, mode, json.dumps(out[spec][mode]), flush=True)
    path = RUNS / "eval" / "val" / ("multistep.json" if a.modes == "open,imputed,oracle" else "multistep_extra.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps({"n_userdays": int(len(uds)), **old}, indent=1))


if __name__ == "__main__":
    main()
