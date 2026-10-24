"""Snapshot-trained RM members (judges m4+): trained the way the RM is used in RL and evaluation.

The window-trained members (train_rm.py) see the same day's earlier outcomes; at a day-start snapshot they
are overconfident (val: predicted 0.96 -> observed 0.91, predicted 0.06 -> 0.16). Here every example is a
user-day of the training period: context = the last 256 listens before the day start, candidates = that
day's algorithmic listens, side features counted at the day start (HistoryCounts), labels = their outcomes
(horizons closed before the cutoff, as in train_rm.py).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rlrec  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rlrec.config import CKPT, DEVICE, N_SPECIAL, SEED, TRAIN_END, day_start  # noqa: E402
from rlrec.data import Seq, adopt_label, rm_train_mask  # noqa: E402
from rlrec.model import limit_gpu, make_batch, save  # noqa: E402
from rlrec.rm import HEADS, HistoryCounts, RewardModel, loss_fn, side_features  # noqa: E402


def train_userdays(seq, users: np.ndarray, first_day: int) -> list[tuple[int, int, int]]:
    """(user_start, day_start_index, day_end_index) for user-days of [first_day, TRAIN_END)."""
    day = np.asarray(seq.day)
    out = []
    for u in users:
        s, e = int(seq.ptr[u]), int(seq.ptr[u + 1])
        d = day[s:e]
        lo = s + int(np.searchsorted(d, first_day))
        hi = s + int(np.searchsorted(d, TRAIN_END))
        if hi <= lo:
            continue
        dd = day[lo:hi]
        cuts = lo + np.flatnonzero(np.r_[True, dd[1:] != dd[:-1]])
        ends = np.r_[cuts[1:], hi]
        out += [(s, int(a), int(b)) for a, b in zip(cuts, ends) if a - s >= 5]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", type=int, required=True)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=48, help="user-days per batch")
    ap.add_argument("--cands", type=int, default=32, help="max candidates per user-day")
    ap.add_argument("--first-day", type=int, default=150, help="training user-days from this day on")
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    seed = SEED + 1000 * (a.member + 1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(DEVICE)
    limit_gpu(dev)
    seq = Seq()
    ok = rm_train_mask(seq)
    org, tok, pr = np.asarray(seq.org), np.asarray(seq.tok), np.asarray(seq.pr)
    like, cont = np.asarray(seq.like), np.asarray(seq.cont)
    n_users = len(seq.ptr) - 1
    keep = np.sort(rng.choice(n_users, size=int(a.frac * n_users), replace=False))
    uds = train_userdays(seq, keep, a.first_day)
    hc = HistoryCounts(seq)
    rm = RewardModel(seq.n_tok).to(dev)
    ck = torch.load(CKPT / "sft.pt", map_location=dev, weights_only=False)
    rm.body.load_state_dict({k: v for k, v in ck["state"].items() if k != "copy_w"})
    opt = torch.optim.AdamW(rm.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    print(f"member {a.member} (snapshot): users {len(keep)}, user-days {len(uds)}, steps {a.steps}", flush=True)
    cutoff = day_start(TRAIN_END)
    t0, run = time.time(), {}
    for step in range(1, a.steps + 1):
        pick = rng.choice(len(uds), a.bs, replace=False)
        ctx, cand_gi = [], []
        for i in pick:
            s, lo, hi = uds[i]
            j = lo + np.flatnonzero(ok[lo:hi])
            if len(j) > a.cands:
                j = np.sort(rng.choice(j, a.cands, replace=False))
            ctx.append((s, lo))
            cand_gi.append(j)
        C = max(1, max(len(j) for j in cand_gi))
        valid = np.zeros((a.bs, C), bool)
        gi = np.zeros((a.bs, C), np.int64)
        for r, j in enumerate(cand_gi):
            valid[r, :len(j)] = True
            gi[r, :len(j)] = j
        if not valid.any():
            continue
        cand = np.where(valid, tok[gi], N_SPECIAL)
        side = side_features(seq.items, cand, hc(ctx, cand))
        g = gi[valid]
        p = pr[g].astype(np.float32)
        y = {"ratio": np.clip(p, 0, 100) / 100, "lp": p >= 50, "skip": p < 10, "like": like[g], "cont": cont[g],
             "adopt": adopt_label(seq, g, cutoff)}
        t = lambda x: torch.from_numpy(np.asarray(x)).to(dev)  # noqa: E731
        b = make_batch(seq, ctx, 256, dev, drop_outcome=0.1, rng=rng)
        vt = t(valid)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = rm.hidden(b)[:, -1]
                logits = rm.head(h[:, None, :].expand(-1, C, -1), t(cand), t(side))[vt]
            loss, parts = loss_fn(logits, {k: t(v) for k, v in y.items()})
            opt.zero_grad(set_to_none=True)
            loss.backward()
        except torch.AcceleratorError:
            print(f"step {step}: CUDA error, exiting for a restart", flush=True)
            sys.exit(3)
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            run[k] = run.get(k, 0.0) + v
        if step % 250 == 0:
            print(f"step {step}/{a.steps} " + " ".join(f"{k} {v / 250:.4f}" for k, v in run.items())
                  + f" {time.time() - t0:.0f}s", flush=True)
            run = {}
    save(rm, CKPT / f"rm_m{a.member}.pt", member=a.member, users=keep, heads=HEADS, mode="snapshot")
    (CKPT / f"rm_m{a.member}_meta.json").write_text(json.dumps({"steps": a.steps, "sec": round(time.time() - t0),
                                                                "users": int(len(keep)), "mode": "snapshot"}))


if __name__ == "__main__":
    main()
