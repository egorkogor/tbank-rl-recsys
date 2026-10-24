"""Reward model member: heads ratio / lp / skip / like / cont on algorithmic listens before TRAIN_END.

Each member sees a bootstrap of `frac` of the users (seeded). Member 0 is the training reward for RL,
members 1-3 are the judges. The backbone starts from the SFT policy.
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

from rlrec.config import CKPT, DEVICE, N_SPECIAL, SEED  # noqa: E402
from rlrec.data import Seq, rm_train_mask  # noqa: E402
from rlrec.model import limit_gpu, make_batch, save  # noqa: E402
from rlrec.rm import HEADS, RewardModel, loss_fn, side_features  # noqa: E402
from train_sft import train_ends, windows  # noqa: E402


def targets(seq, b: dict, ok: np.ndarray, device):
    """Positions whose next event is a usable algorithmic listen; candidate, side features and labels."""
    gi_next = b["gi"][:, 1:]
    g = gi_next.clamp(min=0).cpu().numpy()
    valid = (b["mask"][:, 1:] & b["mask"][:, :-1]).cpu().numpy() & (gi_next.cpu().numpy() >= 0) & ok[g]
    gv = g[valid]
    tok = np.asarray(seq.tok)[gv]
    side = side_features(seq.items["track_len"][tok], seq.items["log_pop"][tok], np.asarray(seq.rep_item)[gv],
                         np.asarray(seq.rep_artist)[gv])
    pr = np.asarray(seq.pr)[gv].astype(np.float32)
    y = {"ratio": np.clip(pr, 0, 100) / 100, "lp": pr >= 50, "skip": pr < 10, "like": np.asarray(seq.like)[gv],
         "cont": np.asarray(seq.cont)[gv]}
    t = lambda x: torch.from_numpy(np.asarray(x)).to(device)  # noqa: E731
    return t(valid), t(tok), t(side), {k: t(v) for k, v in y.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", type=int, required=True)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default="sft")
    ap.add_argument("--drop-outcome", type=float, default=0.3)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()

    seed = SEED + 1000 * (a.member + 1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(DEVICE)
    limit_gpu(dev)
    seq = Seq()
    ok = rm_train_mask(seq)
    ends = train_ends(seq)
    n_users = len(ends)
    keep = np.sort(rng.choice(n_users, size=int(a.frac * n_users), replace=False))
    drop = np.setdiff1d(np.arange(n_users), keep)
    ends_m = ends.copy()
    ends_m[drop] = seq.ptr[drop]  # dropped users contribute no windows
    rm = RewardModel(seq.n_tok).to(dev)
    if a.init:
        ck = torch.load(CKPT / f"{a.init}.pt", map_location=dev, weights_only=False)
        rm.body.load_state_dict(ck["state"])
    opt = torch.optim.AdamW(rm.parameters(), lr=a.lr, weight_decay=0.01)
    n_win = len(windows(seq, ends_m, a.len, np.random.default_rng(0)))
    total = a.epochs * (n_win // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    print(f"member {a.member}: users {len(keep)}, windows/epoch {n_win}, steps {total}", flush=True)
    step, t0, run = 0, time.time(), {}
    for ep in range(a.epochs):
        rm.train()
        wins = windows(seq, ends_m, a.len, rng)
        for i in range(0, len(wins) - a.bs + 1, a.bs):
            b = make_batch(seq, wins[i:i + a.bs], a.len + 1, dev, drop_outcome=a.drop_outcome, rng=rng)
            valid, cand, side, y = targets(seq, b, ok, dev)
            if valid.sum() == 0:
                continue
            inp = {k: v[:, :-1] for k, v in b.items()}
            for attempt in range(6):
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = rm.hidden(inp)[valid]
                        logits = rm.head(h, cand, side)
                    loss, parts = loss_fn(logits, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    break
                except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
                    # Another process took VRAM: wait and retry the same batch.
                    print(f"step {step}: GPU memory error ({type(e).__name__}), retry {attempt + 1}", flush=True)
                    opt.zero_grad(set_to_none=True)
                    h = logits = loss = None
                    torch.cuda.empty_cache()
                    time.sleep(30)
            else:
                raise RuntimeError("GPU memory errors persist")
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
            every = 20 if a.max_steps else 200
            if step % every == 0:
                print(f"ep {ep} step {step}/{total} " + " ".join(f"{k} {v / every:.4f}" for k, v in run.items())
                      + f" {time.time() - t0:.0f}s", flush=True)
                run = {}
            if a.max_steps and step >= a.max_steps:
                break
    save(rm, CKPT / f"rm_m{a.member}.pt", member=a.member, users=keep, heads=HEADS)
    (CKPT / f"rm_m{a.member}_meta.json").write_text(json.dumps({"steps": step, "sec": round(time.time() - t0),
                                                                "users": int(len(keep))}))


if __name__ == "__main__":
    main()
