"""SFT: GPT-2 next-track prediction on every listen before TRAIN_END.

Windows of length+1 events are cut from each user's history with a fresh random offset per epoch.
Early stopping on NDCG@10 of a fixed 1,000-user subset of val (day-start snapshots).
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

from rlrec.config import CKPT, DEVICE, N_SPECIAL, SEED, TRAIN_END, VAL_DAYS  # noqa: E402
from rlrec.data import Seq  # noqa: E402
from rlrec.evaluate import recommend  # noqa: E402
from rlrec.metrics import relevance  # noqa: E402
from rlrec.model import limit_gpu, Policy, chunked_ce, make_batch, save  # noqa: E402

MIN_CTX = 32


def train_ends(seq) -> np.ndarray:
    """Per user: global index of the first listen of day >= TRAIN_END."""
    day = np.asarray(seq.day)
    ends = np.empty(len(seq.ptr) - 1, np.int64)
    for u in range(len(ends)):
        s, e = seq.ptr[u], seq.ptr[u + 1]
        ends[u] = s + np.searchsorted(day[s:e], TRAIN_END)
    return ends


def windows(seq, ends, length, rng) -> list[tuple[int, int]]:
    out = []
    for u in range(len(ends)):
        s, e = int(seq.ptr[u]), int(ends[u])
        end = e - int(rng.integers(0, length))
        while end - s >= 2:
            out.append((max(s, end - length - 1), end))
            end -= length
    rng.shuffle(out)
    return out


def val_subset(seq, n_users: int, seed: int) -> np.ndarray:
    uds = seq.userdays(VAL_DAYS)
    users = np.unique(seq.ud["u"][uds])
    pick = np.random.default_rng(seed).choice(users, size=min(n_users, len(users)), replace=False)
    return uds[np.isin(seq.ud["u"][uds], pick)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--drop-outcome", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after n steps")
    ap.add_argument("--out", default="sft")
    ap.add_argument("--resume", action="store_true", help="continue from <out>_last.pt")
    ap.add_argument("--init", default="", help="start from the weights of this checkpoint file name in CKPT")
    ap.add_argument("--copy", action="store_true", help="add the repeat channel (RepeatNet-style copy logits)")
    ap.add_argument("--copy-init", type=float, nargs=2, default=(4.0, 2.0))
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--stop-at", type=int, default=0, help="stop after this step keeping the full-length schedule "
                    "(the run without the repeat channel was stopped at 6000 of 10616)")
    a = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = torch.device(DEVICE)
    limit_gpu(dev)
    seq = Seq()
    ends = train_ends(seq)
    vsub = val_subset(seq, 1000, SEED)
    model = Policy(seq.n_tok, d=a.d, n_layer=a.layers, n_head=4, copy=a.copy).to(dev)
    if a.init:
        st = torch.load(CKPT / a.init, map_location=dev, weights_only=False)["state"]
        model.load_state_dict(st, strict=not (a.copy and "copy_w" not in st))
        print(f"initialised from {a.init}", flush=True)
    if a.copy and not (a.init and "copy_w" in st):
        with torch.no_grad():
            model.copy_w.copy_(torch.tensor(a.copy_init))
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "tok" not in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "tok" not in n) and n != "copy_w"]
    groups = [{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}]
    if a.copy:
        groups.append({"params": [model.copy_w], "weight_decay": 0.0, "lr": 1e-2})
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.98))
    n_win = len(windows(seq, ends, a.len, np.random.default_rng(0)))
    total = a.max_steps or a.epochs * (n_win // a.bs)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    print(f"windows/epoch {n_win}, steps {total}, val subset {len(vsub)} user-days", flush=True)
    log, best, step, t0 = [], -1.0, 0, time.time()
    start_ep, start_i = 0, 0
    last = CKPT / f"{a.out}_last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, start_ep, start_i, log, best = ck["step"], ck["ep"], ck["i"], ck["log"], ck["best"]
        print(f"resumed at step {step} (epoch {start_ep}, batch offset {start_i})", flush=True)
    for ep in range(start_ep, a.epochs):
        model.train()
        # Windows depend only on (seed, epoch), so a resumed run sees the same order.
        wins = windows(seq, ends, a.len, np.random.default_rng(SEED + ep))
        brng = np.random.default_rng(SEED + 100 + ep)
        run, n_run = 0.0, 0
        for i in range(0, len(wins) - a.bs + 1, a.bs):
            if ep == start_ep and i < start_i:
                continue
            b = make_batch(seq, wins[i:i + a.bs], a.len + 1, dev, drop_outcome=a.drop_outcome, rng=brng)
            inp = {k: v[:, :-1] for k, v in b.items()}
            y = b["tok"][:, 1:]
            # Train on positions with at least MIN_CTX events of context (eval contexts are long).
            ctx_len = inp["mask"].long().cumsum(1)
            keep = b["mask"][:, 1:] & inp["mask"] & (y >= N_SPECIAL) & (ctx_len >= MIN_CTX)
            for attempt in range(6):
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = model.hidden(inp)
                        row, col = keep.nonzero(as_tuple=True)
                        loss = chunked_ce(model, h[keep], y[keep], ctx=(inp["tok"], inp["mask"], row, col))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    break
                except torch.AcceleratorError:
                    # A driver-level CUDA error leaves the context unusable: exit, the wrapper resumes
                    # from the last checkpoint in a fresh process.
                    print(f"step {step}: CUDA error, exiting for a restart from the checkpoint", flush=True)
                    sys.exit(3)
                except torch.OutOfMemoryError as e:
                    # Another process (a game, the desktop) took VRAM: wait and retry the same batch.
                    print(f"step {step}: GPU memory error ({type(e).__name__}), retry {attempt + 1}", flush=True)
                    opt.zero_grad(set_to_none=True)
                    h = loss = None
                    torch.cuda.empty_cache()
                    time.sleep(30)
            else:
                raise RuntimeError("GPU memory errors persist")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % a.save_every == 0:
                torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "ep": ep, "i": i + a.bs, "log": log, "best": best}, last)
            run += loss.item()
            n_run += 1
            every = 20 if 0 < a.max_steps < 1000 else 200
            if step % every == 0:
                mem = torch.cuda.max_memory_reserved() / 2**30 if dev.type == "cuda" else 0.0
                cw = " copy_w " + ",".join(f"{x:.2f}" for x in model.copy_w.tolist()) if a.copy else ""
                print(f"ep {ep} step {step}/{total} loss {run / n_run:.4f} {time.time() - t0:.0f}s gpu {mem:.2f}G"
                      + cw, flush=True)
                run, n_run = 0.0, 0
            if (a.max_steps and step >= a.max_steps) or (a.stop_at and step >= a.stop_at):
                break
        if a.stop_at and step >= a.stop_at:
            torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "ep": ep, "i": 0, "log": log, "best": best}, CKPT / f"{a.out}_s{step}.pt")
            print(f"stopped at step {step}: {a.out}_s{step}.pt", flush=True)
            return
        topk = recommend(model, seq, vsub, dev, a.len)
        rel = relevance(seq, vsub, topk)
        res = {"epoch": ep, "step": step, "sec": round(time.time() - t0),
               **{k: float(np.nanmean(v)) for k, v in rel.items() if not k.endswith("_new") or "@10" in k}}
        log.append(res)
        print(json.dumps(res), flush=True)
        if res["ndcg@10"] > best:
            best = res["ndcg@10"]
            save(model, CKPT / f"{a.out}.pt", epoch=ep, val_subset=res)
        if a.max_steps and step >= a.max_steps:
            break
    (CKPT / f"{a.out}_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
