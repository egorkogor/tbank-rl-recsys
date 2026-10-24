"""How well does RL solve its own T = 1 objective? Compare the RL policy with the closed form.

For J(pi) = E_pi[r] - beta * KL(pi || pi_ref), the optimum is pi* = pi_ref exp(r / beta) / Z and
J(pi*) - J(pi) = beta * KL(pi || pi*) (Rafailov et al., DPO, appendix A.1). Both policies are restricted
and renormalised to the top-1000 of pi_ref for each context (where pi* is computable); r is the training
RM m0. Reported per run: beta * KL(pi_RL || pi*), E[r] under pi_ref / pi_RL / pi*, KL(pi_RL || pi_ref).
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

from rlrec.config import CKPT, DEVICE, RUNS, SEED, TRAIN_END, VAL_DAYS  # noqa: E402
from rlrec.data import Seq  # noqa: E402
from rlrec.evaluate import contexts  # noqa: E402
from rlrec.model import limit_gpu, load_policy, make_batch  # noqa: E402
from rlrec.rl import Scorer  # noqa: E402
from rlrec.rm import load_rm  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="<run>:<step>; reward kind and beta are read from the run name")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-cand", type=int, default=1000)
    a = ap.parse_args()
    dev = torch.device(DEVICE)
    limit_gpu(dev)
    seq = Seq()
    uds = seq.userdays(VAL_DAYS)
    uds = np.sort(np.random.default_rng(SEED).choice(uds, size=min(a.n, len(uds)), replace=False))
    ctx = contexts(seq, uds)
    ref = load_policy(CKPT / "sft.pt", dev).eval()
    rm = load_rm(CKPT / "rm_m0.pt", dev).eval()
    scorer = Scorer(seq, TRAIN_END, dev)
    out = {}
    for spec in a.runs:
        run, step = spec.split(":")
        kind, beta = run.split("_b")[0], float(run.split("_b")[1].split("_")[0])
        pol = load_policy(RUNS / "rl" / run / f"step{step}.pt", dev).eval()
        acc = {k: [] for k in ("gap", "kl_rl_star", "r_ref", "r_rl", "r_star", "kl_rl_ref", "mass_top")}
        for i in range(0, len(uds), 64):
            c = ctx[i:i + 64]
            b = make_batch(seq, c, 256, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lr_full = F.log_softmax(ref.last_logits(b), -1)
                lp_full = F.log_softmax(pol.last_logits(b), -1)
                h = rm.hidden(b)[:, -1]
            lr_top, cand = lr_full.topk(a.n_cand, 1)
            side = scorer.side(c, cand.cpu().numpy())
            if kind == "balanced":
                scorer.set_week(c)
            r = scorer.reward(scorer.probs(rm, h, cand, side), cand, kind, side).float()
            lp_top = lp_full.gather(1, cand)
            lref = F.log_softmax(lr_top, -1)
            lrl = F.log_softmax(lp_top, -1)
            lstar = F.log_softmax(lr_top + r / beta, -1) if beta > 0 else F.log_softmax(r * 1e6, -1)
            prl = lrl.exp()
            kl_rs = (prl * (lrl - lstar)).sum(1)
            acc["kl_rl_star"].append(kl_rs)
            acc["gap"].append(beta * kl_rs)
            acc["r_ref"].append((lref.exp() * r).sum(1))
            acc["r_rl"].append((prl * r).sum(1))
            acc["r_star"].append((lstar.exp() * r).sum(1))
            acc["kl_rl_ref"].append((prl * (lrl - lref)).sum(1))
            acc["mass_top"].append(lp_full.gather(1, cand).exp().sum(1))  # RL mass inside the ref top-1000
        res = {k: float(torch.cat(v).mean()) for k, v in acc.items()}
        res.update(kind=kind, beta=beta, step=int(step))
        out[spec] = res
        print(spec, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
    path = RUNS / "eval" / "val" / "suboptimality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
