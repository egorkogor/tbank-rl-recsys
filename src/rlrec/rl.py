"""Rewards from RM heads, exact KL over the vocabulary, and one policy-gradient step for T = 1 and T > 1.

Policy gradient: REINFORCE with a group baseline (G samples per state, advantage = R - group mean,
scaled by the batch std; as in GRPO / RLOO, no value network), plus beta * exact KL(pi || pi_ref).
One gradient step per batch of fresh samples (strictly on-policy, so no PPO clipping).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .model import append, make_batch
from .rm import HistoryCounts, side_features

LEN_EDGES = np.array([90, 150, 200, 250, 330, 480])
REWARDS = ("ratio", "lp", "fair", "cont", "sec", "mix")


def base_lp_by_length(seq, train_end: int) -> np.ndarray:
    """Per token: Listen+ rate of algorithmic train listens in the token's track-length bucket."""
    org, day, pr, tok = np.asarray(seq.org), np.asarray(seq.day), np.asarray(seq.pr), np.asarray(seq.tok)
    tr = (org == 0) & (day < train_end) & (tok >= 2)
    tlen = seq.items["track_len"]
    bk = np.searchsorted(LEN_EDGES, tlen[tok[tr]])
    rate = np.bincount(bk, weights=(pr[tr] >= 50), minlength=len(LEN_EDGES) + 1) / np.maximum(
        1, np.bincount(bk, minlength=len(LEN_EDGES) + 1))
    return rate[np.searchsorted(LEN_EDGES, tlen)].astype(np.float32)


class Scorer:
    """RM probabilities and rewards for candidate tracks at given context ends."""

    def __init__(self, seq, train_end: int, device):
        self.seq = seq
        self.counts = HistoryCounts(seq)
        self.dev = device
        self.tlen = torch.from_numpy(seq.items["track_len"]).to(device)
        self.base = torch.from_numpy(base_lp_by_length(seq, train_end)).to(device)

    def side(self, user_ctx: list[tuple[int, int]], cands: np.ndarray) -> torch.Tensor:
        ri, ra = self.counts(user_ctx, cands)
        it = self.seq.items
        return torch.from_numpy(side_features(it["track_len"][cands], it["log_pop"][cands], ri, ra)).to(self.dev)

    @torch.no_grad()
    def probs(self, rm, h: torch.Tensor, cands: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """h [R, d] (or [R, C, d]), cands [R, C] -> probabilities [R, C, n_heads]."""
        if h.dim() == 2:
            h = h[:, None, :].expand(-1, cands.shape[1], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.sigmoid(rm.head(h, cands, side))

    def reward(self, p: torch.Tensor, cands: torch.Tensor, kind: str) -> torch.Tensor:
        ratio, lp, skip, like, cont = p.unbind(-1)
        if kind == "ratio":
            return ratio
        if kind == "lp":
            return lp
        if kind == "fair":
            return lp - self.base[cands]
        if kind == "cont":
            return cont
        if kind == "sec":
            return ratio * self.tlen[cands] / 200.0
        if kind == "mix":
            return lp + 0.5 * like - 0.5 * skip
        raise ValueError(kind)


def exact_kl(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """KL(pi || pi_ref) per row over the full vocabulary; forbidden tokens (-inf in both) contribute 0."""
    p = logp.exp()
    d = torch.where(p > 0, logp - logp_ref, torch.zeros_like(logp))
    return (p * d).sum(-1)


def _advantage(R: torch.Tensor) -> torch.Tensor:
    A = R - R.mean(1, keepdim=True)
    return A / (A.std() + 1e-6)


def step_t1(policy, ref, rm, scorer, b: dict, ctx: list, kind: str, beta: float, G: int) -> dict:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = policy.last_logits(b)
    logp = F.log_softmax(z, -1)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logp_ref = F.log_softmax(ref.last_logits(b), -1)
            h = rm.hidden(b)[:, -1]
        a = torch.multinomial(logp.detach().exp(), G, replacement=True)
        side = scorer.side(ctx, a.cpu().numpy())
        r = scorer.reward(scorer.probs(rm, h, a, side), a, kind)
    A = _advantage(r)
    pg = -(A * logp.gather(1, a)).mean()
    kl = exact_kl(logp, logp_ref).mean()
    loss = pg + beta * kl
    return {"loss": loss, "reward": float(r.mean()), "kl": float(kl), "entropy": float(-(logp.exp() * torch.where(
        logp.isfinite(), logp, torch.zeros_like(logp))).sum(-1).mean())}


def rollout(policy, b: dict, T: int, greedy: bool = False, ban_repeats: bool = False) -> tuple[dict, torch.Tensor]:
    """Generate T tracks, each appended to the context with outcome UNKNOWN. Returns final batch, actions [R, T]."""
    acts = []
    with torch.no_grad():
        for _ in range(T):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = policy.last_logits(b)
            if ban_repeats and acts:
                z = z.scatter(1, torch.stack(acts, 1), float("-inf"))
            a = z.argmax(1) if greedy else torch.multinomial(F.softmax(z, -1), 1)[:, 0]
            acts.append(a)
            b = append(b, a)
    return b, torch.stack(acts, 1)


def step_seq(policy, ref, rm, scorer, b: dict, ctx: list, kind: str, beta: float, G: int, T: int) -> dict:
    B, L = b["tok"].shape
    bg = {k: v.repeat_interleave(G, 0) for k, v in b.items()}
    ctx_g = [c for c in ctx for _ in range(G)]
    full, acts = rollout(policy, bg, T)
    pos = torch.arange(L - 1, L - 1 + T, device=acts.device)  # position predicting action t
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hr = rm.hidden(full)[:, pos]  # RM state after the real context + generated prefix
        logp_ref = F.log_softmax(ref.logits(ref.hidden(full)[:, pos]), -1)
    side = torch.stack([scorer.side(ctx_g, acts[:, t:t + 1].cpu().numpy())[:, 0] for t in range(T)], 1)
    p = scorer.probs(rm, hr, acts, side)
    r = scorer.reward(p, acts, kind)  # [R, T]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logp = F.log_softmax(policy.logits(policy.hidden(full)[:, pos]), -1)  # [R, T, V]
    lp_a = logp.gather(2, acts[:, :, None])[:, :, 0].sum(1)
    R = r.sum(1).view(B, G)
    A = _advantage(R).view(-1)
    pg = -(A * lp_a).mean()
    kl = exact_kl(logp, logp_ref).sum(1).mean()
    return {"loss": pg + beta * kl, "reward": float(r.mean()), "kl": float(kl), "entropy": float("nan")}


def batch_states(seq, ctx: list, idx: np.ndarray, length: int, device) -> tuple[dict, list]:
    c = [ctx[i] for i in idx]
    return make_batch(seq, c, length, device), c
