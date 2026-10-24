"""Rewards from RM heads, exact KL over the vocabulary, and one policy-gradient step for T = 1 and T > 1.

Policy gradient: REINFORCE with a group baseline (G samples per state, leave-one-out advantage as in
RLOO, no value network), plus beta * exact KL(pi || pi_ref).
One gradient step per batch of fresh samples (strictly on-policy, so no PPO clipping).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .model import append, make_batch
from .rm import HistoryCounts, side_features

LEN_EDGES = np.array([90, 150, 200, 250, 330, 480])
REWARDS = ("ratio", "lp", "fair", "cont", "sec", "mix", "disc", "placebo", "minutes", "skip", "wtg", "add", "spotify",
           "balanced", "mlp", "quant")
# Balanced minutes (owner's metric, 24.10, revised): B = T * K(s), T = old + new minutes of the week,
# s = new / T, new = tracks the user had not heard (any artist). BC(s) = sqrt(s s*) + sqrt((1-s)(1-s*));
# K = BC^2 below the target share s*, K = 1 - KAPPA (1 - BC^2) above it (too much new costs 4x less than
# too little). B <= T with equality at s = s*; B reads in minutes and grows without bound.
BAL_EPS = 1.0  # minutes added to both parts of the user's week (finite marginal value at zero)
KAPPA = 0.25
WEEK = 7 * 86_400
SPOTIFY_ALPHA = 0.1


def base_lp_by_length(seq, train_end: int) -> np.ndarray:
    """Per token: Listen+ rate of algorithmic train listens in the token's track-length bucket."""
    org, day, pr, tok = np.asarray(seq.org), np.asarray(seq.day), np.asarray(seq.pr), np.asarray(seq.tok)
    tr = (org == 0) & (day < train_end) & (tok >= 2)
    tlen = seq.items["track_len"]
    bk = np.searchsorted(LEN_EDGES, tlen[tok[tr]])
    rate = np.bincount(bk, weights=(pr[tr] >= 50), minlength=len(LEN_EDGES) + 1) / np.maximum(
        1, np.bincount(bk, minlength=len(LEN_EDGES) + 1))
    return rate[np.searchsorted(LEN_EDGES, tlen)].astype(np.float32)


def event_category(rep_item: np.ndarray, rep_artist: np.ndarray) -> np.ndarray:
    """0 = old (the user heard the track before), 1 = new track of a known artist, 2 = new artist."""
    return np.where(rep_item > 0, 0, np.where(rep_artist > 0, 1, 2)).astype(np.int64)


def balance_coef(s: torch.Tensor, s_star: float) -> torch.Tensor:
    bc2 = ((s * s_star).clamp(min=0).sqrt() + ((1 - s) * (1 - s_star)).clamp(min=0).sqrt()) ** 2
    return torch.where(s <= s_star, bc2, 1 - KAPPA * (1 - bc2))


def balanced(M: torch.Tensor, s_star: float) -> torch.Tensor:
    """B for M [..., 3] minutes (old, new track of a known artist, new artist): new = the last two."""
    old, new = M[..., 0].clamp(min=0), M[..., 1:].sum(-1).clamp(min=0)
    T = old + new
    return T * balance_coef(new / T.clamp(min=1e-9), s_star)


def label_stats(seq, train_end: int, n_buckets: int = 20) -> dict:
    """Train-period statistics of logged algorithmic listens for the literature rewards.

    wtg (Zheng et al., MM'22, eq. 3): mean / sd of listened seconds per track-length quantile bucket;
    add (FeedRec-style linear utility): mean / sd of listened minutes, like and skip labels;
    spotify (Hansen et al., WSDM'21, eq. 13): popularity percentile of every track.
    """
    org, day, pr, tok, like = (np.asarray(seq.org), np.asarray(seq.day), np.asarray(seq.pr), np.asarray(seq.tok),
                               np.asarray(seq.like))
    tr = (org == 0) & (day < train_end) & (tok >= 2)
    tlen = seq.items["track_len"]
    L = tlen[tok[tr]]
    sec = np.clip(pr[tr], 0, 100) / 100 * L
    edges = np.quantile(L, np.linspace(0, 1, n_buckets + 1)[1:-1])
    bk = np.searchsorted(edges, L)
    cnt = np.bincount(bk, minlength=n_buckets)
    mu = np.bincount(bk, weights=sec, minlength=n_buckets) / np.maximum(cnt, 1)
    sd = np.sqrt(np.bincount(bk, weights=sec ** 2, minlength=n_buckets) / np.maximum(cnt, 1) - mu ** 2)
    tb = np.searchsorted(edges, tlen)
    # quantile grid per length bucket of the TRACK-level mean listened ratio (>= 5 train listens):
    # single listens pile up at ratio 1.0 (over half are complete), track means are continuous
    ratio_obs = np.clip(pr[tr], 0, 100) / 100
    n_t = np.bincount(tok[tr], minlength=len(tlen))
    m_t = np.bincount(tok[tr], weights=ratio_obs, minlength=len(tlen)) / np.maximum(n_t, 1)
    ok_t = n_t >= 5
    tb_all = np.searchsorted(edges, tlen)
    qgrid = np.stack([np.quantile(m_t[ok_t & (tb_all == b)], np.linspace(0, 1, 101)) if (ok_t & (tb_all == b)).any()
                      else np.linspace(0, 1, 101) for b in range(n_buckets)]).astype(np.float32)
    pop = seq.items["log_pop"]
    # reference mix of minutes over all train listens: old / new track of a known artist / new artist
    trall = day < train_end
    mins = np.clip(pr[trall], 0, 100) / 100 * tlen[tok[trall]] / 60
    cat = event_category(np.asarray(seq.rep_item)[trall], np.asarray(seq.rep_artist)[trall])
    share = np.bincount(cat, weights=mins, minlength=3) / mins.sum()
    return {"qgrid": qgrid, "qbucket": tb.astype(np.int64), "bal_share": share.astype(np.float32), "bal_s_new": float(share[1:].sum()), "wtg_mu": mu[tb].astype(np.float32), "wtg_sd": np.maximum(sd[tb], 1.0).astype(np.float32),
            "min_mu": float((sec / 60).mean()), "min_sd": float((sec / 60).std()),
            "like_mu": float(like[tr].mean()), "like_sd": float(like[tr].std()),
            "skip_mu": float((pr[tr] < 10).mean()), "skip_sd": float((pr[tr] < 10).std()),
            "lp_mu": float((pr[tr] >= 50).mean()), "lp_sd": float((pr[tr] >= 50).std()),
            "pop_pct": (np.argsort(np.argsort(pop)) / len(pop)).astype(np.float32)}


class Scorer:
    """RM probabilities and rewards for candidate tracks at given context ends."""

    def __init__(self, seq, train_end: int, device):
        self.seq = seq
        self.counts = HistoryCounts(seq)
        self.dev = device
        self.tlen = torch.from_numpy(seq.items["track_len"]).to(device)
        self.base = torch.from_numpy(base_lp_by_length(seq, train_end)).to(device)
        st = label_stats(seq, train_end)
        self.st = {k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v) for k, v in st.items()}
        # placebo: a fixed random "reward" per track with the spread of the RM's lp (no signal about users)
        g = torch.Generator().manual_seed(12345)
        self.rand = (0.68 + 0.15 * torch.randn(seq.n_tok, generator=g)).clamp(0, 1).to(device)

    def side(self, user_ctx: list[tuple[int, int]], cands: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(side_features(self.seq.items, cands, self.counts(user_ctx, cands))).to(self.dev)

    @torch.no_grad()
    def probs(self, rm, h: torch.Tensor, cands: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """h [R, d] (or [R, C, d]), cands [R, C] -> probabilities [R, C, n_heads]."""
        if h.dim() == 2:
            h = h[:, None, :].expand(-1, cands.shape[1], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.sigmoid(rm.head(h, cands, side))

    def set_week(self, user_ctx: list[tuple[int, int]]) -> None:
        """The user's minutes per category over the 7 days before each context end -> self.week [R, 3]."""
        seq = self.seq
        t, pr, tok = seq.t, seq.pr, seq.tok
        tlen = seq.items["track_len"]
        out = np.zeros((len(user_ctx), 3), np.float32)
        for r, (s0, e) in enumerate(user_ctx):
            if e <= s0:
                continue
            t_end = int(t[e - 1]) + 1
            lo = s0 + int(np.searchsorted(np.asarray(t[s0:e]), t_end - WEEK))
            if lo >= e:
                continue
            m = np.clip(np.asarray(pr[lo:e]), 0, 100) / 100 * tlen[np.asarray(tok[lo:e])] / 60
            c = event_category(np.asarray(seq.rep_item[lo:e]), np.asarray(seq.rep_artist[lo:e]))
            out[r] = np.bincount(c, weights=m, minlength=3)
        self.week = torch.from_numpy(out).to(self.dev) + BAL_EPS

    def balanced_marginal(self, side: torch.Tensor) -> torch.Tensor:
        """dB/dM of the candidate's part (old if the user heard the track: side feature 5) given self.week,
        by automatic differentiation of B at the user's week."""
        with torch.enable_grad():
            old = self.week[:, 0].clone().requires_grad_(True)
            new = self.week[:, 1:].sum(-1).clone().requires_grad_(True)
            B = balanced(torch.stack([old, new, torch.zeros_like(new)], -1), self.st["bal_s_new"])
            g_old, g_new = torch.autograd.grad(B.sum(), [old, new])
        g_old, g_new = g_old.detach(), g_new.detach()
        rep = side[..., 5] > 0.5
        shape = rep.shape
        go = g_old.view(-1, *([1] * (len(shape) - 1))).expand(shape)
        gn = g_new.view(-1, *([1] * (len(shape) - 1))).expand(shape)
        return torch.where(rep, go, gn)

    def spotify(self, skip, cands, side, first_in_rollout):
        """Hansen et al. (WSDM'21, eq. 13): R = rel - c + alpha * d * rel, rel = 1 - P(skip), c = 0.1.
        d (our adaptation) = [(1 + prior listens of the artist)^-1 + 1 - popularity percentile] / 2, zeroed
        when the artist already appeared earlier in the same rollout."""
        rel = 1 - skip
        n_art = torch.expm1(side[..., 3].float() * 4)  # side feature 3 = log1p(rep_artist) / 4
        d = (1 / (1 + n_art) + 1 - self.st["pop_pct"][cands]) / 2
        if first_in_rollout is not None:
            d = d * first_in_rollout
        return rel - 0.1 + SPOTIFY_ALPHA * d * rel

    def reward(self, p: torch.Tensor, cands: torch.Tensor, kind: str, side: torch.Tensor | None = None) -> torch.Tensor:
        """side (the RM side features) is needed by "disc": discovery only counts for artists new to the user."""
        ratio, lp, skip, like, cont, adopt = p.unbind(-1)
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
        if kind == "minutes":
            # expected listened minutes of the track; over a sequence see session_minutes()
            return ratio * self.tlen[cands] / 60.0
        if kind == "mix":
            return lp + 0.5 * like - 0.5 * skip
        if kind == "skip":
            return skip  # guardrail in evaluation (not a training reward)
        if kind == "wtg":
            # Watch-Time-Gain: predicted listened seconds standardised within the track-length bucket
            return (ratio * self.tlen[cands] - self.st["wtg_mu"][cands]) / self.st["wtg_sd"][cands]
        if kind == "add":
            st = self.st
            return ((ratio * self.tlen[cands] / 60 - st["min_mu"]) / st["min_sd"]
                    + 0.25 * (like - st["like_mu"]) / st["like_sd"] - 0.25 * (skip - st["skip_mu"]) / st["skip_sd"])
        if kind == "spotify":
            return self.spotify(skip, cands, side, None)
        if kind == "quant":
            # D2Q-style (Zhan et al. KDD'22, adapted): percentile of the predicted listened ratio among the
            # track-level mean ratios of tracks of the same length -> in [0, 1], neutral to the track length
            grid = self.st["qgrid"][self.st["qbucket"][cands]]  # [..., 101]
            below = (grid <= ratio[..., None]).float().sum(-1)
            return ((below - 1).clamp(min=0) / 100.0).clamp(0, 1)
        if kind == "mlp":
            # owner's idea: minutes plus a bonus for a completed (Listen+) track, both standardised on the train
            # labels so that neither part needs a hand-picked minutes-per-completion exchange rate
            st = self.st
            return ((ratio * self.tlen[cands] / 60 - st["min_mu"]) / st["min_sd"] + (lp - st["lp_mu"]) / st["lp_sd"])
        if kind == "balanced":
            # expected minutes of the track x marginal value of a minute of its category for this user
            return ratio * self.tlen[cands] / 60.0 * self.balanced_marginal(side)
        if kind == "placebo":
            return self.rand[cands]
        if kind == "disc":
            # long-term proxy: dosage of new artists the user will look up on their own within a week,
            # plus the usual Listen+ so the list stays listenable
            return lp + 2.0 * adopt * side[..., 4]
        raise ValueError(kind)


def session_minutes(p: torch.Tensor, minutes: torch.Tensor) -> torch.Tensor:
    """Per-step expected minutes weighted by the chance the session is still alive at that step:
    r_t = prod_{k<t} P(cont_k) * E[minutes_t]; the sum over t is the expected listened minutes of the sequence."""
    cont = p[..., 4].float()
    alive = torch.cumprod(torch.cat([torch.ones_like(cont[..., :1]), cont[..., :-1]], -1), -1)
    return alive * minutes


def exact_kl(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """KL(pi || pi_ref) per row over the full vocabulary; forbidden tokens (-inf in both) contribute 0."""
    p = logp.exp()
    d = torch.where(p > 0, logp - logp_ref, torch.zeros_like(logp))
    return (p * d).sum(-1)


def _advantage(R: torch.Tensor) -> torch.Tensor:
    """Leave-one-out baseline (RLOO). No std scaling: the gradient stays an unbiased estimate of
    grad E[r], so the stationary point of E[r] - beta * KL is pi* = pi_ref * exp(r / beta) / Z at T = 1
    and RL at a given beta is comparable with the closed form at the same beta."""
    G = R.shape[1]
    return (R - R.mean(1, keepdim=True)) * G / (G - 1)


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
        if kind == "balanced":
            scorer.set_week(ctx)
        r = scorer.reward(scorer.probs(rm, h, a, side), a, kind, side)
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
        logp_ref = F.log_softmax(ref.logits_pos(full, ref.hidden(full), pos.tolist()), -1)
    side = torch.stack([scorer.side(ctx_g, acts[:, t:t + 1].cpu().numpy())[:, 0] for t in range(T)], 1)
    if kind == "balanced":
        scorer.set_week(ctx_g)
    p = scorer.probs(rm, hr, acts, side)
    r = scorer.reward(p, acts, kind, side)  # [R, T]
    if kind == "minutes":
        r = session_minutes(p, r)
    if kind == "spotify":
        art = torch.from_numpy(scorer.seq.items["artist"]).to(acts.device)[acts]  # [R, T]
        first = torch.ones_like(art, dtype=torch.float32)
        for t in range(1, art.shape[1]):
            first[:, t] = (art[:, :t] != art[:, t:t + 1]).all(1).float()
        r = scorer.spotify(p[..., 2].float(), acts, side, first)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logp = F.log_softmax(policy.logits_pos(full, policy.hidden(full), pos.tolist()), -1)  # [R, T, V]
    lp_a = logp.gather(2, acts[:, :, None])[:, :, 0].sum(1)
    R = r.sum(1).view(B, G)
    A = _advantage(R).view(-1)
    pg = -(A * lp_a).mean()
    kl = exact_kl(logp, logp_ref).sum(1).mean()
    return {"loss": pg + beta * kl, "reward": float(r.mean()), "kl": float(kl), "entropy": float("nan")}


def batch_states(seq, ctx: list, idx: np.ndarray, length: int, device) -> tuple[dict, list]:
    c = [ctx[i] for i in idx]
    return make_batch(seq, c, length, device), c
