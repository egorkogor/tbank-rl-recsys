"""Reward model: response of a user to a candidate track given the history.

Causal GPT-2 backbone (same input as the policy) -> h_t; the candidate is the track of event t+1.
score = MLP([h_t, e_c, h_t * e_c, side_c]) -> heads. One pass over a window scores every position
whose next event is algorithmic; h_t sees events <= t only.

side_c (known before the candidate plays): log track length, log train popularity, prior listens of
the track and of its artist by this user (whole history), new-artist and repeat flags, and the quality of
that history (task 1: its strongest hand-made features): smoothed Listen+ and skip rates of the
user x track pair and the Listen+ rate of the user x artist pair.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Policy

HEADS = ("ratio", "lp", "skip", "like", "cont", "adopt")
N_SIDE = 9
COUNT_KEYS = ("rep_item", "rep_artist", "rep_lp", "rep_skip", "rep_artist_lp")
PRIOR_LP, PRIOR_SKIP, K_SMOOTH = 0.68, 0.22, 2.0


class RewardModel(nn.Module):
    def __init__(self, n_tok: int, d: int = 256, n_layer: int = 4, n_head: int = 4, n_pos: int = 320,
                 dropout: float = 0.1):
        super().__init__()
        self.cfg = dict(n_tok=n_tok, d=d, n_layer=n_layer, n_head=n_head, n_pos=n_pos, dropout=dropout)
        self.body = Policy(n_tok, d, n_layer, n_head, n_pos, dropout)
        self.side = nn.Linear(N_SIDE, d)
        self.mlp = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d),
                                 nn.GELU(), nn.Linear(d, len(HEADS)))

    def head(self, h: torch.Tensor, cand: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """h [..., d], cand [...] tokens, side [..., N_SIDE] -> logits [..., n_heads]."""
        e = self.body.tok(cand)
        s = self.side(side.to(h.dtype))
        return self.mlp(torch.cat([h, e, h * e, s], -1)).float()

    def hidden(self, b: dict) -> torch.Tensor:
        return self.body.hidden(b)


def side_features(items: dict, tok: np.ndarray, c: dict) -> np.ndarray:
    """items = seq.items, tok = candidate tokens, c = prior counts (COUNT_KEYS) -> [..., N_SIDE]."""
    ri, ra = c["rep_item"].astype(np.float32), c["rep_artist"].astype(np.float32)
    return np.stack([np.log(np.clip(items["track_len"][tok], 10, 3600)) - 5.3, items["log_pop"][tok] / 5 - 1,
                     np.log1p(ri) / 3, np.log1p(ra) / 4, (ra == 0).astype(np.float32), (ri > 0).astype(np.float32),
                     (c["rep_lp"] + K_SMOOTH * PRIOR_LP) / (ri + K_SMOOTH) - PRIOR_LP,
                     (c["rep_skip"] + K_SMOOTH * PRIOR_SKIP) / (ri + K_SMOOTH) - PRIOR_SKIP,
                     (c["rep_artist_lp"] + K_SMOOTH * PRIOR_LP) / (ra + K_SMOOTH) - PRIOR_LP], -1).astype(np.float32)


def logged_counts(seq, gi: np.ndarray) -> dict:
    """Prior counts stored for logged listens (training and evaluation of the RM on the log)."""
    return {k: np.asarray(getattr(seq, k))[gi] for k in COUNT_KEYS}


def loss_fn(logits: torch.Tensor, y: dict) -> tuple[torch.Tensor, dict]:
    """BCE on every head; ratio is a soft label in [0, 1]; adopt is trained only where defined (>= 0)."""
    parts = {}
    for j, name in enumerate(HEADS):
        if name == "adopt":
            m = y[name] >= 0
            parts[name] = (F.binary_cross_entropy_with_logits(logits[m, j], y[name][m].float())
                           if m.any() else logits[:, j].sum() * 0.0)
        else:
            parts[name] = F.binary_cross_entropy_with_logits(logits[:, j], y[name].float())
    w = {"ratio": 1.0, "lp": 1.0, "skip": 1.0, "like": 1.0, "cont": 0.5, "adopt": 0.5}
    return sum(w[k] * v for k, v in parts.items()), {k: float(v.detach()) for k, v in parts.items()}


def _count_of(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(query.shape, np.int64)
    u, c = np.unique(values, return_counts=True)
    pos = np.searchsorted(u, query).clip(0, len(u) - 1)
    return np.where(u[pos] == query, c[pos], 0)


class HistoryCounts:
    """Prior counts (COUNT_KEYS) of arbitrary candidate tracks in a user's history before a context end.

    Same definitions as the logged columns (data.build / data.build_pair_stats): every prior listen of
    the user, out-of-vocabulary tracks included.
    """

    def __init__(self, seq):
        self.seq = seq
        self.art = seq.items["artist"]

    def __call__(self, ud_ctx: list[tuple[int, int]], cands: np.ndarray) -> dict:
        """ud_ctx[r] = (user_start, ctx_end); cands [R, C] tokens -> {key: counts [R, C]}."""
        out = {k: np.zeros(cands.shape, np.int64) for k in COUNT_KEYS}
        tok, artist, pr = self.seq.tok, self.seq.artist, self.seq.pr
        for r, (s, e) in enumerate(ud_ctx):
            if e <= s:
                continue
            ht, ha, hp = np.asarray(tok[s:e]), np.asarray(artist[s:e]), np.asarray(pr[s:e])
            aa = self.art[cands[r]]
            lp, sk = hp >= 50, hp < 10
            out["rep_item"][r] = _count_of(ht, cands[r])
            out["rep_lp"][r] = _count_of(ht[lp], cands[r])
            out["rep_skip"][r] = _count_of(ht[sk], cands[r])
            out["rep_artist"][r] = _count_of(ha, aa)
            out["rep_artist_lp"][r] = _count_of(ha[lp], aa)
        return out


def load_rm(path, device) -> RewardModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = RewardModel(**ck["cfg"]).to(device)
    m.load_state_dict(ck["state"])
    return m
