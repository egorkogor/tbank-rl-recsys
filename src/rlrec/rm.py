"""Reward model: response of a user to a candidate track given the history.

Causal GPT-2 backbone (same input as the policy) -> h_t; the candidate is the track of event t+1.
score = MLP([h_t, e_c, h_t * e_c, side_c]) -> heads. One pass over a window scores every position
whose next event is algorithmic; h_t sees events <= t only.

side_c (known before the candidate plays): log track length, log train popularity, prior listens of
the track and of its artist by this user (whole history), new-artist flag.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Policy

HEADS = ("ratio", "lp", "skip", "like", "cont")
N_SIDE = 6


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


def side_features(track_len: np.ndarray, log_pop: np.ndarray, rep_item: np.ndarray, rep_artist: np.ndarray):
    return np.stack([np.log(np.clip(track_len, 10, 3600)) - 5.3, log_pop / 5 - 1, np.log1p(rep_item) / 3,
                     np.log1p(rep_artist) / 4, (rep_artist == 0).astype(np.float32),
                     (rep_item > 0).astype(np.float32)], -1).astype(np.float32)


def loss_fn(logits: torch.Tensor, y: dict) -> tuple[torch.Tensor, dict]:
    """BCE on every head; ratio is a soft label in [0, 1]."""
    parts = {}
    for j, name in enumerate(HEADS):
        parts[name] = F.binary_cross_entropy_with_logits(logits[:, j], y[name].float())
    w = {"ratio": 1.0, "lp": 1.0, "skip": 1.0, "like": 1.0, "cont": 0.5}
    return sum(w[k] * v for k, v in parts.items()), {k: float(v.detach()) for k, v in parts.items()}


class HistoryCounts:
    """Prior listens of arbitrary tracks / artists in a user's history before a context end."""

    def __init__(self, seq):
        self.seq = seq
        self.art = seq.items["artist"]

    def __call__(self, ud_ctx: list[tuple[int, int]], cands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ud_ctx[r] = (user_start, ctx_end); cands [R, C] tokens -> counts [R, C] for track and artist."""
        ri = np.zeros(cands.shape, np.int64)
        ra = np.zeros(cands.shape, np.int64)
        tok, artist = self.seq.tok, self.seq.artist
        for r, (s, e) in enumerate(ud_ctx):
            if e <= s:
                continue
            # Same definition as seq.rep_item / seq.rep_artist: every prior listen, OOV tracks included.
            ut, ct = np.unique(np.asarray(tok[s:e]), return_counts=True)
            pos = np.searchsorted(ut, cands[r]).clip(0, len(ut) - 1)
            ri[r] = np.where(ut[pos] == cands[r], ct[pos], 0)
            ua, ca = np.unique(np.asarray(artist[s:e]), return_counts=True)
            aa = self.art[cands[r]]
            pos = np.searchsorted(ua, aa).clip(0, len(ua) - 1)
            ra[r] = np.where(ua[pos] == aa, ca[pos], 0)
        return ri, ra


def load_rm(path, device) -> RewardModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = RewardModel(**ck["cfg"]).to(device)
    m.load_state_dict(ck["state"])
    return m
