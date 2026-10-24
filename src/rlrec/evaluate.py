"""Top-K recommendation from a day-start snapshot and the shared evaluation protocol."""
import json

import numpy as np
import torch

from .metrics import behaviour, relevance, user_bootstrap
from .model import make_batch


def contexts(seq, uds: np.ndarray) -> list[tuple[int, int]]:
    u, e = seq.ud["u"][uds], seq.ud["ctx_end"][uds]
    return list(zip(seq.ptr[u].tolist(), e.tolist()))


@torch.no_grad()
def recommend(model, seq, uds: np.ndarray, device, length: int = 256, k: int = 100, bs: int = 256,
              score_fn=None) -> np.ndarray:
    """Top-k tokens per user-day. score_fn(batch, logits) -> scores lets a reranker replace log pi."""
    model.eval()
    ctx = contexts(seq, uds)
    out = np.zeros((len(uds), k), np.int64)
    for i in range(0, len(uds), bs):
        b = make_batch(seq, ctx[i:i + bs], length, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.last_logits(b)
        if score_fn is not None:
            z = score_fn(b, z)
        out[i:i + bs] = z.topk(k, dim=1).indices.cpu().numpy()
    return out


@torch.no_grad()
def reward_of_lists(rms: list, scorer, seq, uds: np.ndarray, lists: np.ndarray, device, kinds=("lp",),
                    length: int = 256, bs: int = 256) -> dict:
    """Mean RM reward over each user-day's list, per RM member and reward kind -> {kind: [n_members, n]}.

    Also returns the per-item lp probabilities of every member ("lp_items": [n_members, n, k]) for
    ensemble-spread analysis.
    """
    ctx = contexts(seq, uds)
    out = {k: np.zeros((len(rms), len(uds)), np.float32) for k in kinds}
    lp_items = np.zeros((len(rms), *lists.shape), np.float32)
    for i in range(0, len(uds), bs):
        c = ctx[i:i + bs]
        b = make_batch(seq, c, length, device)
        cand = torch.from_numpy(lists[i:i + bs]).to(device)
        side = scorer.side(c, lists[i:i + bs])
        for j, rm in enumerate(rms):
            rm.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = rm.hidden(b)[:, -1]
            p = scorer.probs(rm, h, cand, side)
            lp_items[j, i:i + bs] = p[..., 1].float().cpu().numpy()
            for k in kinds:
                out[k][j, i:i + bs] = scorer.reward(p, cand, k, side).float().mean(1).cpu().numpy()
    out["lp_items"] = lp_items
    return out


def summarize(seq, uds: np.ndarray, topk: np.ndarray, with_behaviour: bool = True) -> dict:
    rel = relevance(seq, uds, topk)
    users = seq.ud["u"][uds]
    res = {m: user_bootstrap(users, v, n_boot=200) for m, v in rel.items()}
    if with_behaviour:
        res["behaviour"] = behaviour(seq, uds, topk)
    return res


def save_eval(path, uds, topk, rel_per_ud: dict | None = None, summary: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), uds=uds, topk=topk.astype(np.int32),
                        **({k: v for k, v in rel_per_ud.items()} if rel_per_ud else {}))
    if summary is not None:
        path.with_suffix(".json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
