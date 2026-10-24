"""Relevance, behaviour and the paired user bootstrap with the decision words of task 1.

A user-day's target set is every distinct track the user listened to >= 50% that day (organic and
algorithmic). Recall@K divides by all targets, out-of-vocabulary ones included, so the vocabulary
ceiling is part of the number and identical for every model.
"""
import numpy as np

KS = (10, 100)


def _dcg_weights(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


_HIST: dict = {}


def history(seq, ud_idx: int) -> np.ndarray:
    """Distinct tokens the user listened to before the day (sorted); cached per user-day across specs."""
    h = _HIST.get(int(ud_idx))
    if h is None:
        u, e = seq.ud["u"][ud_idx], seq.ud["ctx_end"][ud_idx]
        h = np.unique(np.asarray(seq.tok[seq.ptr[u]:e]))
        _HIST[int(ud_idx)] = h
    return h


def relevance(seq, uds: np.ndarray, topk: np.ndarray) -> dict[str, np.ndarray]:
    """Per user-day Recall@K / NDCG@K, plus the same on targets new to the user (NaN if none)."""
    n = len(uds)
    out = {f"{m}@{k}{s}": np.full(n, np.nan) for m in ("recall", "ndcg") for k in KS for s in ("", "_new")}
    w = _dcg_weights(max(KS))
    for r, i in enumerate(uds):
        tg = seq.targets(i)
        n_all = int(seq.ud["n_tgt"][i])
        hist = history(seq, i)
        new = tg[~np.isin(tg, hist, assume_unique=False)]
        for suffix, t, n_t in (("", tg, n_all), ("_new", new, len(new))):
            if n_t == 0:
                continue
            hit = np.isin(topk[r], t)
            for k in KS:
                h = hit[:k]
                out[f"recall@{k}{suffix}"][r] = h.sum() / n_t
                out[f"ndcg@{k}{suffix}"][r] = (h * w[:k]).sum() / w[:min(n_t, k)].sum()
    return out


def behaviour(seq, uds: np.ndarray, topk: np.ndarray, k: int = 10, rng_seed: int = 0) -> dict[str, float]:
    """Is everyone shown the same thing? Coverage, concentration, similarity across users, item profile."""
    top = topk[:, :k]
    n_vocab = seq.n_tok
    flat = top.ravel()
    freq = np.bincount(flat, minlength=n_vocab).astype(float)
    shown = np.sort(freq[freq > 0])
    cum = np.cumsum(shown)
    gini_shown = 1 - 2 * (cum / cum[-1]).sum() / len(shown) + 1 / len(shown)
    most = int(freq.argmax())
    rng = np.random.default_rng(rng_seed)
    a, b = rng.integers(0, len(top), 2000), rng.integers(0, len(top), 2000)
    jac = [len(np.intersect1d(top[x], top[y])) / len(np.union1d(top[x], top[y])) for x, y in zip(a, b) if x != y]
    pop = seq.items["log_pop"]
    pop_rank = np.argsort(np.argsort(pop)) / len(pop)  # percentile of train popularity
    tlen = seq.items["track_len"]
    art = seq.items["artist"]
    rep = new_art = 0.0
    len_dev = []  # personal length profile (owner, 24.10): |mean log length of the list - of the user's tracks|
    loglen = np.log(np.clip(tlen, 10, 3600))
    for r, i in enumerate(uds):
        hist = history(seq, i)
        rep += np.isin(top[r], hist).mean()
        new_art += (~np.isin(art[top[r]], art[hist[hist >= 2]])).mean()
        hv = hist[hist >= 2]
        if len(hv):
            len_dev.append(abs(loglen[top[r]].mean() - loglen[hv].mean()))
    return {
        f"coverage@{k}": float((freq > 0).sum() / (n_vocab - 2)),
        "unique_items": int((freq > 0).sum()),
        "gini_shown": float(gini_shown),
        "top_item_share": float((top == most).any(1).mean()),
        "jaccard_pairs": float(np.mean(jac)),
        "pop_percentile": float(pop_rank[flat].mean()),
        "track_len_mean": float(tlen[flat].mean()),
        "track_len_short_share": float((tlen[flat] <= 150).mean()),
        "repeat_share": rep / len(uds),
        "len_dev_user_log": float(np.mean(len_dev)) if len_dev else float("nan"),
        "new_artist_share": new_art / len(uds),
    }


def user_bootstrap(users: np.ndarray, a: np.ndarray, b: np.ndarray | None = None, n_boot: int = 1000,
                   seed: int = 0) -> dict:
    """Mean of a (or of a - b, paired) with a 95% CI; a user is resampled with all of their days."""
    x = a if b is None else a - b
    ok = ~np.isnan(x)
    users, x = users[ok], x[ok]
    uniq, inv = np.unique(users, return_inverse=True)
    s = np.bincount(inv, weights=x, minlength=len(uniq))
    c = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot).astype(float)
    reps = (w @ s) / (w @ c)
    return {"mean": float(s.sum() / c.sum()), "lo": float(np.quantile(reps, 0.025)),
            "hi": float(np.quantile(reps, 0.975)), "n_users": int(len(uniq))}


def verdict(ci: dict, threshold: float) -> str:
    """Decision words fixed before OOT (same rule as task 1)."""
    if ci["lo"] > 0:
        return "лучше"
    if ci["hi"] < 0:
        return "хуже"
    return "сопоставимо" if abs(ci["mean"]) < threshold else "не определено"
