"""Per-user listen sequences, the item vocabulary, RM labels and the evaluation user-days.

Everything the models see is built here once (scripts/prepare.py) and stored as flat .npy arrays in
artifacts/seq/. A user's listens are a contiguous slice [ptr[u], ptr[u+1]) sorted by time.

Leakage rules:
- vocabulary, item popularity and track length come from days < TRAIN_END only;
- RM labels with a horizon (like within 24 h, next listen within 30 min) are stored for every event,
  and training keeps only events whose horizon closes before day_start(TRAIN_END) (rm_train_mask);
- a user-day context is every listen strictly before the start of that day (ctx_end).
"""
import json

import rlrec  # noqa: F401  (thread limits before polars)
import numpy as np
import polars as pl

from .config import (CONT_GAP, DATA, GAP_EDGES, LIKE_HORIZON, LISTEN_PLUS, N_SPECIAL, OOT_DAYS, PARTIAL, RL_DAYS, SEQ,
                     SKIP, T_END, TRAIN_END, UNK, VAL_DAYS, VOCAB, day_start)

COLS = ["tok", "item", "artist", "t", "day", "org", "pr", "outcome", "gap", "rep_item", "rep_artist", "like", "cont"]
# Prior Listen+ / skip listens of the same track and prior Listen+ of the same artist, per user (pair_stats).
PAIR_COLS = ["rep_lp", "rep_skip", "rep_artist_lp"]


def _outcome(pr: pl.Expr) -> pl.Expr:
    return (pl.when(pr < 10).then(SKIP).when(pr < 50).then(PARTIAL).otherwise(LISTEN_PLUS)).cast(pl.Int8)


def _gap_bucket(gap: pl.Expr) -> pl.Expr:
    e = pl.when(gap.is_null()).then(0)
    for i, edge in enumerate(GAP_EDGES):
        e = e.when(gap < edge).then(i + 1)
    return e.otherwise(len(GAP_EDGES) + 1).cast(pl.Int8)


def build() -> None:
    SEQ.mkdir(parents=True, exist_ok=True)
    ev = pl.read_parquet(DATA / "events.parquet", columns=["uid", "item", "artist", "t", "day", "kind", "org", "pr"])
    likes = ev.filter(pl.col("kind") == 1).select("uid", "item", pl.col("t").alias("t_like")).sort("t_like")
    lis = ev.filter(pl.col("kind") == 0).drop("kind")
    del ev

    # Vocabulary: most listened tracks before TRAIN_END, ties broken by item id.
    cnt = (lis.filter(pl.col("day") < TRAIN_END).group_by("item").len()
           .sort(["len", "item"], descending=[True, False]).head(VOCAB))
    vocab = cnt.with_columns(pl.int_range(N_SPECIAL, N_SPECIAL + cnt.height, dtype=pl.Int32).alias("tok"))

    lis = (lis.with_row_index("rid")  # time order of the source table
           .sort(["uid", "t", "rid"])
           .join(vocab.select("item", "tok"), on="item", how="left")
           .with_columns(pl.col("tok").fill_null(UNK)))

    # Like of the same track within 24 h after the listen started (forward as-of join).
    lk = (lis.select("rid", "uid", "item", "t").sort("t")
          .join_asof(likes, left_on="t", right_on="t_like", by=["uid", "item"], strategy="forward",
                     tolerance=LIKE_HORIZON, check_sortedness=False)
          .select("rid", pl.col("t_like").is_not_null().cast(pl.Int8).alias("like")))
    lis = lis.join(lk, on="rid", how="left")

    nxt = pl.col("t").shift(-1).over("uid")
    lis = lis.with_columns(
        _outcome(pl.col("pr")).alias("outcome"),
        _gap_bucket(pl.col("t") - pl.col("t").shift(1).over("uid")).alias("gap"),
        pl.int_range(pl.len(), dtype=pl.Int32).over(["uid", "item"]).alias("rep_item"),  # prior listens of the track
        pl.int_range(pl.len(), dtype=pl.Int32).over(["uid", "artist"]).alias("rep_artist"),
        ((nxt - pl.col("t")) <= CONT_GAP).fill_null(False).cast(pl.Int8).alias("cont"),
    )

    for c in COLS:
        np.save(SEQ / f"{c}.npy", lis[c].to_numpy())
    users = lis.group_by("uid", maintain_order=True).len()
    ptr = np.zeros(users.height + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(users["len"].to_numpy())
    np.save(SEQ / "uid.npy", users["uid"].to_numpy())
    np.save(SEQ / "ptr.npy", ptr)

    # Item table indexed by token (rows 0..1 = PAD, UNK). Track length and artist from the log.
    items = pl.read_parquet(DATA / "items.parquet")
    art = lis.filter(pl.col("tok") >= N_SPECIAL).group_by("tok").agg(pl.col("artist").first())
    tab = (vocab.join(items, on="item", how="left").join(art, on="tok", how="left").sort("tok"))
    n = N_SPECIAL + tab.height
    item_id = np.zeros(n, np.int64); item_id[N_SPECIAL:] = tab["item"].to_numpy()
    artist = np.zeros(n, np.int64); artist[N_SPECIAL:] = tab["artist"].fill_null(0).to_numpy()
    tlen = np.full(n, 200.0, np.float32); tlen[N_SPECIAL:] = tab["track_len"].fill_null(200.0).to_numpy()
    pop = np.zeros(n, np.float32); pop[N_SPECIAL:] = np.log1p(tab["len"].to_numpy())
    np.savez(SEQ / "items.npz", item_id=item_id, artist=artist, track_len=tlen, log_pop=pop)

    _user_days(lis, ptr)
    build_pair_stats()
    meta = dict(n_listens=lis.height, n_users=users.height, vocab=int(n),
                train_cov=float(vocab["len"].sum() / lis.filter(pl.col("day") < TRAIN_END).height))
    (SEQ / "meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


def _user_days(lis: pl.DataFrame, ptr: np.ndarray) -> None:
    """User-days of the RL, val and OOT windows.

    ctx_end = global index of the user's first listen of that day (context = slice before it).
    Targets = distinct tracks the user listened to >= 50% that day; n_tgt counts out-of-vocabulary ones too.
    """
    days = RL_DAYS + VAL_DAYS + OOT_DAYS
    g = (lis.with_row_index("gi").with_columns(pl.int_range(pl.len()).over("uid").alias("pos"))
         .filter(pl.col("day").is_in(days)))
    uidx = pl.DataFrame({"uid": np.load(SEQ / "uid.npy"), "u": np.arange(len(ptr) - 1)})
    first = g.group_by(["uid", "day"]).agg(pl.col("gi").min().alias("ctx_end"), pl.col("pos").min().alias("n_ctx"))
    tg = (g.filter(pl.col("pr") >= 50).unique(["uid", "day", "item"])
          .group_by(["uid", "day"]).agg(pl.col("tok"), pl.len().alias("n_tgt")))
    ud = (first.join(tg, on=["uid", "day"], how="left").join(uidx, on="uid")
          .with_columns(pl.col("n_tgt").fill_null(0))
          .sort(["day", "uid"]))
    tok = ud["tok"].to_list()
    lens = np.array([0 if x is None else len(x) for x in tok], np.int64)
    tptr = np.zeros(len(tok) + 1, np.int64); tptr[1:] = np.cumsum(lens)
    flat = np.concatenate([np.asarray(x, np.int32) for x in tok if x]) if lens.sum() else np.zeros(0, np.int32)
    np.savez(SEQ / "userdays.npz", u=ud["u"].to_numpy(), day=ud["day"].to_numpy(), ctx_end=ud["ctx_end"].to_numpy(),
             n_ctx=ud["n_ctx"].to_numpy(), n_tgt=ud["n_tgt"].to_numpy(), tgt_ptr=tptr, tgt=flat)


def _prior_count(key: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """For time-ordered events: number of earlier events with the same key and ind = 1."""
    order = np.argsort(key, kind="stable")
    k, x = key[order], ind[order].astype(np.int64)
    cs = np.cumsum(x) - x  # exclusive prefix sum in sorted order
    start = np.r_[True, k[1:] != k[:-1]]
    base = np.maximum.accumulate(np.where(start, cs, 0))
    out = np.empty(len(key), np.int32)
    out[order] = cs - base
    return out


def build_pair_stats(users_per_chunk: int = 1000) -> None:
    """Pair-history quality (task 1: the strongest hand-made features) in user chunks to keep memory low."""
    ptr = np.load(SEQ / "ptr.npy")
    item, artist, pr = (np.load(SEQ / f"{c}.npy", mmap_mode="r") for c in ("item", "artist", "pr"))
    out = {c: np.zeros(int(ptr[-1]), np.int32) for c in PAIR_COLS}
    for u0 in range(0, len(ptr) - 1, users_per_chunk):
        s, e = int(ptr[u0]), int(ptr[min(u0 + users_per_chunk, len(ptr) - 1)])
        local = np.repeat(np.arange(min(users_per_chunk, len(ptr) - 1 - u0), dtype=np.int64), np.diff(ptr[u0:u0 + users_per_chunk + 1]))
        p = np.asarray(pr[s:e])
        ki = (local << 24) + np.asarray(item[s:e]).astype(np.int64)
        ka = (local << 24) + np.asarray(artist[s:e]).astype(np.int64)
        out["rep_lp"][s:e] = _prior_count(ki, p >= 50)
        out["rep_skip"][s:e] = _prior_count(ki, p < 10)
        out["rep_artist_lp"][s:e] = _prior_count(ka, p >= 50)
    for c, v in out.items():
        np.save(SEQ / f"{c}.npy", v)


class Seq:
    """Memory-mapped view of artifacts/seq/."""

    def __init__(self):
        for c in COLS + PAIR_COLS + ["ptr", "uid"]:
            setattr(self, c, np.load(SEQ / f"{c}.npy", mmap_mode="r"))
        self.items = dict(np.load(SEQ / "items.npz"))
        ud = np.load(SEQ / "userdays.npz")
        self.ud = {k: ud[k] for k in ud.files}
        self.meta = json.loads((SEQ / "meta.json").read_text())
        self.n_tok = self.meta["vocab"]
        self.user_of = None

    def user_index(self) -> np.ndarray:
        """Global listen index -> user row (built lazily, 46.5M int32)."""
        if self.user_of is None:
            self.user_of = np.repeat(np.arange(len(self.ptr) - 1, dtype=np.int32), np.diff(self.ptr))
        return self.user_of

    def userdays(self, days: list[int], min_ctx: int = 5, need_targets: bool = True) -> np.ndarray:
        d = self.ud
        m = np.isin(d["day"], days) & (d["n_ctx"] >= min_ctx)
        if need_targets:
            m &= d["n_tgt"] > 0
        return np.flatnonzero(m)

    def targets(self, i: int) -> np.ndarray:
        p = self.ud["tgt_ptr"]
        return self.ud["tgt"][p[i]:p[i + 1]]


def rm_train_mask(seq: Seq, horizon: int = max(CONT_GAP, LIKE_HORIZON)) -> np.ndarray:
    """Algorithmic listens whose label horizon closes before the training cutoff."""
    t = np.asarray(seq.t)
    return (np.asarray(seq.org) == 0) & (t + horizon < day_start(TRAIN_END)) & (np.asarray(seq.tok) >= N_SPECIAL)


def label_valid_eval(seq: Seq) -> np.ndarray:
    """Evaluation labels are valid unless the horizon runs past the end of the log."""
    return np.asarray(seq.t) + max(CONT_GAP, LIKE_HORIZON) < T_END
