"""Post-OOT diagnostics (docs/post_oot_diagnostics_RU.md, written 25.10 before this was run).

On the saved top-10 of every spec, against SFT, paired user bootstrap:
  D1  B gain of the snapshot judges m4-m6 (as in evaluation) and the same with every recommended track's
      length set to the user's habitual length L_u: the share of the gain that is not track length;
  D2  expected session minutes per item split by track-length bucket: composition (longer buckets)
      vs within (better tracks of the same length), as in decompose.py;
  J1  judges' P(like), J1n = P(like) minus the train like rate of the track's length bucket,
  J2  judges' P(Listen+) minus the train Listen+ rate of the length bucket (the fair reward);
  R1  NDCG@10 on the tracks the user listened to and liked that day (from the log, no model),
  R2  share of the day's listened minutes (vocabulary tracks) that falls on the top-10 tracks,
  R3  share of the top-10 the user had disliked before the day (dislike not undone).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rlrec  # noqa: E402,F401
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from rlrec.config import CKPT, DATA, DEVICE, LIKE_HORIZON, N_SPECIAL, RUNS, TRAIN_END, day_start  # noqa: E402
from rlrec.data import Seq  # noqa: E402
from rlrec.evaluate import contexts  # noqa: E402
from rlrec.metrics import user_bootstrap, verdict  # noqa: E402
from rlrec.model import limit_gpu, make_batch  # noqa: E402
from rlrec.rl import LEN_EDGES, Scorer, balanced, session_minutes  # noqa: E402
from rlrec.rm import load_rm  # noqa: E402
from compare import REL_THRESHOLD, REWARD_THRESHOLD  # noqa: E402
from decompose import split  # noqa: E402
from eval_policy import user_loglen  # noqa: E402

K = 10
W = 1.0 / np.log2(np.arange(2, K + 2))


def like_by_length(seq) -> np.ndarray:
    """Per token: like rate of algorithmic train listens (label horizon closed) in the token's length bucket."""
    org, day, tok, t, like = (np.asarray(seq.org), np.asarray(seq.day), np.asarray(seq.tok), np.asarray(seq.t),
                              np.asarray(seq.like))
    tr = (org == 0) & (day < TRAIN_END) & (tok >= N_SPECIAL) & (t + LIKE_HORIZON < day_start(TRAIN_END))
    tlen = seq.items["track_len"]
    bk = np.searchsorted(LEN_EDGES, tlen[tok[tr]])
    rate = np.bincount(bk, weights=like[tr], minlength=len(LEN_EDGES) + 1) / np.maximum(
        1, np.bincount(bk, minlength=len(LEN_EDGES) + 1))
    return rate[np.searchsorted(LEN_EDGES, tlen)].astype(np.float32)


def day_log(seq, uds: np.ndarray) -> dict:
    """Per user-day from the log: liked raw items of the day, minutes per vocabulary token of the day,
    tokens with an active dislike at the start of the day."""
    u, e, d = seq.ud["u"][uds], seq.ud["ctx_end"][uds], seq.ud["day"][uds]
    tlen = seq.items["track_len"]
    liked, mins_tok, mins_val, total = [], [], [], np.zeros(len(uds))
    for r in range(len(uds)):
        s, stop = int(e[r]), int(seq.ptr[u[r] + 1])
        end = s + int(np.searchsorted(np.asarray(seq.day[s:stop]), d[r], side="right"))
        tok, pr = np.asarray(seq.tok[s:end]), np.asarray(seq.pr[s:end])
        liked.append(np.unique(np.asarray(seq.item[s:end])[np.asarray(seq.like[s:end]) > 0]))
        m = tok >= N_SPECIAL
        ut, inv = np.unique(tok[m], return_inverse=True)
        mm = np.bincount(inv, weights=np.clip(pr[m], 0, 100) / 100 * tlen[tok[m]] / 60, minlength=len(ut))
        mins_tok.append(ut)
        mins_val.append(mm)
        total[r] = mm.sum()
    # active dislikes at day start: the last of (dislike = 2, undislike = 4) on the pair before the day is 2
    ev = (pl.scan_parquet(DATA / "events.parquet").filter(pl.col("kind").is_in([2, 4]))
          .select("uid", "item", "t", "kind").collect())
    q = pl.DataFrame({"r": np.arange(len(uds)), "uid": np.asarray(seq.uid)[u].astype(np.uint32),
                      "t0": np.array([day_start(int(x)) for x in d], np.int64)})
    vocab = pl.DataFrame({"item": seq.items["item_id"][N_SPECIAL:].astype(np.uint32),
                          "tok": np.arange(N_SPECIAL, seq.n_tok, dtype=np.int64)})
    dis = (q.join(ev, on="uid").filter(pl.col("t").cast(pl.Int64) < pl.col("t0")).sort("t")
           .group_by(["r", "item"]).agg(pl.col("kind").last()).filter(pl.col("kind") == 2)
           .join(vocab, on="item").group_by("r").agg(pl.col("tok")))
    disliked = [np.zeros(0, np.int64)] * len(uds)
    for r, toks in zip(dis["r"].to_list(), dis["tok"].to_list()):
        disliked[r] = np.asarray(toks, np.int64)
    return {"liked": liked, "mins_tok": mins_tok, "mins_val": mins_val, "total": total, "disliked": disliked}


def log_metrics(seq, log: dict, top: np.ndarray) -> dict:
    n = len(top)
    item = seq.items["item_id"]
    r1, r2, r3 = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    for r in range(n):
        lk = log["liked"][r]
        if len(lk):
            hit = np.isin(item[top[r]], lk)
            r1[r] = (hit * W).sum() / W[:min(len(lk), K)].sum()
        if log["total"][r] > 0:
            r2[r] = log["mins_val"][r][np.isin(log["mins_tok"][r], top[r])].sum() / log["total"][r]
        if len(log["disliked"][r]):
            r3[r] = np.isin(top[r], log["disliked"][r]).mean()
    return {"R1_liked_ndcg": r1, "R2_minutes_share": r2, "R3_disliked_share": r3}


@torch.no_grad()
def judge_metrics(rms, scorer, seq, uds, top, lu, base_like, dev, bs=256) -> dict:
    """Mean over the judges, per user-day (B, B_neutral, J1, J1n, J2) and per item (expected session minutes)."""
    ctx = contexts(seq, uds)
    n = len(uds)
    out = {k: np.zeros(n, np.float32) for k in ("B", "B_neutral", "J1_like", "J1n_like_len", "J2_lp_len")}
    item_min = np.zeros((n, K), np.float32)
    sh = scorer.st["bal_s_new"]
    for i in range(0, n, bs):
        c = ctx[i:i + bs]
        b = make_batch(seq, c, 256, dev)
        cand = torch.from_numpy(top[i:i + bs].astype(np.int64)).to(dev)
        side = scorer.side(c, top[i:i + bs])
        scorer.set_week(c)
        cat = torch.where(side[..., 5] > 0.5, 0, torch.where(side[..., 4] > 0.5, 2, 1)).long()
        L = scorer.tlen[cand]
        Lu = torch.from_numpy(np.exp(lu[i:i + bs])).to(dev)[:, None].expand_as(L)
        Ln = torch.where(torch.isnan(Lu), L, Lu)
        acc = {k: 0.0 for k in out}
        acc_min = 0.0
        for rm in rms:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = rm.hidden(b)[:, -1]
            p = scorer.probs(rm, h, cand, side)
            ratio = p[..., 0].float()
            for key, length in (("B", L), ("B_neutral", Ln)):
                mins = session_minutes(p, ratio * length / 60.0)
                add = torch.zeros_like(scorer.week).scatter_add_(1, cat, mins)
                acc[key] = acc[key] + balanced(scorer.week + add, sh) - balanced(scorer.week, sh)
                if key == "B":
                    acc_min = acc_min + mins
            like, lp = p[..., 3].float(), p[..., 1].float()
            acc["J1_like"] = acc["J1_like"] + like.mean(1)
            acc["J1n_like_len"] = acc["J1n_like_len"] + (like - base_like[cand]).mean(1)
            acc["J2_lp_len"] = acc["J2_lp_len"] + (lp - scorer.base[cand]).mean(1)
        for k in out:
            out[k][i:i + bs] = (acc[k] / len(rms)).cpu().numpy()
        item_min[i:i + bs] = (acc_min / len(rms)).cpu().numpy()
    return out, item_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--stage", default="val", choices=["val", "oot"])
    ap.add_argument("--base", default="sft")
    ap.add_argument("--subset", type=int, default=0, help="first n user-days only, written to diag_subset/ (debug)")
    a = ap.parse_args()
    dev = torch.device(DEVICE)
    limit_gpu(dev)
    seq = Seq()
    d = RUNS / "eval" / a.stage
    od = d / ("diag_subset" if a.subset else "diag")
    od.mkdir(exist_ok=True)
    n = a.subset or None
    uds = np.load(d / f"{a.base}.npz")["uds"][:n]
    users = seq.ud["u"][uds]
    rms = [load_rm(CKPT / f"rm_m{m}.pt", dev).eval() for m in (4, 5, 6)]
    scorer = Scorer(seq, TRAIN_END, dev)
    base_like = torch.from_numpy(like_by_length(seq)).to(dev)
    lu = user_loglen(seq, contexts(seq, uds))
    log = day_log(seq, uds)
    bucket = np.searchsorted(LEN_EDGES, seq.items["track_len"])
    per, item_min, tops = {}, {}, {}
    for spec in [a.base] + [s for s in a.specs if s != a.base]:
        fn = spec.replace(":", "__")
        z = np.load(d / f"{fn}.npz")
        assert (z["uds"][:n] == uds).all(), spec
        top = z["topk"][:n, :K]
        jm, im = judge_metrics(rms, scorer, seq, uds, top, lu, base_like, dev)
        per[spec] = {**jm, **log_metrics(seq, log, top)}
        item_min[spec], tops[spec] = im, top
        # the B of this pass must match the saved evaluation (same judges, same formula)
        err = float(np.abs(jm["B"] - z["rw_balanced"][4:, :n].mean(0)).max())
        np.savez_compressed(od / f"{fn}.npz", **per[spec])
        print(f"{spec}: B {jm['B'].mean():.3f} (max |diff| to saved eval {err:.4f}), B_neutral {jm['B_neutral'].mean():.3f}",
              flush=True)
    b = per[a.base]
    out = {}
    for spec, m in per.items():
        if spec == a.base:
            continue
        row = {}
        for k, v in m.items():
            ci = user_bootstrap(users, v, b[k])
            base_mean = float(np.nanmean(b[k]))
            row[k] = {"base": base_mean, "spec": float(np.nanmean(v)), "delta": ci}
            if k.startswith("R1") or k.startswith("R2"):
                row[k]["verdict"] = verdict(ci, REL_THRESHOLD * base_mean)
            elif k.startswith("J") or k.startswith("R3"):
                row[k]["verdict"] = verdict(ci, REWARD_THRESHOLD)
        dB, dBn = row["B"]["delta"]["mean"], row["B_neutral"]["delta"]["mean"]
        row["B_not_length_share"] = dBn / dB if abs(dB) > 1e-9 else None
        s0, s1 = bucket[tops[a.base]].ravel(), bucket[tops[spec]].ravel()
        row["D2_minutes_split"] = split(s0, item_min[a.base].ravel().astype(np.float64),
                                        s1, item_min[spec].ravel().astype(np.float64))
        js = json.loads((d / f"{spec.replace(':', '__')}.json").read_text())["behaviour"]
        row["track_len_mean"], row["len_dev_user_log"] = js["track_len_mean"], js["len_dev_user_log"]
        out[spec] = row
        sp = row["D2_minutes_split"]
        print(f"{spec:30s} dB {dB:+.2f} dB_neutral {dBn:+.2f} (not length {row['B_not_length_share'] or 0:+.0%});"
              f" minutes/item {sp['delta']:+.3f} = length {sp['composition']:+.3f} + within {sp['within']:+.3f};"
              + " ".join(f" {k.split('_')[0]} {row[k]['delta']['mean']:+.4f} {row[k].get('verdict', '')}"
                         for k in ("J1_like", "J1n_like_len", "J2_lp_len", "R1_liked_ndcg", "R2_minutes_share",
                                   "R3_disliked_share")), flush=True)
    prev = json.loads((od / "diag.json").read_text()) if (od / "diag.json").exists() else {}
    prev.update(out)
    (od / "diag.json").write_text(json.dumps(prev, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
