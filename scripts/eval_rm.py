"""Reward model quality on held-out algorithmic listens (val or OOT days), every member, vs simple rules.

Context = real history up to the listen (last 128-256 events). Metrics: AUC per binary head, GAUC of lp
within user-day, RMSE and calibration of ratio. Slices: repeat / new track, new artist.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rlrec  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from rlrec.config import ART, CKPT, DEVICE, N_SPECIAL, OOT_DAYS, RUNS, T_END, TRAIN_END, VAL_DAYS  # noqa: E402
from rlrec.data import Seq, adopt_label, label_valid_eval  # noqa: E402
from rlrec.model import limit_gpu, make_batch  # noqa: E402
from rlrec.rm import HEADS, load_rm, logged_counts, side_features  # noqa: E402


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gauc(group: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(group, kind="stable")
    g, y, s = group[order], y[order], s[order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    num = den = 0.0
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(g)]):
        v = auc(y[a:b], s[a:b])
        if not np.isnan(v):
            num += v * (b - a)
            den += b - a
    return num / den


def eval_windows(seq, days: list[int], chunk: int = 128) -> list[tuple[int, int, int]]:
    """(window_start, window_end, first_target) covering every listen of `days` once."""
    day = np.asarray(seq.day)
    out = []
    for u in range(len(seq.ptr) - 1):
        s, e = int(seq.ptr[u]), int(seq.ptr[u + 1])
        a = s + int(np.searchsorted(day[s:e], days[0]))
        b = s + int(np.searchsorted(day[s:e], days[-1] + 1))
        for c in range(a, b, chunk):
            out.append((max(s, c - chunk - 1), min(c + chunk, b), c))
    return out


@torch.no_grad()
def score_logged(rms: list, seq, days: list[int], device, bs: int = 32, max_windows: int = 0) -> dict:
    org, tok = np.asarray(seq.org), np.asarray(seq.tok)
    valid_lab = label_valid_eval(seq)
    wins = eval_windows(seq, days)
    if max_windows:
        wins = wins[:max_windows]
    gis, outs = [], [[] for _ in rms]
    for m in rms:
        m.eval()
    for i in range(0, len(wins), bs):
        w = wins[i:i + bs]
        b = make_batch(seq, [(s, e) for s, e, _ in w], 257, device)
        gi_next = b["gi"][:, 1:].cpu().numpy()
        first = np.array([c for _, _, c in w])[:, None]
        g = gi_next.clip(min=0)
        sel = (gi_next >= first) & (org[g] == 0) & (tok[g] >= N_SPECIAL) & valid_lab[g] & (gi_next >= 0)
        if not sel.any():
            continue
        gv = g[sel]
        cand = torch.from_numpy(tok[gv]).to(device)
        side = torch.from_numpy(side_features(seq.items, tok[gv], logged_counts(seq, gv))).to(device)
        inp = {k: v[:, :-1] for k, v in b.items()}
        selt = torch.from_numpy(sel).to(device)
        for j, m in enumerate(rms):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = m.hidden(inp)[selt]
                outs[j].append(torch.sigmoid(m.head(h, cand, side)).cpu().numpy())
        gis.append(gv)
    return {"gi": np.concatenate(gis), "pred": [np.concatenate(o) for o in outs]}


@torch.no_grad()
def score_snapshot(rms: list, seq, days: list[int], device, bs: int = 64, max_ud: int = 0) -> dict:
    """The RM as RL uses it: context = history before the start of the day, candidates = that day's
    algorithmic listens, side features counted at the day start (HistoryCounts). Comparable with task 1."""
    from rlrec.evaluate import contexts
    from rlrec.rm import HistoryCounts
    org, tok, day = np.asarray(seq.org), np.asarray(seq.tok), np.asarray(seq.day)
    uds = seq.userdays(days, need_targets=False)
    if max_ud:
        uds = uds[:max_ud]
    ctx = contexts(seq, uds)
    hc = HistoryCounts(seq)
    gis, outs = [], [[] for _ in rms]
    for i in range(0, len(uds), bs):
        c = ctx[i:i + bs]
        lists = []
        for (s0, e), ud in zip(c, uds[i:i + bs]):
            u = seq.ud["u"][ud]
            end = int(seq.ptr[u + 1])
            j = e + np.flatnonzero(day[e:end] == seq.ud["day"][ud])
            j = j[(org[j] == 0) & (tok[j] >= N_SPECIAL)]
            lists.append(j)
        C = max(1, max(len(j) for j in lists))
        cand = np.ones((len(lists), C), np.int64)  # padding candidate, dropped below
        for r, j in enumerate(lists):
            cand[r, :len(j)] = tok[j]
        side = torch.from_numpy(side_features(seq.items, cand, hc(c, cand))).to(device)
        b = make_batch(seq, c, 256, device)
        ct = torch.from_numpy(cand).to(device)
        for k, m in enumerate(rms):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = m.hidden(b)[:, -1]
                z = m.head(h[:, None, :].expand(-1, C, -1), ct, side)
            p = torch.sigmoid(z).cpu().numpy()
            outs[k].append(np.concatenate([p[r, :len(j)] for r, j in enumerate(lists)]))
        gis.append(np.concatenate(lists))
    return {"gi": np.concatenate(gis), "pred": [np.concatenate(o) for o in outs]}


def baselines(seq, gi: np.ndarray) -> dict:
    """Simple rules scored on the same listens; statistics from algorithmic listens before TRAIN_END."""
    org, day, pr, tok = (np.asarray(seq.org), np.asarray(seq.day), np.asarray(seq.pr), np.asarray(seq.tok))
    tr = (org == 0) & (day < TRAIN_END)
    lp = (pr >= 50).astype(float)
    prior, k = lp[tr].mean(), 20.0
    n_i = np.bincount(tok[tr], minlength=seq.n_tok)
    s_i = np.bincount(tok[tr], weights=lp[tr], minlength=seq.n_tok)
    item_rate = (s_i + k * prior) / (n_i + k)
    uidx = seq.user_index()
    n_u = np.bincount(uidx[tr], minlength=len(seq.ptr) - 1)
    s_u = np.bincount(uidx[tr], weights=lp[tr], minlength=len(seq.ptr) - 1)
    user_rate = (s_u + k * prior) / (n_u + k)
    return {"item_rate": item_rate[tok[gi]], "user_rate": user_rate[uidx[gi]],
            "short_track": -seq.items["track_len"][tok[gi]], "repeat_count": np.asarray(seq.rep_item)[gi].astype(float)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="val", choices=["val", "oot"])
    ap.add_argument("--members", default="0,1,2,3")
    ap.add_argument("--max-windows", type=int, default=0, help="debug")
    ap.add_argument("--snapshot", action="store_true", help="context before the day start, as in RL")
    a = ap.parse_args()
    if a.stage == "oot" and not (ART / "freeze.json").exists():
        sys.exit("OOT needs artifacts/freeze.json")
    days = VAL_DAYS if a.stage == "val" else OOT_DAYS
    dev = torch.device(DEVICE)
    limit_gpu(dev)
    seq = Seq()
    members = [int(x) for x in a.members.split(",")]
    rms = [load_rm(CKPT / f"rm_m{m}.pt", dev) for m in members]
    sc = (score_snapshot(rms, seq, days, dev, max_ud=a.max_windows) if a.snapshot
          else score_logged(rms, seq, days, dev, max_windows=a.max_windows))
    gi = sc["gi"]
    pr = np.asarray(seq.pr)[gi]
    y = {"ratio": np.clip(pr, 0, 100) / 100, "lp": pr >= 50, "skip": pr < 10, "like": np.asarray(seq.like)[gi],
         "cont": np.asarray(seq.cont)[gi], "adopt": adopt_label(seq, gi, T_END)}
    ad_ok = y["adopt"] >= 0
    uidx = seq.user_index()
    group = uidx[gi].astype(np.int64) * 1000 + np.asarray(seq.day)[gi]
    rep = np.asarray(seq.rep_item)[gi] > 0
    new_art = np.asarray(seq.rep_artist)[gi] == 0
    res = {"n": int(len(gi)), "members": {}, "baselines": {}}
    judges = np.mean([p for m, p in zip(members, sc["pred"]) if m != 0], 0) if len(members) > 1 else None
    preds = {f"m{m}": p for m, p in zip(members, sc["pred"])}
    if judges is not None:
        preds["judges_mean"] = judges
    for name, p in preds.items():
        r = {h: auc(y[h], p[:, j]) for j, h in enumerate(HEADS) if h not in ("ratio", "adopt")}
        r["adopt"] = auc(y["adopt"][ad_ok], p[ad_ok, HEADS.index("adopt")])
        r["adopt_base_rate"] = float(y["adopt"][ad_ok].mean())
        r["gauc_lp"] = gauc(group, y["lp"], p[:, 1])
        r["rmse_ratio"] = float(np.sqrt(np.mean((p[:, 0] - y["ratio"]) ** 2)))
        r["auc_lp_repeat"] = auc(y["lp"][rep], p[rep, 1])
        r["auc_lp_new_track"] = auc(y["lp"][~rep], p[~rep, 1])
        r["auc_lp_new_artist"] = auc(y["lp"][new_art], p[new_art, 1])
        bins = np.clip((p[:, 1] * 10).astype(int), 0, 9)
        r["calib_lp"] = [[float(p[bins == k, 1].mean()), float(y["lp"][bins == k].mean()), int((bins == k).sum())]
                         for k in range(10) if (bins == k).any()]
        res["members"][name] = r
    for name, s in baselines(seq, gi).items():
        res["baselines"][name] = {"lp": auc(y["lp"], s), "gauc_lp": gauc(group, y["lp"], s),
                                  "skip": auc(y["skip"], -s)}
    if len(members) > 1:
        res["ensemble_std_lp_mean"] = float(np.std([p[:, 1] for p in sc["pred"]], 0).mean())
    out = RUNS / f"rm_eval_{a.stage}{'_snapshot' if a.snapshot else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items()}, indent=1)[:4000])


if __name__ == "__main__":
    main()
