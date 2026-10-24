"""How many users per arm an online A/B needs to see +1% / +2% on the candidate metrics.

Per-user weekly metrics on the val week (days 287-293, all users active that week):
Listen+ plays, algorithmic Listen+ share (ratio metric, delta method), active days, plays per active day.
Two-sided alpha = 0.05, power = 0.8: n per arm = 2 (z_a + z_b)^2 var / delta^2; with CUPED on the
same metric of the week before, var * (1 - rho^2).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rlrec  # noqa: E402,F401
import numpy as np  # noqa: E402

from rlrec.config import RUNS, VAL_DAYS  # noqa: E402
from rlrec.data import Seq  # noqa: E402

Z = 1.959964 + 0.841621


def n_per_arm(var: float, mean: float, rel: float) -> int:
    return int(np.ceil(2 * Z ** 2 * var / (rel * mean) ** 2))


def main():
    seq = Seq()
    day, org, pr = np.asarray(seq.day), np.asarray(seq.org), np.asarray(seq.pr)
    uidx = seq.user_index()
    m = (day >= VAL_DAYS[0]) & (day <= VAL_DAYS[-1])
    u = uidx[m]
    nu = len(seq.ptr) - 1
    plays = np.bincount(u, minlength=nu).astype(float)
    act = plays > 0
    lp = np.bincount(u, weights=(pr[m] >= 50), minlength=nu)
    algo = np.bincount(u, weights=(org[m] == 0), minlength=nu)
    algo_lp = np.bincount(u, weights=(org[m] == 0) & (pr[m] >= 50), minlength=nu)
    days = np.zeros(nu)
    ud = np.unique(np.stack([u, day[m]], 1), axis=0)
    np.add.at(days, ud[:, 0], 1)
    # the week before (CUPED covariate)
    mp = (day >= VAL_DAYS[0] - 7) & (day < VAL_DAYS[0])
    up = uidx[mp]
    lp_prev = np.bincount(up, weights=(pr[mp] >= 50), minlength=nu)
    plays_prev = np.bincount(up, minlength=nu).astype(float)
    days_prev = np.zeros(nu)
    udp = np.unique(np.stack([up, day[mp]], 1), axis=0)
    np.add.at(days_prev, udp[:, 0], 1)
    # owner's metric tree (24.10): minutes per user-week (north star), sessions per user-week (returns),
    # discovery: adopted new artists per user-week and the share of minutes on artists new to the user
    t_all = np.asarray(seq.t)
    tlen_ev = seq.items["track_len"][np.asarray(seq.tok)]  # OOV tracks get the default 200 s
    mins_ev = np.clip(pr, 0, 100) / 100 * tlen_ev / 60
    gap_new = np.r_[True, (np.diff(t_all) > 1800) | (np.diff(uidx) != 0)]  # session start
    rep_art, adopt = np.asarray(seq.rep_artist), np.asarray(seq.adopt)

    def weekly(mask):
        uu = uidx[mask]
        return {"minutes": np.bincount(uu, weights=mins_ev[mask], minlength=nu),
                "sessions": np.bincount(uu, weights=gap_new[mask], minlength=nu),
                "new_artist_minutes": np.bincount(uu, weights=mins_ev[mask] * (rep_art[mask] == 0), minlength=nu),
                "adopted_artists": np.bincount(uu, weights=(adopt[mask] == 1), minlength=nu)}

    cur, prev = weekly(m), weekly(mp)
    # balanced minutes B = (sum_k sqrt(s*_k M_k))^2, s* = train mix (owner's north star, 24.10)
    from rlrec.config import TRAIN_END
    from rlrec.rl import event_category
    cat_ev = event_category(np.asarray(seq.rep_item), np.asarray(seq.rep_artist))
    tr = day < TRAIN_END
    s_star = np.bincount(cat_ev[tr], weights=mins_ev[tr], minlength=3) / mins_ev[tr].sum()
    for d_, mask in ((cur, m), (prev, mp)):
        Mk = np.stack([np.bincount(uidx[mask], weights=mins_ev[mask] * (cat_ev[mask] == k), minlength=nu)
                       for k in range(3)], 1)
        d_["balanced"] = (np.sqrt(s_star[None] * Mk).sum(1)) ** 2
        d_["balance_coef"] = np.where(Mk.sum(1) > 0, d_["balanced"] / np.maximum(Mk.sum(1), 1e-9), np.nan)
    out = {"users_active_week": int(act.sum())}
    out["balanced_s_star"] = s_star.round(4).tolist()
    out["balance_coef_mean"] = float(np.nanmean(cur["balance_coef"][act]))
    for name in ("balanced", "minutes", "sessions", "adopted_artists"):
        x, xp = cur[name][act], prev[name][act]
        rho = float(np.corrcoef(x, xp)[0, 1])
        out[f"{name}_per_user_week"] = {
            "mean": float(x.mean()), "sd": float(x.std()), "rho_prev_week": rho,
            "n_per_arm_1pct": n_per_arm(x.var(), x.mean(), 0.01), "n_per_arm_2pct": n_per_arm(x.var(), x.mean(), 0.02),
            "cuped_n_per_arm_1pct": n_per_arm(x.var() * (1 - rho ** 2), x.mean(), 0.01),
            "cuped_n_per_arm_2pct": n_per_arm(x.var() * (1 - rho ** 2), x.mean(), 0.02),
            "cuped_n_per_arm_5pct": n_per_arm(x.var() * (1 - rho ** 2), x.mean(), 0.05)}
    # ratio metric: share of minutes on new artists (delta method)
    a_, b_ = cur["new_artist_minutes"][act], cur["minutes"][act]
    ok = b_ > 0
    r_ = a_[ok].sum() / b_[ok].sum()
    v_ = np.var(a_[ok] - r_ * b_[ok]) / b_[ok].mean() ** 2
    out["new_artist_minutes_share"] = {"mean": float(r_), "n_per_arm_2pct": n_per_arm(v_, r_, 0.02),
                                       "n_per_arm_5pct": n_per_arm(v_, r_, 0.05)}
    for name, x, xp in (("listen_plus_per_user_week", lp, lp_prev), ("active_days_per_user_week", days, days_prev),
                        ("plays_per_user_week", plays, plays_prev)):
        x, xp = x[act], xp[act]
        rho = float(np.corrcoef(x, xp)[0, 1])
        v_cuped = x.var() * (1 - rho ** 2)
        out[name] = {"mean": float(x.mean()), "sd": float(x.std()),
                     "n_per_arm_1pct": n_per_arm(x.var(), x.mean(), 0.01),
                     "n_per_arm_2pct": n_per_arm(x.var(), x.mean(), 0.02),
                     "rho_prev_week": rho,
                     "cuped_n_per_arm_1pct": n_per_arm(v_cuped, x.mean(), 0.01),
                     "cuped_n_per_arm_2pct": n_per_arm(v_cuped, x.mean(), 0.02)}
    # ratio metric: algorithmic Listen+ share = sum(algo_lp) / sum(algo), delta method over users
    a, b = algo_lp[act & (algo > 0)], algo[act & (algo > 0)]
    r = a.sum() / b.sum()
    var_ratio = np.var(a - r * b) / b.mean() ** 2  # per-user variance of the linearised ratio
    out["algo_listen_plus_share"] = {"mean": float(r), "sd_linearised": float(np.sqrt(var_ratio)),
                                     "n_per_arm_1pct": n_per_arm(var_ratio, r, 0.01),
                                     "n_per_arm_2pct": n_per_arm(var_ratio, r, 0.02)}
    p = RUNS / "ab_power.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
