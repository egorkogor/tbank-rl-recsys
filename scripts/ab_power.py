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
    out = {"users_active_week": int(act.sum())}
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
