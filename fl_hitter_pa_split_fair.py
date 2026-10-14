"""
fl_hitter_pa_split_fair.py -- the within-season PA-split comparisons redone with a FAIR reference.

Flaw found in fl_hitter_pa_split_test.py / fl_hitter_lowhistory_test.py / the EV90 coefficient fit: every add-on model was cross-validated WITH a fitted intercept (about +.002 to +.009 wOBA), while the
base (regressed wOBA) and the one-parameter xwOBA nudge had none.  The intercept is not information about the hitter: the targets are hitters who reached 150 PA, who are better than the league
average the base regresses to (survivor selection), so a constant shift lowers error for any model that is allowed one.  Part of the reported "gains" was that constant.

Here EVERY model, including the reference, gets the same cross-validated intercept.  Reported: how much the intercept alone buys, and each add-on's change vs the intercept-only reference.

Usage:  python fl_hitter_pa_split_fair.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_ev as hev
import fl_hitter_pa_split_test as ps

warnings.filterwarnings("ignore")
SIZES = (30, 60, 120, 200)
rng = np.random.default_rng(21)
N0_EV90 = hev.N0_EV90


def run():
    d = ps.prep()
    ps.SIZES = SIZES
    D = ps.draws(d)
    t = d[d["tracked"]].groupby(["batter", "year"])["ev"].agg(n="size", ev90=lambda s: np.quantile(s, 0.9)).reset_index()
    lgv = t[t["n"] >= 50].groupby("year")["ev90"].mean()
    out = []
    for m in SIZES:
        S = D[D["m"] == m].reset_index(drop=True)
        y = S["y"].to_numpy(); base = (S["lg_w"] + m * (S["w_in"] - S["lg_w"]) / (m + ps.N0_WOBA)).to_numpy()
        wt, _, _ = ew.weights_from_base(y, base, S["n_out"].to_numpy())
        xs = (m * S["xw_in"] + ps.N0_X * S["lg_x"]) / (m + ps.N0_X); zx = (xs.to_numpy() - base)
        lg = S["year"].map(lgv).to_numpy()
        x = np.where(S["ev90"].isna(), 0.0, (S["ev90"].to_numpy() - lg) * S["bbe"].to_numpy() / (S["bbe"].to_numpy() + N0_EV90))
        xp = x * ps.N0_WOBA / (m + ps.N0_WOBA)
        # PullAir shrunk and standardised for the pair
        pa_ = np.where(S["pull_air"].isna(), 0.0, (S["pull_air"].to_numpy() - S["lg_pull_air"].to_numpy()) * S["bbe"].to_numpy() / (S["bbe"].to_numpy() + ps.N0_VAR["pull_air"])) if "lg_pull_air" in S else None
        groups = (S["batter"] + "|" + S["year"].astype(str)).to_numpy()
        models = {"intercept only (reference)": np.zeros((len(y), 0)), "xwOBA nudge": zx[:, None], "EV90 (prior-weighted)": xp[:, None], "nudge + EV90 (production form)": np.column_stack([zx, xp])}
        if pa_ is not None:
            models["nudge + EV90 + PullAir"] = np.column_stack([zx, xp, pa_ * ps.N0_WOBA / (m + ps.N0_WOBA)])
        pr = {"raw base (no intercept)": base}
        for k, X in models.items():
            pr[k] = ps.cv_group(y, base, X, wt, groups)
        ub = pd.Series(groups).unique(); idx_by = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array)
        names = list(pr)
        bs = {k: [] for k in names}
        for _ in range(300):
            ii = np.concatenate([idx_by[u] for u in rng.choice(ub, len(ub))])
            r0 = ps.rm(y[ii], pr["intercept only (reference)"][ii], wt[ii])
            for k in names:
                bs[k].append(ps.rm(y[ii], pr[k][ii], wt[ii]) - r0)
        for k in names:
            out.append({"m": m, "model": k, "vs intercept-only": np.mean(bs[k]), "lo": np.percentile(bs[k], 5), "hi": np.percentile(bs[k], 95)})
    R = pd.DataFrame(out)
    print("PART 1: EV90 and the nudge, fair reference (every model has the same CV intercept).  Change in wOBA RMSE vs 'intercept only' (negative = better); 90% bootstrap interval over hitters")
    for m in SIZES:
        r = R[R["m"] == m]
        print(f"\n  known sample {m} PA")
        print(r.drop(columns="m").assign(interval=lambda x: [f"[{a:+.5f}, {b:+.5f}]" for a, b in zip(x['lo'], x['hi'])]).drop(columns=["lo", "hi"]).round(5).to_string(index=False))
    return R


if __name__ == "__main__":
    run()
