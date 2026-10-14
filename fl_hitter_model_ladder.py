"""
fl_hitter_model_ladder.py -- how much does each layer of the hitter wOBA model buy?

Rungs (park-neutral wOBA; next-year target with >= 100 PA; league level of the target year treated as known for every rung, so drift does not decide the ranking):
  0  league average
  1  prior year at face value          next = this year's wOBA   (raw, and shifted for the league-level change)
  2  naive shrinkage                   lg + n (x - lg) / (n + n0),  ONE season, one n0 (fit on the other transitions)
  3  current model, no Trackman        two seasons (prior weight .75) regressed wOBA, 50/50 with wOBA rebuilt from regressed K%, BB%, HR/PA, BABIP
  4  current model, with Trackman      3 + 0.485 x (xwOBA - 3) + 0.00377 x shrunk EV90 deviation (mph) x n0w / (PA_eff + n0w)       [2025 -> 2026 only]
Level 1 = all four transitions (2022->23 ... 2025->26), leave-one-transition-out; Level 2 = 2025 -> 2026 rows, where Trackman exists, all rungs on the SAME rows.
Standard inverse-variance weights (frozen from the current model's residuals).  Every rung is shown with and without a constant shift (fit out of sample), because the
targets are hitters who reached 100 PA and sit above the league mean.  Differences carry a paired bootstrap over hitters.

Usage:  python fl_hitter_model_ladder.py
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

import fl_hitter_baseline_backtest as bt
import fl_hitter_contact_vars_test as cv
import fl_hitter_ev as hev
import fl_hitter_model_dev as dev
import fl_hitter_wobacon_test as wc

warnings.filterwarnings("ignore")
rng = np.random.default_rng(4)
GAMMA, B_EV, N0W = 0.485, 0.00377, 440.0


def wrmse(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def wr2(y, p, w, ref):
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - ref) ** 2)


def boot(y, A, B, w, groups, n=1000):
    """Paired bootstrap over hitters of RMSE(A) - RMSE(B)."""
    ug = pd.Series(groups).unique(); by = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array); out = []
    for _ in range(n):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); out.append(wrmse(y[ii], A[ii], w[ii]) - wrmse(y[ii], B[ii], w[ii]))
    return float(np.mean(out)), float(np.percentile(out, 5)), float(np.percentile(out, 95))


def shift_loto(t, y, p, w):
    """Add a constant fit on the OTHER transitions (weighted mean error) -- same free parameter for every rung."""
    out = p.copy()
    for tt in np.unique(t):
        m = t == tt; c = np.average((y - p)[~m], weights=w[~m]); out[m] = p[m] + c
    return out


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    with contextlib.redirect_stdout(io.StringIO()):
        J, q2, xc, _w = wc.main()
    d = bt.transitions(q, "woba")[["name_clean", "t", "x_t", "n_t", "x_prev", "n_prev", "lg_t"]]
    J = J.merge(d, on=["name_clean", "t"], how="left")
    E1, _ = bt.loto(bt.transitions(q, "woba"), 0.0)             # naive: one season
    J = J.merge(E1[["name_clean", "t", "pred"]].rename(columns={"pred": "naive"}), on=["name_clean", "t"], how="left")
    y = J["x_next"].to_numpy(); w = J["w"].to_numpy(); t = J["t"].to_numpy(); g = J["name_clean"].to_numpy()
    preds = {"0 league average": J["lg_next"].to_numpy(), "1a prior year, face value": J["x_t"].to_numpy(), "1b prior year + league-level shift": (J["x_t"] + J["lg_next"] - J["lg_t"]).to_numpy(),
             "2 naive shrinkage (1 season)": J["naive"].to_numpy(), "3 current model, no Trackman": J["shipped_blend"].to_numpy()}
    ref = preds["0 league average"]
    print(f"LEVEL 1: all four transitions, n = {len(J)} hitter-transitions (next-year PA >= 100), scored on identical rows")
    rows = []
    for k, pr in preds.items():
        ps = shift_loto(t, y, pr, w)
        rows.append({"rung": k, "RMSE": wrmse(y, pr, w), "R2 vs league avg": wr2(y, pr, w, ref), "mean error": float(np.average(y - pr, weights=w)), "RMSE with out-of-sample constant": wrmse(y, ps, w)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    n0_naive = bt.fit_n0(bt.transitions(q, "woba"), 0.0)
    print(f"   naive shrinkage n0 (all transitions) = {n0_naive:.0f} PA: a 300 PA hitter keeps {300/(300+n0_naive):.0%} of his deviation from the league; a 100 PA hitter {100/(100+n0_naive):.0%}")
    print("   paired bootstrap of RMSE differences (90% interval):")
    for a, b in (("1b prior year + league-level shift", "1a prior year, face value"), ("2 naive shrinkage (1 season)", "1b prior year + league-level shift"), ("3 current model, no Trackman", "2 naive shrinkage (1 season)")):
        m, lo, hi = boot(y, preds[a], preds[b], w, g)
        print(f"     {a:36s} minus {b:36s}: {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]")
    # by prior-season sample size
    J["bk"] = pd.cut(J["n_t"], [0, 100, 250, 10 ** 5], labels=["< 100 PA", "100-249", "250+"])
    print("\n   by the hitter's PA in the prior season (RMSE; mean error in parentheses):")
    rows = []
    for lab, m in (("< 100 PA", J["bk"] == "< 100 PA"), ("100-249", J["bk"] == "100-249"), ("250+", J["bk"] == "250+")):
        mm = m.to_numpy(); r = {"prior-season PA": lab, "n": int(mm.sum())}
        for k, pr in preds.items():
            r[k.split(" ", 1)[0] + " " + k.split(" ", 1)[1][:18]] = f"{wrmse(y[mm], pr[mm], w[mm]):.4f} ({np.average((y - pr)[mm], weights=w[mm]):+.4f})"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------------------------------------------------------------ Level 2
    d2 = cv.flags(pd.read_parquet(cv.BBE)); s = cv.season_vars(d2)
    T = J[J["t"] == 2025].copy()
    a = s[(s["year"] == 2025)].set_index("name_clean")[["n", "ev90"]].rename(columns={"n": "bbe", "ev90": "ev90_25"})
    qi = q2[q2["year"] == 2025].set_index("name_clean")
    T = T.join(a, on="name_clean"); T["xw25"] = T["name_clean"].map(qi["xwoba"]); T["pa25"] = T["name_clean"].map(qi["pa"])
    T = T.dropna(subset=["xw25"])
    lgx = np.average(T["xw25"], weights=T["pa25"]); xs = (T["pa25"] * T["xw25"] + 200 * lgx) / (T["pa25"] + 200)
    blend = T["shipped_blend"].to_numpy()
    nudged = blend + GAMMA * (xs.to_numpy() - blend)
    lg_ev = 100.17
    evdev = np.where(T["ev90_25"].isna(), 0.0, (T["ev90_25"].to_numpy() - lg_ev) * T["bbe"].fillna(0).to_numpy() / (T["bbe"].fillna(0).to_numpy() + hev.N0_EV90))
    pa_eff = (T["n_t"] + 0.75 * T["n_prev"]).to_numpy()
    full = nudged + B_EV * evdev * N0W / (pa_eff + N0W)
    y2 = T["x_next"].to_numpy(); w2 = T["w"].to_numpy(); g2 = T["name_clean"].to_numpy()
    p2 = {"0 league average": T["lg_next"].to_numpy(), "1a prior year, face value": T["x_t"].to_numpy(), "1b prior year + league-level shift": (T["x_t"] + T["lg_next"] - T["lg_t"]).to_numpy(),
          "2 naive shrinkage (1 season)": T["naive"].to_numpy(), "3 current model, no Trackman": blend, "3b + xwOBA nudge": nudged, "4 current model + xwOBA nudge + EV90 term": full}
    print(f"\nLEVEL 2: 2025 -> 2026 only (Trackman exists), n = {len(T)} hitters, all rungs on the same rows; the constants of rung 4 (gamma, B) were fit on within-season data, not on these rows")
    rows = []
    for k, pr in p2.items():
        shifted = pr + np.average(y2 - pr, weights=w2)                 # in-sample constant (one parameter, same for every rung): upper bound on what a shift buys
        rows.append({"rung": k, "RMSE": wrmse(y2, pr, w2), "R2 vs league avg": wr2(y2, pr, w2, p2["0 league average"]), "mean error": float(np.average(y2 - pr, weights=w2)), "RMSE with constant (in-sample)": wrmse(y2, shifted, w2)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("   paired bootstrap of RMSE differences (90% interval):")
    for a_, b_ in (("2 naive shrinkage (1 season)", "1b prior year + league-level shift"), ("3 current model, no Trackman", "2 naive shrinkage (1 season)"), ("3b + xwOBA nudge", "3 current model, no Trackman"),
                   ("4 current model + xwOBA nudge + EV90 term", "3b + xwOBA nudge"), ("4 current model + xwOBA nudge + EV90 term", "2 naive shrinkage (1 season)")):
        m, lo, hi = boot(y2, p2[a_], p2[b_], w2, g2)
        print(f"     {a_:44s} minus {b_:36s}: {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]")
    print(f"   EV90 term on this transition: SD {np.std(B_EV * evdev * N0W / (pa_eff + N0W)):.4f}; xwOBA nudge SD {np.std(nudged - blend):.4f}; blend SD {np.std(blend):.4f}; actual SD of next-year wOBA {np.std(y2):.4f}")


if __name__ == "__main__":
    main()
