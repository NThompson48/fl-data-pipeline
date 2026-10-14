"""
fl_pwar_short_arm_test.py -- can anything fix the per-arm P(WAR>0) optimism for pitchers projected under 22 IP?  (user 2026-10-12: "P(WAR>0) should already capture uncertainty, but
calibration is a different issue. Anything else we can do about this?")

Backtest 2025 -> 2026 (fl_pwar_calibration, RETURN_ALL; out-of-sample RA9 predictions; P = Phi((repl - pred) / sqrt(185.16 / projected IP))).  Candidate fixes, each fitted on half the
pitchers and scored on the other half (2-fold by pitcher, 50 repeats; nothing is fitted on the scored rows):
  A  production (no change)
  B  global level shift: pred + mean miss of ALL arms (is it just the 2026 league level the backtest does not recentre?)
  C  short-arm level shift: pred + mean miss of arms projected < 22 IP (only those arms move)  -- a hierarchical "fringe-arm prior": arms with little history are, as a group, worse
  D  B + C (global shift plus an extra short-arm shift)
  E  widen: SD x k for short arms (k fitted by log loss) -- tests whether it is a spread problem rather than a level problem
Scored per arm (unweighted: the per-arm claim P(WAR>0) makes) on arms with >= 10 IP in 2026 (the data build's floor): Brier, log loss, calibration in the < 22 group; plus the innings-weighted
Brier (the roster-weighted view, where production is already calibrated) so a fix that helps per arm but hurts the innings-weighted view is visible.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

import fl_pwar_calibration as cal
import fl_ra9_trustmark_projection as ra9

warnings.filterwarnings("ignore")
cal.RETURN_ALL = True


def probs(d, shift_all=0.0, shift_short=0.0, k_short=1.0):
    short = (d["ip_proj"] < 22).to_numpy()
    pred = d["pred"].to_numpy() + shift_all + shift_short * short
    sd = np.sqrt(ra9.RA9_LUCK_VAR_COEF / d["ip_proj"].to_numpy()) * np.where(short, k_short, 1.0)
    return norm.cdf((d["repl"].to_numpy() - pred) / sd)


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    b, _ = cal.main()
    d = b[(b["ip"] >= 10) & b["pred"].notna() & b["target"].notna()].copy().reset_index(drop=True)
    d["repl"] = ra9.LEAGUE_RA9_REPLACEMENT; d["miss"] = d["target"] - d["pred"]; y = (d["target"] < d["repl"]).astype(float).to_numpy()
    short = (d["ip_proj"] < 22).to_numpy(); w_ip = d["ip"].to_numpy()
    print(f"n = {len(d)} arms (>= 10 IP); short (< 22 projected IP) {short.sum()}; mean RA9 miss all {d['miss'].mean():+.3f}, short {d.loc[short, 'miss'].mean():+.3f}, others {d.loc[~short, 'miss'].mean():+.3f}")
    rng = np.random.default_rng(17); R = 50
    names = ["A production", "B global shift", "C short-arm shift", "D global + short shift", "E widen short-arm SD"]
    P = {n: np.zeros((R, len(d))) for n in names}; par = {n: [] for n in names}
    for r in range(R):
        half = rng.random(len(d)) < .5
        for tr, te in ((half, ~half), (~half, half)):
            t = d[tr]; ts = (t["ip_proj"] < 22).to_numpy()
            g = t["miss"].mean(); s_short = t.loc[ts, "miss"].mean(); s_extra = t.loc[ts, "miss"].mean() - g
            yt = (t["target"] < t["repl"]).astype(float).to_numpy()
            k = minimize_scalar(lambda k: ll(yt[ts], probs(t[ts], k_short=k)).mean(), bounds=(.5, 4), method="bounded").x
            dt = d[te]
            P["A production"][r, te] = probs(dt)
            P["B global shift"][r, te] = probs(dt, shift_all=g)
            P["C short-arm shift"][r, te] = probs(dt, shift_short=s_short)
            P["D global + short shift"][r, te] = probs(dt, shift_all=g, shift_short=s_extra)
            P["E widen short-arm SD"][r, te] = probs(dt, k_short=k)
            par["B global shift"].append(g); par["C short-arm shift"].append(s_short); par["D global + short shift"].append(s_extra); par["E widen short-arm SD"].append(k)
    rows = []
    base = {"brier": ((P["A production"] - y) ** 2).mean(1), "ll": ll(y, P["A production"]).mean(1)}
    for n in names:
        br = ((P[n] - y) ** 2).mean(1); lo = ll(y, P[n]).mean(1); brs = ((P[n][:, short] - y[short]) ** 2).mean(1)
        brw = np.array([np.average((P[n][r] - y) ** 2, weights=w_ip) for r in range(R)])
        rows.append({"model": n, "fitted value (mean)": np.mean(par[n]) if par[n] else np.nan, "Brier": br.mean(), "log loss": lo.mean(), "Brier short arms": brs.mean(),
                     "Brier IP-weighted": brw.mean(), "short: pred P": P[n][:, short].mean(), "short: observed": y[short].mean(),
                     "45+ proj IP: pred P": P[n][:, (d['ip_proj'] >= 45).to_numpy()].mean(), "45+: observed": y[(d['ip_proj'] >= 45).to_numpy()].mean()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    # bootstrap the per-arm Brier difference vs production (pitchers resampled, cross-fitted probabilities averaged over repeats)
    bs = np.random.default_rng(4); Pm = {n: P[n].mean(0) for n in names}
    for n in names[1:]:
        dd = []
        for _ in range(2000):
            i = bs.integers(0, len(d), len(d)); dd.append(np.mean((Pm[n][i] - y[i]) ** 2) - np.mean((Pm["A production"][i] - y[i]) ** 2))
        dd = np.array(dd); print(f"  {n:24s} Brier change {np.mean(dd):+.4f}  90% [{np.percentile(dd, 5):+.4f}, {np.percentile(dd, 95):+.4f}]  P(better) {np.mean(dd < 0):.2f}")


if __name__ == "__main__":
    main()
