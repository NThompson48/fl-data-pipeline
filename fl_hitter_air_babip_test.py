"""
fl_hitter_air_babip_test.py -- is Air% (share of batted balls with launch angle >= 10 deg) related to BABIP?
(1) Same-season correlation, hitter-seasons with >= 150 PA and >= 60 batted balls (2025 and 2026), raw and partial (controlling for K%, HR/PA, GB%); BABIP excludes HR, so a fly-ball hitter has fewer balls in play on the ground.
(2) Is it informative about NEXT year's BABIP beyond the regressed BABIP?  Forward 2025 -> 2026 (production BABIP regression, n0 refit on earlier transitions; Air% shrunk by n0 = 89 batted balls, 2025 balls only),
    BABIP_next - base regressed on the shrunk Air% deviation, 10-fold CV with the SAME free intercept in the reference; hitter-cluster bootstrap; weighted RMSE and MAE (reporting standard).
(3) Reliability of Air% vs BABIP (random-half split) so the comparison is not just noise.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import fl_eval_report as er
import fl_hitter_baseline_backtest as bt
import fl_hitter_depth3_test as d3
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore"); rng = np.random.default_rng(12)
N0_AIR = 89.0


def cvfit(y, base, X, w, reps=15):
    pr = np.zeros(len(y))
    for rep in range(reps):
        for tr, te in KFold(10, shuffle=True, random_state=rep).split(y):
            A = np.column_stack([np.ones(len(tr))] + ([X[tr]] if X is not None else [])); B = np.column_stack([np.ones(len(te))] + ([X[te]] if X is not None else []))
            b = np.linalg.lstsq(A * np.sqrt(w[tr])[:, None], (y - base)[tr] * np.sqrt(w[tr]), rcond=None)[0]; pr[te] += (base[te] + B @ b) / reps
    return pr


def main():
    p = dev.load(); t = p[(p["year"] >= 2025) & (p["pa"] >= 150) & (p["bbe"] >= 60)].copy(); t["air"] = t["air_n"] / t["bbe"]; t["hrr"] = t["hr"] / t["pa"]
    print("(1) SAME-SEASON relationship, hitter-seasons >= 150 PA and >= 60 batted balls: n =", len(t))
    cols = ["air", "gb_pct", "k", "hrr", "babip"]; print(t[cols].corr().round(3).to_string())
    X = t[["air", "k", "hrr", "gb_pct"]].dropna(); y = t.loc[X.index, "babip"]; Z = np.column_stack([np.ones(len(X)), (X - X.mean()) / X.std()]); b = np.linalg.lstsq(Z, y.to_numpy(), rcond=None)[0]
    res = y.to_numpy() - Z @ b; se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * res.var(ddof=Z.shape[1]))
    print("  BABIP on standardised [Air%, K%, HR/PA, GB%]: per-SD effects", dict(zip(["Air%", "K%", "HR/PA", "GB%"], [f"{b[i + 1]:+.4f} (se {se[i + 1]:.4f})" for i in range(4)])))
    print("(3) RELIABILITY: see the year-to-year correlations below")
    a25 = t[t.year == 2025].set_index("name_clean"); a26 = t[t.year == 2026].set_index("name_clean"); j = a25.join(a26, lsuffix="_25", rsuffix="_26", how="inner")
    print(f"  2025 -> 2026 correlation (n = {len(j)}): Air% {j['air_25'].corr(j['air_26']):.2f}, BABIP {j['babip_25'].corr(j['babip_26']):.2f}, GB% {j['gb_pct_25'].corr(j['gb_pct_26']):.2f}; last year's Air% vs next year's BABIP {j['air_25'].corr(j['babip_26']):+.2f}")
    # (2) forward predictive test
    q = p.copy(); q["woba"] = q["woba_n"]; d = d3.frame(q, "babip"); n0 = d3.fit_n0(d[d["t"].isin((2022, 2023, 2024))], 0.5, 3)
    te = d[d["t"] == 2025].copy(); te["base"] = d3.pred(te, n0, 0.5, 3); lgair = p[p.year == 2025]["air_n"].sum() / p[p.year == 2025]["bbe"].sum()
    s25 = p[p.year == 2025].drop_duplicates("name_clean").set_index("name_clean"); bb = s25["bbe"].reindex(te["name_clean"]).to_numpy(); an = s25["air_n"].reindex(te["name_clean"]).to_numpy()
    dev_air = np.where(np.isnan(bb) | (bb <= 0), 0.0, (an / np.where(bb > 0, bb, 1) - lgair) * bb / (bb + N0_AIR)); ok = ~np.isnan(bb) & (bb > 0)
    te = te[ok]; dev_air = dev_air[ok]; y = te["x_next"].to_numpy(); w = te["w"].to_numpy(); base = te["base"].to_numpy(); ids = te["name_clean"].to_numpy()
    ref = cvfit(y, base, None, w); new = cvfit(y, base, dev_air[:, None], w)
    b1 = np.linalg.lstsq(np.column_stack([np.ones(len(y)), dev_air]) * np.sqrt(w)[:, None], (y - base) * np.sqrt(w), rcond=None)[0]
    print(f"\n(2) FORWARD 2025 -> 2026 (n = {len(y)} hitters with Trackman 2025 batted balls and >= 100 AB in 2026): coefficient of next-year BABIP on shrunk Air% deviation = {b1[1]:+.3f} per unit (one SD of the deviation = {np.std(dev_air):.3f}: {b1[1] * np.std(dev_air):+.4f} BABIP)")
    print(er.table([{"model": "BABIP regression + Air% term (old = same regression, intercept only)", **er.compare(y, ref, new, w, ids)}], 5))


if __name__ == "__main__":
    main()
