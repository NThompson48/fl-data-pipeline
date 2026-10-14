"""
fl_hitter_bipct_babip_test.py -- is BABIP related to BIP% (balls in play per PA = (AB - K - HR + SF) / PA, the share of PA that end on a ball in play, HR excluded)?
Counting stats exist for 2022-26 (StatsCrew 2022-24 + Trackman-era 2025-26), so this uses five seasons and four transitions.
(1) Same-season correlation, hitter-seasons >= 150 PA (all five seasons), raw and within season (year-demeaned); (2) year-to-year persistence of BIP% vs BABIP and last year's BIP% vs next year's BABIP;
(3) forward / leave-one-transition-out: next-year BABIP = production BABIP regression (three seasons, n0 refit on the training transitions) + b x shrunk BIP% deviation (b fit on training transitions; the reference gets the same fitted
intercept), BIP% shrunk by n0 = 150 PA; weighted RMSE and MAE with hitter-cluster bootstrap intervals (reporting standard).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_depth3_test as d3
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore")
N0_BIP = 150.0


def main():
    p = dev.load(); p = p.copy(); p["bip"] = p["ab"] - p["so"] - p["hr"] + p["sf"].fillna(0); p["bippct"] = p["bip"] / p["pa"]; p["hrr"] = p["hr"] / p["pa"]
    t = p[(p["pa"] >= 150) & p["babip"].notna()].copy()
    for c in ("bippct", "babip", "k", "bbr", "hrr"):
        t[c + "_dm"] = t[c] - t.groupby("year")[c].transform("mean")
    print(f"(1) SAME SEASON: n = {len(t)} hitter-seasons >= 150 PA (2022-26); mean BIP% {t['bippct'].mean():.3f} (SD {t['bippct'].std():.3f}), mean BABIP {t['babip'].mean():.3f}")
    print("   corr with BABIP (year-demeaned):", {c: round(t["babip_dm"].corr(t[c + "_dm"]), 3) for c in ("bippct", "k", "bbr", "hrr")}, "| raw corr with BIP%:", round(t["babip"].corr(t["bippct"]), 3))
    print("   by year corr(BABIP, BIP%):", {int(y): round(g["babip"].corr(g["bippct"]), 3) for y, g in t.groupby("year")})
    rows = []
    for a in (2022, 2023, 2024, 2025):
        A = t[t.year == a].set_index("name_clean"); B = t[t.year == a + 1].set_index("name_clean"); j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        rows.append({"transition": f"{a}->{a + 1}", "n": len(j), "BIP% persistence": j["bippct_a"].corr(j["bippct_b"]), "BABIP persistence": j["babip_a"].corr(j["babip_b"]), "BIP%_t vs BABIP_t+1": j["bippct_a"].corr(j["babip_b"]), "BABIP_t vs BIP%_t+1": j["babip_a"].corr(j["bippct_b"])})
    print("(2) PERSISTENCE\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    q = p.copy(); q["woba"] = q["woba_n"]; d = d3.frame(q, "babip"); bip = p.drop_duplicates(["name_clean", "year"]).set_index(["name_clean", "year"])
    lg = {y: g["bip"].sum() / g["pa"].sum() for y, g in p.groupby("year")}; d["bipdev"] = [((bip["bip"].get((n, t_), np.nan) / bip["pa"].get((n, t_), np.nan) - lg[t_]) * bip["pa"].get((n, t_), np.nan) / (bip["pa"].get((n, t_), np.nan) + N0_BIP)) if (n, t_) in bip.index else np.nan for n, t_ in zip(d["name_clean"], d["t"])]
    d = d.dropna(subset=["bipdev"]); out = {"old": [], "new": [], "y": [], "w": [], "id": [], "fold": []}; coefs = []
    for scheme, plan in (("LEAVE-ONE-TRANSITION-OUT", [(tuple(x for x in (2022, 2023, 2024, 2025) if x != T), T) for T in (2022, 2023, 2024, 2025)]), ("FORWARD (tests 2024->25 and 2025->26)", [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)])):
        Y, O, N, W, I = [], [], [], [], []
        for tr, te in plan:
            dtr = d[d["t"].isin(tr)].copy(); dte = d[d["t"] == te].copy(); n0 = d3.fit_n0(dtr, 0.5, 3); dtr["base"] = d3.pred(dtr, n0, 0.5, 3); dte["base"] = d3.pred(dte, n0, 0.5, 3)
            A = np.column_stack([np.ones(len(dtr)), dtr["bipdev"]]); sw = np.sqrt(dtr["w"].to_numpy()); b = np.linalg.lstsq(A * sw[:, None], (dtr["x_next"] - dtr["base"]).to_numpy() * sw, rcond=None)[0]
            c0 = np.average((dtr["x_next"] - dtr["base"]).to_numpy(), weights=dtr["w"].to_numpy())
            Y.append(dte["x_next"].to_numpy()); O.append(dte["base"].to_numpy() + c0); N.append(dte["base"].to_numpy() + b[0] + b[1] * dte["bipdev"].to_numpy()); W.append(dte["w"].to_numpy()); I.append(dte["name_clean"].to_numpy()); coefs.append(round(float(b[1]), 3))
        y, o, n, w, ids = map(np.concatenate, (Y, O, N, W, I)); print(f"\n(3) {scheme}: n = {len(y)}; BIP% coefficient by fold {coefs[-len(plan):]} (BABIP per unit of shrunk BIP% deviation)")
        print(er.table([{"model": "BABIP regression + BIP% term (old = intercept only)", **er.compare(y, o, n, w, ids, n=500)}], 5))


if __name__ == "__main__":
    main()
