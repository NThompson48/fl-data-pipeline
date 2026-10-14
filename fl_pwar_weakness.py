"""
fl_pwar_weakness.py -- why is P(WAR>0) "weak" (Brier skill +2.3% on 312 pitchers, 2025 -> 2026),
and how much of that can any change to the projection fix?

A. Ceiling.  The production uncertainty law says Var(RA9 error) = 185.16 / IP, all of it
   single-season luck (the talent-error terms a, c fit to 0; 90% upper bounds 0.22 and 0.68).
   Simulate 2026 outcomes inside that world, using the real pitchers, projections, replacement level
   and realized innings, and ask what skill (Brier skill, AUC) P(WAR>0) would show
     (1) if the shipped P were exactly right (a = 0),
     (2) if talent were off by the largest SD the data allow (a = 0.22, 0.68), shipped P vs an ORACLE
         that knows each pitcher's true talent.
   If the real skill sits inside that range, the weakness is the target (one season of luck), not the model.
B. Skill on the scale that matters: IP-weighted accuracy of the RA9 forecast vs naive forecasts, and
   how P(WAR>0) separates pitchers in runs-vs-replacement terms (quintiles).
C. Where P(WAR>0) IS informative: AUC / skill by projected-innings bucket (known pre-season).
D. Stability of the RA9 K%/BB% slopes across the four transitions (the baseline is fit on one).

Usage:  python fl_pwar_weakness.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm, rankdata

import fl_innings_projection as ip
import fl_pwar_calibration as cal
import fl_ra9_trustmark_projection as ra9
import mlb_marcel as mm

B_LUCK = ra9.RA9_LUCK_VAR_COEF
N_SIM = 400


def auc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    n1, n0 = y.sum(), (1 - y).sum()
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def brier_skill(p, y):
    y = np.asarray(y); b0 = np.mean((y.mean() - y) ** 2)
    return 1 - np.mean((np.asarray(p) - y) ** 2) / b0


def part_a(d, repl):
    print("A. CEILING -- skill of P(WAR>0) inside the model's own uncertainty world")
    ip_act, pred, sd = d["ip"].to_numpy(), d["pred"].to_numpy(), d["sd"].to_numpy()
    P_ship = d["P"].to_numpy()
    print(f"   real data: n={len(d)}, base rate {d['y'].mean():.3f}, Brier skill {brier_skill(P_ship, d['y']):+.1%}, AUC {auc(P_ship, d['y']):.3f}")
    print(f"   spread of projected RA9 (SD, IP-weighted): {np.sqrt(np.cov(pred, aweights=ip_act)):.2f} runs/9;  typical luck SD at realized IP "
          f"(median {np.median(np.sqrt(B_LUCK / ip_act)):.2f}, at 100 IP {np.sqrt(B_LUCK/100):.2f})")
    rng = np.random.default_rng(1)
    rows = []
    for a in (0.0, 0.22, 0.68):
        bs_m, au_m, bs_o, au_o = [], [], [], []
        for _ in range(N_SIM):
            tau = pred + np.sqrt(a) * rng.standard_normal(len(pred))
            y = (tau + np.sqrt(B_LUCK / ip_act) * rng.standard_normal(len(pred)) < repl).astype(float)
            if y.min() == y.max():
                continue
            P_or = norm.cdf((repl - tau) / sd)
            bs_m.append(brier_skill(P_ship, y)); au_m.append(auc(P_ship, y))
            bs_o.append(brier_skill(P_or, y)); au_o.append(auc(P_or, y))
        rows.append({"talent error var a": a, "shipped P: Brier skill": np.mean(bs_m), "AUC": np.mean(au_m),
                     "oracle P (true talent): Brier skill": np.mean(bs_o), "oracle AUC": np.mean(au_o)})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    real = (brier_skill(P_ship, d["y"]), auc(P_ship, d["y"]))
    print(f"   -> real skill {real[0]:+.1%} / AUC {real[1]:.3f}. Inside the model's world a perfect-information oracle only reaches "
          f"{t['oracle P (true talent): Brier skill'].max():+.1%} (a=0.68, the loosest bound).")


def part_b(d, war):
    print("\nB. SKILL ON THE SCALE THAT MATTERS (IP-weighted, 2026 outcomes)")
    w = d["ip"].to_numpy(); y = d["target"].to_numpy()
    w25 = war[war["year"] == 2025].groupby("player")[["ra", "ip"]].sum()
    d = d.drop(columns=[c for c in ("ra9_25", "ipw_25") if c in d.columns]).merge(
        pd.DataFrame({"ra9_25": 9 * w25["ra"] / w25["ip"], "ipw_25": w25["ip"]}), left_on="player", right_index=True, how="left")
    lg26 = np.average(d["target"], weights=d["ip"])
    rows = []
    sub = d[d["ipw_25"] >= 20]
    for lab, df in (("all pitchers (n=%d)" % len(d), d), ("with >=20 IP in 2025 (n=%d)" % len(sub), sub)):
        ww = df["ip"].to_numpy(); yy = df["target"].to_numpy()
        rm = lambda f: float(np.sqrt(np.average((yy - f) ** 2, weights=ww)))
        r = {"sample": lab, "model RMSE": rm(df["pred"].to_numpy()), "league mean RMSE": rm(np.full(len(df), lg26)),
             "corr(model, actual)": float(np.corrcoef(df["pred"], yy)[0, 1])}
        if df["ra9_25"].notna().all() and len(df) == len(sub):
            r["last-year RA9 RMSE"] = rm(df["ra9_25"].to_numpy()); r["corr(last yr, actual)"] = float(np.corrcoef(df["ra9_25"], yy)[0, 1])
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    repl = ra9.LEAGUE_RA9_REPLACEMENT
    d["rvr"] = (repl - d["target"]) * d["ip"] / 9                      # realized runs vs replacement
    d["q"] = pd.qcut(d["P"], 5, labels=["Q1 (lowest P)", "Q2", "Q3", "Q4", "Q5 (highest P)"])
    t = d.groupby("q", observed=True).apply(lambda g: pd.Series({
        "n": len(g), "mean P": g["P"].mean(), "actual share beating repl (IP-wtd)": np.average(g["y"], weights=g["ip"]),
        "actual runs vs repl per pitcher": g["rvr"].mean(), "per 9 IP": 9 * g["rvr"].sum() / g["ip"].sum()}))
    print("   by predicted-probability quintile:"); print(t.round(3).to_string())


def part_c(d):
    print("\nC. WHERE P(WAR>0) SEPARATES PITCHERS: by PROJECTED innings (known before the season)")
    d = d.copy()
    d["ipb"] = pd.cut(d["ip_proj"], [0, 25, 45, 70, 1000], labels=["<25 proj IP", "25-45", "45-70", "70+"])
    rows = []
    for k, g in d.groupby("ipb", observed=True):
        if g["y"].nunique() < 2:
            continue
        rows.append({"projected IP": k, "n": len(g), "mean P": g["P"].mean(), "actual": g["y"].mean(),
                     "AUC": auc(g["P"], g["y"]), "Brier skill": brier_skill(g["P"], g["y"]), "SD of P": g["P"].std()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def part_d():
    print("\nD. STABILITY OF THE RA9 SLOPES ACROSS TRANSITIONS (raw prior-year K%, BB% -> next-year RA9; BF>=100 both years)")
    p = ip.build_panel(); tr = ip.transitions(p)
    f = tr[(tr["bf_t"] >= 100) & (tr["bf_next"] >= 100)].dropna(subset=["k_t", "bbr_t", "ra9_next"]).copy()
    rows = []
    for t, g in list(f.groupby("t")) + [("pooled", f)]:
        w = mm.reliability_weight(g["ip_next"], 146).values
        m = smf.wls("ra9_next ~ k_t + bbr_t", data=g, weights=w).fit()
        rows.append({"transition": f"{t}->{t+1}" if t != "pooled" else "pooled (4)", "n": len(g), "K% slope": m.params["k_t"], "se": m.bse["k_t"],
                     "BB% slope": m.params["bbr_t"], "se ": m.bse["bbr_t"], "R2": m.rsquared})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(f"   shipped (2025->2026, projected inputs): K% {ra9.RA9_SHIPPED_K if hasattr(ra9, 'RA9_SHIPPED_K') else -9.24}, BB% {11.53}")


if __name__ == "__main__":
    d, war = cal.main()
    print("\n" + "=" * 100)
    repl = ra9.LEAGUE_RA9_REPLACEMENT
    part_a(d, repl)
    part_b(d, war)
    part_c(d)
    part_d()
