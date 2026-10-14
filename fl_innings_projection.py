"""
fl_innings_projection.py -- project 2027 innings pitched (workload) per pitcher
and re-validate the RA9 luck-variance law on FIVE seasons of FL history
(StatsCrew 2022-24 + war_pitchers 2025-26 = four year-to-year transitions)
instead of the single 2025 -> 2026 transition the P(WAR>0) uncertainty was
originally calibrated on.

WAR is roughly (replacement RA9 - RA9) x innings, and the P(WAR>0) uncertainty
is single-season luck over the innings actually thrown (Var = b / IP), yet the
production code uses the pitcher's LAST-season IP as the next-season workload.

Part 1  How well does prior-season IP predict next-season IP (conditional on
        pitching again), how much does it regress, is the relationship stable
        across transitions, and do two prior seasons beat one? Leave-one-
        transition-out (fit on the other three, score the held-out one).
Part 2  Retention: P(pitches in the league again | prior-season IP).
Part 3  RA9 forecast errors across all four transitions (K%/BB%-only RA9 model,
        each transition's model fit on the OTHER transitions): is Var(error)
        = b/IP stable, and how does it change when the unknown future IP is
        replaced by the PROJECTED IP?
Part 4  Production parameters.

Usage:  python fl_innings_projection.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.stats import norm
from sqlalchemy import text

import fl_statscrew_pull as sc
import mlb_marcel as mm
from fl_supabase import read_sql_df

YEARS = [2022, 2023, 2024, 2025, 2026]


def lastfirst_to_clean(n: str) -> str:
    if "," in n:
        last, first = n.split(",", 1)
        n = f"{first.strip()} {last.strip()}"
    return sc.clean_name(n)


def build_panel() -> pd.DataFrame:
    s = pd.read_parquet("output/fl_statscrew_pitching_r_hr_2022_2024.parquet")[["name_clean", "year", "ip", "bf", "so", "bb", "r"]]
    w = read_sql_df(text("select player, year, g, gs, ip, bf, so, bb, ra from war_pitchers where game_type='reg_season'"))
    w["name_clean"] = w["player"].apply(lastfirst_to_clean)
    w = w.rename(columns={"ra": "r"})[["name_clean", "year", "g", "gs", "ip", "bf", "so", "bb", "r"]]
    p = pd.concat([s, w], ignore_index=True)
    p = p.groupby(["name_clean", "year"], as_index=False).agg(ip=("ip", "sum"), bf=("bf", "sum"), so=("so", "sum"),
                                                              bb=("bb", "sum"), r=("r", "sum"), g=("g", "sum"), gs=("gs", "sum"))
    p["ra9"] = 9 * p["r"] / p["ip"].replace(0, np.nan)
    p["k"] = p["so"] / p["bf"].replace(0, np.nan); p["bbr"] = p["bb"] / p["bf"].replace(0, np.nan)
    return p


def transitions(p: pd.DataFrame) -> pd.DataFrame:
    """One row per (pitcher, t): prior-season stats; next-season stats (NaN if absent)."""
    rows = []
    for t in YEARS[:-1]:
        a = p[p["year"] == t].set_index("name_clean")
        b = p[p["year"] == t + 1].set_index("name_clean")
        prev = p[p["year"] == t - 1].set_index("name_clean") if t - 1 in YEARS else None
        d = a[["ip", "bf", "k", "bbr", "ra9", "g", "gs"]].add_suffix("_t").copy()
        d["ip_prev"] = prev["ip"].reindex(d.index) if prev is not None else np.nan
        d["ip_next"] = b["ip"].reindex(d.index)
        d["ra9_next"] = b["ra9"].reindex(d.index)
        d["bf_next"] = b["bf"].reindex(d.index)
        d["k_next"] = b["k"].reindex(d.index); d["bb_next"] = b["bbr"].reindex(d.index)
        d["t"] = t
        rows.append(d.reset_index())
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
def part1(tr):
    d = tr.dropna(subset=["ip_next"]).copy()
    print("PART 1 -- next-season IP given the pitcher pitches again")
    print(f"  retained pitcher-transitions: {len(d)} across t={sorted(d['t'].unique())}; "
          f"median prior IP {d['ip_t'].median():.0f}, median next IP {d['ip_next'].median():.0f}")
    print("  per-transition OLS  ip_next = a + b*ip_prior :")
    for t, g in d.groupby("t"):
        m = smf.ols("ip_next ~ ip_t", data=g).fit()
        print(f"    {t}->{t+1}: n={len(g)} a={m.params['Intercept']:.1f} b={m.params['ip_t']:.2f} (se {m.bse['ip_t']:.2f}) R2={m.rsquared:.2f}")
    specs = {"persistence (next = prior)": None, "league mean": "ip_next ~ 1",
             "linear shrink: a + b*IP": "ip_next ~ ip_t",
             "sqrt-scale: sqrt(next) ~ sqrt(prior)": "SQRT",
             "+ prior-prior season": "ip_next ~ ip_t + ip_prev"}
    rows = []
    for name, f in specs.items():
        errs = []; r2n = r2d = 0
        for t in sorted(d["t"].unique()):
            tr_, te_ = d[d["t"] != t], d[d["t"] == t].copy()
            if name == "+ prior-prior season":
                tr_ = tr_.dropna(subset=["ip_prev"]); te_ = te_.dropna(subset=["ip_prev"])
                if len(te_) == 0:
                    continue
            if f is None:
                pred = te_["ip_t"].values
            elif f == "SQRT":
                m = smf.ols("np.sqrt(ip_next) ~ np.sqrt(ip_t)", data=tr_).fit()
                pred = np.maximum(m.predict(te_).values, 0) ** 2
            else:
                pred = smf.ols(f, data=tr_).fit().predict(te_).values
            e = te_["ip_next"].values - pred
            errs.append(pd.DataFrame({"t": t, "e": e, "ip_t": te_["ip_t"].values, "ip_next": te_["ip_next"].values}))
        E = pd.concat(errs)
        rows.append({"model": name, "n": len(E), "RMSE": round(np.sqrt((E["e"] ** 2).mean()), 1), "MAE": round(E["e"].abs().mean(), 1),
                     "bias": round(E["e"].mean(), 1),
                     "R2 vs league mean": round(1 - (E["e"] ** 2).sum() / ((E["ip_next"] - E["ip_next"].mean()) ** 2).sum(), 3),
                     "MAE (prior IP>=40)": round(E.loc[E["ip_t"] >= 40, "e"].abs().mean(), 1)})
    print("  leave-one-transition-out accuracy (fit on the other transitions):")
    print(pd.DataFrame(rows).to_string(index=False))
    m = smf.ols("ip_next ~ ip_t", data=d).fit()
    print(f"  pooled fit: ip_next = {m.params['Intercept']:.1f} + {m.params['ip_t']:.3f} * ip_prior   (resid SD {np.sqrt(m.scale):.1f}, R2 {m.rsquared:.2f})")
    # role (only 2025->2026 has g/gs)
    r = d[(d["t"] == 2025)].dropna(subset=["g_t", "gs_t"]).copy()
    r["sp_share"] = r["gs_t"] / r["g_t"].replace(0, np.nan)
    m2 = smf.ols("ip_next ~ ip_t + sp_share", data=r.dropna(subset=["sp_share"])).fit()
    print(f"  role check (2025->2026 only, n={int(m2.nobs)}): ip_next ~ ip_t + SP-share: sp_share coef {m2.params['sp_share']:+.1f} IP (t={m2.tvalues['sp_share']:+.2f}), "
          f"R2 {m.rsquared:.2f} -> {m2.rsquared:.2f} (same-sample base R2 "
          f"{smf.ols('ip_next ~ ip_t', data=r.dropna(subset=['sp_share'])).fit().rsquared:.2f})")
    return m


def part2(tr):
    d = tr[tr["ip_t"] > 0].copy(); d["ret"] = d["ip_next"].notna().astype(int)
    print("\nPART 2 -- retention: P(pitches again | prior IP)")
    d["bucket"] = pd.cut(d["ip_t"], [0, 10, 25, 50, 80, 120, 1000])
    print(d.groupby("bucket", observed=True)["ret"].agg(["size", "mean"]).round(2).T.to_string())
    m = smf.logit("ret ~ np.log(ip_t)", data=d).fit(disp=0)
    print(f"  logit: P(return) = sigmoid({m.params['Intercept']:.2f} + {m.params['np.log(ip_t)']:.2f} * ln(IP)), n={len(d)}")
    by_t = d.groupby("t")["ret"].mean().round(3).to_dict(); print("  overall retention by transition:", by_t)
    return m


def fit_variance(e, ip):
    e, ip = np.asarray(e), np.asarray(ip)
    def nll(p):
        v = p[0] + p[1] / ip
        return 0.5 * np.sum(np.log(v) + e ** 2 / v)
    best = min((minimize(nll, x0, bounds=[(1e-6, None)] * 2, method="L-BFGS-B") for x0 in ([0.3, 100.], [1.0, 20.], [0.05, 200.])), key=lambda r: r.fun)
    return best.x


def part3(tr, ipm, retm):
    print("\nPART 3 -- RA9 luck law across four transitions (K%/BB%-only RA9 model, leave-one-transition-out)")
    d = tr.copy()
    lgk = (d["k_t"] * d["bf_t"]).sum() / d["bf_t"].sum(); lgb = (d["bbr_t"] * d["bf_t"]).sum() / d["bf_t"].sum()
    N0 = 60.0
    d["K_s"] = ((d["k_t"] * d["bf_t"]).fillna(0) + N0 * lgk) / (d["bf_t"].fillna(0) + N0)
    d["B_s"] = ((d["bbr_t"] * d["bf_t"]).fillna(0) + N0 * lgb) / (d["bf_t"].fillna(0) + N0)
    d["K_raw"], d["B_raw"] = d["k_t"], d["bbr_t"]
    fit = d[(d["bf_t"] >= 100) & (d["bf_next"] >= 100)].copy()
    out = []
    for t in sorted(d["t"].unique()):
        trn = fit[fit["t"] != t]
        w = mm.reliability_weight(trn["ip_next"], 146).values
        m = smf.wls("ra9_next ~ K_raw + B_raw", data=trn, weights=w).fit()
        te = d[(d["t"] == t) & (d["ip_next"] >= 10)].copy()
        te["pred"] = m.predict(te.assign(K_raw=te["K_s"], B_raw=te["B_s"])).values
        te["e"] = te["ra9_next"] - te["pred"]
        out.append(te)
    E = pd.concat(out)
    E["ip_hat"] = np.maximum(ipm.predict(E.assign(ip_t=E["ip_t"].fillna(0))).values, 5.0)
    E.loc[E["ip_t"].isna(), "ip_hat"] = 31.0   # debut: median IP of pitchers new to the league
    print(f"  broad sample n={len(E)} (IP_next>=10): errors mean {E['e'].mean():+.2f}")
    print("  per-transition fit  Var(e) = a + b/IP_actual:")
    rows = []
    for t, g in E.groupby("t"):
        a, b = fit_variance(g["e"], g["ip_next"])
        rows.append({"transition": f"{t}->{t+1}", "n": len(g), "a": round(a, 3), "b": round(b, 1), "z-SD at pooled b": np.nan})
    a_all, b_all = fit_variance(E["e"], E["ip_next"])
    for r in rows:
        t = int(r["transition"][:4]); g = E[E["t"] == t]
        r["z-SD at pooled b"] = round((g["e"] / np.sqrt(b_all / g["ip_next"])).std(), 2)
    print(pd.DataFrame(rows).to_string(index=False)); print(f"  pooled: a={a_all:.3f}  b={b_all:.1f}   (shipped b=185.16 from the single 2025->2026 transition)")
    # same law with PROJECTED innings in the denominator
    print("  with PROJECTED innings (what production will actually have) instead of actual IP:")
    for lab, ipcol in (("actual IP", "ip_next"), ("projected IP", "ip_hat")):
        a2, b2 = fit_variance(E["e"], E[ipcol])
        z = E["e"] / np.sqrt(b2 / E[ipcol])
        pit = norm.cdf(z)
        print(f"    {lab:13s}: refit b={b2:7.1f} (a={a2:.2f}); z-SD={z.std():.2f}; coverage 50/80/90% = "
              f"{((pit>.25)&(pit<.75)).mean():.2f}/{((pit>.10)&(pit<.90)).mean():.2f}/{((pit>.05)&(pit<.95)).mean():.2f}")
    z = E["e"] / np.sqrt(185.16 / E["ip_hat"]); pit = norm.cdf(z)
    print(f"    shipped b=185.16 with projected IP: z-SD={z.std():.2f}; coverage 50/80/90% = "
          f"{((pit>.25)&(pit<.75)).mean():.2f}/{((pit>.10)&(pit<.90)).mean():.2f}/{((pit>.05)&(pit<.95)).mean():.2f}")
    a2, b2 = fit_variance(E["e"], E["ip_hat"])
    print(f"    by prior-info bucket at refit b={b2:.0f}:")
    E["z"] = E["e"] / np.sqrt(b2 / E["ip_hat"]); E["pit"] = norm.cdf(E["z"])
    E["bk"] = pd.cut(E["ip_t"].fillna(0), [-1, 0, 20, 60, 1000], labels=["no prior season", "1-20 IP", "21-60", ">60"])
    print(E.groupby("bk", observed=True).apply(lambda g: pd.Series({"n": len(g), "z_SD": g["z"].std(), "cov90": ((g["pit"] > .05) & (g["pit"] < .95)).mean()})).round(2).to_string())
    return b2, a2


if __name__ == "__main__":
    p = build_panel()
    print(f"panel: {len(p)} pitcher-seasons; seasons {p['year'].value_counts().sort_index().to_dict()}")
    tr = transitions(p)
    ipm = part1(tr)
    retm = part2(tr)
    # debut median IP (pitchers whose first season in the panel is t+1>2022)
    first = p.groupby("name_clean")["year"].min()
    deb = p[p.apply(lambda r: first[r["name_clean"]] == r["year"] and r["year"] > 2022, axis=1)]
    print(f"\ndebut-season IP (new to the league, years>2022): median {deb['ip'].median():.0f}, mean {deb['ip'].mean():.0f}, n={len(deb)}; by year {deb.groupby('year')['ip'].median().round(0).to_dict()}")
    gap = []
    for t in (2022, 2023, 2024):
        a = set(p[p.year == t].name_clean); b = set(p[p.year == t + 1].name_clean); c = p[(p.year == t + 2) & p.name_clean.isin(a - b)]
        gap.append(c)
    gap = pd.concat(gap)
    print(f"returners after a skipped season: n={len(gap)}, median IP {gap['ip'].median():.0f}, mean {gap['ip'].mean():.0f}")
    b_new, a_new = part3(tr, ipm, retm)
    print(f"\nPART 4 -- production parameters: IP model a={ipm.params['Intercept']:.2f}, b={ipm.params['ip_t']:.4f}; "
          f"retention logit {retm.params['Intercept']:.3f} + {retm.params['np.log(ip_t)']:.3f}*ln(IP); luck b(projected IP)={b_new:.1f}, a={a_new:.3f}; "
          f"debut IP={deb['ip'].median():.0f}; post-gap returner IP={gap['ip'].median():.0f}")
