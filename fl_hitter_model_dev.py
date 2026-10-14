"""
fl_hitter_model_dev.py -- design checks for the hitter projection (what the production script will freeze).

A  Persistence of Air% and HR-per-air-ball (needed for the Mississippi translation's inputs).
B  Park neutralisation of 2025-26 history (Trustmark share) and what it does to Mississippi hitters.
C  Playing time: PA_next ~ PA_t (+ PA_{t-1}), retention logit, leave-one-transition-out.
D  Final wOBA model on 2025 -> 2026 (the transition with xwOBA): direct, components, blend, + xwOBA term.
E  Luck law for wOBA: Var(error) = a + b / PA.
F  Non-batting WAR parts: fielding and positional adjustment persistence, and shrinkage slopes.
G  Whole-chain WAR backtest, 2025 -> 2026: projected WAR vs actual (bat + position + fielding + replacement), P(WAR>0) calibration.

Usage:  python fl_hitter_model_dev.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import KFold

import fl_hitter_baseline_backtest as bt
import fl_hitter_park as park

warnings.filterwarnings("ignore")
PANEL = "output/fl_hitter_panel.parquet"


def load():
    p = pd.read_parquet(PANEL)
    p["woba_raw"] = p["woba"]
    p["pa_trustmark"] = p["pa_trustmark"].fillna(0); p["pa_pitch"] = p["pa_pitch"].fillna(0)
    add = park.neutralise(p)
    p["woba_n"] = p["woba_raw"] + add
    s = (p["pa_trustmark"] / p["pa_pitch"]).where(p["pa_pitch"] > 0, 0.0).fillna(0.0)
    p["tm_share"] = s
    p["hrr_raw"] = p["hrr"]; p["k_raw"] = p["k"]; p["bbr_raw"] = p["bbr"]
    nr = park.neutralise_rates(p)                                   # 2026-10-13: HR/PA by the HR surface (was the 0.446 HR factor), K% / BB% by the global Trustmark effects
    p["hrr"] = p["hrr_raw"] + nr["hrr"].where(p["hrr_raw"].notna()); p["k"] = p["k_raw"] + nr["k"].where(p["k_raw"].notna()); p["bbr"] = p["bbr_raw"] + nr["bbr"].where(p["bbr_raw"].notna())
    p["air_pct"] = p["air_n"] / p["bbe"].replace(0, np.nan)
    p["hr_air"] = p["air_hr"] / p["air_n"].replace(0, np.nan)
    return p


def partA(p):
    print("A. PERSISTENCE of the translation inputs (2025 -> 2026, n0 = BBE / air-ball equivalents)")
    bt.DEN["air_pct"] = "bbe"; bt.DEN["hr_air"] = "air_n"
    for s, lab in (("air_pct", "Air% of batted balls"), ("hr_air", "HR per air ball")):
        d = bt.transitions(p, s)
        d = d[d["n_next"] >= 40]
        r = np.corrcoef(d["x_t"] - d["lg_t"], d["x_next"] - d["lg_next"])[0, 1]
        n0 = bt.fit_n0(d)
        print(f"   {lab:22s}: n={len(d)}, y/y corr {r:.3f}, n0 {n0:.0f}   (league {d['lg_t'].iloc[0]:.4f} -> {d['lg_next'].iloc[0]:.4f})")


def partB(p):
    print("\nB. PARK NEUTRALISATION of observed seasons (Trustmark PA share x air-ball profile)")
    t = p[p["source"] == "trackman"]
    g = t.assign(mis=t["team"] == "Mississippi Mud Monsters").groupby("mis").apply(lambda d: pd.Series({
        "hitters": len(d), "Trustmark PA share": d["pa_trustmark"].sum() / d["pa_pitch"].sum(),
        "mean wOBA add-back (PA>=100)": (d.loc[d["pa"] >= 100, "woba_n"] - d.loc[d["pa"] >= 100, "woba_raw"]).mean(),
        "max add-back": (d["woba_n"] - d["woba_raw"]).max()}))
    print(g.round(4).to_string())
    m = t[(t["team"] == "Mississippi Mud Monsters") & (t["pa"] >= 100)]
    print(f"   Mississippi hitters PA>=100 (n={len(m)}): raw wOBA {np.average(m['woba_raw'], weights=m['pa']):.4f} -> neutral {np.average(m['woba_n'], weights=m['pa']):.4f}; "
          f"league {np.average(t['woba_raw'], weights=t['pa']):.4f}")


def pa_model(p):
    """PA_next given retained, and retention logit; LOTO."""
    rows = []
    for t in range(2022, 2026):
        a = p[p["year"] == t].set_index("name_clean"); b = p[p["year"] == t + 1].set_index("name_clean")
        prev = p[p["year"] == t - 1].set_index("name_clean") if t > 2022 else None
        d = pd.DataFrame({"pa_t": a["pa"]})
        d["pa_prev"] = prev["pa"].reindex(d.index).fillna(0) if prev is not None else 0.0
        d["pa_next"] = b["pa"].reindex(d.index)
        d["t"] = t
        rows.append(d.reset_index())
    d = pd.concat(rows, ignore_index=True)
    d = d[d["pa_t"] > 0]
    d["ret"] = d["pa_next"].notna().astype(int); d["lnpa"] = np.log(d["pa_t"])
    return d


def partC(p):
    print("\nC. PLAYING TIME (PA), leave-one-transition-out, all hitters with PA_t > 0")
    d = pa_model(p)
    r = d.dropna(subset=["pa_next"])
    print(f"   {len(d)} hitter-transitions, {len(r)} retained ({len(r)/len(d):.0%})")
    rows = []
    for lab, f in (("PA_t only", "pa_next ~ pa_t"), ("+ PA_{t-1}", "pa_next ~ pa_t + pa_prev"), ("PA_t + PA_t^2", "pa_next ~ pa_t + I(pa_t**2)")):
        errs = []
        for t in sorted(r["t"].unique()):
            m = smf.ols(f, data=r[r["t"] != t]).fit(); te = r[r["t"] == t]
            errs.append(te["pa_next"] - m.predict(te))
        e = pd.concat(errs)
        rows.append({"model": lab, "RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(e.abs().mean()), "resid SD": float(e.std())})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    m = smf.ols("pa_next ~ pa_t", data=r).fit()
    print(f"   production fit: PA_next = {m.params['Intercept']:.2f} + {m.params['pa_t']:.4f} x PA_t  (resid SD {np.sqrt(m.scale):.1f}; R2 {m.rsquared:.3f})")
    lo = []
    for t in sorted(d["t"].unique()):
        a, b = d[d["t"] != t], d[d["t"] == t]
        mm_ = smf.logit("ret ~ lnpa", data=a).fit(disp=0); pr = mm_.predict(b).clip(1e-4, 1 - 1e-4)
        lo.append(-(b["ret"] * np.log(pr) + (1 - b["ret"]) * np.log(1 - pr)))
    base = d["ret"].mean()
    print(f"   retention logit on ln PA: LOTO log-loss {pd.concat(lo).mean():.4f} vs base-rate {-(d['ret'].mean()*np.log(base) + (1-d['ret'].mean())*np.log(1-base)):.4f}")
    ml = smf.logit("ret ~ lnpa", data=d).fit(disp=0)
    print(f"   retention: logit = {ml.params['Intercept']:.3f} + {ml.params['lnpa']:.3f} x ln(PA_t)")
    return m, ml


def partD(p):
    print("\nD. FINAL wOBA MODEL on 2025 -> 2026 (park-neutral wOBA; rows with xwOBA, PA>=50 in 2025, >=100 in 2026)")
    q = p.copy(); q["woba"] = q["woba_n"]
    base = bt.transitions(q, "woba")
    E, n0s = bt.loto(base, delta=0.75)
    print(f"   direct regressed neutral wOBA (delta .75): pooled n0 {bt.fit_n0(base, 0.75):.0f}; LOTO RMSE all transitions {bt.rmse(E, E['pred']):.4f} (n={len(E)})")
    comp = {}
    for s in ("k", "bbr", "hrr", "babip"):
        d = bt.transitions(q, s); Es, _ = bt.loto(d, delta=0.5)
        comp[s] = Es.set_index(["name_clean", "t"])["pred"].rename(f"{s}_pred")
    W = base.set_index(["name_clean", "t"]).join(list(comp.values()), how="inner").dropna()
    cols = ["k_pred", "bbr_pred", "hrr_pred", "babip_pred"]
    out = []
    for t in sorted(W.index.get_level_values("t").unique()):
        tr = W[W.index.get_level_values("t") != t]; te = W[W.index.get_level_values("t") == t]
        X = np.column_stack([np.ones(len(tr)), (tr[cols].sub(tr["lg_next"], axis=0)).to_numpy()])
        # regress deviation of next wOBA on deviations of predicted components from THEIR league means (no cross-league mixing)
        Xd = np.column_stack([tr[cols[i]] - tr[cols[i]].mean() for i in range(4)])
        wts = tr["w"].to_numpy()
        A = np.column_stack([np.ones(len(tr)), Xd])
        beta = np.linalg.lstsq(A * np.sqrt(wts)[:, None], tr["x_next"].to_numpy() * np.sqrt(wts), rcond=None)[0]
        Ad = np.column_stack([np.ones(len(te))] + [te[cols[i]] - tr[cols[i]].mean() for i in range(4)])
        out.append(te.assign(pred_comp=Ad @ beta))
    C = pd.concat(out)
    dd = E.set_index(["name_clean", "t"]).loc[C.index]
    print(f"   components model: RMSE {bt.rmse(C, C['pred_comp']):.4f} vs direct {bt.rmse(C, dd['pred']):.4f} on the same {len(C)} rows; 50/50 blend {bt.rmse(C, (C['pred_comp'].to_numpy() + dd['pred'].to_numpy())/2):.4f}")
    # xwOBA on 2025 -> 2026
    x25 = p[p["year"] == 2025].set_index("name_clean")[["xwoba", "pa"]].rename(columns={"xwoba": "xw25", "pa": "pa25"})
    T = C.reset_index()
    T = T[T["t"] == 2025].merge(x25, left_on="name_clean", right_index=True, how="left")
    T["m0"] = dd.reset_index().set_index(["name_clean", "t"]).loc[list(zip(T["name_clean"], T["t"])), "pred"].to_numpy()
    T = T.dropna(subset=["xw25"]); T = T[T["pa25"] >= 50].copy()
    lgx = np.average(T["xw25"], weights=T["pa25"])
    res = []
    yv = T["x_next"].to_numpy(); w = T["w"].to_numpy()
    blend = (T["pred_comp"].to_numpy() + T["m0"].to_numpy()) / 2
    for n0x in (100, 200, 300, 500):
        xs = (T["pa25"] * T["xw25"] + n0x * lgx) / (T["pa25"] + n0x)
        T[f"xs{n0x}"] = xs
    for lab, base_pred, n0x in (("direct", T["m0"].to_numpy(), None), ("blend(direct, components)", blend, None)) + tuple((f"blend + gamma x (xwOBA - blend), n0x={n0x}", blend, n0x) for n0x in (100, 200, 300, 500)):
        if n0x is None:
            pr = base_pred
            res.append({"model": lab, "RMSE": float(np.sqrt(np.average((yv - pr) ** 2, weights=w))), "gamma": np.nan}); continue
        xs = T[f"xs{n0x}"].to_numpy(); z = xs - base_pred
        pr = np.zeros(len(yv)); gs = []
        for rep in range(20):
            for tri, tei in KFold(10, shuffle=True, random_state=rep).split(z):
                g = np.sum(w[tri] * z[tri] * (yv[tri] - base_pred[tri])) / np.sum(w[tri] * z[tri] ** 2)
                gs.append(g); pr[tei] += (base_pred[tei] + g * z[tei]) / 20
        res.append({"model": lab, "RMSE": float(np.sqrt(np.average((yv - pr) ** 2, weights=w))), "gamma": float(np.mean(gs))})
    print(f"   2025 -> 2026 rows with xwOBA: n={len(T)}")
    print(pd.DataFrame(res).round(4).to_string(index=False))
    n0x_best = min((r for r in res if r["gamma"] == r["gamma"]), key=lambda r: r["RMSE"])
    return E, n0s, n0x_best


def partE(E):
    print("\nE. LUCK LAW for park-neutral wOBA errors (all four held-out transitions, regressed-direct predictions)")
    e = (E["x_next"] - E["pred"]).to_numpy(); n = E["n_next"].to_numpy()

    def nll(q):
        v = q[0] + q[1] / n
        return 0.5 * np.sum(np.log(v) + e ** 2 / v)
    best = min((minimize(nll, x0, bounds=[(1e-8, None)] * 2, method="L-BFGS-B") for x0 in ([1e-3, 0.2], [3e-4, 0.1], [1e-4, 0.3])), key=lambda r: r.fun)
    a, b = best.x
    print(f"   Var(err) = a + b/PA: a = {a:.6f} (SD {np.sqrt(a):.4f}), b = {b:.4f};  SD at 100 / 300 / 600 PA = {np.sqrt(a + b/100):.4f} / {np.sqrt(a + b/300):.4f} / {np.sqrt(a + b/600):.4f}")
    z = e / np.sqrt(a + b / n)
    pit = norm.cdf(z)
    print(f"   standardised error SD {z.std():.2f}; coverage 50/80/90% = {((pit>.25)&(pit<.75)).mean():.2f}/{((pit>.10)&(pit<.90)).mean():.2f}/{((pit>.05)&(pit<.95)).mean():.2f}")
    return a, b


def war_parts(p):
    t = p[p["source"] == "trackman"].copy()
    t["fld_pa"] = t["fielding_runs"] / t["pa"]; t["pos_pa"] = t["pos_adj_runs"] / t["pa"]
    return t


def partF(p):
    print("\nF. NON-BATTING WAR PARTS: persistence 2025 -> 2026 (per PA, PA>=100 both years)")
    t = war_parts(p)
    a = t[t["year"] == 2025].set_index("name_clean"); b = t[t["year"] == 2026].set_index("name_clean")
    d = pd.DataFrame({"fld_t": a["fld_pa"], "pos_t": a["pos_pa"], "pa_t": a["pa"], "pos_name_t": a["primary_pos"]}).join(
        pd.DataFrame({"fld_n": b["fld_pa"], "pos_n": b["pos_pa"], "pa_n": b["pa"], "pos_name_n": b["primary_pos"]}), how="inner")
    d = d[(d["pa_t"] >= 100) & (d["pa_n"] >= 100)]
    same = d["pos_name_t"] == d["pos_name_n"]
    print(f"   n={len(d)}, same primary position both years: {same.mean():.0%}")
    out = {}
    for lab, x, y in (("fielding runs/PA", "fld_t", "fld_n"), ("positional adj runs/PA", "pos_t", "pos_n")):
        for who, dd in (("all", d), ("same position", d[same])):
            m = smf.wls(f"{y} ~ {x}", data=dd, weights=dd["pa_n"]).fit()
            print(f"   {lab:24s} [{who:13s}] n={len(dd):3d}  corr {dd[x].corr(dd[y]):.3f}  slope {m.params[x]:.3f} (se {m.bse[x]:.3f})  intercept {m.params['Intercept']:+.5f}")
            out[(lab, who)] = (m.params["Intercept"], m.params[x])
    print("   positional adjustment by primary position (2026, runs per 600 PA):")
    q = t[t["year"] == 2026].groupby("primary_pos").apply(lambda g: 600 * g["pos_adj_runs"].sum() / g["pa"].sum())
    print("   ", q.round(2).to_dict())
    return out


def partG(p, pa_m, ret_m, ab, f_fit, n0x_best):
    print("\nG. WHOLE-CHAIN BACKTEST 2025 -> 2026 (bat + position + fielding + replacement; no baserunning)")
    q = p.copy(); q["woba"] = q["woba_n"]
    n0, delta = bt.fit_n0(bt.transitions(q, "woba").query("t != 2025"), 0.75), 0.75
    t25 = p[(p["year"] == 2025) & (p["source"] == "trackman")].set_index("name_clean")
    t26 = p[(p["year"] == 2026) & (p["source"] == "trackman")].set_index("name_clean")
    lg25 = np.average(q.loc[q["year"] == 2025, "woba"], weights=q.loc[q["year"] == 2025, "pa"])
    lg26 = np.average(q.loc[q["year"] == 2026, "woba"], weights=q.loc[q["year"] == 2026, "pa"])
    RPW, REPL, SC = 11.976, 0.0362, 1.0175
    d = t25[["pa", "woba_n", "fielding_runs", "pos_adj_runs"]].join(t26[["pa", "woba_n", "fielding_runs", "pos_adj_runs"]], lsuffix="_t", rsuffix="_n", how="inner")
    d = d[(d["pa_t"] >= 20) & (d["pa_n"] >= 20)].copy()
    fi, fs = f_fit[("fielding runs/PA", "all")]; pi_, ps = f_fit[("positional adj runs/PA", "all")]
    d["bat_rate_hat"] = (d["pa_t"] * (d["woba_n_t"] - lg25)) / (d["pa_t"] + n0) / SC
    # fielding: reliability-weighted per-PA rate (n0 = 200 PA of zero-fielding mixed in, no extra slope: 10-fold optimum, slope ~1.0)
    d["fld_rate_hat"] = (d["fielding_runs_t"] / d["pa_t"]) * d["pa_t"] / (d["pa_t"] + 200.0)
    d["pos_rate_hat"] = 0.376 * d["pos_adj_runs_t"] / d["pa_t"]
    d["rate_hat"] = d["bat_rate_hat"] + d["fld_rate_hat"] + d["pos_rate_hat"] + REPL            # runs above replacement per PA
    d["bat_act"] = (d["woba_n_n"] - lg26) / SC; d["fld_act"] = d["fielding_runs_n"] / d["pa_n"]; d["pos_act"] = d["pos_adj_runs_n"] / d["pa_n"]
    d["rate_act"] = d["bat_act"] + d["fld_act"] + d["pos_act"] + REPL
    for c in ("bat", "fld", "pos"):
        d[f"{c}_hat_rate"] = d[f"{c}_rate_hat"]
    e = d[d["pa_n"] >= 100].copy()
    print(f"   hitters with PA_2025>=20 and PA_2026>=20: {len(d)}; PA_2026>=100: {len(e)}")
    rows = []
    for lab, hat, act in (("batting", e["bat_rate_hat"], e["bat_act"]), ("fielding", e["fld_rate_hat"], e["fld_act"]), ("position", e["pos_rate_hat"], e["pos_act"]),
                          ("TOTAL runs above replacement / PA", e["rate_hat"], e["rate_act"])):
        w = e["pa_n"]
        rm = lambda x: float(np.sqrt(np.average((act - x) ** 2, weights=w)) * 600)
        rows.append({"component (runs per 600 PA)": lab, "SD actual": float(act.std() * 600), "RMSE model": rm(hat), "RMSE league-average": rm(np.full(len(e), np.average(act, weights=w))),
                     "corr": float(np.corrcoef(hat, act)[0, 1])})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    # P(WAR>0) given realized PA: noise = bat luck (a + b/PA) + fielding and position residuals per PA
    a, b = ab
    res_f = float(np.sqrt(np.average((e["fld_act"] - e["fld_rate_hat"]) ** 2, weights=e["pa_n"])))
    res_p = float(np.sqrt(np.average((e["pos_act"] - e["pos_rate_hat"]) ** 2, weights=e["pa_n"])))
    sd_rate = np.sqrt((a + b / e["pa_n"]) / SC ** 2 + res_f ** 2 + res_p ** 2)
    e["P"] = norm.cdf(e["rate_hat"] / sd_rate); e["y"] = (e["rate_act"] > 0).astype(float)
    for col in ("rate_act",):
        z = (e["rate_act"] - e["rate_hat"]) / sd_rate
        print(f"   standardised error of the total rate: SD {z.std():.2f}; coverage 50/80/90% = {((norm.cdf(z)>.25)&(norm.cdf(z)<.75)).mean():.2f}/{((norm.cdf(z)>.10)&(norm.cdf(z)<.90)).mean():.2f}/{((norm.cdf(z)>.05)&(norm.cdf(z)<.95)).mean():.2f}")
    print(f"   P(WAR>0) given realized PA: mean predicted {e['P'].mean():.3f} vs actual share {e['y'].mean():.3f}; Brier {np.mean((e['P']-e['y'])**2):.4f} vs base-rate {np.mean((e['y'].mean()-e['y'])**2):.4f}")
    e["bin"] = pd.cut(e["P"], [0, .5, .7, .85, .95, 1.0])
    print(e.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("P", "mean"), actual=("y", "mean")).round(3).to_string())
    print(f"   residual SD per 600 PA: fielding {res_f*600:.1f} runs, position {res_p*600:.1f} runs")
    return d


if __name__ == "__main__":
    p = load()
    partA(p)
    partB(p)
    pa_m, ret_m = partC(p)
    E, n0s, nx = partD(p)
    a, b = partE(E)
    f_fit = partF(p)
    d = partG(p, pa_m, ret_m, (a, b), f_fit, nx)
