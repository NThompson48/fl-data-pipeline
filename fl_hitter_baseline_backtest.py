"""
fl_hitter_baseline_backtest.py -- how predictable is each hitter stat from one FL season to the next, and what is the
simplest honest projection (regression to the league level, Marcel-style)?

Data: output/fl_hitter_panel.parquet (2022-26; four transitions t -> t+1).
Method (same logic as the pitcher K%/BB% baseline):
    dev      = player rate - that season's league rate (PA-weighted)           removes league-level drift
    pred     = LG_next + (n_t * dev_t [+ d * n_{t-1} * dev_{t-1}]) / (n_t [+ d n_{t-1}] + n0)
    n0       = the "regression constant" (PA of league-average performance mixed in), fit by weighted least squares
               on the OTHER three transitions (leave-one-transition-out), so the reported error is out of sample.
Part 1  Reliability: n0 per stat, LOTO RMSE vs "last year's raw rate" and "league mean", year-to-year correlations.
Part 2  Multi-year weighting d (does t-1 add anything?).
Part 3  wOBA built directly vs from separately regressed components (K%, BB%, HR/PA, ISO-on-contact...).
Part 4  Cost of league drift: error when the league level of the target year is unknown (use last year's level).
Part 5  xwOBA / Trackman skill metrics on the single transition where they exist (2025 -> 2026).

Usage:  python fl_hitter_baseline_backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import fl_eval_weights as ew

PANEL = "output/fl_hitter_panel.parquet"
STATS = {"k": "K%", "bbr": "BB%", "hrr": "HR/PA", "iso": "ISO", "babip": "BABIP", "woba": "wOBA", "hbpr": "HBP%"}
DEN = {"k": "pa", "bbr": "pa", "hrr": "pa", "hbpr": "pa", "woba": "pa", "iso": "ab", "babip": "ab"}   # sample size behind each rate
MIN_NEXT = 100
DELTA = {"woba": 0.75, "k": 0.75}      # prior-season weight by stat (0.5 otherwise)
WEIGHT_MODE = "invvar"                 # "invvar" (standard: 1/(a + b/n)), "legacy150" (n/(n+150), the old placeholder), "product" (reliability x inverse variance: double counts n)


def league(p, stat):
    d = p.dropna(subset=[stat])
    n = d[DEN[stat]]
    return d.groupby("year").apply(lambda g: np.average(g[stat], weights=g[DEN[stat]]))


def transitions(p, stat):
    lg = league(p, stat)
    rows = []
    for t in range(2022, 2026):
        if t not in lg.index or (t + 1) not in lg.index:       # stat not available for these seasons (e.g. batted-ball stats before 2025)
            continue
        a = p[p["year"] == t].set_index("name_clean"); b = p[p["year"] == t + 1].set_index("name_clean")
        prev = p[p["year"] == t - 1].set_index("name_clean") if t > 2022 else None
        d = pd.DataFrame({"x_t": a[stat], "n_t": a[DEN[stat]]})
        d["x_next"] = b[stat].reindex(d.index); d["n_next"] = b[DEN[stat]].reindex(d.index)
        d["x_prev"] = prev[stat].reindex(d.index) if prev is not None else np.nan
        d["n_prev"] = prev[DEN[stat]].reindex(d.index) if prev is not None else np.nan
        d["t"] = t; d["lg_t"] = lg[t]; d["lg_next"] = lg[t + 1]; d["lg_prev"] = lg.get(t - 1, np.nan)
        rows.append(d.reset_index())
    d = pd.concat(rows, ignore_index=True)
    d = d.dropna(subset=["x_t", "x_next"])
    d = d[(d["n_t"] > 0) & (d["n_next"] >= MIN_NEXT)]
    d["n_prev"] = d["n_prev"].fillna(0); d["x_prev"] = d["x_prev"].fillna(d["lg_prev"])
    return set_weights(d.reset_index(drop=True), DELTA.get(stat, 0.5))


def predict(d, n0, delta=0.0, recenter=True):
    dev_t = d["x_t"] - d["lg_t"]; dev_p = d["x_prev"] - d["lg_prev"].fillna(d["lg_t"])
    num = d["n_t"] * dev_t + delta * d["n_prev"] * dev_p.fillna(0)
    den = d["n_t"] + delta * d["n_prev"] + n0
    base = d["lg_next"] if recenter else d["lg_t"]
    return base + num / den


def wse(d, pred, n0w=None):
    w = d["w"] if "w" in d else d["n_next"] / (d["n_next"] + (n0w or 150.0))
    return float(np.sum(w * (d["x_next"] - pred) ** 2) / w.sum())


def _fit_n0_fixed_w(d, delta, recenter):
    r = minimize_scalar(lambda lg: wse(d, predict(d, np.exp(lg), delta, recenter)), bounds=(np.log(5), np.log(20000)), method="bounded")
    return float(np.exp(r.x))


def set_weights(d, delta):
    """Attach the standard evaluation weight w = 1/(a + b/n_next) (fl_eval_weights): iterate  fit n0 -> residuals -> (a, b) -> weights  a few
    times on the full pooled sample, then FREEZE the weights (columns w, law_a, law_b) so every candidate model is scored with the same ones."""
    d = d.copy()
    n = d["n_next"].to_numpy(dtype=float)
    d["w"] = n / (n + 200.0)
    if WEIGHT_MODE == "legacy150":
        d["w"] = n / (n + 150.0); d["law_a"] = np.nan; d["law_b"] = np.nan
        return d
    a = b = np.nan
    for _ in range(4):
        n0 = _fit_n0_fixed_w(d, delta, True)
        e = (d["x_next"] - predict(d, n0, delta, True)).to_numpy()
        a, b = ew.fit_variance_law(e, n)
        d["w"] = ew.inv_var_weights(n, a, b)
    if WEIGHT_MODE == "product":
        d["w"] = d["w"] * n / (n + b / a)
    d["law_a"] = a; d["law_b"] = b
    return d


def fit_n0(d, delta=0.0, recenter=True):
    return _fit_n0_fixed_w(d, delta, recenter)


def loto(d, delta=0.0, recenter=True):
    out, n0s = [], {}
    for t in sorted(d["t"].unique()):
        tr, te = d[d["t"] != t], d[d["t"] == t]
        n0 = fit_n0(tr, delta, recenter); n0s[t] = n0
        out.append(te.assign(pred=predict(te, n0, delta, recenter)))
    return pd.concat(out), n0s


def rmse(d, pred):
    w = d["w"] if "w" in d else d["n_next"] / (d["n_next"] + 150.0)
    return float(np.sqrt(np.sum(w * (d["x_next"] - pred) ** 2) / w.sum()))


def part1_2(p):
    print("PART 1/2 -- reliability and multi-year weighting (leave-one-transition-out; targets: next-year rate, PA/AB >= %d)" % MIN_NEXT)
    rows = []
    for s, lab in STATS.items():
        d = transitions(p, s)
        E, n0s = loto(d)
        raw = rmse(d, d["x_t"]); mean = rmse(d, d["lg_next"]); mod = rmse(E, E["pred"])
        best_delta, best = 0.0, mod
        for dl in (0.25, 0.5, 0.75, 1.0):
            E2, _ = loto(d, dl)
            r = rmse(E2, E2["pred"])
            if r < best - 1e-9:
                best_delta, best = dl, r
        rr = np.corrcoef(d["x_t"] - d["lg_t"], d["x_next"] - d["lg_next"])[0, 1]
        rows.append({"stat": lab, "n": len(d), "y/y corr": rr, "n0 (pooled)": fit_n0(d), "n0 range by held-out t": f"{min(n0s.values()):.0f}-{max(n0s.values()):.0f}",
                     "RMSE league mean": mean, "RMSE last-year raw": raw, "RMSE regressed (t only)": mod, "best delta (t-1 weight)": best_delta, "RMSE with t-1": best})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    return t


def part3(p):
    print("\nPART 3 -- wOBA: direct vs built from regressed components (LOTO, league level known)")
    base = transitions(p, "woba")
    E, _ = loto(base)
    print(f"   direct regressed wOBA: RMSE {rmse(E, E['pred']):.4f} (n={len(E)})")
    # components: K%, BB%, HR/PA, and 'non-HR contact value' = (wOBA*PA - weighted K/BB/HR parts) approximated by regressing wOBA residual on components
    comp = {}
    for s in ("k", "bbr", "hrr", "babip", "iso"):
        d = transitions(p, s); Es, _ = loto(d)
        comp[s] = Es.set_index(["name_clean", "t"])["pred"].rename(f"{s}_pred")
    Wd = base.set_index(["name_clean", "t"]).join(list(comp.values()), how="inner").dropna()
    # fit next-year wOBA deviation on predicted component deviations (in-sample linear combo, then LOTO)
    X = Wd[["k_pred", "bbr_pred", "hrr_pred", "babip_pred", "iso_pred"]].sub(0)
    Wd = Wd.assign(wpred_direct=Wd["lg_next"] + 0)  # placeholder to keep columns explicit
    outs = []
    for t in sorted(Wd.index.get_level_values("t").unique()):
        tr = Wd[Wd.index.get_level_values("t") != t]; te = Wd[Wd.index.get_level_values("t") == t]
        A = np.column_stack([np.ones(len(tr)), tr[["k_pred", "bbr_pred", "hrr_pred", "babip_pred"]].to_numpy()])
        w = (tr["n_next"] / (tr["n_next"] + 150.0)).to_numpy()
        beta = np.linalg.lstsq(A * np.sqrt(w)[:, None], tr["x_next"].to_numpy() * np.sqrt(w), rcond=None)[0]
        B = np.column_stack([np.ones(len(te)), te[["k_pred", "bbr_pred", "hrr_pred", "babip_pred"]].to_numpy()])
        outs.append(te.assign(pred_comp=B @ beta))
    C = pd.concat(outs)
    dd = E.set_index(["name_clean", "t"]).loc[C.index]
    print(f"   components (K%, BB%, HR/PA, BABIP -> wOBA, weights fit on other transitions): RMSE {rmse(C, C['pred_comp']):.4f} vs direct on the same rows {rmse(C, dd['pred']):.4f} (n={len(C)})")
    avg = (C["pred_comp"].to_numpy() + dd["pred"].to_numpy()) / 2
    print(f"   50/50 average of the two: RMSE {rmse(C, avg):.4f}")


def part4(p):
    print("\nPART 4 -- cost of not knowing next year's league level (recenter=False uses this year's league rate)")
    rows = []
    for s, lab in STATS.items():
        d = transitions(p, s)
        a, _ = loto(d, recenter=True); b, _ = loto(d, recenter=False)
        lgchg = (d.groupby("t")["lg_next"].first() - d.groupby("t")["lg_t"].first())
        rows.append({"stat": lab, "RMSE known league level": rmse(a, a["pred"]), "RMSE last year's level": rmse(b, b["pred"]),
                     "league change by transition": ", ".join(f"{v:+.4f}" for v in lgchg.values)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def part5(p):
    print("\nPART 5 -- Trackman skill metrics for 2025 -> 2026 (the only transition with pitch data)")
    a = p[(p["year"] == 2025)].set_index("name_clean"); b = p[(p["year"] == 2026)].set_index("name_clean")
    d = a[["pa", "woba", "xwoba", "k", "bbr", "hrr", "avg_ev", "ev90", "barrel_pct", "hardhit_pct", "chase_pct", "whiff_pct", "zcontact_pct", "pull_air_pct", "xdamage", "bat_park_factor"]].copy()
    d["woba_next"] = b["woba"].reindex(d.index); d["pa_next"] = b["pa"].reindex(d.index)
    d = d[(d["pa"] >= 50) & (d["pa_next"] >= MIN_NEXT)].dropna(subset=["xwoba", "woba"]).copy()
    lg = {c: np.average(d[c].dropna(), weights=d.loc[d[c].notna(), "pa"]) for c in ("woba", "xwoba")}
    n0 = 300.0
    d["woba_s"] = (d["pa"] * d["woba"] + n0 * lg["woba"]) / (d["pa"] + n0)
    d["xwoba_s"] = (d["pa"] * d["xwoba"] + n0 * lg["xwoba"]) / (d["pa"] + n0)
    print(f"   n={len(d)} hitters (PA>=50 in 2025, PA>={MIN_NEXT} in 2026). corr with 2026 wOBA: wOBA {d['woba'].corr(d['woba_next']):.3f}, xwOBA {d['xwoba'].corr(d['woba_next']):.3f}, "
          f"K% {d['k'].corr(d['woba_next']):+.3f}, BB% {d['bbr'].corr(d['woba_next']):+.3f}, EV90 {d['ev90'].corr(d['woba_next']):+.3f}, barrel% {d['barrel_pct'].corr(d['woba_next']):+.3f}, "
          f"chase% {d['chase_pct'].corr(d['woba_next']):+.3f}")
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict
    y = d["woba_next"].to_numpy(); w = (d["pa_next"] / (d["pa_next"] + 150.0)).to_numpy()
    sets = {"league mean": None, "woba only": ["woba_s"], "xwoba only": ["xwoba_s"], "woba + xwoba": ["woba_s", "xwoba_s"],
            "woba + xwoba + K% + BB%": ["woba_s", "xwoba_s", "k", "bbr"],
            "woba + xwoba + EV90 + chase% + zcontact%": ["woba_s", "xwoba_s", "ev90", "chase_pct", "zcontact_pct"]}
    res = []
    for lab, cols in sets.items():
        if cols is None:
            pr = np.full(len(y), np.average(y, weights=w))
        else:
            X = d[cols].fillna(d[cols].mean()).to_numpy(); pr = np.zeros(len(y))
            for rep in range(10):
                for tri, tei in KFold(10, shuffle=True, random_state=rep).split(X):
                    m = LinearRegression().fit(X[tri], y[tri], sample_weight=w[tri]); pr[tei] += m.predict(X[tei]) / 10
        res.append({"predictors": lab, "10-fold RMSE (wOBA)": float(np.sqrt(np.average((y - pr) ** 2, weights=w))), "corr": float(np.corrcoef(pr, y)[0, 1]) if cols else np.nan})
    print(pd.DataFrame(res).round(4).to_string(index=False))


if __name__ == "__main__":
    p = pd.read_parquet(PANEL)
    part1_2(p)
    part3(p)
    part4(p)
    part5(p)
