"""
k26_projection.py -- simple, prediction-focused analysis for projecting 2026 K%.

Built from three transparent pieces, each with its own figure:

  1. aging_curve()   age effect estimated from individual pitchers' season-to-season
                     K% changes (the "delta method": pair each pitcher's year Y with
                     year Y+1, average the change by age, weight by playing time).

  2. persistence()   year-1 K% -> year-2 K%.  Shows regression toward the mean and
                     that a stabilized year-1 K% (shrunk ~70 TBF toward league)
                     predicts next year better than raw K%.

  3. interactions()  does Stuff+ x {# pitch types / # fastball types / # breaking
                     types} explain next-year K% beyond RTM + aging?  Weighted OLS
                     on the couplets, with the interaction terms + slope plots.

project_2026() then combines them:

    K%_2026_hat  =  stabilized(K%_2025)  +  aging_delta(age_2025)  +  arsenal/stuff adj.

Input: a merged pitcher-season panel with columns (names/symbols flexible)
    Season, PlayerId, K%, TBF, Stuff+, Age,
    n_pitch_types, n_fastball_types, n_breaking_types
Optional: SO (else derived as round(K% * TBF)).

Usage
-----
    from k26_projection import run
    res = run(merged)                 # -> figures in output/, tables in the dict
    res["proj_2026"].head(20)         # 2026 K% forecast per pitcher
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

STAB_TBF = 70          # provisional; run() replaces this with the data-estimated n0
OUT = "output"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25,
                     "axes.axisbelow": True, "font.size": 10})

PREDS = ["stuff", "n_pitch", "n_fb", "n_bb"]
LBL = {"stuff": "Stuff+", "n_pitch": "# pitch types",
       "n_fb": "# fastball types", "n_bb": "# breaking types"}


# --------------------------------------------------------------------------- #
def _find(df, *cands):
    low = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for c in cands:
        k = c.lower().replace(" ", "").replace("_", "")
        if k in low:
            return low[k]
    return None


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    c = {"season": _find(df, "Season", "year"),
         "pid":   _find(df, "PlayerId", "pitcher", "IDfg", "mlbam"),
         "k":     _find(df, "K%", "k_pct", "kpct"),
         "tbf":   _find(df, "TBF", "bf", "pa"),
         "stuff": _find(df, "Stuff+", "stuff_plus", "stuff"),
         "age":   _find(df, "Age"),
         "n_pitch": _find(df, "n_pitch_types"),
         "n_fb":  _find(df, "n_fastball_types"),
         "n_bb":  _find(df, "n_breaking_types"),
         "so":    _find(df, "SO", "strikeouts")}
    need = [k for k in ("season", "pid", "k", "tbf", "age") if c[k] is None]
    if need:
        raise KeyError(f"missing columns for {need}; have {list(df.columns)}")

    o = pd.DataFrame({"season": pd.to_numeric(df[c["season"]], errors="coerce"),
                      "pid": df[c["pid"]]})
    k = pd.to_numeric(df[c["k"]].astype(str).str.replace("%", "", regex=False),
                      errors="coerce")
    o["k_pct"] = k / 100 if k.median(skipna=True) > 1.5 else k
    o["tbf"] = pd.to_numeric(df[c["tbf"]], errors="coerce")
    o["age"] = pd.to_numeric(df[c["age"]], errors="coerce")
    for p in PREDS:
        o[p] = pd.to_numeric(df[c[p]], errors="coerce") if c[p] else np.nan
    so = pd.to_numeric(df[c["so"]], errors="coerce") if c["so"] else pd.Series(np.nan, index=df.index)
    o["so"] = np.where(np.isfinite(so), so, np.round(o["k_pct"] * o["tbf"]))

    o = o[np.isfinite(o.k_pct) & (o.tbf > 0) & np.isfinite(o.age)].copy()
    lg = o.groupby("season").apply(lambda g: np.average(g.k_pct, weights=g.tbf)).rename("lg_k")
    o = o.join(lg, on="season")
    o["k_stab"] = (o.so + STAB_TBF * o.lg_k) / (o.tbf + STAB_TBF)
    return o.reset_index(drop=True)


def couplets(o: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-season pairs for the same pitcher: *1 = year Y, *2 = year Y+1."""
    b = o.copy()
    b["season"] = b["season"] - 1
    m = o.merge(b, on=["pid", "season"], suffixes=("1", "2"))
    m = m.rename(columns={"k_pct1": "k1", "k_pct2": "k2"})
    m["w"] = 2 / (1 / m.tbf1 + 1 / m.tbf2)          # harmonic mean of playing time
    m["age1"] = m.age1.round().astype(int)
    for p in PREDS:                                 # bare name == year-Y (predictor) value
        m[p] = m[p + "1"]
    return m


def estimate_n0(o: pd.DataFrame, min_tbf: int = 25):
    """Data-estimated stabilization point n0 for K%: reliability(n) = n / (n + n0).
    Method of moments -- split observed spread into true-talent + binomial sampling:
        Var(true) = Var_w(observed)  -  p(1-p) / mean(TBF)
        n0        = p(1-p) / Var(true)
    No fixed '70' -- the whole reliability curve falls out of this single number."""
    d = o[o.tbf >= min_tbf]
    n, x = d.tbf.values.astype(float), d.k_pct.values
    p = np.average(x, weights=n)
    var_obs = np.average((x - p) ** 2, weights=n)
    var_true = max(var_obs - p * (1 - p) * len(d) / n.sum(), 1e-6)
    n0 = p * (1 - p) / var_true
    return float(n0), {"p": round(p, 4), "sd_true_K%": round(np.sqrt(var_true), 4),
                       "n_used": len(d), "min_tbf": min_tbf}


def reliability_curve(o, cp, n0):
    """Fitted reliability curve n/(n+n0) vs. the empirical year-over-year signal."""
    yoy = cp.dropna(subset=["k1", "k2"]).copy()
    yoy["tmin"] = yoy[["tbf1", "tbf2"]].min(axis=1)
    edges = [25, 45, 70, 100, 140, 200, 300, 450, 10000]
    xs, rs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = yoy[(yoy.tmin >= lo) & (yoy.tmin < hi)]
        if len(s) < 25:
            continue
        xs.append(s.tmin.median())
        rs.append(np.corrcoef(s.k1, s.k2)[0, 1])
        ns.append(len(s))
    # separate n0 implied by the year-over-year points (measurement + real change)
    xs_a = np.array(xs)
    n0_yoy = np.nan
    if len(xs) >= 4:
        from scipy.optimize import curve_fit
        try:
            n0_yoy = curve_fit(lambda t, a: t / (t + a), xs_a, np.array(rs),
                               p0=[100], bounds=(1, 5000))[0][0]
        except Exception:
            pass

    t = np.arange(1, 700)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, t / (t + n0), lw=2.5, color="#2f6f8e",
            label=f"fitted measurement reliability  n/(n+{n0:.0f})")
    ax.plot(t, t / (t + 70), lw=1.2, ls="--", color="gray", label="reference: n/(n+70)")
    ax.scatter(xs, rs, s=[min(nn / 3, 120) for nn in ns], color="#c1121f", zorder=5,
               label="empirical corr(K%$_Y$, K%$_{Y+1}$) by min TBF")
    if np.isfinite(n0_yoy):
        ax.plot(t, t / (t + n0_yoy), lw=1.8, ls=":", color="#c1121f",
                label=f"fitted to year-over-year  n/(n+{n0_yoy:.0f})")
    ax.axhline(.5, color="k", lw=.8)
    ax.axvline(n0, color="#2f6f8e", ls=":", lw=1)
    ax.set(xlabel="TBF", ylabel="reliability", ylim=(0, 1),
           title="K% reliability curve — estimated from the data, not fixed at 70")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/k26_0_reliability.png", bbox_inches="tight")
    plt.close(fig)
    return {"n0_measurement": round(n0, 1),
            "n0_year_over_year": round(float(n0_yoy), 1) if np.isfinite(n0_yoy) else None,
            "empirical_r_by_bucket": pd.DataFrame({"min_tbf": np.round(xs).astype(int),
                                                   "r": np.round(rs, 3), "n": ns})}


# --------------------------------------------------------------------------- #
def _wmean_ci(y, w):
    yb = np.average(y, weights=w)
    neff = w.sum() ** 2 / np.sum(w ** 2)
    sd = np.sqrt(np.average((y - yb) ** 2, weights=w))
    return yb, 1.96 * sd / np.sqrt(neff)


def _wols(formula, data, wcol="w"):
    import statsmodels.formula.api as smf
    return smf.wls(formula, data=data, weights=data[wcol]).fit()


def _z(s):
    return (s - s.mean()) / s.std()


def _wcorr(x, y, w):
    x, y, w = map(np.asarray, (x, y, w))
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return cov / np.sqrt(vx * vy)


# --------------------------------------------------------------------------- #
#  1. persistence  (year Y -> year Y+1)  -- also yields the projection base
# --------------------------------------------------------------------------- #
def persistence(cp: pd.DataFrame):
    raw = _wols("k2 ~ k1", cp)
    base = _wols("k2 ~ k_stab1", cp)          # <- base model used by everything downstream

    def wrmse(fit):
        return np.sqrt(np.average((cp.k2 - fit.fittedvalues) ** 2, weights=cp.w))

    tab = pd.DataFrame({
        "predictor of next-year K%": ["raw K%_Y",
                                      f"stabilized K%_Y (shrink n0={STAB_TBF:.0f} TBF)"],
        "slope_to_Y+1": [raw.params["k1"], base.params["k_stab1"]],
        "wcorr_with_Y+1": [_wcorr(cp.k1, cp.k2, cp.w), _wcorr(cp.k_stab1, cp.k2, cp.w)],
        "wRMSE": [wrmse(raw), wrmse(base)],
    }).round(4)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(cp.k_stab1 * 100, cp.k2 * 100, s=6 * cp.w / cp.w.mean(),
                  alpha=.18, color="#2f6f8e")
    xs = np.linspace(cp.k_stab1.min(), cp.k_stab1.max(), 50)
    ax[0].plot(xs * 100, (base.params["Intercept"] + base.params["k_stab1"] * xs) * 100,
               color="#c1121f", lw=2,
               label=f"base fit: slope={base.params['k_stab1']:.2f}")
    ax[0].plot(xs * 100, xs * 100, "k--", lw=1, label="y = x")
    ax[0].set(title="Stabilized K%$_Y$  →  K%$_{Y+1}$   (dot size = playing time)",
              xlabel="stabilized K%  year Y", ylabel="K%  year Y+1")
    ax[0].legend()

    ax[1].bar(["raw K%_Y", "stabilized K%_Y"], tab["wcorr_with_Y+1"],
              color=["#9aa7b1", "#2f6f8e"])
    for i, v in enumerate(tab["wcorr_with_Y+1"]):
        ax[1].text(i, v + .02, f"{v:.3f}", ha="center")
    ax[1].set(title="Weighted corr. with next-year K%", ylabel="r", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(f"{OUT}/k26_2_persistence.png", bbox_inches="tight")
    plt.close(fig)
    return base, tab


# --------------------------------------------------------------------------- #
#  2. aging curve -- individual YoY changes, RAW vs net of regression-to-mean
# --------------------------------------------------------------------------- #
def aging_curve(cp: pd.DataFrame, base_model=None):
    cp = cp.copy()
    cp["dk_raw"] = cp.k2 - cp.k1                        # naive delta method
    cp["dk"] = cp.k_stab2 - cp.k_stab1                 # both ends shrunk ~70 TBF -> luck damped

    rows = []
    for age, g in cp.groupby("age1"):
        if len(g) < 25:
            continue
        mr, _ = _wmean_ci(g.dk_raw.values, g.w.values)
        md, ci = _wmean_ci(g.dk.values, g.w.values)
        rows.append((age, mr, md, ci, len(g)))
    curve = pd.DataFrame(rows, columns=["age_from", "mean_dK_raw", "mean_dK", "ci", "n"])

    # zero the young-age plateau (<=25) so the curve is aging relative to youth
    # and any league-wide drift is removed
    off = np.average(curve.loc[curve.age_from <= 25, "mean_dK"],
                     weights=curve.loc[curve.age_from <= 25, "n"])
    curve["mean_dK"] -= off
    curve["mean_dK_raw"] -= np.average(curve.loc[curve.age_from <= 25, "mean_dK_raw"],
                                       weights=curve.loc[curve.age_from <= 25, "n"])
    curve["cum_dK"] = curve.mean_dK.cumsum() - curve.mean_dK.cumsum().max()
    curve["cum_dK_raw"] = curve.mean_dK_raw.cumsum() - curve.mean_dK_raw.cumsum().max()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].axhline(0, color="gray", lw=1)
    ax[0].errorbar(curve.age_from, curve.mean_dK * 100, yerr=curve.ci * 100,
                   fmt="o-", color="#2f6f8e", capsize=3, label="stabilized (Δ of shrunk K%)")
    ax[0].plot(curve.age_from, curve.mean_dK_raw * 100, "s--", color="#c1121f",
               alpha=.6, label="naive (k2 − k1)")
    for _, r in curve.iterrows():
        ax[0].annotate(f"{int(r.n)}", (r.age_from, r.mean_dK * 100),
                       textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax[0].set(title="One-year K% change by age  (PT-weighted)",
              xlabel="age in year Y", ylabel="Δ K%  (points)")
    ax[0].legend(fontsize=8)

    ax[1].plot(curve.age_from + .5, curve.cum_dK * 100, "o-", color="#2f6f8e",
               label="stabilized")
    ax[1].plot(curve.age_from + .5, curve.cum_dK_raw * 100, "s--", color="#c1121f",
               alpha=.6, label="naive")
    ax[1].set(title="Cumulative aging curve (peak = 0)", xlabel="age",
              ylabel="cumulative Δ K%  (points)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/k26_1_aging_curve.png", bbox_inches="tight")
    plt.close(fig)

    delta_by_age = curve.set_index("age_from")["mean_dK"]   # one-year increment, RTM-net
    return curve.round(4), delta_by_age


# --------------------------------------------------------------------------- #
#  3. interactions: Stuff+ x arsenal on next-year K% (beyond RTM + aging)
# --------------------------------------------------------------------------- #
def interactions(cp: pd.DataFrame, base_model, delta_by_age: pd.Series):
    d = cp.dropna(subset=PREDS + ["k_stab1", "k2", "w"]).copy()
    d["age_delta"] = d.age1.map(delta_by_age).fillna(0.0)
    d["resid"] = d.k2 - base_model.predict(d) - d.age_delta   # change beyond RTM + aging
    for p in PREDS:
        d[p + "_z"] = _z(d[p])

    f = ("resid ~ stuff_z + n_pitch_z + n_fb_z + n_bb_z"
         " + stuff_z:n_pitch_z + stuff_z:n_fb_z + stuff_z:n_bb_z")
    m = _wols(f, d)
    coef = m.summary2().tables[1].round(5)

    # per-level straight-line fit of resid on Stuff+ (weighted), over the 10-90
    # pct Stuff+ range, with a 95% band.  Slope difference across levels == the
    # interaction; lowess wiggles are not the story.
    slo, shi = d.stuff.quantile([.10, .90])
    xg = np.linspace(slo, shi, 40)
    slopes = {}
    cols = ["#8ecae6", "#4b9cc4", "#1f6f8b", "#023047"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for a, p in zip(ax, ["n_pitch", "n_fb", "n_bb"]):
        vc = d[p].value_counts()
        levels = sorted(v for v in vc.index if vc[v] >= 80)
        if len(levels) > 4:
            levels = sorted(vc[vc.index.isin(levels)].nlargest(4).index)
        srow = []
        for lvl, col in zip(levels, cols):
            s = d[d[p] == lvl]
            fit = _wols("I(resid*100) ~ stuff", s)
            pr = fit.get_prediction(pd.DataFrame({"stuff": xg})).summary_frame(alpha=.05)
            a.plot(xg, pr["mean"], lw=2.5, color=col,
                   label=f"{lvl:g}  (n={len(s)}, slope={fit.params['stuff']*10:+.2f})")
            a.fill_between(xg, pr["mean_ci_lower"], pr["mean_ci_upper"],
                           color=col, alpha=.12)
            srow.append((lvl, fit.params["stuff"] * 10, fit.pvalues["stuff"], len(s)))
        slopes[p] = pd.DataFrame(srow, columns=[p, "dK%_per_+10_Stuff+", "p", "n"])
        a.axhline(0, color="gray", lw=1)
        a.set(title=f"Stuff+  x  {LBL[p]}", xlabel="Stuff+  year Y")
        a.legend(fontsize=8, title=f"{LBL[p]}  (K% pts / +10 Stuff+)")
    ax[0].set_ylabel("next-year K%  vs. RTM+aging expectation  (pts)")
    fig.suptitle("Does the Stuff+ → next-year-K% slope change with arsenal breadth?", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/k26_3_interactions.png", bbox_inches="tight")
    plt.close(fig)
    return m, coef, slopes


# --------------------------------------------------------------------------- #
#  projection: ONE reliability-weighted year-over-year regression
# --------------------------------------------------------------------------- #
def _yoy_weight(d, kind):
    r1 = d.tbf1 / (d.tbf1 + STAB_TBF)
    r2 = d.tbf2 / (d.tbf2 + STAB_TBF)
    return {"reliability": r2,            # reliability of the season being predicted
            "tbf": d.tbf2,               # raw sample size
            "rel_product": r1 * r2}[kind]  # also down-weight noisy year-Y predictor


def _yoy_design(frame):
    d = frame.copy()
    d["y"] = d.k2 * 100                          # next-year K%, in points
    d["base"] = d.k_stab1 * 100                  # stabilized year-Y K% (kills EIV attenuation)
    d["age_c"] = d.age1 - 29                     # linear age, centered at 29
    d["age_decline"] = np.maximum(d.age1 - 30, 0)  # extra slope after 30
    return d


YOY_FORMULA = ("y ~ base + age_c + age_decline"
               " + stuff_z + n_pitch_z + n_fb_z + n_bb_z"
               " + stuff_z:n_pitch_z + stuff_z:n_fb_z + stuff_z:n_bb_z")


def yoy_model(cp: pd.DataFrame, weight: str = "reliability"):
    import statsmodels.formula.api as smf
    d = _yoy_design(cp.dropna(subset=PREDS + ["k_stab1", "k2", "tbf2", "age1"]))
    stats = {p: (d[p + "1"].mean(), d[p + "1"].std()) for p in PREDS}
    for p in PREDS:
        d[p + "_z"] = (d[p + "1"] - stats[p][0]) / stats[p][1]
    d["wt"] = _yoy_weight(d, weight)
    m = smf.wls(YOY_FORMULA, data=d, weights=d["wt"]).fit()

    wr = _wcorr(m.fittedvalues, d.y, d.wt)
    wrmse = np.sqrt(np.average((d.y - m.fittedvalues) ** 2, weights=d.wt))
    base_only = smf.wls("y ~ base", data=d, weights=d["wt"]).fit()
    wrmse0 = np.sqrt(np.average((d.y - base_only.fittedvalues) ** 2, weights=d.wt))

    coef = m.summary2().tables[1].round(4)
    fit_tab = pd.DataFrame({
        "model": ["stabilized K%_Y only", "+ age + Stuff+ + arsenal + interactions"],
        "wRMSE_Kpts": [wrmse0, wrmse], "wcorr": [_wcorr(base_only.fittedvalues, d.y, d.wt), wr],
    }).round(4)
    return m, stats, coef, fit_tab


def _backtest_yoy(cp, weight):
    """Honest check: fit on couplets with year Y <= 2022, predict the held-out
    2023->2024 and 2024->2025 couplets; compare to the stabilized-K% baseline."""
    import statsmodels.formula.api as smf
    tr = _yoy_design(cp[cp.season <= 2022].dropna(subset=PREDS + ["k_stab1", "k2", "tbf2"]))
    te = _yoy_design(cp[cp.season >= 2023].dropna(subset=PREDS + ["k_stab1", "k2", "tbf2"]))
    if len(tr) < 200 or len(te) < 100:
        return pd.DataFrame()
    st = {p: (tr[p + "1"].mean(), tr[p + "1"].std()) for p in PREDS}
    for d in (tr, te):
        for p in PREDS:
            d[p + "_z"] = (d[p + "1"] - st[p][0]) / st[p][1]
    tr["wt"] = _yoy_weight(tr, weight)
    m = smf.wls(YOY_FORMULA, data=tr, weights=tr["wt"]).fit()
    w = _yoy_weight(te, weight)
    err_full = te.y - m.predict(te)
    err_base = te.y - te.base
    return pd.DataFrame({
        "held-out predictor": ["stabilized K%_2023/24 (baseline)", "full YoY model"],
        "wRMSE_Kpts": [np.sqrt(np.average(err_base ** 2, weights=w)),
                       np.sqrt(np.average(err_full ** 2, weights=w))],
        "wcorr": [_wcorr(te.base, te.y, w), _wcorr(m.predict(te), te.y, w)],
        "n_test": [len(te), len(te)],
    }).round(4)


def project_2026(o, model, stats, target_year=2026):
    src = _yoy_design(o[o.season == target_year - 1]
                      .rename(columns={"k_stab": "k_stab1"})
                      .assign(k2=np.nan, age1=lambda x: x.age)
                      .dropna(subset=PREDS + ["k_stab1"]))
    if src.empty:
        return pd.DataFrame()
    for p in PREDS:
        src[p + "_z"] = (src[p] - stats[p][0]) / stats[p][1]
    src["k26_hat"] = model.predict(src) / 100
    src["delta_vs_2025"] = src["k26_hat"] - src["k_pct"]
    out = (src[["pid", "age", "k_pct", "k_stab1", "k26_hat", "delta_vs_2025", "tbf"] + PREDS]
           .rename(columns={"k_pct": f"k_{target_year-1}",
                            "k_stab1": f"k_{target_year-1}_stab",
                            "k26_hat": f"k_{target_year}_hat",
                            "tbf": f"tbf_{target_year-1}"})
           .sort_values(f"k_{target_year}_hat", ascending=False)
           .reset_index(drop=True))
    return out.round(4)


# --------------------------------------------------------------------------- #
def _yoy_coef_plot(coef):
    c = coef.drop("Intercept")
    b = c.loc["base", "Coef."] if "base" in c.index else np.nan
    order = ["age_c", "age_decline", "stuff_z", "n_pitch_z", "n_fb_z", "n_bb_z",
             "stuff_z:n_pitch_z", "stuff_z:n_fb_z", "stuff_z:n_bb_z"]
    order = [o for o in order if o in c.index][::-1]
    c = c.loc[order]
    lo, hi = c["[0.025"], c["0.975]"]
    colors = ["#c1121f" if ":" in n else ("#3a7d44" if n.startswith("age")
              else "#2f6f8e") for n in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(c["Coef."], range(len(c)), xerr=[c["Coef."] - lo, hi - c["Coef."]],
                fmt="o", color="black", ecolor="gray", capsize=3, zorder=3)
    ax.scatter(c["Coef."], range(len(c)), c=colors, s=70, zorder=4)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels([n.replace("_z", "").replace(":", " × ")
                        .replace("age_c", "age (per yr, linear)")
                        .replace("age_decline", "age >30 (extra per yr)") for n in order])
    ax.set_xlabel("Δ next-year K% points   (per +1 SD for z-scored terms)")
    ax.set_title("Unified year-over-year K% model — reliability-weighted WLS\n"
                 f"carry-over slope on stabilized K%$_Y$ = {b:.2f}   "
                 "(green = aging · blue = arsenal · red = Stuff+ × arsenal)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/k26_4_yoy_model.png", bbox_inches="tight")
    plt.close(fig)


def run(df: pd.DataFrame, verbose: bool = True, weight: str = "reliability",
        stab_tbf=None) -> dict:
    global STAB_TBF
    os.makedirs(OUT, exist_ok=True)
    o = prep(df)

    # --- reliability: estimate n0 from the data instead of hard-coding 70 ---
    if stab_tbf is None:
        STAB_TBF, n0_info = estimate_n0(o)
    else:
        STAB_TBF, n0_info = float(stab_tbf), {"note": "user-supplied"}
    o["k_stab"] = (o.so + STAB_TBF * o.lg_k) / (o.tbf + STAB_TBF)   # recompute with n0
    cp = couplets(o)
    rel = reliability_curve(o, cp, STAB_TBF)

    # --- descriptive pieces (figures 1-3) ---
    base_model, pers = persistence(cp)
    curve, delta_by_age = aging_curve(cp, base_model)
    inter_model, coef, slopes = interactions(cp, base_model, delta_by_age)

    # --- the projection engine: one reliability-weighted YoY regression ---
    m, stats, yoy_coef, fit_tab = yoy_model(cp, weight=weight)
    _yoy_coef_plot(yoy_coef)
    backtest = _backtest_yoy(cp, weight)
    proj = project_2026(o, m, stats)

    res = {"n_seasons": len(o), "n_couplets": len(cp),
           "seasons": sorted(o.season.unique().tolist()),
           "weight": weight, "n0": round(STAB_TBF, 1), "n0_info": n0_info,
           "reliability": rel,
           "aging_curve": curve, "persistence": pers,
           "interaction_coefs": coef, "stuff_slopes_by_level": slopes,
           "yoy_coefs": yoy_coef, "yoy_fit": fit_tab, "yoy_backtest": backtest,
           "yoy_model": m, "proj_2026": proj}

    if verbose:
        pd.set_option("display.width", 170, "display.max_columns", 40)
        print(f"seasons {res['seasons']}   pitcher-seasons {res['n_seasons']}   "
              f"year-to-year couplets {res['n_couplets']}   weight = {weight}")
        print(f"\n-- reliability: estimated n0 = {STAB_TBF:.0f} TBF "
              f"(measurement); year-over-year n0 = {rel['n0_year_over_year']}  "
              f"[true-talent SD = {n0_info.get('sd_true_K%','?')}]\n"
              f"{rel['empirical_r_by_bucket'].to_string(index=False)}")
        ac = curve.assign(dK_net_pts=(curve.mean_dK*100).round(2),
                          dK_naive_pts=(curve.mean_dK_raw*100).round(2),
                          cum_net_pts=(curve.cum_dK*100).round(2))
        print(f"\n-- aging curve: one-year dK% points by age (stabilized vs naive) --\n"
              f"{ac[['age_from','dK_net_pts','dK_naive_pts','cum_net_pts','n']].to_string(index=False)}")
        print(f"\n-- persistence (year Y -> Y+1) --\n{pers}")
        print(f"\n== UNIFIED YoY MODEL (reliability-weighted; this is the projection) ==")
        print(yoy_coef)
        print(f"\nin-sample fit:\n{fit_tab}")
        if not backtest.empty:
            print(f"\nholdout (train Y<=2022, test 2023-24 -> 2024-25):\n{backtest}")
        if not proj.empty:
            print(f"\n-- 2026 K% projection (top 10 / bottom 5) --\n"
                  f"{proj.head(10)}\n...\n{proj.tail(5)}")
        print(f"\nStuff+ x arsenal slopes (descriptive):")
        for p, t in slopes.items():
            print(t.to_string(index=False))
        print(f"\nfigures -> {OUT}/k26_1_aging_curve.png .. k26_4_yoy_model.png")
    return res


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/k_arsenal_merged.csv"
    run(pd.read_csv(path))
