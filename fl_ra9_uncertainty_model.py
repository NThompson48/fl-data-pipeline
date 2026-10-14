"""
fl_ra9_uncertainty_model.py -- information-dependent uncertainty for
P(WAR>0).

The shipped probability used ONE fixed SD (RA9_RESID_SD=1.55, a LOOCV RMSE from
the BF>=100 training population) for every pitcher, so a 5-PA pitcher and a
500-PA pitcher got the same bell curve, and pitchers with no data (league-
average inputs, hence a flattering RA9 point estimate) looked MORE likely to
beat replacement than the average known pitcher.

This fits, on real 2025-inputs -> 2026-RA9 forecast errors for the BROAD
population (anyone with >=MIN_IP 2026 innings, including 2026 debuts with no
FL history -- not just the BF>=100 fit sample), a variance model

    Var(RA9_actual - forecast) = a  +  b / IP_2026  +  c * (1 - r_prior)

    a                 irreducible forecast error of TRUE talent given full info
                      (model misspecification + unexplained talent variation)
    b / IP_2026       single-season luck: shrinks with innings actually thrown
    c * (1 - r_prior) the part of true-talent uncertainty that comes from how
                      little we KNOW about this pitcher, r_prior = PA/(PA+n0)

The talent-level SD used for P(true RA9 < replacement) is sqrt(a + c(1-r)) --
the luck term b/IP is deliberately excluded: WAR>0 here means "is this pitcher
genuinely better than replacement", not "will his next 40 innings happen to
look good". The full outcome-level variance (with b/IP) is validated below by
checking standardized-error coverage by information bucket.

All forecasts are genuinely out-of-sample: K%/BB% inputs are cross-fitted
projections (fl_ra9_input_robustness.oof_kbb_projections) and the RA9 baseline
itself is K-fold cross-fitted on its own BF>=100 training sample.

Also fits the Air% trait model's held-out error as a function of how much
pitch-tracking data a pitcher has, for propagation into the Trustmark scenario.

Usage:  python fl_ra9_uncertainty_model.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import KFold

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_ra9_trustmark_projection as ra9
import fl_ra9_input_robustness as rob
from fl_supabase import read_sql_df

MIN_IP = 10      # broad-sample floor for a 2026 RA9 target to be usable at all
N0_PRIOR_PA = 60.0  # ~ the K/BB n0 (59/62, estimated by mm.estimate_n0)


def fit_variance(e, ip, r):
    """MLE of Var = a + b/ip + c*(1-r), all coefficients >= 0."""
    e, ip, r = map(np.asarray, (e, ip, r))

    def nll(p):
        v = p[0] + p[1] / ip + p[2] * (1 - r)
        return 0.5 * np.sum(np.log(v) + e ** 2 / v)

    best = None
    for x0 in ([1.0, 5.0, 0.5], [0.3, 10.0, 1.0], [1.5, 1.0, 0.1]):
        res = minimize(nll, x0, bounds=[(1e-6, None)] * 3, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best


def bootstrap_params(e, ip, r, n=300, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(e), len(e))
        out.append(fit_variance(e[idx], ip[idx], r[idx])[0])
    return np.array(out)


def build_forecasts(raw, war):
    w25 = (bt.START_2025, bt.END_2025)
    oof = rob.oof_kbb_projections(raw, [w25], [(bt.START_2026, bt.CUTOFF_2026)])

    a = war[war["year"] == 2025].set_index("player")
    b = war[(war["year"] == 2026)].copy()
    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    uniq = names[~names["pitcher"].duplicated(keep=False)].rename(columns={"pitcher": "player"})
    b = b.merge(uniq, on="player", how="inner")
    b = b[b["ip"] >= MIN_IP].copy()
    b["bf_25"] = b["player"].map(a["bf"]); b["ip_25"] = b["player"].map(a["ip"]); b["g_25"] = b["player"].map(a["g"])
    b["so_25"] = b["player"].map(a["so"]); b["bb_25"] = b["player"].map(a["bb"])

    # K/BB inputs: OOF model projection where one exists (had 2025 FL PA);
    # otherwise the other-league bridge rate / league mean, exactly what the
    # production fallback does for a pitcher with no FL history.
    br = bt.bridge_features(b["pitcher_id"], raw, current_year=2025)
    pool25 = fp.filter_period(raw, *w25)
    t25 = pool25[pool25["is_terminal"]]
    lg_k, lg_bb = (t25["k_or_bb"] == "strikeout").mean(), (t25["k_or_bb"] == "walk").mean()
    b = b.merge(oof[["pitcher_id", "K_oof", "BB_oof"]], on="pitcher_id", how="left").merge(br, on="pitcher_id", how="left")
    b["K_pred"] = b["K_oof"].fillna(b["fl_k_rate_bridge"]).fillna(lg_k)
    b["BB_pred"] = b["BB_oof"].fillna(b["fl_bb_rate_bridge"]).fillna(lg_bb)
    kbb25 = fp.build_period_kbb(pool25, min_pa=1)[["pitcher_id", "PA"]].rename(columns={"PA": "PA_25"})
    b = b.merge(kbb25, on="pitcher_id", how="left")
    b["PA_25"] = b["PA_25"].fillna(0)
    b["r_prior"] = b["PA_25"] / (b["PA_25"] + N0_PRIOR_PA)

    ppa = ra9.pitches_per_pa(raw, *w25).rename(columns={"pitches_per_pa": "pitches_per_pa_prior"})
    b = b.merge(ppa, left_on="player", right_on="pitcher", how="left").drop(columns="pitcher")
    b["ip_per_g_prior"] = (b["ip_25"].fillna(0) + ra9.N0_IPG * ra9.LG_IPG) / (b["g_25"].fillna(0) + ra9.N0_IPG)
    b["target"] = b["ra9"]
    w26 = war[(war["year"] == 2026) & (war["ip"] >= 20)]
    league_ppf = float(np.average(w26["ppf"], weights=w26["ip"]))
    league_ipg = float(np.average(w26["ip"] / w26["g"], weights=w26["ip"]))
    thin = b["bf_25"].isna() | (b["bf_25"] < 100)
    b["thin"] = thin
    b.loc[thin, "ppf"] = league_ppf
    b.loc[thin, "ip_per_g_prior"] = league_ipg
    b["pitches_per_pa_prior"] = b["pitches_per_pa_prior"].fillna(b["pitches_per_pa_prior"].mean())
    return b


def crossfit_ra9(b, raw, war):
    """RA9 baseline (shipped spec, trained on raw prior rates exactly as shipped)
    K-fold cross-fitted on the BF>=100 fit sample; full-fit for outsiders."""
    w25 = (bt.START_2025, bt.END_2025)
    fit = rob.build_frame(war, raw, pd.DataFrame({"pitcher_id": b["pitcher_id"], "K_oof": 0.0, "BB_oof": 0.0}),
                          2025, 2026, w25)
    fit["K_pred"], fit["BB_pred"] = fit["K_raw"], fit["BB_raw"]

    def fit_model(d):
        aux = smf.ols(ra9.PPA_RESID_SPEC, data=d).fit()
        d = d.assign(ppa_resid=aux.resid)
        return smf.wls(ra9.RA9_SPEC, data=d, weights=d["w"]).fit(), aux, d

    def predict(model, aux, train, d):
        d = d.copy()
        d["ppa_resid"] = d["pitches_per_pa_prior"] - aux.predict(d)
        d.loc[d["thin"], "ppa_resid"] = 0.0
        for c in ra9.CONTINUOUS_REGRESSORS:
            d[c] = d[c].clip(train[c].min(), train[c].max())
        return model.predict(d).values

    b = b.copy()
    b["pred"] = np.nan
    model, aux, trd = fit_model(fit)
    out_mask = ~b["pitcher_id"].isin(fit["pitcher_id"])
    b.loc[out_mask, "pred"] = predict(model, aux, trd, b[out_mask])
    kf = KFold(10, shuffle=True, random_state=0)
    for tri, tei in kf.split(fit):
        m_, aux_, trd_ = fit_model(fit.iloc[tri])
        te_ids = fit.iloc[tei]["pitcher_id"]
        sel = b["pitcher_id"].isin(te_ids)
        b.loc[sel, "pred"] = predict(m_, aux_, trd_, b[sel])
    print(f"  broad sample n={len(b)}  (fit sample n={len(fit)}, thin/no-history n={int(b['thin'].sum())})")
    return b


def coverage_table(b, sd_col, label):
    z = (b["target"] - b["pred"]) / b[sd_col]
    b = b.assign(z=z, pit=norm.cdf(z))
    b["bucket"] = pd.cut(b["PA_25"], [-1, 0, 50, 150, 10000], labels=["no 2025 FL", "1-50", "51-150", "150+"])
    rows = []
    for k, g in b.groupby("bucket", observed=True):
        rows.append({"bucket": k, "n": len(g), "z_sd": round(g["z"].std(), 2),
                     "cov50": round(((g.pit > .25) & (g.pit < .75)).mean(), 2),
                     "cov80": round(((g.pit > .10) & (g.pit < .90)).mean(), 2),
                     "cov90": round(((g.pit > .05) & (g.pit < .95)).mean(), 2)})
    rows.append({"bucket": "ALL", "n": len(g := b), "z_sd": round(b["z"].std(), 2),
                 "cov50": round(((b.pit > .25) & (b.pit < .75)).mean(), 2),
                 "cov80": round(((b.pit > .10) & (b.pit < .90)).mean(), 2),
                 "cov90": round(((b.pit > .05) & (b.pit < .95)).mean(), 2)})
    print(f"\n{label}"); print(pd.DataFrame(rows).to_string(index=False))


def air_error_model(raw):
    """Held-out error of the Air% trait model, by pitch-tracking sample size
    (LOOCV of the shipped AIR_SPEC on 2025 traits -> 2026 Air%)."""
    r25 = fp.filter_period(raw, bt.START_2025, bt.END_2025)
    rawf = ra9._raw_pitch_features(r25)
    bb = pd.read_csv("output/batted_ball_event_residuals.csv").dropna(subset=["pitcher_id"])
    bb["pitcher_id"] = bb["pitcher_id"].astype("int64").astype(str)
    bb["angle_air"] = bb["angle"] >= 10
    a26 = bb[bb["year"] == 2026].groupby("pitcher_id").agg(air_pct=("angle_air", "mean"), n_bip=("angle_air", "size")).reset_index()
    a26 = a26[a26["n_bip"] >= 30]
    d = ra9._shrink_air_features(rawf).merge(a26, on="pitcher_id", how="inner")
    d["w"] = d["n_bip"] / (d["n_bip"] + 60)  # BIP-reliability weight for the outcome side
    preds = np.full(len(d), np.nan)
    for i in range(len(d)):
        m = smf.ols(ra9.AIR_SPEC, data=d.drop(index=i)).fit()
        preds[i] = float(m.predict(d.loc[[i]]).iloc[0])
    e = d["air_pct"].values - preds
    # outcome noise: binomial sampling of the target Air% itself
    samp_var = d["air_pct"] * (1 - d["air_pct"]) / d["n_bip"]
    talent_var = max(float(np.average(e ** 2 - samp_var, weights=d["n_bip"])), 1e-6)
    print(f"\nAir% trait model LOOCV (n={len(d)}): RMSE={np.sqrt(np.mean(e**2)):.4f}, "
          f"mean sampling var of target={samp_var.mean():.5f}, implied TALENT-level error SD={np.sqrt(talent_var):.4f}, "
          f"sd(air_pct)={d['air_pct'].std():.4f}")
    return float(np.sqrt(talent_var)), float(d["air_pct"].std())


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    war = read_sql_df("select player, year, bf, ip, so, bb, ra, ra9, rar, ppf, g "
                      "from war_pitchers where game_type='reg_season'")
    b = build_forecasts(raw, war)
    b = crossfit_ra9(b, raw, war)
    b["e"] = b["target"] - b["pred"]
    print(f"  forecast error: mean={b['e'].mean():+.3f}  (IP-weighted {np.average(b['e'], weights=b['ip']):+.3f})")

    p, res = fit_variance(b["e"], b["ip"], b["r_prior"])
    bs = bootstrap_params(b["e"].values, b["ip"].values, b["r_prior"].values)
    print("\nVariance model  Var = a + b/IP + c*(1-r_prior)")
    for nme, v, col in zip("abc", p, bs.T):
        print(f"  {nme} = {v:.3f}   90% bootstrap CI [{np.percentile(col, 5):.3f}, {np.percentile(col, 95):.3f}]")
    print(f"  talent SD: fully-known r=1 -> {np.sqrt(p[0]):.3f};  no info r=0 -> {np.sqrt(p[0] + p[2]):.3f}"
          f"   (shipped fixed SD was {ra9.RA9_RESID_SD})")

    b["sd_outcome"] = np.sqrt(p[0] + p[1] / b["ip"] + p[2] * (1 - b["r_prior"]))
    b["sd_fixed"] = ra9.RA9_RESID_SD
    coverage_table(b, "sd_fixed", "SHIPPED fixed SD=1.55 -- standardized error coverage by prior info")
    coverage_table(b, "sd_outcome", "NEW information-dependent SD (outcome level, incl. luck term)")

    # does the point forecast itself show the "no-data pitchers look good" artifact?
    g = b.groupby(pd.cut(b["PA_25"], [-1, 0, 50, 150, 10000], labels=["no 2025 FL", "1-50", "51-150", "150+"]), observed=True)
    print("\nForecast vs actual RA9 by prior-information bucket (mean over pitchers, IP-weighted actual):")
    print(g.apply(lambda d: pd.Series({"n": len(d), "pred": d["pred"].mean(),
                                       "actual_ipw": np.average(d["target"], weights=d["ip"]),
                                       "bias_ipw": np.average(d["e"], weights=d["ip"])})).round(3).to_string())

    sd_air_talent, sd_air_pop = air_error_model(raw)
    print(f"\nPARAMS for production: a={p[0]:.4f} b={p[1]:.4f} c={p[2]:.4f} N0_PRIOR_PA={N0_PRIOR_PA} "
          f"AIR_TALENT_ERR_SD={sd_air_talent:.4f} AIR_POP_SD={sd_air_pop:.4f}")
