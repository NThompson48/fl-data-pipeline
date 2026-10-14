"""
mlb_arsenal_depth_wh_sva_aging.py -- two follow-up aging questions:

1. Arsenal depth (# distinct pitch types actually thrown, by family): does it
   explain aging heterogeneity the role/arsenal-type splits in
   mlb_aging_exploration.py didn't catch, and does it add real out-of-sample
   value to the walk-forward model (k26_projection.py's original Stuff+ x
   arsenal-breadth interaction, revisited with real arsenal-depth counts this
   project never had built until now)?

2. Does wh_sva ITSELF decline with age (i.e. is arsenal-shape erosion the
   actual mechanism behind the K% aging curve)? And can an early-season wh_sva
   surprise -- actual first-half wh_sva vs. what the pitcher's trailing level
   + age would predict -- correct the preseason projection, by predicting
   that SAME season's second half beyond what the preseason model alone says?

Usage:
    python mlb_arsenal_depth_wh_sva_aging.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_aging_exploration import _wmean_ci, build_couplets, empirical_age_curve
from mlb_kbb_model import MIN_PA, PANEL_PATH

pd.set_option("display.width", 160, "display.max_columns", 40)

MIN_USAGE_PCT = 0.05
MIN_USAGE_N = 25


def compute_arsenal_depth(raw: pd.DataFrame) -> pd.DataFrame:
    """# distinct pitch types a pitcher actually uses in a season (>=5% share AND
    >=25 pitches, to exclude incidental/mistake pitches), overall and by family."""
    counts = raw.dropna(subset=["pitch_type"]).groupby(
        ["pitcher", "game_year", "pitch_type"]).size().reset_index(name="n")
    totals = counts.groupby(["pitcher", "game_year"])["n"].sum().rename("total").reset_index()
    counts = counts.merge(totals, on=["pitcher", "game_year"])
    counts["pct"] = counts["n"] / counts["total"]
    counts["qualifies"] = (counts["pct"] >= MIN_USAGE_PCT) & (counts["n"] >= MIN_USAGE_N)
    counts["family"] = counts["pitch_type"].map(mf.PITCH_FAMILY)
    q = counts[counts["qualifies"]]

    n_pitch = q.groupby(["pitcher", "game_year"]).size().rename("n_pitch_types")
    n_fb = q[q["family"] == "FB"].groupby(["pitcher", "game_year"]).size().rename("n_fb_types")
    n_brk = q[q["family"] == "BRK"].groupby(["pitcher", "game_year"]).size().rename("n_brk_types")
    n_off = q[q["family"] == "OFF"].groupby(["pitcher", "game_year"]).size().rename("n_off_types")
    out = pd.concat([n_pitch, n_fb, n_brk, n_off], axis=1).reset_index()
    for c in ["n_pitch_types", "n_fb_types", "n_brk_types", "n_off_types"]:
        out[c] = out[c].fillna(0).astype(int)
    return out


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)

    panel = pd.read_parquet(PANEL_PATH)
    elig = panel[panel["PA"] >= MIN_PA]
    n0_k, _ = mm.estimate_n0(elig["K_rate"], elig["PA"])
    n0_bb, _ = mm.estimate_n0(elig["BB_rate"], elig["PA"])

    # ================================================================= #
    #  PART 1: arsenal depth
    # ================================================================= #
    print("\n" + "=" * 70 + "\nPART 1: ARSENAL DEPTH\n" + "=" * 70)
    depth = compute_arsenal_depth(raw)
    print(depth["n_pitch_types"].value_counts().sort_index())

    panel_d = panel.merge(depth, on=["pitcher", "game_year"], how="left")
    elig_d = panel_d[panel_d["PA"] >= MIN_PA]
    print(f"\nStandalone corr with concurrent K_rate: n_pitch_types={elig_d['n_pitch_types'].corr(elig_d['K_rate']):.3f}, "
          f"n_fb_types={elig_d['n_fb_types'].corr(elig_d['K_rate']):.3f}, "
          f"n_brk_types={elig_d['n_brk_types'].corr(elig_d['K_rate']):.3f}, "
          f"n_off_types={elig_d['n_off_types'].corr(elig_d['K_rate']):.3f}")
    print(f"Standalone corr with concurrent BB_rate: n_pitch_types={elig_d['n_pitch_types'].corr(elig_d['BB_rate']):.3f}")

    cp = build_couplets(elig_d, n0_k, n0_bb)
    tercile = cp["n_pitch_types1"].quantile([1/3, 2/3]).values
    cp["depth_group"] = pd.cut(cp["n_pitch_types1"], bins=[-1, tercile[0], tercile[1], 99],
                                labels=["shallow", "medium", "deep"])
    print(f"\narsenal-depth terciles (# pitch types, year Y): {tercile}")
    for grp, g in cp.groupby("depth_group", observed=True):
        curve = empirical_age_curve(g, "dK_stab")
        cum_final = curve["cum_delta"].iloc[-1] if len(curve) else np.nan
        print(f"  K% aging, {grp:8s} arsenal (n_couplets={len(g):4d}): cumulative decline to oldest bin = {cum_final:+.4f}")

    # does arsenal depth + age-interaction earn its keep in the actual walk-forward model?
    from mlb_kbb_model import (BB_FEATURES, FOLDS, K_FEATURES_SWSTR, _zscore_apply,
                                _zscore_fit, build_model_frame)
    mframe = build_model_frame(panel_d)
    mframe = mframe.merge(depth.rename(columns={"game_year": "target_year"}).assign(
        target_year=lambda d: d["target_year"] + 1), on=["pitcher", "target_year"], how="left")
    for c in ["n_pitch_types", "n_fb_types", "n_brk_types", "n_off_types"]:
        mframe[c] = mframe[c].fillna(0)

    def fit_eval_with_depth(train, test, feature_cols, n0, use_depth, use_interaction):
        d = train.dropna(subset=["Marcel_K_rate", "K_rate", "PA"]).copy()
        d["y"] = d["K_rate"] * 100
        d["base"] = d["Marcel_K_rate"] * 100
        d["wt"] = mm.reliability_weight(d["PA"], n0)
        extra = feature_cols + (["n_pitch_types"] if use_depth else [])
        stats = _zscore_fit(d, extra)
        d = _zscore_apply(d, extra, stats)
        zcols = [f"{c}_z" for c in extra]
        rhs = ["base", "age_c", "age_decline"] + zcols
        if use_interaction:
            rhs.append("age_c:n_pitch_types_z")
        formula = "y ~ " + " + ".join(rhs)
        model = smf.wls(formula, data=d, weights=d["wt"]).fit()
        dt = test.dropna(subset=["Marcel_K_rate", "K_rate", "PA"]).copy()
        dt["base"] = dt["Marcel_K_rate"] * 100
        dt = _zscore_apply(dt, extra, stats)
        pred = model.predict(dt).values / 100
        return mm.evaluate_weighted(dt["K_rate"].values, pred, dt["PA"].values, n0, label="")

    print("\nWalk-forward K% backtest: does n_pitch_types help?")
    for label, use_depth, use_inter in [("baseline (no depth)", False, False),
                                          ("+ n_pitch_types main effect", True, False),
                                          ("+ n_pitch_types x age interaction", True, True)]:
        rows = []
        for train_years, test_year in FOLDS:
            train = mframe[mframe["target_year"].isin(train_years)]
            test = mframe[mframe["target_year"] == test_year]
            elig_train = train[train["PA"] >= MIN_PA]
            n0_k_fold, _ = mm.estimate_n0(elig_train["K_rate"], elig_train["PA"])
            rows.append(fit_eval_with_depth(train, test, K_FEATURES_SWSTR, n0_k_fold, use_depth, use_inter))
        bt = pd.DataFrame(rows)
        print(f"  {label:36s}  R2_w={bt['R2_w'].mean():.4f}  RMSE_w={bt['RMSE_w_pp'].mean():.3f}pp")

    # ================================================================= #
    #  PART 2: does wh_sva itself age, and can early-season wh_sva correct
    #  the projection?
    # ================================================================= #
    print("\n" + "=" * 70 + "\nPART 2: WH_SVA AGING + EARLY-SEASON CORRECTION\n" + "=" * 70)

    print("Refitting pooled overall wh_sva per-pitch (full history)...")
    wh_pitch = mf.pooled_wh_sva_per_pitch(raw)

    season_sva = wh_pitch.groupby(["pitcher", "game_year"]).agg(
        wh_sva_season=("sva", "mean"), n_sva=("sva", "size")).reset_index()
    season_sva_q = season_sva[season_sva["n_sva"] >= 100]

    # 2a. Does wh_sva LEVEL decline with age? (year-over-year couplets on the raw
    # pooled wh_sva mean, playing-time weighted by pitch count)
    b = season_sva_q.copy()
    b["game_year"] -= 1
    cp_sva = season_sva_q.merge(b, on=["pitcher", "game_year"], suffixes=("1", "2"))
    cp_sva["w"] = 2 / (1 / cp_sva["n_sva1"] + 1 / cp_sva["n_sva2"])
    cp_sva["dsva"] = cp_sva["wh_sva_season2"] - cp_sva["wh_sva_season1"]
    age_lookup = panel[["pitcher", "game_year", "age"]].drop_duplicates()
    cp_sva = cp_sva.merge(age_lookup, on=["pitcher", "game_year"], how="left")
    cp_sva["age1_int"] = cp_sva["age"].round().astype("Int64")
    rows = []
    for age, g in cp_sva.dropna(subset=["age1_int"]).groupby("age1_int"):
        if len(g) < 25:
            continue
        m, ci = _wmean_ci(g["dsva"].values, g["w"].values)
        rows.append((age, m, ci, len(g)))
    curve_sva = pd.DataFrame(rows, columns=["age", "mean_dsva", "ci95", "n"])
    curve_sva["cum"] = curve_sva["mean_dsva"].cumsum()
    print("\nYear-over-year change in wh_sva LEVEL, by age (does raw stuff erode?):")
    print(curve_sva.round(5).to_string(index=False))
    print(f"\ncorr(age, dsva) across all couplets = {cp_sva['age'].corr(cp_sva['dsva']):.3f}  (n={len(cp_sva)})")

    # 2b. Within-season wh_sva trend (second-half - first-half, from the delta
    # pipeline) vs age -- an in-season "cliff" signature?
    deltas_full = pd.read_parquet(PANEL_PATH)[["pitcher", "game_year", "wh_sva_delta", "age"]]
    print(f"\ncorr(age, within-season wh_sva_delta) = "
          f"{deltas_full['age'].corr(deltas_full['wh_sva_delta']):.3f}  (n={deltas_full.dropna().shape[0]})")

    # 2c. Early-season surprise -> predicts the SAME season's second-half K_rate
    # beyond the preseason (Marcel+age) projection?
    print("\nBuilding first-half / second-half K_rate split (same chronological cut as the delta pipeline)...")
    term = raw[raw["is_terminal_pitch"]].copy()
    term["is_k"] = term["events"].astype(str).isin(mf.K_EVENTS)
    term = term.sort_values(["pitcher", "game_year", "game_date", "at_bat_number", "pitch_number"])
    grp = term.groupby(["pitcher", "game_year"])
    rank = grp.cumcount()
    counts = grp["is_k"].transform("size")
    term["_half"] = np.where(rank < counts / 2, "first", "second")
    half_k = term.groupby(["pitcher", "game_year", "_half"]).agg(
        PA=("is_k", "size"), K=("is_k", "sum")).reset_index()
    half_k["K_rate_half"] = half_k["K"] / half_k["PA"]
    piv = half_k.pivot(index=["pitcher", "game_year"], columns="_half", values=["K_rate_half", "PA"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv = piv[(piv["PA_first"] >= 50) & (piv["PA_second"] >= 50)]

    wh_first = wh_pitch[wh_pitch.groupby(["pitcher", "game_year"]).cumcount() <
                         wh_pitch.groupby(["pitcher", "game_year"])["sva"].transform("size") / 2]
    wh_first_agg = wh_first.groupby(["pitcher", "game_year"]).agg(
        wh_sva_first=("sva", "mean"), n_first=("sva", "size")).reset_index()
    wh_first_agg = wh_first_agg[wh_first_agg["n_first"] >= 100]

    d2 = piv.merge(wh_first_agg, on=["pitcher", "game_year"], how="inner")
    d2 = d2.merge(panel[["pitcher", "game_year", "Marcel_K_rate", "age_c", "age_decline"]],
                  on=["pitcher", "game_year"], how="left")
    d2 = d2.merge(season_sva[["pitcher", "game_year", "wh_sva_season"]].rename(
        columns={"game_year": "game_year_prior", "wh_sva_season": "wh_sva_prior_year"}),
        left_on=["pitcher"], right_on=["pitcher"], how="left")
    # keep only the correct prior year (game_year - 1) for the "expected level" baseline
    d2 = d2[d2["game_year_prior"] == d2["game_year"] - 1]
    d2 = d2.dropna(subset=["Marcel_K_rate", "wh_sva_prior_year", "K_rate_half_second"])
    d2["surprise"] = d2["wh_sva_first"] - d2["wh_sva_prior_year"]  # early-season vs. last year's level

    print(f"\nn={len(d2)} pitcher-seasons with both a qualifying early-half wh_sva AND a prior-year wh_sva")
    m_base = smf.wls("I(K_rate_half_second*100) ~ I(Marcel_K_rate*100) + age_c + age_decline",
                      data=d2, weights=d2["PA_second"]).fit()
    m_full = smf.wls("I(K_rate_half_second*100) ~ I(Marcel_K_rate*100) + age_c + age_decline + surprise",
                      data=d2, weights=d2["PA_second"]).fit()
    m_inter = smf.wls("I(K_rate_half_second*100) ~ I(Marcel_K_rate*100) + age_c + age_decline "
                       "+ surprise + surprise:age_decline", data=d2, weights=d2["PA_second"]).fit()

    def wrmse(fit, data):
        return np.sqrt(np.average((data["K_rate_half_second"] * 100 - fit.fittedvalues) ** 2, weights=data["PA_second"]))

    print(f"\npredicting the SAME season's second-half K_rate:")
    print(f"  preseason (Marcel+age) only:                  wRMSE={wrmse(m_base, d2):.3f}pp  "
          f"R2={m_base.rsquared:.4f}")
    print(f"  + early-season wh_sva surprise:                wRMSE={wrmse(m_full, d2):.3f}pp  "
          f"R2={m_full.rsquared:.4f}  (surprise coef={m_full.params['surprise']:.3f}, p={m_full.pvalues['surprise']:.4f})")
    print(f"  + surprise x age_decline interaction:          wRMSE={wrmse(m_inter, d2):.3f}pp  "
          f"R2={m_inter.rsquared:.4f}  (interaction coef={m_inter.params['surprise:age_decline']:.3f}, "
          f"p={m_inter.pvalues['surprise:age_decline']:.4f})")
