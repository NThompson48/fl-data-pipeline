"""
fl_halfseason_backtest.py -- the half-season K%/BB% backtest harness (approved
plan: C:\\Users\\215nt\\.claude\\plans\\let-s-discuss-how-this-merry-thacker.md).

Resolved design (confirmed with the user after the plan was approved):
  - Every "model" is a reliability-weighted WLS regression of an ACTUAL future-
    period rate on a named regressor set (naive prior, other-league bridge rate,
    surprise-variable changepoint) -- the same cross-sectional "fit on one
    transition" convention already used for the college bridge and the MLB
    Marcel/aging work elsewhere in this project. "Baseline" vs "challenger" means
    a smaller vs larger regressor set for the SAME target, not a different
    estimator.
  - Experiments 1 and 2 name an explicit later "Test on X" period and DO have real
    held-out data available (FL has two full seasons; 2026 overall is the most
    recent, but 2H2026 and 1H2026 both still postdate the periods those two
    experiments train on) -- so those two get genuine held-out scoring: fit
    coefficients on the named train->target transition, FREEZE them, roll the
    identical model forward onto the next period's analogous features, and score
    against the actual named test period. This is a real generalization check,
    not a second in-sample fit.
  - Experiments 3 and 4 predict the LAST period for which any data exists (2H2026,
    2026 rest-of-season) -- there is no further season to hold out. Per the user's
    explicit call: for these two, score in-sample (fit stats on the named
    transition itself, same convention as the college bridge / Marcel checks
    throughout this project) and weight the comparison toward which model is
    structurally simplest/most defensible, not just which has the highest in-
    sample R2 (an in-sample-only comparison rewards complexity by construction).
  - K-BB%: for every baseline/challenger, also fit K-BB% directly (its own target,
    its own n0) and compare to K_pred - BB_pred from the separately-fit models --
    a cross-cutting question tested empirically here rather than assumed from the
    separate MLB-notebook finding.

Usage:
    python fl_halfseason_backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_college_bridge as bridge
import fl_bridge_matching as bm
import mlb_marcel as mm

START_2025, CUTOFF_2025, END_2025 = "2025-05-08", "2025-07-04", "2025-08-31"
START_2026, CUTOFF_2026, END_2026 = "2026-05-07", "2026-07-07", "2026-09-06"
_DAY_AFTER_2025_CUT = "2025-07-05"
_DAY_AFTER_2026_CUT = "2026-07-08"

MIN_PA_TEST = 10     # floor for a pitcher-period to count in scoring
MIN_PA_N0 = 20       # floor for a pitcher-period to count toward fitting n0


# --------------------------------------------------------------------------- #
#  Shared building blocks
# --------------------------------------------------------------------------- #
def naive_prior(raw: pd.DataFrame, windows: list[tuple[str, str]]):
    frames = [fp.filter_period(raw, s, e) for s, e in windows]
    pool = pd.concat(frames, ignore_index=True)
    kbb = fp.build_period_kbb(pool, min_pa=1)
    n0_k, _ = mm.estimate_n0(kbb["K_rate"], kbb["PA"], min_n=MIN_PA_N0)
    n0_bb, _ = mm.estimate_n0(kbb["BB_rate"], kbb["PA"], min_n=MIN_PA_N0)
    lg_k = kbb["K"].sum() / kbb["PA"].sum()
    lg_bb = kbb["BB"].sum() / kbb["PA"].sum()
    kbb["naive_K_rate"] = (kbb["K"] + n0_k * lg_k) / (kbb["PA"] + n0_k)
    kbb["naive_BB_rate"] = (kbb["BB"] + n0_bb * lg_bb) / (kbb["PA"] + n0_bb)
    return kbb[["pitcher_id", "PA", "naive_K_rate", "naive_BB_rate"]], n0_k, n0_bb


def actual_rate(raw: pd.DataFrame, windows: list[tuple[str, str]], min_pa: int = MIN_PA_TEST):
    frames = [fp.filter_period(raw, s, e) for s, e in windows]
    pool = pd.concat(frames, ignore_index=True)
    kbb = fp.build_period_kbb(pool, min_pa=min_pa)
    return kbb[["pitcher_id", "PA", "K_rate", "BB_rate"]].rename(
        columns={"PA": "PA_test", "K_rate": "K_rate_actual", "BB_rate": "BB_rate_actual"})


_BRIDGE_MODEL = None
_PROSPECT_PROJ = None
_PIONEER_FULL = None
_MILB_FULL = None
_UNIFIED = None
# Which other-league bridge the harness (and therefore production) uses. "unified" =
# fl_unified_bridge (one policy for College/Pioneer/MiLB: TBF shrinkage, one decay per
# metric, staleness damping, pooled mapping -- DECISION_LOG.md "Unified vs by-league
# bridge: head-to-head"); "legacy" = the three league-specific bridges + the R2 x TBF x
# 0.5^gap source pick (kept so earlier experiment results remain reproducible).
BRIDGE_IMPL = "unified"


def bridge_features(pitcher_ids: pd.Series, raw: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """Cross-league other-league match (college/Pioneer/MiLB, recency+
    reliability+TBF weighted single-best-source pick) -- the SAME match
    production (fl_production_projection.py) ships, via the shared
    fl_bridge_matching.recency_weighted_match. Previously used full_match's
    fixed college>Pioneer>MiLB priority chain and only joined the narrow
    bridge table at that, which silently understated every
    'naive+other_league' R2 this harness has ever reported (see
    fl_bridge_matching.py's docstring and DECISION_LOG.md). `current_year`
    is the most recent completed season the caller's feature window
    represents -- see assemble_features."""
    global _BRIDGE_MODEL, _PROSPECT_PROJ, _PIONEER_FULL, _MILB_FULL, _UNIFIED
    if BRIDGE_IMPL == "unified":
        if _UNIFIED is None:
            import fl_unified_bridge as ub   # local import: ub imports this module
            _UNIFIED = ub.fit(raw, verbose=False)
        names_u = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
        ids_u = pd.DataFrame({"pitcher_id": pitcher_ids.astype(str).unique()}).merge(names_u, on="pitcher_id", how="left")
        feat_u = _UNIFIED.match(ids_u["pitcher_id"], ids_u["pitcher"], current_year=current_year)
        return feat_u[["pitcher_id", "fl_k_rate_bridge", "fl_bb_rate_bridge", "has_college_match",
                       "k_bridge_tbf", "bb_bridge_tbf"]]
    if _BRIDGE_MODEL is None:
        import fl_pioneer_bridge as pbridge  # local: pbridge references bt.START_2025 etc.
        import fl_milb_bridge as mbridge     # at module level -- importing at top would deadlock
        _BRIDGE_MODEL = bridge.fit_bridge()
        _PROSPECT_PROJ = bm.build_college_prospect_projections(_BRIDGE_MODEL)
        _PIONEER_FULL = pbridge.build_full_universe_predictions(pbridge.fit_bridge(raw))
        _MILB_FULL = mbridge.build_full_universe_predictions(mbridge.fit_bridge(raw))

    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    ids_df = pd.DataFrame({"pitcher_id": pitcher_ids.astype(str).unique()}).merge(names, on="pitcher_id", how="left")
    feat = bm.recency_weighted_match(ids_df["pitcher_id"], ids_df["pitcher"], _PROSPECT_PROJ,
                                      _PIONEER_FULL, _MILB_FULL, current_year=current_year)
    return feat[["pitcher_id", "fl_k_rate_bridge", "fl_bb_rate_bridge", "has_college_match",
                 "k_bridge_tbf", "bb_bridge_tbf"]]


def wh_sva_level_and_changepoint(train_df: pd.DataFrame):
    """One fit of the two-stage wh_sva model per training window, reused for BOTH
    the per-pitcher LEVEL (mean sva -- a direct prior, like the already-validated
    concurrent corr=0.50 with K_rate) and the changepoint (shown in DECISION_LOG
    not to generalize) -- avoids fitting the XGBoost model twice for the same
    window."""
    wh_sva_pitch = fp.compute_pooled_wh_sva(train_df)
    if len(wh_sva_pitch) == 0:
        return (pd.DataFrame(columns=["pitcher_id", "wh_sva_level"]),
                pd.DataFrame(columns=["pitcher_id", "wh_sva_cp"]))
    level = wh_sva_pitch.groupby("pitcher_id").agg(wh_sva_level=("sva", "mean")).reset_index()
    cp = fp.detect_changepoint(wh_sva_pitch, "sva")
    cp = fp.shrink_changepoint(cp, wh_sva_pitch, "sva")
    cp = cp[["pitcher_id", "changepoint"]].rename(columns={"changepoint": "wh_sva_cp"})
    return level, cp


def dist_levels(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher training-window mean horiz_dist/vert_dist -- the direct prior
    (already-validated concurrent corr ~0.38 with BB_rate), not the changepoint.
    Kept for the changepoint-based MODEL_SPECS entries and other experiment
    scripts; the live production BB% spec uses loc_var_levels (below)
    instead -- see DECISION_LOG.md "Location variance as a BB% regressor
    replacement"."""
    return fp.compute_dist_features(train_df)[["pitcher_id", "horiz_dist", "vert_dist"]]


def loc_var_levels(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher training-window horiz_var/vert_var (location VARIANCE --
    command consistency, not distance-from-center accuracy) AND
    command_score, the OpenCommand-calibrated translation of the same two
    numbers into inferred-miss inches (fp.compute_command_score). horiz_var/
    vert_var replaced horiz_dist/vert_dist in the production BB% spec after
    beating them on every axis checked (held-out corr, stickiness) in a
    genuinely held-out Experiment-2 test -- see DECISION_LOG.md "Location
    variance as a BB% regressor replacement" (BB% R2_w 0.3274 -> 0.3407).
    command_score then replaced horiz_var/vert_var THEMSELVES after another
    genuinely held-out win -- see DECISION_LOG.md "OpenCommand-calibrated
    Command Score replaces horiz_var/vert_var in the BB% model" (BB% R2_w
    0.3517 -> 0.3610). horiz_var/vert_var are kept in this output (just no
    longer the BB% model's own regressors) since other callers/the
    dashboard still display them."""
    return fp.compute_command_score(train_df)[["pitcher_id", "horiz_var", "vert_var", "command_score"]]


def stuff_levels(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher training-window spin_z_norm/extension -- added to the K%
    spec alongside wh_sva_level after a genuinely held-out win (see
    DECISION_LOG.md "Adding spin_rate + extension as direct K% regressors",
    K% R2_w 0.2686 -> 0.3044). The spin baseline is derived from train_df
    itself (never the later test window), matching the "fit on the training
    transition, freeze" discipline every other regressor in this harness
    already follows."""
    return fp.compute_spin_ext_features(train_df)[["pitcher_id", "spin_z_norm", "extension"]]


def dist_changepoints(train_df: pd.DataFrame) -> pd.DataFrame:
    cp_h = fp.detect_changepoint(train_df, "horiz_dist_raw")
    cp_h = fp.shrink_changepoint(cp_h, train_df, "horiz_dist_raw")
    cp_v = fp.detect_changepoint(train_df, "vert_dist_raw")
    cp_v = fp.shrink_changepoint(cp_v, train_df, "vert_dist_raw")
    out = cp_h[["pitcher_id", "changepoint"]].rename(columns={"changepoint": "horiz_dist_cp"}).merge(
        cp_v[["pitcher_id", "changepoint"]].rename(columns={"changepoint": "vert_dist_cp"}),
        on="pitcher_id", how="outer")
    return out


def fit_weighted(data: pd.DataFrame, target: str, regressors: list[str], weight: np.ndarray):
    d = data.copy()
    d["_w"] = weight
    formula = f"{target} ~ " + " + ".join(regressors)
    return smf.wls(formula, data=d, weights=d["_w"]).fit()


def score(y_true, y_pred, w, label):
    return {"label": label, "n": len(y_true),
            "R2_w": round(mm.weighted_r2(y_true, y_pred, w), 4),
            "RMSE_w_pp": round(mm.weighted_rmse(y_true, y_pred, w) * 100, 3)}


# --------------------------------------------------------------------------- #
#  Generic "fit on train transition, score" (in-sample) and
#  "fit, freeze, roll forward, score on next transition" (held-out) runners
# --------------------------------------------------------------------------- #
def run_in_sample(feat_df: pd.DataFrame, target_col: str, regressors: list[str],
                   n0: float, label: str):
    d = feat_df.dropna(subset=[target_col] + regressors).copy()
    if len(d) < 15:
        return {"label": label, "n": len(d), "R2_w": np.nan, "RMSE_w_pp": np.nan}, None
    w = mm.reliability_weight(d["PA_test"], n0).values
    m = fit_weighted(d, target_col, regressors, w)
    return score(d[target_col].values, m.fittedvalues.values, w, label), m


def run_heldout(train_feat: pd.DataFrame, train_target: str, regressors: list[str],
                 test_feat: pd.DataFrame, test_target: str, n0: float, label: str):
    d_tr = train_feat.dropna(subset=[train_target] + regressors).copy()
    if len(d_tr) < 15:
        return {"label": label, "n": 0, "R2_w": np.nan, "RMSE_w_pp": np.nan}
    w_tr = mm.reliability_weight(d_tr["PA_test"], n0).values
    m = fit_weighted(d_tr, train_target, regressors, w_tr)

    d_te = test_feat.dropna(subset=[test_target] + regressors).copy()
    if len(d_te) < 15:
        return {"label": label, "n": 0, "R2_w": np.nan, "RMSE_w_pp": np.nan}
    pred = m.predict(d_te)
    w_te = mm.reliability_weight(d_te["PA_test"], n0).values
    return score(d_te[test_target].values, pred.values, w_te, label)


# --------------------------------------------------------------------------- #
#  Feature assembly for one window: naive prior + bridge + changepoints.
#  `assemble_features` has no notion of a label and is reused directly by the
#  production projection script (no "actual" exists yet for a future season);
#  `assemble` adds the ACTUAL rate of a (possibly different) label window on
#  top, for backtesting.
# --------------------------------------------------------------------------- #
def assemble_features(raw, feature_windows, with_changepoints=True, with_surprise_levels=True):
    naive, n0_k, n0_bb = naive_prior(raw, feature_windows)
    # current_year for cross-league bridge matching = the latest season this
    # feature window represents (its windows' own end dates) -- a player's
    # college/Pioneer/MiLB record from that same year or earlier is knowable
    # at this point; anything later would leak the future into a backtest.
    current_year = max(int(end[:4]) for _, end in feature_windows)
    br = bridge_features(naive["pitcher_id"], raw, current_year)
    feat_pool = pd.concat([fp.filter_period(raw, s, e) for s, e in feature_windows], ignore_index=True)

    df = naive.merge(br, on="pitcher_id", how="left")
    lg_k_rate = (feat_pool[feat_pool["is_terminal"]]["k_or_bb"] == "strikeout").mean()
    lg_bb_rate = (feat_pool[feat_pool["is_terminal"]]["k_or_bb"] == "walk").mean()
    df["fl_k_rate_bridge"] = df["fl_k_rate_bridge"].fillna(lg_k_rate)
    df["fl_bb_rate_bridge"] = df["fl_bb_rate_bridge"].fillna(lg_bb_rate)
    df["has_college_match"] = df["has_college_match"].fillna(False)
    df["k_bridge_tbf"] = df["k_bridge_tbf"].fillna(0.0)
    df["bb_bridge_tbf"] = df["bb_bridge_tbf"].fillna(0.0)

    wh_level = wh_cp = None
    if with_surprise_levels or with_changepoints:
        wh_level, wh_cp = wh_sva_level_and_changepoint(feat_pool)
        df = df.merge(wh_level, on="pitcher_id", how="left")
        d_lvl = dist_levels(feat_pool)
        df = df.merge(d_lvl, on="pitcher_id", how="left")
        d_var = loc_var_levels(feat_pool)
        df = df.merge(d_var, on="pitcher_id", how="left")
        d_stuff = stuff_levels(feat_pool)
        df = df.merge(d_stuff, on="pitcher_id", how="left")
        # impute missing levels with the training population's own mean (neutral,
        # uninformative value) rather than 0 -- these are raw feet/probability-
        # scale quantities, not deltas, so 0 would be a real (wrong) claim
        for c in ["wh_sva_level", "horiz_dist", "vert_dist", "horiz_var", "vert_var", "command_score",
                  "spin_z_norm", "extension"]:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].mean())

    if with_changepoints:
        d_cp = dist_changepoints(feat_pool)
        df = df.merge(wh_cp, on="pitcher_id", how="left")
        df = df.merge(d_cp, on="pitcher_id", how="left")
        for c in ["wh_sva_cp", "horiz_dist_cp", "vert_dist_cp"]:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)

    return df, n0_k, n0_bb


def assemble(raw, feature_windows, label_windows, with_changepoints=True, with_surprise_levels=True):
    df, n0_k, n0_bb = assemble_features(raw, feature_windows, with_changepoints, with_surprise_levels)
    actual = actual_rate(raw, label_windows)
    df = df.merge(actual, on="pitcher_id", how="inner")
    df["KBB_rate_actual"] = df["K_rate_actual"] - df["BB_rate_actual"]
    return df, n0_k, n0_bb


MODEL_SPECS = {
    "naive": {"K": ["naive_K_rate"], "BB": ["naive_BB_rate"]},
    "naive+surprise_prior": {"K": ["naive_K_rate", "wh_sva_level"],
                              "BB": ["naive_BB_rate", "horiz_dist", "vert_dist"]},
    "naive+changepoint": {"K": ["naive_K_rate", "wh_sva_cp"],
                           "BB": ["naive_BB_rate", "horiz_dist_cp", "vert_dist_cp"]},
    "naive+other_league": {"K": ["naive_K_rate", "fl_k_rate_bridge"],
                            "BB": ["naive_BB_rate", "fl_bb_rate_bridge"]},
    "naive+other_league+surprise_prior": {
        # K% adds spin_z_norm (pitch-type-normalized spin rate) and extension
        # alongside wh_sva_level -- validated genuinely held-out, see
        # DECISION_LOG.md "Adding spin_rate + extension as direct K%
        # regressors" (R2_w 0.2686 -> 0.3044, pitch-type-normalized spin beat
        # the raw mean head-to-head). Neither is individually significant on
        # the smaller fold-1 training sample, but both improve held-out
        # accuracy when frozen and scored forward -- the signature of real
        # (if noisily-estimated) signal, not overfitting.
        "K": ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level", "spin_z_norm", "extension"],
        # BB% uses command_score, the OpenCommand-calibrated translation of
        # horiz_var/vert_var into inferred-miss inches, not horiz_var/
        # vert_var directly -- replaced after a genuinely held-out win, see
        # DECISION_LOG.md "OpenCommand-calibrated Command Score replaces
        # horiz_var/vert_var in the BB% model" (R2_w 0.3517 -> 0.3610).
        # horiz_var/vert_var themselves replaced horiz_dist/vert_dist one
        # step earlier (R2_w 0.3274 -> 0.3407, "Location variance as a BB%
        # regressor replacement"). Other MODEL_SPECS entries below still use
        # horiz_dist/vert_dist unchanged -- neither swap was applied blanket
        # to every experiment, only to this validated production spec.
        "BB": ["naive_BB_rate", "fl_bb_rate_bridge", "command_score"]},
    "naive+other_league+changepoint": {
        "K": ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_cp"],
        "BB": ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_dist_cp", "vert_dist_cp"]},
    "naive+other_league+surprise_prior+changepoint": {
        "K": ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level", "wh_sva_cp"],
        "BB": ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_dist", "vert_dist",
               "horiz_dist_cp", "vert_dist_cp"]},
}


def run_model_family(df, n0_k, n0_bb, specs: list[str], mode: str, df_test=None, label_prefix=""):
    """mode='in_sample' scores df itself; mode='heldout' fits on df and scores on
    df_test (both already assembled via assemble())."""
    rows = []
    for name in specs:
        spec = MODEL_SPECS[name]
        if mode == "in_sample":
            k_res, k_m = run_in_sample(df, "K_rate_actual", spec["K"], n0_k, f"{label_prefix}{name} [K%]")
            bb_res, bb_m = run_in_sample(df, "BB_rate_actual", spec["BB"], n0_bb, f"{label_prefix}{name} [BB%]")
            kbb_regr = list(dict.fromkeys(spec["K"] + spec["BB"]))
            d_kbb = df.dropna(subset=["KBB_rate_actual"] + kbb_regr)
            n0_kbb, _ = mm.estimate_n0(d_kbb["KBB_rate_actual"] + 0.5, d_kbb["PA_test"], min_n=MIN_PA_N0)
            kbb_res, _ = run_in_sample(df, "KBB_rate_actual", kbb_regr, n0_kbb, f"{label_prefix}{name} [K-BB% direct]")
            composed = None
            if k_m is not None and bb_m is not None:
                common = df.dropna(subset=spec["K"] + spec["BB"] + ["KBB_rate_actual"])
                w = mm.reliability_weight(common["PA_test"], n0_kbb).values
                pred = k_m.predict(common) - bb_m.predict(common)
                composed = score(common["KBB_rate_actual"].values, pred.values, w, f"{label_prefix}{name} [K-BB% composed]")
            rows.extend([r for r in (k_res, bb_res, kbb_res, composed) if r is not None])
        else:
            k_res = run_heldout(df, "K_rate_actual", spec["K"], df_test, "K_rate_actual", n0_k, f"{label_prefix}{name} [K%]")
            bb_res = run_heldout(df, "BB_rate_actual", spec["BB"], df_test, "BB_rate_actual", n0_bb, f"{label_prefix}{name} [BB%]")
            kbb_regr = list(dict.fromkeys(spec["K"] + spec["BB"]))
            d_kbb = df.dropna(subset=["KBB_rate_actual"] + kbb_regr)
            n0_kbb, _ = mm.estimate_n0(d_kbb["KBB_rate_actual"] + 0.5, d_kbb["PA_test"], min_n=MIN_PA_N0)
            kbb_res = run_heldout(df, "KBB_rate_actual", kbb_regr, df_test, "KBB_rate_actual", n0_kbb, f"{label_prefix}{name} [K-BB% direct]")
            # composed, held-out: fit K and BB separately on train, predict both on test, difference
            d_tr_k = df.dropna(subset=["K_rate_actual"] + spec["K"])
            d_tr_bb = df.dropna(subset=["BB_rate_actual"] + spec["BB"])
            composed = None
            if len(d_tr_k) >= 15 and len(d_tr_bb) >= 15:
                w_tr_k = mm.reliability_weight(d_tr_k["PA_test"], n0_k).values
                w_tr_bb = mm.reliability_weight(d_tr_bb["PA_test"], n0_bb).values
                m_k = fit_weighted(d_tr_k, "K_rate_actual", spec["K"], w_tr_k)
                m_bb = fit_weighted(d_tr_bb, "BB_rate_actual", spec["BB"], w_tr_bb)
                d_te = df_test.dropna(subset=spec["K"] + spec["BB"] + ["KBB_rate_actual"])
                if len(d_te) >= 15:
                    w_te = mm.reliability_weight(d_te["PA_test"], n0_kbb).values
                    pred = m_k.predict(d_te) - m_bb.predict(d_te)
                    composed = score(d_te["KBB_rate_actual"].values, pred.values, w_te,
                                      f"{label_prefix}{name} [K-BB% composed]")
            rows.extend([r for r in (k_res, bb_res, kbb_res, composed) if r is not None])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Experiment 4: per-pitcher "first n0 pitches of 2026" / "rest of season" split
# --------------------------------------------------------------------------- #
def split_rest_of_season(raw, n0: float):
    w2026 = fp.filter_period(raw, START_2026, END_2026)
    term = w2026[w2026["is_terminal"]].copy()
    term = term.sort_values(["pitcher_id", "date", "game_id", "pa_id", "pitch_no"])
    term["cum_pa"] = term.groupby("pitcher_id").cumcount() + 1
    reached = term[term["cum_pa"] >= n0].groupby("pitcher_id")["date"].min().rename("cutoff_date")
    w2026 = w2026.merge(reached, on="pitcher_id", how="inner")
    early = w2026[w2026["date"] <= w2026["cutoff_date"]]
    rest = w2026[w2026["date"] > w2026["cutoff_date"]]
    return early, rest


if __name__ == "__main__":
    print("Loading FL pitches...")
    raw = fp.load_fl_pitches()
    raw = fp.derive_flags(raw)

    all_results = []

    # --------------------------------------------------------------------- #
    # Experiment 1: train on 1H2025 -> actual 2H2025; freeze; roll to
    # 2H2025 -> actual 1H2026 (held-out).
    # --------------------------------------------------------------------- #
    print("\n=== Experiment 1: predicting 2nd half 2025, tested on 1st half 2026 ===")
    df_tr1, n0_k1, n0_bb1 = assemble(raw, [(START_2025, CUTOFF_2025)], [(_DAY_AFTER_2025_CUT, END_2025)],
                                      with_changepoints=False, with_surprise_levels=True)
    df_te1, _, _ = assemble(raw, [(_DAY_AFTER_2025_CUT, END_2025)], [(START_2026, CUTOFF_2026)],
                             with_changepoints=False, with_surprise_levels=True)
    res1 = run_model_family(df_tr1, n0_k1, n0_bb1,
                             ["naive", "naive+surprise_prior", "naive+other_league",
                              "naive+other_league+surprise_prior"],
                             mode="heldout", df_test=df_te1, label_prefix="Exp1 ")
    print(res1.to_string(index=False))
    all_results.append(res1)

    # --------------------------------------------------------------------- #
    # Experiment 2: train on full 2025 -> actual 1H2026; freeze; roll to
    # (2025+1H2026) -> actual 2H2026 (held-out).
    # --------------------------------------------------------------------- #
    print("\n=== Experiment 2: predicting 1st half 2026, tested on 2nd half 2026 ===")
    df_tr2, n0_k2, n0_bb2 = assemble(raw, [(START_2025, END_2025)], [(START_2026, CUTOFF_2026)],
                                      with_changepoints=True)
    df_te2, _, _ = assemble(raw, [(START_2025, END_2025), (START_2026, CUTOFF_2026)],
                             [(_DAY_AFTER_2026_CUT, END_2026)], with_changepoints=True)
    exp23_specs = ["naive", "naive+surprise_prior", "naive+changepoint", "naive+other_league",
                   "naive+other_league+surprise_prior", "naive+other_league+changepoint",
                   "naive+other_league+surprise_prior+changepoint"]
    res2 = run_model_family(df_tr2, n0_k2, n0_bb2, exp23_specs,
                             mode="heldout", df_test=df_te2, label_prefix="Exp2 ")
    print(res2.to_string(index=False))
    all_results.append(res2)

    # --------------------------------------------------------------------- #
    # Experiment 3: (2025+1H2026) -> actual 2H2026, IN-SAMPLE only (no further
    # held-out season exists -- confirmed with the user).
    # --------------------------------------------------------------------- #
    print("\n=== Experiment 3: predicting 2nd half 2026 (in-sample only, no holdout data) ===")
    df3, n0_k3, n0_bb3 = assemble(raw, [(START_2025, END_2025), (START_2026, CUTOFF_2026)],
                                   [(_DAY_AFTER_2026_CUT, END_2026)], with_changepoints=True)
    res3 = run_model_family(df3, n0_k3, n0_bb3, exp23_specs,
                             mode="in_sample", label_prefix="Exp3 ")
    print(res3.to_string(index=False))
    all_results.append(res3)

    # --------------------------------------------------------------------- #
    # Experiment 4: 2025 -> 2026 rest-of-season (per-pitcher, after their own
    # first n0 PA of 2026), IN-SAMPLE only.
    # --------------------------------------------------------------------- #
    print("\n=== Experiment 4: predicting 2026 rest-of-season (in-sample only, no holdout data) ===")
    naive4, n0_k4, n0_bb4 = naive_prior(raw, [(START_2025, END_2025)])
    br4 = bridge_features(naive4["pitcher_id"], raw, current_year=2025)
    early_k, rest_k = split_rest_of_season(raw, n0_k4)
    early_bb, rest_bb = split_rest_of_season(raw, n0_bb4)

    wh_level4, wh_cp4 = wh_sva_level_and_changepoint(early_k)
    dist_level4 = dist_levels(early_bb)
    dist_cp4 = dist_changepoints(early_bb)

    actual_k4 = fp.build_period_kbb(rest_k, min_pa=MIN_PA_TEST)[["pitcher_id", "PA", "K_rate"]].rename(
        columns={"PA": "PA_test", "K_rate": "K_rate_actual"})
    actual_bb4 = fp.build_period_kbb(rest_bb, min_pa=MIN_PA_TEST)[["pitcher_id", "PA", "BB_rate"]].rename(
        columns={"PA": "PA_test", "BB_rate": "BB_rate_actual"})

    df4_k = naive4.merge(br4, on="pitcher_id", how="left").merge(wh_level4, on="pitcher_id", how="left") \
        .merge(wh_cp4, on="pitcher_id", how="left").merge(actual_k4, on="pitcher_id", how="inner")
    df4_bb = naive4.merge(br4, on="pitcher_id", how="left").merge(dist_level4, on="pitcher_id", how="left") \
        .merge(dist_cp4, on="pitcher_id", how="left").merge(actual_bb4, on="pitcher_id", how="inner")
    lg_k_rate4 = df4_k["K_rate_actual"].mean()
    lg_bb_rate4 = df4_bb["BB_rate_actual"].mean()
    df4_k["fl_k_rate_bridge"] = df4_k["fl_k_rate_bridge"].fillna(lg_k_rate4)
    df4_bb["fl_bb_rate_bridge"] = df4_bb["fl_bb_rate_bridge"].fillna(lg_bb_rate4)
    df4_k["wh_sva_level"] = df4_k["wh_sva_level"].fillna(df4_k["wh_sva_level"].mean())
    df4_bb["horiz_dist"] = df4_bb["horiz_dist"].fillna(df4_bb["horiz_dist"].mean())
    df4_bb["vert_dist"] = df4_bb["vert_dist"].fillna(df4_bb["vert_dist"].mean())
    df4_k["wh_sva_cp"] = df4_k["wh_sva_cp"].fillna(0.0)
    df4_bb["horiz_dist_cp"] = df4_bb["horiz_dist_cp"].fillna(0.0)
    df4_bb["vert_dist_cp"] = df4_bb["vert_dist_cp"].fillna(0.0)

    rows4 = []
    for name in ["naive", "naive+surprise_prior", "naive+changepoint", "naive+other_league",
                 "naive+other_league+surprise_prior", "naive+other_league+changepoint",
                 "naive+other_league+surprise_prior+changepoint"]:
        spec = MODEL_SPECS[name]
        k_res, _ = run_in_sample(df4_k, "K_rate_actual", spec["K"], n0_k4, f"Exp4 {name} [K%]")
        bb_res, _ = run_in_sample(df4_bb, "BB_rate_actual", spec["BB"], n0_bb4, f"Exp4 {name} [BB%]")
        rows4.extend([k_res, bb_res])
    res4 = pd.DataFrame(rows4)
    print(res4.to_string(index=False))
    all_results.append(res4)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv("output/fl_halfseason_backtest_results.csv", index=False)
    print("\nSaved -> output/fl_halfseason_backtest_results.csv")
