"""
mlb_pitcher_features.py -- builds the central pitcher-season feature panel (2018-2025)
for the K%/BB% Marcel projection.

Pieces (see the approved plan, `output/../plans/let-s-discuss-how-this-merry-thacker.md`):
  1. Flag derivation (swing/whiff/terminal-pitch, arm-side adjustment, pitch-type family)
  2. Raw K_rate/BB_rate/PA per pitcher-season
  3. Marcel 3-year (5/4/3) weighted K_rate/BB_rate, REG fit via mlb_marcel.estimate_n0
  4. By-pitch-type wh_sva/swstr_sva (BOTH swing-conditioned and unconditional), two-stage
     XGBoost + GroupKFold(5, groups=pitcher) location-vs-shape residual, mirroring
     pitcher_projection_bayesian.ipynb cells 43-49 (loc_cols3/full_cols3/get_oof_proba)
  5. horiz_dist/vert_dist pitcher-season means (winsorized) + within-season deltas
     (overall, not per-family -- per-family halves would be too thin) via apply_shrinkage
  6. BF/G, Age

Column mapping vs. the FL notebook's swing_pop feature set (Statcast names, verified
empirically against statcast_2023.parquet before use):
  effective_velo   -> effective_speed
  ivb              -> api_break_z_with_gravity   (Savant's own IVB, already gravity-adj)
  horz_break_as    -> api_break_x_arm            (ALREADY arm-side-invariant -- checked:
                                                   FF mean api_break_x_arm is ~+0.62/+0.64
                                                   for both L and R pitchers, unlike pfx_x
                                                   which flips sign with handedness)
  spin_rate        -> release_spin_rate
  plate_x_as       -> plate_x * where(p_throws=="R", 1, -1)   (needs the flip -- plate_x
                                                                 itself is NOT arm-adjusted)
  extension        -> release_extension
  rel_height       -> release_pos_z
  rel_side_as      -> release_pos_x * where(p_throws=="R", 1, -1)  (also needs the flip --
                                                                     checked: mean
                                                                     release_pos_x is
                                                                     +1.87 for LHP vs.
                                                                     -1.83 for RHP)

Usage:
    python mlb_pitcher_features.py
"""
from __future__ import annotations

import glob
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBClassifier

import mlb_marcel as mm

YEARS = list(range(2018, 2026))
MIN_PA = 50            # eligibility floor for a pitcher-season to count in reporting
MIN_N_FAMILY = 30      # min pitches in a pitch-type family per pitcher-season for wh_sva/swstr_sva
MIN_HALF_PITCHES = 50  # min pitches per half-season for a within-season delta to be real (else impute 0)

PITCH_FAMILY = {
    "FF": "FB", "FA": "FB", "SI": "FB", "FC": "FB",
    "SL": "BRK", "CU": "BRK", "KC": "BRK", "ST": "BRK", "SV": "BRK", "CS": "BRK", "SC": "BRK",
    "CH": "OFF", "FS": "OFF", "FO": "OFF", "EP": "OFF", "KN": "OFF",
}

SWING_DESC = {"foul", "foul_tip", "hit_into_play", "swinging_strike", "swinging_strike_blocked",
              "foul_bunt", "missed_bunt", "bunt_foul_tip"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
K_EVENTS = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk", "intent_walk"}

LOC_COLS = ["plate_x_as", "plate_z"]
FULL_COLS = ["effective_speed", "ivb", "horz_break_as", "release_spin_rate",
             "plate_x_as", "plate_z", "release_extension", "release_pos_z", "rel_side_as"]

XGB_PARAMS = dict(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
                   colsample_bytree=0.8, eval_metric="logloss", random_state=42,
                   n_jobs=-1, tree_method="hist")


# --------------------------------------------------------------------------- #
#  1. Flag derivation
# --------------------------------------------------------------------------- #
def load_statcast_all(years=YEARS) -> pd.DataFrame:
    frames = []
    for y in years:
        path = f"statcast_{y}.parquet"
        d = pd.read_parquet(path)
        frames.append(d)
        print(f"  loaded {path}: {len(d):,} rows")
    df = pd.concat(frames, ignore_index=True)
    print(f"total: {len(df):,} rows, {df.shape[1]} cols")
    return df


def derive_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    arm_sign = np.where(df["p_throws"] == "R", 1, -1)
    df["plate_x_as"] = df["plate_x"] * arm_sign
    df["rel_side_as"] = df["release_pos_x"] * arm_sign
    df["ivb"] = df["api_break_z_with_gravity"]
    df["horz_break_as"] = df["api_break_x_arm"]

    desc = df["description"].astype(str)
    df["is_swing"] = desc.isin(SWING_DESC)
    df["is_whiff"] = desc.isin(WHIFF_DESC)
    events = df["events"].astype(object)
    df["is_terminal_pitch"] = events.notna() & (events != "truncated_pa")
    df["pitch_family"] = df["pitch_type"].map(PITCH_FAMILY)

    zone_center_z = (df["sz_top"] + df["sz_bot"]) / 2
    df["horiz_dist_raw"] = df["plate_x"].abs()
    df["vert_dist_raw"] = (df["plate_z"] - zone_center_z).abs()
    return df



# --------------------------------------------------------------------------- #
#  2. Raw PA/K/BB rates + BF/G per pitcher-season
# --------------------------------------------------------------------------- #
def season_rates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pitcher, game_year): PA, K, BB, K_rate, BB_rate, BF/G.
    PA = terminal pitches only (one per plate appearance); truncated_pa excluded
    (already handled by is_terminal_pitch)."""
    term = df[df["is_terminal_pitch"]].copy()
    events = term["events"].astype(str)
    term["is_k"] = events.isin(K_EVENTS)
    term["is_bb"] = events.isin(BB_EVENTS)

    agg = term.groupby(["pitcher", "game_year"]).agg(
        PA=("events", "size"),
        K=("is_k", "sum"),
        BB=("is_bb", "sum"),
        n_games=("game_pk", "nunique"),
        player_name=("player_name", lambda s: s.mode().iloc[0] if len(s) else np.nan),
    ).reset_index()
    agg["K_rate"] = agg["K"] / agg["PA"]
    agg["BB_rate"] = agg["BB"] / agg["PA"]
    agg["BF_per_G"] = agg["PA"] / agg["n_games"]
    return agg


# --------------------------------------------------------------------------- #
#  3. Marcel 3-year (5/4/3) weighted K_rate/BB_rate
# --------------------------------------------------------------------------- #
def fit_marcel_reg(season_df: pd.DataFrame, min_pa: int = MIN_PA):
    """REG_K/REG_BB via mlb_marcel.estimate_n0 (method-of-moments), fit once over the
    whole panel's single-season rates as the panel's default hyperparameter -- the
    walk-forward backtest refits this per training fold separately (see the plan's
    validation section); this is the shared/production value."""
    reg_k, info_k = mm.estimate_n0(season_df["K_rate"], season_df["PA"], min_n=min_pa)
    reg_bb, info_bb = mm.estimate_n0(season_df["BB_rate"], season_df["PA"], min_n=min_pa)
    print(f"REG_K  = {reg_k:.1f}  {info_k}")
    print(f"REG_BB = {reg_bb:.1f}  {info_bb}")
    return reg_k, reg_bb


def marcel_projections(season_df: pd.DataFrame, reg_k: float, reg_bb: float) -> pd.DataFrame:
    """For every (pitcher, target_year) with >=1 trailing year of data, compute the
    classic 5/4/3 Marcel-weighted K_rate/BB_rate using ONLY years < target_year.
    Returns one row per (pitcher, target_year) -- this is the preseason projection
    input, joined later against the target_year's ACTUAL K_rate/BB_rate as the label."""
    pivot_pa = season_df.pivot(index="pitcher", columns="game_year", values="PA")
    pivot_k = season_df.pivot(index="pitcher", columns="game_year", values="K")
    pivot_bb = season_df.pivot(index="pitcher", columns="game_year", values="BB")

    league = season_df.groupby("game_year").apply(
        lambda g: pd.Series({"K_rate": g["K"].sum() / g["PA"].sum(),
                              "BB_rate": g["BB"].sum() / g["PA"].sum()})
    )

    pitchers = pivot_pa.index.values
    rows = []
    target_years = sorted(season_df["game_year"].unique())
    for target_year in target_years:
        trailing = [target_year - 1, target_year - 2, target_year - 3]
        weights = [mm.MARCEL_WEIGHTS[lag] for lag in (0, 1, 2)]
        num_k = np.zeros(len(pitchers))
        num_bb = np.zeros(len(pitchers))
        den = np.zeros(len(pitchers))
        lg_num_k, lg_num_bb, lg_den = 0.0, 0.0, 0.0
        any_year = False
        for wt, yr in zip(weights, trailing):
            if yr not in pivot_pa.columns:
                continue
            any_year = True
            pa = pivot_pa[yr].fillna(0).values
            k = pivot_k[yr].fillna(0).values
            bb = pivot_bb[yr].fillna(0).values
            num_k += wt * k
            num_bb += wt * bb
            den += wt * pa
            if yr in league.index:
                lg_num_k += wt * league.loc[yr, "K_rate"]
                lg_num_bb += wt * league.loc[yr, "BB_rate"]
                lg_den += wt
        if not any_year or lg_den == 0:
            continue
        lg_k_blend = lg_num_k / lg_den
        lg_bb_blend = lg_num_bb / lg_den
        with np.errstate(divide="ignore", invalid="ignore"):
            marcel_k = (num_k + reg_k * lg_k_blend) / (den + reg_k)
            marcel_bb = (num_bb + reg_bb * lg_bb_blend) / (den + reg_bb)
        mask = den > 0  # need at least some trailing PA to report a Marcel rate
        rows.append(pd.DataFrame({
            "pitcher": pitchers[mask],
            "game_year": target_year,
            "Marcel_K_rate": marcel_k[mask],
            "Marcel_BB_rate": marcel_bb[mask],
            "Marcel_weighted_PA": den[mask],
        }))
    out = pd.concat(rows, ignore_index=True)
    return out



# --------------------------------------------------------------------------- #
#  4. By-pitch-type wh_sva / swstr_sva, both conditionings
#     Mirrors pitcher_projection_bayesian.ipynb cells 43-49: GroupKFold(5, groups=
#     pitcher), location-only vs. location+shape XGBClassifier, OOF residual = shape
#     value added beyond location. `years` lets the walk-forward backtest later
#     restrict this to a fold's training years only (never fit on the test year).
# --------------------------------------------------------------------------- #
def compute_shape_value_added(df: pd.DataFrame, condition: str, family: str | None = None,
                               years: list[int] | None = None, min_rows: int = 2000) -> pd.DataFrame:
    """condition='swing' -> P(whiff|swing), the established Whiff%-SVA convention.
    condition='all' -> P(swinging-strike) over every pitch of the family regardless of
    swing decision (takes count as 0 automatically, since is_whiff is False for them) --
    the SwStr%-SVA alternative, validated against Whiff%-SVA rather than assumed."""
    pop = df
    if years is not None:
        pop = pop[pop["game_year"].isin(years)]
    if family is not None:
        pop = pop[pop["pitch_family"] == family]
    if condition == "swing":
        pop = pop[pop["is_swing"]]
    elif condition != "all":
        raise ValueError(condition)

    needed = FULL_COLS + ["pitcher", "is_whiff", "game_year"]
    pop = pop.dropna(subset=needed).copy()
    label = f"{condition}/{family or 'ALL'}"
    if len(pop) < min_rows:
        print(f"  [{label}] only {len(pop)} rows -- skipping")
        return pd.DataFrame(columns=["pitcher", "game_year", "n", "rate", "sva"])

    y = pop["is_whiff"].astype(int).values
    groups = pop["pitcher"].values
    gkf = GroupKFold(n_splits=5)

    t0 = time.time()
    oof_loc = cross_val_predict(XGBClassifier(**XGB_PARAMS), pop[LOC_COLS].astype(float).values,
                                 y, cv=gkf, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    oof_full = cross_val_predict(XGBClassifier(**XGB_PARAMS), pop[FULL_COLS].astype(float).values,
                                  y, cv=gkf, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    pop["sva"] = oof_full - oof_loc
    pop["whiff"] = y
    print(f"  [{label}] n={len(pop):,}, rate={y.mean():.4f}, sva_std={pop['sva'].std():.4f}, "
          f"took {time.time()-t0:.0f}s")

    agg = pop.groupby(["pitcher", "game_year"]).agg(
        n=("sva", "size"), rate=("whiff", "mean"), sva=("sva", "mean")
    ).reset_index()
    return agg


def pooled_wh_sva_per_pitch(df: pd.DataFrame, years: list[int] | None = None) -> pd.DataFrame:
    """Overall (pooled across pitch-type families) whiff|swing shape-value-added, at the
    per-PITCH level -- used only for the within-season delta feature (families are too
    thin to split at the median pitch and still have two usable halves). Swing-
    conditioned (the established FL convention), not the unconditional SwStr% version --
    the delta feature tracks trend, and the family-level validation is what decides
    swing-vs-unconditional for the model's main effect."""
    pop = df[df["is_swing"]]
    needed = FULL_COLS + ["pitcher", "is_whiff", "game_year", "game_date", "at_bat_number", "pitch_number"]
    pop = pop.dropna(subset=needed).copy()
    if years is not None:
        pop = pop[pop["game_year"].isin(years)]
    y = pop["is_whiff"].astype(int).values
    groups = pop["pitcher"].values
    gkf = GroupKFold(n_splits=5)
    t0 = time.time()
    oof_loc = cross_val_predict(XGBClassifier(**XGB_PARAMS), pop[LOC_COLS].astype(float).values,
                                 y, cv=gkf, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    oof_full = cross_val_predict(XGBClassifier(**XGB_PARAMS), pop[FULL_COLS].astype(float).values,
                                  y, cv=gkf, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    pop["sva"] = oof_full - oof_loc
    print(f"  [pooled overall wh_sva] n={len(pop):,}, took {time.time()-t0:.0f}s")
    return pop[["pitcher", "game_year", "game_date", "at_bat_number", "pitch_number", "sva"]]


# --------------------------------------------------------------------------- #
#  5. horiz_dist/vert_dist (season means) + within-season deltas
# --------------------------------------------------------------------------- #
def compute_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pitcher-season mean horiz_dist/vert_dist, winsorized at the 1st/99th percentile
    over ALL pitches thrown (not swing-conditioned -- this is a location-precision
    metric independent of the batter's decision)."""
    pop = df.dropna(subset=["horiz_dist_raw", "vert_dist_raw", "pitcher", "game_year"]).copy()
    lo_h, hi_h = pop["horiz_dist_raw"].quantile([0.01, 0.99])
    lo_v, hi_v = pop["vert_dist_raw"].quantile([0.01, 0.99])
    pop["horiz_dist_w"] = pop["horiz_dist_raw"].clip(lo_h, hi_h)
    pop["vert_dist_w"] = pop["vert_dist_raw"].clip(lo_v, hi_v)
    agg = pop.groupby(["pitcher", "game_year"]).agg(
        horiz_dist=("horiz_dist_w", "mean"), vert_dist=("vert_dist_w", "mean")
    ).reset_index()
    return agg


def _half_split_delta(pop: pd.DataFrame, value_col: str, min_half: int = MIN_HALF_PITCHES) -> pd.DataFrame:
    """Split each (pitcher, game_year) at the median pitch, in chronological order
    (game_date, at_bat_number, pitch_number), into first/second half; delta = second -
    first. Below min_half pitches per half: impute delta=0 with has_delta=False rather
    than reporting a noisy tiny-sample difference."""
    pop = pop.sort_values(["pitcher", "game_year", "game_date", "at_bat_number", "pitch_number"])
    grp = pop.groupby(["pitcher", "game_year"])
    rank = grp.cumcount()
    counts = grp[value_col].transform("size")
    half = np.where(rank < counts / 2, "first", "second")
    pop = pop.assign(_half=half)
    piv = pop.groupby(["pitcher", "game_year", "_half"])[value_col].agg(["mean", "size"]).unstack("_half")
    out = pd.DataFrame(index=piv.index)
    out["first_mean"] = piv[("mean", "first")]
    out["second_mean"] = piv[("mean", "second")]
    out["first_n"] = piv[("size", "first")].fillna(0)
    out["second_n"] = piv[("size", "second")].fillna(0)
    out["delta_raw"] = out["second_mean"] - out["first_mean"]
    out["has_delta"] = (out["first_n"] >= min_half) & (out["second_n"] >= min_half)
    out.loc[~out["has_delta"], "delta_raw"] = 0.0
    return out.reset_index()[["pitcher", "game_year", "delta_raw", "has_delta", "first_n", "second_n"]]


def compute_within_season_deltas(df: pd.DataFrame, wh_sva_pitch: pd.DataFrame) -> pd.DataFrame:
    """wh_sva_delta, horiz_dist_delta, vert_dist_delta -- each half-split then shrunk via
    mlb_marcel.apply_shrinkage (small-sample, high-noise quantities; shrinkage params fit
    once on the full-season aggregate of that same raw quantity, never on the delta
    itself, matching the notebook's precedent)."""
    loc_pop = df.dropna(subset=["horiz_dist_raw", "vert_dist_raw", "pitcher", "game_year",
                                 "game_date", "at_bat_number", "pitch_number"])
    d_h = _half_split_delta(loc_pop, "horiz_dist_raw").rename(columns={"delta_raw": "horiz_dist_delta_raw"})
    d_v = _half_split_delta(loc_pop, "vert_dist_raw").rename(columns={"delta_raw": "vert_dist_delta_raw"})
    d_w = _half_split_delta(wh_sva_pitch, "sva").rename(columns={"delta_raw": "wh_sva_delta_raw"})

    out = d_h[["pitcher", "game_year", "horiz_dist_delta_raw", "has_delta"]].rename(
        columns={"has_delta": "has_delta_horiz"})
    out = out.merge(d_v[["pitcher", "game_year", "vert_dist_delta_raw", "has_delta"]].rename(
        columns={"has_delta": "has_delta_vert"}), on=["pitcher", "game_year"], how="outer")
    out = out.merge(d_w[["pitcher", "game_year", "wh_sva_delta_raw", "has_delta"]].rename(
        columns={"has_delta": "has_delta_wh"}), on=["pitcher", "game_year"], how="outer")

    # shrinkage params fit once on the full pooled population's raw quantities
    sw2_h, sb2_h, gm_h = mm.fit_shrinkage_params(loc_pop, "horiz_dist_raw")
    sw2_v, sb2_v, gm_v = mm.fit_shrinkage_params(loc_pop, "vert_dist_raw")
    sw2_w, sb2_w, gm_w = mm.fit_shrinkage_params(wh_sva_pitch, "sva")
    print(f"shrinkage horiz_dist: sw2={sw2_h:.4f} sb2={sb2_h:.6f} gm={gm_h:.4f}")
    print(f"shrinkage vert_dist:  sw2={sw2_v:.4f} sb2={sb2_v:.6f} gm={gm_v:.4f}")
    print(f"shrinkage wh_sva:     sw2={sw2_w:.4f} sb2={sb2_w:.6f} gm={gm_w:.4f}")

    n_h = d_h.set_index(["pitcher", "game_year"])[["first_n", "second_n"]].sum(axis=1)
    n_v = d_v.set_index(["pitcher", "game_year"])[["first_n", "second_n"]].sum(axis=1)
    n_w = d_w.set_index(["pitcher", "game_year"])[["first_n", "second_n"]].sum(axis=1)
    out = out.set_index(["pitcher", "game_year"])
    out["horiz_dist_delta"], _ = mm.apply_shrinkage(out["horiz_dist_delta_raw"].fillna(0).values,
                                                      n_h.reindex(out.index).fillna(1).values, sw2_h, sb2_h, 0.0)
    out["vert_dist_delta"], _ = mm.apply_shrinkage(out["vert_dist_delta_raw"].fillna(0).values,
                                                     n_v.reindex(out.index).fillna(1).values, sw2_v, sb2_v, 0.0)
    out["wh_sva_delta"], _ = mm.apply_shrinkage(out["wh_sva_delta_raw"].fillna(0).values,
                                                  n_w.reindex(out.index).fillna(1).values, sw2_w, sb2_w, 0.0)
    return out.reset_index()


# --------------------------------------------------------------------------- #
#  6. Age
# --------------------------------------------------------------------------- #
def compute_age(season_pitchers: pd.DataFrame, bio_path: str = "output/pitcher_bio.parquet") -> pd.DataFrame:
    """Age as of April 1 of game_year, from the MLB Stats API birthdate pull, reusing
    k26_projection.py's age_c (centered at 29) / age_decline (extra post-30 slope)
    parameterization rather than a new curve shape."""
    bio = pd.read_parquet(bio_path)[["pitcher_id", "birth_date"]].rename(columns={"pitcher_id": "pitcher"})
    out = season_pitchers[["pitcher", "game_year"]].drop_duplicates().merge(bio, on="pitcher", how="left")
    ref_date = pd.to_datetime(out["game_year"].astype(str) + "-04-01")
    out["age"] = (ref_date - out["birth_date"]).dt.days / 365.25
    out["age_c"] = out["age"] - 29
    out["age_decline"] = (out["age"] - 30).clip(lower=0)
    n_missing = out["age"].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} pitcher-seasons missing age (no bio match)")
    return out.drop(columns=["birth_date"])


FAMILIES = ["FB", "BRK", "OFF"]
CONDITIONS = ["swing", "all"]  # swing -> Whiff%-SVA, all -> SwStr%-SVA


def build_family_sva_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (pitcher, game_year), columns wh_sva_fb/brk/off, swstr_sva_fb/brk/off
    (+ n_*/is_imputed_* per family/conditioning). Built over the FULL history -- the
    walk-forward backtest refits this per training fold separately via the same
    compute_shape_value_added(years=...) call."""
    panel = None
    for condition in CONDITIONS:
        prefix = "wh_sva" if condition == "swing" else "swstr_sva"
        for fam in FAMILIES:
            agg = compute_shape_value_added(raw, condition=condition, family=fam)
            fam_lower = fam.lower()
            agg = agg.rename(columns={"sva": f"{prefix}_{fam_lower}", "n": f"n_{prefix}_{fam_lower}",
                                       "rate": f"rate_{prefix}_{fam_lower}"})
            agg[f"is_imputed_{prefix}_{fam_lower}"] = agg[f"n_{prefix}_{fam_lower}"] < MIN_N_FAMILY
            agg.loc[agg[f"is_imputed_{prefix}_{fam_lower}"], f"{prefix}_{fam_lower}"] = 0.0
            panel = agg if panel is None else panel.merge(agg, on=["pitcher", "game_year"], how="outer")
    return panel


if __name__ == "__main__":
    t0 = time.time()
    print("Loading all years...")
    raw = load_statcast_all()
    print("Deriving flags...")
    raw = derive_flags(raw)
    print(f"flags done in {time.time()-t0:.0f}s")
    print(raw[["is_swing", "is_whiff", "is_terminal_pitch"]].mean())
    print(raw["pitch_family"].value_counts(dropna=False))

    print("\nSeason rates...")
    season = season_rates(raw)
    print(season.shape)
    print(season[season["PA"] >= MIN_PA].describe()[["PA", "K_rate", "BB_rate", "BF_per_G"]])

    print("\nMarcel REG fit...")
    reg_k, reg_bb = fit_marcel_reg(season[season["PA"] >= MIN_PA])

    print("\nMarcel projections...")
    marcel = marcel_projections(season, reg_k, reg_bb)
    check = marcel.merge(season[["pitcher", "game_year", "K_rate", "BB_rate", "PA"]],
                          on=["pitcher", "game_year"], suffixes=("", "_actual"))
    check = check[check["PA"] >= MIN_PA]
    print(f"Marcel K_rate vs actual: corr={check['Marcel_K_rate'].corr(check['K_rate']):.3f}, "
          f"RMSE={np.sqrt(((check['Marcel_K_rate']-check['K_rate'])**2).mean())*100:.3f}pp  (n={len(check)})")
    print(f"Marcel BB_rate vs actual: corr={check['Marcel_BB_rate'].corr(check['BB_rate']):.3f}, "
          f"RMSE={np.sqrt(((check['Marcel_BB_rate']-check['BB_rate'])**2).mean())*100:.3f}pp  (n={len(check)})")

    print("\nBy-pitch-type wh_sva/swstr_sva (full history, both conditionings)...")
    t1 = time.time()
    family_sva = build_family_sva_panel(raw)
    print(f"family sva done in {time.time()-t1:.0f}s, shape={family_sva.shape}")

    print("\nhoriz_dist/vert_dist...")
    dist = compute_dist_features(raw)
    print(dist.shape)

    print("\nPooled overall wh_sva (for within-season delta)...")
    t1 = time.time()
    wh_sva_pitch = pooled_wh_sva_per_pitch(raw)
    print(f"pooled wh_sva done in {time.time()-t1:.0f}s")

    print("\nWithin-season deltas...")
    deltas = compute_within_season_deltas(raw, wh_sva_pitch)
    print(deltas.shape)

    print("\nAge...")
    age = compute_age(season)
    print(age.shape)

    print("\nAssembling final panel...")
    panel = season.merge(marcel, on=["pitcher", "game_year"], how="left")
    panel = panel.merge(family_sva, on=["pitcher", "game_year"], how="left")
    panel = panel.merge(dist, on=["pitcher", "game_year"], how="left")
    panel = panel.merge(deltas, on=["pitcher", "game_year"], how="left")
    panel = panel.merge(age, on=["pitcher", "game_year"], how="left")

    id_map = pd.read_parquet("output/pitcher_id_name_map.parquet").rename(columns={"pitcher_id": "pitcher"})
    panel = panel.merge(id_map, on="pitcher", how="left")

    panel.to_parquet("output/mlb_pitcher_season_features.parquet", index=False)
    print(f"\nSaved -> output/mlb_pitcher_season_features.parquet, shape={panel.shape}")
    print(f"total runtime: {time.time()-t0:.0f}s")

    # ------------------------------------------------------------------- #
    #  Validation: standalone correlation of each family/conditioning sva
    #  with the SAME-SEASON K_rate (concurrent check -- the real predictive-
    #  power validation, whether whiff or swstr wins, happens once these
    #  feed the modeling script's incremental CV R2 comparison).
    # ------------------------------------------------------------------- #
    print("\nConcurrent correlation with K_rate (eligible PA>=MIN_PA):")
    elig = panel[panel["PA"] >= MIN_PA]
    for condition in CONDITIONS:
        prefix = "wh_sva" if condition == "swing" else "swstr_sva"
        for fam in FAMILIES:
            col = f"{prefix}_{fam.lower()}"
            if col in elig.columns:
                c = elig[col].corr(elig["K_rate"])
                print(f"  {col}: corr={c:.3f}")
    print(f"  wh_sva_delta: corr with K_rate={elig['wh_sva_delta'].corr(elig['K_rate']):.3f}")
    print(f"  horiz_dist_delta: corr with K_rate={elig['horiz_dist_delta'].corr(elig['K_rate']):.3f}")
    print(f"  vert_dist_delta: corr with K_rate={elig['vert_dist_delta'].corr(elig['K_rate']):.3f}")
    print(f"  BF_per_G: corr with K_rate={elig['BF_per_G'].corr(elig['K_rate']):.3f}")
    print(f"  age_c: corr with K_rate={elig['age_c'].corr(elig['K_rate']):.3f}")
