"""
fl_pitcher_features.py -- FL-side feature engineering for the half-season K%/BB%
backtest harness (see the approved plan at
C:\\Users\\215nt\\.claude\\plans\\let-s-discuss-how-this-merry-thacker.md).

Ports the MLB `mlb_pitcher_features.py` shape-value-added / location-precision
machinery onto FL Supabase's actual columns, plus the per-start chronological
sequencing and real changepoint detection the user clarified "surprise variable
changepoint" to mean (an actual detected shift within a training window's own
time series, not a delta between two pre-defined periods).

Column mapping vs. MLB Statcast (verified against the FL Supabase schema directly,
not assumed -- see inline notes where an earlier pass got something wrong):
  effective_velo  -> effective_speed
  ivb             -> exists directly in FL (NOT approximated from vert_break,
                      an earlier wrong assumption)
  horz_break_as   -> horz_break * arm_sign   (FL's horz_break DOES flip sign with
                      handedness -- checked empirically: four_seam mean horz_break
                      is +9.48 for RHP vs. -9.97 for LHP -- unlike MLB's
                      api_break_x_arm, which is already arm-invariant. Needs the
                      flip, same as plate_x.)
  rel_side_as     -> rel_side * arm_sign     (also flips: +1.60 RHP vs. -1.64 LHP)
  spin_rate       -> release_spin_rate
  plate_x_as      -> plate_x * arm_sign
  extension       -> release_extension
  release_pos_z   -> release_pos_z (direct)
  pitch_type      -> use FL's own `pitch_type` column (NOT `tagged_pitch_type`,
                      which is 60% 'undefined', an earlier wrong read)

Scope simplification (stated, not silently assumed): the user asked for ONE
`wh_sva` surprise variable for K%, not a per-pitch-type-family breakdown -- so
this builds the POOLED (swing-conditioned) two-stage shape-value-added model only,
mirroring mlb_pitcher_features.py's `pooled_wh_sva_per_pitch` (used there for the
within-season delta), not its by-family `compute_shape_value_added`. Simpler and
matches what was actually requested; can be extended to family-level later if the
pooled version underperforms.

Usage:
    import fl_pitcher_features as fp
    raw = fp.load_fl_pitches()
    raw = fp.derive_flags(raw)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBClassifier

import mlb_marcel as mm
from fl_supabase import read_sql_df
from sqlalchemy import text

MIN_HALF_PITCHES = 50  # same floor as MLB's within-season delta, reused for changepoint eligibility
MIN_STARTS_FOR_CHANGEPOINT = 6

LOC_COLS = ["plate_x_as", "plate_z"]
FULL_COLS = ["effective_velo", "ivb", "horz_break_as", "spin_rate",
             "plate_x_as", "plate_z", "extension", "release_pos_z", "rel_side_as"]

XGB_PARAMS = dict(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
                   colsample_bytree=0.8, eval_metric="logloss", random_state=42,
                   n_jobs=-1, tree_method="hist")

RAW_COLS = ["pitcher", "pitcher_id", "pitcher_team", "year", "date", "game_id", "pa_id", "pitch_no",
            "pitch_type", "pitch_call", "swing", "k_or_bb", "is_terminal_pitch",
            "plate_x", "plate_z", "effective_velo", "ivb", "horz_break", "spin_rate",
            "extension", "release_pos_z", "rel_side", "pitcher_throws", "sz_top", "sz_bot",
            "home_team", "catcher_id", "batter_side"]


CACHE_PATH = "output/fl_raw_pitches_flagged.parquet"
# Canonical row order (2026-10-20): the database returns rows in arbitrary order and downstream results (ROS pitchers: mean .013 RA9, max .27) depended on it, so every load is
# sorted on a near-unique key (verified: 0 duplicates on these columns) and the index reset.
ORDER_KEY = ["game_id", "pa_id", "pitch_no", "plate_x", "plate_z", "effective_velo", "spin_rate", "ivb", "pitcher_id"]


def harmonize_names(df: pd.DataFrame) -> pd.DataFrame:
    """One spelling per pitcher_id: the most frequent one in the pitch data (ties alphabetical).  Three ids carry more than one name (Gilleran Jimmy / James = the same pitcher;
    House / McCartney 43 pitches; Parks / Majick 1 pitch) and five downstream places took 'the first name seen', i.e. whichever row the database returned first."""
    cnt = df.dropna(subset=["pitcher_id", "pitcher"]).groupby(["pitcher_id", "pitcher"]).size().rename("n").reset_index().sort_values(["pitcher_id", "n", "pitcher"], ascending=[True, False, True])
    best = cnt.drop_duplicates("pitcher_id").set_index("pitcher_id")["pitcher"]
    out = df.copy(); out["pitcher"] = out["pitcher_id"].map(best).fillna(out["pitcher"]); return out


def canonical_order(df: pd.DataFrame) -> pd.DataFrame:
    df = harmonize_names(df)
    return df.sort_values([c for c in ORDER_KEY if c in df.columns], kind="mergesort", na_position="last").reset_index(drop=True)


def load_fl_pitches_cached(path: str = CACHE_PATH) -> pd.DataFrame:
    """Loads the already-flagged (derive_flags applied) pitch cache if present,
    else builds and saves it. Avoids the ~2min Supabase pull on every script run
    during iterative analysis -- not used by the production backtest harness
    itself (which always pulls fresh), only by downstream analysis scripts."""
    import os
    if os.path.exists(path):
        return canonical_order(pd.read_parquet(path))
    raw = canonical_order(derive_flags(load_fl_pitches()))
    raw.to_parquet(path, index=False)
    return raw


def load_fl_pitches() -> pd.DataFrame:
    q = text(f"""
        SELECT {", ".join(RAW_COLS)}
        FROM pitches
        WHERE game_type = 'reg_season' AND pitcher_id IS NOT NULL
    """)
    df = read_sql_df(q)
    df["date"] = pd.to_datetime(df["date"])
    print(f"loaded {len(df):,} rows")
    return canonical_order(df)


def derive_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize pitcher_throws, arm-side-adjust plate_x/horz_break/rel_side, derive
    swing/whiff/terminal flags and horiz_dist/vert_dist raw columns."""
    df = df.copy()
    throws_map = {"right": "R", "left": "L", "R": "R", "L": "L"}
    df["p_throws_clean"] = df["pitcher_throws"].map(throws_map)  # 'both'/'undefined'/NaN -> NaN
    arm_sign = np.where(df["p_throws_clean"] == "R", 1.0,
                         np.where(df["p_throws_clean"] == "L", -1.0, np.nan))
    df["plate_x_as"] = df["plate_x"] * arm_sign
    df["horz_break_as"] = df["horz_break"] * arm_sign
    df["rel_side_as"] = df["rel_side"] * arm_sign

    df["is_swing"] = df["swing"] == 1.0
    df["is_whiff"] = df["pitch_call"] == "strike_swinging"
    df["is_terminal"] = df["is_terminal_pitch"] == "1"

    zone_center_z = (df["sz_top"] + df["sz_bot"]) / 2
    df["horiz_dist_raw"] = df["plate_x"].abs()
    df["vert_dist_raw"] = (df["plate_z"] - zone_center_z).abs()
    return df


def filter_period(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """start/end are inclusive date strings (or None for open-ended)."""
    out = df
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out


def build_period_kbb(df: pd.DataFrame, min_pa: int = 1) -> pd.DataFrame:
    """One row per pitcher: PA, K, BB, K_rate, BB_rate over whatever rows are passed
    in (caller controls the window via filter_period / concatenating windows).

    Groups by pitcher_id ONLY -- not (pitcher_id, pitcher name). A handful of
    pitcher_ids carry more than one name value across their rows (a known
    upstream data issue, e.g. a single mislabeled game -- confirmed directly:
    pitcher_id 1000248636 is 'House, Tristan' on 26 appearances and
    'McCartney, Seth' on 1). Grouping on the composite key silently split a
    real pitcher's counts across multiple output rows, which is a correctness
    bug (undercounts each row's PA) and, worse, caused a cross-product
    explosion when two independently-built tables keyed on pitcher_id were
    later merged together (946 rows for 820 real pitchers, caught directly
    while adding the per-year K%/BB% split)."""
    term = df[df["is_terminal"]].copy()
    term["is_k"] = term["k_or_bb"] == "strikeout"
    term["is_bb"] = term["k_or_bb"] == "walk"
    agg = term.groupby("pitcher_id").agg(
        PA=("is_terminal", "size"), K=("is_k", "sum"), BB=("is_bb", "sum"),
    ).reset_index()
    agg["K_rate"] = agg["K"] / agg["PA"]
    agg["BB_rate"] = agg["BB"] / agg["PA"]
    return agg[agg["PA"] >= min_pa]


PARK_FACTOR_YEAR = "2025-2026"  # the combined-period row in park_factors_fip --
# more PA per park (~2x either single year alone) for a stabler estimate, same
# "fit once on the full available cross-section" convention used for n0/lg_rate
# elsewhere in this project.


def load_park_factors() -> pd.DataFrame:
    """Per-FL-team SO%/BB% park factors from Supabase's park_factors_fip table
    (already computed from real FL play-by-play, FIP-component-level, scale
    100=league average -- not something this project is deriving itself, just
    reading). Returns one row per team with so_factor/bb_factor as fractions
    (e.g. 1.02, not 102) ready to divide into a raw rate."""
    from fl_supabase import read_sql_df
    from sqlalchemy import text
    q = text('SELECT team, venue, "SO" as so_pf, "BB" as bb_pf FROM park_factors_fip WHERE year = :yr')
    df = read_sql_df(q, params={"yr": PARK_FACTOR_YEAR})
    df["so_factor"] = df["so_pf"] / 100.0
    df["bb_factor"] = df["bb_pf"] / 100.0
    return df[["team", "venue", "so_factor", "bb_factor"]]


def attach_park_factors(df: pd.DataFrame, park_factors: pd.DataFrame) -> pd.DataFrame:
    """Joins each pitch row to its GAME's park factor via home_team (the park
    a game is played at is always the home team's, regardless of which team
    the pitcher is on) -- not the pitcher's own team. Missing/unmatched games
    (e.g. a team not in park_factors_fip) get factor=1.0 (neutral, no
    adjustment) rather than being dropped."""
    out = df.merge(park_factors[["team", "so_factor", "bb_factor"]],
                    left_on="home_team", right_on="team", how="left")
    out["so_factor"] = out["so_factor"].fillna(1.0)
    out["bb_factor"] = out["bb_factor"].fillna(1.0)
    return out.drop(columns=["team"])


def build_period_kbb_park_adjusted(df: pd.DataFrame, park_factors: pd.DataFrame, min_pa: int = 1) -> pd.DataFrame:
    """Same shape as build_period_kbb, but K/BB credit for each terminal PA is
    divided by that GAME's park factor before summing -- a strikeout at a
    K-favoring park (so_factor>1) counts for less, a walk at a BB-suppressing
    park (bb_factor<1) counts for more. PA itself is NOT adjusted (park
    affects event PROBABILITY, not how many batters a pitcher faces).
    adjusted_K/adjusted_BB are non-integer by construction (a park-factor-
    scaled credit, not a raw count) -- expected, not a bug."""
    term = df[df["is_terminal"]].copy()
    term = attach_park_factors(term, park_factors)
    term["is_k"] = term["k_or_bb"] == "strikeout"
    term["is_bb"] = term["k_or_bb"] == "walk"
    term["k_adj"] = term["is_k"] / term["so_factor"]
    term["bb_adj"] = term["is_bb"] / term["bb_factor"]
    agg = term.groupby("pitcher_id").agg(
        PA=("is_terminal", "size"), K=("is_k", "sum"), BB=("is_bb", "sum"),
        K_adj=("k_adj", "sum"), BB_adj=("bb_adj", "sum"),
    ).reset_index()
    agg["K_rate"] = agg["K"] / agg["PA"]
    agg["BB_rate"] = agg["BB"] / agg["PA"]
    agg["K_rate_park_adj"] = agg["K_adj"] / agg["PA"]
    agg["BB_rate_park_adj"] = agg["BB_adj"] / agg["PA"]
    return agg[agg["PA"] >= min_pa]


def load_run_values() -> dict:
    """FL-specific linear weights (runs above a league-average PA) and the
    league's own runs-per-win, both from Supabase tables already maintained
    by this project (`woba_weights`, `war_context`) -- empirically derived
    from real FL play-by-play, not an imported generic MLB constant.

    run_value_k: `woba_weights.out_value` (a strikeout is a specific kind of
    out; this table's "out_value" is the same constant used as every other
    row's baseline, i.e. the run value of making an out vs. a league-average
    PA). run_value_bb: `woba_weights.rv_above_avg` for the BB row (already
    zero-centered on league-average PA, same convention as run_value_k, so
    the two are directly combinable). rpw: `war_context` lg/rpw, PA-weighted
    across every available reg_season year (not a single year, same "fit on
    the full available cross-section" convention used for n0/lg_rate
    elsewhere in this project)."""
    from fl_supabase import read_sql_df
    from sqlalchemy import text

    wq = read_sql_df(text("SELECT outcome, out_value, rv_above_avg FROM woba_weights"))
    run_value_bb = float(wq.loc[wq["outcome"] == "BB", "rv_above_avg"].iloc[0])
    run_value_k = float(wq["out_value"].iloc[0])

    rpw_q = read_sql_df(text("SELECT year, value FROM war_context "
                              "WHERE scope='lg' AND key='rpw' AND game_type='reg_season'"))
    pa_q = read_sql_df(text("SELECT year, value FROM war_context "
                             "WHERE scope='lg' AND key='pa' AND game_type='reg_season'"))
    rpw_m = rpw_q.set_index("year")["value"]
    pa_m = pa_q.set_index("year")["value"]
    rpw = float((rpw_m * pa_m).sum() / pa_m.sum())

    return {"run_value_k": run_value_k, "run_value_bb": run_value_bb, "rpw": rpw}


def compute_bf_per_g(df: pd.DataFrame) -> pd.DataFrame:
    """Batters faced per game (BF/G) -- a usage/role proxy (starters: high,
    typically 15-25+; relievers: low, typically 3-8) over whatever window is
    passed in. Already validated as a strong lever in this project's earlier FL
    `wh_sva` precedent work; not yet tested as a direct K%/BB% regressor."""
    term = df[df["is_terminal"]].dropna(subset=["pitcher_id", "game_id"])
    agg = term.groupby("pitcher_id").agg(
        pa=("is_terminal", "size"), n_games=("game_id", "nunique"),
    ).reset_index()
    agg["bf_per_g"] = agg["pa"] / agg["n_games"]
    return agg[["pitcher_id", "bf_per_g", "n_games"]]


ZONE_HALF_WIDTH = 0.83  # standard plate_x in-zone half-width (ft), accounting for ball radius
FASTBALL_TYPES = ["four_seam", "sinker"]


def compute_fb_zone_pct(df: pd.DataFrame, min_fb: int = 1) -> pd.DataFrame:
    """Per-pitcher fraction of four-seam/sinker fastballs thrown in the strike
    zone, over whatever window is passed in -- tested as a BB% candidate
    regressor (see DECISION_LOG.md "Fastball Zone% as a BB% regressor"):
    held-out corr with future BB_rate = -0.415 (train-2025 -> actual
    1H2026), stronger than horiz_dist/vert_dist's own concurrent corr
    (~0.38) and only moderately redundant with them (corr -0.38/-0.46).
    Moderately sticky (self-corr ~0.39-0.45 across splits, vs BB_rate's own
    ~0.50-0.55) with n0~157 fastballs (~77 PA-equivalent) -- noisier than the
    model's existing regressors but a real, usable signal, not pure noise."""
    fb = df[df["pitch_type"].isin(FASTBALL_TYPES)].dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot"]).copy()
    fb["in_zone"] = (fb["plate_x"].abs() <= ZONE_HALF_WIDTH) & (fb["plate_z"] >= fb["sz_bot"]) & (fb["plate_z"] <= fb["sz_top"])
    agg = fb.groupby("pitcher_id").agg(fb_n=("in_zone", "size"), fb_zone_pct=("in_zone", "mean")).reset_index()
    return agg[agg["fb_n"] >= min_fb]


WH_SVA_ESTIMATOR = "xgb"      # 'xgb' (original; the default, so every script that does not choose keeps its numbers) or 'torch3'; fl_production_projection.py sets 'torch3' for the 2027 pitcher projection
_NN_CACHE = {}


def compute_pooled_wh_sva(df: pd.DataFrame, min_rows: int = 2000, estimator: str | None = None) -> pd.DataFrame:
    """Per-swing stuff value (sva) with the chosen estimator: 'xgb' = the original XGBoost pair, 'torch3' = PyTorch MLPs averaged over 3 seeds (fl_wh_sva_nn_worker.py, run as a separate process).
    Same inputs, same pitcher-held-out cross-fit, same output columns."""
    est = estimator or WH_SVA_ESTIMATOR
    if est == "xgb":
        return compute_pooled_wh_sva_xgb(df, min_rows)
    if est == "torch3":
        return _wh_sva_torch3(df, min_rows)
    raise ValueError(f"unknown wh_sva estimator {est!r}")


def _wh_sva_torch3(df: pd.DataFrame, min_rows: int = 2000) -> pd.DataFrame:
    import os
    import subprocess
    import sys
    cols = FULL_COLS + ["pitcher_id", "is_whiff", "date", "game_id", "pa_id", "pitch_no"]
    pop = df[df["is_swing"]].dropna(subset=cols).copy()
    if len(pop) < min_rows:
        print(f"  [pooled wh_sva, torch3] only {len(pop)} rows -- skipping")
        return pd.DataFrame(columns=["pitcher_id", "date", "game_id", "pa_id", "pitch_no", "sva"])
    key = int(pd.util.hash_pandas_object(pop[["pitcher_id", "game_id", "pa_id", "pitch_no"]].astype(str), index=False).sum())
    if key in _NN_CACHE:
        return _NN_CACHE[key].copy()
    here = os.path.dirname(os.path.abspath(__file__)); tmp = os.path.join(here, "output", "tmp_nn"); os.makedirs(tmp, exist_ok=True)
    fin, fout = os.path.join(tmp, f"in_{key}.parquet"), os.path.join(tmp, f"out_{key}.parquet")
    pop[cols].to_parquet(fin, index=False)
    r = subprocess.run([sys.executable, os.path.join(here, "fl_wh_sva_nn_worker.py"), fin, fout], cwd=here, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if r.returncode != 0 or not os.path.exists(fout):
        raise RuntimeError("fl_wh_sva_nn_worker.py failed (see its output above)")
    out = pd.read_parquet(fout)
    for p in (fin, fout):
        os.remove(p)
    _NN_CACHE[key] = out; return out.copy()


def compute_pooled_wh_sva_xgb(df: pd.DataFrame, min_rows: int = 2000) -> pd.DataFrame:
    """Swing-conditioned, pooled-across-pitch-type shape-value-added at the per-pitch
    level: P(whiff | swing, location+shape)_oof - P(whiff | swing, location)_oof,
    via GroupKFold(5, groups=pitcher_id) so no pitcher's own pitches inform their own
    OOF prediction. Fit fresh on whatever window `df` already represents -- callers
    must pass only a single experiment's TRAINING window to avoid leaking the test
    period's shape-location relationship into the model."""
    pop = df[df["is_swing"]]
    needed = FULL_COLS + ["pitcher_id", "is_whiff", "date", "game_id", "pa_id", "pitch_no"]
    pop = pop.dropna(subset=needed).copy()
    if len(pop) < min_rows:
        print(f"  [pooled wh_sva] only {len(pop)} rows -- skipping")
        return pd.DataFrame(columns=["pitcher_id", "date", "game_id", "pa_id", "pitch_no", "sva"])

    y = pop["is_whiff"].astype(int).values
    groups = pop["pitcher_id"].values
    gkf = GroupKFold(n_splits=5)
    t0 = time.time()
    oof_loc = cross_val_predict(XGBClassifier(**XGB_PARAMS), pop[LOC_COLS].astype(float).values,
                                 y, cv=gkf, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    oof_full = cross_val_predict(XGBClassifier(**XGB_PARAMS), pop[FULL_COLS].astype(float).values,
                                  y, cv=gkf, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    pop["sva"] = oof_full - oof_loc
    print(f"  [pooled wh_sva] n={len(pop):,}, whiff_rate={y.mean():.4f}, "
          f"sva_std={pop['sva'].std():.4f}, took {time.time()-t0:.0f}s")
    return pop[["pitcher_id", "date", "game_id", "pa_id", "pitch_no", "sva"]]


def compute_stuff_plus(wh_sva_pitch: pd.DataFrame, mean_target: float = 100.0,
                        sd_target: float = 30.0) -> tuple[pd.DataFrame, float, float]:
    """Display-only rescaling of wh_sva to a 'Stuff+'-style metric: 100 = league
    average, SD = 30. Standardized at the PITCH level (mean/SD computed over every
    individual pitch, not over pitcher-level averages, which would have much less
    spread) and then averaged per pitcher -- same convention as the existing Stuff+
    Dashboard's own MU/SIGMA rescaling. This is purely for human-readable display;
    the production projection model keeps using the raw (unscaled) wh_sva_level as
    its regressor, since that's the scale its frozen coefficients were fit on."""
    mu = wh_sva_pitch["sva"].mean()
    sigma = wh_sva_pitch["sva"].std()
    pitch_level = mean_target + (wh_sva_pitch["sva"] - mu) / sigma * sd_target
    out = (wh_sva_pitch.assign(stuff_plus=pitch_level)
           .groupby("pitcher_id").agg(stuff_plus=("stuff_plus", "mean")).reset_index())
    return out, float(mu), float(sigma)


MIN_TYPE_N_FOR_LOC_VAR = 15  # per-(pitcher, pitch_type) pitch floor before that type counts
MIN_TYPE_N_FOR_SPIN_BASELINE = 20  # per-pitch-type pitch floor before it gets its own spin baseline


def compute_spin_ext_features(df: pd.DataFrame, baseline_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-pitcher spin_z_norm (pitch-type-normalized spin rate) and
    extension (raw mean) -- validated K% regressors added alongside
    wh_sva_level (see DECISION_LOG.md "Adding spin_rate + extension as
    direct K% regressors").

    spin_rate spans ~1200rpm across pitch types in this data (splitter
    ~1170 vs sweeper ~2420) -- a pitcher's raw mean is driven as much by
    THEIR PITCH MIX as by how much spin they generate on a given pitch.
    spin_z_norm instead expresses each pitch's spin_rate as a z-score
    relative to that SAME pitch type's own league baseline, then averages
    per pitcher -- beat the raw mean head-to-head in a genuinely held-out
    check (K% R2_w 0.2979->0.3044). extension varies far less by pitch type
    (~0.8ft span) so its raw mean needs no such correction.

    baseline_df sets where the pitch-type mean/std for the z-score come
    from -- pass the TRAINING window explicitly (not df itself) when df is
    a test/scoring window, to avoid leaking test-window spin distributions
    into the normalization; defaults to df when not given (in-sample use,
    e.g. fitting on a single window with no separate held-out target)."""
    baseline_df = df if baseline_df is None else baseline_df
    type_stats = (baseline_df.dropna(subset=["spin_rate", "pitch_type"])
                  .groupby("pitch_type")["spin_rate"].agg(["mean", "std", "size"]))
    type_stats = type_stats[type_stats["size"] >= MIN_TYPE_N_FOR_SPIN_BASELINE]
    type_stats = type_stats.rename(columns={"mean": "_type_mean", "std": "_type_std"})[["_type_mean", "_type_std"]]

    d = df.dropna(subset=["spin_rate", "extension", "pitch_type", "pitcher_id"]).copy()
    d = d.merge(type_stats, on="pitch_type", how="left")
    d["_spin_z"] = (d["spin_rate"] - d["_type_mean"]) / d["_type_std"]

    out = d.groupby("pitcher_id").agg(
        spin_z_norm=("_spin_z", "mean"), extension=("extension", "mean"), n_spin_ext=("_spin_z", "size"),
    ).reset_index()
    return out


def compute_loc_var(df: pd.DataFrame, min_pitches: int = 1) -> pd.DataFrame:
    """Per-pitcher location VARIANCE (command consistency/repeatability) --
    distinct from compute_dist_features' horiz_dist/vert_dist (mean absolute
    distance from zone center, i.e. AIM/accuracy). Computed WITHIN each pitch
    type first (a slider sitting lower than a fastball is intentional
    targeting, not inconsistency) then pooled (n-weighted) across pitch types
    per pitcher, so between-pitch-type location differences don't inflate it.

    Tested as a BB% candidate (see DECISION_LOG.md "Location variance as a
    BB% regressor/informed-prior input"): beats horiz_dist/vert_dist on every
    axis checked -- held-out corr with future BB_rate (0.274/0.365 vs.
    0.214/0.310), comparable-or-better stickiness (~0.74-0.78 vs ~0.70-0.77)
    -- but is also highly correlated with them (r=0.85/0.87), reading as a
    cleaner measurement of the same underlying command trait rather than an
    independent one."""
    d = df.dropna(subset=["plate_x_as", "plate_z", "pitch_type", "pitcher_id"]).copy()
    grp = d.groupby(["pitcher_id", "pitch_type"]).agg(
        n=("plate_x_as", "size"), hvar=("plate_x_as", "var"), vvar=("plate_z", "var"),
    ).reset_index()
    grp = grp[grp["n"] >= MIN_TYPE_N_FOR_LOC_VAR]

    def _pool(g):
        w = g["n"]
        return pd.Series({"horiz_var": np.average(g["hvar"], weights=w),
                           "vert_var": np.average(g["vvar"], weights=w), "n_pitches": w.sum()})

    out = grp.groupby("pitcher_id").apply(_pool, include_groups=False).reset_index()
    return out[out["n_pitches"] >= min_pitches]


# Frozen MLB-fit translation (mlb_opencommand_locvar_test.py / fl_command_score_model_test.py):
# WLS fit of real github.com/tomdoyo/open-command "inferred_in" (computer-vision-derived
# distance from the catcher's actual inferred target, NOT dispersion around a pitcher's own
# mean) on horiz_var/vert_var computed the SAME way as compute_loc_var above, on n=769
# workload-qualified MLB pitchers (2024+2025, >=300 pitches AND >=300 OpenCommand-tracked
# pitches each -- excludes position-player mop-up cameos, confirmed directly: the raw,
# unfiltered match included real position players like Amed Rosario/Tucker Barnhart with
# only 17-31 tracked pitches). R2=0.68. Z-scored check confirms vert_var genuinely matters
# more than horiz_var (0.70in/SD vs 0.45in/SD), not an artifact of raw-scale coefficients
# looking similar. See DECISION_LOG.md "OpenCommand-calibrated Command Score replaces
# horiz_var/vert_var in the BB% model" for the full validation, including the held-out win
# over horiz_var+vert_var used directly (R2_w 0.3517->0.3610) and the n0~=126-pitch
# reliability check (real but not a concern for the full-season population this feeds).
COMMAND_SCORE_INTERCEPT = 4.424373963818792
COMMAND_SCORE_HORIZ_COEF = 3.9415731433625307
COMMAND_SCORE_VERT_COEF = 4.281096060834336
COMMAND_SCORE_HORIZ_RANGE = (0.353837, 1.152937)  # MLB training support -- clip FL inputs to this
COMMAND_SCORE_VERT_RANGE = (0.262952, 1.694887)   # before predicting, same discipline as RA9's
                                                   # clip_to_training_range (can't trust the fit
                                                   # outside the range it was validated on)


def compute_command_score(df: pd.DataFrame, min_pitches: int = 1) -> pd.DataFrame:
    """Per-pitcher horiz_var/vert_var (compute_loc_var) translated into
    OpenCommand-equivalent inferred-miss inches via the frozen MLB
    regression above. Lower = better command (fewer inches of real miss
    from the catcher's actual target, as OpenCommand measures it on MLB
    broadcast video) -- opposite direction from raw horiz_var/vert_var only
    in the sense that this is now a genuine "miss" scale, not a variance."""
    lv = compute_loc_var(df, min_pitches=min_pitches)
    hv_c = lv["horiz_var"].clip(*COMMAND_SCORE_HORIZ_RANGE)
    vv_c = lv["vert_var"].clip(*COMMAND_SCORE_VERT_RANGE)
    lv["command_score"] = COMMAND_SCORE_INTERCEPT + COMMAND_SCORE_HORIZ_COEF * hv_c + COMMAND_SCORE_VERT_COEF * vv_c
    return lv


def compute_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pitcher-window mean horiz_dist/vert_dist, winsorized at 1st/99th pct of
    whatever window is passed in (not swing-conditioned -- location-precision metric
    independent of the batter's decision)."""
    pop = df.dropna(subset=["horiz_dist_raw", "vert_dist_raw", "pitcher_id"]).copy()
    lo_h, hi_h = pop["horiz_dist_raw"].quantile([0.01, 0.99])
    lo_v, hi_v = pop["vert_dist_raw"].quantile([0.01, 0.99])
    pop["horiz_dist_w"] = pop["horiz_dist_raw"].clip(lo_h, hi_h)
    pop["vert_dist_w"] = pop["vert_dist_raw"].clip(lo_v, hi_v)
    agg = pop.groupby("pitcher_id").agg(
        horiz_dist=("horiz_dist_w", "mean"), vert_dist=("vert_dist_w", "mean"),
        n_dist=("horiz_dist_w", "size"),
    ).reset_index()
    return agg


def _per_start_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-(pitcher, game) mean of value_col, in full chronological order via
    (date, game_id, pa_id, pitch_no) -- one row per start, ordered."""
    d = df.dropna(subset=[value_col, "pitcher_id", "date", "game_id", "pa_id", "pitch_no"]).copy()
    d = d.sort_values(["pitcher_id", "date", "game_id", "pa_id", "pitch_no"])
    agg = d.groupby(["pitcher_id", "date", "game_id"], sort=False).agg(
        mean_val=(value_col, "mean"), n=(value_col, "size")
    ).reset_index()
    # re-sort by (date, game_id) within pitcher to restore chronological start order
    # (groupby(sort=False) preserves first-seen order, which already follows the
    # pre-sorted frame, but make it explicit rather than relying on that)
    agg = agg.sort_values(["pitcher_id", "date", "game_id"]).reset_index(drop=True)
    return agg


def detect_changepoint(df: pd.DataFrame, value_col: str,
                        min_starts: int = MIN_STARTS_FOR_CHANGEPOINT) -> pd.DataFrame:
    """Real changepoint detection (ruptures, PELT, L2/mean-shift cost) run on each
    pitcher's own per-start chronological series of `value_col`, WITHIN whatever
    window `df` represents. Returns one row per pitcher_id with the raw detected
    shift magnitude (post-breakpoint mean - pre-breakpoint mean; 0 if no changepoint
    or too few starts) and n_starts (used for shrinkage weighting by the caller)."""
    starts = _per_start_series(df, value_col)
    rows = []
    for pid, g in starts.groupby("pitcher_id", sort=False):
        vals = g["mean_val"].values
        n_starts = len(vals)
        if n_starts < min_starts:
            rows.append({"pitcher_id": pid, "changepoint_raw": 0.0, "n_starts": n_starts,
                         "has_changepoint": False})
            continue
        algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(vals.reshape(-1, 1))
        try:
            bkps = algo.predict(pen=np.log(n_starts) * np.var(vals) if np.var(vals) > 0 else 1.0)
        except Exception:
            bkps = [n_starts]
        bkps = [b for b in bkps if b < n_starts]  # drop the trailing len(series) sentinel
        if not bkps:
            rows.append({"pitcher_id": pid, "changepoint_raw": 0.0, "n_starts": n_starts,
                         "has_changepoint": False})
            continue
        # use the single largest-magnitude breakpoint if ruptures finds more than one
        best_mag, best_bk = 0.0, None
        for bk in bkps:
            before, after = vals[:bk], vals[bk:]
            if len(before) == 0 or len(after) == 0:
                continue
            mag = after.mean() - before.mean()
            if abs(mag) > abs(best_mag):
                best_mag, best_bk = mag, bk
        rows.append({"pitcher_id": pid, "changepoint_raw": best_mag, "n_starts": n_starts,
                     "has_changepoint": best_bk is not None})
    return pd.DataFrame(rows)


def shrink_changepoint(cp_df: pd.DataFrame, full_pop_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Shrink detected changepoint magnitudes via the same empirical-Bayes mechanism
    already used for MLB's within-season deltas (mlb_marcel.fit_shrinkage_params /
    apply_shrinkage) -- reusing the small-sample discipline, not the delta
    definition. Shrinkage params fit on the full population's per-start values of
    value_col (grouped by pitcher_id), never on the changepoint magnitudes
    themselves."""
    starts = _per_start_series(full_pop_df, value_col)
    sw2, sb2, gm = mm.fit_shrinkage_params(starts.rename(columns={"mean_val": value_col}),
                                            value_col, group_cols=("pitcher_id",), min_n=min(5, MIN_STARTS_FOR_CHANGEPOINT))
    out = cp_df.copy()
    shrunk, reliab = mm.apply_shrinkage(out["changepoint_raw"].fillna(0).values,
                                         out["n_starts"].clip(lower=1).values, sw2, sb2, 0.0)
    out["changepoint"] = shrunk
    out["changepoint_reliab"] = reliab
    return out
