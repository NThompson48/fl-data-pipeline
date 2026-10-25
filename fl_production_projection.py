"""
fl_production_projection.py -- builds real 2027 preseason K%/BB% projections for
the FL Player Projections Dashboard (approved plan:
C:\\Users\\215nt\\.claude\\plans\\let-s-discuss-how-this-merry-thacker.md).

Reuses Experiment 2's exact fitted `naive+other_league+surprise_prior` model
(the best genuinely held-out performer in fl_halfseason_backtest.py -- see
DECISION_LOG.md) WITHOUT refitting: those coefficients were fit on full-2025
features -> actual 1H2026 and already validated by freezing them and scoring on
2025+1H2026 -> actual 2H2026. Projecting 2027 is one more roll-forward of that
same frozen model onto full-2025+2026 features. `changepoint` is excluded
entirely, per the DECISION_LOG finding that it doesn't survive held-out testing.

Usage:
    python fl_production_projection.py
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_college_bridge as bridge
import fl_pioneer_bridge as pbridge
import fl_milb_bridge as mbridge
import fl_unified_bridge as ub
import fl_bridge_matching as bm
import mlb_marcel as mm

SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]

# BB% only: the bridge enters as an informative PRIOR (Marcel-style pseudo-PA
# blend, mlb_marcel.blended_rate/fit_n0_bridge) instead of a free OLS term --
# a genuinely held-out win for BB%, a wash for K% at full-season PA (so K%
# keeps SPEC["K"]'s plain OLS bridge term unchanged). See DECISION_LOG.md
# "BB% bridge as an informative prior".
BB_SPEC_BLENDED = ["blended_BB_rate", "command_score"]

K_SE_CALIBRATION = 1.25  # empirical coverage correction for K%'s prediction SE.
# Checked directly against real held-out outcomes (2025+1H2026 -> actual
# 2H2026, n=260): the raw WLS se_obs UNDERcovers -- a nominal 90% interval
# only actually contained the true K_rate 83.8% of the time (BB% was fine:
# 89.6% actual vs 90% nominal, no correction needed there). Solved for the
# scale factor that restores nominal coverage at 50%/80%/90% separately
# (1.27/1.18/1.27 -- consistent enough across levels to mean this is a
# simple uniform under-estimate, not a shape problem) and took 1.25 as a
# round, representative value; re-verified it restores ~49%/83%/89% actual
# coverage at those same nominal levels. See DECISION_LOG.md "K% prediction
# interval calibration correction".


def build_canonical_id_map(raw: pd.DataFrame) -> dict:
    """Map pitcher_id -> canonical pitcher_id, consolidating ids that are the
    same real person reissued a new id mid-career (confirmed directly: 'Williams,
    Brian' has two ids, both on 'Mississippi Mud Monsters', with all 2026 data
    under only one of them -- the Player Page was silently showing the OTHER
    id's row since both share one display name. 22 such (name, team) groups
    exist across the full roster, not just this one case).

    Heuristic: group by (pitcher name, most-common team); within a group with
    >1 distinct id, remap every id to whichever has the most total rows (the
    fullest-history id). Ids that share a name but NOT a team are left
    unmerged -- deliberately conservative, since this project has also
    confirmed genuine different-real-people name collisions elsewhere (~1% of
    cases): an unmerged pair is a recoverable display gap, a wrongly-merged
    pair would silently blend two different people's stats."""
    primary_team = raw.groupby("pitcher_id")["pitcher_team"].agg(lambda s: s.mode().iloc[0])
    name_of = raw.groupby("pitcher_id")["pitcher"].first()
    row_counts = raw.groupby("pitcher_id").size()
    key_df = pd.DataFrame({"name": name_of, "team": primary_team, "n": row_counts}).reset_index()

    id_map = {}
    for (name, team), g in key_df.groupby(["name", "team"]):
        if len(g) == 1:
            continue
        canonical = g.loc[g["n"].idxmax(), "pitcher_id"]
        for pid in g["pitcher_id"]:
            if pid != canonical:
                id_map[pid] = canonical
    return _extend_id_map_cross_team(raw, id_map, row_counts)


# --- cross-team merge: same name, DIFFERENT teams (a traded/released-and-
# signed pitcher whose id was reissued). Legacy rule above refused these on
# purpose ("an unmerged pair is a recoverable display gap, a wrongly-merged
# pair silently blends two people"). Resolved with real evidence instead of a
# blanket refusal -- see DECISION_LOG.md "Duplicate pitcher names: resolved by
# handedness + date/game overlap + arsenal distance". Of the 16 duplicated
# names, 15 were the same person (sequential dates, same hand, arsenal within
# the same-person range); 'Davis, Tyler' is two people (opposite hands).
ARSENAL_TYPES = ["four_seam", "sinker", "slider", "changeup", "curveball", "cutter"]
ARSENAL_FEATS = ["effective_velo", "ivb", "horz_break_as", "spin_rate"]
ARSENAL_MIN_PITCHES = 15      # per pitch type, per id, to compare that type
ARSENAL_MIN_FEATURES = 7      # comparable features needed to trust the distance
ARSENAL_MAX_DIST = 0.80       # mean |diff| in between-pitcher SDs; the same
# person's 2025-vs-2026 arsenal had a 99th percentile of 0.83, different
# same-handed pitchers a median of 1.05 / 5th percentile of 0.63


def _arsenal_table(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[raw["pitch_type"].notna() & ~raw["pitch_type"].isin(["not_him", ""])]
    rows = {}
    for pid, s in d.groupby("pitcher_id"):
        r = {"n": len(s), "rz": s["release_pos_z"].mean(), "rs": s["rel_side_as"].mean(),
             "ext": s["extension"].mean()}
        for t in ARSENAL_TYPES:
            st = s[s["pitch_type"] == t]
            for k in ARSENAL_FEATS:
                r[f"{t}_{k}"] = st[k].mean() if len(st) >= ARSENAL_MIN_PITCHES else np.nan
        rows[pid] = r
    return pd.DataFrame(rows).T


def _extend_id_map_cross_team(raw: pd.DataFrame, id_map: dict, row_counts: pd.Series) -> dict:
    names = raw.groupby("pitcher_id")["pitcher"].first()
    hand = raw.groupby("pitcher_id")["p_throws_clean"].agg(lambda x: x.mode().iloc[0] if x.notna().any() else None)
    dup = names[names.duplicated(keep=False)]
    # work on ids still distinct after the legacy merge
    dup = dup[~dup.index.isin(id_map.keys())]
    if dup.empty:
        return id_map
    F = _arsenal_table(raw)
    feat_cols = [c for c in F.columns if c != "n"]
    sd = {c: F.loc[F["n"] >= 150, c].std() for c in feat_cols}
    dates = raw.groupby("pitcher_id")["date"].agg(set)
    games = raw.groupby("pitcher_id")["game_id"].agg(set)

    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    merged_log = []
    for name, grp in dup.groupby(dup):
        ids = list(grp.index)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if hand.get(a) != hand.get(b):
                    continue
                if dates[a] & dates[b] or games[a] & games[b]:
                    continue
                if a in F.index and b in F.index:
                    ds = [abs(F.at[a, c] - F.at[b, c]) / sd[c] for c in feat_cols
                          if pd.notna(F.at[a, c]) and pd.notna(F.at[b, c]) and sd[c] > 0]
                    if len(ds) >= ARSENAL_MIN_FEATURES and np.mean(ds) > ARSENAL_MAX_DIST:
                        continue
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
                merged_log.append(name)
    groups = {}
    for pid in {x for x in parent} | {find(x) for x in parent}:
        groups.setdefault(find(pid), set()).add(pid)
    out = dict(id_map)
    for root, members in groups.items():
        canonical = max(members, key=lambda p: row_counts.get(p, 0))
        for pid in members:
            if pid != canonical:
                out[pid] = canonical
    # flatten chains (legacy id -> canonical1 -> canonical2) so a single
    # Series.replace pass lands every id on its final canonical
    for k in list(out):
        v = out[k]
        while v in out and out[v] != v:
            v = out[v]
        out[k] = v
    print(f"  cross-team same-person merges: {len(set(merged_log))} names ({sorted(set(merged_log))})")
    return out

# Pitchers with a 2025 FL record but no 2026 appearance get a DISCOUNTED 2025
# prior instead of nothing (they were previously dropped to bridge/league mean,
# so 299 of 797 roster pitchers had no model projection). Equivalent-n0 for a
# lag-2 prior estimated from FL 2022-24 StatsCrew + 2025-26 pitch data:
# K% ~230 pseudo-PA (range 160-300), BB% ~150 (70-170), vs ~60 for a lag-1
# prior, i.e. a 2025 PA is worth 59/230 ~ 0.26 K-PA and 62/150 ~ 0.41 BB-PA.
# Weak evidence (the gap-only sample is n=18; lag-2 slopes K 0.56, BB 0.70 vs
# lag-1 0.86/0.87) -- shipped as a flagged fallback, not a validated model.
# Returns after a year off are rare (~3%). See DECISION_LOG.md "Gap prior".
STALE_K_FEATURE_RETAIN = 0.65
STALE_BB_FEATURE_RETAIN = 0.80
GAP_N0_K = 230.0
GAP_N0_BB = 150.0

OUT_PROJECTIONS = "output/fl_2027_projections.parquet"
OUT_PROSPECTS = "output/fl_college_prospect_lookup.parquet"
OUT_PROSPECT_PROJ = "output/fl_college_prospect_projections.parquet"
OUT_PIONEER_PROSPECTS = "output/fl_pioneer_prospect_lookup.parquet"
OUT_PIONEER_PROSPECT_PROJ = "output/fl_pioneer_prospect_projections.parquet"
OUT_MILB_PROSPECTS = "output/fl_milb_prospect_lookup.parquet"
OUT_MILB_PROSPECT_PROJ = "output/fl_milb_prospect_projections.parquet"
OUT_COEFS = "output/fl_projection_model_coefs.json"


def fit_frozen_model(raw):
    """Fit `naive+other_league+surprise_prior` on the exact Experiment 2 training
    transition (full-2025 -> actual 1H2026) -- the same, already held-out-
    validated fit used in fl_halfseason_backtest.py. Returns (m_k, m_bb, n0_k,
    n0_bb, n0_bridge_bb, rho_kbb) where the league rates are the 2025-window
    values used to neutrally impute missing regressors during that fit.

    BB%'s bridge is folded in as an informative prior (blended_BB_rate,
    n0_bridge_bb fit here by nonlinear least squares against this exact
    training transition's real outcomes, then frozen) instead of entering
    the regression as its own free OLS term -- see BB_SPEC_BLENDED above."""
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                      [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    d_tr_k = df_tr.dropna(subset=["K_rate_actual"] + SPEC["K"])
    d_tr_bb = df_tr.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA",
                                    "command_score"]).copy()
    w_k = mm.reliability_weight(d_tr_k["PA_test"], n0_k).values
    w_bb = mm.reliability_weight(d_tr_bb["PA_test"], n0_bb).values
    m_k = bt.fit_weighted(d_tr_k, "K_rate_actual", SPEC["K"], w_k)

    term_tr = fp.filter_period(raw, bt.START_2025, bt.END_2025)
    term_tr = term_tr[term_tr["is_terminal"]]
    lg_bb_rate_tr = (term_tr["k_or_bb"] == "walk").mean()
    d_tr_bb["BB_raw"] = d_tr_bb["naive_BB_rate"] * (d_tr_bb["PA"] + n0_bb) - n0_bb * lg_bb_rate_tr
    n0_bridge_bb = mm.fit_n0_bridge(d_tr_bb["BB_raw"], d_tr_bb["PA"], n0_bb, lg_bb_rate_tr,
                                     d_tr_bb["fl_bb_rate_bridge"], d_tr_bb["BB_rate_actual"], w_bb)
    d_tr_bb["blended_BB_rate"] = mm.blended_rate(d_tr_bb["BB_raw"], d_tr_bb["PA"], n0_bb, lg_bb_rate_tr,
                                                  n0_bridge_bb, d_tr_bb["fl_bb_rate_bridge"])
    print(f"  BB% bridge informative prior: n0_bridge_bb={n0_bridge_bb:.1f} "
          f"(fit on full-2025->actual-1H2026, n={len(d_tr_bb)})")
    m_bb = bt.fit_weighted(d_tr_bb, "BB_rate_actual", BB_SPEC_BLENDED, w_bb)

    # K%/BB% residual correlation (fit on the same training transition, common
    # pitchers only) -- needed to combine the two SEPARATELY-fit models' SEs
    # into a proper K-BB% uncertainty rather than naively assuming
    # independence (checked directly: real but modest, rho=-0.13, not 0).
    common_ids = set(d_tr_k["pitcher_id"]) & set(d_tr_bb["pitcher_id"])
    ck = d_tr_k[d_tr_k["pitcher_id"].isin(common_ids)].sort_values("pitcher_id")
    cbb = d_tr_bb[d_tr_bb["pitcher_id"].isin(common_ids)].sort_values("pitcher_id")
    resid_k = ck["K_rate_actual"].values - m_k.predict(ck).values
    resid_bb = cbb["BB_rate_actual"].values - m_bb.predict(cbb).values
    rho_kbb = float(np.corrcoef(resid_k, resid_bb)[0, 1])
    print(f"K%/BB% residual correlation (n={len(common_ids)}): rho={rho_kbb:.3f}")

    return m_k, m_bb, n0_k, n0_bb, n0_bridge_bb, rho_kbb


def _fl_name_to_college_clean(pitcher_name: str) -> str:
    return bm.fl_name_to_college_clean(pitcher_name)


def extend_match_by_name(feat: pd.DataFrame, prospect_proj: pd.DataFrame) -> pd.DataFrame:
    """Thin wrapper (+ match-count logging) around fl_bridge_matching's
    shared implementation -- see that module's docstring for why this is
    shared with fl_halfseason_backtest.py rather than duplicated (the two had
    quietly diverged, which understated the backtest's real held-out R2)."""
    before = int(feat["has_college_match"].sum())
    out = bm.extend_match_by_name(feat, prospect_proj)
    print(f"  name-based match extension: +{int(out['has_college_match'].sum()) - before} roster players "
          f"({int(out['has_college_match'].sum())} total matched of {len(out)})")
    return out


def extend_match_by_other_leagues(feat: pd.DataFrame, pioneer_full: pd.DataFrame,
                                   milb_full: pd.DataFrame) -> pd.DataFrame:
    """Thin wrapper (+ match-count logging) around fl_bridge_matching's
    shared implementation -- see that module's docstring."""
    out = bm.extend_match_by_other_leagues(feat, pioneer_full, milb_full)
    use_pio_k = out["k_bridge_source"] == "pioneer"
    use_milb_k = out["k_bridge_source"] == "milb"
    use_milb_bb = out["bb_bridge_source"] == "milb"
    use_pio_bb = out["bb_bridge_source"] == "pioneer"
    print(f"  other-league match extension: +{int(use_pio_k.sum())} via Pioneer (K%), "
          f"+{int(use_milb_k.sum())} via MiLB (K%), +{int(use_milb_bb.sum())} via MiLB (BB%), "
          f"+{int(use_pio_bb.sum())} via Pioneer (BB%) "
          f"({int(out['has_college_match'].sum())} total matched of {len(out)})")
    return out


def build_2027_projections(raw, m_k, m_bb, n0_bridge_bb: float, rho_kbb: float, run_values: dict,
                            ubridge: "ub.UnifiedBridge") -> pd.DataFrame:
    """Full 2025+2026 features (current roster: every pitcher with any FL
    activity in that window) -> 2027 projected K_rate/BB_rate via the frozen
    model fit above. Also attaches display-only extras: raw K%/BB% split by
    year (2025 vs 2026, not pooled) and a pitch-level-standardized Stuff+.

    The naive prior (naive_K_rate/naive_BB_rate/PA) uses 2026 ONLY, not the
    pooled 2025+2026 window -- tested directly in fl_flatpool_tuning.py: a
    1-year trailing window beats 2 or 3 years, consistently, on both K%/BB%
    and both the validation (2025) and held-out test (2026) checks (see
    DECISION_LOG.md "Flat-pool window size"). Everything else here (bridge
    match population, wh_sva_level/horiz_var/vert_var, team lookup) keeps
    the full 2025+2026 window -- that wasn't what was tested, so it isn't
    changed on the strength of a finding that was specifically about the
    naive rate."""
    feat, n0_k, n0_bb = bt.assemble_features(
        raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.END_2026)], with_changepoints=False)

    naive_2026, n0_k, n0_bb = bt.naive_prior(raw, [(bt.START_2026, bt.END_2026)])
    feat = feat.drop(columns=["naive_K_rate", "naive_BB_rate", "PA"]).merge(
        naive_2026[["pitcher_id", "PA", "naive_K_rate", "naive_BB_rate"]], on="pitcher_id", how="left")

    # Bridge match BEFORE computing blended_BB_rate below -- previously this
    # ran AFTER, so BB%'s informative prior silently used only the narrow
    # bridge-fitting match (the ~150-250 real transition pairs), never the
    # broader name/Pioneer/MiLB-matched fl_bb_rate_bridge the rest of this
    # project believed was live (K% was fine -- it reads fl_k_rate_bridge
    # later, already past this point). Also switches from the fixed
    # college>Pioneer>MiLB priority chain to the recency+reliability+TBF
    # weighted single-best-source pick -- see fl_bridge_matching.
    # recency_weighted_match and DECISION_LOG.md "Cross-league bridge match:
    # weighted blend vs single-best-source" / "Preston Johnson" entries.
    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    feat = feat.merge(names, on="pitcher_id", how="left")
    term_2026 = fp.filter_period(raw, bt.START_2026, bt.END_2026)
    term_2026 = term_2026[term_2026["is_terminal"]]
    lg_k_rate_2026 = (term_2026["k_or_bb"] == "strikeout").mean()
    lg_bb_rate_2026 = (term_2026["k_or_bb"] == "walk").mean()

    # Gap prior for pitchers with a 2025 line but no 2026 PA (see GAP_N0_*)
    feat["PA_model"] = feat["PA"]
    gap = feat["naive_K_rate"].isna()
    feat["is_gap"] = gap
    if gap.any():
        kbb25 = fp.build_period_kbb(fp.filter_period(raw, bt.START_2025, bt.END_2025), min_pa=1).set_index("pitcher_id")
        ids = feat.loc[gap, "pitcher_id"]
        pa25, k25, bb25 = (ids.map(kbb25[c]).astype(float) for c in ("PA", "K", "BB"))
        dk, dbb = n0_k / GAP_N0_K, n0_bb / GAP_N0_BB
        feat.loc[gap, "naive_K_rate"] = (dk * k25 + n0_k * lg_k_rate_2026) / (dk * pa25 + n0_k)
        feat.loc[gap, "naive_BB_rate"] = (dbb * bb25 + n0_bb * lg_bb_rate_2026) / (dbb * pa25 + n0_bb)
        feat.loc[gap, "PA_model"] = dbb * pa25
        print(f"  gap prior (2025 record, no 2026 PA): {int(gap.sum())} pitchers, "
              f"{int(feat.loc[gap, 'naive_K_rate'].notna().sum())} with a usable 2025 line")

    # UNIFIED other-league bridge (fl_unified_bridge.py): one TBF-shrunk, staleness-damped
    # input policy and one pooled mapping for College / Pioneer / MiLB, the source with the
    # highest reliability per metric. Replaces the three league bridges + the R2 x TBF x
    # 0.5^gap pick (DECISION_LOG.md "Unified vs by-league bridge: head-to-head").
    matched = ubridge.match(feat["pitcher_id"], feat["pitcher"], current_year=2026)
    feat = feat.drop(columns=["fl_k_rate_bridge", "fl_bb_rate_bridge", "has_college_match"], errors="ignore").merge(
        matched[["pitcher_id", "fl_k_rate_bridge", "fl_bb_rate_bridge", "has_college_match",
                 "k_bridge_source", "bb_bridge_source", "k_bridge_tbf", "bb_bridge_tbf"]], on="pitcher_id", how="left")
    feat["fl_k_rate_bridge"] = feat["fl_k_rate_bridge"].fillna(lg_k_rate_2026)
    feat["fl_bb_rate_bridge"] = feat["fl_bb_rate_bridge"].fillna(lg_bb_rate_2026)
    feat["has_college_match"] = feat["has_college_match"].fillna(False)
    feat["k_bridge_source"] = feat["k_bridge_source"].fillna("none")
    feat["bb_bridge_source"] = feat["bb_bridge_source"].fillna("none")
    print(f"  cross-league bridge match (unified bridge, highest-reliability source): "
          f"{int(feat['has_college_match'].sum())}/{len(feat)} matched "
          f"(K% source counts: {feat['k_bridge_source'].value_counts().to_dict()}, "
          f"BB% source counts: {feat['bb_bridge_source'].value_counts().to_dict()})")

    # BB%'s bridge is an informative prior (see fit_frozen_model) -- back out
    # this SAME 2026-only naive window's raw BB count from naive_BB_rate (the
    # window's own league BB rate, recomputed here to match exactly what
    # bt.naive_prior used internally) and blend it with the (now fully
    # matched) bridge via the frozen n0_bridge_bb.
    feat["BB_raw"] = feat["naive_BB_rate"] * (feat["PA_model"] + n0_bb) - n0_bb * lg_bb_rate_2026
    feat["blended_BB_rate"] = mm.blended_rate(feat["BB_raw"], feat["PA_model"], n0_bb, lg_bb_rate_2026,
                                               n0_bridge_bb, feat["fl_bb_rate_bridge"])

    # most recent team per pitcher within the 2025+2026 window -- a pitcher who
    # changed teams mid-window shows their LATEST team, matching "current roster"
    recent_team = (raw.sort_values("date").groupby("pitcher_id")["pitcher_team"]
                   .last().rename("team").reset_index())
    feat = feat.merge(recent_team, on="pitcher_id", how="left")

    # raw (unpooled) K%/BB% by year, for display -- the model itself still uses
    # the pooled naive_K_rate/naive_BB_rate above, unaffected by this
    for yr, start, end in [(2025, bt.START_2025, bt.END_2025), (2026, bt.START_2026, bt.END_2026)]:
        yr_kbb = fp.build_period_kbb(fp.filter_period(raw, start, end), min_pa=1)
        yr_kbb = yr_kbb[["pitcher_id", "PA", "K_rate", "BB_rate"]].rename(columns={
            "PA": f"PA_{yr}", "K_rate": f"k_rate_{yr}", "BB_rate": f"bb_rate_{yr}"})
        feat = feat.merge(yr_kbb, on="pitcher_id", how="left")

    # Stuff+ (pitch-level standardized wh_sva, display-only) over the same
    # 2025+2026 window the rest of this table uses
    feat_pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                           fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    wh_sva_pitch = fp.compute_pooled_wh_sva(feat_pool)
    stuff_plus_df, stuff_mu, stuff_sigma = fp.compute_stuff_plus(wh_sva_pitch)
    print(f"Stuff+ scaling: pitch-level mu={stuff_mu:.4f}, sigma={stuff_sigma:.4f}")
    feat = feat.merge(stuff_plus_df, on="pitcher_id", how="left")

    # A gap pitcher's pitch-derived level features (wh_sva, spin, extension,
    # command_score) are also a year older than everyone else's, so they get the
    # same lag-2 discount as the naive rate: the lag-2/lag-1 persistence ratio
    # (K 0.56/0.86 ~ 0.65, BB 0.70/0.87 ~ 0.80) of the pooled FL history. An
    # ASSUMPTION extended from the rate persistence (pitch features have only
    # one measurable lag in the data), not a fitted value.
    g = feat["is_gap"]
    for cols, keep in ((["wh_sva_level", "spin_z_norm", "extension"], STALE_K_FEATURE_RETAIN),
                       (["command_score"], STALE_BB_FEATURE_RETAIN)):
        for c in cols:
            mu = feat.loc[~g, c].mean()
            feat.loc[g, c] = mu + keep * (feat.loc[g, c] - mu)

    d_k = feat.dropna(subset=SPEC["K"]).copy()
    d_bb = feat.dropna(subset=BB_SPEC_BLENDED).copy()
    d_k["proj_K_rate"] = m_k.predict(d_k)
    d_bb["proj_BB_rate"] = m_bb.predict(d_bb)

    # Prediction uncertainty: OBSERVATION-level SE (se_obs includes both
    # parameter uncertainty and residual variance -- "how much could this
    # specific player's real future rate differ from the point estimate",
    # not just "how uncertain is the fitted line"). WLS already models
    # Var(residual) = scale/weight, so a low-PA pitcher's own reliability
    # weight naturally widens their own band -- no extra modeling needed.
    # Same reliability weight used for FITTING is reused here for scoring.
    w_k_new = mm.reliability_weight(d_k["PA_model"], n0_k).values
    w_bb_new = mm.reliability_weight(d_bb["PA_model"], n0_bb).values
    d_k["proj_K_rate_se"] = m_k.get_prediction(d_k, weights=w_k_new).se_obs * K_SE_CALIBRATION
    d_bb["proj_BB_rate_se"] = m_bb.get_prediction(d_bb, weights=w_bb_new).se_obs

    out = feat[["pitcher_id", "pitcher", "team", "PA", "PA_model", "naive_K_rate", "naive_BB_rate", "blended_BB_rate",
                "fl_k_rate_bridge", "fl_bb_rate_bridge", "has_college_match",
                "k_bridge_source", "bb_bridge_source",
                "wh_sva_level", "stuff_plus", "horiz_var", "vert_var", "command_score",
                "spin_z_norm", "extension",
                "PA_2025", "k_rate_2025", "bb_rate_2025", "PA_2026", "k_rate_2026", "bb_rate_2026"]].copy()
    out = out.merge(d_k[["pitcher_id", "proj_K_rate", "proj_K_rate_se"]], on="pitcher_id", how="left")
    out = out.merge(d_bb[["pitcher_id", "proj_BB_rate", "proj_BB_rate_se"]], on="pitcher_id", how="left")
    out["proj_basis"] = np.where(out["PA"].fillna(0) > 0, "recent", np.where(out["proj_K_rate"].notna(), "gap_prior", "none"))
    out["PA"] = out["PA"].fillna(0)   # true 2026 PA (the gap prior's discounted PA lives in PA_model)
    out["proj_KBB_rate"] = out["proj_K_rate"] - out["proj_BB_rate"]
    # Var(K-BB) = Var(K) + Var(BB) - 2*rho*SE_K*SE_BB -- NOT naive independence
    # (rho=-0.13, checked directly in fit_frozen_model; negative correlation
    # means K-BB%'s true uncertainty is slightly WIDER than independence would
    # suggest, since a high-K miss and a high-BB miss partially offset in K-BB%
    # less often than chance would predict).
    out["proj_KBB_rate_se"] = np.sqrt(
        out["proj_K_rate_se"] ** 2 + out["proj_BB_rate_se"] ** 2
        - 2 * rho_kbb * out["proj_K_rate_se"] * out["proj_BB_rate_se"])
    out["reliability_weight"] = mm.reliability_weight(out["PA_model"], n0_k).values

    # Runs/wins translation of the SAME uncertainty, using FL's own linear
    # weights and runs-per-win (fl_pitcher_features.load_run_values) --
    # RunsAllowed = PA*(run_value_k*K_rate + run_value_bb*BB_rate), so its
    # variance is the standard Var(aX+bY) = a^2 Var(X) + b^2 Var(Y) +
    # 2ab*Cov(X,Y) expansion, PA-scaled, using the SAME rho_kbb already
    # fit for the K-BB% combination (NOT independence -- run_value_k is
    # negative and run_value_bb positive, and rho_kbb is also negative, so
    # their product is positive: the real correlation structure makes
    # RunsAllowed uncertainty WIDER than independence would suggest, same
    # direction as the K-BB% finding this is built on top of).
    rv_k, rv_bb, rpw = run_values["run_value_k"], run_values["run_value_bb"], run_values["rpw"]
    runs_var = out["PA_model"] ** 2 * (
        rv_k ** 2 * out["proj_K_rate_se"] ** 2 + rv_bb ** 2 * out["proj_BB_rate_se"] ** 2
        + 2 * rv_k * rv_bb * rho_kbb * out["proj_K_rate_se"] * out["proj_BB_rate_se"])
    out["proj_runs_se"] = np.sqrt(runs_var)
    out["proj_wins_se"] = out["proj_runs_se"] / rpw

    return out.sort_values("proj_KBB_rate", ascending=False).reset_index(drop=True)


RECENTER = True


def league_level_2026(raw):
    t = fp.filter_period(raw, bt.START_2026, bt.END_2026)
    t = t[t["is_terminal"]]
    return (t["k_or_bb"] == "strikeout").mean(), (t["k_or_bb"] == "walk").mean()


def recenter_to_league(proj: pd.DataFrame, lg_k: float, lg_bb: float):
    """Additive shift so the PA-weighted mean projection (weights = true 2026
    PA, over pitchers who actually pitched in 2026) equals the latest full-
    season league K%/BB%. The regression's intercept carries the league level
    of the single transition it was fit on, and league-wide K%/BB% moves
    between seasons (BB% went 10.4% -> 11.4% from 2025 to 2026), a common
    offset no per-pitcher feature can see. On the held-out Exp-2 transition
    recentering turned a +0.63pp K% / +0.54pp BB% PA-weighted bias into
    ~0/+0.28pp and lifted R2_w (fl_league_drift_check.py, DECISION_LOG.md
    "League-level drift"). Returns (proj, shift_k, shift_bb), shifts being the
    amount SUBTRACTED from the raw model output."""
    if not RECENTER:
        return proj, 0.0, 0.0
    m = (proj["PA"] > 0) & proj["proj_K_rate"].notna() & proj["proj_BB_rate"].notna()
    shift_k = float(np.average(proj.loc[m, "proj_K_rate"], weights=proj.loc[m, "PA"]) - lg_k)
    shift_bb = float(np.average(proj.loc[m, "proj_BB_rate"], weights=proj.loc[m, "PA"]) - lg_bb)
    proj = proj.copy()
    proj["proj_K_rate"] -= shift_k
    proj["proj_BB_rate"] -= shift_bb
    proj["proj_KBB_rate"] = proj["proj_K_rate"] - proj["proj_BB_rate"]
    print(f"  recentered to 2026 league level: K% shift {-shift_k*100:+.2f}pp, BB% shift {-shift_bb*100:+.2f}pp")
    return proj, shift_k, shift_bb


PIONEER_TEAM_NAMES = {
    "bh10500": "Boise Hawks", "gr11842": "Glacier Range Riders",
    "gj11797": "Grand Junction Jackalopes", "gv11791": "Great Falls Voyagers",
    "ic12132": "Idaho Falls Chukars", "mp13196": "Missoula Paddleheads",
    "no13328": "Northern Colorado Owlz", "ob13593": "Oakland Ballers",
    "or13455": "Ogden Raptors", "rv14089": "Rocky Mountain Vibes",
}


def build_pioneer_prospect_lookup() -> pd.DataFrame:
    """Every Pioneer League pitcher-season pulled (2021-2025), for the
    Other-League Lookup page's per-season history table -- the Pioneer-side
    analog of build_college_prospect_lookup. `teams` is a team CODE
    ("bh10500") for the original 2023-2025 pull and a plain full name
    ("Boise Hawks") for the 2021-2022 extension -- PIONEER_TEAM_NAMES.get's
    fallback-to-itself handles both without a separate code path."""
    import os
    df = pd.read_parquet("output/fl_pioneer_pitching_2021_2025.parquet")
    if os.path.exists("output/fl_pioneer_pitching_2026.parquet"):   # 2026 from Baseball Reference (fl_pioneer_2026.py)
        df = pd.concat([df, pd.read_parquet("output/fl_pioneer_pitching_2026.parquet")[df.columns.intersection(
            pd.read_parquet("output/fl_pioneer_pitching_2026.parquet").columns)]], ignore_index=True)
    out = df.rename(columns={"player": "name", "year": "season", "bf": "tbf"})
    out["team"] = out["teams"].apply(
        lambda s: "/".join(PIONEER_TEAM_NAMES.get(c, c) for c in s.split(",")))
    return out[["name", "team", "season", "tbf", "k_pct", "bb_pct"]].sort_values(
        ["name", "season"], ascending=[True, False]).reset_index(drop=True)


def build_milb_prospect_lookup() -> pd.DataFrame:
    """Every MiLB pitcher-season/team-stint in the cached MLB Stats API pull
    (2018-2026), for the Other-League Lookup page's per-season history table --
    the MiLB-side analog of build_college_prospect_lookup. Kept per level/team
    stint (not summed across a mid-season promotion) so a lookup can see e.g.
    'AA then AAA' within one year, same spirit as showing college conference
    moves separately."""
    df = pd.read_parquet("output/milb_pitching_2018_2026.parquet")
    out = df.rename(columns={"player_name": "name", "battersFaced": "tbf",
                              "K_rate": "k_pct", "BB_rate": "bb_pct"}).copy()
    out["team"] = out["team"] + " (" + out["level"] + ")"
    return out[["name", "team", "season", "tbf", "k_pct", "bb_pct"]].sort_values(
        ["name", "season"], ascending=[True, False]).reset_index(drop=True)


def finalize_other_league_projections(full: pd.DataFrame) -> pd.DataFrame:
    """Pioneer/MiLB's build_full_universe_predictions -> the same display shape
    build_college_prospect_projections already produces, so the dashboard's
    lookup page can treat all three leagues identically."""
    out = full.rename(columns={"player": "name"}).drop(columns=["name_clean"])
    out["fl_kbb_rate_bridge"] = out["fl_k_rate_bridge"] - out["fl_bb_rate_bridge"]
    return out.sort_values("fl_kbb_rate_bridge", ascending=False).reset_index(drop=True)


def build_college_prospect_lookup() -> pd.DataFrame:
    """Every college pitcher-season in the cached FanGraphs pulls (now including
    2026), for the 'hasn't played FL yet' search's per-season history table.
    Reuses the already-pulled/validated data -- no new scraping."""
    frames = []
    for path in sorted(glob.glob("output/fg_college_pit_20*.parquet")):
        frames.append(pd.read_parquet(path)[["name", "team", "season", "tbf", "k_pct", "bb_pct"]])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["name", "season"], ascending=[True, False]).reset_index(drop=True)


def build_college_prospect_projections(bridge_model: dict) -> pd.DataFrame:
    """One row per distinct college pitcher (every name in the cached FanGraphs
    pulls, 2023-2026): k_pct from their single most recent season, bb_pct_multi
    TBF-weighted across every season they have, and the bridge-predicted FL
    K%/BB% from both -- precomputed so the dashboard's lookup page never needs
    to recompute a multi-year average live.

    `college_years` is keyed on FanGraphs' own upid (see
    fl_college_bridge.load_all_college_years), not name_clean alone -- groups
    by upid FIRST, then collapses to one row per name_clean by keeping
    whichever real person's history is most recent (ties by total TBF), so a
    genuine different-person name collision doesn't blend two careers into
    one fictional composite the way a direct name_clean groupby would."""
    cy = bridge_model["college_years"]
    per_id_latest = (cy.sort_values("season").groupby("upid").tail(1)
                      [["upid", "name_clean", "name", "team", "season", "k_pct"]]
                      .rename(columns={"season": "latest_season"}))
    per_id_multi = cy.groupby("upid").apply(lambda g: pd.Series({
        "bb_pct_multi": np.average(g["bb_pct"], weights=g["tbf"]),
        "n_years": len(g), "tbf_latest": g.loc[g["season"].idxmax(), "tbf"], "tbf_multi": g["tbf"].sum(),
    }), include_groups=False).reset_index()
    per_id = per_id_latest.merge(per_id_multi, on="upid")

    out = (per_id.sort_values(["latest_season", "tbf_multi"])
           .groupby("name_clean").tail(1).reset_index(drop=True))
    out["fl_k_rate_bridge"] = bridge_model["k_model"].predict(out).values
    out["fl_bb_rate_bridge"] = bridge_model["bb_model"].predict(out).values
    out["fl_kbb_rate_bridge"] = out["fl_k_rate_bridge"] - out["fl_bb_rate_bridge"]
    out["college_r2_k"] = bridge_model["k_r2"]
    out["college_r2_bb"] = bridge_model["bb_r2"]
    return out.drop(columns=["upid"]).sort_values("fl_kbb_rate_bridge", ascending=False).reset_index(drop=True)


def project_from_college_stats(k_pct: float, bb_pct_multi: float, m_k, m_bb, bridge_model,
                                lg_k_rate: float, lg_bb_rate: float, n0_bb: float, n0_bridge_bb: float,
                                wh_sva_mean: float, command_score_mean: float,
                                spin_z_norm_mean: float, extension_mean: float,
                                level_shift_k: float = 0.0, level_shift_bb: float = 0.0) -> dict:
    """Project a pitcher with NO FL history from college stats -- used by BOTH
    the lookup (passing their real multi-year bb_pct_multi) and the manual-input
    calculator (a single typed BB% treated as if it were that average -- a
    reasonable stand-in in the absence of real season-by-season history), so
    they're one code path. K%: naive_K_rate = the league mean (a zero-FL-PA
    pitcher gets full shrinkage, exactly as fl_halfseason_backtest.naive_prior
    already produces for any real zero-PA pitcher). BB%: blended_BB_rate at
    raw_count=0, n=0 is the SAME informative-prior formula used everywhere
    else (mlb_marcel.blended_rate) -- at zero real PA it reduces to a pure
    n0_bb/n0_bridge_bb-weighted blend of the league mean and the bridge rate,
    rather than a separate zero-PA special case. wh_sva/command_score/
    spin_z_norm/extension = the population mean (no pitch-level FL data
    exists for a college-only arm, same neutral-imputation convention
    assemble_features already uses for an unmatched pitcher's bridge rate)."""
    row_k = pd.DataFrame([{"k_pct": k_pct}])
    row_bb = pd.DataFrame([{"bb_pct_multi": bb_pct_multi}])
    fl_k_rate_bridge = float(bridge_model["k_model"].predict(row_k).iloc[0])
    fl_bb_rate_bridge = float(bridge_model["bb_model"].predict(row_bb).iloc[0])

    k_row = pd.DataFrame([{"naive_K_rate": lg_k_rate, "fl_k_rate_bridge": fl_k_rate_bridge,
                            "wh_sva_level": wh_sva_mean, "spin_z_norm": spin_z_norm_mean,
                            "extension": extension_mean}])
    blended_bb = mm.blended_rate(0.0, 0.0, n0_bb, lg_bb_rate, n0_bridge_bb, fl_bb_rate_bridge)
    bb_row = pd.DataFrame([{"blended_BB_rate": blended_bb, "command_score": command_score_mean}])
    # same league-level recentering the roster projections get (see
    # recenter_to_league), so the calculator agrees with the leaderboard
    proj_k = float(m_k.predict(k_row).iloc[0]) - level_shift_k
    proj_bb = float(m_bb.predict(bb_row).iloc[0]) - level_shift_bb
    return {"fl_k_rate_bridge": fl_k_rate_bridge, "fl_bb_rate_bridge": fl_bb_rate_bridge,
            "proj_K_rate": proj_k, "proj_BB_rate": proj_bb, "proj_KBB_rate": proj_k - proj_bb}


def project_from_bridge_rates(fl_k_rate_bridge: float, fl_bb_rate_bridge: float, m_k, m_bb,
                               lg_k_rate: float, lg_bb_rate: float, n0_bb: float, n0_bridge_bb: float,
                               wh_sva_mean: float, command_score_mean: float,
                               spin_z_norm_mean: float, extension_mean: float,
                               level_shift_k: float = 0.0, level_shift_bb: float = 0.0) -> dict:
    """Project a pitcher with NO FL history from his (unified-bridge) other-league translation: the
    same neutral-imputation convention as project_from_college_stats (naive rate = league mean,
    stuff/command = population means), with the bridge rates already computed by
    fl_unified_bridge.translate()."""
    k_row = pd.DataFrame([{"naive_K_rate": lg_k_rate, "fl_k_rate_bridge": fl_k_rate_bridge,
                            "wh_sva_level": wh_sva_mean, "spin_z_norm": spin_z_norm_mean, "extension": extension_mean}])
    blended_bb = mm.blended_rate(0.0, 0.0, n0_bb, lg_bb_rate, n0_bridge_bb, fl_bb_rate_bridge)
    bb_row = pd.DataFrame([{"blended_BB_rate": blended_bb, "command_score": command_score_mean}])
    proj_k = float(m_k.predict(k_row).iloc[0]) - level_shift_k
    proj_bb = float(m_bb.predict(bb_row).iloc[0]) - level_shift_bb
    return {"fl_k_rate_bridge": fl_k_rate_bridge, "fl_bb_rate_bridge": fl_bb_rate_bridge,
            "proj_K_rate": proj_k, "proj_BB_rate": proj_bb, "proj_KBB_rate": proj_k - proj_bb}


def save_model_coefs(path, m_k, m_bb, bridge_coefs: dict,
                      lg_k_rate, lg_bb_rate, n0_bb, n0_bridge_bb,
                      wh_sva_mean, command_score_mean,
                      spin_z_norm_mean, extension_mean,
                      level_shift_k=0.0, level_shift_bb=0.0):
    """Persist the frozen model + all three "other league" bridges as plain
    coefficients (not statsmodels objects) so the Streamlit app can apply them
    with pure arithmetic -- keeps the app's own dependencies to
    streamlit/pandas/numpy/plotly only, no statsmodels/sqlalchemy/xgboost
    needed at runtime. Pioneer/MiLB bridges were fit via WLS (PA-weighted,
    see DECISION_LOG.md), but .params has the same shape as the college
    bridge's OLS fit, so this is the same payload shape for all three."""
    payload = {
        "k_model": {k: float(v) for k, v in m_k.params.items()},
        "bb_model": {k: float(v) for k, v in m_bb.params.items()},
        # the ONE unified other-league bridge (shrinkage, decay, staleness damping, pooled mapping)
        "bridge_unified": bridge_coefs,
        "lg_k_rate": float(lg_k_rate),
        "lg_bb_rate": float(lg_bb_rate),
        "n0_bb": float(n0_bb),
        "n0_bridge_bb": float(n0_bridge_bb),
        "wh_sva_mean": float(wh_sva_mean),
        "command_score_mean": float(command_score_mean),
        "spin_z_norm_mean": float(spin_z_norm_mean),
        "extension_mean": float(extension_mean),
        # amount SUBTRACTED from the raw model output (recenter_to_league)
        "level_shift_k": float(level_shift_k),
        "level_shift_bb": float(level_shift_bb),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    fp.WH_SVA_ESTIMATOR = "swstr_ct"    # user 2026-10-25 ("add it"): the per-pitch swinging-strike stuff network with pitch type and count feeds the 2027 pitcher projection as the K% model's stuff term (column wh_sva_level; the additive K% spec is unchanged
                                        # and the frozen model is refit with it below; replaced the swing-based network 'torch3' of 2026-10-23; ROS keeps its own XGBoost estimator until validated; see DECISION_LOG.md 2026-10-24/25 entries)
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()
    import fl_position_player_flag as ppf
    raw = ppf.exclude(raw)     # position players who threw mop-up innings are not pitchers (DECISION_LOG 2026-10-08); excluded from the fit, the league levels and the projections

    print("Fitting frozen Experiment-2 model (full-2025 -> actual 1H2026)...")
    # Fit on the ORIGINAL (unconsolidated) ids -- this must stay byte-for-byte
    # the same population Experiment 2 was validated on.
    m_k, m_bb, n0_k, n0_bb, n0_bridge_bb, rho_kbb = fit_frozen_model(raw)
    print(m_k.params)
    print(m_bb.params)

    print("\nConsolidating same-person, multi-id pitchers for the roster build...")
    id_map = build_canonical_id_map(raw)
    print(f"  {len(id_map)} ids remapped across {len(set(id_map.values()))} canonical pitchers")
    raw_roster = raw.copy()
    raw_roster["pitcher_id"] = raw_roster["pitcher_id"].replace(id_map)

    print("\nFitting the unified other-league bridge (College / Pioneer / MiLB debut pairs, one policy)...")
    ubridge = ub.fit(raw)
    prospect_tables = ubridge.prospect_tables(current_year=2026)
    for lg, path in (("college", OUT_PROSPECT_PROJ), ("pioneer", OUT_PIONEER_PROSPECT_PROJ), ("milb", OUT_MILB_PROSPECT_PROJ)):
        prospect_tables[lg].to_parquet(path, index=False)
        print(f"Saved -> {path}, shape={prospect_tables[lg].shape}; latest season counts "
              f"{prospect_tables[lg]['latest_season'].value_counts().sort_index().tail(3).to_dict()}")

    print("\nLoading FL-specific run values + runs-per-win (Supabase woba_weights/war_context)...")
    run_values = fp.load_run_values()
    print(f"  run_value_k={run_values['run_value_k']:.4f}  run_value_bb={run_values['run_value_bb']:.4f}  "
          f"rpw={run_values['rpw']:.2f}")

    print("\nBuilding 2027 projections (full 2025+2026 -> 2027)...")
    proj = build_2027_projections(raw_roster, m_k, m_bb, n0_bridge_bb, rho_kbb, run_values, ubridge)

    # (PA-stratified bootstrap risk tiers removed -- the dashboard replaced the
    # Risk column with P(WAR>0); see DECISION_LOG.md. fl_bootstrap_risk_tier.py
    # is left in place but no longer imported here.)
    print("\nRecentering projections to the latest observed league level...")
    lg_k_2026, lg_bb_2026 = league_level_2026(raw)
    proj, shift_k, shift_bb = recenter_to_league(proj, lg_k_2026, lg_bb_2026)

    proj.to_parquet(OUT_PROJECTIONS, index=False)
    print(f"Saved -> {OUT_PROJECTIONS}, shape={proj.shape}")
    print(proj[["pitcher", "PA", "proj_basis", "proj_K_rate", "proj_BB_rate", "proj_KBB_rate"]].head(15).to_string(index=False))

    print("\nBuilding college prospect lookup (per-season history table)...")
    prospects = build_college_prospect_lookup()
    prospects.to_parquet(OUT_PROSPECTS, index=False)
    print(f"Saved -> {OUT_PROSPECTS}, shape={prospects.shape}")

    print("\nBuilding Pioneer/MiLB prospect lookup + projection tables (Other-League Lookup page)...")
    pioneer_prospects = build_pioneer_prospect_lookup()
    pioneer_prospects.to_parquet(OUT_PIONEER_PROSPECTS, index=False)
    print(f"Saved -> {OUT_PIONEER_PROSPECTS}, shape={pioneer_prospects.shape}")

    milb_prospects = build_milb_prospect_lookup()
    milb_prospects.to_parquet(OUT_MILB_PROSPECTS, index=False)
    print(f"Saved -> {OUT_MILB_PROSPECTS}, shape={milb_prospects.shape}")

    # sanity check: the typed-in (calculator) path agrees with the lookup-table path for a real prospect
    # (single-season history, so the table's decayed K% average equals the calculator's single-season input)
    tb = prospect_tables["college"]
    sample = tb[(tb["n_years"] == 1) & (tb["tbf_latest"] >= 100)].iloc[0]
    fk, fb = ubridge.translate("college", sample["k_pct"], sample["bb_pct_multi"], sample["tbf_latest"], sample["tbf_multi"], int(sample["gap"]))
    assert abs(fk - sample["fl_k_rate_bridge"]) < 1e-9 and abs(fb - sample["fl_bb_rate_bridge"]) < 1e-9, \
        "calculator and lookup disagree on the unified bridge translation"
    # league rates = 2026 ONLY, the same ones build_2027_projections uses for
    # the naive prior and the BB% blend (the dashboard's calculator previously
    # got the pooled 2025+2026 rates, so it disagreed with the leaderboard)
    lg_k_rate, lg_bb_rate = lg_k_2026, lg_bb_2026
    wh_mean = proj["wh_sva_level"].mean()
    command_score_mean = proj["command_score"].mean()
    spin_z_norm_mean = proj["spin_z_norm"].mean()
    extension_mean = proj["extension"].mean()
    result = project_from_bridge_rates(fk, fb, m_k, m_bb, lg_k_rate, lg_bb_rate, n0_bb, n0_bridge_bb,
                                        wh_mean, command_score_mean, spin_z_norm_mean, extension_mean,
                                        level_shift_k=shift_k, level_shift_bb=shift_bb)
    print(f"\nSample prospect {sample['name']} (latest season {sample['latest_season']}, {sample['n_years']} yr(s), "
          f"{sample['tbf_latest']:.0f} TBF): college K%={sample['k_pct']:.3f} BB%={sample['bb_pct_multi']:.3f} -> {result}")

    print("\nSaving plain-arithmetic model coefficients for the dashboard...")
    save_model_coefs(OUT_COEFS, m_k, m_bb, ubridge.coef_dict(2026),
                      lg_k_rate, lg_bb_rate, n0_bb, n0_bridge_bb, wh_mean, command_score_mean,
                      spin_z_norm_mean, extension_mean, shift_k, shift_bb)
    print(f"Saved -> {OUT_COEFS}")
