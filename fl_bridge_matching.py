"""
fl_bridge_matching.py -- the "other-league" bridge name-matching chain, shared
between the production pipeline (fl_production_projection.py) and the
half-season backtest harness (fl_halfseason_backtest.py).

Extracted because the two had quietly diverged: fl_halfseason_backtest.py's
bridge_features() only ever joined the NARROW bridge-fitting table (the
college bridge's own ~152 real transition pairs, matched by exact pitcher_id),
never the name-extended college match (any season, "most recent on record")
or the Pioneer/MiLB bridges that production actually ships via
extend_match_by_name/extend_match_by_other_leagues. Every "naive+other_league"
R2 reported from the backtest was therefore understating what's live in
production. One shared implementation (used by both files now) keeps that
from happening again.

Three-tier match, same priority order used throughout this project: narrow
bridge-fitting match (exact pitcher_id, real transition pair) > name-matched
college (any season) > Pioneer/MiLB name match. Never overwrites an earlier
(higher-priority) match -- see extend_match_by_other_leagues' per-metric
priority (college > Pioneer/MiLB, validated real R2 decides which of Pioneer
vs MiLB per metric).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_college_bridge as bridge


def fl_name_to_college_clean(pitcher_name: str) -> str:
    """FL names are 'Last, First' -> reformat to 'First Last' before the same
    normalization fl_college_bridge.load_all_college_years uses, so FL and
    college names compare on equal footing."""
    s = str(pitcher_name)
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return bridge._clean_name(s)


def build_college_prospect_projections(bridge_model: dict) -> pd.DataFrame:
    """One row per distinct college pitcher (every name in the cached
    FanGraphs pulls): k_pct from their single most recent season,
    bb_pct_multi TBF-weighted across every season they have, and the
    bridge-predicted FL K%/BB% from both -- precomputed so a name-match
    lookup never needs to recompute a multi-year average live.

    Same upid-first grouping as fl_production_projection's copy of this
    function (kept as a separate copy so the backtest harness and
    production can each call it without a cross-module dependency) -- see
    that function's docstring for why name_clean alone isn't safe to group
    on directly."""
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


def extend_match_by_name(feat: pd.DataFrame, prospect_proj: pd.DataFrame) -> pd.DataFrame:
    """The narrow bridge-fitting sample only covers true FL rookies whose
    debut-year college stats were part of that specific transition match --
    most of any roster are returning veterans never eligible for that match
    by construction, not a missing-data bug -- but many DO have a findable
    college record by name once matched against the full college universe
    (prospect_proj). Fill in fl_k_rate_bridge/fl_bb_rate_bridge for any
    pitcher the narrow match missed but a name match finds -- never
    overwrites a real narrow match."""
    feat = feat.copy()
    feat["_name_clean"] = feat["pitcher"].apply(fl_name_to_college_clean)
    extra = pd.DataFrame({
        "_name_clean": prospect_proj["name"].apply(bridge._clean_name),
        "_extra_k": prospect_proj["fl_k_rate_bridge"],
        "_extra_bb": prospect_proj["fl_bb_rate_bridge"],
    }).drop_duplicates("_name_clean")

    feat = feat.merge(extra, on="_name_clean", how="left")
    need_fill = ~feat["has_college_match"] & feat["_extra_k"].notna()
    feat.loc[need_fill, "fl_k_rate_bridge"] = feat.loc[need_fill, "_extra_k"]
    feat.loc[need_fill, "fl_bb_rate_bridge"] = feat.loc[need_fill, "_extra_bb"]
    feat.loc[need_fill, "has_college_match"] = True
    # source tracking: every match here, narrow or name-extended, is college
    # data -- per-metric sources diverge once extend_match_by_other_leagues
    # can fill in Pioneer/MiLB instead.
    feat["k_bridge_source"] = np.where(feat["has_college_match"], "college", "none")
    feat["bb_bridge_source"] = np.where(feat["has_college_match"], "college", "none")
    return feat.drop(columns=["_name_clean", "_extra_k", "_extra_bb"])


def extend_match_by_other_leagues(feat: pd.DataFrame, pioneer_full: pd.DataFrame,
                                   milb_full: pd.DataFrame) -> pd.DataFrame:
    """Fills fl_k_rate_bridge/fl_bb_rate_bridge for pitchers the college match
    (narrow + name-extended) still misses, using Pioneer League and
    affiliated MiLB as two more 'other league' sources -- never overwrites an
    existing (college) match.

    Priority is set PER METRIC by each source's own validated held-out R2
    (fl_pioneer_bridge.py / fl_milb_bridge.py, PA-weighted fits, real
    Pioneer/MiLB-year-Y -> FL-year-Y+1 transition pairs):
        K%:  college > Pioneer (R2=0.096) > MiLB (R2=0.037)
        BB%: college > MiLB   (R2=0.198) > Pioneer (R2=0.096)
    Each source turned out stronger on a DIFFERENT metric -- MiLB's walk-rate
    signal transfers much better than its strikeout-rate signal, Pioneer is
    mediocre-but-even on both -- so K% and BB% are filled independently
    rather than picking one "best overall" source, the same asymmetric
    philosophy fl_college_bridge.py already uses (single-year K% vs
    multi-year BB%)."""
    feat = feat.copy()
    feat["_name_clean"] = feat["pitcher"].apply(fl_name_to_college_clean)

    pio = pioneer_full.rename(columns={"name_clean": "_name_clean",
                                        "fl_k_rate_bridge": "_pio_k", "fl_bb_rate_bridge": "_pio_bb"})
    mil = milb_full.rename(columns={"name_clean": "_name_clean",
                                     "fl_k_rate_bridge": "_milb_k", "fl_bb_rate_bridge": "_milb_bb"})
    feat = feat.merge(pio[["_name_clean", "_pio_k", "_pio_bb"]].drop_duplicates("_name_clean"),
                       on="_name_clean", how="left")
    feat = feat.merge(mil[["_name_clean", "_milb_k", "_milb_bb"]].drop_duplicates("_name_clean"),
                       on="_name_clean", how="left")

    # fl_k_rate_bridge/fl_bb_rate_bridge are NEVER actually NaN by this point --
    # an unmatched pitcher already carries a league-average FALLBACK CONSTANT
    # (or, in the backtest harness, still NaN pre-fill -- callers fillna after
    # this). has_college_match (not isna()) is the real "do we have genuine
    # bridge info" signal, same convention extend_match_by_name uses.
    still_unmatched = ~feat["has_college_match"]

    use_pio_k = still_unmatched & feat["_pio_k"].notna()
    feat.loc[use_pio_k, "fl_k_rate_bridge"] = feat.loc[use_pio_k, "_pio_k"]
    use_milb_k = still_unmatched & ~use_pio_k & feat["_milb_k"].notna()
    feat.loc[use_milb_k, "fl_k_rate_bridge"] = feat.loc[use_milb_k, "_milb_k"]

    use_milb_bb = still_unmatched & feat["_milb_bb"].notna()
    feat.loc[use_milb_bb, "fl_bb_rate_bridge"] = feat.loc[use_milb_bb, "_milb_bb"]
    use_pio_bb = still_unmatched & ~use_milb_bb & feat["_pio_bb"].notna()
    feat.loc[use_pio_bb, "fl_bb_rate_bridge"] = feat.loc[use_pio_bb, "_pio_bb"]

    feat.loc[use_pio_k, "k_bridge_source"] = "pioneer"
    feat.loc[use_milb_k, "k_bridge_source"] = "milb"
    feat.loc[use_milb_bb, "bb_bridge_source"] = "milb"
    feat.loc[use_pio_bb, "bb_bridge_source"] = "pioneer"

    feat["has_college_match"] = feat["has_college_match"] | use_pio_k | use_milb_k | use_pio_bb | use_milb_bb
    return feat.drop(columns=["_name_clean", "_pio_k", "_pio_bb", "_milb_k", "_milb_bb"])


def full_match(pitcher_ids: pd.Series, pitcher_names: pd.Series, bridge_model: dict,
               prospect_proj: pd.DataFrame, pioneer_full: pd.DataFrame,
               milb_full: pd.DataFrame) -> pd.DataFrame:
    """The full three-tier match chain in one call, for any list of
    (pitcher_id, pitcher_name) pairs: narrow bridge-fitting match -> name-
    extended college -> Pioneer/MiLB. Returns one row per pitcher_id with
    fl_k_rate_bridge/fl_bb_rate_bridge (NaN only for a true non-match -- callers
    fill with their own league-average fallback) and has_college_match/
    k_bridge_source/bb_bridge_source."""
    feat = pd.DataFrame({"pitcher_id": pd.Series(pitcher_ids).astype(str).values,
                          "pitcher": pd.Series(pitcher_names).values})
    narrow = bridge.predict_other_league_prior(bridge_model, feat["pitcher_id"])
    feat = feat.merge(narrow, on="pitcher_id", how="left")
    feat["has_college_match"] = feat["has_college_match"].fillna(False)
    feat = extend_match_by_name(feat, prospect_proj)
    feat = extend_match_by_other_leagues(feat, pioneer_full, milb_full)
    return feat


RECENCY_DECAY = 0.5  # per-year-of-staleness weight decay, reused from
# fl_college_bridge.BB_MULTI_YEAR_DECAY -- the one real decay-rate sweep this
# project has actually run (on the within-college multi-year BB% average,
# where decay=0.5 beat both flat TBF-weighting and single-year-only on real
# held-out data). There is no real sample of players who match >1 of
# college/Pioneer/MiLB at once to grid-search a CROSS-league decay on
# directly, so this transplants the validated within-league value rather
# than picking a number with no empirical grounding at all. See
# fl_other_league_bridge_test.py variant C for the held-out check of the
# resulting blend against the fixed college>Pioneer>MiLB priority order.


def _source_candidate(full: pd.DataFrame, source: str) -> pd.DataFrame:
    """One row per _name_clean, this source's own latest_season/tbf/predicted
    rate/validated R2, columns prefixed by `source` so college/pioneer/milb
    can be merged onto the same feat frame without colliding."""
    cols = {
        "name_clean": "_name_clean", "latest_season": f"{source}_latest_season",
        "tbf_latest": f"{source}_tbf_latest", "tbf_multi": f"{source}_tbf_multi",
        "fl_k_rate_bridge": f"{source}_k", "fl_bb_rate_bridge": f"{source}_bb",
        f"{source}_r2_k": f"{source}_r2_k", f"{source}_r2_bb": f"{source}_r2_bb",
    }
    return full.rename(columns=cols)[list(cols.values())].drop_duplicates("_name_clean")


def recency_weighted_match(pitcher_ids: pd.Series, pitcher_names: pd.Series,
                            college_full: pd.DataFrame, pioneer_full: pd.DataFrame,
                            milb_full: pd.DataFrame, current_year: int,
                            decay: float = RECENCY_DECAY) -> pd.DataFrame:
    """Alternative to full_match's fixed college>Pioneer>MiLB priority chain:
    for each pitcher, per metric, picks the SINGLE source with the highest
    weight = (that source's own validated held-out R2) x (the sample size,
    TBF, behind that source's rate estimate) x (decay**years-stale, relative
    to current_year -- the most recent completed FL season) -- rather than
    the fixed-priority version's "college wins whenever it has ANY match,
    no matter how old or how much better a Pioneer/MiLB record is."

    Tested a straight weighted BLEND across all matching sources first (this
    function's original design) -- held out WORSE than the fixed-priority
    chain on the real Experiment-2 test (K% R2_w 0.287 vs 0.294, BB% 0.348
    vs 0.353): diluting an already-good college match with weaker Pioneer/
    MiLB signal costs more than the extra information is worth. Picking the
    single best-weighted source instead beat the fixed-priority chain on K%
    (R2_w 0.295, decay=0.5) while landing close on BB% (0.349 vs 0.353) --
    see DECISION_LOG.md "Cross-league bridge match: weighted blend vs
    single-best-source" for the full comparison table. A source with no
    match for a player gets zero weight, so an unmatched metric naturally
    falls through to whichever source DOES match, or to the caller's own
    league-average fallback if none do.

    Returns the same shape as full_match: one row per pitcher_id with
    fl_k_rate_bridge/fl_bb_rate_bridge, has_college_match (kept under its
    original name for compatibility with every existing caller/column
    consumer, even though it now means "has ANY bridge match"), and
    k_bridge_source/bb_bridge_source (which single source won, for
    display -- "college"/"pioneer"/"milb"/"none")."""
    feat = pd.DataFrame({"pitcher_id": pd.Series(pitcher_ids).astype(str).values,
                          "pitcher": pd.Series(pitcher_names).values})
    feat["_name_clean"] = feat["pitcher"].apply(fl_name_to_college_clean)

    sources = ["college", "pioneer", "milb"]
    fulls = {"college": college_full, "pioneer": pioneer_full, "milb": milb_full}
    for s in sources:
        feat = feat.merge(_source_candidate(fulls[s], s), on="_name_clean", how="left")

    out = feat[["pitcher_id", "pitcher"]].copy()
    for metric, tbf_col in [("k", "tbf_latest"), ("bb", "tbf_multi")]:
        weights = {}
        for s in sources:
            gap = current_year - feat[f"{s}_latest_season"]
            w = feat[f"{s}_r2_{metric}"] * feat[f"{s}_{tbf_col}"] * (decay ** gap)
            weights[s] = w.where(feat[f"{s}_{metric}"].notna(), 0.0).fillna(0.0)
        w_frame = pd.concat({s: weights[s] for s in sources}, axis=1)
        wsum = w_frame.sum(axis=1)
        has_any = wsum > 0
        best_source = w_frame.idxmax(axis=1)

        pred = pd.Series(np.nan, index=feat.index)
        for s in sources:
            mask = has_any & (best_source == s)
            pred.loc[mask] = feat.loc[mask, f"{s}_{metric}"]
        out[f"fl_{metric}_rate_bridge"] = pred
        out[f"{metric}_bridge_source"] = np.where(has_any, best_source, "none")
        # batters faced behind the WINNING source's rate (K%: its latest season; BB%: its
        # multi-year total) -- how much real data stands behind this bridge value
        tbf_sel = pd.Series(0.0, index=feat.index)
        for s_ in sources:
            m_ = has_any & (best_source == s_)
            tbf_sel.loc[m_] = feat.loc[m_, f"{s_}_{tbf_col}"].astype(float)
        out[f"{metric}_bridge_tbf"] = tbf_sel

    out["has_college_match"] = out["fl_k_rate_bridge"].notna() | out["fl_bb_rate_bridge"].notna()
    return out
