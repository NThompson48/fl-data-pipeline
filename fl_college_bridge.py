"""
fl_college_bridge.py -- persists the college->FL raw-rate K%/BB% bridge (previously
only ad hoc `python -c` one-liners, never saved as a script -- see DECISION_LOG.md's
"Conference-season-relative K%/BB%: tested on real data, does NOT beat raw rate"
entry for why raw rate, not conference-relative, is the right input).

Asymmetric multi-year design (tested directly on real data, see DECISION_LOG.md
"Multi-year college input: helps BB%, not K%"):
  - K% uses the pitcher's SINGLE most recent college season (`k_pct`).
  - BB% uses a TBF-weighted AVERAGE across every available college season up to
    and including their most recent one (`bb_pct_multi`).
This was tested by fitting both forms on the n=152 (deduped) FL-rookie bridge
pairs, weighted-R2 and a combined-model significance check, evaluated against
real FL K_rate/BB_rate: single-year clearly wins for K% (R2_w 0.120 vs 0.080,
multi-year term p=0.93 in the combined model -- adds nothing), multi-year
clearly wins for BB% (R2_w 0.240 vs 0.214, multi-year term p=0.005 in the
combined model, single-year term drops to p=0.089 once multi-year is included).

Exposes the "other league prior" used by the half-season backtest harness: for a
pitcher with a matched college pitching season, predict their FL K_rate/BB_rate
from college stats via the already-validated linear fit; for an unmatched
pitcher, return NaN (the harness blends this with the naive FL-history prior by
reliability, so an unmatched pitcher's "other league prior" model collapses
exactly to the naive-only baseline -- see plan Decision 2).

This was fit ONLY on the college-year-Y -> FL-rookie-year-Y+1 transition (n=152,
deduped from 160 -- a handful of `fl_college_bridge_clean.parquet` rows carried
the same pre-existing pitcher_id/name duplication bug fixed elsewhere in
`fl_pitcher_features.build_period_kbb`; this older file predates that fix and
wasn't rebuilt from scratch here, just deduped defensively at load time).
Applying its predicted rate to a RETURNING FL veteran is a real scope extension
the plan flags explicitly -- handled by letting reliability weighting naturally
downweight it for pitchers who already have substantial FL history, not by
pretending the bridge was validated for that case.

Usage:
    import fl_college_bridge as bridge
    model = bridge.fit_bridge()
    preds = bridge.predict_other_league_prior(model, pitcher_ids)
"""
from __future__ import annotations

import glob
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BRIDGE_PATH = "output/fl_college_bridge_clean.parquet"
COLLEGE_GLOB = "output/fg_college_pit_20*.parquet"
FL_TRANSITION_YEARS = [(2025, "2025-05-08", "2025-08-31"), (2026, "2026-05-07", "2026-09-06")]
# matches fl_halfseason_backtest.START_2025/END_2025/START_2026/END_2026 --
# duplicated as plain strings here rather than importing that module, to
# avoid a circular import (fl_halfseason_backtest imports fl_college_bridge).


def _clean_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fl_name_clean(pitcher_name: str) -> str:
    """FL names are 'Last, First' -> reformat to 'First Last' before
    normalizing, same convention as fl_pioneer_bridge._fl_name_clean /
    fl_bridge_matching.fl_name_to_college_clean -- duplicated rather than
    imported to avoid a circular import (fl_bridge_matching already imports
    this module)."""
    s = str(pitcher_name)
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return _clean_name(s)


def load_all_college_years(glob_pat: str = COLLEGE_GLOB) -> pd.DataFrame:
    """Every pulled college pitching season (2023-2026, as more years get added
    this picks them up automatically), name-normalized, one row per
    (upid, season) -- keyed on FanGraphs' own stable player id, NOT on
    name_clean alone.

    Grouping by (name_clean, season) alone (the original approach) silently
    BLENDS different real people who share a cleaned name into one fictional
    composite -- confirmed directly: 137/11,788 name_clean values span >1
    distinct upid (e.g. two different real "Alex Hernandez"es, SLU vs GT,
    same 2025 season), and the old groupby averaged their k_pct/bb_pct
    together as if they were one person. Keeping upid as the grouping key
    avoids that; name_clean is retained as a plain column so callers can
    still look a player up by name, but _latest_and_multi now resolves which
    upid is the right real person BEFORE averaging across seasons, instead of
    averaging across whichever upids happen to share that name."""
    frames = []
    for path in sorted(glob.glob(glob_pat)):
        d = pd.read_parquet(path)[["name", "team", "season", "tbf", "k_pct", "bb_pct", "upid"]]
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["name_clean"] = out["name"].apply(_clean_name)
    agg = out.groupby(["upid", "season"]).agg(
        tbf=("tbf", "sum"), k_pct=("k_pct", "mean"), bb_pct=("bb_pct", "mean"),
        name=("name", "first"), team=("team", "first"), name_clean=("name_clean", "first"),
    ).reset_index()
    return agg


BB_MULTI_YEAR_DECAY = 0.5  # per-year-of-lag weight decay on top of the existing
# TBF weight for bb_pct_multi -- tested directly (fl_bridge_recency_test.py):
# flat TBF-only averaging (decay=1.0) treats a 3-year-old season the same as
# last year's, TBF for TBF. A decay sweep found a real, non-trivial peak
# around decay=0.5 (R2=0.2604 vs flat's 0.2461, corr 0.510 vs 0.496) on the
# n=152 real transition-pair sample -- moderate recency-weighting, not flat
# and not single-year-only (decay=0.0 was WORSE than flat, R2=0.2138). See
# DECISION_LOG.md "Recency-weighted multi-year BB% bridge input". This is
# COLLEGE-SPECIFIC: the same sweep found flat pooling already optimal for
# MiLB and no clear signal either way for Pioneer (too few multi-year
# players) -- NOT applied to fl_pioneer_bridge.py/fl_milb_bridge.py.


def _latest_and_multi(college_years: pd.DataFrame, name_clean: str, as_of_season: int | None = None,
                       bb_decay: float = BB_MULTI_YEAR_DECAY):
    """For one player: k_pct from their single most recent season (<=
    as_of_season if given, else their overall most recent), bb_pct_multi as a
    TBF x recency-decay-weighted average across every season up to and
    including that one (weight = tbf * bb_decay**lag, lag = years before the
    player's own most recent matched season).

    When name_clean spans >1 distinct upid (a genuine different-person name
    collision -- see load_all_college_years), resolve to the SINGLE upid
    whose own most recent in-range season is latest (ties broken by that
    upid's total in-range TBF), then restrict every computation below to
    ONLY that person's rows -- never blend two different people's seasons
    together just because they share a cleaned name.

    Also returns tbf_latest (TBF behind k_pct) and tbf_multi (summed TBF
    behind bb_pct_multi) -- the sample-size side of cross-league match
    weighting (see fl_bridge_matching.py)."""
    hist = college_years[college_years["name_clean"] == name_clean]
    if as_of_season is not None:
        hist = hist[hist["season"] <= as_of_season]
    if hist.empty:
        return None
    if hist["upid"].nunique() > 1:
        per_upid_latest = hist.sort_values("season").groupby("upid").agg(
            latest_season=("season", "last"), tbf_sum=("tbf", "sum")).reset_index()
        best = per_upid_latest.sort_values(["latest_season", "tbf_sum"], ascending=False).iloc[0]
        hist = hist[hist["upid"] == best["upid"]]
    latest = hist.sort_values("season").iloc[-1]
    lag = latest["season"] - hist["season"]
    bb_w = hist["tbf"] * (bb_decay ** lag)
    bb_pct_multi = np.average(hist["bb_pct"], weights=bb_w)
    return {"k_pct": latest["k_pct"], "bb_pct_multi": bb_pct_multi,
            "latest_season": int(latest["season"]), "n_years": len(hist),
            "tbf_latest": float(latest["tbf"]), "tbf_multi": float(hist["tbf"].sum())}


def build_fl_transition_pairs(raw: pd.DataFrame, college_years: pd.DataFrame) -> pd.DataFrame:
    """Real FL-year-Y outcomes, name-matched against the pitcher's college
    history at or before Y-1 -- builds `fl_college_bridge_clean.parquet`
    FRESH from live data, the college-side analog of
    fl_pioneer_bridge.build_fl_transition_pairs/fl_milb_bridge's equivalent.
    `fl_college_bridge_clean.parquet` previously had no owning script at all
    (built once, outside this codebase, with no way to pick up a future FL
    season's debuts) -- this closes that gap so the college bridge self-
    refreshes the same way Pioneer/MiLB already do, instead of silently going
    stale. See DECISION_LOG.md "College bridge: no rebuild script existed".

    Restricted to genuine FL DEBUTS (no FL PA in any earlier year this script
    checks) -- same guard as the Pioneer/MiLB versions, for the same reason
    (a returning veteran's year-2 performance reflects their OWN FL track
    record, not a now-stale college season).

    Also dedupes same-person multi-pitcher_id cases (a pitcher reissued a new
    id mid-career -- confirmed directly, caught here: "Brouwer, Adam"/
    "Jones, Logan"/"Martzolf, Max" and 11 others each had 2-3 distinct ids
    sharing one name_clean within the same fl_year, fragmenting one real
    person's PA across multiple noisy partial-sample rows and corrupting the
    fit -- K% corr dropped 0.365->0.251, BB% 0.510->0.279 before this fix).
    Self-contained version of the same principle as
    fl_production_projection.build_canonical_id_map (always keep the
    fullest-history id), not imported from there to avoid a circular import
    (that module already imports this one)."""
    import fl_pitcher_features as fp

    seen_pitcher_ids: set = set()
    rows = []
    for fl_year, start, end in FL_TRANSITION_YEARS:
        kbb = fp.build_period_kbb(fp.filter_period(raw, start, end), min_pa=1)
        kbb = kbb[~kbb["pitcher_id"].isin(seen_pitcher_ids)]
        seen_pitcher_ids |= set(kbb["pitcher_id"])

        names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
        kbb = kbb.merge(names, on="pitcher_id", how="left")
        kbb["name_clean"] = kbb["pitcher"].apply(_fl_name_clean)
        kbb = kbb.sort_values("PA", ascending=False).drop_duplicates("name_clean", keep="first")
        for _, r in kbb.iterrows():
            feats = _latest_and_multi(college_years, r["name_clean"], as_of_season=fl_year - 1)
            if feats is None:
                continue
            rows.append({"pitcher_id": r["pitcher_id"], "pitcher": r["pitcher"],
                         "name_clean": r["name_clean"], "fl_year": fl_year,
                         "college_year": fl_year - 1, "pa": r["PA"],
                         "k_rate": r["K_rate"], "bb_rate": r["BB_rate"]})
    return pd.DataFrame(rows)


def rebuild_bridge_file(raw: pd.DataFrame, out_path: str = BRIDGE_PATH) -> pd.DataFrame:
    """Rebuilds `fl_college_bridge_clean.parquet` from scratch against
    whatever FL seasons are currently in `raw` -- run this (or just
    `python fl_college_bridge.py`) whenever a new FL season's debuts should
    be folded in. `fit_bridge()`'s own interface/signature is UNCHANGED (it
    still just reads this file), so none of this module's other callers
    need to change."""
    college_years = load_all_college_years()
    d = build_fl_transition_pairs(raw, college_years)
    print(f"[fl_college_bridge] rebuilt {out_path}: {len(d)} real college-year-Y -> FL-year-Y+1 "
          f"transition pairs (fl_year breakdown: {d['fl_year'].value_counts().to_dict() if len(d) else {}})")
    d.to_parquet(out_path, index=False)
    return d


def fit_bridge(path: str = BRIDGE_PATH) -> dict:
    """Refit the bridge: K% on single-year, BB% on the multi-year weighted
    average (see module docstring for why). Reports the same K% correlation
    already recorded in DECISION_LOG.md as a regression check."""
    d = pd.read_parquet(path).drop_duplicates("pitcher_id").copy()
    college_years = load_all_college_years()

    rows = []
    for _, r in d.iterrows():
        feats = _latest_and_multi(college_years, r["name_clean"], as_of_season=r["college_year"])
        if feats is None:
            continue
        rows.append({"pitcher_id": r["pitcher_id"], **feats})
    feat_df = pd.DataFrame(rows)
    d = d.merge(feat_df, on="pitcher_id", how="inner", suffixes=("", "_hist"))

    k_model = smf.ols("k_rate ~ k_pct", data=d).fit()
    bb_model = smf.ols("bb_rate ~ bb_pct_multi", data=d).fit()
    k_corr = d["k_pct"].corr(d["k_rate"])
    bb_corr = d["bb_pct_multi"].corr(d["bb_rate"])
    print(f"[fl_college_bridge] refit on n={len(d)}: K% (single-year) corr={k_corr:.3f} "
          f"(DECISION_LOG: 0.365 on the older n=160), BB% (multi-year avg) corr={bb_corr:.3f}")

    k_r2 = max(k_model.rsquared, 0.0)
    bb_r2 = max(bb_model.rsquared, 0.0)

    college_table = d[["pitcher_id", "k_pct", "bb_pct_multi"]].drop_duplicates("pitcher_id")
    return {
        "k_model": k_model, "bb_model": bb_model,
        "k_r2": k_r2, "bb_r2": bb_r2,
        "college_table": college_table,
        "college_years": college_years,
    }


def predict_other_league_prior(model: dict, pitcher_ids: pd.Series) -> pd.DataFrame:
    """For each pitcher_id, the bridge-predicted FL K_rate/BB_rate (NaN if no
    college match) -- for FL roster pitchers who happen to appear in the
    bridge's own fitting sample (true recent rookies). See bridge_pseudo_pa for
    the blending weight."""
    tbl = model["college_table"].copy()
    tbl["fl_k_rate_bridge"] = model["k_model"].predict(tbl).values
    tbl["fl_bb_rate_bridge"] = model["bb_model"].predict(tbl).values

    out = pd.DataFrame({"pitcher_id": pitcher_ids.astype(str).unique()})
    out = out.merge(tbl[["pitcher_id", "fl_k_rate_bridge", "fl_bb_rate_bridge"]],
                     on="pitcher_id", how="left")
    out["has_college_match"] = out["fl_k_rate_bridge"].notna()
    return out


def predict_from_name(model: dict, name: str, as_of_season: int | None = None) -> dict | None:
    """For ANY college pitcher by name (prospect lookup) -- single-year K%,
    multi-year-averaged BB%, through the same fitted bridge."""
    feats = _latest_and_multi(model["college_years"], _clean_name(name), as_of_season=as_of_season)
    if feats is None:
        return None
    row = pd.DataFrame([feats])
    fl_k = float(model["k_model"].predict(row).iloc[0])
    fl_bb = float(model["bb_model"].predict(row).iloc[0])
    return {"fl_k_rate_bridge": fl_k, "fl_bb_rate_bridge": fl_bb,
            "k_pct": feats["k_pct"], "bb_pct_multi": feats["bb_pct_multi"],
            "latest_season": feats["latest_season"], "n_years": feats["n_years"]}


def bridge_pseudo_pa(model: dict, n0_k: float, n0_bb: float) -> tuple[float, float]:
    """pseudo_PA_k/bb: the "equivalent sample size" of the bridge prediction, on the
    same n/(n+n0) reliability scale as the FL-side naive prior, so the two sources
    can be combined with ordinary reliability-weighted averaging (plan
    Decision 2) -- not a privileged override."""
    k_r2 = min(max(model["k_r2"], 1e-6), 0.95)
    bb_r2 = min(max(model["bb_r2"], 1e-6), 0.95)
    pseudo_pa_k = n0_k * k_r2 / (1 - k_r2)
    pseudo_pa_bb = n0_bb * bb_r2 / (1 - bb_r2)
    return pseudo_pa_k, pseudo_pa_bb


if __name__ == "__main__":
    import fl_pitcher_features as fp

    print("Rebuilding fl_college_bridge_clean.parquet from live FL data...")
    raw = fp.load_fl_pitches_cached()
    rebuild_bridge_file(raw)

    m = fit_bridge()
    pk, pbb = bridge_pseudo_pa(m, n0_k=70.0, n0_bb=150.0)
    print(f"k_r2={m['k_r2']:.3f} bb_r2={m['bb_r2']:.3f}  pseudo_PA (at illustrative n0=70/150): "
          f"K={pk:.1f} BB={pbb:.1f}")
