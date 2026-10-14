"""
fl_milb_bridge.py -- affiliated MiLB (AAA/AA/A+/A/ROK) -> FL raw-rate K%/BB%
bridge, same structure as fl_pioneer_bridge.py (see that module's docstring
for the genuine-debuts-only transition-pair construction and the PA-weighted
fit rationale -- both reused here unchanged).

Source data: output/milb_pitching_2018_2026.parquet, pulled via the official
MLB Stats API by the pre-existing pull_milb_pitching.py (2026 added
specifically to let build_full_universe_predictions's "most recent season on
record" lookup catch a SAME-YEAR MiLB->FL transition for a player who arrives
in FL mid-season -- previously only year-1-or-earlier MiLB records could
match at all, since 2026 wasn't pulled; see DECISION_LOG.md "mixed-league
ROS/same-year"). A pitcher can appear at multiple levels/teams within one
season (a mid-season promotion/demotion) -- aggregated by summing
strikeOuts/baseOnBalls/battersFaced across levels within a season before
computing that season's K%/BB%, the same mid-season-move handling
fl_statscrew_pull.py uses for in-season trades.

Usage:
    import fl_milb_bridge as mbridge
    raw = fp.load_fl_pitches_cached()
    model = mbridge.fit_bridge(raw)
    preds = mbridge.predict_other_league_prior(model, pitcher_ids)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt

MILB_PATH = "output/milb_pitching_2018_2026.parquet"
FL_TRANSITION_YEARS = [
    (2025, bt.START_2025, bt.END_2025),
    (2026, bt.START_2026, bt.END_2026),
]


def _clean_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fl_name_clean(pitcher_name: str) -> str:
    """FL names are 'Last, First' -> reformat to 'First Last' before
    normalizing, same convention used throughout this project."""
    s = str(pitcher_name)
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return _clean_name(s)


def load_all_milb_years(path: str = MILB_PATH) -> pd.DataFrame:
    """One row per (pitcher MLBAM id, season), summed across any mid-season
    level/team moves.

    Keyed on the real MLBAM id ('pitcher'), NOT name_clean alone -- grouping
    by name_clean alone (the original approach) silently blends different
    real people who share a name: confirmed directly, 368/13,943 lowercase
    names in this file span >1 distinct MLBAM id, the same class of bug
    found and fixed in fl_college_bridge.load_all_college_years. name_clean
    is kept as a plain column for name-based lookup; _latest_and_multi
    resolves which id is the right real person before averaging across
    seasons."""
    df = pd.read_parquet(path)
    df["name_clean"] = df["player_name"].apply(_clean_name)
    agg = df.groupby(["pitcher", "season"]).agg(
        tbf=("battersFaced", "sum"), so=("strikeOuts", "sum"), bb=("baseOnBalls", "sum"),
        player=("player_name", "first"), name_clean=("name_clean", "first"),
    ).reset_index().rename(columns={"season": "year"})
    agg["k_pct"] = agg["so"] / agg["tbf"]
    agg["bb_pct"] = agg["bb"] / agg["tbf"]
    return agg[agg["tbf"] > 0]


def _latest_and_multi(milb_years: pd.DataFrame, name_clean: str, as_of_season: int):
    """Single most recent MiLB season (<= as_of_season) for k_pct; TBF-weighted
    average across every MiLB season up to and including that one for
    bb_pct_multi -- mirrors fl_college_bridge._latest_and_multi exactly,
    including the same upid-style collision resolution (here keyed on the
    MLBAM id, 'pitcher') and the same tbf_latest/tbf_multi outputs used for
    cross-league match weighting (see fl_bridge_matching.py)."""
    hist = milb_years[(milb_years["name_clean"] == name_clean) & (milb_years["year"] <= as_of_season)]
    if hist.empty:
        return None
    if hist["pitcher"].nunique() > 1:
        per_id_latest = hist.sort_values("year").groupby("pitcher").agg(
            latest_year=("year", "last"), tbf_sum=("tbf", "sum")).reset_index()
        best = per_id_latest.sort_values(["latest_year", "tbf_sum"], ascending=False).iloc[0]
        hist = hist[hist["pitcher"] == best["pitcher"]]
    latest = hist.sort_values("year").iloc[-1]
    bb_pct_multi = np.average(hist["bb_pct"], weights=hist["tbf"])
    return {"k_pct": latest["k_pct"], "bb_pct_multi": bb_pct_multi,
            "latest_season": int(latest["year"]), "n_years": len(hist),
            "tbf_latest": float(latest["tbf"]), "tbf_multi": float(hist["tbf"].sum())}


def build_fl_transition_pairs(raw: pd.DataFrame, milb_years: pd.DataFrame) -> pd.DataFrame:
    """Real FL-year-Y outcomes, name-matched against the pitcher's MiLB
    history at or before Y-1. Restricted to genuine FL DEBUTS (no FL PA in an
    earlier year this script checks) for the same reason as
    fl_pioneer_bridge.build_fl_transition_pairs -- without it, a returning FL
    veteran gets counted again against a stale MiLB season that has nothing
    to do with his actual current performance."""
    seen_pitcher_ids: set = set()
    rows = []
    for fl_year, start, end in FL_TRANSITION_YEARS:
        kbb = fp.build_period_kbb(fp.filter_period(raw, start, end), min_pa=1)
        kbb = kbb[~kbb["pitcher_id"].isin(seen_pitcher_ids)]
        seen_pitcher_ids |= set(kbb["pitcher_id"])

        names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
        kbb = kbb.merge(names, on="pitcher_id", how="left")
        kbb["name_clean"] = kbb["pitcher"].apply(_fl_name_clean)
        for _, r in kbb.iterrows():
            feats = _latest_and_multi(milb_years, r["name_clean"], as_of_season=fl_year - 1)
            if feats is None:
                continue
            rows.append({"pitcher_id": r["pitcher_id"], "pitcher": r["pitcher"], "fl_year": fl_year,
                         "pa": r["PA"], "k_rate": r["K_rate"], "bb_rate": r["BB_rate"], **feats})
    return pd.DataFrame(rows)


def fit_bridge(raw: pd.DataFrame) -> dict:
    milb_years = load_all_milb_years()
    d = build_fl_transition_pairs(raw, milb_years)
    print(f"[fl_milb_bridge] {len(d)} real MiLB-year-Y -> FL-year-Y+1 transition pairs found "
          f"(fl_year breakdown: {d['fl_year'].value_counts().to_dict() if len(d) else {}})")

    if len(d) < 5:
        print("[fl_milb_bridge] too few pairs to fit meaningfully -- returning degenerate model")
        return {"k_model": None, "bb_model": None, "k_r2": 0.0, "bb_r2": 0.0,
                "milb_table": pd.DataFrame(columns=["pitcher_id", "k_pct", "bb_pct_multi"]),
                "milb_years": milb_years, "transition_pairs": d}

    # PA-weighted (WLS), not plain OLS -- same rationale as fl_pioneer_bridge.py:
    # rate-estimate noise scales ~1/PA, and many FL-debut rows here are small-PA
    # cameos that would otherwise dominate an unweighted fit.
    k_model = smf.wls("k_rate ~ k_pct", data=d, weights=d["pa"]).fit()
    bb_model = smf.wls("bb_rate ~ bb_pct_multi", data=d, weights=d["pa"]).fit()
    k_corr = d["k_pct"].corr(d["k_rate"])
    bb_corr = d["bb_pct_multi"].corr(d["bb_rate"])
    print(f"[fl_milb_bridge] unweighted corr -- K%={k_corr:.3f}, BB%={bb_corr:.3f}")
    print(f"[fl_milb_bridge] PA-weighted fit -- K% R2={max(k_model.rsquared, 0):.3f}, "
          f"BB% R2={max(bb_model.rsquared, 0):.3f}")

    k_r2 = max(k_model.rsquared, 0.0)
    bb_r2 = max(bb_model.rsquared, 0.0)
    milb_table = d[["pitcher_id", "k_pct", "bb_pct_multi"]].drop_duplicates("pitcher_id")
    return {"k_model": k_model, "bb_model": bb_model, "k_r2": k_r2, "bb_r2": bb_r2,
            "milb_table": milb_table, "milb_years": milb_years, "transition_pairs": d}


def predict_other_league_prior(model: dict, pitcher_ids: pd.Series) -> pd.DataFrame:
    tbl = model["milb_table"].copy()
    out = pd.DataFrame({"pitcher_id": pitcher_ids.astype(str).unique()})
    if model["k_model"] is None or tbl.empty:
        out["fl_k_rate_bridge"] = np.nan
        out["fl_bb_rate_bridge"] = np.nan
        out["has_milb_match"] = False
        return out
    tbl["fl_k_rate_bridge"] = model["k_model"].predict(tbl).values
    tbl["fl_bb_rate_bridge"] = model["bb_model"].predict(tbl).values
    out = out.merge(tbl[["pitcher_id", "fl_k_rate_bridge", "fl_bb_rate_bridge"]], on="pitcher_id", how="left")
    out["has_milb_match"] = out["fl_k_rate_bridge"].notna()
    return out


def build_full_universe_predictions(model: dict) -> pd.DataFrame:
    """One row per distinct name_clean ever pulled (2018-2026), using that
    player's own single most recent season for k_pct and a TBF-weighted
    multi-year average for bb_pct_multi, with the fitted bridge's predicted
    FL K%/BB% -- the MiLB-side analog of
    fl_production_projection.build_college_prospect_projections, used to
    extend match coverage to ANY roster player with an MiLB record, not just
    the narrow transition-fitting sample.

    Resolves the same name_clean-spans->1-real-person risk _latest_and_multi
    guards against: groups by the real MLBAM id ('pitcher') first, then -- if
    a name_clean still maps to >1 distinct id -- keeps only the one whose own
    history is most recent (ties by total TBF), rather than blending
    different real people's seasons into one fictional composite."""
    my = model["milb_years"]
    per_id_latest = (my.sort_values("year").groupby("pitcher").tail(1)
                      [["pitcher", "name_clean", "player", "year", "k_pct"]]
                      .rename(columns={"year": "latest_season"}))
    per_id_multi = my.groupby("pitcher").apply(lambda g: pd.Series({
        "bb_pct_multi": np.average(g["bb_pct"], weights=g["tbf"]),
        "n_years": len(g), "tbf_latest": g.loc[g["year"].idxmax(), "tbf"], "tbf_multi": g["tbf"].sum(),
    }), include_groups=False).reset_index()
    per_id = per_id_latest.merge(per_id_multi, on="pitcher")

    out = (per_id.sort_values(["latest_season", "tbf_multi"])
           .groupby("name_clean").tail(1).reset_index(drop=True))
    if model["k_model"] is not None:
        out["fl_k_rate_bridge"] = model["k_model"].predict(out).values
        out["fl_bb_rate_bridge"] = model["bb_model"].predict(out).values
    else:
        out["fl_k_rate_bridge"] = np.nan
        out["fl_bb_rate_bridge"] = np.nan
    out["milb_r2_k"] = model["k_r2"]
    out["milb_r2_bb"] = model["bb_r2"]
    return out


def predict_from_name(model: dict, name: str, as_of_season: int) -> dict | None:
    feats = _latest_and_multi(model["milb_years"], _clean_name(name), as_of_season=as_of_season)
    if feats is None or model["k_model"] is None:
        return None
    row = pd.DataFrame([feats])
    fl_k = float(model["k_model"].predict(row).iloc[0])
    fl_bb = float(model["bb_model"].predict(row).iloc[0])
    return {"fl_k_rate_bridge": fl_k, "fl_bb_rate_bridge": fl_bb, **feats}


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()
    m = fit_bridge(raw)
    if len(m["transition_pairs"]):
        print("\nTransition pairs (sorted by PA):")
        print(m["transition_pairs"].sort_values("pa", ascending=False)
              [["pitcher", "fl_year", "pa", "k_rate", "bb_rate", "k_pct", "bb_pct_multi", "latest_season", "n_years"]]
              .to_string(index=False))
