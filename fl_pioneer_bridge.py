"""
fl_pioneer_bridge.py -- Pioneer League -> FL raw-rate K%/BB% bridge, same
asymmetric design as fl_college_bridge.py (K% on the pitcher's single most
recent season, BB% on a TBF-weighted multi-year average -- see that module's
docstring for why, validated on real college data in DECISION_LOG.md
"Multi-year college input: helps BB%, not K%").

Unlike the college bridge, there was no pre-existing "clean" transition-pairs
file to load -- this script BUILDS one from scratch: for each FL season with
real outcomes (2025, 2026), it name-matches FL pitchers against their most
recent Pioneer League season at or before (FL year - 1) -- the same
Pioneer-year-Y -> FL-year-Y+1 transition logic as the college bridge's
college-year -> FL-rookie-year pairing, scored against real FL K_rate/BB_rate
(never assumed).

Usage:
    import fl_pioneer_bridge as pbridge
    raw = fp.load_fl_pitches_cached()
    model = pbridge.fit_bridge(raw)
    preds = pbridge.predict_other_league_prior(model, pitcher_ids)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt

PIONEER_PATH = "output/fl_pioneer_pitching_2021_2025.parquet"
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
    normalizing, same convention as
    fl_production_projection._fl_name_to_college_clean."""
    s = str(pitcher_name)
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return _clean_name(s)


PIONEER_2026_PATH = "output/fl_pioneer_pitching_2026.parquet"   # built by fl_pioneer_2026.py (Baseball Reference, 9 of 12 teams, BF estimated)


def load_all_pioneer_years(path: str = PIONEER_PATH) -> pd.DataFrame:
    import os
    df = pd.read_parquet(path)
    if path == PIONEER_PATH and os.path.exists(PIONEER_2026_PATH):      # 2026 season, when available
        df = pd.concat([df, pd.read_parquet(PIONEER_2026_PATH)], ignore_index=True)
    return df.rename(columns={"bf": "tbf"})[["name_clean", "year", "tbf", "k_pct", "bb_pct", "player"]]


def _latest_and_multi(pioneer_years: pd.DataFrame, name_clean: str, as_of_season: int):
    """Single most recent Pioneer season (<= as_of_season) for k_pct; TBF-weighted
    average across every Pioneer season up to and including that one for
    bb_pct_multi -- mirrors fl_college_bridge._latest_and_multi exactly,
    including the tbf_latest/tbf_multi outputs used for cross-league match
    weighting (see fl_bridge_matching.py). NOTE: unlike college/MiLB, the
    StatsCrew Pioneer pull has no stable per-player id to resolve a
    different-real-person name collision against -- that residual risk is
    accepted here (small-n source, ~1% rate measured elsewhere), not fixed."""
    hist = pioneer_years[(pioneer_years["name_clean"] == name_clean)
                          & (pioneer_years["year"] <= as_of_season)]
    if hist.empty:
        return None
    latest = hist.sort_values("year").iloc[-1]
    bb_pct_multi = np.average(hist["bb_pct"], weights=hist["tbf"])
    return {"k_pct": latest["k_pct"], "bb_pct_multi": bb_pct_multi,
            "latest_season": int(latest["year"]), "n_years": len(hist),
            "tbf_latest": float(latest["tbf"]), "tbf_multi": float(hist["tbf"].sum())}


def build_fl_transition_pairs(raw: pd.DataFrame, pioneer_years: pd.DataFrame) -> pd.DataFrame:
    """Real FL-year-Y outcomes, name-matched against the pitcher's Pioneer
    history at or before Y-1 -- the Pioneer-side analog of
    fl_college_bridge_clean.parquet (built fresh here since no such file
    pre-exists for Pioneer).

    Restricted to genuine FL DEBUTS (no FL PA in any earlier year this script
    checks): without this guard, a pitcher who debuted in FL 2025 and kept
    pitching in 2026 gets counted AGAIN as a fl_year=2026 "transition" against
    a now-stale Pioneer season, even though his real 2026 FL performance is
    driven by his own 2025 FL track record, not Pioneer ball two years prior.
    Confirmed directly: including those returning-veteran duplicates collapses
    the fit to near-zero (K% R2=0.001); restricting to true debuts is what
    makes this a fair test of the bridge itself."""
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
            feats = _latest_and_multi(pioneer_years, r["name_clean"], as_of_season=fl_year - 1)
            if feats is None:
                continue
            rows.append({"pitcher_id": r["pitcher_id"], "pitcher": r["pitcher"], "fl_year": fl_year,
                         "pa": r["PA"], "k_rate": r["K_rate"], "bb_rate": r["BB_rate"], **feats})
    return pd.DataFrame(rows)


def fit_bridge(raw: pd.DataFrame) -> dict:
    pioneer_years = load_all_pioneer_years()
    d = build_fl_transition_pairs(raw, pioneer_years)
    print(f"[fl_pioneer_bridge] {len(d)} real Pioneer-year-Y -> FL-year-Y+1 transition pairs found "
          f"(fl_year breakdown: {d['fl_year'].value_counts().to_dict() if len(d) else {}})")

    if len(d) < 5:
        print("[fl_pioneer_bridge] too few pairs to fit meaningfully -- returning degenerate model")
        return {"k_model": None, "bb_model": None, "k_r2": 0.0, "bb_r2": 0.0,
                "pioneer_table": pd.DataFrame(columns=["pitcher_id", "k_pct", "bb_pct_multi"]),
                "pioneer_years": pioneer_years, "transition_pairs": d}

    # WLS weighted by FL debut-season PA, not plain OLS (unlike fl_college_bridge.py,
    # which fit unweighted and still found signal). Checked directly: many Pioneer
    # transition pairs are tiny-PA FL cameos (as low as 1 PA) whose k_rate/bb_rate
    # are almost pure noise; plain OLS let that noise dominate (K% R2=0.008, BB%
    # R2=0.019) while a PA>=50 subset alone showed real correlation (K% 0.37, BB%
    # 0.39). Rate-estimate variance scales ~1/PA, so weighting by PA is the
    # standard fix (the same justification mlb_marcel's reliability weighting
    # already uses elsewhere in this project) rather than an arbitrary filter.
    k_model = smf.wls("k_rate ~ k_pct", data=d, weights=d["pa"]).fit()
    bb_model = smf.wls("bb_rate ~ bb_pct_multi", data=d, weights=d["pa"]).fit()
    k_corr = d["k_pct"].corr(d["k_rate"])
    bb_corr = d["bb_pct_multi"].corr(d["bb_rate"])
    print(f"[fl_pioneer_bridge] unweighted corr -- K%={k_corr:.3f}, BB%={bb_corr:.3f} "
          f"(swamped by tiny-PA debut noise)")
    print(f"[fl_pioneer_bridge] PA-weighted fit -- K% R2={max(k_model.rsquared, 0):.3f}, "
          f"BB% R2={max(bb_model.rsquared, 0):.3f}")

    k_r2 = max(k_model.rsquared, 0.0)
    bb_r2 = max(bb_model.rsquared, 0.0)
    pioneer_table = d[["pitcher_id", "k_pct", "bb_pct_multi"]].drop_duplicates("pitcher_id")
    return {"k_model": k_model, "bb_model": bb_model, "k_r2": k_r2, "bb_r2": bb_r2,
            "pioneer_table": pioneer_table, "pioneer_years": pioneer_years, "transition_pairs": d}


def predict_other_league_prior(model: dict, pitcher_ids: pd.Series) -> pd.DataFrame:
    tbl = model["pioneer_table"].copy()
    out = pd.DataFrame({"pitcher_id": pitcher_ids.astype(str).unique()})
    if model["k_model"] is None or tbl.empty:
        out["fl_k_rate_bridge"] = np.nan
        out["fl_bb_rate_bridge"] = np.nan
        out["has_pioneer_match"] = False
        return out
    tbl["fl_k_rate_bridge"] = model["k_model"].predict(tbl).values
    tbl["fl_bb_rate_bridge"] = model["bb_model"].predict(tbl).values
    out = out.merge(tbl[["pitcher_id", "fl_k_rate_bridge", "fl_bb_rate_bridge"]], on="pitcher_id", how="left")
    out["has_pioneer_match"] = out["fl_k_rate_bridge"].notna()
    return out


def build_full_universe_predictions(model: dict) -> pd.DataFrame:
    """One row per distinct Pioneer League pitcher ever pulled (2023-2025),
    using their own single most recent season for k_pct and a TBF-weighted
    multi-year average for bb_pct_multi, with the fitted bridge's predicted FL
    K%/BB% -- the Pioneer-side analog of
    fl_production_projection.build_college_prospect_projections, used to
    extend match coverage to ANY roster player with a Pioneer record, not
    just the narrow transition-fitting sample."""
    py = model["pioneer_years"]
    latest = (py.sort_values("year").groupby("name_clean").tail(1)
              [["name_clean", "player", "year", "k_pct"]]
              .rename(columns={"year": "latest_season"}))
    multi = py.groupby("name_clean").apply(lambda g: pd.Series({
        "bb_pct_multi": np.average(g["bb_pct"], weights=g["tbf"]),
        "n_years": len(g), "tbf_latest": g.loc[g["year"].idxmax(), "tbf"], "tbf_multi": g["tbf"].sum(),
    }), include_groups=False).reset_index()
    out = latest.merge(multi, on="name_clean")
    if model["k_model"] is not None:
        out["fl_k_rate_bridge"] = model["k_model"].predict(out).values
        out["fl_bb_rate_bridge"] = model["bb_model"].predict(out).values
    else:
        out["fl_k_rate_bridge"] = np.nan
        out["fl_bb_rate_bridge"] = np.nan
    out["pioneer_r2_k"] = model["k_r2"]
    out["pioneer_r2_bb"] = model["bb_r2"]
    return out


def predict_from_name(model: dict, name: str, as_of_season: int) -> dict | None:
    feats = _latest_and_multi(model["pioneer_years"], _clean_name(name), as_of_season=as_of_season)
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
