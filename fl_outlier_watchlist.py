"""
fl_outlier_watchlist.py -- the frozen RECORD of outlier pitchers / pitch types, to be checked against 2027 outcomes  (user 2026-10-27: "we'll keep an eye on it. Can we keep the list of outlier pitchers/types on record?")
Window = the 2027 projection's feature window: 2025 + 2026 regular season (data as of the 2026-10-26 refresh).  Method = fl_outlier_test / fl_outlier_within: 8 main pitch types, 7 standardised characteristics (velocity, ivb, arm-side horizontal break, spin, extension, release height, release side), each pitch scored
against a distribution fitted on a 10,000-pitch sample from OTHER pitchers (5 pitcher folds, seed 0): robust Mahalanobis (MinCovDet support .9) -> log -> z, and isolation forest (300 trees, 512-pitch subsamples) anomaly -> z, both z-scored against the type's own training distribution (median / MAD).
Pitcher indices: oi_if / oi = mean z over all scored pitches; ot_if / ot = the largest pitch-type mean z among types with >= 30 of his pitches.  Flags: oi_if in the top 5% / 10% (threshold = the quantile among pitchers with >= 150 scored pitches in the window; frozen below); ot_if top 5%; and per pitch type the top 5% of pitcher cell means (>= 30 pitches of that type).
Writes prereg/outlier_watchlist_2027.json (every pitcher, every cell, thresholds, parameters) and prereg/outlier_watchlist_2027.md (the flagged lists).  Watch rule (report only; no production change): after 2027, compare the flagged pitchers' K% with the frozen production projections (prereg/c10_frozen_projections.json).
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_halfseason_backtest as bt
import fl_outlier_within as ow
import fl_pitcher_features as fp
import fl_position_player_flag as ppf

warnings.filterwarnings("ignore")
MIN_PITCHES, MIN_CELL = 150, 30


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); win = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025), fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    d = win[win["pitch_type"].isin(ow.TYPES)].dropna(subset=ow.FEATS + ["pitcher_id"]).reset_index(drop=True); d["pid"] = d["pitcher_id"].astype(str); print(f"{len(d):,} pitches; scoring ...", flush=True)
    d = ow.pitch_scores(d).dropna(subset=["z", "zi"]); names = raw.drop_duplicates("pitcher_id").assign(pid=lambda x: x["pitcher_id"].astype(str)).set_index("pid")["pitcher"]
    proj = pd.read_parquet("output/fl_2027_projections.parquet").assign(pid=lambda x: x["pitcher_id"].astype(str)).drop_duplicates("pid").set_index("pid")
    cell = d.groupby(["pid", "pitch_type"]).agg(n=("zi", "size"), zi=("zi", "mean"), z=("z", "mean")).reset_index(); cell = cell[cell["n"] >= MIN_CELL].copy()
    p = d.groupby("pid").agg(n_pitches=("zi", "size"), oi_if=("zi", "mean"), oi=("z", "mean"))
    for col, nm, tn in (("zi", "ot_if", "ot_if_type"), ("z", "ot", "ot_type")):
        i = cell.loc[cell.groupby("pid")[col].idxmax()].set_index("pid"); p[nm] = i[col]; p[tn] = i["pitch_type"]
    p["pitcher"] = p.index.map(names); p["team_2026"] = p.index.map(proj["team"]); p["stuff_plus"] = p.index.map(proj["stuff_plus"]); p["proj_K_rate_2027"] = p.index.map(proj["proj_K_rate"])
    elig = p[p["n_pitches"] >= MIN_PITCHES]; thr = {"oi_if_top5": float(elig["oi_if"].quantile(.95)), "oi_if_top10": float(elig["oi_if"].quantile(.90)), "ot_if_top5": float(elig["ot_if"].dropna().quantile(.95)), "n_eligible_pitchers": int(len(elig))}
    p["flag_oi_if_top5"] = (p["n_pitches"] >= MIN_PITCHES) & (p["oi_if"] >= thr["oi_if_top5"]); p["flag_oi_if_top10"] = (p["n_pitches"] >= MIN_PITCHES) & (p["oi_if"] >= thr["oi_if_top10"]); p["flag_ot_if_top5"] = (p["n_pitches"] >= MIN_PITCHES) & (p["ot_if"] >= thr["ot_if_top5"])
    cell["pitcher"] = cell["pid"].map(names); cell["team_2026"] = cell["pid"].map(proj["team"]); cell["cell_top5_within_type"] = cell.groupby("pitch_type")["zi"].transform(lambda s: s >= s.quantile(.95))
    meta = {"frozen_on": "2026-10-27", "window": "2025 + 2026 regular season (data as of the 2026-10-26 refresh)", "pitches_scored": int(len(d)), "pitch_types": ow.TYPES, "characteristics": ow.FEATS, "thresholds": thr, "min_pitches_pitcher": MIN_PITCHES, "min_pitches_cell": MIN_CELL,
            "method": "per-type distribution fitted on a 10,000-pitch sample from other pitchers (5 pitcher folds, seed 0): robust Mahalanobis (MinCovDet support .9) and isolation forest (300 trees, 512 subsamples), each z-scored against the type's training distribution; pitcher index = mean z over his pitches; ot = largest type mean z (>= 30 pitches)",
            "evidence_for_watching": "2026-10-27 outlier dummy test: top 5% isolation-forest dummy, K% RMSE better in 3 of 4 pairings (mean -.00005), flagged pitchers' K% residual +0.3 to +1.7 points in all four, coefficient +.003 to +.027; exploratory (one of 7 variants), 5-14 flagged pitchers per pairing.  Not in production.",
            "watch_rule": "report only: after 2027, flagged pitchers' (flag_oi_if_top5, flag_oi_if_top10) mean K% residual vs the frozen production projection (prereg/c10_frozen_projections.json), PA-weighted, level-free, with a pitcher bootstrap"}
    json.dump({"meta": meta, "pitchers": p.reset_index().rename(columns={"index": "pid"}).to_dict("records"), "cells": cell.to_dict("records")}, open("prereg/outlier_watchlist_2027.json", "w"), default=float)
    f = lambda v, k=2: "—" if pd.isna(v) else f"{v:.{k}f}"
    L = ["# Outlier watch list (frozen 2026-10-27)", "", f"Window: {meta['window']}. {meta['pitches_scored']:,} pitches of 8 main types scored against distributions fitted on other pitchers. Method and parameters are in `outlier_watchlist_2027.json` (`meta`) and `fl_outlier_watchlist.py`.", "",
         f"Thresholds on the isolation-forest mean score (pitchers with >= {MIN_PITCHES} scored pitches, n = {thr['n_eligible_pitchers']}): top 5% >= {thr['oi_if_top5']:.2f}, top 10% >= {thr['oi_if_top10']:.2f}; most-unusual-type score top 5% >= {thr['ot_if_top5']:.2f}.", "",
         "Why it is on record: " + meta["evidence_for_watching"], "", "Watch rule: " + meta["watch_rule"] + ".", "", "## Pitchers flagged top 5% (isolation-forest mean score)", "", "| Pitcher | 2026 team | Pitches | Score | Most unusual type (score) | Stuff+ | Projected 2027 K% |", "|---|---|---|---|---|---|---|"]
    for r in p[p["flag_oi_if_top5"]].sort_values("oi_if", ascending=False).itertuples():
        L.append(f"| {r.pitcher} | {r.team_2026} | {r.n_pitches} | {r.oi_if:.2f} | {r.ot_if_type} ({f(r.ot_if)}) | {f(r.stuff_plus, 0)} | {f(100 * r.proj_K_rate_2027, 1)}% |")
    L += ["", "## Additional pitchers flagged top 10% (not top 5%)", "", "| Pitcher | 2026 team | Pitches | Score | Most unusual type (score) | Stuff+ | Projected 2027 K% |", "|---|---|---|---|---|---|---|"]
    for r in p[p["flag_oi_if_top10"] & ~p["flag_oi_if_top5"]].sort_values("oi_if", ascending=False).itertuples():
        L.append(f"| {r.pitcher} | {r.team_2026} | {r.n_pitches} | {r.oi_if:.2f} | {r.ot_if_type} ({f(r.ot_if)}) | {f(r.stuff_plus, 0)} | {f(100 * r.proj_K_rate_2027, 1)}% |")
    L += ["", "## Outlier pitches by type (top 5% of pitcher cell means within each type; >= 30 pitches of that type)", ""]
    for t in ow.TYPES:
        c = cell[(cell["pitch_type"] == t) & cell["cell_top5_within_type"]].sort_values("zi", ascending=False)
        if c.empty:
            continue
        L += [f"### {t} ({len(c)} of {int((cell['pitch_type'] == t).sum())} cells)", "", "| Pitcher | 2026 team | Pitches | Isolation-forest score | Mahalanobis score |", "|---|---|---|---|---|"]
        L += [f"| {r.pitcher} | {r.team_2026} | {r.n} | {r.zi:.2f} | {r.z:.2f} |" for r in c.itertuples()] + [""]
    open("prereg/outlier_watchlist_2027.md", "w", encoding="utf-8").write("\n".join(L))
    print(f"flagged: top5 {int(p['flag_oi_if_top5'].sum())}, top10 {int(p['flag_oi_if_top10'].sum())}, ot_if top5 {int(p['flag_ot_if_top5'].sum())}; cells {len(cell)} ({int(cell['cell_top5_within_type'].sum())} top-5% cells); thresholds {thr}")


if __name__ == "__main__":
    main()
