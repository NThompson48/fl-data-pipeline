"""
fl_velo_trend_prereg.py -- freeze check C12 (REPORT ONLY, before any 2027 data): pitcher K% with fastball velocity LEVEL + velocity RECENCY  (user 2026-10-28: "looks promising")
Candidate (fl_pitcher_velo_trend_test.py): proj_K += b1 x (fb_velo_shrunk - mean) + b2 x (velo_rec25_shrunk - mean), constant form, b fit jointly on the pooled ROS25 + ROS26 K% residuals of the REBUILT pitcher error-analysis samples (production network stuff term); fb_velo = mean effective velocity of
four-seams / sinkers in the feature window shrunk by n / (n + n0) with n0 from the split-half stickiness table; velo_rec25 = mean of the last 25% of his pitches minus his window mean, velocity standardised within pitch type, in SD units (no shrinkage).  C1 (level only, frozen 2026-10-13) stays as registered.
Freezes: the coefficients and constants, and per pitcher the 2027 preseason features (window 2025 + 2026 regular season), the published production K% projection and the counterfactual K% projection.  Rule (after 2027): weighted RMSE / MAE of the production, C1 and C12 projections against actual 2027 K%
(frozen standard weights from the production projection, level-free, pitcher bootstrap); the preseason sample and the 2027 mid-season ROS sample; report only, no production change.
Usage: python fl_velo_trend_prereg.py --freeze
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

import fl_error_analysis_common as ea
import fl_error_analysis_pitchers as epi
import fl_error_analysis_select as sel
import fl_pitcher_breakout as fpb
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_halfseason_backtest as bt

L = ["fb_velo", "velo_rec25"]
C12 = "C12 (report only): pitcher K% with fastball velocity LEVEL + velocity RECENCY (last 25% of pitches vs window, within-type velocity z) vs production and vs C1; weighted RMSE / MAE, level-free, frozen standard weights, pitcher bootstrap; no production change"


def main():
    fpb.SERIES = ["velo"]; t, rawc = fpb.tracking_with_pitcher_columns(); raw = ppf.exclude(rawc); id2name = t.drop_duplicates("pitcher_id").set_index(t.drop_duplicates("pitcher_id")["pitcher_id"].astype(str))["pit"].to_dict()
    st, S, n0t = pickle.load(open(fpb.CACHE, "rb"))
    for name, s in S.items():
        w = epi.window(t, name, s.get("asof")).copy(); w["key"] = w[s["key"]]; s["traits"] = ea.build(w, "p"); s["traits"].update(fpb.cp_pitcher(fpb.window_raw(raw, name, s.get("asof")), id2name if s["key"] == "pit" else {k: k for k in id2name}))
    st.loc["velo_rec25", "n0"] = 0.0; n0_fb = float(st.loc["fb_velo", "n0"])
    Fs = [S[s]["targets"]["k"] for s in ("ROS25", "ROS26")]; Xs = [sel.X(S[s], "k", L, st, n0t, False) for s in ("ROS25", "ROS26")]; b = sel.fit(pd.concat(Fs), np.vstack(Xs)); per = {s: sel.fit(S[s]["targets"]["k"], sel.X(S[s], "k", L, st, n0t, False)) for s in ("ROS25", "ROS26", "PRE")}
    print(f"coefficients (K per unit of the shrunk deviation): fb_velo {b[0]:+.5f} (per mph), velo_rec25 {b[1]:+.5f} (per SD); by sample: " + "; ".join(f"{s}: {v[0]:+.5f} / {v[1]:+.5f}" for s, v in per.items()) + f"; n0 fb_velo {n0_fb:.1f}", flush=True)
    # 2027 features: window = 2025 + 2026 regular season
    win = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025), fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True); win["pid"] = win["pitcher_id"].astype(str)
    fb = win[win["pitch_type"].isin(fp.FASTBALL_TYPES) & win["effective_velo"].notna()].groupby("pid")["effective_velo"].agg(["mean", "size"]); x1 = ea.shrink(fb["mean"], fb["size"].astype(float), n0_fb)
    cp = fpb.cp_pitcher(win, {k: k for k in win["pid"].unique()}); v2, n2 = cp["velo_rec25"]; x2 = v2
    pub = pd.read_parquet("output/fl_2027_projections.parquet"); pub["pid"] = pub["pitcher_id"].astype(str); d = pub[["pid", "pitcher", "team", "PA", "proj_K_rate"]].copy(); d["fb_velo_shrunk"] = d["pid"].map(x1); d["velo_rec25"] = d["pid"].map(x2)
    wts = d["PA"].clip(lower=1); mu1 = float(np.average(d["fb_velo_shrunk"].dropna(), weights=wts[d["fb_velo_shrunk"].notna()])); mu2 = float(np.average(d["velo_rec25"].dropna(), weights=wts[d["velo_rec25"].notna()]))
    d["term_level"] = b[0] * (d["fb_velo_shrunk"] - mu1); d["term_recency"] = b[1] * (d["velo_rec25"] - mu2); d[["term_level", "term_recency"]] = d[["term_level", "term_recency"]].fillna(0.0)
    d["proj_K_rate_c1_level"] = d["proj_K_rate"] + d["term_level"]; d["proj_K_rate_c12"] = d["proj_K_rate"] + d["term_level"] + d["term_recency"]
    print(f"{len(d)} pitchers; with a velocity level {int(d['fb_velo_shrunk'].notna().sum())}, with a recency feature {int(d['velo_rec25'].notna().sum())}; shift (K points) SD: level {100 * d['term_level'].std():.2f}, recency {100 * d['term_recency'].std():.2f}, max |total| {100 * (d['term_level'] + d['term_recency']).abs().max():.2f}")
    snap = {"frozen_on": "2026-10-28", "note": C12, "coefficients": {"fb_velo_per_mph": float(b[0]), "velo_rec25_per_sd": float(b[1])}, "constants": {"mu_fb_velo": mu1, "mu_velo_rec25": mu2, "n0_fb_velo": n0_fb, "rec25_shrinkage": "none"}, "by_sample": {s: [float(v[0]), float(v[1])] for s, v in per.items()},
            "pitchers": d.drop(columns=["pid"]).assign(pitcher_id=pub["pitcher_id"].astype(str)).to_dict("records")}
    json.dump(snap, open("prereg/velo_trend_frozen_2027.json", "w"), default=float)
    for path in ("output/preregistered_2027.json", "prereg/preregistered_2027.json"):
        try:
            spec = json.load(open(path))
        except FileNotFoundError:
            continue
        spec["checks"]["C12 pitcher K% velocity level + recency (report only)"] = C12; spec["frozen"]["p:k:c12"] = {"coefficients": snap["coefficients"], "constants": snap["constants"], "n_pitchers": int(len(d)), "projections_file": "prereg/velo_trend_frozen_2027.json"}
        if not any("C12" in a.get("note", "") for a in spec.setdefault("amendments", [])):
            spec["amendments"].append({"date": "2026-10-28", "note": "C12 added before any 2027 data (report only, no production change): pitcher K% with fastball velocity level + velocity recency vs production and C1; coefficients and per-pitcher 2027 features / projections frozen (prereg/velo_trend_frozen_2027.json). Evidence: held-out K% RMSE -1.2 to -2.5% (level alone -1.0 to -2.0%), feature picked from a 36-feature screen (q .42), so optimistic."})
        json.dump(spec, open(path, "w"), indent=1); print("updated", path)


if __name__ == "__main__":
    main() if "--freeze" in sys.argv else sys.exit(__doc__)
