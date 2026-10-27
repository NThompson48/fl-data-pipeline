"""
fl_fbcmd_prereg.py -- freeze check C11 (REPORT ONLY, before any 2027 data): the fastball-blend command term for pitcher BB%  (user 2026-10-27: "run checks")
Production BB% (preseason) = [blended_BB_rate, command_score] (coefficient on command_score in output/fl_projection_model_coefs.json).  COUNTERFACTUAL = the same model with the all-pitch Command Score replaced by the average of the all-pitch score and the frozen-formula score on four-seams + sinkers
(>= 50 fastballs in the 2025 + 2026 regular-season window; otherwise unchanged), SAME coefficient (no refit; a first-order substitution; pitchers with no 2026 PA get the blend shift x .80, the project's stale-feature retention).
Freezes per pitcher: the all-pitch score, the fastball score, the blend, the production and counterfactual 2027 BB% projections (prereg/fastball_blend_frozen_2027.json) and adds C11 + an amendment to output/ and prereg/ pre-registration files.
Rule (after 2027): PA-weighted, level-free weighted RMSE / MAE of both projections against actual 2027 BB% (frozen standard weights from the production projection, pitcher bootstrap 1,000); report only, no production change.
Evidence at freeze: four-pairing walk-forward harness blend -0.9% RMSE (P .72-.75); ROS forward -.0001 (P .75), ROS reverse +.0004 (P .15, worse): not replicated, so NOT adopted.
Usage: python fl_fbcmd_prereg.py --freeze
"""
import json
import sys

import numpy as np
import pandas as pd

import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_ros_fbcmd_test as fb

C11 = "C11 (report only): preseason pitcher BB% with the fastball-blend command term (average of the all-pitch Command Score and the frozen-formula score on four-seams + sinkers, >= 50 fastballs) vs production; weighted RMSE / MAE, level-free, frozen standard weights, pitcher bootstrap; no production change"


def freeze():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025), fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True); s = fb.scores(pool)
    pub = pd.read_parquet("output/fl_2027_projections.parquet"); coefs = json.load(open("output/fl_projection_model_coefs.json")); b = float(coefs["bb_model"]["command_score"])
    d = pub[["pitcher_id", "pitcher", "PA", "PA_2026", "command_score", "proj_BB_rate"]].copy(); d["pid"] = d["pitcher_id"]; s2 = s.reset_index().rename(columns={"pitcher_id": "pid", "command_score": "cmd_raw"}); s2["pid"] = s2["pid"].astype(d["pid"].dtype)
    d = d.merge(s2[["pid", "cmd_raw", "fb_cmd", "cmd_blend", "n_fb"]], on="pid", how="left"); shift = (d["cmd_blend"] - d["cmd_raw"]).fillna(0.0); shift = np.where(d["PA_2026"].fillna(0) > 0, shift, .8 * shift)
    d["cmd_blend_term"] = d["command_score"] + shift; d["proj_BB_rate_blend"] = d["proj_BB_rate"] + b * shift
    print(f"{len(d)} pitchers; {int(d['fb_cmd'].notna().sum())} with a fastball score; shift in the command term: SD {np.std(shift):.3f} in, max |{np.max(np.abs(shift)):.2f}|; projected BB% change: SD {100 * (d['proj_BB_rate_blend'] - d['proj_BB_rate']).std():.3f} pts, max |{100 * (d['proj_BB_rate_blend'] - d['proj_BB_rate']).abs().max():.2f}| pts")
    snap = {"frozen_on": "2026-10-27", "note": C11, "b_command_score": b, "pitchers": d.assign(pitcher_id=d["pitcher_id"].astype(str)).drop(columns="pid").to_dict("records")}; json.dump(snap, open("prereg/fastball_blend_frozen_2027.json", "w"), default=float)
    for path in ("output/preregistered_2027.json", "prereg/preregistered_2027.json"):
        try:
            spec = json.load(open(path))
        except FileNotFoundError:
            continue
        spec["checks"]["C11 fastball-blend command term for BB% (report only)"] = C11; spec["frozen"]["p:bb:c11"] = {"b_command_score": b, "n_pitchers": int(len(d)), "projections_file": "prereg/fastball_blend_frozen_2027.json"}
        if not any("C11" in a.get("note", "") for a in spec.setdefault("amendments", [])):
            spec["amendments"].append({"date": "2026-10-27", "note": "C11 added before any 2027 data (report only, no production change): fastball-blend command term vs production for pitcher BB%; production and counterfactual 2027 projections frozen (prereg/fastball_blend_frozen_2027.json). Evidence: walk-forward harness -0.9% RMSE, ROS forward +, ROS reverse - (not replicated)."})
        json.dump(spec, open(path, "w"), indent=1); print("updated", path)


if __name__ == "__main__":
    freeze() if "--freeze" in sys.argv else sys.exit(__doc__)
