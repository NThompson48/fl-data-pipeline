"""
fl_ros_stuff_estimator_test.py -- should the pitcher ROS (rest-of-season) model use the swstr_ct stuff level instead of the XGBoost wh_sva?  (user 2026-10-25: "what about the ROS model?")
The ROS pitcher model's K% 'power prior' is  proj = ROS update + B x (shrunk stuff deviation) x n0 / (PA_eff + n0)  (fl_pitcher_ros_stuff.py), with the stuff level rebuilt at each as-of date from last season's pitches + this season's pitches
BEFORE the date (nothing from the ROS window enters the features).  This reruns that harness's K% test with the stuff estimator swapped, everything else identical (same cutoffs, same two-stage design: the coefficient is fit on one season's ROS rows and frozen, scored on the other season,
forward 2025 -> 2026 and reverse 2026 -> 2025; pitcher-cluster two-stage bootstrap; every model gets the same constant):
    xgb = production-until-now ROS estimator (reproduces the logged numbers) | swstr_ct = per-pitch swinging strike with pitch type and count (the new production 2027 estimator) | torch3 = swing-based network.
Reliability n0 is recomputed per estimator (rows are swings for xgb / torch3, pitches for swstr_ct).  The metric is the change in weighted RMSE of ROS K% vs the ROS update without a stuff term, with the bootstrap 90% interval and P(helps); also split by the pitcher's
effective PA at the as-of date (thin < 100, 100+).  BB% is untouched (the stuff estimator does not enter it).   Rows are saved under estimator-specific names (the production rows files are NOT overwritten).   Output: output/ros_stuff_estimator_test.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_pitcher_features as fp
import fl_pitcher_ros as pr
import fl_pitcher_ros_backtest as pb
import fl_pitcher_ros_stuff as ps
import fl_pitcher_ros_terms as tm

warnings.filterwarnings("ignore")


def one(est):
    fp.WH_SVA_ESTIMATOR = est; raw = fp.load_fl_pitches_cached(); names = ps.idmap(raw); T = pr.Tables(); n0s = ps.reliability(raw)
    print(f"[{est}] reliability (random-half r) and n0: {{'wh_sva': ({n0s['wh_sva'][1]:.3f}, {n0s['wh_sva'][0]:.0f})}}", flush=True)
    consts = {2025: pr.fit_constants(2024), 2026: pr.fit_constants(2025)}; R = {y: ps.rows(T, raw, names, consts[y], y, n0s) for y in (2025, 2026)}
    for y in R:
        R[y].to_parquet(f"output/pitcher_ros_stuff_rows_{est}_{y}.parquet", index=False)
    res = {"n0": n0s["wh_sva"][0], "r": n0s["wh_sva"][1]}
    for tag, tr, te in (("forward 2025->2026", 2025, 2026), ("reverse 2026->2025", 2026, 2025)):
        full, reps = tm.stage1(R[tr], "k", ["wh_sva"], consts[tr]["k"]["n0"]); out = {"coef": [float(c) for c in full[1:]]}
        for grp, sel in (("all", np.ones(len(R[te]), bool)), ("thin (PA_eff < 100)", R[te]["pa_eff_k"].to_numpy() < 100), ("100+ PA_eff", R[te]["pa_eff_k"].to_numpy() >= 100)):
            sub = R[te][sel].reset_index(drop=True)
            if len(sub) < 30:
                continue
            point, ds = tm.stage2(sub, "k", ["wh_sva"], consts[te]["k"]["n0"], full, reps)
            y = sub["y_k"].to_numpy(); b = sub["base_k"].to_numpy(); w = ew.weights_from_base(y, b, sub["n_ros"].to_numpy())[0]
            out[grp] = {"n": int(len(sub)), "base RMSE": float(pb.wrmse(y, b, w)), "dRMSE": float(point), "90%": [float(np.percentile(ds, 5)), float(np.percentile(ds, 95))], "P(helps)": float((ds < 0).mean())}
        res[tag] = out
        print(f"[{est}] {tag}: stuff coefficient {out['coef']}  " + "; ".join(f"{g}: n {v['n']}, base {v['base RMSE']:.5f}, dRMSE {v['dRMSE']:+.5f} [{v['90%'][0]:+.5f}, {v['90%'][1]:+.5f}] P(helps) {v['P(helps)']:.2f}" for g, v in out.items() if g != "coef"), flush=True)
    return res


def main():
    out = {est: one(est) for est in ("xgb", "swstr_ct", "torch3")}
    json.dump(out, open("output/ros_stuff_estimator_test.json", "w"), indent=1, default=float)
    print("\nSUMMARY: change in ROS K% RMSE vs the ROS update (negative = the stuff term helps); P(helps)")
    for tag in ("forward 2025->2026", "reverse 2026->2025"):
        for g in ("all", "thin (PA_eff < 100)", "100+ PA_eff"):
            print(f"  {tag:20s} {g:20s} " + " | ".join(f"{est}: {out[est][tag][g]['dRMSE']:+.5f} (P {out[est][tag][g]['P(helps)']:.2f})" for est in out if g in out[est][tag]))


if __name__ == "__main__":
    main()
