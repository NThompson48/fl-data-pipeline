"""
fl_hitter_battrack_hrbabip.py -- follow-up to fl_hitter_battrack_test.py: do the user's estimated bat-tracking measures (bat_speed_est, PAA, pad_lac, under_pct) help the CONTACT components, HR/PA and BABIP?  (user 2026-10-27: "run those follow ups")
They describe the swing path rather than total wOBA, so the natural targets are the components the production hitter model regresses without any contact-quality term: HR/PA and BABIP (production = the regressed component, fl_hitter_ros.predict 'hrr' / 'babip', and the preseason path's proj_hrr / proj_babip).
Same pre-specified design as fl_error_analysis_common.py: samples PRE (2025 -> 2026) and ROS25 / ROS26 (mid-season -> rest of season) rebuilt exactly as in fl_error_analysis_hitters.py, level-free residuals, frozen standard weights (HR/PA by PA, BABIP by balls in play), traits measured only in the feature window and shrunk by their own reliability;
screen = residual ~ z(trait) + centred prediction + log history (BH q over ALL 32 traits, pooled ROS25 / ROS26); improvement = faded and constant add-ons, b fit on one season and scored on the other, transferred to PRE, 10-fold CV inside PRE.
Positive controls in the same tables: barrel%, hard-hit%, average EV, mean launch angle, pull-air% (the model's HR/PA regression has no contact-quality term, so these should predict its misses).
The second follow-up (re-test with a 2027 season of tracking) cannot be run until 2027 data exist.
Usage: python fl_hitter_battrack_hrbabip.py      (log: output/battrack_hrbabip_log.txt)    Output: output/battrack_hrbabip.json
"""
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import fl_error_analysis_common as ea
import fl_error_analysis_hitters as eh
import fl_error_analysis_select as sel
import fl_hitter_battrack_test as bk

warnings.filterwarnings("ignore")
NEW = bk.NEW; CONTROLS = ["barrel_pct", "hardhit_pct", "avg_ev", "la_mean", "pull_air_pct", "bat_speed"]
N0_T = {"hr": 136.8, "babip": 851.6}                         # the production regression constants (fl_hitter_ros.SPEC)
MIN_N = 30; MIN_BIP = 30


def new_targets(S, t):
    """add the 'hr' and 'babip' residual frames to S[...]['targets'] (PRE from the preseason path, ROS25 / ROS26 from fl_hitter_ros.predict)."""
    import fl_hitter_model_dev as dev
    import fl_hitter_pre2026 as pre
    import fl_hitter_ros as hr
    d, c, fit = pre.build_shifted(); p = dev.load(); a = p[p["year"] == 2026].drop_duplicates("name_clean").set_index("name_clean")
    a = a.assign(bip26=a["ab"] - a["so"] - a["hr"] + a["sf"]); j = d.set_index("name_clean").join(a[["pa", "hrr", "babip", "bip26"]].rename(columns={"pa": "pa26", "hrr": "hr26", "babip": "bab26"}), how="inner")
    j = j[(j["pa26"] >= MIN_N) & j["proj_woba"].notna()]; hist = j["pa_t"] + j["pa_prev"]
    for tg, pc, ac, nc in (("hr", "proj_hrr", "hr26", "pa26"), ("babip", "proj_babip", "bab26", "bip26")):
        jj = j.dropna(subset=[pc, ac]); jj = jj[jj[nc] >= MIN_BIP]; S["PRE"]["targets"][tg] = ea.frame(jj[ac] - jj[pc], jj[pc], jj[nc], hist.reindex(jj.index), np.zeros(len(jj), bool), jj.index)
    T = hr.Tables()
    for yr in (2025, 2026):
        asof = S[f"ROS{yr % 100}"]["asof"]; f, lg = hr.features(T, yr, asof); pr_ = hr.predict(f, lg, trackman=True)
        ros = T.pa[(T.pa["year"] == yr) & (T.pa["date"] >= asof)].groupby("name_clean").agg(n_ros=("woba", "size"), hr_ros=("hr", "mean"), hit=("hit", "sum"), bip=("bip", "sum")); ros["bab_ros"] = ros["hit"] / ros["bip"].replace(0, np.nan)
        f = f.join(ros[["hr_ros", "bab_ros", "bip"]].rename(columns={"bip": "bip_ros"})); prior = T.panel[T.panel["year"] < yr].groupby("name_clean")["pa"].sum().reindex(f.index).fillna(0); hist = prior + f["n"]; rookie = (prior == 0).to_numpy()
        for tg, pc, ac, nc in (("hr", "hrr", "hr_ros", "n_ros"), ("babip", "babip", "bab_ros", "bip_ros")):
            m = f[ac].notna() & pr_[pc].notna() & (f["n_ros"] >= MIN_N) & (f[nc] >= MIN_BIP)
            S[f"ROS{yr % 100}"]["targets"][tg] = ea.frame((f[ac] - pr_[pc])[m], pr_[pc][m], f[nc][m], hist[m], rookie[m.to_numpy()], f.index[m])
    return S


def main():
    pd.set_option("display.width", 260); t = bk.load_tracking_tolerant(); t["key"] = t["bat"]
    st, S = pickle.load(open("output/_battrack_samples.pkl", "rb"))
    CACHE = "output/_battrack_hrbabip_samples.pkl"
    for name, s_ in S.items():
        s_["window"] = eh.window(t, name, s_.get("asof"))
    if os.path.exists(CACHE):
        S2 = pickle.load(open(CACHE, "rb")); [S[k]["targets"].update(v) for k, v in S2.items()]
    else:
        S = new_targets(S, t); pickle.dump({k: {tg: F for tg, F in v["targets"].items() if tg in N0_T} for k, v in S.items()}, open(CACHE, "wb"))
    for name, s_ in S.items():
        print(f"{name}: " + ", ".join(f"{tg} n={len(F)}" for tg, F in s_["targets"].items() if tg in N0_T), flush=True)
        s_["traits"] = bk.build_all(s_["window"].assign(key=s_["window"]["bat"]), "h")
    n0s = st["n0"].to_dict(); res = {}
    for sname, s in S.items():
        for tg in N0_T:
            r = ea.screen(s["targets"][tg], s["traits"], n0s); r["q"] = ea.bh(r["p"].to_numpy()); res[(sname, tg)] = r
    tables = {}
    for tg in N0_T:
        a, b, pr = res[("ROS25", tg)], res[("ROS26", tg)], res[("PRE", tg)]; k = a.index.intersection(b.index)
        wa, wb = 1 / a.loc[k, "se"] ** 2, 1 / b.loc[k, "se"] ** 2; pc = (a.loc[k, "coef per SD"] * wa + b.loc[k, "coef per SD"] * wb) / (wa + wb); ps = 1 / np.sqrt(wa + wb)
        T = pd.DataFrame({"t ROS25": a.loc[k, "t"], "t ROS26": b.loc[k, "t"], "t PRE": pr["t"].reindex(k), "pooled coef/SD": pc, "pooled t": pc / ps}); T["pooled q"] = ea.bh(2 * norm.sf(np.abs(T["pooled t"].to_numpy()))); T["same sign"] = np.sign(T["t ROS25"]) == np.sign(T["t ROS26"])
        T = T.join(st[["split-half r", "y/y disattenuated"]]); T["shortlist"] = (T["pooled q"] <= .10) & T["same sign"] & (T["split-half r"] >= .40) & (T["y/y disattenuated"] >= .50); tables[tg] = T
        print(f"\nSCREEN, target {tg} (coef per SD in {tg} units; BH q over {len(T)} traits)\n" + T.loc[[x for x in NEW + CONTROLS if x in T.index]].round(4).to_string(), flush=True)
        print("   shortlisted overall: " + (", ".join(T.index[T["shortlist"]]) or "none"), flush=True)
    single = []; _, which = sel.designs("woba", S)
    for tg in N0_T:
        for c in NEW + ["barrel_pct", "avg_ev"]:
            for fade in (True, False):
                for r in sel.evaluate([c], tg, S, st, N0_T, fade, which):
                    single.append({"target": tg, "trait": c, "form": "faded" if fade else "constant", **r})
    A = pd.DataFrame(single); print("\nIMPROVEMENT, single-trait add-ons (dRMSE < 0 = better; overall rows)\n" + A[A["group"] == "all"].drop(columns=["group", "RMSE old", "MAE old", "RMSE new", "MAE new", "RMSE 90%"]).round(5).to_string(index=False))
    for tg in N0_T:
        for lab, L in (("PAA + pad_lac + under_pct (the new swing-path set)", ["paa", "pad_lac", "under_pct"]), ("new set + barrel%", ["paa", "pad_lac", "under_pct", "barrel_pct"])):
            for fade in (False,):
                rows = sel.evaluate(L, tg, S, st, N0_T, fade, which)
                print(f"\nJOINT {lab}, target {tg}, constant form\n" + pd.DataFrame(rows)[lambda d: d["group"] == "all"][["test", "n", "dRMSE", "RMSE 90%", "P(RMSE better)", "dMAE", "P(MAE better)"]].round(5).to_string(index=False))
    json.dump({"screen": {tg: T.reset_index().to_dict("records") for tg, T in tables.items()}, "improvement": single}, open("output/battrack_hrbabip.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
