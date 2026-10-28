"""
fl_pitcher_velo_trend_test.py -- follow-up to fl_pitcher_breakout.py: does a pitcher's RECENT velocity (a velocity changepoint / the last 25% of his pitches vs his window) add to K% prediction beyond the fastball velocity level the project already tracks (pre-registered C1)?
In the changepoint screen the velocity-recency features were the most consistent (same sign in ROS25, ROS26 and PRE, pooled t about +2, but BH q .42 over 66 traits).  Same design as the error analysis (fl_error_analysis_select): the cached rebuilt pitcher samples (output/_pitcher_breakout_samples.pkl),
b fit on one sample and scored on another (ROS25 <-> ROS26, ROS25 -> PRE, CV inside PRE), faded and constant add-ons, weighted RMSE / MAE with frozen weights and a player bootstrap.  Targets K% and BB%.   Output: output/pitcher_velo_trend_log.txt
"""
import warnings

import numpy as np
import pandas as pd
import pickle

import fl_error_analysis_common as ea
import fl_error_analysis_pitchers as epi
import fl_error_analysis_select as sel
import fl_pitcher_breakout as fpb
import fl_pitcher_features as fp
import fl_position_player_flag as ppf

warnings.filterwarnings("ignore")
fpb.SERIES = [s for s in fpb.SERIES if s != "stuff"]


def main():
    pd.set_option("display.width", 250); t, rawc = fpb.tracking_with_pitcher_columns(); raw = ppf.exclude(rawc); id2name = t.drop_duplicates("pitcher_id").set_index(t.drop_duplicates("pitcher_id")["pitcher_id"].astype(str))["pit"].to_dict()
    st, S, n0t = pickle.load(open(fpb.CACHE, "rb"))
    for name, s in S.items():
        w = epi.window(t, name, s.get("asof")).copy(); w["key"] = w[s["key"]]; s["window"] = w; s["traits"] = ea.build(w, "p"); s["traits"].update(fpb.cp_pitcher(fpb.window_raw(raw, name, s.get("asof")), id2name if s["key"] == "pit" else {k: k for k in id2name}))
    CP = [k for k in S["ROS25"]["traits"] if k.split("_")[0] in fpb.SERIES and k.split("_", 1)[1] in ("cp_t", "cp_shift", "cp_recent", "rec25")]
    for k in CP:
        st.loc[k, "n0"] = 0.0
    _, which = sel.designs("k", S); rows = []; sets = {"fastball velocity (the level; pre-registered C1)": ["fb_velo"], "velocity recency (last 25% of pitches)": ["velo_rec25"], "velocity changepoint (post-split level)": ["velo_cp_recent"], "fb velocity + velocity recency": ["fb_velo", "velo_rec25"],
                                                                      "fb velocity + velocity changepoint": ["fb_velo", "velo_cp_recent"], "fb velocity + extension": ["fb_velo", "extension"], "fb velocity + extension + velocity recency": ["fb_velo", "extension", "velo_rec25"], "release-side recency": ["rels_rec25"], "whiff recency": ["whiff_rec25"]}
    for tg in ("k", "bb"):
        for lab, L in sets.items():
            for fade in (True, False):
                for r in sel.evaluate(L, tg, S, st, n0t, fade, which):
                    rows.append({"target": tg, "set": lab, "form": "faded" if fade else "constant", **r})
    A = pd.DataFrame(rows); A = A[A["group"] == "all"].copy(); A["rel %"] = 100 * A["dRMSE"] / A["RMSE old"]
    for tg in ("k", "bb"):
        for form in ("constant", "faded"):
            T = A[(A["target"] == tg) & (A["form"] == form)]; print(f"\nTARGET {tg.upper()} ({form} form): dRMSE as % of production (negative = better); P(RMSE better)\n" + T.pivot_table(index="set", columns="test", values="rel %", sort=False).round(2).to_string() + "\n" + T.pivot_table(index="set", columns="test", values="P(RMSE better)", sort=False).round(2).to_string())
    A.to_csv("output/pitcher_velo_trend.csv", index=False)


if __name__ == "__main__":
    main()
