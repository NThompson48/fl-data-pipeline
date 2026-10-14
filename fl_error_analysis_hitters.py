"""
fl_error_analysis_hitters.py -- HITTER projection error analysis (design: fl_error_analysis_common.py docstring).

Samples (all out of sample for the outcome):
  PRE    production preseason path on 2025-and-earlier data -> 2026 (fl_hitter_pre2026.build_shifted); targets park-neutral wOBA, K%, BB% over 2026; >= 30 PA in 2026;
         traits from the 2025 tracking season; history = PA_t + PA_t-1; no rookies (a preseason projection needs an FL season).
  ROS25  the production rest-of-season model (fl_hitter_ros.predict, Trackman terms on) as of mid-season 2025 -> the rest of 2025 (>= 30 PA after the date);
  ROS26  the same for 2026.  Traits from that season's pitches before the date; history = all earlier FL PA + PA so far this season; rookies = no earlier FL season.
ROS25 and ROS26 have disjoint outcomes (discovery / replication); PRE overlaps ROS26 in outcome season.
Usage:  python fl_error_analysis_hitters.py      (log: output/error_analysis_hitters_log.txt)
"""
import warnings

import numpy as np
import pandas as pd

import fl_error_analysis_common as ea

warnings.filterwarnings("ignore")
N0_T = {"woba": 434.2, "k": 79.7, "bb": 146.7}
MIN_N = 30
CACHE = "output/_error_analysis_hitter_samples.pkl"


def window(t, name, asof=None):
    return t[t["year"] == 2025] if name == "PRE" else t[(t["year"] == 2000 + int(name[-2:])) & (t["date"] < asof)]


def samples(t):
    import fl_hitter_baseline_backtest as bt
    import fl_hitter_model_dev as dev
    import fl_hitter_pre2026 as pre
    import fl_hitter_ros as hr
    S = {}
    # ---- PRE
    d, c, fit = pre.build_shifted()
    p = dev.load(); a = p[p["year"] == 2026].drop_duplicates("name_clean").set_index("name_clean")
    j = d.set_index("name_clean").join(a[["pa", "woba_n", "k", "bbr"]].rename(columns={"pa": "pa26", "woba_n": "woba26", "k": "k26", "bbr": "bb26"}), how="inner")
    j = j[(j["pa26"] >= MIN_N) & j["proj_woba"].notna()]; hist = j["pa_t"] + j["pa_prev"]
    tr = {}
    for tg, pc, ac in (("woba", "proj_woba", "woba26"), ("k", "proj_k", "k26"), ("bb", "proj_bbr", "bb26")):
        jj = j.dropna(subset=[pc, ac]); tr[tg] = ea.frame(jj[ac] - jj[pc], jj[pc], jj["pa26"], hist.reindex(jj.index), np.zeros(len(jj), bool), jj.index)
    S["PRE"] = {"targets": tr, "window": t[t["year"] == 2025]}
    # ---- ROS25 / ROS26
    T = hr.Tables()
    for yr in (2025, 2026):
        t0, t1 = T.season_bounds(yr); asof = t0 + (t1 - t0) / 2; asof = pd.Timestamp(asof.date())
        f, lg = hr.features(T, yr, asof); pr_ = hr.predict(f, lg, trackman=True)
        prior = T.panel[T.panel["year"] < yr].groupby("name_clean")["pa"].sum().reindex(f.index).fillna(0)
        hist = prior + f["n"]; rookie = (prior == 0).to_numpy()
        tr = {}
        for tg, pc, ac in (("woba", "woba_final", "woba_ros"), ("k", "k_final", "k_ros"), ("bb", "bbr_final", "bb_ros")):
            m = f[ac].notna() & pr_[pc].notna() & (f["n_ros"] >= MIN_N)
            tr[tg] = ea.frame((f[ac] - pr_[pc])[m], pr_[pc][m], f["n_ros"][m], hist[m], rookie[m.to_numpy()], f.index[m])
        S[f"ROS{yr % 100}"] = {"targets": tr, "window": t[(t["year"] == yr) & (t["date"] < asof)], "asof": asof}
        print(f"ROS{yr % 100}: as of {asof.date()}, {m.sum()} hitters with >= {MIN_N} PA after it; rookies {int(rookie[m.to_numpy()].sum())}; low-history (< 150 PA) {int((hist[m] < 150).sum())}", flush=True)
    print(f"PRE: {len(j)} hitters (>= {MIN_N} PA in 2026), low-history {int(((j['pa_t'] + j['pa_prev']) < 150).sum())}")
    return S


def main():
    import os
    import pickle
    t = ea.load_tracking(); t["key"] = t["bat"]
    print("STICKINESS (alternating games, 2025 and 2026; y/y 2025 -> 2026)")
    if os.path.exists(CACHE):
        st, S = pickle.load(open(CACHE, "rb"))
        for name, s_ in S.items():
            s_["window"] = window(t, name, s_.get("asof"))
    else:
        st = ea.stickiness(t, "h"); S = samples(t)
        pickle.dump((st, {k: {kk: vv for kk, vv in v.items() if kk != "window"} for k, v in S.items()}), open(CACHE, "wb"))
    print(st.round(3).to_string()); n0s = st["n0"].to_dict()
    res = {}
    for sname, s in S.items():
        tw = s["window"].copy(); traits = ea.build(tw, "h")
        for tg, F in s["targets"].items():
            r = ea.screen(F, traits, n0s); r["q"] = ea.bh(r["p"].to_numpy()); res[(sname, tg)] = r
    out = {}
    for tg in N0_T:
        a, b, pr = res[("ROS25", tg)], res[("ROS26", tg)], res[("PRE", tg)]
        k = a.index.intersection(b.index)
        wa, wb = 1 / a.loc[k, "se"] ** 2, 1 / b.loc[k, "se"] ** 2; pc = (a.loc[k, "coef per SD"] * wa + b.loc[k, "coef per SD"] * wb) / (wa + wb); ps = 1 / np.sqrt(wa + wb)
        from scipy.stats import norm
        T_ = pd.DataFrame({"t ROS25": a.loc[k, "t"], "t ROS26": b.loc[k, "t"], "t PRE": pr["t"].reindex(k), "pooled coef/SD": pc, "pooled t": pc / ps})
        T_["pooled q"] = ea.bh(2 * norm.sf(np.abs(T_["pooled t"].to_numpy()))); T_["same sign"] = np.sign(T_["t ROS25"]) == np.sign(T_["t ROS26"])
        T_ = T_.join(st[["split-half r", "y/y disattenuated"]]); T_["shortlist"] = (T_["pooled q"] <= .10) & T_["same sign"] & (T_["split-half r"] >= .40) & (T_["y/y disattenuated"] >= .50)
        out[tg] = T_.sort_values("pooled t", key=np.abs, ascending=False)
        print(f"\nSCREEN, target {tg} (residual = actual - projected; coef per SD of the shrunk trait, in {tg} units; controls: centred prediction, log history)\n" + out[tg].round(4).to_string())
    pd.concat({tg: T_ for tg, T_ in out.items()}).to_csv("output/error_analysis_hitters_screen.csv"); st.to_csv("output/error_analysis_hitters_stickiness.csv")


if __name__ == "__main__":
    main()
