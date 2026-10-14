"""
fl_error_analysis_pitchers.py -- PITCHER projection error analysis (design: fl_error_analysis_common.py docstring).

Samples (all out of sample for the outcome):
  PRE    K% / BB%: the frozen production model (fit full 2025 -> 1H2026, position players excluded) on fold 2: features 2025 + 1H2026 -> actual 2H2026 (>= 30 BF);
         traits from 2025 + 1H2026 tracking; history = BF in the feature window; rookies = no 2025 FL appearance.
  RA9    preseason RA9 2025 -> 2026 (fl_pwar_calibration out-of-sample build, >= 10 IP in 2026); traits from 2025; history = 2025 BF; rookies = no 2025 record.
  ROS25  the production pitcher ROS model (fl_ros_production.pitcher_snapshot) as of mid-season 2025 -> rest of 2025 (K%, BB%; >= 30 BF after the date);
  ROS26  the same for 2026 (K%, BB%, and RA9 from the play-by-play, >= 10 IP after the date).  Traits from that season before the date; history = earlier FL BF + BF so far;
         rookies = no earlier FL season.  Caveat: ROS constants were fit on 2025 (fl_ros_build), so ROS25 is not fully out of sample for the existing terms.
Usage:  python fl_error_analysis_pitchers.py      (log: output/error_analysis_pitchers_log.txt)
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import fl_error_analysis_common as ea

warnings.filterwarnings("ignore")
MIN_N = 30
CACHE = "output/_error_analysis_pitcher_samples.pkl"
N0_RA9 = 300.0           # BF; fade constant for RA9 (no RA9 regression n0 exists; constant b reported alongside)


def window(t, name, asof=None):
    import fl_halfseason_backtest as bt
    if name == "PRE":
        return t[(t["year"] == 2025) | ((t["year"] == 2026) & (t["date"] <= pd.Timestamp(bt.CUTOFF_2026)))]
    if name == "RA9":
        return t[t["year"] == 2025]
    yr = 2000 + int(name[-2:]); return t[(t["year"] == yr) & (t["date"] < asof)]


def samples(t):
    import fl_halfseason_backtest as bt
    import fl_position_player_flag as ppf
    import fl_posplayer_refit_check as chk
    import fl_production_projection as pp
    import fl_pitcher_features as fp
    import fl_pwar_calibration as cal
    import fl_hitter_ros as hr
    import fl_pitcher_ros as pr
    import fl_pitcher_ros_stuff as pst
    import fl_ros_production as rp
    import fl_ros_workload as wl
    S = {}; n0t = {}
    tp = pr.Tables(); id2name = t.drop_duplicates("pitcher_id").set_index("pitcher_id")["pit"]
    prior_bf = lambda yr: tp.panel[tp.panel["year"] < yr].groupby("name_clean")["bf"].sum()
    # ---- PRE (K%, BB%)
    raw = ppf.exclude(fp.load_fl_pitches_cached()); m_k, m_bb, n0_k, n0_bb, n0b, rho = pp.fit_frozen_model(raw); dk, db = chk.fold2(raw, m_k, m_bb, n0b)
    n0t.update({"k": float(n0_k), "bb": float(n0_bb), "ra9": N0_RA9})
    win = t[((t["year"] == 2025)) | ((t["year"] == 2026) & (t["date"] <= pd.Timestamp(bt.CUTOFF_2026)))]
    tr = {}
    for tg, d, ac in (("k", dk, "K_rate_actual"), ("bb", db, "BB_rate_actual")):
        d = d.dropna(subset=["pred", ac]); d = d[d["PA_test"] >= MIN_N]; d = d.drop_duplicates("pitcher_id").set_index(d.drop_duplicates("pitcher_id")["pitcher_id"].astype(str))
        nm = d.index.map(id2name); rookie = ~nm.isin(prior_bf(2026).index[prior_bf(2026) > 0])
        tr[tg] = ea.frame(d[ac] - d["pred"], d["pred"], d["PA_test"], d["PA"], rookie, d.index)
    S["PRE"] = {"targets": tr, "window": win, "key": "pitcher_id"}
    print(f"PRE: K% n = {len(tr['k'])}, BB% n = {len(tr['bb'])}; rookies {int(tr['k']['rookie'].sum())}, low-history {int((tr['k']['hist'] < 150).sum())}", flush=True)
    # ---- RA9 preseason
    cal.RETURN_ALL = True; b, war = cal.main(); b = b[(b["ip"] >= 10) & b["pred"].notna() & b["target"].notna()].copy(); b["pitcher_id"] = b["pitcher_id"].astype(str); b = b.drop_duplicates("pitcher_id").set_index("pitcher_id")
    bf25 = war[war["year"] == 2025].groupby("player")["bf"].sum(); h = b["player"].map(bf25).fillna(0)
    S["RA9"] = {"targets": {"ra9": ea.frame(b["target"] - b["pred"], b["pred"], b["ip"], h, (h == 0).to_numpy(), b.index)}, "window": t[t["year"] == 2025], "key": "pitcher_id"}
    print(f"RA9: n = {len(b)}; no 2025 record {int((h == 0).sum())}", flush=True)
    # ---- ROS25 / ROS26
    T_h = hr.Tables(); ctx = rp.PitcherContext(tp); D = wl.collect(T_h, tp); wf_p = rp.workload_fit(D, "pit", (2025,)); consts = pr.fit_constants(2025); n0s = pst.reliability(ctx.raw); B = rp.STUFF_B["clean"]
    for yr in (2025, 2026):
        t0, t1 = tp.season_bounds(yr); asof = pd.Timestamp((t0 + (t1 - t0) / 2).date())
        o = rp.pitcher_snapshot(tp, ctx, yr, asof, consts, n0s, B, wf_p)
        pb = prior_bf(yr).reindex(o.index).fillna(0); hist = pb + o["ytd_bf"]; rookie = (pb == 0).to_numpy(); tr = {}
        for tg, pc, ac in (("k", "ros_k", "act_k"), ("bb", "ros_bb", "act_bb")):
            m = (o["act_bf"] >= MIN_N) & o[ac].notna(); tr[tg] = ea.frame((o[ac] - o[pc])[m], o[pc][m], o["act_bf"][m], hist[m], rookie[m.to_numpy()], o.index[m])
        if yr == 2026:
            m = (o["act_ip"] >= 10) & o["act_ra9"].notna(); tr["ra9"] = ea.frame((o["act_ra9"] - o["ros_ra9"])[m], o["ros_ra9"][m], o["act_ip"][m], hist[m], rookie[m.to_numpy()], o.index[m])
        S[f"ROS{yr % 100}"] = {"targets": tr, "window": t[(t["year"] == yr) & (t["date"] < asof)], "key": "pit", "asof": asof}
        print(f"ROS{yr % 100}: as of {asof.date()}, K% n = {len(tr['k'])}, rookies {int(tr['k']['rookie'].sum())}, low-history {int((tr['k']['hist'] < 150).sum())}" + (f"; RA9 n = {len(tr['ra9'])}" if "ra9" in tr else ""), flush=True)
    return S, n0t


def traits_for(S_, name, cache={}):
    k = (name, S_["key"])
    if k not in cache:
        w = S_["window"].copy(); w["key"] = w[S_["key"]]; cache[k] = ea.build(w, "p")
    return cache[k]


def main():
    t = ea.load_tracking()
    import os
    import pickle
    tt = t.copy(); tt["key"] = tt["pitcher_id"]
    print("STICKINESS (alternating games, 2025 and 2026; y/y 2025 -> 2026)")
    if os.path.exists(CACHE):
        st, S, n0t = pickle.load(open(CACHE, "rb"))
        for name, s in S.items():                               # windows are not pickled (large): rebuild from the tracking table
            s["window"] = window(t, name, s.get("asof"))
    else:
        st = ea.stickiness(tt, "p"); S, n0t = samples(t)
        pickle.dump((st, {k: {kk: vv for kk, vv in v.items() if kk != "window"} for k, v in S.items()}, n0t), open(CACHE, "wb"))
    print(st.round(3).to_string()); n0s = st["n0"].to_dict()
    res = {}
    for sname, s in S.items():
        traits = traits_for(s, sname)
        for tg, F in s["targets"].items():
            r = ea.screen(F, traits, n0s); r["q"] = ea.bh(r["p"].to_numpy()); res[(sname, tg)] = r
    out = {}
    for tg in ("k", "bb"):
        a, b, pr_ = res[("ROS25", tg)], res[("ROS26", tg)], res[("PRE", tg)]; k = a.index.intersection(b.index)
        wa, wb = 1 / a.loc[k, "se"] ** 2, 1 / b.loc[k, "se"] ** 2; pc = (a.loc[k, "coef per SD"] * wa + b.loc[k, "coef per SD"] * wb) / (wa + wb); ps = 1 / np.sqrt(wa + wb)
        T_ = pd.DataFrame({"t ROS25": a.loc[k, "t"], "t ROS26": b.loc[k, "t"], "t PRE": pr_["t"].reindex(k), "pooled coef/SD": pc, "pooled t": pc / ps})
        T_["pooled q"] = ea.bh(2 * norm.sf(np.abs(T_["pooled t"].to_numpy()))); T_["same sign"] = np.sign(T_["t ROS25"]) == np.sign(T_["t ROS26"])
        T_ = T_.join(st[["split-half r", "y/y disattenuated"]]); T_["shortlist"] = (T_["pooled q"] <= .10) & T_["same sign"] & (T_["split-half r"] >= .40) & (T_["y/y disattenuated"] >= .50)
        out[tg] = T_.sort_values("pooled t", key=np.abs, ascending=False); print(f"\nSCREEN, target {tg}\n" + out[tg].round(4).to_string())
    # RA9: two samples share the 2026 outcome season (preseason 2025 -> 2026 and ROS26) -> no disjoint replication; pooled anyway, flagged
    a, b = res[("RA9", "ra9")], res[("ROS26", "ra9")]; k = a.index.intersection(b.index)
    wa, wb = 1 / a.loc[k, "se"] ** 2, 1 / b.loc[k, "se"] ** 2; pc = (a.loc[k, "coef per SD"] * wa + b.loc[k, "coef per SD"] * wb) / (wa + wb); ps = 1 / np.sqrt(wa + wb)
    T_ = pd.DataFrame({"t RA9 pre": a.loc[k, "t"], "t ROS26": b.loc[k, "t"], "pooled coef/SD": pc, "pooled t (not independent)": pc / ps, "q RA9 pre": a.loc[k, "q"], "q ROS26": b.loc[k, "q"]})
    T_ = T_.join(st[["split-half r", "y/y disattenuated"]]); T_["shortlist"] = (T_["q RA9 pre"] <= .10) & (np.sign(T_["t RA9 pre"]) == np.sign(T_["t ROS26"])) & (T_["t ROS26"].abs() >= 1.5) & (T_["split-half r"] >= .40) & (T_["y/y disattenuated"] >= .50)
    out["ra9"] = T_.sort_values("pooled t (not independent)", key=np.abs, ascending=False); print("\nSCREEN, target RA9 (positive = more runs than projected)\n" + out["ra9"].round(4).to_string())
    # ---- improvement tests
    print("\nIMPROVEMENT TESTS (shortlist only)")
    rows = []
    pairs = {"k": [("ROS25", "ROS26"), ("ROS26", "ROS25"), ("ROS25", "PRE")], "bb": [("ROS25", "ROS26"), ("ROS26", "ROS25"), ("ROS25", "PRE")], "ra9": [("RA9", "ROS26"), ("ROS26", "RA9")]}
    for tg, T_ in out.items():
        for trn in T_.index[T_["shortlist"]]:
            for fade in (True, False):
                for a_, b_ in pairs[tg]:
                    A, B_ = S[a_], S[b_]; FA, FB = A["targets"][tg], B_["targets"][tg]
                    va, na = traits_for(A, a_)[trn]; vb, nb = traits_for(B_, b_)[trn]
                    xa = ea.shrink(va, na, n0s[trn]).reindex(FA.index); xb = ea.shrink(vb, nb, n0s[trn]).reindex(FB.index)
                    xa = xa.fillna(np.average(xa.dropna(), weights=FA.loc[xa.notna(), "w"])); xb = xb.fillna(np.average(xb.dropna(), weights=FB.loc[xb.notna(), "w"]))
                    r = ea.improvement(FA, FB, xa, xb, n0t[tg], fade=fade)
                    for grp in ("all", "low-history", "rookies"):
                        if grp in r:
                            rows.append({"target": tg, "trait": trn, "fade": "faded" if fade else "constant", "train -> test": f"{a_} -> {b_}", "group": grp, "b": r["b"], **r[grp]})
    if rows:
        R = pd.DataFrame(rows); print(R.round(5).to_string(index=False)); R.to_csv("output/error_analysis_pitchers_improvement.csv", index=False)
    else:
        print("  (no trait met the shortlist criteria)")
    pd.concat({tg: T_ for tg, T_ in out.items()}).to_csv("output/error_analysis_pitchers_screen.csv"); st.to_csv("output/error_analysis_pitchers_stickiness.csv")


if __name__ == "__main__":
    main()
