"""
fl_hitter_seager_test.py -- swing-DECISION measures in the style of the BP "SEAGER" article (Baseball Prospectus, 'Quantifying the Corey Seager approach'), tested in the hitter model  (user 2026-10-28: might help HR/PA or wOBA overall)
IDEA (article): for every count x pitch location, the expected run value of swinging vs taking; a pitch is HITTABLE when swinging is worth more.  Hittable pitches taken (lower is better), correct non-swings at unhittable pitches, and their combination (SEAGER) describe a hitter's
decision quality; the article reports that it relates to next-season power (ISO) better than chase rate or Z-O swing but is less stable.  It does not use pitch type, damage or exit velocity.
THIS IMPLEMENTATION (an ANALOG; the article gives no exact formula, so the definitions below are ours and were fixed before any result was seen):
  pitch run value   count-state value V(count) = mean run value of the plate-appearance outcomes (tracked rv_batter) of the PAs that passed through the count; a non-terminal pitch is worth V(next count) - V(count), a terminal pitch rv_batter - V(count)
  EV of a decision  separately for swings and for takes (called strikes / balls / balls in the dirt): XGBoost regression of the pitch run value on side-adjusted location (inside = +), height, balls, strikes, cross-fitted by hitter (5 folds, a hitter's own pitches never train his own surface);
                    swing edge = predicted swing value - predicted take value, for EVERY pitch
  traits (per hitter window): hit_taken_pct (share of his takes with swing edge > 0; lower = better), correct_take_pct (takes at edge < 0 over [swings at edge > 0 + takes at edge < 0]), seager = correct_take_pct - hit_taken_pct, regret_100 (runs lost per 100 pitches to
                    decisions worse than the alternative: max(0, edge) on takes + max(0, -edge) on swings), swing_regret_100 (aggression errors), take_regret_100 (passivity errors)
DESIGN: the project's hitter error analysis (fl_error_analysis_common.py; samples PRE / ROS25 / ROS26, rebuilt on the refreshed data as in fl_hitter_battrack_test.py), targets wOBA, K%, BB%, HR/PA, BABIP; stickiness; screen (BH q over ALL traits, now 38); single-trait add-ons and forward selection
(does a decision trait enter once the existing discipline / contact traits are available?); weighted RMSE / MAE, frozen weights, player bootstrap.
CAVEAT: the swing / take surfaces use league pitches of 2025-26 (hitter cross-fitted); they share seasons with the outcome samples but never a hitter's own outcomes.
Usage: python fl_hitter_seager_test.py    (log: output/seager_log.txt)    Output: output/seager_test.json
"""
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

import fl_error_analysis_common as ea
import fl_error_analysis_hitters as eh
import fl_error_analysis_select as sel
import fl_hitter_battrack_test as bk
import fl_pitcher_features as fp

warnings.filterwarnings("ignore")
SEAGER = ["hit_taken_pct", "correct_take_pct", "seager", "regret_100", "swing_regret_100", "take_regret_100"]
N0T = {"woba": 434.2, "k": 79.7, "bb": 146.7, "hr": 136.8, "babip": 851.6}


def add_pitch_values(t):
    """per-pitch run value, swing edge (swing EV - take EV), cross-fitted by hitter.  Adds columns pv, edge, valid."""
    t = t.copy(); key = t["game_id"].astype(str) + "|" + t["pa_id"].astype(str); t["pakey"] = key
    t = t.sort_values(["game_id", "pa_id", "pa_pitch_count"], kind="mergesort").reset_index(drop=False).rename(columns={"index": "orig"}); key = t["pakey"]
    ok = t["balls"].between(0, 3) & t["strikes"].between(0, 2); t["bs"] = t["balls"] * 3 + t["strikes"]
    pa_rv = t.loc[t["rv_batter"].notna()].drop_duplicates("pakey").set_index("pakey")["rv_batter"]; t["pa_rv"] = key.map(pa_rv)
    V = t[ok & t["pa_rv"].notna()].groupby("bs")["pa_rv"].mean(); t["V"] = t["bs"].map(V)
    nxt = t.groupby("pakey")["V"].shift(-1); term = t["rv_batter"].notna()
    t["pv"] = np.where(term, t["rv_batter"] - t["V"], nxt - t["V"]); t["pv"] = t["pv"].where(ok)
    side = t["batter_side"].astype(str); t["x_in"] = np.where(side == "right", -t["plate_x"], t["plate_x"])
    pc = t["pitch_call"].astype(str); t["is_swing"] = t["swing"] == 1; t["is_take"] = t["take"] == 1
    feat = ["x_in", "plate_z", "balls", "strikes"]; base = t["pv"].notna() & t[feat].notna().all(axis=1) & side.isin(["left", "right"]) & (t["is_swing"] | t["is_take"])
    t["valid"] = base; d = t[base]; X = d[feat].to_numpy(float); y = d["pv"].to_numpy(float); g = d["bat"].to_numpy(); sw = d["is_swing"].to_numpy(); es = np.zeros(len(d)); et = np.zeros(len(d))
    prm = {**fp.XGB_PARAMS, "n_jobs": 8, "n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 50}
    for tr, te in GroupKFold(5).split(X, y, g):
        ms = XGBRegressor(**prm).fit(X[tr][sw[tr]], y[tr][sw[tr]]); mt = XGBRegressor(**prm).fit(X[tr][~sw[tr]], y[tr][~sw[tr]]); es[te] = ms.predict(X[te]); et[te] = mt.predict(X[te])
    t["edge"] = np.nan; t.loc[d.index, "edge"] = es - et; t["ev_swing"] = np.nan; t["ev_take"] = np.nan; t.loc[d.index, "ev_swing"] = es; t.loc[d.index, "ev_take"] = et
    print(f"pitches with a swing / take decision and a run value: {int(base.sum()):,}; mean swing edge {np.nanmean(t['edge']):+.4f} runs; share of pitches with edge > 0: {np.nanmean(t.loc[base, 'edge'] > 0):.3f}; swing share {t.loc[base, 'is_swing'].mean():.3f}", flush=True)
    return t.sort_values("orig").set_index("orig").loc[:, ["edge", "pv", "valid", "ev_swing", "ev_take", "is_swing", "is_take"]]


def seager_traits(t):
    """{trait: (value Series by key, n Series)} over the window's decision pitches."""
    d = t[t["valid"].fillna(False)].copy(); d["good_edge"] = d["edge"] > 0; sw = d["is_swing"]; tk = d["is_take"]
    d["regret"] = np.where(tk, np.maximum(0, d["edge"]), np.maximum(0, -d["edge"])); d["sreg"] = np.where(sw, np.maximum(0, -d["edge"]), 0.0); d["treg"] = np.where(tk, np.maximum(0, d["edge"]), 0.0)
    g = d.groupby("key"); n = g.size().astype(float); ntk = d[tk].groupby("key").size().astype(float)
    hit_taken = d[tk].groupby("key")["good_edge"].mean(); good_take = d[tk & ~d["good_edge"]].groupby("key").size(); good_swing = d[sw & d["good_edge"]].groupby("key").size()
    ct = (good_take / (good_take.add(good_swing, fill_value=0))).dropna(); ngood = good_take.add(good_swing, fill_value=0).astype(float)
    out = {"hit_taken_pct": (hit_taken, ntk.reindex(hit_taken.index)), "correct_take_pct": (ct, ngood.reindex(ct.index))}
    sg = (ct.reindex(hit_taken.index) - hit_taken).dropna(); out["seager"] = (sg, n.reindex(sg.index))
    for k, c in (("regret_100", "regret"), ("swing_regret_100", "sreg"), ("take_regret_100", "treg")):
        v = 100 * g[c].mean(); out[k] = (v, n.reindex(v.index))
    return out


def build_all(t, kind):
    return {**ea.hitter_traits(t), **bk.battrack_traits(t), **seager_traits(t)}


def main():
    pd.set_option("display.width", 270)
    ea.COLS = list(dict.fromkeys(ea.COLS + ["plate_x", "plate_z", "rv_batter", "woba"])); t = bk.load_tracking_tolerant(); t["key"] = t["bat"]
    pv = add_pitch_values(t); t = t.join(pv); print(f"tracking rows {len(t):,}", flush=True)
    st, S = pickle.load(open("output/_battrack_samples.pkl", "rb")); S2 = pickle.load(open("output/_battrack_hrbabip_samples.pkl", "rb"))
    for name, s_ in S.items():
        s_["targets"].update(S2[name]); s_["window"] = eh.window(t, name, s_.get("asof")); s_["traits"] = build_all(s_["window"], "h")
    orig = ea.build; ea.build = lambda tt, kind: seager_traits(tt); st_new = ea.stickiness(t, "h"); ea.build = orig; st = pd.concat([st, st_new.loc[[k for k in SEAGER if k in st_new.index]]]); n0s = st["n0"].to_dict()
    REF = ["swing_pct", "chase_pct", "zswing_pct", "whiff_pct", "called_strike_pct", "two_strike_whiff", "barrel_pct", "avg_ev"]
    print("STICKINESS (alternating games; y/y 2025 -> 2026)\n" + st.loc[SEAGER + REF].round(3).to_string(), flush=True)
    for yr in (2025, 2026):
        w = t[t["year"] == yr]; tr = build_all(w, "h"); df = pd.DataFrame({k: tr[k][0] for k in SEAGER + REF}).dropna(); print(f"\n{yr}: correlations of the decision traits with plate-discipline traits (n {len(df)})\n" + df.corr().loc[SEAGER, REF].round(2).to_string())
    res = {}; tables = {}
    for sname, s in S.items():
        for tg in N0T:
            r = ea.screen(s["targets"][tg], s["traits"], n0s); r["q"] = ea.bh(r["p"].to_numpy()); res[(sname, tg)] = r
    for tg in N0T:
        a, b, pr = res[("ROS25", tg)], res[("ROS26", tg)], res[("PRE", tg)]; k = a.index.intersection(b.index); wa, wb = 1 / a.loc[k, "se"] ** 2, 1 / b.loc[k, "se"] ** 2; pc = (a.loc[k, "coef per SD"] * wa + b.loc[k, "coef per SD"] * wb) / (wa + wb); ps = 1 / np.sqrt(wa + wb)
        T = pd.DataFrame({"t ROS25": a.loc[k, "t"], "t ROS26": b.loc[k, "t"], "t PRE": pr["t"].reindex(k), "pooled coef/SD": pc, "pooled t": pc / ps}); T["pooled q"] = ea.bh(2 * norm.sf(np.abs(T["pooled t"].to_numpy()))); T["same sign"] = np.sign(T["t ROS25"]) == np.sign(T["t ROS26"])
        T = T.join(st[["split-half r", "y/y disattenuated"]]); T["shortlist"] = (T["pooled q"] <= .10) & T["same sign"] & (T["split-half r"] >= .40) & (T["y/y disattenuated"] >= .50); tables[tg] = T
        print(f"\nSCREEN, target {tg} ({len(T)} traits; coef per SD in {tg} units; BH q over all traits)\n" + T.loc[[x for x in SEAGER + REF if x in T.index]].round(4).to_string(), flush=True)
        print("   shortlisted overall: " + (", ".join(T.index[T["shortlist"]]) or "none"), flush=True)
    single = []; _, which = sel.designs("woba", S)
    for tg in N0T:
        for c in SEAGER:
            for fade in (True, False):
                for r in sel.evaluate([c], tg, S, st, N0T, fade, which):
                    single.append({"target": tg, "trait": c, "form": "faded" if fade else "constant", **r})
    A = pd.DataFrame(single); A = A[A["group"] == "all"]; A["rel %"] = 100 * A["dRMSE"] / A["RMSE old"]
    P = A[A["form"] == "constant"].pivot_table(index=["target", "trait"], columns="test", values="rel %"); Q = A[A["form"] == "constant"].pivot_table(index=["target", "trait"], columns="test", values="P(RMSE better)")
    print("\nIMPROVEMENT, single-trait add-ons, constant form: dRMSE as % of production (negative = better)\n" + P.round(2).to_string() + "\nP(RMSE better)\n" + Q.round(2).to_string())
    print("\nFaded form\n" + A[A["form"] == "faded"].pivot_table(index=["target", "trait"], columns="test", values="rel %").round(2).to_string())
    fwd = {}
    for tg in ("woba", "hr", "k", "bb"):
        cands = list(tables[tg].index[tables[tg]["shortlist"]])
        for fade in (True, False):
            L, path = sel.forward(tg, cands, S, st, N0T, fade); fwd[f"{tg}-{'faded' if fade else 'constant'}"] = {"candidates": cands, "selected": L}
            print(f"\nFORWARD SELECTION, {tg}, {'faded' if fade else 'constant'}: candidates {cands}\n   selected: {L if L else 'nothing'}" + (("\n" + pd.DataFrame(path).to_string(index=False)) if path else ""))
    json.dump({"stickiness": st.loc[SEAGER].reset_index().to_dict("records"), "screen": {tg: T.reset_index().to_dict("records") for tg, T in tables.items()}, "improvement": single, "forward": fwd}, open("output/seager_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
