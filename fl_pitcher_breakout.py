"""
fl_pitcher_breakout.py -- PITCHER changepoints and a model fitted on big upside misses  (user 2026-10-28: "I think the pitcher version might be more fruitful"; hitter version: fl_hitter_changepoint_upside.py)
Samples = the project's pitcher error analysis (fl_error_analysis_pitchers.samples, REBUILT on the refreshed data with the production network stuff estimator): PRE (frozen K% / BB% model, fold 2), RA9 (preseason RA9 2025 -> 2026), ROS25 / ROS26 (production ROS pitcher model as of mid-season -> rest of season); level-free residuals, frozen standard weights;
traits measured only in the feature window and shrunk by their own reliability.  The pitcher tracking columns dropped by the 2026-10-26 hitting pull (velocity, spin, break, release, extension) are merged back from the pitch cache.
CHANGEPOINT FEATURES per pitcher from his chronological pitches in the feature window (series standardised WITHIN pitch type so mix changes do not masquerade as skill changes): velocity, spin, induced vertical break, arm-side horizontal break, release height, release side, extension, per-pitch STUFF (the production network swstr_ct value) and whiff per swing:
  cp_t (signed t of the best single split), cp_shift (post - pre, SD units), cp_recent (post-split mean - window mean), rec25 (last 25% - window mean; fixed split, the control for the search).  9 series x 4 = 36 features.
PART 1 SCREEN (as the hitter version; BH q over all traits; K%, BB% pooled ROS25 / ROS26 with PRE as a check, RA9 on RA9-pre and ROS26).   PART 2 UPSIDE MODEL: breakout = the top 20% of the residual in the GOOD direction (more K%, fewer walks, lower RA9); ridge logistic on the shrunk traits + log history + the prediction,
fit on one sample and scored on another, with / without the changepoint features, vs the history + prediction baseline; AUC for the upside and the downside tail, hit rate and mean surprise of the 15 pitchers flagged most strongly (base rate .20).
Usage: python fl_pitcher_breakout.py     (log: output/pitcher_breakout_log.txt)     Output: output/pitcher_breakout.json
"""
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import fl_error_analysis_common as ea
import fl_error_analysis_pitchers as epi
import fl_halfseason_backtest as bt
import fl_hitter_battrack_test as bk
import fl_hitter_changepoint_upside as hcu
import fl_pitcher_features as fp
import fl_position_player_flag as ppf

warnings.filterwarnings("ignore")
CACHE = "output/_pitcher_breakout_samples.pkl"
SERIES = ["velo", "spin", "ivb", "hb", "relz", "rels", "ext", "stuff", "whiff"]; MIN_N = {s: 150 for s in SERIES}; MIN_N["whiff"] = 60
SIGN = {"k": 1.0, "bb": -1.0, "ra9": -1.0}                    # +1 = a higher residual is the good direction


def tracking_with_pitcher_columns():
    """tracking table (hitting-pull columns) + the pitcher-side attributes of the pitch cache, matched per pitch on (game, PA, pitch type, plate location): the tracking table's pa_pitch_count is the PA's pitch total, not a pitch index"""
    ea.COLS = list(dict.fromkeys(ea.COLS + ["rel_speed", "plate_x", "plate_z"])); t = bk.load_tracking_tolerant(); raw = fp.load_fl_pitches_cached()
    cols = [c for c in ("effective_velo", "ivb", "spin_rate", "extension", "horz_break_as", "release_pos_z", "rel_side_as") if c in raw.columns]
    r = raw[["game_id", "pa_id", "pitch_type", "plate_x", "plate_z"] + cols].dropna(subset=["plate_x", "plate_z"]).copy(); r["px"] = r["plate_x"].round(3); r["pz"] = r["plate_z"].round(3); r = r.drop_duplicates(["game_id", "pa_id", "pitch_type", "px", "pz"]).drop(columns=["plate_x", "plate_z"]).rename(columns={"horz_break_as": "hb_as_", "release_pos_z": "rel_height_"})
    t["px"] = t["plate_x"].round(3); t["pz"] = t["plate_z"].round(3); drop = [c for c in ("effective_velo", "ivb", "spin_rate", "extension", "hb_as", "rel_height") if c in t.columns]
    t = t.drop(columns=drop).merge(r, on=["game_id", "pa_id", "pitch_type", "px", "pz"], how="left")
    t["hb_as"] = t["hb_as_"]; t["rel_height"] = t["rel_height_"]
    for c in ("arm_angle", "vert_appr_angle"):
        if c not in t.columns:
            t[c] = np.nan
    t["key"] = t["pit"]; print(f"tracking {len(t):,} rows; pitcher attributes matched on {t['effective_velo'].notna().mean():.1%} of rows", flush=True); return t, raw


def window_raw(raw, name, asof):
    if name == "PRE":
        return pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025), fp.filter_period(raw, bt.START_2026, bt.CUTOFF_2026)], ignore_index=True)
    if name == "RA9":
        return fp.filter_period(raw, bt.START_2025, bt.END_2025)
    yr = 2000 + int(name[-2:]); s, e = (bt.START_2025, "2025-12-31") if yr == 2025 else (bt.START_2026, "2026-12-31"); return fp.filter_period(raw, s, (pd.Timestamp(asof) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))


def cp_pitcher(w, id2name):
    """changepoint features per pitcher from a window of the pitch cache (w): {trait: (Series by name key, n Series)}"""
    w = w.sort_values(["pitcher_id", "date", "game_id", "pa_id", "pitch_no"], kind="mergesort").reset_index(drop=True)
    cols = {"velo": "effective_velo", "spin": "spin_rate", "ivb": "ivb", "hb": "horz_break_as", "relz": "release_pos_z", "rels": "rel_side_as", "ext": "extension"}
    for s, c in cols.items():
        if c in w.columns:
            g = w.groupby("pitch_type")[c]; w[s] = (w[c] - g.transform("mean")) / g.transform("std")
    w["pid"] = w["pitcher_id"].astype(str)
    if "stuff" in SERIES:
        sva = fp.compute_pooled_wh_sva(w, estimator="swstr_ct"); sva = sva.copy(); sva["pitcher_id"] = sva["pitcher_id"].astype(str)
        w = w.merge(sva[["pitcher_id", "date", "game_id", "pa_id", "pitch_no", "sva"]].rename(columns={"pitcher_id": "pid"}), on=["pid", "date", "game_id", "pa_id", "pitch_no"], how="left"); w["stuff"] = w["sva"]
    swing = w["pitch_call"].isin(fp._STRIKE_CALLS - {"strike_called"}) if hasattr(fp, "_STRIKE_CALLS") else w["is_swing"]
    w["whiff"] = np.where(w["pitch_call"].isin(["strike_swinging"]), 1.0, np.where(w["pitch_call"].isin(["strike_swinging", "foul_not_fieldable", "foul_fieldable", "foul_ball", "in_play"]), 0.0, np.nan))
    out = {f"{s}_{f}": {} for s in SERIES for f in ("cp_t", "cp_shift", "cp_recent", "rec25")}; ns = {s: {} for s in SERIES}
    for s in SERIES:
        if s not in w.columns:
            continue
        d = w[w[s].notna()]
        for pid, g in d.groupby("pid"):
            r = hcu.cp_one(g[s].to_numpy(float), MIN_N[s])
            if r is None:
                continue
            nm = id2name.get(pid)
            if nm is None:
                continue
            for f in ("cp_t", "cp_shift", "cp_recent", "rec25"):
                out[f"{s}_{f}"][nm] = r[f]
            ns[s][nm] = r["n"]
    return {name: (pd.Series(v, dtype=float), pd.Series(ns[name.split("_")[0]], dtype=float).reindex(pd.Series(v, dtype=float).index)) for name, v in out.items()}


def main():
    pd.set_option("display.width", 250); t, rawc = tracking_with_pitcher_columns(); raw = ppf.exclude(rawc); fp.WH_SVA_ESTIMATOR = "swstr_ct"
    import fl_production_projection as pp; pp.BB_SPEC_BLENDED = ["blended_BB_rate", "command_score"]      # the PUBLISHED BB% spec (raw Command Score), whatever the working tree holds
    id2name = t.drop_duplicates("pitcher_id").set_index(t.drop_duplicates("pitcher_id")["pitcher_id"].astype(str))["pit"].to_dict()
    tt = t.copy(); tt["key"] = tt["pitcher_id"]
    if os.path.exists(CACHE):
        st, S, n0t = pickle.load(open(CACHE, "rb"))
    else:
        st = ea.stickiness(tt, "p"); S, n0t = epi.samples(t); pickle.dump((st, {k: {kk: vv for kk, vv in v.items() if kk != "window"} for k, v in S.items()}, n0t), open(CACHE, "wb"))
    n0s = st["n0"].to_dict(); print("samples:", {k: {tg: len(F) for tg, F in v["targets"].items()} for k, v in S.items()}, flush=True)
    for name, s in S.items():
        w = epi.window(t, name, s.get("asof")).copy(); w["key"] = w[s["key"]]; s["window"] = w; s["traits"] = ea.build(w, "p")
        # changepoint features: keyed like the sample (pitcher_id for PRE / RA9, clean name for ROS)
        cp = cp_pitcher(window_raw(raw, name, s.get("asof")), id2name if s["key"] == "pit" else {k: k for k in id2name})
        s["traits"].update(cp); print(f"  {name}: traits {len(s['traits'])}", flush=True)
    CP = [k for k in S["ROS25"]["traits"] if k.split("_")[0] in SERIES and k.split("_", 1)[1] in ("cp_t", "cp_shift", "cp_recent", "rec25")]; OLD = [k for k in S["ROS25"]["traits"] if k not in CP]
    for k in CP:
        n0s[k] = 0.0
    print(f"{len(OLD)} earlier traits, {len(CP)} changepoint features", flush=True)
    res = {}; tables = {}
    for sname, s in S.items():
        for tg, F in s["targets"].items():
            r = ea.screen(F, s["traits"], n0s); r["q"] = ea.bh(r["p"].to_numpy()); res[(sname, tg)] = r
    pooled = {"k": ("ROS25", "ROS26", "PRE"), "bb": ("ROS25", "ROS26", "PRE"), "ra9": ("RA9", "ROS26", None)}
    for tg, (sa, sb, sc) in pooled.items():
        a, b = res[(sa, tg)], res[(sb, tg)]; k = a.index.intersection(b.index); wa, wb = 1 / a.loc[k, "se"] ** 2, 1 / b.loc[k, "se"] ** 2; pc = (a.loc[k, "coef per SD"] * wa + b.loc[k, "coef per SD"] * wb) / (wa + wb); ps = 1 / np.sqrt(wa + wb)
        T = pd.DataFrame({f"t {sa}": a.loc[k, "t"], f"t {sb}": b.loc[k, "t"], "pooled coef/SD": pc, "pooled t": pc / ps}); T["pooled q"] = ea.bh(2 * norm.sf(np.abs(T["pooled t"].to_numpy()))); T["same sign"] = np.sign(T[f"t {sa}"]) == np.sign(T[f"t {sb}"])
        if sc:
            T["t PRE"] = res[(sc, tg)]["t"].reindex(k)
        tables[tg] = T; cpt = T.loc[[x for x in CP if x in T.index]].sort_values("pooled t", key=np.abs, ascending=False)
        print(f"\nSCREEN, target {tg} ({'positive = more than projected' if tg != 'ra9' else 'positive = more runs than projected'}): changepoint features (BH q over {len(T)} traits)\n" + cpt.head(8).round(4).to_string(), flush=True)
        print("   changepoint features with pooled q <= .10 and the same sign: " + (", ".join(cpt.index[(cpt["pooled q"] <= .10) & cpt["same sign"]]) or "none"), flush=True)
        oldt = T.loc[[x for x in OLD if x in T.index]].sort_values("pooled t", key=np.abs, ascending=False); print("   (earlier traits, for reference) " + "; ".join(f"{i} t {v:+.1f} q {q:.2f}" for i, v, q in zip(oldt.index[:5], oldt["pooled t"][:5], oldt["pooled q"][:5])), flush=True)
    # ---- Part 2
    def flip(F, tg):
        F2 = F.copy(); F2["e"] = SIGN[tg] * F["e"]; return F2
    designs = {"k": [("ROS25", "ROS26"), ("ROS26", "ROS25"), ("ROS25", "PRE")], "bb": [("ROS25", "ROS26"), ("ROS26", "ROS25"), ("ROS25", "PRE")], "ra9": [("RA9", "ROS26"), ("ROS26", "RA9")]}; out = []
    for tg in ("k", "bb", "ra9"):
        for lab, names in (("baseline: history + prediction only", []), ("earlier traits", OLD), ("earlier traits + changepoint features", OLD + CP), ("changepoint features only", CP)):
            for a, b in designs[tg]:
                Fa, Fb = flip(S[a]["targets"][tg], tg), flip(S[b]["targets"][tg], tg); Xa, ua = hcu.design(Fa, S[a]["traits"], n0s, names); Xb, ub = hcu.design(Fb, S[b]["traits"], n0s, names); ua_, ub_ = ua + ["lh", "pc"], ub + ["lh", "pc"]; common = [k for k in ua_ if k in ub_]
                Xa = Xa[:, [ua_.index(k) for k in common]]; Xb = Xb[:, [ub_.index(k) for k in common]]; yua, yda = hcu.tails(Fa); yub, ydb = hcu.tails(Fb); pu, _ = hcu.fit_score(Xa, yua, Xb, 0.03); pdn, _ = hcu.fit_score(Xa, yda, Xb, 0.03)
                m = hcu.metrics(Fb, pu, pdn, yub, ydb); out.append({"target": tg, "model": lab, "test": f"{a} -> {b}", **{k: v for k, v in m.items() if not k.startswith("top-15 mean") and k != "all mean"}, "top-15 mean surprise (SD of residual)": float(Fb["e"].to_numpy()[np.argsort(-pu)[:15]].mean() / Fb["e"].std()), "all mean (SD)": float(Fb["e"].mean() / Fb["e"].std())})
            if tg != "ra9":
                F = flip(S["PRE"]["targets"][tg], tg); X, u = hcu.design(F, S["PRE"]["traits"], n0s, names); yu, yd = hcu.tails(F); pu = np.zeros(len(F)); pdn = np.zeros(len(F))
                for rep in range(5):
                    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, yu):
                        pu[te] += hcu.fit_score(X[tr], yu[tr], X[te], 0.03)[0] / 5; pdn[te] += hcu.fit_score(X[tr], yd[tr], X[te], 0.03)[0] / 5
                m = hcu.metrics(F, pu, pdn, yu, yd); out.append({"target": tg, "model": lab, "test": "CV within PRE", **{k: v for k, v in m.items() if not k.startswith("top-15 mean") and k != "all mean"}, "top-15 mean surprise (SD of residual)": float(F["e"].to_numpy()[np.argsort(-pu)[:15]].mean() / F["e"].std()), "all mean (SD)": float(F["e"].mean() / F["e"].std())})
    O = pd.DataFrame(out); print("\nUPSIDE MODEL (breakout = the top 20% of the residual in the GOOD direction)\n" + O.round(3).to_string(index=False)); g = O.groupby(["target", "model"]).mean(numeric_only=True).round(3); print("\nMEAN OVER THE TESTS\n" + g.to_string())
    for tg in ("k",):
        D_ = {s: hcu.design(flip(S[s]["targets"][tg], tg), S[s]["traits"], n0s, OLD + CP) for s in ("ROS25", "ROS26")}; nm_ = {s: D_[s][1] + ["lh", "pc"] for s in D_}; common = [k for k in nm_["ROS25"] if k in nm_["ROS26"]]
        Xs = [D_[s][0][:, [nm_[s].index(k) for k in common]] for s in D_]; ys = [hcu.tails(flip(S[s]["targets"][tg], tg))[0] for s in D_]; u = common[:-2]
        m = LogisticRegression(C=0.03, max_iter=2000).fit(np.vstack(Xs), np.concatenate(ys)); co = pd.Series(m.coef_[0], index=common).sort_values(key=np.abs, ascending=False); print(f"\nLARGEST COEFFICIENTS of the pooled ROS25 + ROS26 upside model ({tg}; standardised features)\n" + co.head(12).round(3).to_string())
    json.dump({"upside": out, "screen": {tg: T.reset_index().to_dict("records") for tg, T in tables.items()}}, open("output/pitcher_breakout.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
