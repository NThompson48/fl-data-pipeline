"""
fl_hitter_battrack_test.py -- do the user's ESTIMATED BAT-TRACKING measures help the hitter projection?  (user 2026-10-27: "pull from some of my bat tracking estimation work to see if it helps the hitter model")
The measures are the swing-suite estimators (swing-suite/update_data.py, used unchanged), computed per hitter from his tracked balls in play (exit speed, launch angle, spray direction, pitch speed):
  bat_speed_est  estimated bat speed from exit speed and pitch speed (energy-transfer model, 15 hardest / softest balls anchor the efficiency scale)
  paa            Peak Attack Angle: the launch angle at which his exit speed peaks (quadratic envelope on the top 20% per 3-degree bin, averaged with the top-15 mean)
  pad_lac        launch-angle-corrected Peak Attack Direction: his pull spray expected at his peak attack angle (regression of spray on launch angle, evaluated at PAA)
  under_pct      share of balls in play launched above 40 degrees (under the ball)
References in the same screen: the project's measured bat_speed (tracking 'batspeed'), average EV, hard-hit%, barrel%, mean launch angle, popup%, fly-ball%.
DESIGN = the project's hitter error analysis (fl_error_analysis_common.py, pre-specified): samples PRE (preseason path on 2025 -> 2026), ROS25, ROS26 (production rest-of-season model as of mid-season -> rest of season), targets wOBA, K%, BB%;
level-free residuals, frozen standard weights; traits measured only in each sample's feature window and shrunk by their own reliability.
  (1) STICKINESS  split-half reliability (alternating games) and 2025 -> 2026 disattenuated.   (2) SCREEN  residual ~ z(trait) + centred prediction + log history; pooled across ROS25 / ROS26, BH q over ALL traits (the 29 earlier + these 4).
  (3) IMPROVEMENT (every new trait, shortlisted or not): faded and constant add-on, b fit on one season and scored on the other (ROS25 <-> ROS26), transferred to PRE, and 10-fold CV inside PRE; weighted RMSE and MAE, frozen weights, player bootstrap; overall / low-history / rookies.
Usage: python fl_hitter_battrack_test.py      (log: output/battrack_test_log.txt)    Output: output/battrack_test.json
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "swing-suite"))
import update_data as ud  # noqa: E402  (the user's estimators; importing only attaches a log handler)

import fl_error_analysis_common as ea  # noqa: E402
import fl_error_analysis_hitters as eh  # noqa: E402
import fl_error_analysis_select as sel  # noqa: E402

warnings.filterwarnings("ignore")
NEW = ["bat_speed_est", "paa", "pad_lac", "under_pct"]
REF = ["bat_speed", "avg_ev", "hardhit_pct", "barrel_pct", "la_mean", "pu_pct", "fb_pct"]


def battrack_traits(t):
    """{trait: (value Series by key, n BBE Series)} with the swing-suite estimators."""
    b = t[t["is_bip"] & t["direction"].between(*ud.DIR_RANGE) & t["rel_speed"].notna() & t["batter_side"].isin(["left", "right"])]
    out = {k: {} for k in NEW}; n = {}
    for key, g in b.groupby("key"):
        if len(g) < 12:
            continue
        la = g["angle"].to_numpy(float); ev = g["exit_speed"].to_numpy(float); rel = g["rel_speed"].to_numpy(float)
        sp = np.where(g["batter_side"].to_numpy() == "right", -g["direction"].to_numpy(float), g["direction"].to_numpy(float))
        paa = ud._envelope_peak_arrays(la, ev, ud.ANGLE_MIN, ud.ANGLE_MAX)
        out["bat_speed_est"][key] = ud.estimate_bat_speed_arrays(ev, rel); out["paa"][key] = paa; out["pad_lac"][key] = ud._la_corrected_pad_arrays(la, sp, paa); out["under_pct"][key] = float((la > 40).mean()); n[key] = len(g)
    ns = pd.Series(n, dtype=float); res = {}
    for k in NEW:
        v = pd.Series(out[k], dtype=float).dropna(); res[k] = (v, ns.reindex(v.index))
    return res


def build_all(t, kind):
    return {**ea.hitter_traits(t), **battrack_traits(t)}


def load_tracking_tolerant():
    """ea.load_tracking(), but tolerant of the pitcher-side columns the 2026-10-26 hitting pull no longer carries (effective_velo, ivb, horz_break, spin_rate, extension, rel_height, arm_angle: only the pitcher traits and the
    hitter trait fb_velo_faced use them) and with rel_speed added for the bat-speed estimator."""
    import pyarrow.parquet as pq
    have = set(pq.read_schema(ea.TRACK).names); orig = pd.read_parquet; ea.COLS = list(dict.fromkeys(ea.COLS + ["rel_speed"]))

    def rp(path, columns=None, **kw):
        if path != ea.TRACK:
            return orig(path, columns=columns, **kw)
        df = orig(path, columns=[c for c in columns if c in have], **kw)
        for c in columns:
            if c not in df:
                df[c] = np.nan
        return df
    pd.read_parquet = rp
    try:
        return ea.load_tracking()
    finally:
        pd.read_parquet = orig


def main():
    t = load_tracking_tolerant(); t["key"] = t["bat"]; print(f"tracking rows {len(t):,}; balls in play with pitch speed {int((t['is_bip'] & t['rel_speed'].notna()).sum()):,}", flush=True)
    CACHE = "output/_battrack_samples.pkl"; orig_build = ea.build; ea.build = lambda tt, kind: build_all(tt, kind)
    if os.path.exists(CACHE):
        st, S = pickle.load(open(CACHE, "rb"))
    else:
        st = ea.stickiness(t, "h"); S = eh.samples(t); pickle.dump((st, {k: {kk: vv for kk, vv in v.items() if kk != "window"} for k, v in S.items()}), open(CACHE, "wb"))
    ea.build = orig_build; print("STICKINESS (alternating games; y/y 2025 -> 2026)\n" + st.loc[NEW + REF].round(3).to_string(), flush=True)
    n0s = st["n0"].to_dict()
    for name, s_ in S.items():
        s_["window"] = eh.window(t, name, s_.get("asof")); s_["traits"] = build_all(s_["window"].assign(key=s_["window"]["bat"]), "h")
    res = {}
    for sname, s in S.items():
        for tg, F in s["targets"].items():
            r = ea.screen(F, s["traits"], n0s); r["q"] = ea.bh(r["p"].to_numpy()); res[(sname, tg)] = r
    tables = {}
    for tg in eh.N0_T:
        a, b, pr = res[("ROS25", tg)], res[("ROS26", tg)], res[("PRE", tg)]; k = a.index.intersection(b.index)
        wa, wb = 1 / a.loc[k, "se"] ** 2, 1 / b.loc[k, "se"] ** 2; pc = (a.loc[k, "coef per SD"] * wa + b.loc[k, "coef per SD"] * wb) / (wa + wb); ps = 1 / np.sqrt(wa + wb)
        T = pd.DataFrame({"t ROS25": a.loc[k, "t"], "t ROS26": b.loc[k, "t"], "t PRE": pr["t"].reindex(k), "pooled coef/SD": pc, "pooled t": pc / ps}); T["pooled q"] = ea.bh(2 * norm.sf(np.abs(T["pooled t"].to_numpy()))); T["same sign"] = np.sign(T["t ROS25"]) == np.sign(T["t ROS26"])
        T = T.join(st[["split-half r", "y/y disattenuated"]]); T["shortlist"] = (T["pooled q"] <= .10) & T["same sign"] & (T["split-half r"] >= .40) & (T["y/y disattenuated"] >= .50); tables[tg] = T
        show = T.loc[[x for x in NEW + REF if x in T.index]]; print(f"\nSCREEN, target {tg} (n traits {len(T)}; coef per SD in {tg} units; BH q over all traits)\n" + show.round(4).to_string(), flush=True)
        print("   shortlisted overall: " + (", ".join(T.index[T["shortlist"]]) or "none"), flush=True)
    # improvement tests for every new trait
    single = []; n0t = eh.N0_T; _, which = sel.designs("woba", S); pd.set_option("display.width", 260)
    for tg in eh.N0_T:
        for c in NEW:
            for fade in (True, False):
                for r in sel.evaluate([c], tg, S, st, n0t, fade, which):
                    single.append({"target": tg, "trait": c, "form": "faded" if fade else "constant", **r})
    A = pd.DataFrame(single); print("\nIMPROVEMENT, single-trait add-ons (dRMSE < 0 = better; overall rows)\n" + A[A["group"] == "all"].drop(columns=["group", "RMSE old", "MAE old"]).round(5).to_string(index=False))
    json.dump({"stickiness": st.loc[NEW + REF].reset_index().to_dict("records"), "screen": {tg: T.reset_index().to_dict("records") for tg, T in tables.items()}, "improvement": single}, open("output/battrack_test.json", "w"), indent=1, default=float)
    # how the estimates relate to the measured bat speed and EV90 in the same season (context)
    for yr in (2025, 2026):
        w = t[t["year"] == yr]; tr = build_all(w, "h"); df = pd.DataFrame({k: tr[k][0] for k in NEW + ["bat_speed", "avg_ev", "barrel_pct"]}).dropna(); print(f"\n{yr}: correlations among the estimates and measured traits (n {len(df)})\n" + df.corr().round(2).to_string())


if __name__ == "__main__":
    main()
