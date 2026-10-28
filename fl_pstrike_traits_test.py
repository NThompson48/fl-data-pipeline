"""
fl_pstrike_traits_test.py -- do a pitcher's STRIKE-NETWORK measures in the feature window predict his next projection miss?  (user 2026-10-28: "sure!" to testing the expected strike rate and the residual as error-analysis traits)
Measures per pitcher window, from the cross-fitted P(strike | stuff, location, count, handedness) network (output/pstrike_pitch_level.parquet; fl_pstrike_network.py):
  exp_strike     mean predicted strike probability (what his pitches earn: stuff + location + count + handedness)      loc_expected  the same from the network without stuff (location + count + handedness)
  stuff_value    exp_strike - loc_expected (what the stuff adds)                                                          strike_resid  actual strike rate minus exp_strike (strikes beyond what the pitches earn: deception / sequencing / framing / luck)
Each is shrunk by its own split-half reliability (alternating games, 2025 and 2026 seasons; n0 in decision pitches).  Design = the pitcher error analysis (fl_error_analysis_pitchers; the rebuilt samples cached by fl_pitcher_breakout.py: PRE 229, RA9 311, ROS25 204, ROS26 226 pitchers): screen (residual ~ z(trait) +
centred prediction + log history, player bootstrap; pooled ROS25 / ROS26 for K% and BB%, RA9 on RA9-pre and ROS26; BH q over the 35 earlier traits + these 4) and held-out add-ons (b fit on one sample, scored on another: ROS25 <-> ROS26, ROS25 -> PRE, CV in PRE; faded and constant forms; weighted RMSE / MAE, frozen weights, player bootstrap).
Output: output/pstrike_traits_log.txt
"""
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import fl_error_analysis_common as ea
import fl_error_analysis_pitchers as epi
import fl_error_analysis_select as sel
import fl_pitcher_breakout as fpb

warnings.filterwarnings("ignore")
NEW = ["exp_strike", "loc_expected", "stuff_value", "strike_resid"]


def strike_traits(d, keymap):
    """d: decision pitches of the window with p_full, p_loc, target; returns {trait: (Series by key, n Series)}"""
    d = d.assign(key=d["pid"].map(keymap)).dropna(subset=["key"]); g = d.groupby("key"); n = g.size().astype(float)
    return {"exp_strike": (g["p_full"].mean(), n), "loc_expected": (g["p_loc"].mean(), n), "stuff_value": ((d["p_full"] - d["p_loc"]).groupby(d["key"]).mean(), n), "strike_resid": ((d["target"] - d["p_full"]).groupby(d["key"]).mean(), n)}


def split_half_n0(pop):
    out = {}
    for yr in (2025, 2026):
        y = pop[pop["year"] == yr]; gm = y[["pid", "game_id", "date"]].drop_duplicates(["pid", "game_id"]).sort_values("date"); gm["half"] = gm.groupby("pid").cumcount() % 2; y = y.merge(gm[["pid", "game_id", "half"]], on=["pid", "game_id"])
        A = strike_traits(y[y["half"] == 0], {k: k for k in y["pid"].unique()}); B = strike_traits(y[y["half"] == 1], {k: k for k in y["pid"].unique()})
        for k in NEW:
            df = pd.DataFrame({"a": A[k][0], "na": A[k][1], "b": B[k][0], "nb": B[k][1]}).dropna(); df = df[(df["na"] >= 100) & (df["nb"] >= 100)]; r = np.corrcoef(df["a"], df["b"])[0, 1]; nh = float(((df["na"] + df["nb"]) / 2).mean()); out.setdefault(k, []).append((r, nh, nh * (1 - r) / r if r > 0.02 else np.inf, len(df)))
    return {k: {"split-half r": float(np.mean([x[0] for x in v])), "n0": float(np.nanmedian([x[2] for x in v])), "units per half": float(np.mean([x[1] for x in v])), "pitchers": int(np.mean([x[3] for x in v]))} for k, v in out.items()}


def main():
    pd.set_option("display.width", 250); t, rawc = fpb.tracking_with_pitcher_columns(); id2name = t.drop_duplicates("pitcher_id").set_index(t.drop_duplicates("pitcher_id")["pitcher_id"].astype(str))["pit"].to_dict()
    pop = pd.read_parquet("output/pstrike_pitch_level.parquet"); pop["pid"] = pop["pitcher_id"].astype(str); pop["date"] = pd.to_datetime(pop["date"])
    st, S, n0t = pickle.load(open(fpb.CACHE, "rb")); sh = split_half_n0(pop); print("RELIABILITY of the strike-network measures (alternating games; 2025, 2026 averaged)\n" + pd.DataFrame(sh).T.round(3).to_string(), flush=True)
    for k in NEW:
        st.loc[k, "n0"] = sh[k]["n0"]
    import fl_halfseason_backtest as bt
    for name, s in S.items():
        w = epi.window(t, name, s.get("asof")).copy(); w["key"] = w[s["key"]]; s["traits"] = ea.build(w, "p")
        if name == "PRE":
            d = pop[(pop["year"] == 2025) | ((pop["year"] == 2026) & (pop["date"] <= pd.Timestamp(bt.CUTOFF_2026)))]
        elif name == "RA9":
            d = pop[pop["year"] == 2025]
        else:
            yr = 2000 + int(name[-2:]); d = pop[(pop["year"] == yr) & (pop["date"] < pd.Timestamp(s["asof"]))]
        s["traits"].update(strike_traits(d, id2name if s["key"] == "pit" else {k: k for k in id2name}))
    n0s = st["n0"].to_dict(); res = {}
    for sname, s in S.items():
        for tg, F in s["targets"].items():
            r = ea.screen(F, s["traits"], n0s); r["q"] = ea.bh(r["p"].to_numpy()); res[(sname, tg)] = r
    for tg, (sa, sb, sc) in {"k": ("ROS25", "ROS26", "PRE"), "bb": ("ROS25", "ROS26", "PRE"), "ra9": ("RA9", "ROS26", None)}.items():
        a, b = res[(sa, tg)], res[(sb, tg)]; k = a.index.intersection(b.index); wa, wb = 1 / a.loc[k, "se"] ** 2, 1 / b.loc[k, "se"] ** 2; pc = (a.loc[k, "coef per SD"] * wa + b.loc[k, "coef per SD"] * wb) / (wa + wb); ps = 1 / np.sqrt(wa + wb)
        T = pd.DataFrame({f"t {sa}": a.loc[k, "t"], f"t {sb}": b.loc[k, "t"], "pooled coef/SD": pc, "pooled t": pc / ps}); T["pooled q"] = ea.bh(2 * norm.sf(np.abs(T["pooled t"].to_numpy()))); T["same sign"] = np.sign(T[f"t {sa}"]) == np.sign(T[f"t {sb}"])
        if sc:
            T["t PRE"] = res[(sc, tg)]["t"].reindex(k)
        print(f"\nSCREEN, target {tg} ({'positive = more than projected' if tg != 'ra9' else 'positive = more runs than projected'}; BH q over {len(T)} traits)\n" + T.loc[[x for x in NEW if x in T.index]].round(4).to_string(), flush=True)
        top = T.sort_values("pooled t", key=np.abs, ascending=False).head(4); print("   top traits overall: " + "; ".join(f"{i} t {v:+.1f} q {q:.2f}" for i, v, q in zip(top.index, top["pooled t"], top["pooled q"])), flush=True)
    _, which = sel.designs("k", S); rows = []
    sets = {"strike residual": ["strike_resid"], "expected strike rate": ["exp_strike"], "location-expected": ["loc_expected"], "stuff value": ["stuff_value"], "strike residual + expected": ["strike_resid", "exp_strike"], "fastball velocity (C1)": ["fb_velo"], "fb velocity + strike residual": ["fb_velo", "strike_resid"]}
    for tg in ("k", "bb"):
        for lab, L in sets.items():
            for fade in (True, False):
                for r in sel.evaluate(L, tg, S, st, n0t, fade, which):
                    rows.append({"target": tg, "set": lab, "form": "faded" if fade else "constant", **r})
    A = pd.DataFrame(rows); A = A[A["group"] == "all"].copy(); A["rel %"] = 100 * A["dRMSE"] / A["RMSE old"]
    for tg in ("k", "bb"):
        for form in ("constant", "faded"):
            T = A[(A["target"] == tg) & (A["form"] == form)]; print(f"\nTARGET {tg.upper()} ({form}): dRMSE as % of production (negative = better); P(RMSE better)\n" + T.pivot_table(index="set", columns="test", values="rel %", sort=False).round(2).to_string() + "\n" + T.pivot_table(index="set", columns="test", values="P(RMSE better)", sort=False).round(2).to_string())
    A.to_csv("output/pstrike_traits_improvement.csv", index=False)


if __name__ == "__main__":
    main()
