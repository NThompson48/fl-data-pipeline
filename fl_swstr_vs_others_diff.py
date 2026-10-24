"""
fl_swstr_vs_others_diff.py -- biggest disagreements between the per-pitch swinging-strike stuff (swstr_ct, network) and (a) the swing-based network, (b) XGBoost  (user 2026-10-24)
Window = the fold-2 feature window (2025 + 1H2026), pitchers with a stuff level from all three estimators.  Levels z-scored across pitchers (so scales are comparable); disagreement = z(swstr_ct) - z(other).
Per pitcher: the three z levels, own K% and PA, pitch mix (top two types, share), the three ADDITIVE K% projections (each fit on the standard training transition) and the actual 2H2026 K% (where PA_test >= 40).
WHO IS RIGHT: across all pitchers with PA_test >= 40, regress the other model's K% error on the disagreement (positive slope = swstr_ct's extra information is right), weighted by the standard inverse-variance weights; plus the top-25 each way.
Output: output/swstr_vs_others_diff.json / csv
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_k_prior_production_test as pt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import mlb_marcel as mm

warnings.filterwarnings("ignore")
ADD = pp.SPEC["K"]


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); F = {est: pt.frames(raw, est) for est in ("xgb", "torch3", "swstr_ct")}; D = {}
    for est, (tr, n0_k, f2, n0_k2) in F.items():
        d = tr.dropna(subset=["K_rate_actual"] + ADD); w = mm.reliability_weight(d["PA_test"], n0_k).values; m = bt.fit_weighted(d, "K_rate_actual", ADD, w); x = f2.dropna(subset=ADD + ["K_rate_actual"]).copy()
        x["pred"] = m.predict(x).to_numpy(); D[est] = x.set_index("pitcher_id")[["wh_sva_level", "naive_K_rate", "PA", "PA_test", "K_rate_actual", "pred"]]
    ids = sorted(set.intersection(*(set(v.index) for v in D.values()))); t = pd.DataFrame(index=ids)
    for est, nm in (("swstr_ct", "sw"), ("torch3", "net"), ("xgb", "xgb")):
        v = D[est].loc[ids]; t[f"z_{nm}"] = (v["wh_sva_level"] - v["wh_sva_level"].mean()) / v["wh_sva_level"].std(); t[f"pred_{nm}"] = v["pred"]
    t["K_own"] = D["xgb"].loc[ids, "naive_K_rate"]; t["PA"] = D["xgb"].loc[ids, "PA"]; t["PA_test"] = D["xgb"].loc[ids, "PA_test"]; t["K_actual"] = D["xgb"].loc[ids, "K_rate_actual"]
    win = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025), fp.filter_period(raw, bt.START_2026, bt.CUTOFF_2026)], ignore_index=True)
    nm = win.drop_duplicates("pitcher_id").set_index("pitcher_id")["pitcher"]; mix = win.groupby(["pitcher_id", "pitch_type"]).size().unstack(fill_value=0); mixs = mix.div(mix.sum(axis=1), axis=0)
    top2 = mixs.apply(lambda r: ", ".join(f"{k} {100 * v:.0f}%" for k, v in r.nlargest(2).items()), axis=1); t.insert(0, "pitcher", nm.reindex(ids).to_numpy()); t["mix"] = top2.reindex(ids).to_numpy(); t["velo"] = win.groupby("pitcher_id")["effective_velo"].mean().reindex(ids).to_numpy()
    t["d_net"] = t["z_sw"] - t["z_net"]; t["d_xgb"] = t["z_sw"] - t["z_xgb"]; pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 40)
    print(f"{len(t)} pitchers; correlations of z levels:\n{t[['z_sw', 'z_net', 'z_xgb']].corr().round(3).to_string()}\nSD of disagreement (z units): swstr-net {t['d_net'].std():.3f}, swstr-xgb {t['d_xgb'].std():.3f}")
    cols = ["pitcher", "mix", "velo", "PA", "z_sw", "z_net", "z_xgb", "K_own", "pred_sw", "pred_net", "pred_xgb", "K_actual", "PA_test"]
    for col, lab in (("d_net", "swstr_ct vs the swing network"), ("d_xgb", "swstr_ct vs XGBoost")):
        big = t[t["PA"] >= 40]; hi = big.nlargest(10, col)[cols].round(3); lo = big.nsmallest(10, col)[cols].round(3)
        print(f"\n== {lab}: swstr_ct HIGHER (PA >= 40)\n{hi.to_string()}\n== {lab}: swstr_ct LOWER\n{lo.to_string()}")
    o = t[t["PA_test"] >= 40].copy(); y = o["K_actual"].to_numpy(); base = o["pred_xgb"].to_numpy(); w = ew.weights_from_base(y, base, o["PA_test"].to_numpy())[0]; res = {}
    print(f"\nWHO IS RIGHT (n = {len(o)} with PA_test >= 40; level-free; frozen weights)")
    for other, col in (("net", "d_net"), ("xgb", "d_xgb")):
        e = y - o[f"pred_{other}"].to_numpy(); e = e - np.average(e, weights=w); dd = o[col].to_numpy(); dd = dd - np.average(dd, weights=w); slope = float(np.sum(w * dd * e) / np.sum(w * dd ** 2)); se = float(np.sqrt(np.sum(w * (e - slope * dd) ** 2) / (len(o) - 2) / np.sum(w * dd ** 2)))
        rows = []
        for lab, sel in (("top 25 swstr_ct higher", o[col].nlargest(25).index), ("top 25 swstr_ct lower", o[col].nsmallest(25).index)):
            s = o.loc[sel]; rows.append({"group": lab, "mean disagreement (z)": s[col].mean(), "actual K%": s["K_actual"].mean(), f"{other} pred": s[f"pred_{other}"].mean(), "swstr_ct pred": s["pred_sw"].mean(),
                                         f"|err| {other}": np.abs(s["K_actual"] - s[f"pred_{other}"]).mean(), "|err| swstr_ct": np.abs(s["K_actual"] - s["pred_sw"]).mean()})
        print(f"\nvs {other}: slope of the {other} K% error on the disagreement = {slope:+.4f} (SE {se:.4f}, t {slope / se:+.1f}; positive = swstr_ct's extra information is right; 1 z of disagreement ~ {100 * slope:+.2f} K points)\n" + pd.DataFrame(rows).round(4).to_string(index=False)); res[other] = {"slope": slope, "se": se, "rows": rows}
    t.to_csv("output/swstr_vs_others_diff.csv"); pd.Series({k: v for k, v in res.items()}).to_json("output/swstr_vs_others_diff.json")


if __name__ == "__main__":
    main()
