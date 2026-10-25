"""
fl_strike_stuff_test.py -- P(strike | stuff)  (user 2026-10-25: "What about P(strike|stuff)?")
MODEL: P(strike | 7 stuff inputs: velocity, ivb, arm-side horizontal break, spin, extension, release height, release side) -- no location, count or pitch type, no baseline subtracted -- network (3 seeds, MLP 128-64-32) and XGBoost reference, cross-fitted by pitcher
(5-fold GroupKFold).  strike = called strike, swinging strike or foul; ball = called ball or ball in dirt; balls in play, hit batters, intentional and automatic calls excluded.  Level = mean prediction over the pitcher's strike / ball pitches.
(1) STABLE / STICKY: split-half reliability (alternating games; >= 150 thrown pitches per half) 2025 | 2026; year-to-year r at >= 150, 300, 500, 800 and 1,100 pitches in each year, next to the saved levels of the earlier measures (output/y2y_levels_{2025,2026}.csv)
    and the stuff-only whiff measures recomputed here (P(whiff | stuff), P(swstr | stuff) networks).   Levels saved to output/strike_stuff_levels_{year}.csv.
(2) PREDICTIVE: the standard K% harness (fit full 2025 -> 1H2026, fold 2 -> 2H2026, level-free, frozen weights from the PUBLISHED model, pitcher bootstrap): published | additive swstr_ct | strike|stuff replaces the stuff term | swstr_ct + strike|stuff;
    and BB% (the production blended rate + command score): published | + swstr_ct | + strike|stuff | + both.   Output: output/strike_stuff_test.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_k_prior_production_test as pt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import fl_wh_sva_prior_k_test as pk
import mlb_marcel as mm

warnings.filterwarnings("ignore")
STUFF = [c for c in fp.FULL_COLS if c not in fp.LOC_COLS]; ADD = pp.SPEC["K"]; BBS = pp.BB_SPEC_BLENDED; THRESH = (150, 300, 500, 800, 1100)
SEASONS = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}


def xgb_strike_stuff(s):
    d = s[s["pitch_call"] != "undefined"].dropna(subset=STUFF + ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]); d = d[d["pitch_call"].isin(fp._STRIKE_CALLS | fp._BALL_CALLS)].copy()
    y = d["pitch_call"].isin(fp._STRIKE_CALLS).astype(int).to_numpy(); X = d[STUFF].astype(float).to_numpy(); g = d["pitcher_id"].to_numpy(); p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        p[te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return d[["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]].assign(sva=p - p.mean())


def stability(raw):
    out = {}; new = {}
    for yr, (a, b) in SEASONS.items():
        s = fp.filter_period(raw, a, b); g = s.drop_duplicates(["game_id", "pitcher_id"])[["game_id", "pitcher_id", "date"]].sort_values("date"); g["half"] = g.groupby("pitcher_id").cumcount() % 2
        thrown = s.merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"]).groupby(["pitcher_id", "half"]).size().unstack()
        for lab, sv in (("strike|stuff network", fp.compute_pooled_wh_sva(s, estimator="strike_stuff")), ("strike|stuff XGBoost", xgb_strike_stuff(s)), ("whiff|stuff network", fp.compute_pooled_wh_sva(s, estimator="whiff_stuff")),
                        ("swstr|stuff network", fp.compute_pooled_wh_sva(s, estimator="swstr_stuff"))):
            d = sv.merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"]); lv = d.groupby(["pitcher_id", "half"])["sva"].mean().unstack(); ok = (thrown.reindex(lv.index).min(axis=1) >= 150) & lv.notna().all(axis=1)
            r = float(np.corrcoef(lv.loc[ok, 0], lv.loc[ok, 1])[0, 1]); out.setdefault(lab, {})[yr] = {"split_half_r": r, "S-B": 2 * r / (1 + r), "pitchers": int(ok.sum())}
            lvl = sv.groupby("pitcher_id")["sva"].mean(); lvl.index = lvl.index.astype(str); new.setdefault(lab, {})[yr] = lvl
        pd.DataFrame({k: v[yr] for k, v in new.items()}).to_csv(f"output/strike_stuff_levels_{yr}.csv"); print(f"  {yr} done", flush=True)
    ref = {yr: pd.read_csv(f"output/y2y_levels_{yr}.csv", index_col=0, dtype={0: str}) for yr in SEASONS}
    for yr in SEASONS:
        ref[yr].index = ref[yr].index.astype(str)
    n = {yr: ref[yr]["n_pitches"] for yr in SEASONS}; levels = {lab: (new[lab][2025], new[lab][2026]) for lab in new}
    for lab in ("XGBoost wh_sva", "swing network wh_sva", "swstr_ct", "strike_loc", "strike_ct", "raw SwStr% (per pitch)"):
        levels[lab] = (ref[2025][lab], ref[2026][lab])
    rows = []
    for lab, (a, b) in levels.items():
        r = {"measure": lab}
        for thr in THRESH:
            j = pd.concat([a.rename("a"), b.rename("b"), n[2025].rename("n25"), n[2026].rename("n26")], axis=1).dropna(); j = j[(j["n25"] >= thr) & (j["n26"] >= thr)]
            rr = float(np.corrcoef(j["a"], j["b"])[0, 1]); z = np.arctanh(rr); se = 1 / np.sqrt(len(j) - 3); r[f"r >= {thr}"] = rr; r[f"CI {thr}"] = f"[{np.tanh(z - 1.96 * se):.2f}, {np.tanh(z + 1.96 * se):.2f}]"; r[f"n {thr}"] = len(j)
        rows.append(r)
    for yr in SEASONS:
        M = pd.DataFrame({lab: levels[lab][0 if yr == 2025 else 1] for lab in levels}).dropna(); print(f"\n{yr}: correlation of pitcher levels (n = {len(M)})\n" + M.corr().round(2).to_string())
    pd.set_option("display.width", 280); print("\nSPLIT-HALF (S-B) 2025 | 2026\n" + "\n".join(f"  {k:26s} {v[2025]['S-B']:.4f} | {v[2026]['S-B']:.4f}  ({v[2025]['pitchers']} | {v[2026]['pitchers']} pitchers)" for k, v in out.items()))
    print("\nYEAR-TO-YEAR r, by pitches-per-year threshold\n" + pd.DataFrame(rows).round(3).to_string(index=False)); return {"split_half": out, "y2y": rows}


def lg_rate(raw, windows, what):
    t = pd.concat([fp.filter_period(raw, a, b) for a, b in windows], ignore_index=True); t = t[t["is_terminal"]]; return float((t["k_or_bb"] == what).mean())


def predictive(raw):
    F = {e: pt.frames(raw, e) for e in ("xgb", "swstr_ct", "strike_stuff")}; tr, n0_k, f2, n0_k2 = F["xgb"]; lv = {"sw": "swstr_ct", "st": "strike_stuff"}
    def merged(frame, which):
        d = frame.copy()
        for k, e in lv.items():
            d = d.merge(F[e][which][["pitcher_id", "wh_sva_level"]].rename(columns={"wh_sva_level": k}), on="pitcher_id", how="left")
        return d
    dtr, dte = merged(tr, 0).copy(), merged(f2, 2).copy(); lg_bb_tr = lg_rate(raw, [(bt.START_2025, bt.END_2025)], "walk"); lg_bb_f2 = lg_rate(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], "walk")
    _, _, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    need_bb = ["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score", "sw", "st"]
    dbb = dtr.dropna(subset=need_bb).copy(); wbb = mm.reliability_weight(dbb["PA_test"], n0_bb).values; dbb["BB_raw"] = dbb["naive_BB_rate"] * (dbb["PA"] + n0_bb) - n0_bb * lg_bb_tr
    n0b = mm.fit_n0_bridge(dbb["BB_raw"], dbb["PA"], n0_bb, lg_bb_tr, dbb["fl_bb_rate_bridge"], dbb["BB_rate_actual"], wbb); dbb["blended_BB_rate"] = mm.blended_rate(dbb["BB_raw"], dbb["PA"], n0_bb, lg_bb_tr, n0b, dbb["fl_bb_rate_bridge"])
    tb = dte.dropna(subset=need_bb).copy(); tb["BB_raw"] = tb["naive_BB_rate"] * (tb["PA"] + n0_bb) - n0_bb * lg_bb_f2; tb["blended_BB_rate"] = mm.blended_rate(tb["BB_raw"], tb["PA"], n0_bb, lg_bb_f2, n0b, tb["fl_bb_rate_bridge"])
    dk = dtr.dropna(subset=["K_rate_actual"] + ADD + ["sw", "st"]).copy(); wk = mm.reliability_weight(dk["PA_test"], n0_k).values; tk = dte.dropna(subset=["K_rate_actual"] + ADD + ["sw", "st"]).copy(); base_k = [f for f in ADD if f != "wh_sva_level"]; res = {}
    for stat, dtrain, wtr, dtest, target, specs in (
            ("K%", dk, wk, tk, "K_rate_actual", {"published": ADD, "additive swstr_ct": base_k + ["sw"], "strike|stuff": base_k + ["st"], "swstr_ct + strike|stuff": base_k + ["sw", "st"]}),
            ("BB%", dbb, wbb, tb, "BB_rate_actual", {"published": BBS, "+ swstr_ct": BBS + ["sw"], "+ strike|stuff": BBS + ["st"], "+ swstr_ct + strike|stuff": BBS + ["sw", "st"]})):
        P = {}; coefs = {}
        for lab, spec in specs.items():
            m = bt.fit_weighted(dtrain, target, spec, wtr); P[lab] = m.predict(dtest).to_numpy(); coefs[lab] = {k: (round(float(m.params[k]), 4), round(float(m.tvalues[k]), 2)) for k in spec if k in ("sw", "st", "wh_sva_level")}
        y = dtest[target].to_numpy(); n_t = dtest["PA_test"].to_numpy(); ids = dtest["pitcher_id"].to_numpy(); base = P["published"]; w = ew.weights_from_base(y, base, n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w); rows = []
        for lab, q in P.items():
            rm, ma = er.wmetrics(y, cen(q), w); r = {"model": lab, "RMSE": rm, "MAE": ma}
            if lab != "published":
                c = er.compare(y, cen(base), cen(q), w, ids, n=1000); r.update({"dRMSE": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "MAE 90%": c["MAE 90%"], "P(MAE better)": c["P(MAE better)"]})
            rows.append(r)
        pd.set_option("display.width", 260); print(f"\n{stat}: fold 2, n = {len(y)}, level-free, frozen weights (coefficients (b, t): {coefs})\n" + er.table(rows, 5)); res[stat] = {"n": len(y), "table": rows, "coefs": coefs}
        if stat == "K%":
            c = er.compare(y, cen(P["additive swstr_ct"]), cen(P["strike|stuff"]), w, ids, n=1000); print(f"\nstrike|stuff vs additive swstr_ct: dRMSE {c['dRMSE']:+.5f} {c['RMSE 90%']} P {c['P(RMSE better)']:.2f} | dMAE {c['dMAE']:+.5f} P {c['P(MAE better)']:.2f}")
    return res


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); print("STABILITY", flush=True); st = stability(raw); print("\nPREDICTIVE", flush=True); pr = predictive(raw)
    json.dump({"stability": st, "predictive": pr}, open("output/strike_stuff_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
