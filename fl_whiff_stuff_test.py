"""
fl_whiff_stuff_test.py -- a network for P(whiff | stuff)  (user 2026-10-25: "What if we built a neural network for P(whiff|stuff)")
MODEL: P(whiff | swing, 7 stuff inputs: velocity, ivb, arm-side horizontal break, spin, extension, release height, release side) -- no location, no count, no baseline to subtract -- trained on swings, cross-fitted by pitcher (5-fold GroupKFold),
scored on EVERY pitch a pitcher threw; level = mean prediction over all his pitches ('whiff_stuff') or over his swings only ('whiff_stuff_sw').  Network = 3-seed MLP 128-64-32 (fl_wh_sva_whiffstuff_worker.py).  Also an XGBoost version (production params) for reference.
(1) STABLE / STICKY: split-half reliability (alternating games; >= 150 thrown pitches per half) 2025 and 2026; year-to-year correlation at >= 150, 300, 500, 800 and 1,100 pitches in each year, alongside the saved levels of the earlier measures (output/y2y_levels_{2025,2026}.csv).
(2) PREDICTIVE: the standard K% harness (fit full 2025 -> 1H2026, fold 2 -> 2H2026, level-free, frozen weights from the PUBLISHED model, pitcher bootstrap): published | additive swstr_ct | whiff_stuff replaces the stuff term | whiff_stuff_sw replaces the stuff term.
Output: output/whiff_stuff_test.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBClassifier

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_k_prior_production_test as pt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import mlb_marcel as mm

warnings.filterwarnings("ignore")
STUFF = [c for c in fp.FULL_COLS if c not in fp.LOC_COLS]; ADD = pp.SPEC["K"]; THRESH = (150, 300, 500, 800, 1100)
SEASONS = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}


def xgb_stuff(s, swstr=False):
    """XGBoost P(whiff | swing, stuff) (trained on swings) or, with swstr=True, P(swinging strike | stuff) per pitch (trained on all pitches); cross-fitted by pitcher, scored on every pitch; returns sva rows like the network estimator."""
    pop = s[s["pitch_call"] != "undefined"].dropna(subset=STUFF + ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]).copy(); sw = pop["is_swing"].to_numpy().astype(bool); wh = pop["is_whiff"].fillna(False).astype(int).to_numpy()
    y = (sw & (wh == 1)).astype(int) if swstr else wh; mask = np.ones(len(y), dtype=bool) if swstr else sw
    X = pop[STUFF].astype(float).to_numpy(); g = pop["pitcher_id"].to_numpy(); p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        trs = tr[mask[tr]]; p[te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(X[trs], y[trs]).predict_proba(X[te])[:, 1]
    return pop[["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]].assign(sva=p - p.mean())


def stability(raw):
    out = {}; new = {}
    for yr, (a, b) in SEASONS.items():
        s = fp.filter_period(raw, a, b); g = s.drop_duplicates(["game_id", "pitcher_id"])[["game_id", "pitcher_id", "date"]].sort_values("date"); g["half"] = g.groupby("pitcher_id").cumcount() % 2
        thrown = s.merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"]).groupby(["pitcher_id", "half"]).size().unstack()
        for lab, sv in (("whiff|stuff network (all pitches)", fp.compute_pooled_wh_sva(s, estimator="whiff_stuff")), ("whiff|stuff network (swings only)", fp.compute_pooled_wh_sva(s, estimator="whiff_stuff_sw")), ("whiff|stuff XGBoost (all pitches)", xgb_stuff(s)),
                        ("swstr|stuff network (all pitches)", fp.compute_pooled_wh_sva(s, estimator="swstr_stuff")), ("swstr|stuff XGBoost (all pitches)", xgb_stuff(s, swstr=True))):
            d = sv.merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"]); lv = d.groupby(["pitcher_id", "half"])["sva"].mean().unstack(); ok = (thrown.reindex(lv.index).min(axis=1) >= 150) & lv.notna().all(axis=1)
            r = float(np.corrcoef(lv.loc[ok, 0], lv.loc[ok, 1])[0, 1]); out.setdefault(lab, {})[yr] = {"split_half_r": r, "S-B": 2 * r / (1 + r), "pitchers": int(ok.sum())}; new.setdefault(lab, {})[yr] = sv.groupby("pitcher_id")["sva"].mean()
        print(f"  {yr} done", flush=True)
    ref = {yr: pd.read_csv(f"output/y2y_levels_{yr}.csv", index_col=0, dtype={0: str}) for yr in SEASONS}
    for yr in SEASONS:
        ref[yr].index = ref[yr].index.astype(str)
    for lab in new:
        for yr in SEASONS:
            new[lab][yr].index = new[lab][yr].index.astype(str)
    n ={yr: ref[yr]["n_pitches"] for yr in SEASONS}; rows = []
    levels = {lab: (new[lab][2025], new[lab][2026]) for lab in new}
    for lab in ("XGBoost wh_sva", "swing network wh_sva", "swstr_plain", "swstr_ct", "raw Whiff% (per swing)", "raw SwStr% (per pitch)", "raw K% (PA >= 40)"):
        levels[lab] = (ref[2025][lab], ref[2026][lab])
    for lab, (a, b) in levels.items():
        r = {"measure": lab}
        for thr in THRESH:
            j = pd.concat([a.rename("a"), b.rename("b"), n[2025].rename("n25"), n[2026].rename("n26")], axis=1).dropna(); j = j[(j["n25"] >= thr) & (j["n26"] >= thr)]
            rr = float(np.corrcoef(j["a"], j["b"])[0, 1]); z = np.arctanh(rr); se = 1 / np.sqrt(len(j) - 3); r[f"r >= {thr}"] = rr; r[f"CI {thr}"] = f"[{np.tanh(z - 1.96 * se):.2f}, {np.tanh(z + 1.96 * se):.2f}]"; r[f"n {thr}"] = len(j)
        rows.append(r)
    for yr in SEASONS:
        M = pd.DataFrame({lab: (levels[lab][0] if yr == 2025 else levels[lab][1]) for lab in levels}).dropna(); print(f"\n{yr}: correlation of pitcher levels (n = {len(M)})\n" + M.corr().round(2).to_string())
    pd.set_option("display.width", 280); print("\nSPLIT-HALF (S-B) 2025 | 2026\n" + "\n".join(f"  {k:36s} {v[2025]['S-B']:.4f} | {v[2026]['S-B']:.4f}  ({v[2025]['pitchers']} | {v[2026]['pitchers']} pitchers)" for k, v in out.items()))
    print("\nYEAR-TO-YEAR r, by pitches-per-year threshold\n" + pd.DataFrame(rows).round(3).to_string(index=False)); return {"split_half": out, "y2y": rows}


def predictive(raw):
    F = {e: pt.frames(raw, e) for e in ("xgb", "swstr_ct", "whiff_stuff", "whiff_stuff_sw", "swstr_stuff")}; tr, n0_k, f2, n0_k2 = F["xgb"]; lv = {"sw": "swstr_ct", "ws": "whiff_stuff", "wsw": "whiff_stuff_sw", "ss": "swstr_stuff"}
    def merged(frame, which):
        d = frame.copy()
        for k, e in lv.items():
            d = d.merge(F[e][which][["pitcher_id", "wh_sva_level"]].rename(columns={"wh_sva_level": k}), on="pitcher_id", how="left")
        return d
    dk = merged(tr, 0).dropna(subset=["K_rate_actual"] + ADD + list(lv)).copy(); tk = merged(f2, 2).dropna(subset=["K_rate_actual"] + ADD + list(lv)).copy(); w_tr = mm.reliability_weight(dk["PA_test"], n0_k).values
    base = [f for f in ADD if f != "wh_sva_level"]; specs = {"published": ADD, "additive swstr_ct": base + ["sw"], "whiff_stuff (all pitches)": base + ["ws"], "whiff_stuff (swings only)": base + ["wsw"], "swstr_ct + whiff_stuff": base + ["sw", "ws"], "swstr_stuff (all pitches)": base + ["ss"], "swstr_stuff + whiff_stuff": base + ["ss", "ws"]}; P = {}; coefs = {}
    for lab, spec in specs.items():
        m = bt.fit_weighted(dk, "K_rate_actual", spec, w_tr); P[lab] = m.predict(tk).to_numpy(); coefs[lab] = {k: (round(float(m.params[k]), 4), round(float(m.tvalues[k]), 2)) for k in spec if k in ("sw", "ws", "wsw", "wh_sva_level")}
    y = tk["K_rate_actual"].to_numpy(); n_t = tk["PA_test"].to_numpy(); ids = tk["pitcher_id"].to_numpy(); b = P["published"]; w = ew.weights_from_base(y, b, n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w); rows = []
    for lab, q in P.items():
        rm, ma = er.wmetrics(y, cen(q), w); r = {"model": lab, "RMSE": rm, "MAE": ma}
        if lab != "published":
            c = er.compare(y, cen(b), cen(q), w, ids, n=1000); r.update({"dRMSE": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "MAE 90%": c["MAE 90%"], "P(MAE better)": c["P(MAE better)"]})
        rows.append(r)
    for a, bb in (("additive swstr_ct", "whiff_stuff (all pitches)"), ("whiff_stuff (all pitches)", "swstr_stuff (all pitches)"), ("additive swstr_ct", "swstr_stuff (all pitches)")):
        c = er.compare(y, cen(P[a]), cen(P[bb]), w, ids, n=1000); print(f"\n{bb} vs {a}: dRMSE {c['dRMSE']:+.5f} {c['RMSE 90%']} P {c['P(RMSE better)']:.2f} | dMAE {c['dMAE']:+.5f} P {c['P(MAE better)']:.2f}")
    print(f"\nK%: fold 2, n = {len(y)}, coefficients (b, t): {coefs}\n" + er.table(rows, 5)); return {"n": len(y), "table": rows, "coefs": coefs}


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); print("STABILITY", flush=True); st = stability(raw); print("\nPREDICTIVE", flush=True); pr = predictive(raw)
    json.dump({"stability": st, "predictive": pr}, open("output/whiff_stuff_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
