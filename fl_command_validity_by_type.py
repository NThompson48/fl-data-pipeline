"""
fl_command_validity_by_type.py -- can horiz_var and vert_var estimate OpenCommand miss distance better for some pitch types than others, and what does the relationship look like WITHIN each type?  (user 2026-10-27)
OpenCommand (github.com/tomdoyo/open-command) publishes, per pitcher and pitch type and season, the number of tracked pitches and 'inferred_in' = the computer-vision-derived distance (inches) from the catcher's actual inferred target.  Our global Command Score is
4.4244 + 3.9416 x horiz_var + 4.2811 x vert_var (inputs clipped to the MLB training support), fitted to the pitcher-level 'ALL' rows (769 MLB pitchers).  Here the unit is the PITCHER x PITCH-TYPE CELL (MLB 2024 and 2025; Statcast plate_x_as / plate_z variances computed within the type exactly as
fl_pitcher_features.compute_loc_var does; cells with >= 75 Statcast pitches AND >= 75 OpenCommand-tracked pitches; OpenCommand cells for the same name summed n-weighted).
PER TYPE: (1) correlation of horiz_var, vert_var and the global score with inferred_in;  (2) FIT ON 2024, SCORE ON 2025: the GLOBAL formula as is (bias, RMSE, R^2), the global formula with a per-type intercept, the global score recalibrated by type (affine), and a full per-type regression inferred ~ horiz_var + vert_var;
(3) the same four by 5-fold pitcher-grouped cross-validation on both seasons pooled; (4) standardised weights of horiz_var vs vert_var within the type (inches per SD, cluster-robust by pitcher) next to the global weights; (5) F-tests across types: do the SLOPES differ by type (vs equal slopes with type intercepts)?
Output: output/command_validity_by_type.json
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

import fl_pitcher_features as fp
import mlb_opencommand_locvar_test as mo

warnings.filterwarnings("ignore")
I0, CH, CV = fp.COMMAND_SCORE_INTERCEPT, fp.COMMAND_SCORE_HORIZ_COEF, fp.COMMAND_SCORE_VERT_COEF; HR, VR = fp.COMMAND_SCORE_HORIZ_RANGE, fp.COMMAND_SCORE_VERT_RANGE; MIN_N = 75
MAP = {"FF": "four_seam", "SI": "sinker", "SL": "slider", "CH": "changeup", "FC": "cutter", "ST": "sweeper", "CU": "curveball", "FS": "splitter"}; TYPES = list(MAP.values())
gscore = lambda h, v: I0 + CH * np.clip(h, *HR) + CV * np.clip(v, *VR)


def build():
    names = pd.read_parquet("output/pitcher_id_name_map.parquet"); names["name_clean"] = names["name"].apply(mo.statcast_name_clean); idn = names.drop_duplicates("pitcher_id").set_index("pitcher_id")["name_clean"]; cells = []
    for y in (2024, 2025):
        d = pd.read_parquet(f"statcast_{y}.parquet", columns=["pitcher", "pitch_type", "plate_x", "plate_z", "p_throws"]); d = d[d["pitch_type"].isin(MAP)].dropna(subset=["plate_x", "plate_z"]).copy()
        d["plate_x_as"] = d["plate_x"] * np.where(d["p_throws"] == "R", 1, -1); d["pitch_type"] = d["pitch_type"].map(MAP)
        g = d.groupby(["pitcher", "pitch_type"]).agg(n_sc=("plate_x_as", "size"), hvar=("plate_x_as", "var"), vvar=("plate_z", "var")).reset_index(); g["name_clean"] = g["pitcher"].map(idn); g["year"] = y
        o = pd.read_csv(f"oc_{y}_command_scores.csv"); o = o[o["pitch_type"].isin(MAP)].copy(); o["pitch_type"] = o["pitch_type"].map(MAP); o["name_clean"] = o["pitcher"].apply(mo.clean_name); o["ws"] = o["inferred_in"] * o["n"]
        o = o.groupby(["name_clean", "pitch_type"], as_index=False).agg(n_oc=("n", "sum"), ws=("ws", "sum")); o["oc"] = o["ws"] / o["n_oc"]
        cells.append(g.merge(o[["name_clean", "pitch_type", "n_oc", "oc"]], on=["name_clean", "pitch_type"], how="inner"))
    c = pd.concat(cells, ignore_index=True); c = c[(c["n_sc"] >= MIN_N) & (c["n_oc"] >= MIN_N)].dropna(subset=["hvar", "vvar", "oc", "name_clean"]).copy(); c["g"] = gscore(c["hvar"], c["vvar"]); return c


def metrics(y, p, w):
    e = y - p; sst = np.sum(w * (y - np.average(y, weights=w)) ** 2); return {"RMSE": float(np.sqrt(np.average(e ** 2, weights=w))), "R2": float(1 - np.sum(w * e ** 2) / sst), "bias": float(np.average(p - y, weights=w))}


def wls(X, y, w):
    X1 = np.column_stack([np.ones(len(y)), X]); sw = np.sqrt(w); b = np.linalg.lstsq(X1 * sw[:, None], y * sw, rcond=None)[0]; return b


def pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def main():
    c = build(); print(f"{len(c):,} pitcher x type cells (>= {MIN_N} Statcast and tracked pitches)", c.groupby("pitch_type").size().to_dict(), flush=True); res = {}; pd.set_option("display.width", 260)
    rows = []; coefs = []
    for t in TYPES:
        s = c[c["pitch_type"] == t].reset_index(drop=True)
        if len(s) < 40:
            continue
        w = s["n_oc"].to_numpy(float); y = s["oc"].to_numpy(float); H = s[["hvar", "vvar"]].to_numpy(float); gs = s["g"].to_numpy(float); tr = (s["year"] == 2024).to_numpy(); te = (s["year"] == 2025).to_numpy(); r = {"type": t, "cells": len(s), "pitchers": int(s["name_clean"].nunique()), "OC mean in": float(y.mean()), "OC SD": float(y.std()),
                                                                                                                                                                                                                                                              "corr horiz_var": float(np.corrcoef(s["hvar"], y)[0, 1]), "corr vert_var": float(np.corrcoef(s["vvar"], y)[0, 1]), "corr global score": float(np.corrcoef(gs, y)[0, 1])}
        if tr.sum() >= 25 and te.sum() >= 25:                                                  # fit 2024 -> score 2025
            a = metrics(y[te], gs[te], w[te]); bg = (y[tr] - gs[tr]).mean(); b = metrics(y[te], gs[te] + bg, w[te]); aff = wls(gs[tr, None], y[tr], w[tr]); c_ = metrics(y[te], pred(aff, gs[te, None]), w[te]); full = wls(H[tr], y[tr], w[tr]); d_ = metrics(y[te], pred(full, H[te]), w[te])
            r.update({"n 2024 / 2025": f"{int(tr.sum())} / {int(te.sum())}", "2025 R2 global as is": a["R2"], "RMSE global as is": a["RMSE"], "bias global (in)": a["bias"], "R2 + type intercept": b["R2"], "RMSE + type intercept": b["RMSE"], "R2 type affine": c_["R2"], "R2 type regression": d_["R2"], "RMSE type regression": d_["RMSE"]})
        out = {k: np.zeros(len(s)) for k in ("b", "a", "f")}                                    # pitcher-grouped CV on both seasons pooled
        for trn, tst in GroupKFold(5).split(s, y, s["name_clean"]):
            out["b"][tst] = gs[tst] + (y[trn] - gs[trn]).mean(); out["a"][tst] = pred(wls(gs[trn, None], y[trn], w[trn]), gs[tst, None]); out["f"][tst] = pred(wls(H[trn], y[trn], w[trn]), H[tst])
        r.update({"CV R2 global as is": metrics(y, gs, w)["R2"], "CV R2 + type intercept": metrics(y, out["b"], w)["R2"], "CV R2 type affine": metrics(y, out["a"], w)["R2"], "CV R2 type regression": metrics(y, out["f"], w)["R2"], "CV RMSE global": metrics(y, gs, w)["RMSE"], "CV RMSE type regression": metrics(y, out["f"], w)["RMSE"]})
        Z = (H - H.mean(0)) / H.std(0); m = sm.WLS(y, sm.add_constant(Z), weights=w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s["name_clean"])[0]}); coefs.append({"type": t, "inches per SD horiz_var": float(m.params[1]), "t": float(m.tvalues[1]), "inches per SD vert_var": float(m.params[2]), "t ": float(m.tvalues[2]), "SD horiz_var": float(H[:, 0].std()), "SD vert_var": float(H[:, 1].std()),
                                                                                                                                                                                                                                                                  "per-unit horiz (global 3.94)": float(m.params[1] / H[:, 0].std()), "per-unit vert (global 4.28)": float(m.params[2] / H[:, 1].std())}); rows.append(r)
    T = pd.DataFrame(rows).set_index("type"); K = pd.DataFrame(coefs).set_index("type")
    print("\n(1) WITHIN-TYPE CORRELATIONS WITH OPENCOMMAND INFERRED MISS (inches)\n" + T[["cells", "pitchers", "OC mean in", "OC SD", "corr horiz_var", "corr vert_var", "corr global score"]].round(3).to_string())
    print("\n(2) FIT ON 2024, SCORE ON 2025 (R2 vs OpenCommand miss within the type; RMSE and bias in inches)\n" + T[["n 2024 / 2025", "2025 R2 global as is", "RMSE global as is", "bias global (in)", "R2 + type intercept", "R2 type affine", "R2 type regression", "RMSE type regression"]].round(3).to_string())
    print("\n(3) PITCHER-GROUPED 5-FOLD CV, BOTH SEASONS POOLED\n" + T[["CV R2 global as is", "CV R2 + type intercept", "CV R2 type affine", "CV R2 type regression", "CV RMSE global", "CV RMSE type regression"]].round(3).to_string())
    print("\n(4) WITHIN-TYPE WEIGHTS (inches of miss per SD of the variance; cluster-robust t) and per-unit slopes vs the global 3.94 / 4.28\n" + K.round(2).to_string())
    P = c.copy(); P["pt"] = pd.Categorical(P["pitch_type"]); D = pd.get_dummies(P["pt"], drop_first=True).astype(float); w = P["n_oc"].to_numpy(float); y = P["oc"].to_numpy(float)
    Hh = P[["hvar", "vvar"]].to_numpy(float); Dn = D.to_numpy(float); X0 = np.column_stack([np.ones(len(P)), Hh]); X1 = np.column_stack([X0, Dn]); X2 = np.column_stack([X1, Dn * Hh[:, [0]], Dn * Hh[:, [1]]]).astype(float)
    f = {k: sm.WLS(y, X, weights=w).fit() for k, X in (("pooled", X0), ("type intercepts", X1), ("type slopes", X2))}
    print("\n(5) F-TESTS ACROSS TYPES (WLS): " + "; ".join(f"{a} -> {b}: F = {(f[a].ssr - f[b].ssr) / (f[b].df_model - f[a].df_model) / (f[b].ssr / f[b].df_resid):.1f}, p = {f[b].compare_f_test(f[a])[1]:.2g}" for a, b in (("pooled", "type intercepts"), ("type intercepts", "type slopes")))
          + f"\n    R2: one pooled line {f['pooled'].rsquared:.3f}; + type intercepts {f['type intercepts'].rsquared:.3f}; + type slopes {f['type slopes'].rsquared:.3f} (n = {len(P):,} cells)")
    json.dump({"by_type": T.reset_index().to_dict("records"), "weights": K.reset_index().to_dict("records")}, open("output/command_validity_by_type.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
