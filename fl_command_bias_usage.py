"""
fl_command_bias_usage.py -- what does a USAGE-WEIGHTED PITCH-TYPE BIAS CORRECTION do to the fit of the global Command Score?  (user 2026-10-27)
CURRENT: pitcher-level Command Score = 4.4244 + 3.9416 x horiz_var + 4.2811 x vert_var (variances computed within each pitch type, pooled n-weighted across types, inputs clipped to the MLB support), fitted to OpenCommand's 'ALL' inferred miss (inches).
CORRECTED: score + sum over types of (share of the pitcher's pitches of that type) x bias_type, where bias_type = the weighted mean of (OpenCommand inferred miss - the global formula applied to that type's own variances) over pitcher x type cells (the cell-level bias of fl_command_validity_by_type.py;
types without a bias, i.e. the rare ones, get 0).  A changeup-heavy pitcher is moved up by the changeup bias, a four-seam-only pitcher by almost nothing.
TARGET: OpenCommand 'ALL' inferred miss per pitcher-season (MLB 2024, 2025; pitchers with >= 300 Statcast pitches and >= 300 tracked pitches; weights = tracked pitches).  Biases are ESTIMATED ON 2024 CELLS and applied to 2025 pitchers (and 5-fold pitcher-grouped CV on both seasons, biases estimated on the training folds).
VARIANTS: (a) the global formula as is (frozen coefficients); (b) + usage-weighted bias correction; (c) the 3 coefficients REFIT on the training data (the best this variable pair can do without type information); (d) refit + correction.   Reported: R2, RMSE, mean error (inches), and how much the correction varies across pitchers.
Output: output/command_bias_usage.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import fl_command_validity_by_type as cv
import fl_pitcher_features as fp
import mlb_opencommand_locvar_test as mo

warnings.filterwarnings("ignore")
MIN_TYPE = 15; MIN_P = 300; MAIN = cv.MAP; gscore = cv.gscore


def statcast_cells(years=(2024, 2025)):
    names = pd.read_parquet("output/pitcher_id_name_map.parquet"); names["name_clean"] = names["name"].apply(mo.statcast_name_clean); idn = names.drop_duplicates("pitcher_id").set_index("pitcher_id")["name_clean"]; out = []
    for y in years:
        d = pd.read_parquet(f"statcast_{y}.parquet", columns=["pitcher", "pitch_type", "plate_x", "plate_z", "p_throws"]).dropna(subset=["plate_x", "plate_z", "pitch_type"]); d["plate_x_as"] = d["plate_x"] * np.where(d["p_throws"] == "R", 1, -1)
        g = d.groupby(["pitcher", "pitch_type"]).agg(n=("plate_x_as", "size"), hvar=("plate_x_as", "var"), vvar=("plate_z", "var")).reset_index(); g = g[g["n"] >= MIN_TYPE].copy(); g["name_clean"] = g["pitcher"].map(idn); g["year"] = y; g["type"] = g["pitch_type"].map(MAIN); out.append(g)
    return pd.concat(out, ignore_index=True)


def pitcher_table(cells, bias):
    """pitcher-season pooled variances (n-weighted, as compute_loc_var), usage-weighted correction from `bias` (dict type -> inches)."""
    cells = cells.copy(); cells["b"] = cells["type"].map(bias).fillna(0.0); cells["hn"] = cells["hvar"] * cells["n"]; cells["vn"] = cells["vvar"] * cells["n"]; cells["bn"] = cells["b"] * cells["n"]
    p = cells.groupby(["pitcher", "name_clean", "year"]).agg(n=("n", "sum"), hn=("hn", "sum"), vn=("vn", "sum"), bn=("bn", "sum")).reset_index(); p["hvar"] = p["hn"] / p["n"]; p["vvar"] = p["vn"] / p["n"]; p["corr"] = p["bn"] / p["n"]; p["g"] = gscore(p["hvar"], p["vvar"]); return p


def oc_all():
    out = []
    for y in (2024, 2025):
        o = pd.read_csv(f"oc_{y}_command_scores.csv"); o = o[o["pitch_type"] == "ALL"].copy(); o["name_clean"] = o["pitcher"].apply(mo.clean_name); o["ws"] = o["inferred_in"] * o["n"]; o = o.groupby("name_clean", as_index=False).agg(n_oc=("n", "sum"), ws=("ws", "sum")); o["oc"] = o["ws"] / o["n_oc"]; o["year"] = y; out.append(o[["name_clean", "year", "n_oc", "oc"]])
    return pd.concat(out, ignore_index=True)


def type_bias(cells, years):
    c = cv.build(); c = c[c["year"].isin(years)]; return {t: float(np.average(s["oc"] - s["g"], weights=s["n_oc"])) for t, s in c.groupby("pitch_type")}


def fit(X, y, w):
    X1 = np.column_stack([np.ones(len(y)), X]); sw = np.sqrt(w); return np.linalg.lstsq(X1 * sw[:, None], y * sw, rcond=None)[0]


def metr(y, p, w):
    e = y - p; sst = np.sum(w * (y - np.average(y, weights=w)) ** 2); return {"R2": float(1 - np.sum(w * e ** 2) / sst), "RMSE": float(np.sqrt(np.average(e ** 2, weights=w))), "mean error (pred - OC, in)": float(np.average(p - y, weights=w))}


def main():
    cells = statcast_cells(); oc = oc_all(); res = {}
    def table(bias, years):
        p = pitcher_table(cells, bias); p = p[p["year"].isin(years)].merge(oc, on=["name_clean", "year"], how="inner"); return p[(p["n"] >= MIN_P) & (p["n_oc"] >= MIN_P)].dropna(subset=["oc", "hvar", "vvar"]).reset_index(drop=True)
    b24 = type_bias(cells, (2024,)); ball = type_bias(cells, (2024, 2025)); print("type biases (OpenCommand - global formula, inches), 2024 cells:", {k: round(v, 3) for k, v in b24.items()}, "\n   both seasons:", {k: round(v, 3) for k, v in ball.items()}, flush=True); res["bias_2024"] = b24
    tr = table(b24, (2024,)); te = table(b24, (2025,)); print(f"\npitcher-seasons: 2024 {len(tr)}, 2025 {len(te)}; correction (inches) across 2025 pitchers: mean {te['corr'].mean():+.3f}, SD {te['corr'].std():.3f}, range [{te['corr'].min():+.2f}, {te['corr'].max():+.2f}]; OC miss SD {te['oc'].std():.2f}", flush=True)
    rows = []
    def run(label, trn, tst, bias_for_corr):
        wtr, wte = trn["n_oc"].to_numpy(float), tst["n_oc"].to_numpy(float); y = tst["oc"].to_numpy(float); out = {}
        out["(a) global as is"] = metr(y, tst["g"].to_numpy(), wte); out["(b) + usage-weighted bias correction"] = metr(y, (tst["g"] + tst["corr"]).to_numpy(), wte)
        c0 = float(np.average(trn["corr"], weights=wtr)); out["(b2) + correction CENTERED on the training mean"] = metr(y, (tst["g"] + tst["corr"] - c0).to_numpy(), wte)
        b = fit(trn[["hvar", "vvar"]].to_numpy(float), trn["oc"].to_numpy(float), wtr); out["(c) coefficients refit"] = metr(y, np.column_stack([np.ones(len(tst)), tst[["hvar", "vvar"]].to_numpy(float)]) @ b, wte)
        trn2 = trn.assign(y2=trn["oc"] - trn["corr"]); b2 = fit(trn2[["hvar", "vvar"]].to_numpy(float), trn2["y2"].to_numpy(float), wtr); out["(d) refit + correction"] = metr(y, np.column_stack([np.ones(len(tst)), tst[["hvar", "vvar"]].to_numpy(float)]) @ b2 + tst["corr"].to_numpy(), wte)
        for k, v in out.items():
            rows.append({"design": label, "variant": k, **v})
    run("fit 2024 -> score 2025", tr, te, b24)
    # pitcher-grouped CV on both seasons: biases estimated from the TRAINING pitchers' cells only
    both = table(ball, (2024, 2025)); cc = cv.build(); folds = list(GroupKFold(5).split(both, both["oc"], both["name_clean"])); pred = {k: np.zeros(len(both)) for k in ("a", "b", "b2", "c", "d")}
    for trn_i, tst_i in folds:
        trn_names = set(both.loc[trn_i, "name_clean"]); sub = cc[cc["name_clean"].isin(trn_names)]; bias = {t: float(np.average(s["oc"] - s["g"], weights=s["n_oc"])) for t, s in sub.groupby("pitch_type")}
        P = pitcher_table(cells, bias).set_index(["pitcher", "year"]); T = both.set_index(["pitcher", "year"]).join(P[["corr"]], rsuffix="_f"); T = T.reset_index(); T["corr"] = T["corr_f"]; trn, tst = T.iloc[trn_i], T.iloc[tst_i]
        pred["a"][tst_i] = tst["g"]; pred["b"][tst_i] = tst["g"] + tst["corr"]; pred["b2"][tst_i] = tst["g"] + tst["corr"] - np.average(trn["corr"], weights=trn["n_oc"]); b = fit(trn[["hvar", "vvar"]].to_numpy(float), trn["oc"].to_numpy(float), trn["n_oc"].to_numpy(float)); pred["c"][tst_i] = np.column_stack([np.ones(len(tst)), tst[["hvar", "vvar"]].to_numpy(float)]) @ b
        b2 = fit(trn[["hvar", "vvar"]].to_numpy(float), (trn["oc"] - trn["corr"]).to_numpy(float), trn["n_oc"].to_numpy(float)); pred["d"][tst_i] = np.column_stack([np.ones(len(tst)), tst[["hvar", "vvar"]].to_numpy(float)]) @ b2 + tst["corr"].to_numpy()
    y = both["oc"].to_numpy(float); w = both["n_oc"].to_numpy(float)
    for k, lab in (("a", "(a) global as is"), ("b", "(b) + usage-weighted bias correction"), ("b2", "(b2) + correction CENTERED on the training mean"), ("c", "(c) coefficients refit"), ("d", "(d) refit + correction")):
        rows.append({"design": f"pitcher-grouped 5-fold CV, both seasons (n = {len(both)})", "variant": lab, **metr(y, pred[k], w)})
    T = pd.DataFrame(rows); pd.set_option("display.width", 220); print("\n" + T.round(4).to_string(index=False))
    # who moves most
    te["pred_b"] = te["g"] + te["corr"]; te["err_a"] = te["g"] - te["oc"]; te["err_b"] = te["pred_b"] - te["oc"]; mv = te.assign(abs_move=te["corr"].abs()).sort_values("abs_move", ascending=False)
    big = mv.head(int(len(mv) * 0.2)); print(f"\nthe 20% of 2025 pitchers with the largest |correction| (n {len(big)}): mean |error| as is {np.average(np.abs(big['err_a']), weights=big['n_oc']):.3f} in -> corrected {np.average(np.abs(big['err_b']), weights=big['n_oc']):.3f} in; the rest: {np.average(np.abs(mv.iloc[len(big):]['err_a']), weights=mv.iloc[len(big):]['n_oc']):.3f} -> {np.average(np.abs(mv.iloc[len(big):]['err_b']), weights=mv.iloc[len(big):]['n_oc']):.3f}")
    print("correlation of the correction with the 2025 residual (OC - global score): %.3f (a perfect correction would be 1)" % np.corrcoef(te["corr"], te["oc"] - te["g"])[0, 1])
    json.dump({"rows": rows, "bias_2024": b24}, open("output/command_bias_usage.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
