"""
fl_command_variants.py -- more ways to improve the Command Score, judged by (A) fit to MLB OpenCommand, (B) FL year-to-year stickiness, (C) FL BB% prediction  (user 2026-10-27: "Try some stuff. Try to keep stability/stickiness/BB% model in mind.")
All variants are built by ONE function (build) from pitch-level rows with the same columns for MLB Statcast and FL tracking: pitcher, pitch_type, x (plate_x, arm-side signed), z (plate_z), same (batter same-handed as the pitcher), balls, strikes.
VARIANTS (pitcher-level features, each pooled n-weighted across its cells):
  hvar, vvar    plain variances within pitcher x pitch type (cells >= 15)                       [the current inputs]
  hvar_c, vvar_c  conditional variances within type x same-handedness x 3 count buckets (cells >= 10)        [M3 of fl_command_improve.py]
  area_c        sqrt of the 2D covariance determinant in the same cells (a miss-shape / ellipse-area term)
  hvar_e, vvar_e  conditional variances within type x same-handedness x all 12 counts, EMPIRICAL-BAYES shrunk toward the type x side variance (k = 20 pseudo-pitches; cells >= 3)
  gh, gv        INFERRED-TARGET residual: within pitcher x type x side, a Gaussian mixture (1-3 components by BIC, >= 40 pitches per component) on (x, z); the variance of pitches around their own component centre
SCORES: OLS of OpenCommand 'ALL' inferred miss on the variant's features (MLB 2024-25, weights = tracked pitches), features clipped to their MLB 0.5-99.5% range, then FROZEN and applied to FL features.
(A) MLB fit: pitcher-grouped 5-fold CV and fit-2024 -> score-2025 (R2, RMSE inches).
(B) FL stickiness: year-to-year r (2025 vs 2026 regular season) at >= 150 / 300 / 500 / 800 pitches in each year -- the scores are frozen, so there is no refit noise.
(C) FL BB%: the project's four pairings (T0->T1b, T0->T2, T1->T2, T2->T0), production BB spec [blended_BB_rate, command_score] with command_score swapped for the variant; frozen weights from the published model, level-free, weighted RMSE and MAE, pitcher bootstrap (1,000).
Output: output/command_variants.json (+ cached FL features output/command_variants_{window}.parquet)
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import mlb_marcel as mm
import mlb_opencommand_locvar_test as mo

warnings.filterwarnings("ignore")
K_EB = 20; GMM_MIN = 40
WINDOWS = {"T0": ([(bt.START_2025, bt.CUTOFF_2025)], [(bt._DAY_AFTER_2025_CUT, bt.END_2025)]), "T1b": ([(bt._DAY_AFTER_2025_CUT, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)]),
           "T1": ([(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)]), "T2": ([(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)])}
PAIRS = (("T0", "T1b"), ("T1", "T2"), ("T0", "T2"), ("T2", "T0"))
SEASONS = {"S2025": [(bt.START_2025, bt.END_2025)], "S2026": [(bt.START_2026, bt.END_2026)]}
VARIANTS = {"v1 cond3": ["hvar_c", "vvar_c"], "v2 cond12 shrunk": ["hvar_e", "vvar_e"], "v3 cond3 + area": ["hvar_c", "vvar_c", "area_c"], "v4 target-cluster": ["gh", "gv"], "v5 cluster + cond3": ["gh", "gv", "hvar_c", "vvar_c"],
            "v6 plain + cond3": ["hvar", "vvar", "hvar_c", "vvar_c"], "v7 cond12 shrunk + cluster": ["hvar_e", "vvar_e", "gh", "gv"]}
ALLF = ["hvar", "vvar", "hvar_c", "vvar_c", "area_c", "hvar_e", "vvar_e", "gh", "gv"]


def cells(d, keys):
    k = ["pid", "pitch_type"] + keys; g = d.groupby(k)[["x", "z", "xx", "zz", "xz"]].sum(); g["n"] = d.groupby(k).size(); g = g.reset_index(); n = g["n"]
    g["hv"] = (g["xx"] - g["x"] ** 2 / n) / (n - 1); g["vv"] = (g["zz"] - g["z"] ** 2 / n) / (n - 1); g["cv"] = (g["xz"] - g["x"] * g["z"] / n) / (n - 1); g["area"] = np.sqrt(np.maximum(g["hv"] * g["vv"] - g["cv"] ** 2, 0)); return g


def pool(g, cols, minn, rename, gk=("pid",)):
    g = g[g["n"] >= minn]; gk = list(gk); n = g.groupby(gk)["n"].sum(); return pd.DataFrame({new: (g[c] * g["n"]).groupby([g[k] for k in gk]).sum() / n for c, new in zip(cols, rename)})


def gmm_cells(d, gk=("pid",)):
    out = []
    for (pid, pt, sd), s in d.groupby(["pid", "pitch_type", "same"]):
        n = len(s)
        if n < 15:
            continue
        X = s[["x", "z"]].to_numpy(float)
        if n < 2 * GMM_MIN:
            out.append((pid, pt, n, X[:, 0].var(ddof=1), X[:, 1].var(ddof=1))); continue
        best, bic = None, np.inf
        for k in range(1, min(3, n // GMM_MIN) + 1):
            m = GaussianMixture(k, covariance_type="full", reg_covar=1e-3, random_state=0, max_iter=100).fit(X); b = m.bic(X)
            if b < bic:
                best, bic = m, b
        R = best.predict_proba(X); res = X[:, None, :] - best.means_[None, :, :]; v = (R[:, :, None] * res ** 2).sum((0, 1)) / max(n - best.n_components, 1); out.append((pid, pt, n, v[0], v[1]))
    g = pd.DataFrame(out, columns=["pid", "pitch_type", "n", "gh", "gv"]); return pool(g, ["gh", "gv"], 15, ["gh", "gv"], gk)


def build(d, by_type=False):
    """d: columns pid, pitch_type, x, z, same, balls, strikes (no missing). Returns one row per pitcher (or per pitcher x pitch type with by_type=True: the same features computed inside the type)."""
    gk = ["pid", "pitch_type"] if by_type else ["pid"]
    d = d.copy(); d["xx"] = d["x"] ** 2; d["zz"] = d["z"] ** 2; d["xz"] = d["x"] * d["z"]; d["cb"] = np.where(d["strikes"] == 2, 2, np.where(d["balls"] >= 2, 0, 1)); d["c12"] = d["balls"].clip(0, 3) * 3 + d["strikes"].clip(0, 2)
    P = pool(cells(d, []), ["hv", "vv"], 15, ["hvar", "vvar"], gk); C = pool(cells(d, ["same", "cb"]), ["hv", "vv", "area"], 10, ["hvar_c", "vvar_c", "area_c"], gk)
    e = cells(d, ["same", "c12"]); ref = cells(d, ["same"])[["pid", "pitch_type", "same", "hv", "vv"]].rename(columns={"hv": "hv0", "vv": "vv0"}); e = e.merge(ref, on=["pid", "pitch_type", "same"]); df = e["n"] - 1
    e["hv"] = (df * e["hv"] + K_EB * e["hv0"]) / (df + K_EB); e["vv"] = (df * e["vv"] + K_EB * e["vv0"]) / (df + K_EB); E = pool(e.dropna(subset=["hv", "vv"]), ["hv", "vv"], 3, ["hvar_e", "vvar_e"], gk)
    G = gmm_cells(d, gk); n = d.groupby(gk).size().rename("n"); return pd.concat([n, P, C, E, G], axis=1).reset_index()


# ---------------------------------------------------------------- MLB
def mlb_frame():
    names = pd.read_parquet("output/pitcher_id_name_map.parquet"); names["name_clean"] = names["name"].apply(mo.statcast_name_clean); idn = names.drop_duplicates("pitcher_id").set_index("pitcher_id")["name_clean"]; out = []
    for y in (2024, 2025):
        cache = f"output/command_variants_mlb_{y}.parquet"; cache_t = f"output/command_variants_mlbtype_{y}.parquet"
        if os.path.exists(cache) and os.path.exists(cache_t):
            F = pd.read_parquet(cache)
        else:
            d = pd.read_parquet(f"statcast_{y}.parquet", columns=["pitcher", "pitch_type", "plate_x", "plate_z", "p_throws", "stand", "balls", "strikes"]).dropna(subset=["plate_x", "plate_z", "pitch_type", "pitcher", "balls", "strikes"])
            d = d.assign(pid=d["pitcher"], x=d["plate_x"] * np.where(d["p_throws"] == "R", 1, -1), z=d["plate_z"], same=(d["stand"] == d["p_throws"]).astype(int))[["pid", "pitch_type", "x", "z", "same", "balls", "strikes"]]
            print(f"  MLB {y}: {len(d):,} pitches ...", flush=True); F = build(d); F.to_parquet(cache); build(d, by_type=True).to_parquet(cache_t)
        F["year"] = y; F["name_clean"] = F["pid"].map(idn); out.append(F)
    F = pd.concat(out, ignore_index=True); oc = []
    for y in (2024, 2025):
        o = pd.read_csv(f"oc_{y}_command_scores.csv"); o = o[o["pitch_type"] == "ALL"].copy(); o["name_clean"] = o["pitcher"].apply(mo.clean_name); o["ws"] = o["inferred_in"] * o["n"]; o = o.groupby("name_clean", as_index=False).agg(n_oc=("n", "sum"), ws=("ws", "sum")); o["oc"] = o["ws"] / o["n_oc"]; o["year"] = y; oc.append(o[["name_clean", "year", "n_oc", "oc"]])
    D = F.merge(pd.concat(oc), on=["name_clean", "year"], how="inner"); D = D[(D["n"] >= 300) & (D["n_oc"] >= 300)].dropna(subset=ALLF + ["oc"]).reset_index(drop=True); return D


def ols(X, y, w):
    A = np.column_stack([np.ones(len(y)), X]); sw = np.sqrt(w); return np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]


def metr(y, p, w):
    e = y - p; return {"R2": float(1 - np.sum(w * e ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)), "RMSE": float(np.sqrt(np.average(e ** 2, weights=w)))}


def mlb_fit(D):
    y = D["oc"].to_numpy(float); w = D["n_oc"].to_numpy(float); rows = []; coefs = {}; lo = D[ALLF].quantile(.005); hi = D[ALLF].quantile(.995); grp = D["name_clean"].to_numpy(); tr24 = (D["year"] == 2024).to_numpy(); te25 = (D["year"] == 2025).to_numpy()
    g0 = 4.424373963818792 + 3.9415731433625307 * D["hvar"].clip(*fp.COMMAND_SCORE_HORIZ_RANGE) + 4.281096060834336 * D["vvar"].clip(*fp.COMMAND_SCORE_VERT_RANGE); g0 = g0.to_numpy(float)
    r = {"variant": "v0 current (frozen)", **{"CV " + k: v for k, v in metr(y, g0, w).items()}, **{"24->25 " + k: v for k, v in metr(y[te25], g0[te25], w[te25]).items()}}; rows.append(r)
    for lab, cols in VARIANTS.items():
        X = D[cols].clip(lo[cols], hi[cols], axis=1).to_numpy(float); p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, grp):
            b = ols(X[tr], y[tr], w[tr]); p[te] = np.column_stack([np.ones(len(te)), X[te]]) @ b
        b24 = ols(X[tr24], y[tr24], w[tr24]); p25 = np.column_stack([np.ones(te25.sum()), X[te25]]) @ b24
        rows.append({"variant": lab, **{"CV " + k: v for k, v in metr(y, p, w).items()}, **{"24->25 " + k: v for k, v in metr(y[te25], p25, w[te25]).items()}}); coefs[lab] = {"cols": cols, "b": ols(X, y, w).tolist()}
    return pd.DataFrame(rows), coefs, lo, hi


def score_fl(F, coefs, lo, hi):
    out = pd.DataFrame({"pid": F["pid"]})
    for lab, c in coefs.items():
        X = F[c["cols"]].clip(lo[c["cols"]], hi[c["cols"]], axis=1).to_numpy(float); out[lab] = np.column_stack([np.ones(len(F)), X]) @ np.array(c["b"])
    return out


def by_type(coefs, lo, hi):
    """Within-pitch-type validity: the pitcher-level scores (frozen coefficients) applied to pitcher x TYPE cells (variants computed inside the type), vs OpenCommand's per-type inferred miss; cells >= 75 Statcast and tracked pitches (as fl_command_validity_by_type.py)."""
    import fl_command_validity_by_type as cvt
    c = cvt.build()[["pitcher", "pitch_type", "year", "n_sc", "n_oc", "oc", "g", "name_clean"]]; inv = {v: k for k, v in cvt.MAP.items()}; c["code"] = c["pitch_type"].map(inv); T = []
    for y in (2024, 2025):
        t = pd.read_parquet(f"output/command_variants_mlbtype_{y}.parquet"); t["year"] = y; T.append(t)
    T = pd.concat(T, ignore_index=True).rename(columns={"pid": "pitcher", "pitch_type": "code"}); M = c.merge(T, on=["pitcher", "code", "year"], how="inner").dropna(subset=ALLF); rows = []
    for lab in ["v0 current (frozen)"] + list(VARIANTS):
        if lab.startswith("v0"):
            M[lab] = M["g"]
        else:
            cl = coefs[lab]; X = M[cl["cols"]].clip(lo[cl["cols"]], hi[cl["cols"]], axis=1).to_numpy(float); M[lab] = np.column_stack([np.ones(len(M)), X]) @ np.array(cl["b"])
    for t, s in M.groupby("pitch_type"):
        if len(s) < 40:
            continue
        r = {"type": t, "cells": len(s)}; w = s["n_oc"].to_numpy(float); y = s["oc"].to_numpy(float)
        for lab in ["v0 current (frozen)"] + list(VARIANTS):
            r[lab] = metr(y, s[lab].to_numpy(float), w)["R2"]; r["r2 " + lab] = float(np.corrcoef(s[lab], y)[0, 1] ** 2)
        rows.append(r)
    allr = {"type": "ALL types pooled", "cells": len(M)}
    for lab in ["v0 current (frozen)"] + list(VARIANTS):
        allr[lab] = metr(M["oc"].to_numpy(float), M[lab].to_numpy(float), M["n_oc"].to_numpy(float))["R2"]; allr["r2 " + lab] = float(np.corrcoef(M[lab], M["oc"])[0, 1] ** 2)
    rows.append(allr); return pd.DataFrame(rows)


# ---------------------------------------------------------------- FL
def fl_prep():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); r = fp.add_count(raw); return raw, r


def fl_window(r, name, wins):
    cache = f"output/command_variants_{name}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    d = pd.concat([fp.filter_period(r, a, b) for a, b in wins], ignore_index=True); d = d[d["batter_side"].isin(["left", "right"]) & d["pitcher_throws"].isin(["left", "right"])].dropna(subset=["plate_x_as", "plate_z", "pitch_type", "pitcher_id", "balls", "strikes"])
    d = d.assign(pid=d["pitcher_id"], x=d["plate_x_as"], z=d["plate_z"], same=(d["batter_side"] == d["pitcher_throws"]).astype(int))[["pid", "pitch_type", "x", "z", "same", "balls", "strikes"]]
    print(f"  FL {name}: {len(d):,} pitches ...", flush=True); F = build(d); F.to_parquet(cache); return F


def stickiness(Fs, S):
    rows = []; n = {k: Fs[k].set_index("pid")["n"] for k in Fs}
    for lab in ["v0 current (frozen)"] + list(VARIANTS):
        a, b = S["S2025"].set_index("pid")[lab], S["S2026"].set_index("pid")[lab]; r = {"variant": lab}
        for thr in (150, 300, 500, 800):
            j = pd.concat([a.rename("a"), b.rename("b"), n["S2025"].rename("n25"), n["S2026"].rename("n26")], axis=1).dropna(); j = j[(j["n25"] >= thr) & (j["n26"] >= thr)]; rr = float(np.corrcoef(j["a"], j["b"])[0, 1]); z = np.arctanh(rr); se = 1 / np.sqrt(len(j) - 3); r[f"r>={thr}"] = rr; r[f"CI{thr}"] = f"[{np.tanh(z - 1.96 * se):.2f},{np.tanh(z + 1.96 * se):.2f}]"; r[f"n{thr}"] = len(j)
        rows.append(r)
    return pd.DataFrame(rows)


def lg_bb(raw, windows):
    t = pd.concat([fp.filter_period(raw, a, b) for a, b in windows], ignore_index=True); t = t[t["is_terminal"]]; return float((t["k_or_bb"] == "walk").mean())


def bb_frame(raw, name, S_fl):
    feat, lab = WINDOWS[name]; df, n0k, n0bb = bt.assemble_features(raw, feat, with_changepoints=False, with_surprise_levels=False); df = df.merge(bt.actual_rate(raw, lab), on="pitcher_id", how="inner")
    cur = fp.compute_command_score(pd.concat([fp.filter_period(raw, a, b) for a, b in feat], ignore_index=True))[["pitcher_id", "command_score"]]; df = df.merge(cur, on="pitcher_id", how="left"); sc = S_fl[name].rename(columns={"pid": "pitcher_id"}); df = df.merge(sc, on="pitcher_id", how="left")
    for c in ["command_score"] + list(VARIANTS):
        df[c] = df[c].fillna(df[c].mean())
    return df, n0bb, lg_bb(raw, feat)


def bb_harness(raw, S_fl):
    sf = {l: l.split()[0] for l in VARIANTS}; labels = ["command_score"] + list(sf.values()); rows = []; coefs = []; fr = {n: (bb_frame(raw, n, S_fl)[0].rename(columns=sf),) + bb_frame(raw, n, S_fl)[1:] for n in WINDOWS}
    for a, b in PAIRS:
        da, n0, lga = fr[a]; db, _, lgb = fr[b]; need = ["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA"]; ta = da.dropna(subset=need).copy(); tb = db.dropna(subset=need).copy(); wtr = mm.reliability_weight(ta["PA_test"], n0).values
        ta["BB_raw"] = ta["naive_BB_rate"] * (ta["PA"] + n0) - n0 * lga; n0b = mm.fit_n0_bridge(ta["BB_raw"], ta["PA"], n0, lga, ta["fl_bb_rate_bridge"], ta["BB_rate_actual"], wtr); ta["blended_BB_rate"] = mm.blended_rate(ta["BB_raw"], ta["PA"], n0, lga, n0b, ta["fl_bb_rate_bridge"])
        tb["BB_raw"] = tb["naive_BB_rate"] * (tb["PA"] + n0) - n0 * lgb; tb["blended_BB_rate"] = mm.blended_rate(tb["BB_raw"], tb["PA"], n0, lgb, n0b, tb["fl_bb_rate_bridge"]); P = {}
        for c in labels:
            m = bt.fit_weighted(ta, "BB_rate_actual", ["blended_BB_rate", c], wtr); P[c] = m.predict(tb).to_numpy(); coefs.append({"pairing": f"{a}->{b}", "model": c, "b": float(m.params[c]), "t": float(m.tvalues[c])})
        y = tb["BB_rate_actual"].to_numpy(); n_t = tb["PA_test"].to_numpy(); ids = tb["pitcher_id"].to_numpy(); base = P["command_score"]; w = ew.weights_from_base(y, base, n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w)
        for c in labels:
            rm, ma = er.wmetrics(y, cen(P[c]), w); r = {"pairing": f"{a}->{b}", "n": len(y), "model": c, "RMSE": rm, "MAE": ma}
            if c != "command_score":
                q = er.compare(y, cen(base), cen(P[c]), w, ids, n=1000); r.update({"dRMSE": q["dRMSE"], "90%": q["RMSE 90%"], "P(RMSE)": q["P(RMSE better)"], "dMAE": q["dMAE"], "P(MAE)": q["P(MAE better)"]})
            rows.append(r)
        print(f"  BB {a}->{b} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(coefs)


def main():
    pd.set_option("display.width", 250); D = mlb_frame(); print(f"(A) MLB: {len(D)} pitcher-seasons, {D['name_clean'].nunique()} pitchers", flush=True); A, coefs, lo, hi = mlb_fit(D); print(A.round(4).to_string(index=False), flush=True)
    BT = by_type(coefs, lo, hi); keys = ["v0 current (frozen)"] + list(VARIANTS)
    print("\n(A2) BY PITCH TYPE: R2 vs OpenCommand's per-type miss (1 - SSE/SST, weighted; frozen pitcher-level coefficients applied to type-level cells, so it includes bias)\n" + BT[["type", "cells"] + keys].round(3).to_string(index=False))
    print("\n(A3) BY PITCH TYPE: squared correlation within the type (bias / level removed)\n" + BT[["type", "cells"] + ["r2 " + k for k in keys]].round(3).to_string(index=False), flush=True)
    raw, r = fl_prep(); wins = {**{n: w[0] for n, w in WINDOWS.items()}, **SEASONS}; Fs = {n: fl_window(r, n, w) for n, w in wins.items()}; S = {}
    cur_names = {}
    for n, F in Fs.items():
        s = score_fl(F, coefs, lo, hi); cs = fp.compute_command_score(pd.concat([fp.filter_period(raw, a, b) for a, b in wins[n]], ignore_index=True))[["pitcher_id", "command_score"]].rename(columns={"pitcher_id": "pid", "command_score": "v0 current (frozen)"}); S[n] = s.merge(cs, on="pid", how="outer")
    St = stickiness(Fs, S); print("\n(B) FL YEAR-TO-YEAR STICKINESS (2025 vs 2026 regular season; frozen scores)\n" + St.round(3).to_string(index=False), flush=True)
    j = S["S2026"].set_index("pid").dropna(); print("\nFL 2026 score correlations with the current Command Score:", {k: round(float(j[k].corr(j["v0 current (frozen)"])), 3) for k in VARIANTS}, "; level (mean, SD): " + "; ".join(f"{k} {j[k].mean():.2f}/{j[k].std():.2f}" for k in ["v0 current (frozen)"] + list(VARIANTS)), flush=True)
    S_bb = {n: S[n].rename(columns={"v0 current (frozen)": "_cur"}).drop(columns="_cur") for n in WINDOWS}; T, K = bb_harness(raw, S_bb)
    print("\n(C) FL BB%: per pairing (change vs the production command_score; P = bootstrap P(better))\n" + T[T["model"] != "command_score"].round(5).to_string(index=False))
    g = T.groupby("model").agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean")).sort_values("RMSE"); g["dRMSE vs production"] = g["RMSE"] - g.loc["command_score", "RMSE"]; g["dMAE vs production"] = g["MAE"] - g.loc["command_score", "MAE"]
    print("\nMEAN OVER THE FOUR PAIRINGS\n" + g.round(5).to_string()); print("\nCOEFFICIENTS (per unit of the score; t)\n" + K.pivot(index="model", columns="pairing", values="t").round(1).to_string())
    json.dump({"mlb_fit": A.to_dict("records"), "by_type": BT.to_dict("records"), "stickiness": St.to_dict("records"), "bb": T.to_dict("records"), "bb_mean": g.reset_index().to_dict("records"), "coefs": coefs}, open("output/command_variants.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
