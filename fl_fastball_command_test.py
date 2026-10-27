"""
fl_fastball_command_test.py -- a FASTBALL-SPECIFIC command measure for the BB% projection model  (user 2026-10-27: "fastball command is the most important ... focus on the projection model for now")
FASTBALLS = four-seam + sinker (FL names four_seam / sinker; Statcast FF / SI; the project's fp.FASTBALL_TYPES, as in the fb_zone_pct test; cutters excluded).
STEP 1 (MLB): target = OpenCommand inferred miss on the pitcher's fastballs (FF + SI rows, tracked-pitch-weighted), pitcher-season, >= 150 Statcast and >= 150 tracked fastballs.  Features computed on the pitcher's FASTBALLS only with fl_command_variants.build (plain, conditional, 12-count shrunk, area, inferred-target cluster).
   Models (OLS, pitcher-grouped 5-fold CV R2 / RMSE): the current all-pitch formula applied to the fastball variances (frozen coefficients), and fastball-specific fits of each feature set.
STEP 2 (FL): the fastball scores (frozen MLB coefficients, features clipped to the MLB range) -- year-to-year stickiness (2025 vs 2026) and BB% in the four-pairing harness (T0->T1b, T0->T2, T1->T2, T2->T0; production spec [blended_BB_rate, command_score]; frozen weights, level-free, RMSE / MAE, pitcher bootstrap 1,000):
   (a) fastball score ADDED to the production command_score; (b) fastball score REPLACES it; (c) fb_blend = all-pitch score with the fastball part up-weighted (score computed on all pitches plus the fastball score, equal weights).   Pitchers with too few fastballs get the training-window mean (neutral).
Output: output/fastball_command_test.json
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import fl_command_variants as cvr
import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import mlb_marcel as mm
import mlb_opencommand_locvar_test as mo

warnings.filterwarnings("ignore")
MIN_FB = 150; FB_FL = list(fp.FASTBALL_TYPES); FB_MLB = ["FF", "SI"]
SETS = {"plain refit": ["hvar", "vvar"], "cond3": ["hvar_c", "vvar_c"], "cond3 + area": ["hvar_c", "vvar_c", "area_c"], "cond12 shrunk": ["hvar_e", "vvar_e"], "cluster": ["gh", "gv"], "plain + cond3": ["hvar", "vvar", "hvar_c", "vvar_c"], "cond12 shrunk + cluster": ["hvar_e", "vvar_e", "gh", "gv"]}


def mlb_fb():
    names = pd.read_parquet("output/pitcher_id_name_map.parquet"); names["name_clean"] = names["name"].apply(mo.statcast_name_clean); idn = names.drop_duplicates("pitcher_id").set_index("pitcher_id")["name_clean"]; out = []; oc = []
    for y in (2024, 2025):
        cache = f"output/command_variants_mlbfb_{y}.parquet"
        if os.path.exists(cache):
            F = pd.read_parquet(cache)
        else:
            d = pd.read_parquet(f"statcast_{y}.parquet", columns=["pitcher", "pitch_type", "plate_x", "plate_z", "p_throws", "stand", "balls", "strikes"]); d = d[d["pitch_type"].isin(FB_MLB)].dropna(subset=["plate_x", "plate_z", "pitcher", "balls", "strikes"])
            d = d.assign(pid=d["pitcher"], x=d["plate_x"] * np.where(d["p_throws"] == "R", 1, -1), z=d["plate_z"], same=(d["stand"] == d["p_throws"]).astype(int))[["pid", "pitch_type", "x", "z", "same", "balls", "strikes"]]; F = cvr.build(d); F.to_parquet(cache)
        F["year"] = y; F["name_clean"] = F["pid"].map(idn); out.append(F)
        o = pd.read_csv(f"oc_{y}_command_scores.csv"); o = o[o["pitch_type"].isin(FB_MLB)].copy(); o["name_clean"] = o["pitcher"].apply(mo.clean_name); o["ws"] = o["inferred_in"] * o["n"]; o = o.groupby("name_clean", as_index=False).agg(n_oc=("n", "sum"), ws=("ws", "sum")); o["oc"] = o["ws"] / o["n_oc"]; o["year"] = y; oc.append(o[["name_clean", "year", "n_oc", "oc"]])
    D = pd.concat(out, ignore_index=True).merge(pd.concat(oc), on=["name_clean", "year"], how="inner"); return D[(D["n"] >= MIN_FB) & (D["n_oc"] >= MIN_FB)].dropna(subset=cvr.ALLF + ["oc"]).reset_index(drop=True)


def step1(D):
    y = D["oc"].to_numpy(float); w = D["n_oc"].to_numpy(float); grp = D["name_clean"].to_numpy(); lo = D[cvr.ALLF].quantile(.005); hi = D[cvr.ALLF].quantile(.995); rows = []; coefs = {}
    g0 = 4.424373963818792 + 3.9415731433625307 * D["hvar"].clip(*fp.COMMAND_SCORE_HORIZ_RANGE) + 4.281096060834336 * D["vvar"].clip(*fp.COMMAND_SCORE_VERT_RANGE); g0 = g0.to_numpy(float); m = cvr.metr(y, g0, w)
    rows.append({"model": "all-pitch formula on fastball variances (frozen)", "R2": m["R2"], "RMSE": m["RMSE"], "r2 (bias removed)": float(np.corrcoef(g0, y)[0, 1] ** 2)})
    for lab, cols in SETS.items():
        X = D[cols].clip(lo[cols], hi[cols], axis=1).to_numpy(float); p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, grp):
            b = cvr.ols(X[tr], y[tr], w[tr]); p[te] = np.column_stack([np.ones(len(te)), X[te]]) @ b
        m = cvr.metr(y, p, w); rows.append({"model": "fastball fit: " + lab, "R2": m["R2"], "RMSE": m["RMSE"], "r2 (bias removed)": float(np.corrcoef(p, y)[0, 1] ** 2)}); coefs[lab] = {"cols": cols, "b": cvr.ols(X, y, w).tolist()}
    return pd.DataFrame(rows), coefs, lo, hi


def fl_fb_window(r, name, wins):
    cache = f"output/command_variants_fb_{name}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    d = pd.concat([fp.filter_period(r, a, b) for a, b in wins], ignore_index=True); d = d[d["pitch_type"].isin(FB_FL) & d["batter_side"].isin(["left", "right"]) & d["pitcher_throws"].isin(["left", "right"])].dropna(subset=["plate_x_as", "plate_z", "pitcher_id", "balls", "strikes"])
    d = d.assign(pid=d["pitcher_id"], x=d["plate_x_as"], z=d["plate_z"], same=(d["batter_side"] == d["pitcher_throws"]).astype(int))[["pid", "pitch_type", "x", "z", "same", "balls", "strikes"]]
    print(f"  FL fastballs {name}: {len(d):,}", flush=True); F = cvr.build(d); F.to_parquet(cache); return F


def main():
    pd.set_option("display.width", 250); D = mlb_fb(); print(f"STEP 1: MLB fastball cells: {len(D)} pitcher-seasons, {D['name_clean'].nunique()} pitchers; OpenCommand fastball miss mean {D['oc'].mean():.2f} SD {D['oc'].std():.2f}", flush=True); A, coefs, lo, hi = step1(D); print(A.round(4).to_string(index=False), flush=True)
    raw, r = cvr.fl_prep(); wins = {**{n: w[0] for n, w in cvr.WINDOWS.items()}, **cvr.SEASONS}; Fb = {n: fl_fb_window(r, n, w) for n, w in wins.items()}; S = {}
    for n in wins:
        F = Fb[n]; F = F[F["n"] >= 50].copy(); s = cvr.score_fl(F, coefs, lo, hi); g = 4.424373963818792 + 3.9415731433625307 * F["hvar"].clip(*fp.COMMAND_SCORE_HORIZ_RANGE) + 4.281096060834336 * F["vvar"].clip(*fp.COMMAND_SCORE_VERT_RANGE); s["fb frozen formula"] = g.to_numpy(); s["n_fb"] = F["n"].to_numpy(); S[n] = s
    labs = ["fb frozen formula"] + ["fastball fit: " + k for k in SETS]; rows = []
    for lab in labs:
        key = lab.replace("fastball fit: ", ""); res = {"measure": lab}
        for thr in (50, 100, 200, 400):
            a, b = S["S2025"].set_index("pid"), S["S2026"].set_index("pid"); j = pd.concat([a[key].rename("a"), b[key].rename("b"), a["n_fb"].rename("n25"), b["n_fb"].rename("n26")], axis=1).dropna(); j = j[(j["n25"] >= thr) & (j["n26"] >= thr)]
            rr = float(np.corrcoef(j["a"], j["b"])[0, 1]); z = np.arctanh(rr); se = 1 / np.sqrt(len(j) - 3); res[f"r>={thr}"] = rr; res[f"CI{thr}"] = f"[{np.tanh(z - 1.96 * se):.2f},{np.tanh(z + 1.96 * se):.2f}]"; res[f"n{thr}"] = len(j)
        rows.append(res)
    St = pd.DataFrame(rows); print("\nSTEP 2a: FL FASTBALL-SCORE STICKINESS (2025 vs 2026; minimum fastballs per year)\n" + St.round(3).to_string(index=False), flush=True)
    # BB% harness
    ren = {lab: "f%d" % i for i, lab in enumerate(["fb frozen formula"] + list(SETS))}; key_of = {"fb frozen formula": "fb frozen formula", **{k: k for k in SETS}}
    out = []; coefs_out = []; fr = {}
    for name in cvr.WINDOWS:
        feat_w, lab_w = cvr.WINDOWS[name]; df, n0bb = None, None; df, n0k, n0bb = bt.assemble_features(raw, feat_w, with_changepoints=False, with_surprise_levels=False); df = df.merge(bt.actual_rate(raw, lab_w), on="pitcher_id", how="inner")
        cur = fp.compute_command_score(pd.concat([fp.filter_period(raw, a, b) for a, b in feat_w], ignore_index=True))[["pitcher_id", "command_score"]]; df = df.merge(cur, on="pitcher_id", how="left")
        sc = S[name].rename(columns={"pid": "pitcher_id", **{k: v for k, v in ren.items()}}); df = df.merge(sc[["pitcher_id"] + list(ren.values())], on="pitcher_id", how="left")
        for c in ["command_score"] + list(ren.values()):
            df[c] = df[c].fillna(df[c].mean())
        for v in ren.values():
            df["bl_" + v] = (df["command_score"] + df[v]) / 2     # blend: raw averaged (all-pitch score and fastball score are both inches-scale)
        fr[name] = (df, n0bb, cvr.lg_bb(raw, feat_w))
    cand = {}
    for lab, v in ren.items():
        cand["+ " + lab] = ["blended_BB_rate", "command_score", v]; cand["replace: " + lab] = ["blended_BB_rate", v]; cand["blend: " + lab] = ["blended_BB_rate", "bl_" + v]
    for a, b in cvr.PAIRS:
        da, n0, lga = fr[a]; db, _, lgb = fr[b]; need = ["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA"]; ta = da.dropna(subset=need).copy(); tb = db.dropna(subset=need).copy(); wtr = mm.reliability_weight(ta["PA_test"], n0).values
        ta["BB_raw"] = ta["naive_BB_rate"] * (ta["PA"] + n0) - n0 * lga; n0b = mm.fit_n0_bridge(ta["BB_raw"], ta["PA"], n0, lga, ta["fl_bb_rate_bridge"], ta["BB_rate_actual"], wtr); ta["blended_BB_rate"] = mm.blended_rate(ta["BB_raw"], ta["PA"], n0, lga, n0b, ta["fl_bb_rate_bridge"])
        tb["BB_raw"] = tb["naive_BB_rate"] * (tb["PA"] + n0) - n0 * lgb; tb["blended_BB_rate"] = mm.blended_rate(tb["BB_raw"], tb["PA"], n0, lgb, n0b, tb["fl_bb_rate_bridge"]); P = {"production": bt.fit_weighted(ta, "BB_rate_actual", ["blended_BB_rate", "command_score"], wtr).predict(tb).to_numpy()}
        for lab, spec in cand.items():
            m = bt.fit_weighted(ta, "BB_rate_actual", spec, wtr); P[lab] = m.predict(tb).to_numpy(); coefs_out.append({"pairing": f"{a}->{b}", "model": lab, "t_last": float(m.tvalues[spec[-1]]), "b_last": float(m.params[spec[-1]])})
        y = tb["BB_rate_actual"].to_numpy(); n_t = tb["PA_test"].to_numpy(); ids = tb["pitcher_id"].to_numpy(); base = P["production"]; w = ew.weights_from_base(y, base, n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w)
        for lab, q in P.items():
            rm, ma = er.wmetrics(y, cen(q), w); rr = {"pairing": f"{a}->{b}", "n": len(y), "model": lab, "RMSE": rm, "MAE": ma}
            if lab != "production":
                c = er.compare(y, cen(base), cen(q), w, ids, n=1000); rr.update({"dRMSE": c["dRMSE"], "P(RMSE)": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE)": c["P(MAE better)"]})
            out.append(rr)
        print(f"  BB {a}->{b} done", flush=True)
    T = pd.DataFrame(out); g = T.groupby("model").agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), P_RMSE_mean=("P(RMSE)", "mean"), better_in=("dRMSE", lambda s: int((s < 0).sum()))).sort_values("RMSE"); g["dRMSE"] = g["RMSE"] - g.loc["production", "RMSE"]; g["dMAE"] = g["MAE"] - g.loc["production", "MAE"]
    print("\nSTEP 2b: FL BB% -- MEAN OVER THE FOUR PAIRINGS (labels f0 = frozen formula on fastballs, f1.. = fastball fits in SETS order: " + ", ".join(f"f{i + 1} {k}" for i, k in enumerate(SETS)) + ")\n" + g.round(5).to_string(), flush=True)
    print("\nper pairing, best candidates vs production\n" + T[T["model"].isin(g.index[:6]) & (T["model"] != "production")].round(5).to_string(index=False))
    json.dump({"mlb": A.to_dict("records"), "stickiness": St.to_dict("records"), "bb": T.to_dict("records"), "bb_mean": g.reset_index().to_dict("records"), "coefs": coefs}, open("output/fastball_command_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
