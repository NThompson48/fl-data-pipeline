"""
fl_command_improve.py -- can the Command Score's fit to OpenCommand be improved, and how large is its practical uncertainty?  (user 2026-10-27)
TARGET: OpenCommand 'ALL' inferred miss (inches from the catcher's inferred target) per pitcher-season, MLB 2024 and 2025 (>= 300 Statcast pitches and >= 300 tracked pitches; 1,098 pitcher-seasons; weights = tracked pitches).  Everything is scored OUT OF SAMPLE: 5-fold CV grouped by pitcher (a pitcher's two seasons stay together), plus fit 2024 -> score 2025.
CURRENT (M0): 4.4244 + 3.9416 x horiz_var + 4.2811 x vert_var (frozen; variances within each pitch type, pooled n-weighted).   CANDIDATE IMPROVEMENTS (all computable from FL tracking too: plate location, batter side, count, zone limits, velocity, pitch type):
  M1 refit of the same two variances;  M2 sqrt (SD) form;  M3 CONDITIONAL variances (within pitch type x batter handedness relative to the pitcher x count bucket, so intended-target shifts across batters and counts are not counted as misses), raw height;  M4 conditional variances with height NORMALISED to the batter's zone
  (plate_z relative to sz_bot / sz_top);  M5 M0's variances + the conditional ones;  M6 robust spread (MAD) instead of variance;  M7 M5 + location / zone features (mean |x|, mean |height - zone middle|, share in zone, share in the heart);  M8 M7 + pitch mix and velocity;  M9 ridge on everything;  M10 gradient boosting on everything.
UNCERTAINTY: for the current formula and the best candidate: out-of-sample residual SD, empirical 50 / 80 / 90% prediction intervals, calibration (slope of OpenCommand on the score), by tracked-pitch count and by score tercile, and what it means on the Frontier League scale and in BB% points.   Output: output/command_improve.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import fl_command_validity_by_type as cv
import fl_pitcher_features as fp
import mlb_opencommand_locvar_test as mo

warnings.filterwarnings("ignore")
HR, VR = fp.COMMAND_SCORE_HORIZ_RANGE, fp.COMMAND_SCORE_VERT_RANGE; FB, BRK, OFF = {"FF", "SI", "FC"}, {"SL", "ST", "CU", "KC", "SV", "CS"}, {"CH", "FS", "FO"}


def pooled(g, cols, minn):
    g = g[g["n"] >= minn]; n = g.groupby("pitcher")["n"].sum(); out = pd.DataFrame({c: (g[c] * g["n"]).groupby(g["pitcher"]).sum() / n for c in cols}); out["n_kept"] = n; return out


def features(y):
    d = pd.read_parquet(f"statcast_{y}.parquet", columns=["pitcher", "pitch_type", "plate_x", "plate_z", "p_throws", "stand", "balls", "strikes", "sz_top", "sz_bot", "release_speed", "zone"]).dropna(subset=["plate_x", "plate_z", "pitch_type", "pitcher"])
    d["x"] = d["plate_x"] * np.where(d["p_throws"] == "R", 1, -1); d["z"] = d["plate_z"]; zh = (d["sz_top"] - d["sz_bot"]).where(lambda s: s > 0.5); d["zn"] = ((d["plate_z"] - d["sz_bot"]) / zh).clip(-1, 2.5); d["same"] = (d["stand"] == d["p_throws"]).astype(int)
    d["cb"] = np.where(d["strikes"] == 2, 2, np.where(d["balls"] >= 2, 0, 1)); d["ax"] = d["x"].abs(); d["az"] = (d["zn"] - 0.5).abs(); d["inz"] = d["zone"].between(1, 9).astype(float); d["heart"] = ((d["x"].abs() < 0.28) & d["zn"].between(0.33, 0.67)).astype(float)
    # M0 variables: variance within pitcher x type, n >= 15
    a = d.groupby(["pitcher", "pitch_type"]).agg(n=("x", "size"), hvar=("x", "var"), vvar=("z", "var"), hmad=("x", lambda s: 1.4826 * np.median(np.abs(s - np.median(s)))), vmad=("z", lambda s: 1.4826 * np.median(np.abs(s - np.median(s))))).reset_index(); P = pooled(a, ["hvar", "vvar", "hmad", "vmad"], 15)
    # conditional variances: within pitcher x type x same-handed x count bucket, n >= 10
    b = d.groupby(["pitcher", "pitch_type", "same", "cb"]).agg(n=("x", "size"), hvar_c=("x", "var"), vvar_c=("z", "var"), vvarn_c=("zn", "var")).reset_index(); C = pooled(b, ["hvar_c", "vvar_c", "vvarn_c"], 10).rename(columns={"n_kept": "n_kept_c"})
    L = d.groupby("pitcher").agg(n=("x", "size"), ax=("ax", "mean"), az=("az", "mean"), inz=("inz", "mean"), heart=("heart", "mean"), velo=("release_speed", "mean"), fb=("pitch_type", lambda s: s.isin(FB).mean()), brk=("pitch_type", lambda s: s.isin(BRK).mean()), off=("pitch_type", lambda s: s.isin(OFF).mean()))
    f = P.join(C, how="inner").join(L, how="inner"); f["year"] = y; return f.reset_index()


def main():
    names = pd.read_parquet("output/pitcher_id_name_map.parquet"); names["name_clean"] = names["name"].apply(mo.statcast_name_clean); idn = names.drop_duplicates("pitcher_id").set_index("pitcher_id")["name_clean"]
    F = pd.concat([features(y) for y in (2024, 2025)], ignore_index=True); F["name_clean"] = F["pitcher"].map(idn); oc = []
    for y in (2024, 2025):
        o = pd.read_csv(f"oc_{y}_command_scores.csv"); o = o[o["pitch_type"] == "ALL"].copy(); o["name_clean"] = o["pitcher"].apply(mo.clean_name); o["ws"] = o["inferred_in"] * o["n"]; o = o.groupby("name_clean", as_index=False).agg(n_oc=("n", "sum"), ws=("ws", "sum")); o["oc"] = o["ws"] / o["n_oc"]; o["year"] = y; oc.append(o[["name_clean", "year", "n_oc", "oc"]])
    D = F.merge(pd.concat(oc), on=["name_clean", "year"], how="inner"); D = D[(D["n"] >= 300) & (D["n_oc"] >= 300)].dropna().reset_index(drop=True); print(f"{len(D)} pitcher-seasons ({D['name_clean'].nunique()} pitchers)", flush=True)
    D["g"] = 4.424373963818792 + 3.9415731433625307 * D["hvar"].clip(*HR) + 4.281096060834336 * D["vvar"].clip(*VR); y = D["oc"].to_numpy(float); w = D["n_oc"].to_numpy(float); grp = D["name_clean"].to_numpy()
    for c in ("hvar", "vvar", "hvar_c", "vvar_c", "vvarn_c"):
        D["s_" + c] = np.sqrt(D[c])
    sets = {"M1 refit (hvar, vvar)": ["hvar", "vvar"], "M2 sqrt form": ["s_hvar", "s_vvar"], "M3 conditional variances": ["hvar_c", "vvar_c"], "M4 conditional, height normalised": ["hvar_c", "vvarn_c"], "M5 M0 + conditional": ["hvar", "vvar", "hvar_c", "vvar_c", "vvarn_c"],
            "M6 robust spread (MAD)": ["hmad", "vmad"], "M7 M5 + location / zone": ["hvar", "vvar", "hvar_c", "vvar_c", "vvarn_c", "ax", "az", "inz", "heart"], "M8 M7 + mix + velocity": ["hvar", "vvar", "hvar_c", "vvar_c", "vvarn_c", "ax", "az", "inz", "heart", "fb", "brk", "off", "velo"]}
    ALL = sets["M8 M7 + mix + velocity"]; models = {"M0 current formula (frozen)": None, **{k: ("ols", v) for k, v in sets.items()}, "M9 ridge on everything": ("ridge", ALL), "M10 gradient boosting on everything": ("gbm", ALL)}
    def metr(yt, p, ww):
        e = yt - p; return {"R2": float(1 - np.sum(ww * e ** 2) / np.sum(ww * (yt - np.average(yt, weights=ww)) ** 2)), "RMSE": float(np.sqrt(np.average(e ** 2, weights=ww)))}
    def fitpred(kind, cols, tr, te):
        X = D[cols].to_numpy(float); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd
        if kind == "ols":
            A = np.column_stack([np.ones(tr.sum()), Z[tr]]); sw = np.sqrt(w[tr]); b = np.linalg.lstsq(A * sw[:, None], y[tr] * sw, rcond=None)[0]; return np.column_stack([np.ones(te.sum()), Z[te]]) @ b
        if kind == "ridge":
            return Ridge(alpha=3.0).fit(Z[tr], y[tr], sample_weight=w[tr]).predict(Z[te])
        return GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.8, random_state=0).fit(Z[tr], y[tr], sample_weight=w[tr]).predict(Z[te])
    rows = []; oof = {}
    for lab, spec in models.items():
        p = np.zeros(len(D))
        for trn, tst in GroupKFold(5).split(D, y, grp):
            tm = np.zeros(len(D), bool); tm[trn] = True; sm = np.zeros(len(D), bool); sm[tst] = True; p[sm] = D["g"].to_numpy()[sm] if spec is None else fitpred(spec[0], spec[1], tm, sm)
        t24 = (D["year"] == 2024).to_numpy(); t25 = ~t24; p25 = D["g"].to_numpy()[t25] if spec is None else fitpred(spec[0], spec[1], t24, t25); oof[lab] = p
        rows.append({"model": lab, "n features": 2 if spec is None else len(spec[1]), **{f"CV {k}": v for k, v in metr(y, p, w).items()}, **{f"2024->2025 {k}": v for k, v in metr(y[t25], p25, w[t25]).items()}})
    T = pd.DataFrame(rows); pd.set_option("display.width", 230); print("\nFIT TO OPENCOMMAND 'ALL' INFERRED MISS (out of sample)\n" + T.round(4).to_string(index=False), flush=True)
    # standardised ridge weights (what carries the signal)
    Z = (D[ALL] - D[ALL].mean()) / D[ALL].std(); r = Ridge(alpha=3.0).fit(Z, y, sample_weight=w); print("\nridge weights, inches per SD of the feature:\n" + pd.Series(r.coef_, index=ALL).round(3).to_string(), flush=True)
    # ---- uncertainty: current formula and the best candidate
    best = T.sort_values("CV RMSE").iloc[0]["model"]; res = {}
    for lab in ("M0 current formula (frozen)", best):
        p = oof[lab]; e = y - p; q = lambda a: float(np.quantile(np.abs(e), a))
        slope = float(np.polyfit(p, y, 1)[0]); bins = pd.cut(D["n_oc"], [0, 600, 1200, 1e9], labels=["300-600 tracked", "600-1200", "1200+"]); ter = pd.Series(pd.qcut(p, 3, labels=["best third (low score)", "middle", "worst third (high score)"]))
        out = {"residual SD (in)": float(np.sqrt(np.average(e ** 2, weights=w))), "|error| 50% / 80% / 90% (in)": [q(.5), q(.8), q(.9)], "OpenCommand mean / SD (in)": [float(np.average(y, weights=w)), float(np.sqrt(np.average((y - np.average(y, weights=w)) ** 2, weights=w)))], "calibration slope of OC on the score": slope,
               "residual SD by tracked pitches": {str(k): float(np.sqrt(np.average(e[(bins == k).to_numpy()] ** 2, weights=w[(bins == k).to_numpy()]))) for k in bins.cat.categories}, "residual SD by score third": {str(k): float(np.sqrt(np.average(e[(ter == k).to_numpy()] ** 2, weights=w[(ter == k).to_numpy()]))) for k in ter.cat.categories}}
        res[lab] = out; print(f"\nUNCERTAINTY, {lab}:\n" + json.dumps(out, indent=1), flush=True)
    fl_sd = 1.123; e90 = res["M0 current formula (frozen)"]["|error| 50% / 80% / 90% (in)"][2]
    print(f"\nFL scale: the Command Score's SD across FL pitchers is {fl_sd:.2f} in (2027 file); the out-of-sample 90% half-width for the current formula is {e90:.2f} in ({100 * e90 / res['M0 current formula (frozen)']['OpenCommand mean / SD (in)'][0]:.1f}% of the mean MLB miss); in BB% terms at .0345 BB% per inch that is +-{100 * .0345 * e90:.1f} BB points")
    json.dump({"models": rows, "uncertainty": res, "best": best}, open("output/command_improve.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
