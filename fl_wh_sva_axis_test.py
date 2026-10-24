"""
fl_wh_sva_axis_test.py -- does adding SPIN AXIS to the stuff model (wh_sva) help?  (user 2026-10-21: "Can we add it to the model, encoded and mirrored?")

ENCODING (verified on the data, 517,799 pitches pulled to output/spin_axis_supplement.parquet): spin_axis is 0-360 degrees; left-handers are the mirror image of right-handers about the vertical, so
  axis_as = (360 - spin_axis) mod 360 for left-handed pitchers, spin_axis for right-handed  ('arm-side' axis, like plate_x_as / horz_break_as / rel_side_as),   then   axis_sin = sin(axis_as),  axis_cos = cos(axis_as)
(circular, so 359 and 1 degrees are neighbours).  Check: circular mean axis, right vs 360 - left: four-seam 211 vs 212, changeup 246 vs 243, sinker 238 vs 236, curveball 44 vs 47, sweeper 85 vs 79, cutter 176 vs 186;
after mirroring, corr(axis_sin, arm-side horizontal break) -.94 (L) / -.93 (R) and corr(axis_cos, ivb) -.89 / -.88: the same for both hands.  98.7% of pitches have an axis; baseline and axis models use the SAME swings.
TEST (pre-stated; same protocol as the rest of the wh_sva work): each season separately (2025 tuning-style, 2026 confirmation-style), swings with all 9 inputs AND an axis, 5-fold GroupKFold BY PITCHER; full model with 9 inputs
(baseline) vs 11 inputs (+ axis_sin, axis_cos); location-only model unchanged (2 inputs).  Estimators: production XGBoost (fp.XGB_PARAMS) and the PyTorch network (128-64-32, unregularised baseline, 3 seeds; the location net 64-64).
Scores: out-of-fold per-pitch log-loss of the full model (XGBoost; torch single-seed mean and 3-seed ensemble), AUC, stuff increment (loc - full log-loss), pitcher-level reliability of sva = full - loc (alternating games, >= 150
pitches per half), correlation between the baseline and axis pitcher levels, and the log-loss change by pitch type (ensemble / XGBoost).
RULE (pre-stated): adding the axis 'helps' if the full-model log-loss improves by at least one noise unit (.00023, the SE of a 2-seed mean in the 2025 sweeps) for the XGBoost model AND the torch ensemble in BOTH seasons, and the
reliability does not fall by more than .01.  Passing = go on to the downstream tests (whiff-rate target, first-N, K% harness); it is not production adoption.
Output: output/wh_sva_axis_test.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_nn_pilot as pilot
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
KEY9 = ["game_id", "pa_id", "pitch_no", "plate_x", "plate_z", "effective_velo", "spin_rate", "ivb", "pitcher_id"]
AXIS = ["axis_sin", "axis_cos"]; SEEDS = (1, 2, 3); NOISE = 0.00023


def add_spin_axis(raw):
    """merge the pulled spin axis onto the pitch cache and add the mirrored, circular encoding (the same function would go into fl_pitcher_features on adoption)."""
    s = pd.read_parquet("output/spin_axis_supplement.parquet")[KEY9 + ["spin_axis"]]
    for c in ("game_id", "pitcher_id"):
        s[c] = s[c].astype(str)
    r = raw.copy()
    for c in ("game_id", "pitcher_id"):
        r[c] = r[c].astype(str)
    assert not s.duplicated(KEY9).any(); n = len(r); r = r.merge(s, on=KEY9, how="left"); assert len(r) == n
    a = np.where(r["p_throws_clean"] == "L", (360.0 - r["spin_axis"]) % 360.0, r["spin_axis"]); a = np.where(r["p_throws_clean"].isin(["L", "R"]), a, np.nan)
    r["axis_as"] = a; r["axis_sin"] = np.sin(np.radians(a)); r["axis_cos"] = np.cos(np.radians(a)); return r


def tjob(tag, cols, cfg, seed, k, tr, te, X, y, g):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg=cfg); return tag, seed, te, p


def season(raw, yr):
    pop = raw[raw["is_swing"] & (raw["year"] == yr)].dropna(subset=fp.FULL_COLS + AXIS + ["pitcher_id", "is_whiff", "date", "game_id", "pa_id", "pitch_no"]).reset_index(drop=True)
    y = pop["is_whiff"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); X9 = pop[fp.FULL_COLS].astype(float).to_numpy(); X11 = pop[fp.FULL_COLS + AXIS].astype(float).to_numpy(); Xl = pop[fp.LOC_COLS].astype(float).to_numpy()
    folds = list(GroupKFold(5).split(X9, y, g)); t0 = time.time(); print(f"\n== {yr}: {len(pop):,} swings, {len(np.unique(g))} pitchers, whiff rate {y.mean():.4f}", flush=True)
    jobs = [delayed(tjob)(tag, None, cfg, s, k, tr, te, X, y, g) for tag, X, cfg in (("t9", X9, {}), ("t11", X11, {}), ("tloc", Xl, {"hidden": (64, 64)})) for s in SEEDS for k, (tr, te) in enumerate(folds)]
    O = {}
    for tag, s, te, p in Parallel(n_jobs=8, backend="loky")(jobs):
        O.setdefault((tag, s), np.zeros(len(y)))[te] = p
    X = {"x9": np.zeros(len(y)), "x11": np.zeros(len(y)), "xloc": np.zeros(len(y))}
    for tr, te in folds:
        for tag, M in (("x9", X9), ("x11", X11), ("xloc", Xl)):
            X[tag][te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(M[tr], y[tr]).predict_proba(M[te])[:, 1]
    print(f"   fitted in {(time.time() - t0) / 60:.1f} min", flush=True)
    clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6); ens = {t: np.mean([O[(t, s)] for s in SEEDS], axis=0) for t in ("t9", "t11", "tloc")}
    res = {}
    for lab, full, loc, singles in (("XGBoost 9 inputs", X["x9"], X["xloc"], None), ("XGBoost 11 (+axis)", X["x11"], X["xloc"], None), ("torch 9 inputs", ens["t9"], ens["tloc"], [O[("t9", s)] for s in SEEDS]), ("torch 11 (+axis)", ens["t11"], ens["tloc"], [O[("t11", s)] for s in SEEDS])):
        res[lab] = {"logloss": log_loss(y, clip(full)), "single_seed_mean": float(np.mean([log_loss(y, clip(p)) for p in singles])) if singles else np.nan, "auc": roc_auc_score(y, full), "stuff_increment": log_loss(y, clip(loc)) - log_loss(y, clip(full)), "sva": full - loc}
    rel = pilot.reliability(raw[raw["year"] == yr], {k: v["sva"] for k, v in res.items()}, pop)
    lv = {k: pd.Series(v["sva"]).groupby(g).mean() for k, v in res.items()}
    for k in res:
        res[k].update(rel[k]); res[k]["corr_with_9input_level"] = float(np.corrcoef(lv[k], lv[k.replace("11 (+axis)", "9 inputs")])[0, 1]) if "11" in k else 1.0
    T = pd.DataFrame({k: {m: v for m, v in r.items() if m != "sva"} for k, r in res.items()}).T
    pd.set_option("display.width", 250); print(T[["logloss", "single_seed_mean", "auc", "stuff_increment", "split_half_r", "full_SB", "n0_pitches", "corr_with_9input_level"]].round(5).to_string())
    pt = pop["pitch_type"].to_numpy(); rows = []
    for lab, a, b in (("XGBoost", X["x9"], X["x11"]), ("torch ensemble", ens["t9"], ens["t11"])):
        la = -(y * np.log(clip(a)) + (1 - y) * np.log(1 - clip(a))); lb = -(y * np.log(clip(b)) + (1 - y) * np.log(1 - clip(b)))
        d = pd.DataFrame({"pitch_type": pt, "gain": la - lb}).groupby("pitch_type")["gain"].agg(["mean", "size"]); d = d[d["size"] >= 1500].rename(columns={"mean": f"{lab} log-loss gain from axis"}); rows.append(d[f"{lab} log-loss gain from axis"])
    P = pd.concat(rows, axis=1).join(pd.Series(pt).value_counts().rename("swings")).sort_values("swings", ascending=False); print("\nlog-loss improvement from adding the axis, by pitch type (positive = better with axis):\n" + P.round(5).to_string())
    return T, P


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); raw = add_spin_axis(raw)
    print(f"pitches {len(raw):,}; with a usable mirrored axis {raw['axis_sin'].notna().mean():.4f}; check circular mean axis_as for four-seam right {np.degrees(np.arctan2(raw[(raw.pitch_type=='four_seam')&(raw.p_throws_clean=='R')].axis_sin.mean(), raw[(raw.pitch_type=='four_seam')&(raw.p_throws_clean=='R')].axis_cos.mean()))%360:.0f} / "
          f"left (mirrored) {np.degrees(np.arctan2(raw[(raw.pitch_type=='four_seam')&(raw.p_throws_clean=='L')].axis_sin.mean(), raw[(raw.pitch_type=='four_seam')&(raw.p_throws_clean=='L')].axis_cos.mean()))%360:.0f}", flush=True)
    out = {}
    for yr in (2025, 2026):
        T, P = season(raw, yr); out[yr] = {"table": T.reset_index().to_dict("records"), "by_pitch_type": P.reset_index().to_dict("records")}
    print("\nRULE CHECK (log-loss improvement >= %.5f for XGBoost and the torch ensemble in BOTH seasons; reliability not lower by > .01):" % NOISE)
    ok = True
    for fam, a, b in (("XGBoost", "XGBoost 9 inputs", "XGBoost 11 (+axis)"), ("torch ensemble", "torch 9 inputs", "torch 11 (+axis)")):
        for yr in (2025, 2026):
            T = pd.DataFrame(out[yr]["table"]).set_index("index"); d = T.loc[a, "logloss"] - T.loc[b, "logloss"]; dr = T.loc[b, "full_SB"] - T.loc[a, "full_SB"]; passed = d >= NOISE and dr >= -0.01; ok &= passed
            print(f"  {fam:15s} {yr}: log-loss {d:+.5f} ({d / NOISE:+.1f} noise units), reliability {dr:+.3f} -> {'pass' if passed else 'no'}")
    print("\nVERDICT:", "HELPS in both seasons: go on to the downstream tests" if ok else "does not meet the rule in every cell (see above)")
    json.dump(out, open("output/wh_sva_axis_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
