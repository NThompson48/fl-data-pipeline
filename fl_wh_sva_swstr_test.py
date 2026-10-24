"""
fl_wh_sva_swstr_test.py -- a PER-PITCH swinging-strike stuff model with pitch type and count  (user 2026-10-24: "P(swstr|stuff, location, pitch type, count) - P(swstr|location, pitch type, count)", "Swinging Strike% (all pitches denominator) ... count matters more for that")
swstr_sva = P(swinging strike on a PITCH | 9 stuff+location inputs, pitch type, balls, strikes) - P(swinging strike | location, pitch type, balls, strikes);  pitcher level = mean over ALL his pitches.  Network (PyTorch MLP 128-64-32 on the 9 inputs + balls + strikes,
pitch type through a 4-d embedding; location net 64-64 on plate_x_as, plate_z + balls + strikes + embedding), 3 seeds, 5-fold GroupKFold BY PITCHER, each season separately (2025 tuning-style, 2026 confirmation-style).
COUNT is rebuilt from the pitch sequence inside each plate appearance (balls = prior ball calls; strikes = min(2, prior called/swinging strikes and fouls)); checked against the terminal pitch (strikeouts should arrive at 2 strikes, walks at 3 balls).  Pitches with call 'undefined' are dropped.
VARIANTS (same pitches within a row of the table): [swing] current production-style model, swings only, 9 inputs (P(whiff | swing));  [pp_plain] per-pitch swinging-strike, 9 inputs, no pitch type or count (isolates the denominator change);  [pp_ct] per-pitch + pitch type + count (the user's spec).
SCORES: out-of-fold log-loss of the full model (3-seed ensemble), stuff increment (loc - full), pitcher-level reliability (alternating games, >= 150 pitches per half; Spearman-Brown), pitcher-level correlation between variants, and the increment by pitch type.
DECISION RULE (pre-stated): a per-pitch variant goes on to the K% harness only if its reliability (S-B) is at least the swing model's in BOTH seasons; the K% harness (fl_wh_sva_prior_k_test style) then decides adoption.   Output: output/wh_sva_swstr_test.json
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

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_nn_pilot as pilot
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
SEEDS = (1, 2, 3); BALL = {"ball_called", "ball_in_dirt", "ball_intentional"}; STRIKE = {"strike_called", "strike_swinging", "foul_not_fieldable", "foul_fieldable"}


def add_count(raw):
    r = raw.sort_values(["game_id", "pa_id", "pitch_no"], kind="mergesort").copy(); key = [r["game_id"].astype(str), r["pa_id"]]
    b = r["pitch_call"].isin(BALL).astype(int); s = r["pitch_call"].isin(STRIKE).astype(int)
    r["balls"] = b.groupby(key).cumsum() - b; r["strikes"] = np.minimum(2, s.groupby(key).cumsum() - s); r["balls"] = np.minimum(3, r["balls"]); return r


def check_count(r):
    t = r[r["is_terminal"]]; k = t[t["k_or_bb"] == "strikeout"]; w = t[t["k_or_bb"] == "walk"]
    print(f"count check: first pitches 0-0: {((r['pitch_no'] == r.groupby(['game_id', 'pa_id'])['pitch_no'].transform('min')) & (r['balls'] == 0) & (r['strikes'] == 0)).sum() / (r['pitch_no'] == r.groupby(['game_id', 'pa_id'])['pitch_no'].transform('min')).sum():.4f}; "
          f"strikeouts arriving at 2 strikes {np.mean(k['strikes'] == 2):.4f} (n={len(k):,}); walks arriving at 3 balls {np.mean(w['balls'] == 3):.4f} (n={len(w):,})", flush=True)


def job(tag, cfg, seed, k, tr, te, X, y, g, C):
    torch.set_num_threads(2)
    if C is None:
        p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg=cfg)
    else:
        p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg={**cfg, "n_cat": int(C.max()) + 1}, cat_tr=C[tr], cat_te=C[te])
    return tag, seed, te, p


def run_variant(pop, label, full_cols, loc_cols, cat, raw_season, folds_seed=0):
    y = pop["target"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); Xf = pop[full_cols].astype(float).to_numpy(); Xl = pop[loc_cols].astype(float).to_numpy()
    C = pd.Categorical(pop["pitch_type"].fillna("unk")).codes.astype(int) if cat else None
    folds = list(GroupKFold(5).split(Xf, y, g)); t0 = time.time()
    jobs = [delayed(job)(tag, cfg, s, k, tr, te, X, y, g, C) for tag, X, cfg in (("full", Xf, {}), ("loc", Xl, {"hidden": (64, 64)})) for s in SEEDS for k, (tr, te) in enumerate(folds)]
    O = {}
    for tag, s, te, p in Parallel(n_jobs=8, backend="loky")(jobs):
        O.setdefault((tag, s), np.zeros(len(y)))[te] = p
    full = np.mean([O[("full", s)] for s in SEEDS], axis=0); loc = np.mean([O[("loc", s)] for s in SEEDS], axis=0); clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6)
    res = {"n": len(y), "base_rate": float(y.mean()), "logloss_full": log_loss(y, clip(full)), "logloss_loc": log_loss(y, clip(loc)), "stuff_increment": log_loss(y, clip(loc)) - log_loss(y, clip(full)), "auc_full": roc_auc_score(y, full)}
    sva = full - loc; rel = pilot.reliability(raw_season, {label: sva}, pop)[label]; res.update(rel)
    print(f"   {label:9s} n={len(y):,} rate {y.mean():.4f}  full {res['logloss_full']:.5f}  increment {res['stuff_increment']:.5f}  AUC {res['auc_full']:.4f}  reliability S-B {rel['full_SB']:.4f}  n0 {rel['n0_pitches']:.0f} pitches  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    lv = pd.Series(sva).groupby(g).mean(); pt = pop["pitch_type"].fillna("unk").to_numpy(); d = pd.DataFrame({"pt": pt, "inc": -(y * np.log(clip(full)) + (1 - y) * np.log(1 - clip(full))) * -1}).iloc[:0]
    ll = lambda p: -(y * np.log(clip(p)) + (1 - y) * np.log(1 - clip(p))); inc = pd.DataFrame({"pt": pt, "inc": ll(loc) - ll(full)}).groupby("pt")["inc"].agg(["mean", "size"]); res["by_pitch_type"] = inc[inc["size"] >= 3000]["mean"].to_dict()
    return res, lv


def season(raw, yr):
    s = raw[raw["year"] == yr]; pit = s[(s["pitch_call"] != "undefined")].dropna(subset=fp.FULL_COLS + ["pitcher_id", "date", "game_id", "pa_id", "pitch_no", "balls", "strikes"]).reset_index(drop=True)
    pit["target"] = (pit["is_swing"] & pit["is_whiff"]).astype(int); sw = pit[pit["is_swing"]].copy().assign(target=lambda d: d["is_whiff"].astype(int)).reset_index(drop=True)
    print(f"\n== {yr}: {len(pit):,} pitches ({int(pit['target'].sum()):,} swinging strikes, {pit['target'].mean():.4f}), {len(sw):,} swings, {pit['pitcher_id'].nunique()} pitchers", flush=True)
    out, lv = {}, {}
    CNT = ["balls", "strikes"]
    for label, pop, fc, lc, cat in (("swing", sw, fp.FULL_COLS, fp.LOC_COLS, False), ("pp_plain", pit, fp.FULL_COLS, fp.LOC_COLS, False), ("pp_ct", pit, fp.FULL_COLS + CNT, fp.LOC_COLS + CNT, True)):
        out[label], lv[label] = run_variant(pop, label, fc, lc, cat, s, 0)
    L = pd.DataFrame(lv).dropna(); print("   pitcher-level correlation between variants:\n" + L.corr().round(3).to_string(), flush=True); out["level_corr"] = L.corr().round(4).to_dict()
    return out


def main():
    raw = add_count(ppf.exclude(fp.load_fl_pitches_cached())); check_count(raw); out = {}
    for yr in (2025, 2026):
        out[yr] = season(raw, yr)
    print("\nRULE CHECK (per-pitch variant reliability S-B >= the swing model's in BOTH seasons):")
    for v in ("pp_plain", "pp_ct"):
        ok = all(out[yr][v]["full_SB"] >= out[yr]["swing"]["full_SB"] for yr in (2025, 2026))
        verdict = "goes on to the K% harness" if ok else "no"
        print(f"  {v}: " + "; ".join(f"{yr} {out[yr][v]['full_SB']:.4f} vs {out[yr]['swing']['full_SB']:.4f}" for yr in (2025, 2026)) + f" -> {verdict}")
    json.dump(out, open("output/wh_sva_swstr_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
