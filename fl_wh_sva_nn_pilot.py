"""
fl_wh_sva_nn_pilot.py -- PILOT: would a neural network estimate the stuff signal (wh_sva) better than the production XGBoost?  (user 2026-10-21: "a neural network for wh_sva due to the
extreme nonlinearity and interactions present")

Production wh_sva (fl_pitcher_features.compute_pooled_wh_sva) = P(whiff | swing, shape + location)_oof - P(whiff | swing, location)_oof, both XGBoost (depth 4, 100 trees), 5-fold GroupKFold BY PITCHER so
no pitcher's own pitches inform his prediction; pitcher level = mean per-pitch difference.  Inputs (9): effective velo, ivb, arm-side horizontal break, spin rate, plate_x (arm-side), plate_z, extension, release
height, release side.  Location-only model: plate_x, plate_z.  Population here = the harness's: position players excluded, swings with all 9 inputs (226,750 swings, 817 pitchers, whiff rate .233).
Estimators (same cross-fit, same 9 inputs, sva = full - loc in every case, so only the model family changes):
  prod     XGBoost as production (control)
  xgb_deep XGBoost depth 8, 400 trees, learning rate .05 (control: is 'more flexible trees' enough, or does the family matter?)
  nn       MLP 128-64-32 (location model 64-64), ReLU, L2 1e-4, Adam 1e-3, batch 1024, early stopping on a random 10% of the TRAINING fold (patience 6), inputs standardised on the training fold; ONE seed
  nn3      the same MLP averaged over 3 seeds (neural nets differ run to run: averaging removes initialisation noise)
  blend    mean of the prod and nn3 per-pitch sva
Hyper-parameters fixed in advance (no tuning on the held-out folds).
Reported: (1) per-pitch out-of-fold log-loss and AUC of the full and location models, and the 'stuff increment' = loc log-loss - full log-loss (how much the shape inputs add out of fold);
(2) per-pitch sva SD and its correlation with prod; (3) pitcher-level reliability (alternating games, pitchers >= 150 pitches per half, Spearman-Brown, implied n0 in pitches) and pitcher-level correlation
with production; (4) concurrent correlation of the pitcher level with K% (>= 100 PA).
DECISION RULE for spending the hours on the downstream test (frozen K% model refit with the variant's level, fold 2 = 2H2026, standard weights, weighted RMSE AND MAE, pitcher bootstrap; fl_wh_sva_variants_test.py
protocol): proceed with a candidate if its full-model log-loss beats prod by >= .0005 OR its pitcher-level reliability beats prod by >= .03.  Adoption (separate, later) = RMSE and MAE both better with P(better) >= .90.
Output: output/wh_sva_nn_pilot.json.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

import fl_pitcher_features as fp
import fl_position_player_flag as ppf

warnings.filterwarnings("ignore")
DEEP = dict(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=3, tree_method="hist")
PROD = {**fp.XGB_PARAMS, "n_jobs": 3}
SEEDS = (1, 2, 3)


def _mlp(Xtr, ytr, Xte, seed, hidden):
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-4, batch_size=1024, learning_rate_init=1e-3, max_iter=80, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=6, random_state=seed).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1], m.n_iter_


def fold_job(k, tr, te, X_loc, X_full, y):
    out = {"k": k, "te": te}; t0 = time.time()
    with threadpool_limits(limits=3):
        for tag, X in (("loc", X_loc), ("full", X_full)):
            out[f"prod_{tag}"] = XGBClassifier(**PROD).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            out[f"deep_{tag}"] = XGBClassifier(**DEEP).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            hidden = (64, 64) if tag == "loc" else (128, 64, 32); ps, its = [], []
            for s in SEEDS:
                p, it = _mlp(X[tr], y[tr], X[te], s, hidden); ps.append(p); its.append(it)
            out[f"nn_{tag}"] = ps[0]; out[f"nn3_{tag}"] = np.mean(ps, axis=0); out[f"epochs_{tag}"] = its
    out["secs"] = time.time() - t0; return out


def reliability(raw, sva_by_row, pop):
    """alternating games, pitchers >= 150 pitches per half; level = mean per-pitch sva over swings (same as fl_wh_sva_variants_test.reliability)."""
    g = raw.drop_duplicates(["game_id", "pitcher_id"])[["game_id", "pitcher_id", "date"]].sort_values("date"); g["half"] = g.groupby("pitcher_id").cumcount() % 2
    thrown = raw.merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"]).groupby(["pitcher_id", "half"]).size().unstack()
    res = {}
    for name, sva in sva_by_row.items():
        d = pop[["pitcher_id", "game_id"]].assign(sva=sva).merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"])
        lv = d.groupby(["pitcher_id", "half"])["sva"].mean().unstack(); ok = (thrown.reindex(lv.index).min(axis=1) >= 150) & lv.notna().all(axis=1)
        r = float(np.corrcoef(lv.loc[ok, 0], lv.loc[ok, 1])[0, 1]); nh = float(thrown.reindex(lv.index)[ok].mean(axis=1).mean())
        res[name] = {"split_half_r": r, "full_SB": 2 * r / (1 + r), "n0_pitches": nh * (1 - r) / r if r > 0 else None, "pitchers": int(ok.sum())}
    return res


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached())
    pop = raw[raw["is_swing"]].dropna(subset=fp.FULL_COLS + ["pitcher_id", "is_whiff", "date", "game_id", "pa_id", "pitch_no"]).copy().reset_index(drop=True)
    y = pop["is_whiff"].astype(int).to_numpy(); X_loc = pop[fp.LOC_COLS].astype(float).to_numpy(); X_full = pop[fp.FULL_COLS].astype(float).to_numpy(); grp = pop["pitcher_id"].to_numpy()
    print(f"{len(pop):,} swings, {pd.Series(grp).nunique()} pitchers, whiff rate {y.mean():.4f}", flush=True)
    folds = list(GroupKFold(n_splits=5).split(X_full, y, grp)); t0 = time.time()
    outs = Parallel(n_jobs=5, backend="loky")(delayed(fold_job)(k, tr, te, X_loc, X_full, y) for k, (tr, te) in enumerate(folds))
    print(f"all folds done in {(time.time() - t0) / 60:.1f} min; per-fold seconds {[round(o['secs']) for o in outs]}; MLP epochs (full model, 3 seeds) {[o['epochs_full'] for o in outs]}", flush=True)
    P = {f"{m}_{t}": np.zeros(len(y)) for m in ("prod", "deep", "nn", "nn3") for t in ("loc", "full")}
    for o in outs:
        for key in P:
            P[key][o["te"]] = o[key]
    S = {"prod": P["prod_full"] - P["prod_loc"], "xgb_deep": P["deep_full"] - P["deep_loc"], "nn": P["nn_full"] - P["nn_loc"], "nn3": P["nn3_full"] - P["nn3_loc"]}
    S["blend"] = 0.5 * (S["prod"] + S["nn3"])
    rows = {}
    for lab, fk, lk in (("prod", "prod_full", "prod_loc"), ("xgb_deep", "deep_full", "deep_loc"), ("nn", "nn_full", "nn_loc"), ("nn3", "nn3_full", "nn3_loc")):
        ll_f, ll_l = log_loss(y, np.clip(P[fk], 1e-6, 1 - 1e-6)), log_loss(y, np.clip(P[lk], 1e-6, 1 - 1e-6))
        rows[lab] = {"logloss_full": ll_f, "logloss_loc": ll_l, "stuff_increment": ll_l - ll_f, "auc_full": roc_auc_score(y, P[fk]), "auc_loc": roc_auc_score(y, P[lk])}
    rows["blend"] = {}
    for lab, s in S.items():
        rows[lab].update({"sva_sd": float(s.std()), "corr_with_prod_pitch": float(np.corrcoef(s, S["prod"])[0, 1])})
    rel = reliability(raw, S, pop)
    lv = {k: pd.Series(v).groupby(grp).mean() for k, v in S.items()}
    term = raw[raw["is_terminal"]]; kr = term.groupby("pitcher_id").agg(PA=("is_terminal", "size"), K=("k_or_bb", lambda s: (s == "strike_out" if False else s == "strikeout").mean())); kr = kr[kr["PA"] >= 100]
    for lab in S:
        rows[lab].update(rel[lab]); common = lv[lab].index.intersection(lv["prod"].index)
        rows[lab]["corr_with_prod_pitcher_level"] = float(np.corrcoef(lv[lab].loc[common], lv["prod"].loc[common])[0, 1]); j = kr.join(lv[lab].rename("lv"), how="inner"); rows[lab]["corr_with_K_same_period"] = float(np.corrcoef(j["K"], j["lv"])[0, 1])
    T = pd.DataFrame(rows).T[["logloss_full", "logloss_loc", "stuff_increment", "auc_full", "auc_loc", "sva_sd", "corr_with_prod_pitch", "split_half_r", "full_SB", "n0_pitches", "corr_with_prod_pitcher_level", "corr_with_K_same_period"]]
    pd.set_option("display.width", 250); print("\n" + T.round(4).to_string())
    base = T.loc["prod"]; print("\nDECISION RULE (full-model log-loss better by >= .0005 OR reliability better by >= .03 vs prod):")
    for lab in ("xgb_deep", "nn", "nn3", "blend"):
        d_ll = base["logloss_full"] - T.loc[lab, "logloss_full"] if lab != "blend" else np.nan; d_rel = T.loc[lab, "full_SB"] - base["full_SB"]
        go = (d_ll >= 0.0005 if d_ll == d_ll else False) or d_rel >= 0.03; print(f"  {lab:9s} d logloss {d_ll:+.5f}  d reliability {d_rel:+.3f}  -> {'PROCEED to the K% harness' if go else 'no'}")
    json.dump(T.to_dict("index"), open("output/wh_sva_nn_pilot.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
