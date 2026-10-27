"""
fl_outlier_within.py -- WITHIN the outlier pool, do a pitch's characteristics relate to how that pitch performs (Whiff% = whiffs per swing)?  Per pitch type.  (user 2026-10-27: "Is there any correlation of pitch characteristics to performance within the outlier pool? ... per-type";
"specifically how the pitch itself performs by Whiff%")
POOL: per pitch type, the most unusual 10% of pitches by the held-out outlier score (z = robust Mahalanobis, zi = isolation forest; fl_outlier_test's definition: 7 standardised characteristics, a distribution fitted on a 10,000-pitch sample from OTHER pitchers, 5 pitcher folds), pooled 2025-26 regular season.
OUTCOME: Whiff% = whiff per swing, so only pitches that were SWUNG AT enter the regressions (the pool is defined on all pitches).  Secondary: SwStr% (swinging strike per pitch) on all pitches, and a LOCATION-ADJUSTED Whiff% (whiff minus the cross-fitted P(whiff | swing, plate location, count, pitch type)), because shape and location are entangled.
FOR EACH TYPE: (1) Whiff% of pool vs non-pool swings;  (2) linear-probability regression of whiff on the 7 characteristics (standardised within the type; coefficient = percentage points of Whiff% per SD), pitcher-clustered SE, fitted in the POOL and in the NON-POOL;  (3) joint Wald test that the 7 slopes differ between pool and non-pool;
(4) PREDICTION inside the pool (5 pitcher folds): AUC and log-loss gain over the pool average for a model fitted within the pool, and for a general model fitted on non-pool pitches of other pitchers.   Isolation-forest pool is primary; the Mahalanobis pool is reported as a robustness line.   Output: output/outlier_within.json
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf

warnings.filterwarnings("ignore")
FEATS = ["effective_velo", "ivb", "horz_break_as", "spin_rate", "extension", "release_pos_z", "rel_side_as"]; SHORT = ["velo", "ivb", "horz brk", "spin", "extension", "release ht", "release side"]
TYPES = ["four_seam", "sinker", "cutter", "slider", "sweeper", "curveball", "changeup", "splitter"]; K = 5; TOP = 0.10


def pitch_scores(d, seed=0):
    d = d.copy(); d["fold"] = d["pid"].map(dict(zip(*(lambda u: (u, np.random.default_rng(seed).permutation(len(u)) % K))(np.unique(d["pid"]))))); d["z"] = np.nan; d["zi"] = np.nan
    for t in TYPES:
        sub = d[d["pitch_type"] == t]
        if len(sub) < 1500:
            continue
        for k in range(K):
            tr = sub[sub["fold"] != k]; te = sub[sub["fold"] == k]
            if len(tr) < 800 or len(te) == 0:
                continue
            samp = tr.sample(min(len(tr), 10000), random_state=seed)[FEATS]; mu, sd = samp.mean(), samp.std().replace(0, 1.0); Xs = (samp - mu) / sd; Xte = (te[FEATS] - mu) / sd
            mcd = MinCovDet(support_fraction=0.9, random_state=seed).fit(Xs); ltr = np.log(mcd.mahalanobis(Xs) + 1e-9); med = np.median(ltr); mad = 1.4826 * np.median(np.abs(ltr - med)); d.loc[te.index, "z"] = (np.log(mcd.mahalanobis(Xte) + 1e-9) - med) / mad
            iso = IsolationForest(n_estimators=300, max_samples=512, random_state=seed).fit(Xs); atr = -iso.score_samples(Xs); amed = np.median(atr); amad = 1.4826 * np.median(np.abs(atr - amed)); d.loc[te.index, "zi"] = (-iso.score_samples(Xte) - amed) / amad
    return d


def location_baseline(d):
    """cross-fitted P(whiff | swing, plate location, balls, strikes, pitch type) on swings, by pitcher folds; returns a Series aligned to d (NaN for non-swings)."""
    sw = d["is_swing"].to_numpy(bool); oh = pd.get_dummies(d["pitch_type"]).astype(float).to_numpy(); X = np.column_stack([d[["plate_x_as", "plate_z", "balls", "strikes"]].astype(float).to_numpy(), oh]); y = d["whiff"].to_numpy(int); p = np.full(len(d), np.nan)
    for tr, te in GroupKFold(K).split(X, y, d["pid"].to_numpy()):
        trs = tr[sw[tr]]; m = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(X[trs], y[trs]); p[te] = m.predict_proba(X[te])[:, 1]
    return pd.Series(np.where(sw, p, np.nan), index=d.index)


def ols_cluster(X, y, g):
    codes = pd.factorize(g)[0]; r = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="cluster", cov_kwds={"groups": codes}); return r


def analyse(sub, pool_mask, outcome, label):
    """regressions inside / outside the pool on the standardised characteristics (swings only for a whiff outcome)."""
    Z = ((sub[FEATS] - sub[FEATS].mean()) / sub[FEATS].std()).to_numpy(); y = sub[outcome].to_numpy(float); g = sub["pid"].to_numpy(); ok = np.isfinite(y); out = {}
    for nm, m in (("pool", pool_mask & ok), ("non-pool", (~pool_mask) & ok)):
        r = ols_cluster(Z[m], y[m], g[m]); out[nm] = {"n": int(m.sum()), "pitchers": int(len(np.unique(g[m]))), "rate": float(y[m].mean()), "coef_pp": (100 * r.params[1:]).tolist(), "t": r.tvalues[1:].tolist()}
    m = ok; pm = pool_mask[m].astype(float); X = np.column_stack([Z[m], pm, Z[m] * pm[:, None]]); r = ols_cluster(X, y[m], g[m]); R = np.zeros((7, len(r.params))); R[np.arange(7), 1 + 8 + np.arange(7)] = 1
    out["p_slopes_differ"] = float(r.wald_test(R, scalar=True).pvalue); return out


def predict_in_pool(sub, pool_mask, outcome):
    """5 pitcher folds: AUC / log-loss gain over the training pool mean for (a) a model fitted within the pool, (b) a general model fitted on NON-pool pitches of other pitchers; evaluated on the held-out pitchers' pool swings."""
    ok = np.isfinite(sub[outcome].to_numpy(float)); sw = sub[ok]; pm = pool_mask[ok]; Z = ((sw[FEATS] - sw[FEATS].mean()) / sw[FEATS].std()).to_numpy(); y = sw[outcome].to_numpy(int); g = sw["pid"].to_numpy()
    pa, pb, pc, yy = np.full(len(y), np.nan), np.full(len(y), np.nan), np.full(len(y), np.nan), y
    for tr, te in GroupKFold(K).split(Z, y, g):
        trp = tr[pm[tr]]; trn = tr[~pm[tr]]; tep = te[pm[te]]
        if len(tep) == 0 or len(np.unique(y[trp])) < 2:
            continue
        pa[tep] = LogisticRegression(C=1.0, max_iter=500).fit(Z[trp], y[trp]).predict_proba(Z[tep])[:, 1]; pb[tep] = LogisticRegression(C=1.0, max_iter=500).fit(Z[trn], y[trn]).predict_proba(Z[tep])[:, 1]; pc[tep] = y[trp].mean()
    m = np.isfinite(pa); c = lambda p: np.clip(p, 1e-6, 1 - 1e-6)
    return {"n_eval": int(m.sum()), "AUC within-pool model": float(roc_auc_score(yy[m], pa[m])), "AUC general model": float(roc_auc_score(yy[m], pb[m])), "logloss gain within-pool": float(log_loss(yy[m], c(pc[m])) - log_loss(yy[m], c(pa[m]))), "logloss gain general": float(log_loss(yy[m], c(pc[m])) - log_loss(yy[m], c(pb[m])))}


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); win = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025), fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    d = fp.add_count(win[win["pitch_call"] != "undefined"]); d = d[d["pitch_type"].isin(TYPES)].dropna(subset=FEATS + ["plate_x_as", "plate_z", "balls", "strikes", "pitcher_id"]).reset_index(drop=True); d["pid"] = d["pitcher_id"].astype(str)
    d["is_swing"] = d["is_swing"].fillna(False).astype(bool); d["whiff"] = (d["is_swing"] & d["is_whiff"].fillna(False).astype(bool)).astype(int); d["swstr"] = d["whiff"]
    print(f"{len(d):,} pitches of the 8 main types, {int(d['is_swing'].sum()):,} swings; scoring outlier-ness against other pitchers ...", flush=True); d = pitch_scores(d); d = d.dropna(subset=["z", "zi"]).reset_index(drop=True)
    d["p_loc"] = location_baseline(d); d["whiff_adj"] = np.where(d["is_swing"], d["whiff"] - d["p_loc"], np.nan); d["whiff_sw"] = np.where(d["is_swing"], d["whiff"], np.nan); print("location baseline done", flush=True)
    res = {}; pd.set_option("display.width", 250)
    for t in TYPES + ["ALL (types pooled)"]:
        sub = d if t.startswith("ALL") else d[d["pitch_type"] == t]; sub = sub.reset_index(drop=True)
        if len(sub) < 3000:
            continue
        pool_if = (sub["zi"] >= sub["zi"].quantile(1 - TOP)).to_numpy() if not t.startswith("ALL") else (sub.groupby("pitch_type")["zi"].transform(lambda s: s >= s.quantile(1 - TOP))).to_numpy(bool)
        pool_mc = (sub["z"] >= sub["z"].quantile(1 - TOP)).to_numpy() if not t.startswith("ALL") else (sub.groupby("pitch_type")["z"].transform(lambda s: s >= s.quantile(1 - TOP))).to_numpy(bool)
        if t.startswith("ALL"):                                                       # characteristics standardised WITHIN each type for the pooled run
            for f in FEATS:
                sub[f] = sub.groupby("pitch_type")[f].transform(lambda s: (s - s.mean()) / s.std())
        R = {"if": analyse(sub, pool_if, "whiff_sw", "whiff"), "mcd": analyse(sub, pool_mc, "whiff_sw", "whiff"), "if_adj": analyse(sub, pool_if, "whiff_adj", "whiff adjusted"), "swstr_if": analyse(sub.assign(allp=sub["swstr"].astype(float)), pool_if, "allp", "swstr"), "predict_if": predict_in_pool(sub, pool_if, "whiff_sw")}
        res[t] = R; a = R["if"]
        print(f"\n=== {t}: pool = top {int(TOP * 100)}% by isolation-forest score  |  Whiff%: pool {100 * a['pool']['rate']:.1f}% ({a['pool']['n']:,} swings, {a['pool']['pitchers']} pitchers) vs non-pool {100 * a['non-pool']['rate']:.1f}% ({a['non-pool']['n']:,})  |  slopes differ (joint p) {a['p_slopes_differ']:.3f}")
        tab = pd.DataFrame({"characteristic": SHORT, "pool: pp Whiff% per SD": a["pool"]["coef_pp"], "pool t": a["pool"]["t"], "non-pool: pp per SD": a["non-pool"]["coef_pp"], "non-pool t": a["non-pool"]["t"], "pool, location-adj pp": R["if_adj"]["pool"]["coef_pp"], "adj t": R["if_adj"]["pool"]["t"]})
        print(tab.round(2).to_string(index=False)); p = R["predict_if"]
        print(f"   prediction inside the pool (held-out pitchers, n {p['n_eval']:,} pool swings): AUC within-pool model {p['AUC within-pool model']:.3f}, general (non-pool) model {p['AUC general model']:.3f}; log-loss gain over the pool average: within-pool {p['logloss gain within-pool']:+.5f}, general {p['logloss gain general']:+.5f}")
        cv = np.corrcoef(R["if"]["pool"]["coef_pp"], R["mcd"]["pool"]["coef_pp"])[0, 1]; print(f"   robustness: Mahalanobis pool vs isolation-forest pool, correlation of the 7 pool coefficients {cv:.2f}; Mahalanobis pool Whiff% {100 * R['mcd']['pool']['rate']:.1f}%, slopes differ p {R['mcd']['p_slopes_differ']:.3f}", flush=True)
    json.dump(res, open("output/outlier_within.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
