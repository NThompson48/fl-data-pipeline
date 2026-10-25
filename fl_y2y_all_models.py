"""
fl_y2y_all_models.py -- year-to-year correlation (2025 level vs 2026 level, each from its own season's cross-fit) of EVERY pitcher-level stuff measure tried, plus raw reference rates  (user 2026-10-25)
Measures: XGBoost wh_sva (production until now) | swing network wh_sva | swstr_plain (per-pitch swinging strike, 9 inputs, no pitch type or count; network) | swstr_ct (per-pitch, pitch type + count; network) | strike_loc | strike_ct (networks)
References (raw, no model): Whiff% per swing, SwStr% per pitch, K% per PA (PA >= 40 each year).   Pitcher sets: >= 150, 300 (the earlier test), 500, 800 and 1,100 pitches in EACH year (nested: higher thresholds are subsets); 95% CI by Fisher z.
Same set of pitchers within each row of the output.   Output: output/y2y_all_models.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
SEEDS = (1, 2, 3); THRESH = (150, 300, 500, 800, 1100)


def _job(tag, cfg, seed, tr, te, X, y, g):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg=cfg); return tag, seed, te, p


def swstr_plain(s):
    d = fp.add_count(s[s["pitch_call"] != "undefined"]); pop = d.dropna(subset=fp.FULL_COLS + ["pitcher_id"]).copy(); y = (pop["is_swing"] & pop["is_whiff"]).astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy()
    Xf = pop[fp.FULL_COLS].astype(float).to_numpy(); Xl = pop[fp.LOC_COLS].astype(float).to_numpy(); folds = list(GroupKFold(5).split(Xf, y, g)); jobs = []
    for tag, X, cfg in (("full", Xf, {}), ("loc", Xl, {"hidden": (64, 64)})):
        jobs += [delayed(_job)(tag, cfg, sd, tr, te, X, y, g) for sd in SEEDS for tr, te in folds]
    O = {}
    for tag, sd, te, p in Parallel(n_jobs=8, backend="loky")(jobs):
        O.setdefault((tag, sd), np.zeros(len(y)))[te] = p
    sva = np.mean([O[("full", sd)] for sd in SEEDS], axis=0) - np.mean([O[("loc", sd)] for sd in SEEDS], axis=0)
    return pd.Series(sva).groupby(g).mean()


def levels(s):
    L = {"XGBoost wh_sva": fp.compute_pooled_wh_sva(s, estimator="xgb").groupby("pitcher_id")["sva"].mean(), "swing network wh_sva": fp.compute_pooled_wh_sva(s, estimator="torch3").groupby("pitcher_id")["sva"].mean(),
         "swstr_plain": swstr_plain(s), "swstr_ct": fp.compute_pooled_wh_sva(s, estimator="swstr_ct").groupby("pitcher_id")["sva"].mean(),
         "strike_loc": fp.compute_pooled_wh_sva(s, estimator="strike_loc").groupby("pitcher_id")["sva"].mean(), "strike_ct": fp.compute_pooled_wh_sva(s, estimator="strike_ct").groupby("pitcher_id")["sva"].mean()}
    sw = s[s["is_swing"]]; t = s[s["is_terminal"]]; pa = t.groupby("pitcher_id").size(); k = (t["k_or_bb"] == "strikeout").groupby(t["pitcher_id"]).mean()
    L["raw Whiff% (per swing)"] = sw.groupby("pitcher_id")["is_whiff"].mean(); L["raw SwStr% (per pitch)"] = (s["is_swing"] & s["is_whiff"]).groupby(s["pitcher_id"]).mean(); L["raw K% (PA >= 40)"] = k[pa >= 40]
    return L, s.groupby("pitcher_id").size()


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); res = {}; D = {}
    for yr, (a, b) in {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}.items():
        D[yr] = levels(fp.filter_period(raw, a, b)); print(f"  {yr} levels done", flush=True); pd.DataFrame(D[yr][0]).join(D[yr][1].rename("n_pitches")).to_csv(f"output/y2y_levels_{yr}.csv")
    names = list(D[2025][0]); rows = []
    for lab in names:
        r = {"measure": lab}
        for thr in THRESH:
            n25, n26 = D[2025][1], D[2026][1]; j = pd.concat([D[2025][0][lab].rename("a"), D[2026][0][lab].rename("b"), n25.rename("n25"), n26.rename("n26")], axis=1).dropna(); j = j[(j["n25"] >= thr) & (j["n26"] >= thr)]
            rr = float(np.corrcoef(j["a"], j["b"])[0, 1]); z = np.arctanh(rr); se = 1 / np.sqrt(len(j) - 3); r[f"r (>= {thr} pitches)"] = rr; r[f"95% CI {thr}"] = f"[{np.tanh(z - 1.96 * se):.2f}, {np.tanh(z + 1.96 * se):.2f}]"; r[f"n {thr}"] = len(j)
        rows.append(r)
    pd.set_option("display.width", 250); T = pd.DataFrame(rows); print("\nYEAR-TO-YEAR CORRELATION OF THE PITCHER LEVEL (2025 vs 2026)\n" + T.round(3).to_string(index=False))
    json.dump(rows, open("output/y2y_all_models.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
