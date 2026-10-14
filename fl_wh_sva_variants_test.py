"""
fl_wh_sva_variants_test.py -- which stuff-value-added target and conditioning should feed the pitcher K% model?  (user 2026-10-13)
Production wh_sva = P(whiff | swing, stuff + location) - P(whiff | swing, location), no count (fl_pitcher_features.compute_pooled_wh_sva).
Six variants = 3 targets x {no count, count}:
  whiff   P(whiff | swing, ...)                        population: swings
  swstr   P(swinging strike | pitch, ...)              population: every pitch
  strike  P(strike | pitch, ...), strike = not a ball  population: every pitch  (called / swinging strike, foul, ball in play; HBP = not a strike)
  count = pre-pitch balls and strikes (rebuilt from each PA's pitch sequence; distribution matches the tracking file's own count columns within .004 per cell)
  added to BOTH the location model and the full model, so the difference is "stuff beyond location AND count".
Same estimator as production for every variant: XGBoost (XGB_PARAMS), 5-fold GroupKFold by pitcher, out-of-fold, pitcher level = mean per-pitch difference.
Evaluation = the production harness: the frozen K% model is REFIT with each variant's level (training transition full 2025 -> actual 1H2026), then fold 2 (features 2025 + 1H2026,
targets actual 2H2026), position players excluded.  Level-free (each prediction gets its own weighted mean error removed: production recentres anyway), standard inverse-variance
weights from the production variant's residuals (frozen, fl_eval_weights), weighted RMSE and MAE, pitcher bootstrap (1,000) vs production.
Also: (a) BB% model + each variant's level as an extra regressor (is stuff-for-strikes a BB% signal?), same scoring; (b) split-half reliability of the pitcher level
(alternating games, 2025-26 pooled, pitchers with >= 150 pitches per half) with Spearman-Brown and the implied n0 in pitches thrown; (c) concurrent corr with K%.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBClassifier

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_posplayer_refit_check as chk
import fl_production_projection as pp

warnings.filterwarnings("ignore")
BALL = {"ball_called", "ball_in_dirt", "ball_intentional", "automatic_ball"}
CSTR = {"strike_called", "strike_swinging", "automaticstrike"}
FOUL = {"foul_not_fieldable", "foul_fieldable", "foul_ball"}
VARIANTS = [("whiff", False), ("whiff", True), ("swstr", False), ("swstr", True), ("strike", False), ("strike", True)]
CUR = {"target": "whiff", "count": False}
_orig = fp.compute_pooled_wh_sva


def add_count_and_targets(raw):
    r = raw.sort_values(["game_id", "pa_id", "pitch_no"]).copy()
    key = (r["game_id"].astype(str) + "|" + r["pa_id"].astype(str)).to_numpy(); pc = r["pitch_call"].astype(str).to_numpy()
    b = np.zeros(len(r), int); s = np.zeros(len(r), int); cb = cs = 0; prev = None
    for i in range(len(r)):
        if key[i] != prev:
            cb = cs = 0; prev = key[i]
        b[i] = min(cb, 3); s[i] = min(cs, 2)
        if pc[i] in BALL:
            cb += 1
        elif pc[i] in CSTR or (pc[i] in FOUL and cs < 2):
            cs += 1
    r["balls"] = b; r["strikes"] = s
    r["is_swstr"] = (r["pitch_call"] == "strike_swinging").astype(int)
    r["is_strike"] = (~r["pitch_call"].isin(BALL | {"hit_by_pitch", "undefined"})).astype(int)
    return r.sort_index()


def sva_variant(df, min_rows=2000):
    """Drop-in replacement for fp.compute_pooled_wh_sva, driven by CUR."""
    tgt, use_count = CUR["target"], CUR["count"]
    pop = df[df["is_swing"]] if tgt == "whiff" else df[df["pitch_call"] != "undefined"]
    ycol = {"whiff": "is_whiff", "swstr": "is_swstr", "strike": "is_strike"}[tgt]
    loc = fp.LOC_COLS + (["balls", "strikes"] if use_count else []); full = fp.FULL_COLS + (["balls", "strikes"] if use_count else [])
    pop = pop.dropna(subset=list(dict.fromkeys(full + ["pitcher_id", ycol, "date", "game_id", "pa_id", "pitch_no"]))).copy()
    if len(pop) < min_rows:
        return pd.DataFrame(columns=["pitcher_id", "date", "game_id", "pa_id", "pitch_no", "sva"])
    y = pop[ycol].astype(int).values; g = pop["pitcher_id"].values; gkf = GroupKFold(n_splits=5); t0 = time.time()
    o_loc = cross_val_predict(XGBClassifier(**fp.XGB_PARAMS), pop[loc].astype(float).values, y, cv=gkf, groups=g, method="predict_proba", n_jobs=5)[:, 1]
    o_full = cross_val_predict(XGBClassifier(**fp.XGB_PARAMS), pop[full].astype(float).values, y, cv=gkf, groups=g, method="predict_proba", n_jobs=5)[:, 1]
    pop["sva"] = o_full - o_loc
    print(f"    [{tgt}{' + count' if use_count else ''}] n={len(pop):,} rate={y.mean():.4f} sva_sd={pop['sva'].std():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return pop[["pitcher_id", "date", "game_id", "pa_id", "pitch_no", "sva"]]


def fit_bb_with(raw, extra):
    """Production BB% fit (fl_production_projection.fit_frozen_model, BB part) with optional extra regressors; returns the fitted model + n0 bridge."""
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    d = df_tr.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score"] + extra).copy()
    w = mm.reliability_weight(d["PA_test"], n0_bb).values
    t = fp.filter_period(raw, bt.START_2025, bt.END_2025); t = t[t["is_terminal"]]; lg = (t["k_or_bb"] == "walk").mean()
    d["BB_raw"] = d["naive_BB_rate"] * (d["PA"] + n0_bb) - n0_bb * lg
    n0b = mm.fit_n0_bridge(d["BB_raw"], d["PA"], n0_bb, lg, d["fl_bb_rate_bridge"], d["BB_rate_actual"], w)
    d["blended_BB_rate"] = mm.blended_rate(d["BB_raw"], d["PA"], n0_bb, lg, n0b, d["fl_bb_rate_bridge"])
    return bt.fit_weighted(d, "BB_rate_actual", pp.BB_SPEC_BLENDED + extra, w), n0b


def reliability(raw):
    sva = sva_variant(raw); g = raw.drop_duplicates(["game_id", "pitcher_id"])[["game_id", "pitcher_id", "date"]].sort_values("date")
    g["half"] = g.groupby("pitcher_id").cumcount() % 2; sva = sva.merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"])
    thrown = raw.merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"]).groupby(["pitcher_id", "half"]).size().unstack()
    lv = sva.groupby(["pitcher_id", "half"])["sva"].mean().unstack(); ok = (thrown.reindex(lv.index).min(axis=1) >= 150) & lv.notna().all(axis=1)
    r = np.corrcoef(lv.loc[ok, 0], lv.loc[ok, 1])[0, 1]; n_half = thrown.reindex(lv.index)[ok].mean(axis=1).mean()
    full_lv = sva.groupby("pitcher_id")["sva"].mean()
    return {"split-half r": r, "full (S-B)": 2 * r / (1 + r), "pitches per half": n_half, "implied n0 (pitches)": n_half * (1 - r) / r if r > 0 else np.nan, "pitchers": int(ok.sum())}, full_lv


def main():
    raw = ppf.exclude(add_count_and_targets(fp.load_fl_pitches_cached()))
    fp.compute_pooled_wh_sva = sva_variant
    res = {}; rel = []
    term = raw[raw["is_terminal"]]; kr = term.groupby("pitcher_id").agg(PA=("is_terminal", "size"), K=("k_or_bb", lambda s: (s == "strikeout").mean()))
    for tgt, cnt in VARIANTS:
        CUR.update(target=tgt, count=cnt); lab = f"{tgt}{' + count' if cnt else ''}"; print(f"\n=== {lab}", flush=True)
        m_k, m_bb, n0_k, n0_bb, n0b, rho = pp.fit_frozen_model(raw)
        dk, db = chk.fold2(raw, m_k, m_bb, n0b)
        mb2, n0b2 = fit_bb_with(raw, ["wh_sva_level"]); db2 = chk.fold2(raw, m_k, mb2, n0b2)[1]; db2 = db2.dropna(subset=["wh_sva_level"]); db2["pred_x"] = mb2.predict(db2)
        res[lab] = {"k": dk.set_index("pitcher_id")[["pred", "K_rate_actual", "PA_test"]], "bb": db.set_index("pitcher_id")[["pred", "BB_rate_actual", "PA_test"]],
                    "bbx": db2.set_index("pitcher_id")[["pred_x"]], "coef_k": float(m_k.params.get("wh_sva_level", np.nan)), "coef_bb": float(mb2.params.get("wh_sva_level", np.nan)),
                    "t_k": float(m_k.tvalues.get("wh_sva_level", np.nan)), "t_bb": float(mb2.tvalues.get("wh_sva_level", np.nan))}
        rr, lv = reliability(raw); k = kr[kr["PA"] >= 100].join(lv.rename("lv"), how="inner"); rr["corr with K% (same period)"] = np.corrcoef(k["K"], k["lv"])[0, 1]; rr["variant"] = lab; rel.append(rr)
    base = "whiff"
    # ---- K%
    ids = set.intersection(*[set(v["k"].index) for v in res.values()]); ids = sorted(ids)
    B = res[base]["k"].loc[ids]; y = B["K_rate_actual"].to_numpy(); w = ew.weights_from_base(y, B["pred"].to_numpy(), B["PA_test"].to_numpy())[0]
    c = lambda p: p + np.average(y - p, weights=w)
    rows = []
    for lab, v in res.items():
        p = c(v["k"].loc[ids, "pred"].to_numpy()); rm, ma = er.wmetrics(y, p, w); row = {"variant": lab, "coef in K% model": v["coef_k"], "t": v["t_k"], "RMSE": rm, "MAE": ma}
        if lab != base:
            rep = er.compare(y, c(B["pred"].to_numpy()), p, w, np.array(ids), n=1000); row.update({"dRMSE vs prod": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
        rows.append(row)
    print(f"\nK%: fold 2 held out (2H2026 targets), n = {len(ids)} pitchers, level-free, frozen weights; production = whiff (no count)\n" + er.table(rows, 5))
    # ---- BB% with the variant added
    ids = sorted(set.intersection(*[set(v["bbx"].index) for v in res.values()]) & set(res[base]["bb"].index))
    Bb = res[base]["bb"].loc[ids]; y = Bb["BB_rate_actual"].to_numpy(); w = ew.weights_from_base(y, Bb["pred"].to_numpy(), Bb["PA_test"].to_numpy())[0]
    c = lambda p: p + np.average(y - p, weights=w); ref = c(Bb["pred"].to_numpy()); rows = []
    rm, ma = er.wmetrics(y, ref, w); rows.append({"BB% model": "production (no stuff term)", "RMSE": rm, "MAE": ma})
    for lab, v in res.items():
        p = c(v["bbx"].loc[ids, "pred_x"].to_numpy()); rm, ma = er.wmetrics(y, p, w); rep = er.compare(y, ref, p, w, np.array(ids), n=1000)
        rows.append({"BB% model": f"+ {lab}", "coef": v["coef_bb"], "t": v["t_bb"], "RMSE": rm, "MAE": ma, "dRMSE": rep["dRMSE"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
    print(f"\nBB%: fold 2 held out, n = {len(ids)}, production BB% model vs + each variant's level\n" + er.table(rows, 5))
    print("\nRELIABILITY of the pitcher level (alternating games, 2025-26)\n" + pd.DataFrame(rel).set_index("variant").round(3).to_string())


if __name__ == "__main__":
    main()
