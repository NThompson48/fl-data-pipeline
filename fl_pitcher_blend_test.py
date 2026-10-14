"""
fl_pitcher_blend_test.py -- does a 50/50 (or any) blend of the full pitcher model with prior history help, as the 50/50 direct / component blend does for hitter wOBA?
(1) K% and BB% (fold 2 of Experiment 2: features 2025 + 1H2026, targets = actual 2H2026; frozen production model fit on full-2025 -> 1H2026 without position players): blend(lam) = lam x model + (1 - lam) x PRIOR HISTORY,
    prior history = the naive regressed rate (naive_K_rate / naive_BB_rate: the pitcher's own Marcel rate pooled and shrunk toward the league, no stuff / bridge / command terms).  lam = 0 .. 1 by .1; level-free (same fitted intercept in
    every blend, as the production output is recentred); standard inverse-variance weights from the model's own residuals; pitcher-cluster bootstrap vs lam = 1 (the model alone) and vs lam = .5; weighted RMSE and MAE.
(2) RA9, 2025 -> 2026 (fl_pwar_calibration out-of-sample build, >= 10 IP in 2026): blend(lam) = lam x projected RA9 + (1 - lam) x his own 2025 RA9 shrunk toward the league (n0 = 50 / 100 / 200 IP), IP-weighted
    (RA9 luck variance ~ 1 / IP), same fitted intercept, weighted RMSE and MAE.
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_posplayer_refit_check as chk
import fl_production_projection as pp

warnings.filterwarnings("ignore")
LAMS = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]


def blend_table(y, model, hist, w, ids, label, nboot=500):
    c = lambda a: a + np.average(y - a, weights=w)                      # same fitted intercept for every blend
    rows = []; ref1 = c(model); ref5 = c(0.5 * model + 0.5 * hist)
    for lam in LAMS:
        new = c(lam * model + (1 - lam) * hist); r, m = er.wmetrics(y, new, w); row = {"weight on model": lam, "RMSE": r, "MAE": m}
        for nm, ref in (("vs model alone", ref1), ("vs .5", ref5)):
            if abs(lam - (1.0 if nm == "vs model alone" else 0.5)) > 1e-9:
                rep = er.compare(y, ref, new, w, ids, n=nboot); row.update({f"dRMSE {nm}": rep["dRMSE"], f"{nm} 90%": rep["RMSE 90%"], f"P(better) {nm}": rep["P(RMSE better)"], f"dMAE {nm}": rep["dMAE"]})
        rows.append(row)
    print(f"\n{label}\n" + er.table(rows, 5))
    best = min(LAMS, key=lambda l: er.wmetrics(y, c(l * model + (1 - l) * hist), w)[0]); print(f"  best weight on the model by RMSE: {best}")


def main():
    raw = fp.load_fl_pitches_cached(); rawx = ppf.exclude(raw); m_k, m_bb, n0_k, n0_bb, n0b, rho = pp.fit_frozen_model(rawx); dk, db = chk.fold2(rawx, m_k, m_bb, n0b)
    print(f"fold 2: K% n = {len(dk)}, BB% n = {len(db)} pitchers with 2H2026 targets")
    for lab, d, tcol, naive in (("K%", dk, "K_rate_actual", "naive_K_rate"), ("BB%", db, "BB_rate_actual", "naive_BB_rate")):
        d = d.dropna(subset=[naive, "pred", tcol]); y = d[tcol].to_numpy(); w = ew.weights_from_base(y, d["pred"].to_numpy(), d["PA_test"].to_numpy())[0]
        blend_table(y, d["pred"].to_numpy(), d[naive].to_numpy(), w, d["pitcher_id"].to_numpy(), f"(1) {lab}: model vs prior history (naive regressed rate)")
    d = dk.merge(db[["pitcher_id", "pred", "naive_BB_rate", "BB_rate_actual"]], on="pitcher_id", suffixes=("", "_bb")).dropna(subset=["pred", "pred_bb", "naive_K_rate", "naive_BB_rate"])
    y = (d["K_rate_actual"] - d["BB_rate_actual"]).to_numpy(); w = ew.weights_from_base(y, (d["pred"] - d["pred_bb"]).to_numpy(), d["PA_test"].to_numpy())[0]
    blend_table(y, (d["pred"] - d["pred_bb"]).to_numpy(), (d["naive_K_rate"] - d["naive_BB_rate"]).to_numpy(), w, d["pitcher_id"].to_numpy(), "(1c) K-BB%: model vs prior history")
    import fl_pwar_calibration as cal
    from sqlalchemy import text
    from fl_supabase import read_sql_df
    cd, war = cal.main(); cd = cd[cd["ip"] >= 10].copy(); w25 = war[war["year"] == 2025].groupby("player").agg(ip25=("ip", "sum"), ra25=("ra", "sum")); cd = cd.merge(w25, left_on="player", right_index=True, how="left"); cd["ip25"] = cd["ip25"].fillna(0)
    lg25 = (war[war["year"] == 2025]["ra"].sum() * 9) / war[war["year"] == 2025]["ip"].sum(); cd["ra9_25"] = np.where(cd["ip25"] > 0, 9 * cd["ra25"] / cd["ip25"].replace(0, np.nan), lg25)
    y = cd["target"].to_numpy(); w = cd["ip"].to_numpy(); ids = cd["pitcher_id"].to_numpy()
    for n0 in (50, 100, 200):
        hist = ((cd["ip25"] * cd["ra9_25"] + n0 * lg25) / (cd["ip25"] + n0)).to_numpy()
        blend_table(y, cd["pred"].to_numpy(), hist, w, ids, f"(2) RA9 2025 -> 2026 (n = {len(cd)}): projected RA9 vs own 2025 RA9 shrunk toward the league (n0 = {n0} IP), IP-weighted", nboot=300)


if __name__ == "__main__":
    main()
