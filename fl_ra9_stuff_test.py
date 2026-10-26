"""
fl_ra9_stuff_test.py -- does a pitcher stuff level (P(strike | stuff) and the other candidates) add to the RA9 baseline model beyond K%, BB% and innings per appearance?  (user 2026-10-26: "is strike given stuff perhaps helpful for the RA9 part of the model?")
PRODUCTION RA9 BASELINE (fl_ra9_trustmark_projection): target RA9 ~ K_pred + BB_pred + ip_per_g_prior, WLS (weights reliability_weight(target IP, RA9_N0_EVAL)), fit on the single Trackman-era transition 2025 actual -> 2026 actual, BF >= 100 both years.
This test ADDS ONE STUFF LEVEL to that regression and scores both versions by LEAVE-ONE-PITCHER-OUT predictions (hat-matrix shortcut for WLS): weighted RMSE and MAE of RA9 (level-free; standard inverse-variance weights from the baseline's leave-one-out residuals), pitcher bootstrap (1,000) of the difference,
and the coefficient with its t-value from the full-sample fit.   FORWARD: features from 2025 (K%, BB%, innings per appearance, stuff level) -> 2026 RA9.   REVERSE CHECK: features from 2026 -> 2025 RA9 (a 'predict the past' check used elsewhere in this project).
STUFF LEVELS come from the HELD-OUT, frozen-model design of fl_y2y_heldout.py (forward: the model trained on 2025 pitches of OTHER pitchers only; reverse: trained on 2026 only), so no level has seen the pitcher or the target season.   Levels are z-scored across the pitchers in the test so coefficients are per SD.
Note the production model applies the RA9 coefficients to PROJECTED K% and BB% (which already contain stuff through the K% model); this test uses ACTUAL prior-season K% and BB%, as the model is fit.   Output: output/ra9_stuff_test.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import text

import fl_eval_report as er
import fl_eval_weights as ew
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_ra9_trustmark_projection as ra9
import mlb_marcel as mm
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
MEAS = ["strike|stuff (net)", "strike|stuff (xgb)", "swstr|stuff (net)", "swstr|stuff (xgb)", "swstr_ct (net)", "swstr_ct (xgb)", "swing network wh_sva", "XGBoost wh_sva", "whiff|stuff (net)", "strike_ct (net)", "strike_loc (net)"]
nm = lambda s: s.replace(" ", "_").replace("|", "-").replace("(", "").replace(")", "")


def wls_t(X, y, w):
    A = X.T @ (X * w[:, None]); Ai = np.linalg.inv(A); beta = Ai @ X.T @ (w * y); e = y - X @ beta; s2 = np.sum(w * e ** 2) / (len(y) - X.shape[1]); return beta, beta / np.sqrt(np.diag(Ai) * s2)


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); idmap = raw[["pitcher", "pitcher_id"]].drop_duplicates(); idmap["pitcher_id"] = idmap["pitcher_id"].astype(str)
    uniq = idmap.groupby("pitcher")["pitcher_id"].nunique(); idmap = idmap[idmap["pitcher"].isin(uniq[uniq == 1].index)].drop_duplicates("pitcher").set_index("pitcher")["pitcher_id"]
    war = read_sql_df(text("select player, year, bf, ip, so, bb, ra, ra9, g from war_pitchers where game_type='reg_season'")); out = {}; rows = []
    for design, fy, ty in (("forward 2025 -> 2026", 2025, 2026), ("reverse 2026 -> 2025", 2026, 2025)):
        a = war[war["year"] == fy]; b = war[war["year"] == ty]; m = a.merge(b, on="player", suffixes=("_f", "_t")); m = m[(m["bf_f"] >= 100) & (m["bf_t"] >= 100)].copy()
        m["K_pred"] = m["so_f"] / m["bf_f"]; m["BB_pred"] = m["bb_f"] / m["bf_f"]; m["ip_per_g_prior"] = (m["ip_f"] + ra9.N0_IPG * ra9.LG_IPG) / (m["g_f"] + ra9.N0_IPG); m["target"] = m["ra9_t"]
        m["pid"] = m["player"].map(idmap); m = m.dropna(subset=["pid", "target"]).copy(); wd = "fwd25" if design.startswith("forward") else "rev26"; w0 = mm.reliability_weight(m["ip_t"], ra9.RA9_N0_EVAL).values
        print(f"\n== {design}: n = {len(m)} pitchers (BF >= 100 both seasons)", flush=True); o = {}
        for meas in MEAS:
            lv = pd.read_csv(f"output/y2y_heldout_{wd}_{nm(meas)}.csv", index_col=0, dtype={0: str})[str(fy)]; d = m.assign(lvl=m["pid"].map(lv)).dropna(subset=["lvl"]).copy(); d["lvl"] = (d["lvl"] - d["lvl"].mean()) / d["lvl"].std()
            y = d["target"].to_numpy(float); w = mm.reliability_weight(d["ip_t"], ra9.RA9_N0_EVAL).values; X0 = np.column_stack([np.ones(len(d)), d[["K_pred", "BB_pred", "ip_per_g_prior"]].to_numpy(float)]); X1 = np.column_stack([X0, d["lvl"].to_numpy(float)])
            def lo(X):
                A = X.T @ (X * w[:, None]); beta = np.linalg.solve(A, X.T @ (w * y)); e = y - X @ beta; h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(A), X); return y - e / (1 - h)
            p0, p1 = lo(X0), lo(X1); wt = ew.weights_from_base(y, p0, d["ip_t"].to_numpy(float))[0]; cen = lambda q: q + np.average(y - q, weights=wt); c = er.compare(y, cen(p0), cen(p1), wt, d["pid"].to_numpy(), n=1000); beta, t = wls_t(X1, y, w)
            r = {"design": design, "measure": meas, "n": len(d), "RMSE base": c["RMSE old"], "RMSE +stuff": c["RMSE new"], "dRMSE": c["dRMSE"], "90%": c["RMSE 90%"], "P(better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE better)": c["P(MAE better)"],
                 "coef per SD (runs/9)": float(beta[-1]), "t": float(t[-1])}
            rows.append(r); o[meas] = r
        T = pd.DataFrame(rows); pd.set_option("display.width", 280); print(T[T["design"] == design].drop(columns="design").round(4).to_string(index=False), flush=True)
    json.dump(rows, open("output/ra9_stuff_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
