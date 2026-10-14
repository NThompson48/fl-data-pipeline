"""
fl_pitcher_global_only_test.py -- MODEL CHANGE REPORT: pitcher Mississippi park shift, profile model -> GLOBAL-ONLY  (user 2026-10-10; reporting standard: weighted RMSE and MAE, old vs new, frozen weights, bootstrap)
Same unit and target as fl_park_pitcher_fb25_model_test.py: visiting pitcher-season at Trustmark (205), target = mean fixed-effects-residualised wOBA allowed over his Trustmark PA, prediction = the fitted profile part (0 for global-only: the global park effect is
absorbed by the park fixed effect, so the profile part is the only thing that can differ between pitchers).
  OLD-final   = the shipped profile model's FINAL coefficients (HR-prone -.0016, K% +.0115, Air% -.0025, left 0) applied in-sample (favours the old model: its coefficients were fitted on these very PA)
  OLD-CV      = the same four production terms re-fitted and bootstrap-shrunk on training pitchers / the training season (honest out-of-sample version of the old model)
  NEW         = global-only (0)
Weights: 1/(a + b/n) fitted on OLD-CV residuals and frozen; pitcher-cluster bootstrap (B = 1000), P(new better).
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_park_pitcher_final as pf
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
TERMS = ["z_hr", "z_k", "z_air", "left_c"]


def shrunk_fit(Xr, yr, cl, rows):
    b, se = cluster_ols(Xr[rows], yr[rows], cl[rows]); t = b / se; return b * np.where(np.abs(t) > 1, np.maximum(0.0, 1 - 1 / t ** 2), 0.0)


def main():
    est, prof, stdz = pf.build(); est["b_home"] = (est["batter_team"] == est["home_team"]).astype(int); est["left_c"] = est["lefty"] - est["lefty"].mean()
    home_batter = (est["b_home"] == 1); g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pid"])[0], pd.factorize(np.where(~home_batter, est["pid"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    tmv = est["tm"].to_numpy(float) * home_batter.to_numpy(float); on = tmv == 1; yr = partial_out(est["woba"].to_numpy(float)[:, None], g, iters=30)[:, 0]; cl = pd.factorize(est["game_id"])[0]
    Xr = partial_out(np.column_stack([tmv * est[t].to_numpy(float) for t in TERMS]), g, iters=30); pid = est["pid"].to_numpy(); yrs = est["year"].to_numpy()
    fin = json.load(open("output/_backup_before_pitcher_global_only/trustmark_pitcher_coefs.json"))["coefs"]; b_final = np.array([fin["hr_prone"]["final"], fin["k_rate"]["final"], fin["air"]["final"], fin["left"]["final"]]); print("shipped final profile coefficients:", dict(zip(TERMS, b_final.round(4))))
    keys, uidx = np.unique(pd.Series(pid[on]) + "|" + pd.Series(yrs[on]).astype(str), return_inverse=True); nu = np.bincount(uidx); u_pid = np.array([k.split("|")[0] for k in keys]); u_year = np.array([int(k.split("|")[1]) for k in keys]); y_u = np.bincount(uidx, weights=yr[on]) / nu

    def unit(b):
        return np.bincount(uidx, weights=Xr[on] @ b) / nu
    pred = {"OLD-final (shipped coefficients, in-sample)": {s: unit(b_final) for s in ("fold", "season")}, "OLD-CV (4 terms refit + shrunk)": {}, "NEW global-only": {s: np.zeros(len(keys)) for s in ("fold", "season")}}
    rng = np.random.default_rng(5); upid = pd.unique(pid); P = np.zeros((5, len(keys)))
    for rep in range(5):
        fold = dict(zip(upid, rng.integers(0, 5, len(upid)))); f_row = np.array([fold[p] for p in pid]); f_unit = np.array([fold[p] for p in u_pid])
        for f in range(5):
            b = shrunk_fit(Xr, yr, cl, f_row != f); pu = unit(b); P[rep, f_unit == f] = pu[f_unit == f]
    pred["OLD-CV (4 terms refit + shrunk)"]["fold"] = P.mean(axis=0); S = np.zeros(len(keys))
    for tr_y, te_y in ((2025, 2026), (2026, 2025)):
        b = shrunk_fit(Xr, yr, cl, yrs == tr_y); pu = unit(b); S[u_year == te_y] = pu[u_year == te_y]
    pred["OLD-CV (4 terms refit + shrunk)"]["season"] = S
    for scheme, title in (("fold", "5-fold by pitcher (x5)"), ("season", "cross-season (fit one season, predict the other)")):
        base = pred["OLD-CV (4 terms refit + shrunk)"][scheme]; a, b = ew.fit_variance_law(y_u - base, nu.astype(float)); w = ew.inv_var_weights(nu.astype(float), a, b); rows = []
        for m, v in pred.items():
            p = v[scheme]; rm, ma = er.wmetrics(y_u, p, w); row = {"model": m, "RMSE": rm, "MAE": ma}
            if m != "NEW global-only":
                rep = er.compare(y_u, p, pred["NEW global-only"][scheme], w, u_pid, n=1000); row.update({"dRMSE new - this": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(new RMSE better)": rep["P(RMSE better)"], "dMAE new - this": rep["dMAE"], "MAE 90%": rep["MAE 90%"], "P(new MAE better)": rep["P(MAE better)"]})
            rows.append(row)
        print(f"\n{title}: {len(keys)} pitcher-seasons (law a {a:.5f}, b {b:.3f}; frozen)\n" + er.table(rows, 5))


if __name__ == "__main__":
    main()
