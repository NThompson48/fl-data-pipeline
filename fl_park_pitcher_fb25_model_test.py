"""
fl_park_pitcher_fb25_model_test.py -- DOES FLY BALL + POP-UP SHARE (>= 25 deg) HELP THE TRUSTMARK PITCHER MODEL?  (user 2026-10-09: "Does it help the model?")
Unit: a visiting pitcher-season at Trustmark.  Target = mean fixed-effects-residualised wOBA allowed over his Trustmark PA (year + hitter + pitcher + [pitcher x home] + park effects removed, as in the production design), n = his Trustmark PA.
Prediction = fitted interaction part (residualised Trustmark x profile columns times the coefficients) averaged over the same PA, so every model is scored on the same quantity: how much the pitcher's profile moves his Trustmark result relative to the Trustmark average.
Models (all fitted on training pitchers / the training season, coefficients shrunk by the production rule max(0, 1 - 1/t^2) with the training-sample cluster t):
  A  none (profile adds nothing, 0)                      B  PRODUCTION profile: HR-prone, K%, Air% (>= 10), left        C  B + fly ball + pop-up share
  D  K% allowed only (what survives production shrinkage)  E  K% + fly ball + pop-up share                              F  K% + BB% + fly ball + pop-up share
Validation: (1) 5-fold CV by PITCHER (all seasons pooled; each pitcher's Trustmark results are predicted by coefficients fitted on other pitchers), repeated 5 times with different fold assignments;
            (2) cross-SEASON: fit on one season, predict the other (both directions pooled).
Weights: inverse-variance 1/(a + b/n), law fitted on model B's residuals and then frozen for every model (fl_eval_weights); report weighted RMSE and weighted MAE, difference vs B (production), pitcher-cluster bootstrap 90% interval and P(better) for both metrics.
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_park_pitcher_final as pf
import fl_park_pitcher_profile_screen as sc
import fl_pitcher_fbshare_sticky as st
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
MODELS = {"A none": [], "B production (HR-prone, K%, Air%, left)": ["z_hr", "z_k", "z_air", "left_c"], "C production + FB+PU share": ["z_hr", "z_k", "z_air", "left_c", "z_fbpu"], "D K% only": ["z_k"], "E K% + FB+PU": ["z_k", "z_fbpu"], "F K% + BB% + FB+PU": ["z_k", "z_bb", "z_fbpu"]}


def fit_shrunk(Xr, yr, cl, rows, cols):
    if not cols:
        return np.zeros(0)
    X = Xr[:, cols][rows]; b, se = cluster_ols(X, yr[rows], cl[rows]); t = b / se; sh = np.where(np.abs(t) > 1, np.maximum(0.0, 1 - 1 / t ** 2), 0.0); return b * sh


def main():
    est, prof, stdz = pf.build(); est["b_home"] = (est["batter_team"] == est["home_team"]).astype(int); C = sc.build_candidates(est)
    bb = pf.pitcher_bbe(); bb["pid"] = bb["pitcher_id"].astype(str); bb["park"] = bb["game_id"].str.split("-").str[1]; bb = bb[bb["park"] != "TrustmarkPark"]
    est["z_fbpu"] = est["pid"].map(st.share_z(bb, 25, 91, est, 92)); est["z_bb"] = est["pid"].map(C["BB% allowed"]); est["left_c"] = est["lefty"] - est["lefty"].mean()
    est = est.dropna(subset=["z_fbpu", "z_bb"]).reset_index(drop=True); allterms = ["z_hr", "z_k", "z_air", "left_c", "z_fbpu", "z_bb"]
    home_batter = (est["b_home"] == 1); g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pid"])[0], pd.factorize(np.where(~home_batter, est["pid"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    tmv = est["tm"].to_numpy(float) * home_batter.to_numpy(float); on = tmv == 1; yr = partial_out(est["woba"].to_numpy(float)[:, None], g, iters=30)[:, 0]; cl = pd.factorize(est["game_id"])[0]
    Xr = partial_out(np.column_stack([tmv * est[t].to_numpy(float) for t in allterms]), g, iters=30); ix = {t: i for i, t in enumerate(allterms)}; pid = est["pid"].to_numpy(); yrs = est["year"].to_numpy()
    print(f"visiting-pitcher Trustmark PA {int(on.sum()):,}; pitcher-seasons {len(set(zip(pid[on], yrs[on])))}")
    unit = pd.DataFrame({"pid": pid[on], "year": yrs[on]}); unit["key"] = unit["pid"] + "|" + unit["year"].astype(str); keys, uidx = np.unique(unit["key"], return_inverse=True); nu = np.bincount(uidx); u_pid = np.array([k.split("|")[0] for k in keys]); u_year = np.array([int(k.split("|")[1]) for k in keys])
    y_u = np.bincount(uidx, weights=yr[on]) / nu; row_idx = np.where(on)[0]

    def unit_pred(b, cols):
        if not cols:
            return np.zeros(len(keys))
        p = Xr[:, cols][on] @ b; return np.bincount(uidx, weights=p) / nu
    results = {m: [] for m in MODELS}; ys, ws, ids, ns = [], [], [], []
    rows_all = np.arange(len(est)); rng = np.random.default_rng(5); upid = pd.unique(pid)
    # (1) 5-fold by pitcher, repeated
    P1 = {m: np.zeros((5, len(keys))) for m in MODELS}
    for rep in range(5):
        fold = dict(zip(upid, rng.integers(0, 5, len(upid)))); f_row = np.array([fold[p] for p in pid]); f_unit = np.array([fold[p] for p in u_pid])
        for f in range(5):
            tr = f_row != f
            for m, terms in MODELS.items():
                cols = [ix[t] for t in terms]; b = fit_shrunk(Xr, yr, cl, tr, cols); te = f_unit == f; pu = unit_pred(b, cols); P1[m][rep, te] = pu[te]
    pred1 = {m: P1[m].mean(axis=0) for m in MODELS}
    # (2) cross-season
    P2 = {m: np.zeros(len(keys)) for m in MODELS}
    for tr_y, te_y in ((2025, 2026), (2026, 2025)):
        tr = yrs == tr_y
        for m, terms in MODELS.items():
            cols = [ix[t] for t in terms]; b = fit_shrunk(Xr, yr, cl, tr, cols); te = u_year == te_y; pu = unit_pred(b, cols); P2[m][te] = pu[te]

    def report(pred, title, mask=None, ref="B production (HR-prone, K%, Air%, left)", law="B production (HR-prone, K%, Air%, left)"):
        mk = np.ones(len(keys), bool) if mask is None else mask; y = y_u[mk]; n = nu[mk].astype(float); ids_ = u_pid[mk]; base = pred[ref][mk]; a, b = ew.fit_variance_law(y - pred[law][mk], n); w = ew.inv_var_weights(n, a, b); rows = []
        for m in MODELS:
            p = pred[m][mk]; rm, ma = er.wmetrics(y, p, w); row = {"model": m, "RMSE": rm, "MAE": ma}
            if m != ref:
                rep = er.compare(y, base, p, w, ids_, n=1000); row.update({"dRMSE vs ref": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE vs ref": rep["dMAE"], "MAE 90%": rep["MAE 90%"], "P(MAE better)": rep["P(MAE better)"]})
            rows.append(row)
        print(f"\n{title}: {mk.sum()} pitcher-seasons (variance law a {a:.5f}, b {b:.3f} => n0_eval {b / a:.0f} PA; frozen)\n" + er.table(rows, 5))
    report(pred1, "(1) 5-FOLD BY PITCHER (repeated x5), reference = B production"); report(P2, "(2) CROSS-SEASON, reference = B production")
    report(pred1, "(1b) 5-FOLD BY PITCHER, reference = D K%-only (what production keeps after shrinkage)", ref="D K% only"); report(P2, "(2b) CROSS-SEASON, reference = D K%-only", ref="D K% only")
    # coefficients on all data (shrunk), for reference
    allrows = np.ones(len(est), bool)
    for m in ("B production (HR-prone, K%, Air%, left)", "C production + FB+PU share", "E K% + FB+PU"):
        cols = [ix[t] for t in MODELS[m]]; X = Xr[:, cols]; b, se = cluster_ols(X, yr, cl); print(f"\nfull-sample {m}: " + "; ".join(f"{t} {b[i]:+.4f} (t {b[i] / se[i]:+.1f})" for i, t in enumerate(MODELS[m])))


if __name__ == "__main__":
    main()
