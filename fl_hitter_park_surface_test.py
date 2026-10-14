"""
fl_hitter_park_surface_test.py -- MODEL-CHANGE REPORT: hitter Trustmark translation, residual-surface model vs the profile (Air% / power / left) model  (user 2026-10-10)
Unit: a visiting hitter-season with >= 8 balls in play at Trustmark.  Targets (both within hitter, per ball in play):
  T1 PARK EFFECT ON SAME-QUALITY CONTACT = mean (wOBA - xwOBA) at Trustmark minus the hitter's own non-Trustmark mean residual (the user's definition);
  T2 TOTAL CHANGE IN wOBA ON CONTACT   = mean wOBA at Trustmark minus the hitter's own non-Trustmark mean wOBA (shrunk toward the league mean, n0 100) -- includes contact-quality shifts the surface does not model.
Cross-season: everything (the surface; the profile regression; the hitter's profile / batted-ball mix, taken from his NON-Trustmark balls) comes from the TRAIN season; the target is the TEST season (both directions pooled).
Models: NONE = a single level (weighted mean of the train targets); OLD = level + Air% + power (HR per ball in play) + left on the train units (bootstrap-free: t from OLS, positive-part James-Stein shrink); NEW-k = the surface effect averaged over the hitter's balls
(features: EV+LA | + side | + side + spray direction), shrunk toward the league mean by n/(n+60), NO fitted level (the surface supplies it).  Weights 1/(a + b/n) from NONE's residuals, frozen; hitter-cluster bootstrap (B = 1000).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_park_surface as ps

warnings.filterwarnings("ignore")
MIN_N = 8


def profile_table(balls):
    """hitter-level Air%, HR per BIP (shrunk n0 89 / 100), left share from the given NON-Trustmark balls"""
    x = balls[balls["tm"] == 0]; g = x.groupby("name_clean").agg(n=("la", "size"), air=("la", lambda s: (s >= 10).mean()), hr=("woba", lambda s: (s > 1.8).mean()), left=("left", "mean")); lg_air = np.average(g["air"], weights=g["n"]); lg_hr = np.average(g["hr"], weights=g["n"])
    out = pd.DataFrame({"z_air": (g["air"] - lg_air) * g["n"] / (g["n"] + 89), "z_hr": (g["hr"] - lg_hr) * g["n"] / (g["n"] + 100), "left": g["left"], "n_prof": g["n"]}); return out


def units(b, season, col_res="res_adj", woba_base=None):
    v = b[(b["tm"] == 1) & (b["vis"] == 1) & (b["year"] == season)]; g = v.groupby("name_clean").agg(n=("res", "size"), t1=(col_res, "mean"), woba=("woba", "mean")); g = g[g["n"] >= MIN_N].copy(); g["t2"] = g["woba"] - g.index.map(woba_base).to_numpy(); return g.drop(columns="woba")


def main():
    b = ps.load_balls(); b["res_adj"] = b["res"] - b["name_clean"].map(ps.hitter_baseline(b)).fillna(0.0)
    non = b[b["tm"] == 0].groupby("name_clean")["woba"].agg(["sum", "size"]); lgw = b.loc[b["tm"] == 0, "woba"].mean(); woba_base = (non["sum"] + 100 * lgw) / (non["size"] + 100); rows = []; preds = {}
    allu = []
    for tr_y, te_y in ((2025, 2026), (2026, 2025)):
        train = b[b["year"] == tr_y]; U_tr = units(b, tr_y, woba_base=woba_base); U_te = units(b, te_y, woba_base=woba_base); P = profile_table(train); U_te = U_te.join(P, how="left"); U_tr = U_tr.join(P, how="left"); U_te["train_year"] = tr_y
        for tgt in ("t1", "t2"):
            pass
        ok_tr = U_tr["n"].notna() & U_tr["z_air"].notna()
        out = U_te.copy()
        for tgt in ("t1", "t2"):
            w = U_tr.loc[ok_tr, "n"].to_numpy(float); y = U_tr.loc[ok_tr, tgt].to_numpy(float); out[f"{tgt}_NONE"] = np.average(y, weights=w)
            X = np.column_stack([np.ones(ok_tr.sum()), U_tr.loc[ok_tr, ["z_air", "z_hr", "left"]].to_numpy(float)]); XtWX = X.T @ (X * w[:, None]); beta = np.linalg.solve(XtWX, X.T @ (w * y)); e = y - X @ beta; s2 = np.sum(w * e ** 2) / (len(y) - 4)
            se = np.sqrt(np.diag(np.linalg.inv(XtWX)) * s2 * np.mean(w)); t = beta / se; sh = np.where(np.abs(t) > 1, np.maximum(0.0, 1 - 1 / t ** 2), 0.0); sh[0] = 1.0; bs = beta * sh
            Xt = np.column_stack([np.ones(len(out)), out[["z_air", "z_hr", "left"]].fillna({"z_air": 0, "z_hr": 0, "left": train["left"].mean()}).to_numpy(float)]); out[f"{tgt}_OLD"] = Xt @ bs
        for f in ps.FEATS:
            eff = ps.fit_effect(train, f); sh_, mu = ps.hitter_shift(train, eff, f); out[f"NEW_{f}"] = out.index.map(sh_).to_numpy(); out[f"NEW_{f}"] = out[f"NEW_{f}"].fillna(mu)
        allu.append(out)
    U = pd.concat(allu); print(f"test units: {len(U)} hitter-seasons (>= {MIN_N} Trustmark balls; {U.index.nunique()} hitters)\n")
    for tgt, title in (("t1", "T1  PARK EFFECT ON SAME-QUALITY CONTACT (wOBA - xwOBA, within hitter)"), ("t2", "T2  TOTAL CHANGE IN wOBA ON CONTACT (within hitter)")):
        y = U[tgt].to_numpy(float); n = U["n"].to_numpy(float); base = U[f"{tgt}_NONE"].to_numpy(float); a, b_ = ew.fit_variance_law(y - base, n); w = ew.inv_var_weights(n, a, b_); ids = U.index.to_numpy()
        models = {"NONE (level only)": base, "OLD profile (Air%, power, left)": U[f"{tgt}_OLD"].to_numpy(float)}
        for f in ps.FEATS:
            models[f"NEW surface: {f}"] = U[f"NEW_{f}"].to_numpy(float)
        old = models["OLD profile (Air%, power, left)"]; rows = []
        for k, v in models.items():
            rm, ma = er.wmetrics(y, v, w); row = {"model": k, "RMSE": rm, "MAE": ma, "mean prediction": np.average(v, weights=w)}
            if k != "OLD profile (Air%, power, left)":
                rep = er.compare(y, old, v, w, ids, n=1000); row.update({"dRMSE vs OLD": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE vs OLD": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
            rows.append(row)
        print(f"{title}: weighted mean target {np.average(y, weights=w):+.4f}; variance law a {a:.5f}, b {b_:.3f} (n0_eval {b_ / max(a, 1e-12):.0f} balls)\n" + er.table(rows, 5) + "\n")


if __name__ == "__main__":
    main()
