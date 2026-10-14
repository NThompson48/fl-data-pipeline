"""fl_pitcher_park_surface_air.py -- held-out-season test of an AIR%-ONLY pitcher shift: shift_i = league mean + slope x (his shrunk Air% - league Air%), slope = regression of the pitcher's surface shift on Air% in the TRAIN season (pitchers >= 100 balls).  Same units / target / weights / bootstrap as fl_pitcher_park_surface_test.py; compared with NONE (level only) and HIST (surface over his own balls)."""
import warnings
import numpy as np
import pandas as pd
import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_park_ms_check as ms
import fl_pitcher_park_surface_test as pt
warnings.filterwarnings("ignore")
b = pt.load(); pbase = pt.pitcher_baseline(b); allu = []
for tr_y, te_y in ((2025, 2026), (2026, 2025)):
    train = b[b["year"] == tr_y]; Utr = pt.units(b, tr_y, pbase); Ute = pt.units(b, te_y, pbase); eff = ms.fit_effect_pooled(train, "evla_side"); sh, raw, n, mu = pt.shifts(train, eff)
    x = train[train["tm"] == 0].copy(); x["air"] = (x["la"] >= 10).astype(float); a = x.groupby("pid")["air"].agg(["mean", "size"]); lg = np.average(a["mean"], weights=a["size"]); air_s = lg + (a["mean"] - lg) * a["size"] / (a["size"] + 100)
    J = pd.DataFrame({"raw": raw, "n": n, "air": air_s}).dropna(); J = J[J["n"] >= 100]; w = J["n"].to_numpy(float); A = np.column_stack([np.ones(len(J)), J["air"] - lg]); beta = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * J["raw"].to_numpy()))
    level = np.average(Utr["t1"], weights=Utr["n"]); out = Ute.copy(); out["NONE"] = level; h = out.index.map(sh).to_numpy(); out["HIST"] = level + np.where(np.isnan(h), 0.0, h - mu)
    ai = out.index.map(air_s).to_numpy(); out["AIR"] = level + np.where(np.isnan(ai), 0.0, beta[1] * (ai - lg)); out["slope"] = beta[1]; allu.append(out)
    print(f"train {tr_y}: shift-on-Air% slope {beta[1]:+.4f} per unit of Air% share (i.e. {beta[1] * 0.0434:+.4f} per SD of Air%), corr {np.corrcoef(J['raw'], J['air'])[0, 1]:+.2f}, n {len(J)}")
U = pd.concat(allu)
for lab, sel in (("ALL", np.ones(len(U), bool)), ("MISSISSIPPI pitchers at home", (U["ms"] == 1).to_numpy()), ("VISITING pitchers", (U["ms"] == 0).to_numpy())):
    Us = U[sel]; y = Us["t1"].to_numpy(float); n = Us["n"].to_numpy(float); base = Us["NONE"].to_numpy(float); a_, b_ = ew.fit_variance_law(y - base, n); w = ew.inv_var_weights(n, a_, b_); rows = []
    for k in ("NONE", "HIST", "AIR"):
        v = Us[k].to_numpy(float); rm, ma = er.wmetrics(y, v, w); row = {"model": k, "RMSE": rm, "MAE": ma, "SD of prediction": np.std(v)}
        if k != "NONE":
            rep = er.compare(y, base, v, w, Us.index.to_numpy(), n=1000); row.update({"dRMSE vs NONE": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
        rows.append(row)
    print(f"\n{lab} ({len(Us)} pitcher-seasons)\n" + er.table(rows, 5))
