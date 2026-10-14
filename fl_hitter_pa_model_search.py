"""
fl_hitter_pa_model_search.py -- how good is the hitter playing-time model, and can it be improved?
Target: PA next year per team game, returners (any PA next year; the projection is conditional on playing), leave-one-transition-out over 2022->23 ... 2025->26 (524 hitter-transitions), equal weights, RMSE and MAE in PA at 102 games.
Baselines: league mean | last year's PA per game at face value | current (PA_t/G + projected wOBA quality).  Candidates add: second prior season's PA, nonlinearities, hit-quality interactions, a regular/part-timer split
(logit for >= 300-PA-pace x conditional means), and a small gradient-boosting / random-forest model.  Also R^2 and the error by prior-PA group, and the irreducible-noise picture: what share of returners change
status (regular <-> part-timer) between seasons.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import fl_eval_report as er
import fl_hitter_baseline_backtest as bt
import fl_hitter_depth3_test as d3
import fl_hitter_model_dev as dev
import fl_hitter_projection as hp

warnings.filterwarnings("ignore"); G = hp.GAMES; TG = 102


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; delta = hp.SPEC["woba"][0]; bt.MIN_NEXT = 1; fr = d3.frame(q, "woba")
    pm = dev.pa_model(p); r = pm.dropna(subset=["pa_next"]).copy(); r["g_t"] = r["t"].map(G); r["g_next"] = r["t"].map(lambda t: G[t + 1]); r["x"] = r["pa_t"] / r["g_t"]; r["y"] = r["pa_next"] / r["g_next"]
    r["x_prev"] = r["pa_prev"] / r["t"].map(lambda t: G.get(t - 1, 96)); parts = []
    for T in (2022, 2023, 2024, 2025):
        bt.MIN_NEXT = 100; tr = d3.frame(q, "woba"); tr = tr[tr["t"] != T]; bt.MIN_NEXT = 1; n0 = d3.fit_n0(tr, delta, 3); te = fr[fr["t"] == T].copy(); te["q"] = d3.pred(te, n0, delta, 3) - te["lg_next"]; parts.append(te[["name_clean", "t", "q"]])
    r = r.merge(pd.concat(parts), on=["name_clean", "t"], how="inner").reset_index(drop=True); r["x2"] = r["x"] ** 2; r["xq"] = r["x"] * r["q"]; r["reg"] = (r["x"] >= 300 / 102).astype(float)
    print(f"{len(r)} returners; PA next (at 102 games) mean {TG * r['y'].mean():.0f}, SD {TG * r['y'].std():.0f}; share who were regulars (>= 300 PA pace) last year {r['reg'].mean():.0%}")
    feats = {"A: PA_t/G": ["x"], "B (current): + quality q": ["x", "q"], "C: + PA_{t-1}/G": ["x", "q", "x_prev"], "D: + x^2 + x*q": ["x", "q", "x2", "xq"], "E: + x_prev + x^2 + x*q": ["x", "q", "x_prev", "x2", "xq"]}
    pred = {"league mean": [], "last year at face value": []}; pred.update({k: [] for k in feats}); pred.update({"F: regular/part-timer split": [], "G: gradient boosting": [], "H: random forest": []}); Y, I, T_ = [], [], []
    for T in (2022, 2023, 2024, 2025):
        tr, te = r[r["t"] != T], r[r["t"] == T]; gn = te["g_next"].to_numpy()
        pred["league mean"].append(gn * tr["y"].mean() * np.ones(len(te))); pred["last year at face value"].append(gn * te["x"].to_numpy())
        for k, cols in feats.items():
            X = np.column_stack([np.ones(len(tr))] + [tr[c] for c in cols]); b = np.linalg.lstsq(X, tr["y"].to_numpy(), rcond=None)[0]; pred[k].append(np.maximum(gn * (np.column_stack([te[c] for c in cols]) @ b[1:] + b[0]), 20))
        cols = ["x", "q", "x_prev"]; lr = LogisticRegression(C=10.0, max_iter=500).fit(tr[cols], (tr["y"] * TG >= 300 / 1.0)); ph = lr.predict_proba(te[cols])[:, 1]
        hi, lo = tr[tr["y"] * TG >= 300], tr[tr["y"] * TG < 300]; mh, ml = [np.linalg.lstsq(np.column_stack([np.ones(len(d)), d[cols]]), d["y"].to_numpy(), rcond=None)[0] for d in (hi, lo)]
        Xt = np.column_stack([np.ones(len(te)), te[cols]]); pred["F: regular/part-timer split"].append(gn * (ph * (Xt @ mh) + (1 - ph) * (Xt @ ml)))
        for nm, mdl in (("G: gradient boosting", GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)), ("H: random forest", RandomForestRegressor(n_estimators=300, min_samples_leaf=12, random_state=0))):
            mdl.fit(tr[["x", "q", "x_prev"]], tr["y"]); pred[nm].append(gn * mdl.predict(te[["x", "q", "x_prev"]]))
        Y.append(te["pa_next"].to_numpy()); I.append(te["name_clean"].to_numpy()); T_.append(np.full(len(te), T))
    y, ids, tt = map(np.concatenate, (Y, I, T_)); P = {k: np.concatenate(v) for k, v in pred.items()}; w = np.ones(len(y)); rows = []; cur = P["B (current): + quality q"]
    for k, v in P.items():
        rm, ma = er.wmetrics(y, v, w); r2 = 1 - ((y - v) ** 2).sum() / ((y - y.mean()) ** 2).sum(); row = {"model": k, "RMSE": rm, "MAE": ma, "R^2 vs a constant": r2}
        if k != "B (current): + quality q":
            rep = er.compare(y, cur, v, w, ids, n=400); row.update({"dRMSE vs current": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE vs current": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
        rows.append(row)
    print("LEAVE-ONE-TRANSITION-OUT, PA next year (returners)\n" + er.table(rows, 3))
    bins = [0, 100, 200, 300, 400, 1000]; lab = ["<100", "100-200", "200-300", "300-400", "400+"]; gb = pd.cut(pd.Series(r.set_index(["t", "name_clean"]).reindex(list(zip(tt, ids)))["pa_t"].to_numpy()), bins, labels=lab)
    print("\nerror of the current model by prior-season PA:\n" + pd.DataFrame({"g": gb, "err": y - cur, "abs": np.abs(y - cur), "actual": y}).groupby("g").agg(n=("err", "size"), mean_actual=("actual", "mean"), bias=("err", "mean"), rmse=("err", lambda s: np.sqrt((s ** 2).mean())), mae=("abs", "mean")).round(0).to_string())
    reg_t = r["x"] * TG >= 300; reg_n = r["y"] * TG >= 300; print(f"\nstatus change: of last year's regulars (>= 300 PA pace) {reg_n[reg_t].mean():.0%} are regulars again; of last year's part-timers {reg_n[~reg_t].mean():.0%} become regulars; the PA distribution is bimodal, which caps any conditional-mean model")


if __name__ == "__main__":
    main()
