"""
fl_hitter_survivor_test.py -- survivorship correction for hitter wOBA (thin-history hitters are projected ~.01 low).

TARGET (decided with the user): the rate over the PA he actually gets next year, PA-weighted, i.e. every hitter with >= 1 PA next year, weight = PA next year.  (Alternative rejected: rate conditional on >= 100 PA.)
BASE: the production regression, lg_next + (n_t dev_t + .75 n_prev dev_p)/(n_t + .75 n_prev + n0), n0 fit on the training transitions only (>= 100 PA sample, inverse-variance weights, as in production).
CORRECTIONS added to the base (all fit on training transitions only, PA-weighted least squares; the reference gets the same free intercept):
  R   intercept only                                 c0
  S1  c0 + c1 x fade,            fade = n0 / (n_t + .75 n_prev + n0)   -- the prior's share (what "regress to the survivor mean" means)
  S2  c0 + c1 x 1[history < 100]                      -- a thin-history step
  S3  c0 + c1 x (1 - P(retain | PA_t))                -- selection strength: the less likely he is to be here at all, the bigger the survivor premium (logit on ln PA_t, fit in the training folds)
  S4  c0 + c1 x 1[hist < 50] + c2 x 1[50 <= hist < 200]   -- two bins
VALIDATION: forward only (train on earlier transitions, score the next one: 22->23 / 23->24 / 24->25 -> tests 23->24, 24->25, 25->26) and leave-one-transition-out as a check; hitter-cluster bootstrap of the
held-out change in PA-weighted RMSE.  Also scored on the old target (>= 100 PA, inverse-variance weights) to see what the correction does to regulars.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1234)
DELTA = 0.75


def wr(e, w):
    return float(np.sqrt(np.average(e ** 2, weights=w)))


def design(d, kind, n0, ret_fit):
    hist = d["n_t"] + DELTA * d["n_prev"]; one = np.ones(len(d))
    if kind == "R":
        return np.column_stack([one])
    if kind == "S1":
        return np.column_stack([one, n0 / (hist + n0)])
    if kind == "S2":
        return np.column_stack([one, (hist < 100).astype(float)])
    if kind == "S3":
        pr = 1 / (1 + np.exp(-(ret_fit[0] + ret_fit[1] * np.log(d["n_t"]))))
        return np.column_stack([one, 1 - pr])
    if kind == "S4":
        return np.column_stack([one, (hist < 50).astype(float), ((hist >= 50) & (hist < 200)).astype(float)])


def retention_fit(p, years):
    rows = []
    for t in years:
        a = p[p["year"] == t].set_index("name_clean"); b = p[p["year"] == t + 1].set_index("name_clean")
        r = pd.DataFrame({"pa_t": a["pa"]}); r["ret"] = b["pa"].reindex(r.index).notna().astype(int); rows.append(r[r["pa_t"] > 0])
    r = pd.concat(rows); m = sm.Logit(r["ret"], sm.add_constant(np.log(r["pa_t"]))).fit(disp=0)
    return m.params.to_numpy()


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    bt.MIN_NEXT = 1; D = bt.transitions(q, "woba"); D["n_prev"] = D["n_prev"].fillna(0)
    bt.MIN_NEXT = 100; D100 = bt.transitions(q, "woba")
    kinds = ["R", "S1", "S2", "S3", "S4"]; folds = [(list(range(2022, T)), T) for T in (2023, 2024, 2025)]
    out = {k: [] for k in kinds}; meta = []
    for scheme, plan in (("FORWARD", [([t for t in (2022, 2023, 2024, 2025) if t < T], T) for T in (2023, 2024, 2025)]),
                         ("LOTO", [([t for t in (2022, 2023, 2024, 2025) if t != T], T) for T in (2022, 2023, 2024, 2025)])):
        res = {k: [] for k in kinds}
        for tr_years, T in plan:
            if not tr_years:
                continue
            dtr = D[D["t"].isin(tr_years)]; dte = D[D["t"] == T].copy()
            n0 = bt.fit_n0(D100[D100["t"].isin(tr_years)], DELTA)                      # production n0, training transitions only
            base_tr = bt.predict(dtr, n0, DELTA); base_te = bt.predict(dte, n0, DELTA)
            ret_fit = retention_fit(p, tr_years)
            for k in kinds:
                Xtr = design(dtr, k, n0, ret_fit); y = (dtr["x_next"] - base_tr).to_numpy(); w = dtr["n_next"].to_numpy()
                coef = np.linalg.lstsq(Xtr * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
                pred = base_te + design(dte, k, n0, ret_fit) @ coef
                res[k].append(dte.assign(pred=pred, kind=k, coef=str(np.round(coef, 4)), n0=n0))
        print(f"\n{'=' * 100}\n{scheme}: held-out PA-weighted RMSE (target = rate over the PA he gets next year)\n{'=' * 100}")
        R = {k: pd.concat(v, ignore_index=True) for k, v in res.items()}
        ref = R["R"]; y = ref["x_next"].to_numpy(); w = ref["n_next"].to_numpy(); ids = ref["name_clean"].to_numpy()
        by = pd.Series(np.arange(len(ref))).groupby(ids).apply(np.array); ug = by.index.to_numpy(); rows = []
        for k in kinds:
            e = y - R[k]["pred"].to_numpy(); e0 = y - ref["pred"].to_numpy(); r = {"model": k, "PA-wtd RMSE": wr(e, w)}
            if k != "R":
                dd = []
                for _ in range(500):
                    ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wr(e[ii], w[ii]) - wr(e0[ii], w[ii]))
                dd = np.array(dd); r.update({"vs R": np.mean(dd), "90% interval": f"[{np.percentile(dd, 5):+.5f}, {np.percentile(dd, 95):+.5f}]", "P(better)": float((dd < 0).mean())})
            # old target: hitters with >= 100 PA next year, inverse-variance weights
            m100 = (R[k]["n_next"] >= 100).to_numpy(); ww = ew_weights(R[k][m100]); r["RMSE old target (>=100 PA)"] = wr((R[k]["x_next"] - R[k]["pred"]).to_numpy()[m100], ww)
            r["coef (last fold)"] = R[k]["coef"].iloc[-1]; rows.append(r)
        print(pd.DataFrame(rows).round(5).to_string(index=False))
        # mean error by prior PA bucket (PA-weighted)
        bins = [0, 50, 100, 200, 400, 5000]; lab = ["<50", "50-100", "100-200", "200-400", "400+"]
        tab = {}
        for k in ("R", "S1", "S3", "S4"):
            d = R[k].assign(err=R[k]["x_next"] - R[k]["pred"], b=pd.cut(R[k]["n_t"], bins, labels=lab)); tab[k] = d.groupby("b").apply(lambda g: np.average(g["err"], weights=g["n_next"]))
        print("\n  held-out mean error (actual - predicted, PA-weighted) by prior-season PA:\n", pd.DataFrame(tab).round(4).to_string())
        print("  n by bucket:", R["R"].assign(b=pd.cut(R["R"]["n_t"], bins, labels=lab)).groupby("b").size().to_dict())


def ew_weights(d):
    import fl_eval_weights as ew
    n = d["n_next"].to_numpy(float); a, b = 0.000266, 0.4545            # the hitter wOBA law (fl_hitter_model_dev part E)
    return ew.inv_var_weights(n, a, b)


if __name__ == "__main__":
    main()
