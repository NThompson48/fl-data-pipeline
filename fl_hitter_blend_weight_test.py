"""
fl_hitter_blend_weight_test.py -- is the 50/50 blend of the direct wOBA regression and the component-built wOBA the best blend?
blend(lam) = lam x direct + (1 - lam) x components, lam in {0, .1, ..., 1}; production lam = .5 (a fixed choice, never fitted).
Everything is out of sample: for each held-out transition, n0 per stat (three-season regression) is fit on the other transitions only, and the component weights (K, BB, HR/PA, BABIP -> wOBA) are fit on
the TRAINING transitions only (in-sample predictions there).  Schemes: leave-one-transition-out (4 folds) and forward (train on earlier, test 2024->25 and 2025->26).
Target: next-year park-neutral wOBA, hitters with >= 100 PA next year, standard inverse-variance weights; hitter-cluster bootstrap; weighted RMSE AND MAE (reporting standard).  No Trackman terms (they are added after the blend).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_depth3_test as d3
import fl_hitter_model_dev as dev
import fl_hitter_projection as hp

warnings.filterwarnings("ignore")
COMP = ("k", "bbr", "hrr", "babip")
LAMS = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]


def fold_preds(frames, tr_years, te_year):
    """Per stat: n0 on the training transitions; predictions on train + test rows; then component weights on the training rows; returns the test frame with direct / comp columns."""
    P = {}
    for s, d in frames.items():
        delta = hp.SPEC[s][0]; n0 = d3.fit_n0(d[d["t"].isin(tr_years)], delta, 3)
        dd = d[d["t"].isin(list(tr_years) + [te_year])].copy(); dd["pred"] = d3.pred(dd, n0, delta, 3); P[s] = dd.set_index(["name_clean", "t"])
    idx = P["woba"].index
    for s in COMP:
        idx = idx.intersection(P[s].index)
    W = P["woba"].loc[idx]; tr = W.index.get_level_values("t").isin(tr_years)
    X = np.column_stack([(P[s].loc[idx, "pred"] - P[s].loc[idx, "lg_next"]).to_numpy() for s in COMP]); y = (W["x_next"] - W["lg_next"]).to_numpy(); w = W["w"].to_numpy()
    beta = np.linalg.lstsq(X[tr] * np.sqrt(w[tr])[:, None], y[tr] * np.sqrt(w[tr]), rcond=None)[0]
    te = ~tr; out = W[te].copy(); out["direct"] = out["pred"]; out["comp"] = out["lg_next"] + X[te] @ beta; out["hist"] = out["n_t"] + 0.75 * out["n_prev"]
    return out, beta


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; frames = {s: d3.frame(q, s) for s in ("woba",) + COMP}
    for scheme, plan in (("LEAVE-ONE-TRANSITION-OUT", [(tuple(t for t in (2022, 2023, 2024, 2025) if t != T), T) for T in (2022, 2023, 2024, 2025)]),
                         ("FORWARD (train on earlier transitions; tests 2024->25 and 2025->26)", [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)])):
        folds = []; betas = []
        for tr, te in plan:
            o, b = fold_preds(frames, tr, te); folds.append(o.assign(fold=te)); betas.append(b)
        D = pd.concat(folds); y = D["x_next"].to_numpy(); w = D["w"].to_numpy(); ids = D.index.get_level_values("name_clean").to_numpy()
        print(f"\n{'=' * 110}\n{scheme}: n = {len(D)} hitter-transitions; component weights by fold (K, BB, HR, BABIP): " + " | ".join(str(np.round(b, 2).tolist()) for b in betas) + f"\n{'=' * 110}")
        base = 0.5 * D["direct"].to_numpy() + 0.5 * D["comp"].to_numpy(); rows = []
        for lam in LAMS:
            new = lam * D["direct"].to_numpy() + (1 - lam) * D["comp"].to_numpy(); rep = er.compare(y, base, new, w, ids, n=400) if lam != 0.5 else {k: np.nan for k in ("dRMSE", "RMSE 90%", "P(RMSE better)", "dMAE", "MAE 90%", "P(MAE better)")}
            r, m = er.wmetrics(y, new, w); rows.append({"direct weight": lam, "RMSE": r, "MAE": m, "dRMSE vs .5": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE vs .5": rep["dMAE"], "MAE 90%": rep["MAE 90%"], "P(MAE better)": rep["P(MAE better)"]})
        print(er.table(rows))
        # by fold and by history
        print("\n  best direct weight by held-out fold (RMSE):", {int(f): LAMS[int(np.argmin([er.wmetrics(g['x_next'], l * g['direct'] + (1 - l) * g['comp'], g['w'])[0] for l in LAMS]))] for f, g in D.groupby("fold")})
        for lab, m in (("history < 200 PA", D["hist"] < 200), ("200-500 PA", (D["hist"] >= 200) & (D["hist"] < 500)), ("500+ PA", D["hist"] >= 500)):
            g = D[m]; print(f"  {lab:18s} n={len(g):3d}  best direct weight {LAMS[int(np.argmin([er.wmetrics(g['x_next'], l * g['direct'] + (1 - l) * g['comp'], g['w'])[0] for l in LAMS]))]}  RMSE at 0 / .5 / 1: " + " / ".join(f"{er.wmetrics(g['x_next'], l * g['direct'] + (1 - l) * g['comp'], g['w'])[0]:.5f}" for l in (0, .5, 1)))


if __name__ == "__main__":
    main()
