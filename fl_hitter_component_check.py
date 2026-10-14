"""
fl_hitter_component_check.py -- (1) how well does each regressed component (K%, BB%, HR/PA, BABIP, HBP/PA, and wOBA itself) predict next year?  (2) do the components need MORE shrinkage than the per-stat regression already applies?
Leave-one-transition-out (n0 per stat refit on the other transitions, three-season regression, standard weights, next-year PA (AB for BABIP) >= 100): weighted RMSE and MAE of league mean / last season at face value / regressed;
skill = 1 - RMSE^2 / RMSE_league^2; and the calibration SLOPE of (next - league) on (predicted - league) with a hitter-cluster bootstrap interval (1 = the shrinkage is right; < 1 = shrink more; > 1 = shrunk too much).
Same slope for the wOBA built from the components (weights fit on the training transitions only), the direct wOBA and the 50/50 blend.
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_blend_weight_test as bw
import fl_hitter_depth3_test as d3
import fl_hitter_model_dev as dev
import fl_hitter_projection as hp

warnings.filterwarnings("ignore"); rng = np.random.default_rng(3)


def slope(dev_pred, dev_act, w, ids, n=500):
    def f(ii):
        x, y, ww = dev_pred[ii], dev_act[ii], w[ii]; xm, ym = np.average(x, weights=ww), np.average(y, weights=ww)
        return np.sum(ww * (x - xm) * (y - ym)) / np.sum(ww * (x - xm) ** 2)
    by = pd.Series(np.arange(len(ids))).groupby(ids).apply(np.array); ug = by.index.to_numpy()
    b = [f(np.concatenate([by[u] for u in rng.choice(ug, len(ug))])) for _ in range(n)]
    return f(np.arange(len(ids))), np.percentile(b, 5), np.percentile(b, 95)


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; stats = list(hp.SPEC); frames = {s: d3.frame(q, s) for s in stats}; rows = []; srows = []
    for s, d in frames.items():
        delta = hp.SPEC[s][0]; parts = []
        for T in (2022, 2023, 2024, 2025):
            tr = [t for t in (2022, 2023, 2024, 2025) if t != T]; n0 = d3.fit_n0(d[d["t"].isin(tr)], delta, 3); te = d[d["t"] == T].copy(); te["pred"] = d3.pred(te, n0, delta, 3); parts.append(te)
        D = pd.concat(parts); y = D["x_next"].to_numpy(); w = D["w"].to_numpy(); ids = D["name_clean"].to_numpy()
        lg, last, reg = D["lg_next"].to_numpy(), D["x_t"].to_numpy(), D["pred"].to_numpy()
        rl, ml = er.wmetrics(y, lg, w); rf, mf = er.wmetrics(y, last, w); rr, mr = er.wmetrics(y, reg, w)
        b, lo, hi = slope(reg - lg, y - lg, w, ids)
        rows.append({"stat": s, "n": len(D), "RMSE league": rl, "RMSE last yr": rf, "RMSE regressed": rr, "MAE league": ml, "MAE last yr": mf, "MAE regressed": mr, "skill vs league": 1 - rr ** 2 / rl ** 2, "slope (reg dev -> next dev)": b, "slope 90%": f"[{lo:.2f}, {hi:.2f}]"})
    print("(1)+(2a) EACH REGRESSED STAT, leave-one-transition-out\n" + er.table(rows, 4))
    # wOBA from components, direct, blend (component weights on training transitions only)
    frames2 = {s: frames[s] for s in ("woba",) + bw.COMP}; folds = []
    for T in (2022, 2023, 2024, 2025):
        o, _ = bw.fold_preds(frames2, tuple(t for t in (2022, 2023, 2024, 2025) if t != T), T); folds.append(o)
    D = pd.concat(folds); y = D["x_next"].to_numpy(); w = D["w"].to_numpy(); ids = D.index.get_level_values("name_clean").to_numpy(); lg = D["lg_next"].to_numpy(); out = []
    for lab, pr in (("league mean", lg), ("last year at face value", D["x_t"].to_numpy()), ("direct regression", D["direct"].to_numpy()), ("components (fitted weights)", D["comp"].to_numpy()), ("50/50 blend (production)", 0.5 * D["direct"].to_numpy() + 0.5 * D["comp"].to_numpy())):
        r, m = er.wmetrics(y, pr, w); row = {"wOBA projection": lab, "RMSE": r, "MAE": m}
        if lab not in ("league mean", "last year at face value"):
            b, lo, hi = slope(pr - lg, y - lg, w, ids); row.update({"slope on (proj - league)": b, "slope 90%": f"[{lo:.2f}, {hi:.2f}]"})
        out.append(row)
    print("\n(2b) wOBA FROM THE COMPONENTS vs DIRECT vs BLEND (n = %d), leave-one-transition-out, component weights fit on training transitions only\n" % len(D) + er.table(out, 4))


if __name__ == "__main__":
    main()
