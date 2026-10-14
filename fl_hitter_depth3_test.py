"""
fl_hitter_depth3_test.py -- does a THIRD history season help the hitter PREseason projection?  (production: latest season + previous season, weight delta on the previous.)
Depth-3 model: the same regression with a third season at weight delta^2 (geometric, no new parameter):
    pred = lg_next + (n_t dev_t + d n_p dev_p + d^2 n_pp dev_pp) / (n_t + d n_p + d^2 n_pp + n0).   n0 refit for each depth on the training transitions; delta as in production.
Transitions 2022->23 ... 2025->26; a 3-season history exists only for 2024->25 and 2025->26 (earlier rows are identical across depths).
Forward: n0 fit on earlier transitions, scored on 2025->26 (and on 2024->25 with 2022->23 / 2023->24); leave-one-transition-out as a check.  Standard inverse-variance weights (>= 100 PA next year); hitter-cluster bootstrap.
Stats: wOBA, K%, BB%, HR/PA, BABIP (wOBA also as the production blend of direct and component projections).
"""
import warnings
import numpy as np
import pandas as pd
import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev
import fl_hitter_projection as hp
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore"); rng = np.random.default_rng(21)


def frame(q, s):
    d = bt.transitions(q, s); lg = bt.league(q, s); den = bt.DEN[s]
    pp = q[["name_clean", "year", s, den]].rename(columns={s: "x_pp", den: "n_pp"}); pp["t"] = pp["year"] + 2
    d = d.merge(pp[["name_clean", "t", "x_pp", "n_pp"]], on=["name_clean", "t"], how="left")
    d["lg_pp"] = d["t"].map(lambda t: lg.get(t - 2, np.nan)); d["n_pp"] = d["n_pp"].fillna(0.0); d["x_pp"] = d["x_pp"].fillna(d["lg_pp"]).fillna(d["lg_t"])
    d["n_prev"] = d["n_prev"].fillna(0.0)
    return d


def pred(d, n0, delta, depth):
    dev_t = d["x_t"] - d["lg_t"]; dev_p = (d["x_prev"] - d["lg_prev"].fillna(d["lg_t"])).fillna(0.0); dev_pp = (d["x_pp"] - d["lg_pp"].fillna(d["lg_t"])).fillna(0.0)
    num = d["n_t"] * dev_t + delta * d["n_prev"] * dev_p; den = d["n_t"] + delta * d["n_prev"] + n0
    if depth == 3:
        num = num + delta ** 2 * d["n_pp"] * dev_pp; den = den + delta ** 2 * d["n_pp"]
    return d["lg_next"] + num / den


def fit_n0(d, delta, depth):
    f = lambda lg: float(np.sum(d["w"] * (d["x_next"] - pred(d, np.exp(lg), delta, depth)) ** 2) / d["w"].sum())
    return float(np.exp(minimize_scalar(f, bounds=(np.log(5), np.log(20000)), method="bounded").x))


def wr(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; fit = hp.fit_all(p)
    stats = {s: frame(q, s) for s in hp.SPEC if s != "hbpr"}
    for scheme, plan in (("FORWARD (train on earlier transitions)", [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]), ("LOTO (all other transitions)", [(tuple(t for t in (2022, 2023, 2024, 2025) if t != T), T) for T in (2024, 2025)])):
        print(f"\n{'=' * 100}\n{scheme}: held-out RMSE on the folds where a 3-season history exists (2024->25 and 2025->26)\n{'=' * 100}")
        rows = []; held = {}
        for s, d in stats.items():
            delta = hp.SPEC[s][0]; P = {2: [], 3: []}; Y = []
            for tr, te in plan:
                dtr = d[d["t"].isin(tr)]; dte = d[d["t"] == te].copy()
                for depth in (2, 3):
                    n0 = fit_n0(dtr, delta, depth); P[depth].append(pred(dte, n0, delta, depth).to_numpy())
                Y.append(dte)
            Dte = pd.concat(Y); y = Dte["x_next"].to_numpy(); w = Dte["w"].to_numpy(); ids = Dte["name_clean"].to_numpy(); p2, p3 = np.concatenate(P[2]), np.concatenate(P[3])
            by = pd.Series(np.arange(len(y))).groupby(ids).apply(np.array); ug = by.index.to_numpy(); dd = []
            for _ in range(600):
                ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wr(y[ii], p3[ii], w[ii]) - wr(y[ii], p2[ii], w[ii]))
            dd = np.array(dd); h3 = (Dte["n_pp"] > 0).to_numpy()
            rows.append({"stat": s, "n": len(y), "n with 3rd season": int(h3.sum()), "RMSE 2 seasons": wr(y, p2, w), "RMSE 3 seasons": wr(y, p3, w), "change": wr(y, p3, w) - wr(y, p2, w), "90% interval": f"[{np.percentile(dd, 5):+.5f}, {np.percentile(dd, 95):+.5f}]", "P(better)": float((dd < 0).mean()),
                         "change, rows with a 3rd season": wr(y[h3], p3[h3], w[h3]) - wr(y[h3], p2[h3], w[h3])})
            held[s] = (Dte, p2, p3)
        print(pd.DataFrame(rows).round(5).to_string(index=False))
    # production wOBA blend: direct + components (fit weights from the production fit), forward on 2025->26 only
    print("\nBLENDED wOBA (production: 0.5 direct + 0.5 components), forward 2025->26, n0 for each stat refit on 2022-24 transitions per depth:")
    res = {}
    for depth in (2, 3):
        pr = {}
        for s, d in stats.items():
            delta = hp.SPEC[s][0]; n0 = fit_n0(d[d["t"].isin((2022, 2023, 2024))], delta, depth); dte = d[d["t"] == 2025]; pr[s] = pd.Series(pred(dte, n0, delta, depth).to_numpy(), index=dte["name_clean"].to_numpy())
        idx = pr["woba"].index; comp = sum(fit["comp_beta"][s] * (pr[s].reindex(idx) - stats[s][stats[s]["t"] == 2025]["lg_next"].iloc[0]) for s in ("k", "bbr", "hrr", "babip")).fillna(0.0)
        lgw = stats["woba"][stats["woba"]["t"] == 2025]["lg_next"].iloc[0]; res[depth] = 0.5 * pr["woba"] + 0.5 * (lgw + comp)
    dte = stats["woba"][stats["woba"]["t"] == 2025].set_index("name_clean"); y = dte["x_next"].reindex(res[2].index).to_numpy(); w = dte["w"].reindex(res[2].index).to_numpy(); ok = ~np.isnan(y)
    a, b = res[2].to_numpy()[ok], res[3].to_numpy()[ok]; y, w = y[ok], w[ok]; ids = res[2].index.to_numpy()[ok]
    by = pd.Series(np.arange(len(y))).groupby(ids).apply(np.array); ug = by.index.to_numpy(); dd = []
    for _ in range(800):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wr(y[ii], b[ii], w[ii]) - wr(y[ii], a[ii], w[ii]))
    dd = np.array(dd); print(f"  n={len(y)}: 2 seasons {wr(y, a, w):.5f}, 3 seasons {wr(y, b, w):.5f}, change {wr(y, b, w) - wr(y, a, w):+.5f} [{np.percentile(dd, 5):+.5f}, {np.percentile(dd, 95):+.5f}] P(better) {np.mean(dd < 0):.2f}")


if __name__ == "__main__":
    main()
