"""
fl_rookie_value_test.py -- WHAT ARE THE ROOKIE BRIDGES WORTH IN WINS?  (user 2026-10-10: "what about on rookies?")  FL DEBUT players with other-league history; out-of-sample (leave-one-out / leave-one-debut-class-out) bridge predictions vs naive ways of using the same raw stats.
HITTERS (200 debut hitters 2023-26 with MiLB history, fl_hitter_bridge_test.py): MODEL = bridge wOBA (0.5 direct + 0.5 component-built, out-of-fold, tuned inside the folds); RAW = rank by the raw MiLB wOBA proxy of the latest source season (no level adjustment, no shrinkage);
   RANDOM = no information.  Realised = first-FL-season wOBA deviation from the FL league, converted to batting WAR per 400 PA = 400 x (dev / 1.0175) / 11.98 (the replacement term cancels in differences; fielding / position / baserunning not included).
PITCHERS (unified bridge debut pairs, fl_unified_bridge.py): MODEL = bridge K% and BB% (leave-one-out), RAW = rank by raw latest-season K% minus raw multi-year BB%; realised = the first FL season's K% and BB% turned into runs per 9 with the production RA9 baseline slope
   (RA9 = const - 9.238 K% + 11.53 BB%; innings per game cancel); WAR per 100 IP = 100/9 x (runs saved vs the pool mean)/10.706.  (K/BB-skill outcome: BABIP/HR luck is not in it.)
ACCURACY: weighted RMSE / MAE (inverse-variance weights from the league-mean guess residuals, frozen), MODEL vs the debut-class-mean guess, bootstrap over players.  SELECTION: the K best by MODEL vs by RAW from random candidate pools of M rookies, realised WAR difference, bootstrap over rookies.
"""
import io
import contextlib
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_bridge_test as bt
import fl_pitcher_features as fp
import fl_unified_bridge as ub

warnings.filterwarnings("ignore")
BETA = {"k": -0.177, "bbr": 0.516, "hrr": 1.414}; RPW_H = 11.98; RPW_P = 10.706


def hitters():
    with contextlib.redirect_stdout(io.StringIO()):
        res = bt.main()
    r = res["woba"]; dd = r["dd"].reset_index(drop=True); idx = {nm: i for i, nm in enumerate(dd["nm"])}; comp = np.zeros(len(dd))
    for s, b in BETA.items():
        q = res[s]; j = {nm: i for i, nm in enumerate(q["dd"]["nm"])}; comp += b * np.array([q["P"]["M2 pooled slope"][j[nm]] if nm in j else 0.0 for nm in dd["nm"]])
    model = 0.5 * r["P"]["M2 pooled slope"] + 0.5 * comp; base = r["P"]["M1 debut-class mean"]; y = r["y"]; n = r["n"]
    m = bt.milb_table(); raw = []
    for nm, yr in zip(dd["nm"], dd["year"]):
        g = m[(m["nm"] == nm) & (m["season"] < yr)]; ls = g["season"].max(); g = g[g["season"] == ls]; raw.append(np.average(g["woba"], weights=g["pa"]))
    D = pd.DataFrame({"name": dd["nm"], "n": n, "real": y, "MODEL": model, "BASE": base, "RAW": raw}); D["RAW"] = D["RAW"].replace([np.inf, -np.inf], np.nan); D["RAW"] = D["RAW"].fillna(D["RAW"].median()); return D, 400 / 1.0175 / RPW_H


def pitchers():
    raw = fp.load_fl_pitches_cached()
    with contextlib.redirect_stdout(io.StringIO()):
        mdl = ub.fit(raw, verbose=False)
    p = mdl.pairs.copy().reset_index(drop=True); w = p["pa"].to_numpy(float); leagues = sorted(p["league"].unique()); I = np.column_stack([(p["league"] == lg).astype(float) for lg in leagues])

    def loo(X, y):
        XtWX = X.T @ (X * w[:, None]); beta = np.linalg.solve(XtWX, X.T @ (w * y)); e = y - X @ beta; h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(XtWX), X); return y - e / (1 - h), (np.sum(w * y) - w * y) / (np.sum(w) - w)
    pk, bk = loo(np.column_stack([I, p["u_k"].to_numpy()]), p["k_rate"].to_numpy()); pb, bb = loo(np.column_stack([np.ones(len(p)), p["u_bb"].to_numpy()]), p["bb_rate"].to_numpy())
    ra = lambda k, b: -9.238 * k + 11.53 * b
    D = pd.DataFrame({"name": p["pitcher_id"] + "|" + p["league"], "pid": p["pitcher_id"], "n": w, "real": ra(p["k_rate"], p["bb_rate"]), "MODEL": ra(pk, pb), "BASE": ra(bk, bb), "RAW": ra(p["k_raw_latest"], p["bb_raw_multi"])})
    D = D.replace([np.inf, -np.inf], np.nan).dropna().sort_values("n", ascending=False).drop_duplicates("pid").reset_index(drop=True); return D, 100 / 9 / RPW_P


def run(D, unit, label, sign):
    """sign = +1 if higher = better (wOBA), -1 if lower = better (RA9-equivalent)"""
    y, mo, ba, rw, n = D["real"].to_numpy(), D["MODEL"].to_numpy(), D["BASE"].to_numpy(), D["RAW"].to_numpy(), D["n"].to_numpy(float)
    a, b = ew.fit_variance_law(y - ba, n); w = ew.inv_var_weights(n, a, b); rep = er.compare(y, ba, mo, w, D["name"].to_numpy(), n=1000)
    print(f"\n=== {label}: n = {len(D)}\nACCURACY (weighted; converted to WAR {('per 400 PA' if sign > 0 else 'per 100 IP')} by x {unit:.2f}): league-mean guess RMSE {rep['RMSE old']:.4f} ({rep['RMSE old'] * unit:.2f} WAR) / MAE {rep['MAE old']:.4f} ({rep['MAE old'] * unit:.2f}) -> MODEL RMSE {rep['RMSE new']:.4f} ({rep['RMSE new'] * unit:.2f}) / MAE {rep['MAE new']:.4f} ({rep['MAE new'] * unit:.2f}); "
          f"dRMSE {rep['dRMSE']:+.4f} {rep['RMSE 90%']} P(better) {rep['P(RMSE better)']:.2f}; dMAE {rep['dMAE']:+.4f} P {rep['P(MAE better)']:.2f}")
    print(f"correlation with realised: MODEL {np.corrcoef(mo, y)[0, 1]:.2f}, RAW {np.corrcoef(rw, y)[0, 1]:.2f}; Spearman MODEL {pd.Series(mo).corr(pd.Series(y), method='spearman'):.2f}, RAW {pd.Series(rw).corr(pd.Series(y), method='spearman'):.2f}")
    rng = np.random.default_rng(31)

    def picks(Dx, K, M, reps):
        yy, mm, rr = Dx["real"].to_numpy(), Dx["MODEL"].to_numpy(), Dx["RAW"].to_numpy(); nn = len(yy)
        if M is None:
            s = lambda p: np.argsort(-sign * p)[:K]; return np.array([yy[s(mm)].mean() * sign, yy[s(rr)].mean() * sign, yy.mean() * sign])
        out = np.zeros((reps, 3))
        for r_ in range(reps):
            i = rng.choice(nn, M, replace=False); s = lambda p: i[np.argsort(-sign * p[i])[:K]]; out[r_] = [yy[s(mm)].mean() * sign, yy[s(rr)].mean() * sign, yy[i].mean() * sign]
        return out.mean(axis=0)
    rows = []
    for K in (3, 5):
        for M in (None, 40, 20):
            est = picks(D, K, M, 1500); bs = np.array([picks(D.iloc[rng.integers(0, len(D), len(D))].reset_index(drop=True), K, M, 250) for _ in range(250)])
            d_raw = (bs[:, 0] - bs[:, 1]) * unit * K; d_rnd = (bs[:, 0] - bs[:, 2]) * unit * K
            rows.append({"K picked": K, "from pool of": f"whole pool ({len(D)})" if M is None else M, "wins/yr MODEL vs RAW stats": (est[0] - est[1]) * unit * K, "90%": f"[{np.percentile(d_raw, 5):+.2f}, {np.percentile(d_raw, 95):+.2f}]", "wins/yr MODEL vs RANDOM": (est[0] - est[2]) * unit * K, "90% ": f"[{np.percentile(d_rnd, 5):+.2f}, {np.percentile(d_rnd, 95):+.2f}]"})
    print(f"SELECTION ({'each pick = 400 PA' if sign > 0 else 'each pick = 100 IP'}; wins per season for the K picks together, 1 WAR ~ 1 win)\n" + pd.DataFrame(rows).round(2).to_string(index=False))


def main():
    H, uh = hitters(); P, up = pitchers(); run(H, uh, "ROOKIE HITTERS (MiLB -> FL debut)", +1); run(P, up, "ROOKIE PITCHERS (college / Pioneer / MiLB -> FL debut)", -1)


if __name__ == "__main__":
    main()
