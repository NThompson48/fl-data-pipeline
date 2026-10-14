"""
fl_hitter_air_babip_component_test.py -- Air% as an informative prior in the BABIP COMPONENT of the wOBA projection.
Three versions of the component half of the blend (the blend is always 0.5 direct + 0.5 components; direct = the production wOBA regression):
  V0  K%, BB%, HR/PA components only (production since the BABIP drop)
  V1  + BABIP regressed as before (the old production component)
  V2  + BABIP with an Air% prior: BABIP_pred = regressed BABIP + b x shrunk Air% deviation (Air% = share of tracked balls with launch angle >= 10, shrunk by n0 = 89 balls; b fit on the training folds)
Forward 2025 -> 2026 (production three-season regressions, n0 per stat fit on the 2022-24 transitions; Air% from 2025 balls only; Trackman exists only for 2025-26 so the Air% coefficient and the component weights are fit by
hitter-grouped 10-fold CV x 15 repeats on this one transition, every version with the same free intercept): next-year park-neutral wOBA, >= 100 PA, standard weights, weighted RMSE and MAE with hitter-cluster bootstrap intervals.
Also scored: the BABIP prediction itself.  Forward only (a reverse 2026 -> 2025 check was not run: the 2025 position / fielding style gaps do not matter here but the user flagged reverse as leaky).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import fl_eval_report as er
import fl_hitter_air_babip_test as ab
import fl_hitter_depth3_test as d3
import fl_hitter_ev as hev
import fl_hitter_model_dev as dev
import fl_hitter_projection as hp

warnings.filterwarnings("ignore")
STATS = ("woba", "k", "bbr", "hrr", "babip")


def air_table():
    b = hev.load_bbe(); b["air"] = (b["angle"] >= 10).astype(int)
    g = b.groupby(["name_clean", "year"]).agg(n=("air", "size"), air=("air", "sum")).reset_index(); lg = {y: x["air"].sum() / x["n"].sum() for y, x in g.groupby("year")}
    return g.set_index(["name_clean", "year"]), lg


def build_forward(p, g, lg):
    q = p.copy(); q["woba"] = q["woba_n"]; fr = {}
    for s in STATS:
        d = d3.frame(q, s); n0 = d3.fit_n0(d[d["t"].isin((2022, 2023, 2024))], hp.SPEC[s][0], 3); te = d[d["t"] == 2025].copy(); te["pred"] = d3.pred(te, n0, hp.SPEC[s][0], 3); fr[s] = te.set_index("name_clean")
    idx = fr["woba"].index
    for s in STATS:
        idx = idx.intersection(fr[s].index)
    D = pd.DataFrame(index=idx)
    for s in STATS:
        D[f"pred_{s}"] = fr[s].loc[idx, "pred"]; D[f"lg_{s}"] = fr[s].loc[idx, "lg_next"]
    D["y"] = fr["woba"].loc[idx, "x_next"]; D["y_babip"] = fr["babip"].loc[idx, "x_next"]; D["w"] = fr["woba"].loc[idx, "w"]; D["w_babip"] = fr["babip"].loc[idx, "w"]
    n = np.array([g["n"].get((nm, 2025), np.nan) for nm in idx]); a = np.array([g["air"].get((nm, 2025), np.nan) for nm in idx]); ok = ~np.isnan(n) & (n > 0)
    D["air"] = np.where(ok, (a / np.where(ok, n, 1) - lg[2025]) * n / (n + ab.N0_AIR), np.nan); return D.dropna(subset=["air", "y", "y_babip"])


def cv_variants(D, reps=15):
    y = D["y"].to_numpy(); w = D["w"].to_numpy(); n = len(D); out = {k: np.zeros(n) for k in ("V0", "V1", "V2", "babip_ref", "babip_air")}
    lgw = D["lg_woba"].to_numpy(); P = {s: D[f"pred_{s}"].to_numpy() - D[f"lg_{s}"].to_numpy() for s in STATS}; air = D["air"].to_numpy(); yb = D["y_babip"].to_numpy(); wb = D["w_babip"].to_numpy(); base_b = D["pred_babip"].to_numpy()
    for rep in range(reps):
        for tr, te in KFold(10, shuffle=True, random_state=rep).split(y):
            sw = np.sqrt(wb[tr]); A = np.column_stack([np.ones(len(tr)), air[tr]]); bcoef = np.linalg.lstsq(A * sw[:, None], (yb - base_b)[tr] * sw, rcond=None)[0]
            c0b = np.average((yb - base_b)[tr], weights=wb[tr]); out["babip_ref"][te] += (base_b[te] + c0b) / reps; out["babip_air"][te] += (base_b[te] + bcoef[0] + bcoef[1] * air[te]) / reps
            babip_air = P["babip"] + bcoef[0] + bcoef[1] * air                                   # BABIP deviation with the Air% prior (all rows; coefficient from the training fold)
            for name, cols in (("V0", ("k", "bbr", "hrr")), ("V1", ("k", "bbr", "hrr", "babip")), ("V2", ("k", "bbr", "hrr", "babip_air"))):
                X = np.column_stack([babip_air if c == "babip_air" else P[c] for c in cols]); sw2 = np.sqrt(w[tr]); A2 = np.column_stack([np.ones(len(tr)), X[tr]])
                beta = np.linalg.lstsq(A2 * sw2[:, None], (y - lgw)[tr] * sw2, rcond=None)[0]; comp = lgw[te] + beta[0] + X[te] @ beta[1:]
                direct = D["pred_woba"].to_numpy()[te]; c_dir = np.average((y - D["pred_woba"].to_numpy())[tr], weights=w[tr])                       # same fitted intercept for the direct half
                out[name][te] += (0.5 * (direct + c_dir) + 0.5 * comp) / reps
    return out


def main():
    p = dev.load(); g, lg = air_table(); D = build_forward(p, g, lg); out = cv_variants(D); y = D["y"].to_numpy(); w = D["w"].to_numpy(); ids = D.index.to_numpy()
    print(f"FORWARD 2025 -> 2026: n = {len(D)} hitters (>= 100 PA in 2026, Trackman balls in 2025)")
    rows = [{"comparison": "BABIP prediction: Air% prior vs none (BABIP target, BABIP weights)", **er.compare(D["y_babip"].to_numpy(), out["babip_ref"], out["babip_air"], D["w_babip"].to_numpy(), ids, n=500)},
            {"comparison": "wOBA blend: V2 (BABIP + Air% prior) vs V0 (no BABIP) [current]", **er.compare(y, out["V0"], out["V2"], w, ids, n=500)},
            {"comparison": "wOBA blend: V2 (BABIP + Air% prior) vs V1 (BABIP, no Air%) [old]", **er.compare(y, out["V1"], out["V2"], w, ids, n=500)},
            {"comparison": "wOBA blend: V1 (BABIP, no Air%) vs V0 (no BABIP)", **er.compare(y, out["V0"], out["V1"], w, ids, n=500)}]
    print(er.table(rows, 5))
    r0, m0 = er.wmetrics(y, out["V0"], w); print(f"\n  levels: V0 RMSE {r0:.5f} MAE {m0:.5f}; V1 {er.wmetrics(y, out['V1'], w)[0]:.5f} / {er.wmetrics(y, out['V1'], w)[1]:.5f}; V2 {er.wmetrics(y, out['V2'], w)[0]:.5f} / {er.wmetrics(y, out['V2'], w)[1]:.5f}")


if __name__ == "__main__":
    main()
