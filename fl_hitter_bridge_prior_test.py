"""
fl_hitter_bridge_prior_test.py -- thin FL history: does the MiLB bridge, used as the PRIOR the debut season shrinks toward (instead of the league mean), improve the SECOND-season projection?  (user 2026-10-09: thin history includes rookies)
Sample: FL debut hitters 2023-25 with MiLB history (fl_hitter_bridge_test pairs, >= 20 debut PA) who have >= 20 PA in the next FL season; target = next-season rate minus that season's FL league mean.
  old: dev_hat = n/(n + n0) * dev_debut                      (shrink to the league mean; n0 = production single-season n0: wOBA 436, K 79.7, BB 146.7, HR 139.8)
  new: dev_hat = n/(n + n0) * dev_debut + n0/(n + n0) * prior (prior = out-of-fold bridge prediction for the debut class, leave-one-debut-class-out)
wOBA = 0.5 direct + 0.5 component-built (K -.177, BB +.516, HR +1.414), in both arms.  Weights: frozen inverse-variance weights from the OLD model's residuals at the next-season PA.  Player-cluster bootstrap (fl_eval_report); RMSE and MAE.
Also: how much of the debut deviation is shrunk away (mean shrink n0/(n+n0)) -- the more shrink, the more the prior matters.
"""
import io
import contextlib
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_bridge_test as bt

warnings.filterwarnings("ignore")
N0 = {"woba": 436.0, "k": 79.7, "bbr": 146.7, "hrr": 139.8}
BETA = {"k": -0.177, "bbr": 0.516, "hrr": 1.414}


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        res = bt.main()
    h = pd.read_parquet("output/fl_hitter_panel.parquet"); h["nm"] = h["name_clean"].map(bt.norm); h["k"] = h["so"] / h["pa"]; h["bbr"] = h["bb"] / h["pa"]; h["hrr"] = h["hr"] / h["pa"]
    lg = {s: h.dropna(subset=[s]).groupby("year").apply(lambda g: np.average(g[s], weights=g["pa"])) for s in bt.STATS}
    nxt = h[["nm", "year", "pa"] + list(bt.STATS)].copy(); nxt["year"] = nxt["year"] - 1; nxt = nxt.drop_duplicates(["nm", "year"]).set_index(["nm", "year"])
    r0 = res["woba"]["dd"]; keep = []
    for i, (nm, yr) in enumerate(zip(r0["nm"], r0["year"])):
        if yr <= 2025 and (nm, yr) in nxt.index and nxt.loc[(nm, yr), "pa"] >= 20:
            keep.append(i)
    keep = np.array(keep); names = r0["nm"].to_numpy()[keep]; yrs = r0["year"].to_numpy()[keep]; n_deb = r0["pa"].to_numpy(float)[keep]; n_next = np.array([nxt.loc[(a, b), "pa"] for a, b in zip(names, yrs)], float)
    print(f"{len(keep)} debut hitters (2023-25 classes) with MiLB history and >= 20 PA in their second FL season")
    old, new, Y = {}, {}, {}
    for s in bt.STATS:
        q = res[s]; idx = {nm: i for i, nm in enumerate(q["dd"]["nm"])}; sel = np.array([idx[nm] for nm in names]); dev = q["dd"][f"y_{s}"].to_numpy()[sel]; prior = q["P"]["M2 pooled slope"][sel]; sh = n_deb / (n_deb + N0[s])
        old[s] = sh * dev; new[s] = sh * dev + (1 - sh) * prior; Y[s] = np.array([nxt.loc[(a, b), s] - lg[s].loc[b + 1] for a, b in zip(names, yrs)]); print(f"  {bt.STATS[s]}: mean shrink factor n/(n+n0) {sh.mean():.2f} (weight on the prior {1 - sh.mean():.2f}); mean prior {prior.mean():+.4f}")
    direct_old, direct_new = old["woba"].copy(), new["woba"].copy()
    for arm in (old, new):
        arm["woba_comp"] = sum(BETA[s] * arm[s] for s in BETA)
        arm["woba_direct"] = arm["woba"].copy()
        arm["woba"] = 0.5 * arm["woba"] + 0.5 * arm["woba_comp"]
    rows = []
    for s, lab in bt.STATS.items():
        y = Y[s]; ok = np.isfinite(y); a, b = ew.fit_variance_law(y[ok] - old[s][ok], n_next[ok]); w = ew.inv_var_weights(n_next[ok], a, b); rep = er.compare(y[ok], old[s][ok], new[s][ok], w, names[ok], n=1000); rows.append({"stat": lab, **rep})
    for k, lab in (("woba_direct", "wOBA direct only"), ("woba_comp", "wOBA component-built only")):
        y = Y["woba"]; ok = np.isfinite(y); a, b = ew.fit_variance_law(y[ok] - old[k][ok], n_next[ok]); w = ew.inv_var_weights(n_next[ok], a, b); rows.append({"stat": lab, **er.compare(y[ok], old[k][ok], new[k][ok], w, names[ok], n=1000)})
    print(er.table(rows, 5))


if __name__ == "__main__":
    main()
