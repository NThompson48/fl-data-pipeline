"""
fl_pwar_calibration_lens.py -- evaluate P(WAR>0) for what it is meant to be (a decision-ready statement of prediction uncertainty, not a classifier): is the predictive distribution of RA9 calibrated?
Same out-of-sample 2025 -> 2026 build as fl_pwar_calibration.py (broad sample, >= 10 IP in 2026).  Checks:
 (1) reliability of P: predicted vs observed frequency by P bin with binomial intervals (pooled and by PROJECTED-IP bucket, the information available before the season);
 (2) coverage of the central 50 / 80 / 90 % predictive intervals of RA9 (pred +- z sd), i.e. is the spread right?  and the PIT histogram (uniform if calibrated);
 (3) decile calibration slope on the logit scale.  Brier skill and AUC are NOT used.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
import fl_pwar_calibration as cal


def wilson(k, n, z=1.645):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    d, war = cal.main(); d = d.copy()
    d["pit"] = norm.cdf((d["target"] - d["pred"]) / d["sd"])       # PIT of the realized RA9 under the predictive normal
    d["ipb"] = pd.cut(d["ip_proj"], [0, 22, 30, 45, 1000], labels=["<22 proj IP", "22-30", "30-45", "45+"])
    print("\n" + "=" * 100 + "\n(1) RELIABILITY of P(RA9 < replacement): predicted vs observed, 90% Wilson interval\n" + "=" * 100)
    for lab, g0 in [("all", d)] + [(str(k), g) for k, g in d.groupby("ipb", observed=True)]:
        g0 = g0.assign(bin=pd.cut(g0["P"], [0, .5, .6, .7, .8, 1.0])); rows = []
        for b, g in g0.groupby("bin", observed=True):
            lo, hi = wilson(g["y"].sum(), len(g)); rows.append({"P bin": str(b), "n": len(g), "mean predicted": g["P"].mean(), "observed": g["y"].mean(), "90% interval": f"[{lo:.2f}, {hi:.2f}]", "predicted inside?": lo <= g["P"].mean() <= hi})
        print(f"\n  {lab} (n={len(g0)}; mean predicted {g0['P'].mean():.3f}, observed {g0['y'].mean():.3f})"); print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\n" + "=" * 100 + "\n(2) COVERAGE of central predictive intervals for RA9 (nominal 50 / 80 / 90%) and PIT\n" + "=" * 100)
    rows = []
    for lab, g in [("all", d)] + [(str(k), g) for k, g in d.groupby("ipb", observed=True)]:
        z = ((g["target"] - g["pred"]) / g["sd"]).abs(); r = {"group": lab, "n": len(g)}
        for lev in (.5, .8, .9):
            r[f"{int(lev * 100)}%"] = float((z <= norm.ppf(0.5 + lev / 2)).mean())
        r["mean z"] = float(((g["target"] - g["pred"]) / g["sd"]).mean()); r["sd of z"] = float(((g["target"] - g["pred"]) / g["sd"]).std()); rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    h = np.histogram(d["pit"], bins=np.linspace(0, 1, 11))[0] / len(d); print("  PIT histogram, deciles (uniform = .10 each):", np.round(h, 3).tolist())
    print("\n" + "=" * 100 + "\n(3) logit-scale calibration slope by projected-IP bucket (1.0 / 0 = calibrated)\n" + "=" * 100)
    for lab, g in [("all", d)] + [(str(k), g) for k, g in d.groupby("ipb", observed=True)]:
        g = g.assign(l=np.log(g["P"].clip(1e-3, 1 - 1e-3) / (1 - g["P"].clip(1e-3, 1 - 1e-3))))
        if g["y"].nunique() > 1:
            m = smf.logit("y ~ l", data=g).fit(disp=0); print(f"  {lab:12s} n={len(g):3d}  intercept {m.params['Intercept']:+.2f} (se {m.bse['Intercept']:.2f})  slope {m.params['l']:.2f} (se {m.bse['l']:.2f})")


if __name__ == "__main__":
    main()
