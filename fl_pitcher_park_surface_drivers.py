"""fl_pitcher_park_surface_drivers.py -- what drives a pitcher's Trustmark surface shift: Air% (predictable from traits) or the hard-hit air-ball share (needs exit speed allowed)?  Pitcher level (>= 150 non-Trustmark balls), both seasons pooled."""
import warnings
import numpy as np
import pandas as pd
import fl_hitter_park_ms_check as ms
import fl_pitcher_park_surface_test as pt
warnings.filterwarnings("ignore")
b = pt.load(); eff = ms.fit_effect_pooled(b, "evla_side"); x = b[b["tm"] == 0].copy(); x["eff"] = eff(x); x["air"] = (x["la"] >= 10).astype(float); x["hardair"] = ((x["ev"] >= 90) & (x["la"] >= 10) & (x["la"] < 40)).astype(float); x["hard"] = (x["ev"] >= 90).astype(float); x["fbpu"] = (x["la"] >= 25).astype(float)
g = x.groupby("pid").agg(n=("eff", "size"), shift=("eff", "mean"), air=("air", "mean"), hardair=("hardair", "mean"), hard=("hard", "mean"), fbpu=("fbpu", "mean"), ev=("ev", "mean")); g = g[g["n"] >= 150]
print(f"{len(g)} pitchers; corr with the pitcher's mean surface effect (more negative = bigger park penalty):")
for c in ("air", "fbpu", "hard", "ev", "hardair"):
    print(f"  {c:8s} corr {g['shift'].corr(g[c]):+.2f}")
rng = np.random.default_rng(5); x["h"] = rng.integers(0, 2, len(x))
for c in ("air", "hard", "ev", "hardair", "eff"):
    t = x.groupby(["pid", "h"])[c].agg(["mean", "size"]).unstack("h").dropna(); t = t[(t[("size", 0)] >= 40) & (t[("size", 1)] >= 40)]; r = np.corrcoef(t[("mean", 0)], t[("mean", 1)])[0, 1]; print(f"  split-half full-length reliability of {c:8s}: {2 * r / (1 + r):.2f} (n {len(t)})")
