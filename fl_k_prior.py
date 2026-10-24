"""
fl_k_prior.py -- the production K% model with STUFF as a fading informative prior  (user 2026-10-24: "Add it and push"; tested in fl_wh_sva_prior_k_test.py, robustness in fl_wh_sva_prior_*_robust_test.py)

    P_stuff   = lg + c * (wh_sva_level - stuff_mean)
    blended_K = (K_count + n0*lg + n0_s*P_stuff) / (PA + n0 + n0_s)        (K_count = naive_K_rate*(PA+n0) - n0*lg, i.e. the pitcher's own strikeouts)
    K% model  = WLS on [blended_K_stuff, fl_k_rate_bridge, spin_z_norm, extension]
(c, n0_s) are fit JOINTLY with the regression by nonlinear least squares on the training transition (inner WLS refit at each trial; Nelder-Mead on (log c-multiplier, log n0_s)), replacing the earlier c-grid x
n0_s fit whose network optimum sat on the grid edge.  The returned object behaves like the statsmodels result the rest of the project calls (predict / params / get_prediction / tvalues) so every existing caller keeps working;
predict(df) builds blended_K_stuff itself from naive_K_rate, PA (or PA_model_k when present) and wh_sva_level, using the frozen n0 and league rate unless df carries 'lg_k_frame' / 'n0_k_frame'.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import fl_halfseason_backtest as bt
import mlb_marcel as mm

PRIOR_SPEC = ["blended_K_stuff", "fl_k_rate_bridge", "spin_z_norm", "extension"]


def add_blend(df, c, n0_s, stuff_mean, n0_k, lg):
    d = df.copy(); pa = d["PA_model_k"] if "PA_model_k" in d else d["PA"]
    lg = d["lg_k_frame"] if "lg_k_frame" in d else lg; n0 = d["n0_k_frame"] if "n0_k_frame" in d else n0_k
    d["K_raw"] = d["naive_K_rate"] * (pa + n0) - n0 * lg; d["P_stuff"] = lg + c * (d["wh_sva_level"] - stuff_mean)
    d["blended_K_stuff"] = mm.blended_rate(d["K_raw"], pa, n0, lg, n0_s, d["P_stuff"]); return d


class PriorKModel:
    def __init__(self, ols, c, n0_s, stuff_mean, n0_k, lg, c0):
        self.ols, self.c, self.n0_s, self.stuff_mean, self.n0_k, self.lg, self.c0 = ols, c, n0_s, stuff_mean, n0_k, lg, c0
        self.params, self.tvalues = ols.params, ols.tvalues

    def frame(self, df):
        return add_blend(df, self.c, self.n0_s, self.stuff_mean, self.n0_k, self.lg)

    def predict(self, df):
        return self.ols.predict(self.frame(df))

    def get_prediction(self, df, weights=None):
        return self.ols.get_prediction(self.frame(df), weights=weights)

    def prior_dict(self):
        return {"c": float(self.c), "n0_s": float(self.n0_s), "stuff_mean": float(self.stuff_mean), "n0_k": float(self.n0_k), "lg": float(self.lg), "c_ols_slope": float(self.c0)}


def fit(d, n0_k, lg, w, target="K_rate_actual", extra=()):
    """d: training frame with naive_K_rate, PA, wh_sva_level, fl_k_rate_bridge, spin_z_norm, extension, target.  Returns a PriorKModel."""
    spec = PRIOR_SPEC + list(extra); stuff_mean = float(d["wh_sva_level"].mean()); ds = (d["wh_sva_level"] - stuff_mean).to_numpy(); y = d[target].to_numpy()
    c0 = float(np.sum(w * ds * (y - np.average(y, weights=w))) / np.sum(w * ds ** 2))

    def sse(theta):
        c, n0_s = c0 * np.exp(theta[0]), np.exp(theta[1]); m = bt.fit_weighted(add_blend(d, c, n0_s, stuff_mean, n0_k, lg), target, spec, w)
        return float(np.sum(w * (y - m.predict(add_blend(d, c, n0_s, stuff_mean, n0_k, lg)).to_numpy()) ** 2))

    best = min((minimize(sse, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-10, "maxiter": 200}) for x0 in ([0.0, np.log(30.0)], [0.4, np.log(60.0)])), key=lambda r: r.fun)
    c, n0_s = c0 * np.exp(best.x[0]), float(np.exp(best.x[1])); ols = bt.fit_weighted(add_blend(d, c, n0_s, stuff_mean, n0_k, lg), target, spec, w)
    return PriorKModel(ols, c, n0_s, stuff_mean, n0_k, lg, c0)
