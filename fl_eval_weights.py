"""
fl_eval_weights.py -- the project's standard evaluation / fitting weight for a target observed at sample size n.

A target rate y observed over n trials (PA, batted balls, innings...) has error variance around the projection
        Var_i = a + b / n_i          a = talent + model-misfit variance (does not shrink with n),  b / n = sampling noise
and the efficient weight is its inverse, w_i = 1 / (a + b / n_i).  This is exactly the "reliability weight" n / (n + n0_eval)
(reliability = share of Var_i that is signal at that n) up to a constant, with  n0_eval = b / a;  it is ONE weight, not two:
multiplying a reliability weight by an inverse variance would count the sample-size effect twice (the sensitivity run in
fl_hitter_weight_audit.py shows that variant).  a and b are estimated by Gaussian maximum likelihood from the residuals of a
reference model, then FIXED for every candidate model being compared, so the weights cannot favour the model they came from.

fit_variance_law(err, n)   -> (a, b)
inv_var_weights(n, a, b)   -> w   (mean-normalised)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def fit_variance_law(err, n, floor: float = 1e-10):
    e = np.asarray(err, dtype=float); n = np.asarray(n, dtype=float)
    ok = np.isfinite(e) & np.isfinite(n) & (n > 0)
    e, n = e[ok], n[ok]
    s2 = float(np.mean(e ** 2))

    def nll(p):
        v = p[0] + p[1] / n
        return 0.5 * np.sum(np.log(v) + e ** 2 / v)
    starts = ([0.3 * s2, 0.7 * s2 * np.median(n)], [0.05 * s2, 0.95 * s2 * np.median(n)], [floor, s2 * np.median(n)], [0.8 * s2, 0.2 * s2 * np.median(n)])
    best = min((minimize(nll, x0, bounds=[(floor, None), (floor, None)], method="L-BFGS-B") for x0 in starts), key=lambda r: r.fun)
    return float(best.x[0]), float(best.x[1])


def inv_var_weights(n, a, b):
    n = np.asarray(n, dtype=float)
    w = 1.0 / (a + b / n)
    return w / np.mean(w)


def weights_from_base(y, base, n):
    """Fit the variance law on a reference model's residuals and return (w, a, b); used to score a target that has no precomputed weight column."""
    a, b = fit_variance_law(np.asarray(y, dtype=float) - np.asarray(base, dtype=float), n)
    return inv_var_weights(n, a, b), a, b
