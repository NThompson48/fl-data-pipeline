"""
fl_hierarchical_k_model.py -- a genuinely hierarchical (binomial GLMM)
alternative to the current two-stage "EB-shrink the naive prior, then WLS
regress" approach, motivated by the PA-weighting investigation this session
(see DECISION_LOG.md): rather than approximating "how much to trust a
low-PA point" via an external weight function, model it directly.

Level 1 (observation/sampling): K_i ~ Binomial(PA_test_i, theta_i) -- the
    observed strikeout count is a binomial draw at the pitcher's TRUE rate,
    with their own real PA_test. This is what correctly represents "small
    samples are noisier" -- it falls out of the likelihood itself, no
    weight formula needed.
Level 2 (regression): logit(theta_i) = X_i @ beta + eps_i,
    eps_i ~ Normal(0, sigma^2) -- the true rate depends on the same
    covariates already used (naive_K_rate, fl_k_rate_bridge, wh_sva_level,
    spin_z_norm, extension), plus a residual random-effect term for
    genuine player-to-player heterogeneity the covariates don't explain.

Non-centered parameterization (eps_i = sigma * z_i, z_i ~ Normal(0,1)) for
sampling efficiency -- standard practice for hierarchical models in PyMC.

Usage:
    python fl_hierarchical_k_model.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

SPEC_K = bt.MODEL_SPECS["naive+other_league+surprise_prior"]["K"]


def build_hierarchical_model(d, spec):
    X = d[spec].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)  # standardize for sane priors/sampling
    K = (d["K_rate_actual"].values * d["PA_test"].values).round().astype(int)
    n = d["PA_test"].values.astype(int)
    n_obs, n_feat = X_std.shape

    with pm.Model() as model:
        X_data = pm.Data("X", X_std)
        intercept = pm.Normal("intercept", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2, shape=n_feat)
        sigma = pm.HalfNormal("sigma", sigma=1)
        z = pm.Normal("z", mu=0, sigma=1, shape=n_obs)
        eps = z * sigma
        logit_theta = intercept + pm.math.dot(X_data, beta) + eps
        theta = pm.Deterministic("theta", pm.math.invlogit(logit_theta))
        pm.Binomial("K_obs", n=n, p=theta, observed=K)
    return model, X.mean(axis=0), X.std(axis=0)


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    d_tr = df_tr.dropna(subset=["K_rate_actual"] + SPEC_K).copy()
    d_te = df_te.dropna(subset=["K_rate_actual"] + SPEC_K).copy()
    print(f"n_train={len(d_tr)}  n_test={len(d_te)}")

    print("\nBuilding and sampling the hierarchical binomial GLMM...")
    model, X_mean, X_std = build_hierarchical_model(d_tr, SPEC_K)
    with model:
        idata = pm.sample(draws=1500, tune=1500, chains=4, target_accept=0.95,
                           random_seed=42, progressbar=True)

    print("\nPosterior summary (intercept, beta, sigma):")
    import arviz as az
    summary = az.summary(idata, var_names=["intercept", "beta", "sigma"])
    print(summary)

    post_intercept = idata.posterior["intercept"].values.mean()
    post_beta = idata.posterior["beta"].values.mean(axis=(0, 1))
    post_sigma = idata.posterior["sigma"].values.mean()
    print(f"\npoint estimate -- intercept={post_intercept:.4f}  beta={post_beta.round(4)}  sigma={post_sigma:.4f}")

    # predict on held-out: standardize test features with TRAIN mean/std, apply posterior mean beta
    X_te = d_te[SPEC_K].values
    X_te_std = (X_te - X_mean) / X_std
    logit_pred = post_intercept + X_te_std @ post_beta
    pred = 1 / (1 + np.exp(-logit_pred))

    w_te = mm.reliability_weight(d_te["PA_test"], n0_k).values
    r2 = mm.weighted_r2(d_te["K_rate_actual"].values, pred, w_te)
    mae = mm.weighted_mae(d_te["K_rate_actual"].values, pred, w_te)
    rmse = mm.weighted_rmse(d_te["K_rate_actual"].values, pred, w_te)
    print(f"\nheld-out (current n0=64.1 weight): R2_w={r2:.4f}  RMSE_w={rmse:.4f}  MAE_w={mae:.4f}")

    d_te = d_te.copy()
    d_te["pred"] = pred
    d_te["w"] = w_te
    d_te["weighted_sq_err"] = d_te["w"] * (d_te["K_rate_actual"] - d_te["pred"]) ** 2
    d_te["pa_bucket"] = pd.cut(d_te["PA_test"], [0, 15, 30, 60, 100, 200, 99999],
                               labels=["1-15", "16-30", "31-60", "61-100", "101-200", "200+"])
    g = d_te.groupby("pa_bucket", observed=True)["weighted_sq_err"].mean()
    print("\nweighted_sq_err by PA bucket:", g.round(5).to_dict())
    print(f"flatness ratio (lowest/highest): {g.iloc[0] / g.iloc[-1]:.3f}")

    idata.to_netcdf("output/_scratch_hierarchical_k_idata.nc")
    print("\nSaved posterior trace -> output/_scratch_hierarchical_k_idata.nc")
