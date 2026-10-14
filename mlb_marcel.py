"""
mlb_marcel.py -- Marcel-the-Monkey building blocks for the MLB K%/BB% projection:
empirically-derived reliability (no fixed constant), classic 5/4/3 multi-year
weighting, and the reliability-weighted evaluation helpers used throughout the
projection notebook.

Reliability philosophy (matches k26_projection.py's estimate_n0(), extended to a
full empirical curve rather than a single point estimate): a rate stat observed
over n trials has Var(observed) = Var(true talent) + p(1-p)/n. Splitting the
league-wide observed variance into those two pieces gives a stabilization point
n0 such that reliability(n) = n / (n + n0). This is fit directly from the data,
never copied from a textbook constant, and should be refit per-metric (K%, BB%)
and per-fold in the walk-forward backtest (never on data that includes the test
year).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Empirical reliability: single-season method-of-moments n0, and the fuller
#  year-over-year empirical curve used to sanity-check it.
# --------------------------------------------------------------------------- #
def estimate_n0(rate: pd.Series, n: pd.Series, min_n: int = 25):
    """Data-estimated stabilization point n0 for a rate stat: reliability(n) = n/(n+n0).
    Method of moments -- splits observed cross-sectional spread into true-talent
    variance + binomial sampling variance:
        Var(true) = Var_w(observed) - p(1-p) * mean(1/n)   [approx, PA-weighted]
        n0        = p(1-p) / Var(true)
    Fit ONCE per (metric, training fold) and reused as a fixed hyperparameter --
    never re-estimated inside a test year.
    """
    mask = n >= min_n
    x, w = rate[mask].values, n[mask].values.astype(float)
    p = np.average(x, weights=w)
    var_obs = np.average((x - p) ** 2, weights=w)
    var_true = max(var_obs - p * (1 - p) * len(x) / w.sum(), 1e-8)
    n0 = p * (1 - p) / var_true
    return float(n0), {"p": round(float(p), 4), "sd_true": round(float(np.sqrt(var_true)), 4),
                        "n_used": int(mask.sum())}


def empirical_reliability_curve(pairs: pd.DataFrame, rate1: str, rate2: str, n1: str, n2: str,
                                 edges=(25, 45, 70, 100, 140, 200, 300, 450, 10_000)):
    """Bucket dual-year pitcher pairs by min(PA_year1, PA_year2) and report the raw
    correlation between year1 and year2 rate in each bucket -- the empirical curve
    that estimate_n0()'s fitted n/(n+n0) should be checked against before it's
    trusted for weighting. Mirrors k26_projection.py's reliability_curve(), minus
    the plotting (kept separate so this can run headless in a pipeline)."""
    d = pairs.dropna(subset=[rate1, rate2, n1, n2]).copy()
    d["tmin"] = d[[n1, n2]].min(axis=1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = d[(d["tmin"] >= lo) & (d["tmin"] < hi)]
        if len(s) < 25:
            continue
        rows.append((s["tmin"].median(), np.corrcoef(s[rate1], s[rate2])[0, 1], len(s)))
    return pd.DataFrame(rows, columns=["median_n", "empirical_r", "n_pairs"])


def fit_n0_from_curve(curve: pd.DataFrame, p0: float = 100.0):
    """Fit n/(n+n0) to the empirical (median_n, empirical_r) points directly --
    an independent cross-check against estimate_n0()'s single-season method-of-
    moments value. The two should be in the same ballpark; a big divergence
    signals a bug (e.g. real talent change contaminating the 'noise' estimate)
    rather than a reason to just trust whichever number is more convenient."""
    from scipy.optimize import curve_fit
    if len(curve) < 4:
        return np.nan
    try:
        n0 = curve_fit(lambda t, a: t / (t + a), curve["median_n"].values,
                        curve["empirical_r"].values, p0=[p0], bounds=(1, 5000))[0][0]
        return float(n0)
    except Exception:
        return np.nan


def reliability_weight(n: pd.Series, n0: float) -> pd.Series:
    """reliability(n) = n / (n + n0) -- the row weight used both for fitting
    (WLS) and for evaluation (weighted R2/RMSE/MAE), so a pitcher-season's
    influence on the reported metrics matches how much real signal it actually
    carries."""
    return n / (n + n0)


# --------------------------------------------------------------------------- #
#  Bridge-as-informative-prior: blend a raw per-player count toward BOTH the
#  league mean AND an external "other league" bridge rate in one Marcel-style
#  shrinkage formula, instead of entering the bridge as a free OLS regressor.
#  Motivated by FLHitting's fl_ros_coef_drift_test.py finding that the
#  bridge's OLS coefficient was the LEAST stable regressor across in-season
#  cutoffs (decaying from ~0.6 early-season to ~0.1 late-season) -- exactly
#  the pattern this formula produces automatically from one frozen n0_bridge,
#  without per-cutoff refitting. Validated (fl_ros_bridge_prior_test.py,
#  scratch_verify_bb_prior_exp2.py): a genuine held-out win for BB% (beats
#  naive-only AND the OLS approach in both year directions), a wash for K%
#  at full-season PA (OLS kept for K%). See DECISION_LOG.md "BB% bridge as
#  an informative prior".
# --------------------------------------------------------------------------- #
def blended_rate(raw_count, n, n0_lg: float, lg_rate: float, n0_bridge: float, bridge_rate):
    """(raw_count + n0_lg*lg_rate + n0_bridge*bridge_rate) / (n + n0_lg + n0_bridge).
    n0_bridge=0 recovers the plain league-shrunk naive rate; n=0 (a brand-new
    entrant with no real PA at all) recovers a pure n0_lg/n0_bridge-weighted
    blend of the league mean and the bridge rate -- no separate zero-PA case
    needed, the same formula covers the whole range."""
    return (raw_count + n0_lg * lg_rate + n0_bridge * bridge_rate) / (n + n0_lg + n0_bridge)


def fit_n0_bridge(raw_count, n, n0_lg: float, lg_rate: float, bridge_rate, target, weight, p0: float = 50.0) -> float:
    """Fits the bridge's pseudo-PA (n0_bridge) by nonlinear least squares
    against REAL outcomes. Unlike estimate_n0 (pure cross-sectional method of
    moments -- needs no outcome), this needs a train -> actual-outcome
    transition, because it's answering 'how much is the bridge worth, in
    equivalent PA', which only a real outcome can calibrate. Fit ONCE on a
    training transition and reused frozen, same discipline as estimate_n0."""
    from scipy.optimize import curve_fit
    raw_count, n, bridge_rate, target, weight = (np.asarray(x, dtype=float)
                                                   for x in (raw_count, n, bridge_rate, target, weight))

    def f(_, n0_bridge):
        return blended_rate(raw_count, n, n0_lg, lg_rate, n0_bridge, bridge_rate)

    popt, _ = curve_fit(f, np.zeros(len(n)), target, p0=[p0],
                         sigma=1.0 / np.sqrt(weight + 1e-9), bounds=(0, 2000))
    return float(popt[0])


# --------------------------------------------------------------------------- #
#  Reliability-weighted evaluation metrics (R2, RMSE, MAE) -- used for every
#  backtest fold instead of unweighted metrics, per the requirement that noisy
#  small-PA pitcher-seasons shouldn't count as much as well-established ones.
# --------------------------------------------------------------------------- #
def weighted_r2(y_true, y_pred, w):
    y_true, y_pred, w = map(np.asarray, (y_true, y_pred, w))
    y_bar = np.average(y_true, weights=w)
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - y_bar) ** 2)
    return 1 - ss_res / ss_tot


def weighted_rmse(y_true, y_pred, w):
    y_true, y_pred, w = map(np.asarray, (y_true, y_pred, w))
    return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=w)))


def weighted_mae(y_true, y_pred, w):
    y_true, y_pred, w = map(np.asarray, (y_true, y_pred, w))
    return float(np.average(np.abs(y_true - y_pred), weights=w))


def evaluate_weighted(y_true, y_pred, n, n0, label=""):
    """One-stop reliability-weighted evaluation table row: weight = n/(n+n0)."""
    w = reliability_weight(pd.Series(n), n0).values
    return {
        "label": label,
        "n": len(y_true),
        "R2_w": round(weighted_r2(y_true, y_pred, w), 4),
        "RMSE_w_pp": round(weighted_rmse(y_true, y_pred, w) * 100, 3),
        "MAE_w_pp": round(weighted_mae(y_true, y_pred, w) * 100, 3),
    }


# --------------------------------------------------------------------------- #
#  Classic Marcel 5/4/3 multi-year weighting
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  Empirical-Bayes shrinkage for small-sample within-season delta features
#  (ported verbatim from pitcher_projection_bayesian.ipynb cell 362) -- applied only
#  to the engineered within-season delta features (wh_sva_delta, horiz_dist_delta,
#  vert_dist_delta), never re-applied to the already-Marcel-regressed rate itself
#  (no double-shrinking).
# --------------------------------------------------------------------------- #
def fit_shrinkage_params(full_season_df: pd.DataFrame, valcol: str,
                          group_cols=("pitcher", "game_year"), min_n: int = 50):
    """Empirical-Bayes variance components (within-pitcher, between-pitcher, grand mean)
    for `valcol`, estimated ONCE from a full population of data. Reused as fixed
    hyperparameters when shrinking small-sample values of the same quantity -- do NOT
    re-estimate from the small sample itself."""
    g = full_season_df.groupby(list(group_cols)).agg(
        n=(valcol, "size"), mean_val=(valcol, "mean"), var_val=(valcol, "var")).reset_index()
    g = g[g["n"] >= min_n]
    sigma_within2 = g["var_val"].mean()
    sigma_between2 = max(g["mean_val"].var() - sigma_within2 / g["n"].mean(), 1e-6)
    grand_mean = g["mean_val"].mean()
    return sigma_within2, sigma_between2, grand_mean


def apply_shrinkage(raw_values, n_values, sigma_within2: float, sigma_between2: float, grand_mean: float):
    """Shrink per-pitcher raw values toward the population mean, weighted by each
    pitcher's own sample size. Only has a real effect when n_values varies across rows
    (true here: within-season deltas naturally have varying pitch counts per half)."""
    reliability_i = sigma_between2 / (sigma_between2 + sigma_within2 / n_values)
    return grand_mean + reliability_i * (raw_values - grand_mean), reliability_i


MARCEL_WEIGHTS = {0: 5, 1: 4, 2: 3}  # 0 = most recent year, 1 = year before, 2 = two years before


def residual_correlation(y_true_k, y_pred_k, y_true_bb, y_pred_bb):
    """corr(K residual, BB residual) -- empirically ~ -0.18 to -0.26 year-over-year
    in real MLB data (checked directly on 2022-2025 dK/dBB, consistent with an
    identical finding on the separate FL dataset earlier this session), and
    concurrently ~ -0.11. NOT assumable as zero: a pitcher who gets more
    aggressive tends to see K% and BB% move in opposite directions together, so
    naive independence would understate K-BB% projection uncertainty, not just
    be a simplifying approximation."""
    resid_k = np.asarray(y_true_k) - np.asarray(y_pred_k)
    resid_bb = np.asarray(y_true_bb) - np.asarray(y_pred_bb)
    return float(np.corrcoef(resid_k, resid_bb)[0, 1])


def propagate_kbb_uncertainty(se_k, se_bb, rho):
    """Combine independent per-pitcher K% and BB% projection standard errors into
    a K-BB% standard error, accounting for the real (typically negative)
    correlation between the two components' residuals rather than assuming
    independence:
        Var(K-BB) = Var(K) + Var(BB) - 2*rho*SE(K)*SE(BB)
    A negative rho (the empirically observed case) INCREASES the combined
    variance above the naive-independence sum -- report this plainly rather than
    defaulting rho=0 for convenience."""
    se_k, se_bb = np.asarray(se_k), np.asarray(se_bb)
    var_kbb = se_k**2 + se_bb**2 - 2 * rho * se_k * se_bb
    return np.sqrt(np.maximum(var_kbb, 0.0))


def marcel_weighted_rate(events_by_year: dict[int, tuple[float, float]], reg: float, league_rate_fn):
    """events_by_year: {year: (successes_or_rate*PA, PA)} for up to 3 trailing years
    (missing years simply absent, not zero-filled). league_rate_fn(year) -> that
    year's league-wide rate, used for the regression term with the SAME 5/4/3
    recency weights as the pitcher's own numerator/denominator.
    Returns the Marcel-projected rate for the upcoming season."""
    num, den = 0.0, 0.0
    for lag, (succ, pa) in events_by_year.items():
        wt = MARCEL_WEIGHTS.get(lag)
        if wt is None or pa <= 0:
            continue
        num += wt * succ
        den += wt * pa
    if den == 0:
        return np.nan
    # regression term uses the same recency-weighted league rate, scaled by `reg`
    lg_num, lg_den = 0.0, 0.0
    for lag in events_by_year:
        wt = MARCEL_WEIGHTS.get(lag)
        if wt is None:
            continue
        lg_rate = league_rate_fn(lag)
        if lg_rate is None:
            continue
        lg_num += wt * lg_rate
        lg_den += wt
    lg_rate_blend = lg_num / lg_den if lg_den > 0 else np.nan
    return (num + reg * lg_rate_blend) / (den + reg)


def marcel_reliability_prior(season_rates: dict[int, tuple[float, float]], target_year: int,
                              n0: float, reg: float, lg_rate: float) -> float:
    """Marcel recency weights (5/4/3 by lag), with each season's contribution ALSO
    scaled by that season's own reliability (n/(n+n0)) -- a reliability-weighted
    AVERAGE OF SEASON RATES, not classic marcel_weighted_rate's weighted POOLING of
    raw counts. Deliberately different: pooling raw counts already implicitly
    weights by PA, so multiplying by an explicit reliability term on top would
    double-count sample size. Weighting already-computed per-season rates by
    recency x reliability keeps the two effects cleanly separable -- recency
    controls how much an old season matters regardless of its size, reliability
    controls how much a small season matters regardless of its age.

    season_rates: {year: (rate, PA)} for up to 3 trailing years (missing years
    simply absent, not zero-filled). n0/reg fit ONCE via estimate_n0 on the pooled
    cross-section, not re-tuned here -- same convention as every other reliability
    constant in this project."""
    num, den = 0.0, 0.0
    for year, (rate, pa) in season_rates.items():
        lag = target_year - year
        recency_w = MARCEL_WEIGHTS.get(lag)
        if recency_w is None or pa <= 0 or np.isnan(rate):
            continue
        reliab = pa / (pa + n0)
        w = recency_w * reliab
        num += w * rate
        den += w
    if den == 0:
        return np.nan
    return (num + reg * lg_rate) / (den + reg)
