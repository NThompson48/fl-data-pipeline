"""
fl_ra9_trustmark_projection.py -- RA9 projection (target) + TrustmarkPark-specific
adjustment (applied on top), built from this session's validated findings. See
DECISION_LOG.md for the full trail; summary of what's baked in here:

RA9 BASELINE MODEL
  Walk-forward validated (2022->23, 23->24, 24->25, 25->26, each trained only
  on strictly-earlier transitions) down to the simplest spec that survives:
  RA9 ~ K% + BB% alone. naive_ra9 (the pitcher's own prior-year RA9 level) and
  HR% were BOTH tested and dropped -- naive_ra9's coefficient is never
  significant once K%/BB% are in the model (p=0.64-0.88 across every spec
  tried), and HR%'s apparent value turned out to be an artifact that reverses
  under a MAE-based R2 and disappears once pooled across eras (p=0.187).
  This module fits that 2-predictor spec on the one real Trackman-era
  transition available (2025 actual -> 2026 actual, via war_pitchers) and
  freezes it, same "fit on the training transition, freeze, roll forward"
  convention as fl_production_projection.py's K%/BB% model.

AIR% PROJECTION (for the park adjustment)
  Raw year-over-year Air% stickiness is unreliable -- LOOCV R2_w=-0.45 for
  "just use last year's Air%" (reliability-weighted; looked OK, 0.40 raw
  correlation, until weighted by target reliability). A pitch-characteristic
  trait model (IVB, fraction of pitches thrown in the upper third of the
  zone or above, effective velocity, four-seam usage rate) fit on 2025
  traits -> real 2026 Air%, LOOCV R2_w=0.23 -- genuinely predictive, and
  critically available even for a pitcher with NO batted-ball track record
  at all (it only needs pitch-tracking, not outcomes). This is the Air%
  source used here, not naive shrinkage of the pitcher's own Air% history.

TRUSTMARKPARK ADJUSTMENT
  100% of home runs in this league come from air-ball contact (angle>=10deg;
  zero from ground balls) -- confirmed directly, not assumed -- which is why
  this decomposes as HR% = Contact% x Air% x HR_per_air_ball, with the park
  effect applied to the HR_per_air_ball term specifically; a pitcher's Air%
  itself is assumed park-invariant (standard sabermetric convention: a
  batted-ball profile is a pitcher trait, what a park does to that contact
  is the park's effect, not the pitcher's).
  Two stacked, independently-measured park effects, both isolated to
  air-ball contact specifically (not all contact, which would dilute/confound
  them with ground-ball behavior):
    1. HR-per-air-ball suppression: TrustmarkPark 0.0260 vs league 0.0694 (raw ratio
       0.375; 0.0268 vs 0.0694 in the earlier two-season-less pull). The factor used is the
       empirical-Bayes estimate 0.446 over all 18 parks, NOT the official park_factors.pf_hr
       = 0.624 the first version used: the official table (one season of data for Trustmark)
       is a compressed version of the raw ratios, and the raw ratio holds in same-team
       home/road splits, so the shrinkage that matters is the park-to-park one (true SD of
       the log factor 0.30) -- see TRUSTMARK_HR_FACTOR and DECISION_LOG.md "Trustmark HR factor".
    2. Hit-rate-on-non-HR-air-ball suppression: initially measured via the
       overall wobacon/xwobacon ratio (0.859) and found to be LARGELY A
       CONFOUND with the HR effect (wobacon includes HRs, so Trustmark's
       huge HR suppression was bleeding into that number). Recomputed in
       isolation (HR-excluded hit rate on air balls specifically) across all
       18 FL parks, with a real cross-park EB shrinkage (n0=835, estimated
       via mm.estimate_n0 on the park-level panel): raw delta 0.0099 ->
       shrunk delta 0.0073 -- a real but SMALL effect, not the dramatic one
       the confounded number implied. TrustmarkPark is not unusual among FL
       parks on this dimension.
  Both effects are blended for a pitcher assumed to play `pct_home_games` of
  their games at TrustmarkPark and the rest at a neutral-average park
  (factor=1.0): blended = pct_home*park_factor + (1-pct_home)*1.0.
  Converted to runs via this project's own empirical linear weights
  (`woba_weights`, not generic constants): a non-HR air-ball HIT is worth
  rv_above_out=1.041 runs more than an out, using the REAL FL hit-type mix
  on air balls specifically (65.5% 1B / 31.5% 2B / 3.0% 3B, not the generic
  all-contact mix). HR's own run value above average (1.4, the standard
  "a home run costs about 1.4 runs more than an average out" heuristic,
  consistent with woba_weights' own HR rv_above_avg=1.464) converts the HR
  side.

Usage:
    python fl_ra9_trustmark_projection.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
from fl_supabase import read_sql_df
import mlb_marcel as mm

OUT_PATH = "output/fl_ra9_trustmark_projection.parquet"

# --- RA9 baseline model constants ---
# `ppf` (real, player-season-specific park factor, already in war_pitchers) and
# `ip_per_g_prior` (PRIOR-season innings-per-appearance -- a usage/trust signal)
# were added after an error-analysis pass on the plain K%+BB% spec's held-out
# residuals (fl_error_analysis_v2.py): team-level RA9 residuals showed a ~1.7
# run spread with zero team/park info in the model, and a regression of the
# residual on candidate explanations found prior-season IP/appearance the
# single strongest, most robust predictor (p=0.0015, survives everything else
# in the model), with `ppf` a smaller but real, independently significant
# contributor (p=0.035 alone). Deliberately uses the PRIOR season's IP/G, not
# the target season's own (which was also tested and dropped -- R2_w=0.265 but
# not a legitimate predictor, since a pitcher's OWN target-season usage is
# partly an outcome of that season, not knowable in advance). Validated
# genuinely held out BOTH directions (2025->2026 forward AND the non-causal
# 2026->2025 reversal, same robustness-check convention as the BB% bridge-
# prior work): R2_w 0.105->0.140 forward, 0.002->0.055 reversed -- same
# feature ordering and monotonic improvement in both directions, confirming
# this isn't a fluke of which year happened to be the training transition.
# See DECISION_LOG.md "RA9: park factor + prior-season usage as regressors".
#
# `pitches_per_pa_prior` (PRIOR-season pitches thrown per plate appearance --
# a pitch-economy/efficiency signal) added on top: NOT redundant with
# ip_per_g_prior despite both being "usage" measures -- the two are only
# modestly correlated (r=-0.29: starters are slightly more economical, but
# it's a different axis), and ip_per_g_prior's distribution is bimodal
# (SP-shaped vs RP-shaped clusters) while pitches_per_pa_prior's is unimodal/
# bell-shaped, a genuinely different kind of signal, not a restatement.
# Tested both as a straight swap-in for ip_per_g_prior (slightly WORSE alone,
# R2_w=0.135 vs 0.140) and added on top (real additional lift, R2_w=0.159) --
# it complements rather than duplicates. Validated genuinely held out BOTH
# directions, same discipline as ppf/ip_per_g_prior: R2_w 0.140->0.159
# forward (2025->2026), 0.055->0.088 reversed (2026->2025) -- confirmed
# before shipping, not assumed from the single-direction result.
#
# `ip_per_g_prior` is now an EB-SHRUNK rate (Marcel-style blend toward the
# league-wide games-weighted mean), not the raw ratio -- user-prompted follow
# up after the clipping fix above: a method-of-moments decomposition (same
# convention as mm.estimate_n0, swapping the binomial p(1-p)/n sampling-noise
# term for the real one) gave a NEGATIVE implied n0 for ip_per_g_prior (the
# cross-sectional variance doesn't shrink with g the simple model assumes --
# plausible, since ip_per_g is a bimodal, role-driven variable, not noise
# around one central tendency). Fell back to empirically grid-searching n0
# via the same held-out LOOCV discipline (same precedent as the team-random-
# effect credibility constant tested earlier): clean, consistently-peaked
# optimum at n0=12 GAMES, R2_w 0.159->0.164 forward, 0.088->0.089 reversed
# (a tie, not a regression -- clears the bar in both directions).
#
# `pitches_per_pa_prior` -> `ppa_resid`: user flagged, correctly, that the
# usage/efficiency regressors might be PROXYING for skill rather than adding
# independent signal, and asked for them to be too influential relative to
# K%/BB%. Checked directly: `ip_per_g_prior` is fine (only 21% of its own
# variance explained by K_pred+BB_pred+ppf, VIF=1.27, and its PARTIAL
# relationship with RA9 is stronger than its raw bivariate one -- genuine
# independent signal). `pitches_per_pa_prior` was the real problem: 57.5% of
# its variance is explained by K_pred+BB_pred+ppf (VIF=2.35), and that
# collinearity was cannibalizing K_pred's own coefficient (K_pred collapsed
# from -8.74 (p=0.009, no efficiency term) to -4.17 (p=0.317, raw efficiency
# in the model) -- confirmed by the per-regressor contribution-spread on the
# real roster: K_pred's spread (0.132) was SMALLER than either usage
# variable's, backward for what should be a skill-driven model).
#
# Fix: residualize `pitches_per_pa_prior` against `K_pred + BB_pred + ppf`
# (fit on the training transition, frozen) and use the RESIDUAL
# (`ppa_resid`) as the regressor instead of the raw ratio. By construction
# this is exactly orthogonal to K_pred/BB_pred/ppf (corr=0.0000 in training),
# so it cannot steal their explanatory credit. Also tested whether a K%xBB%
# interaction explains pitches_per_pa_prior better than K%+BB% alone first --
# it doesn't (R2 0.569->0.570, interaction p=0.76, purely additive).
# Verified this is a free fix, not a tradeoff: LOOCV R2_w/RMSE_w are IDENTICAL
# between raw and residualized (0.1638/1.5552 forward, 0.0885/1.4258
# reversed, both directions byte-for-byte) -- expected by the Frisch-Waugh-
# Lovell theorem, since ppa_resid is just pitches_per_pa_prior minus a linear
# combination of regressors ALREADY in the model, so the model's SPAN (and
# therefore its predictions) is unchanged; only the coefficient attribution
# is. With the residual: K_pred=-9.04 (p=0.006, MORE significant than the
# original no-efficiency model), ppa_resid itself keeps essentially the same
# coefficient the raw version had (-1.945 vs -1.894, p=0.071 vs 0.079) --
# the real efficiency signal is preserved, just no longer mis-attributed.
# See DECISION_LOG.md "RA9: pitches_per_pa_prior residualized against K%/BB%/ppf".
LG_IPG = 2.2887  # games-weighted league mean (sum(ip)/sum(g)), pooled both years, g>=1
N0_IPG = 12.0    # empirically chosen (LOOCV-optimal), games
PPA_RESID_SPEC = "pitches_per_pa_prior ~ K_pred + BB_pred + ppf"  # the auxiliary
# regression used to build ppa_resid -- fit on the training transition, frozen,
# then applied identically to the full roster (never refit per-pitcher).
RA9_SPEC = "target ~ K_pred + BB_pred + ppf + ip_per_g_prior + ppa_resid"
# ^ LEGACY. `ppa_resid` (and PPA_RESID_SPEC / CONTINUOUS_REGRESSORS below) was
# REMOVED from production on 2026-10-07: it failed the user's own criterion --
# no held-out lift when added to the K%/BB% models (Exp2/Exp1 both worse), no
# consistent contact-quality link, unstable components (YoY 0.10-0.18 for the
# 84% that is fouls+balls), counter-intuitive sign, and an RA9 gain whose 90%
# CI includes zero (+0.042 forward RMSE_w, [-0.015, +0.100]). The legacy names
# are kept ONLY so the many experiment scripts that reproduce earlier results
# (fl_ra9_*_test.py etc.) keep importing; production uses RA9_SPEC_PROD.
# See DECISION_LOG.md "ppa_resid routed through K%/BB% and contact quality".
PITCHER_NAME_ALIASES = {"Gilleran, James": "Gilleran, Jimmy"}
RA9_SPEC_PROD = "target ~ K_pred + BB_pred + ip_per_g_prior"
RA9_CONTINUOUS = ["ip_per_g_prior"]
# PARK-NEUTRAL by construction (2026-10-07): the park term (`ppf`) was dropped from the
# baseline. Out-of-sample (LOOCV, n=79) it adds nothing -- forward R2_w 0.151 without vs
# 0.139 with `ppf`, reversed 0.045 vs 0.052 -- and neither does a FIP-consistent
# game-by-game park exposure (own game-mix: 0.121 / 0.031; fl_park_term_test.py). `ppf` is
# also internally inconsistent with the FIP park factors the Trustmark adjustment is built
# from: team-level corr 0.61, and Mississippi pitchers average ppf ~101 although half their
# games are at a park whose FIP factor is 93 (war_pitchers.pf, by contrast, tracks the FIP
# factor at corr 0.99). K%/BB% are not park-adjusted (tested: no gain), so the baseline is
# the pitcher's RA9 in an average park; the Mississippi projection adds the explicit,
# Air%-driven TrustmarkPark shift ONCE on top, for every pitcher alike -- including
# Mississippi's own. See DECISION_LOG.md "Park-neutral RA9 + Mississippi translation".
RA9_N0_EVAL = 146  # IP-based reliability n0, fit on the real 2025->2026 transition residuals
# (unchanged by this spec update -- only the regressor set was validated/changed, not the weight)
RA9_RESID_SD = 1.5507  # LOOCV (held-out, not in-sample) RMSE_w for RA9_SPEC -- the honest
# uncertainty of a single point projection, used below for P(true RA9 < replacement).
# NOTE: applied as-is to the TrustmarkPark-adjusted scenarios too, which understates
# their true uncertainty slightly (the Air% trait model has its own real prediction
# error, LOOCV R2_w~=0.23, not propagated here) -- acceptable since the TrustmarkPark
# shift itself is narrow (~0.3-0.4 runs/9) relative to this SD, but a genuine
# simplification, not a free pass.

# Information-dependent uncertainty for P(WAR>0) (replaces the single fixed SD
# above as the thing the probability uses; RA9_RESID_SD is kept only as the
# reference value of the old behaviour). fl_ra9_uncertainty_model.py fit, on
# genuinely out-of-sample 2025-inputs -> 2026-RA9 forecast errors for the BROAD
# population (n=312, incl. 2026 debuts with no FL history, not just the BF>=100
# fit sample), Var(error) = a + b/IP + c(1 - r_prior). The MLE put essentially
# ALL the variance in the single-season luck term b/IP (a, c -> 0; their 90%
# bootstrap upper bounds are 0.22 and 0.68, i.e. talent-level uncertainty is
# small and NOT separately identifiable at this n), so the validated model is
# Var = RA9_LUCK_VAR_COEF / IP. Standardized-error SD 0.98 overall (vs 1.65 for
# the fixed 1.55 SD, whose nominal-90% interval covered only 76%); coverage by
# prior-information bucket 80-93% at nominal 90%. Interpretation: P(WAR>0) is
# the probability the pitcher's actual RA9 over his expected 2027 innings beats
# replacement -- a thin-workload pitcher is genuinely less certain because
# fewer innings = more luck, which is exactly what WAR>0 measures.
RA9_LUCK_VAR_COEF = 185.16   # b, in (runs per 9)^2 * innings
UNCERTAINTY_MIN_IP = 5.0     # floor so a 1-IP pitcher doesn't get an infinite SD
DEBUT_MEDIAN_IP = 18.0       # median first-season IP of pitchers new to the league (2023-26, n=1,107;
# StatsCrew + war_pitchers) -- workload assumption for a pitcher with no FL innings at all
# --- 2027 workload (innings) projection, fl_innings_projection.py ---
# Next-season IP (given the pitcher pitches again) regresses hard toward the mean: across FOUR
# year-to-year transitions (2022-23 ... 2025-26, 543 pitcher-transitions) the OLS slope on prior
# IP is 0.64/0.63/0.67/0.59 (stable), pooled IP_next = 19.3 + 0.628 * IP_prior (resid SD 29.5,
# R2 0.33). Leave-one-transition-out beats "next = prior" (RMSE 29.7 vs 32.0, R2 0.32 vs 0.21).
# A second prior season adds nothing out-of-sample (RMSE 28.1 vs 28.1); SP share helps a little
# (CV RMSE 30.8 -> 29.7) but exists for ONE transition only, so it is not shipped.
IP_PROJ_A, IP_PROJ_B = 19.31, 0.6278
GAP_RETURNER_IP = 24.0       # median IP of pitchers who return after skipping a season (n=22)
GAP_RETURN_PROB = 0.03       # share of 'skipped a season' pitchers who ever return (24/758)
# Retention: P(pitches again | prior-season IP) = sigmoid(-2.286 + 0.541 * ln(IP)), n=1,577;
# overall 32-36% every year. Projected IP is CONDITIONAL on pitching, so it is shown with this.
RETURN_LOGIT_A, RETURN_LOGIT_B = -2.286, 0.541
# The luck law was re-validated on those four transitions (K%/BB%-only RA9 model, leave-one-
# transition-out): b = 135 / 174 / 179 / 210 (pooled 184.7, matching the 185.16 above), a ~ 0;
# standardized-error SD 0.95-0.99 per transition. With the PROJECTED (not actual) IP in the
# denominator, b = 185 still gives nominal 50/80/90% coverage of 0.50/0.79/0.90.
# Air% trait model held-out error (LOOCV, fl_ra9_uncertainty_model.air_error_model):
# talent-level SD 0.0443 once the binomial sampling noise of the 2026 Air% target is
# removed; population SD 0.0723 is what a pitcher with NO tracked pitches gets.
AIR_TALENT_ERR_SD = 0.0410  # was 0.0443 for traits-only; 0.0410 with own-history Air% (LOOCV, same sampling-noise removal)
AIR_POP_SD = 0.0723

# --- "probability of beating replacement level" constants ---
# Replacement level is a REAL, already-computed WAR concept in this project
# (war_pitchers.rar = runs above replacement), not invented here -- but it must
# be LEAGUE-relative, not team-relative. Backing it out per-pitcher
# (ra9_replacement = ra9 + rar*9/ip) was tried and rejected: regressing real
# repl_runs on playing time + role (the only things a true replacement baseline
# should depend on) gets R2=0.91, not 1.0 -- rar carries real team/park-specific
# noise (presumably each team's own opponent/park context), which has no business
# being "replacement level" baked into a single-pitcher threshold.
# Fix: average within role across the WHOLE league, which cancels that
# team-specific noise out instead of importing it. Checked directly: SP and RP
# converge to nearly the SAME value (6.962 vs 6.965, 2026) -- the role
# adjustment already baked into repl_runs apparently offsets the raw role-driven
# RA9 gap, so one single league-wide number works and a role split isn't needed.
LEAGUE_RA9_REPLACEMENT = 6.963  # IP-weighted mean across ALL 2026 pitchers (ip>=20),
# used identically for every pitcher regardless of team/role -- this IS the fix,
# not a fallback for missing data.
LEAGUE_K_RATE = 0.1940   # 2026 league rates -- the generic/average-pitcher profile
LEAGUE_BB_RATE = 0.1142
LEAGUE_AVG_AIR_PCT = 0.5347  # used to shift the REPLACEMENT baseline itself under
# TrustmarkPark (see build_replacement_trustmark_shift): a replacement-level
# pitcher benefits from TrustmarkPark's HR/defense suppression too, by the SAME
# mechanism as any other pitcher -- comparing an adjusted pitcher against an
# UNADJUSTED replacement baseline would overstate how much TrustmarkPark helps
# them RELATIVE to replacement, since part of that apparent gain is really "any
# pitcher, including a replacement one, does better at Trustmark." No data
# exists on what a replacement-level pitcher's own Air%/contact profile looks
# like (vs. an average rostered one), so league averages are the defensible
# neutral assumption, not an attempt at precision.

# --- Air% trait model constants ---
AIR_SPEC = "air_pct ~ ivb + frac_high + effective_velo + fourseam_rate"
# A pitcher with zero 2026 pitches gets the trait model's prediction evaluated AT
# the league-mean feature values (computed naturally below, not a separate
# constant) -- acceptable here specifically since the TrustmarkPark effect's
# sensitivity to Air% is narrow (~0.33-0.41 runs/9 across the full real range),
# much smaller than K%/BB%'s direct effect on the RA9 baseline itself.
#
# Every pitcher WITH at least one 2026 pitch instead gets each INPUT FEATURE
# (ivb, frac_high, effective_velo, fourseam_rate) shrunk toward its own league mean
# with reliability n/(n+n0), n0 estimated per-feature via real method-of-moments
# (pooled within-pitcher variance vs. real cross-pitcher variance, same formula as
# N0_NAIVE below) on the full 2026 pitcher population (n=477-482) -- not a single
# hard pitch-count gate, which created an arbitrary discontinuity (29 pitches ->
# pure league average, 31 -> the full, unshrunk model) where split-half reliability
# testing showed a smooth, fast-stabilizing curve instead. The four features
# stabilize at very different rates -- ivb/effective_velo/fourseam_rate are nearly
# fully reliable within single-digit pitches (n0=6.6/2.4/3.7: these are mechanical
# pitch properties, not outcomes), frac_high is the slow one (n0=46.3, matching its
# weaker split-half correlation at low N) -- so a single shared cutoff was always
# going to be wrong for at least some of the four regardless of where it was set.
AIR_FEATURE_N0 = {"ivb": 6.6, "effective_velo": 2.4, "fourseam_rate": 3.7, "frac_high": 46.3}
AIR_FEATURE_LEAGUE_MEAN = {"ivb": 8.9804, "effective_velo": 84.1721,
                            "fourseam_rate": 0.3263, "frac_high": 0.3460}
LEAGUE_BF_PER_IP = 4.512  # real 2026 league rate -- used as a placeholder workload
# ratio for pitchers with no real 2026 IP/BF (bridge-only prospects). The RA9 SHIFT
# is a rate and is invariant to the absolute IP/BF magnitude assumed, only to this
# ratio, so an assumed 100 IP at the league BF/IP ratio gives the same shift as any
# other IP would.

# --- TrustmarkPark adjustment constants (see module docstring for derivation) ---
HR_PER_AIR_LEAGUE = 0.0668
# HR-per-air-ball ratio, TrustmarkPark vs the league: 0.446 = empirical-Bayes estimate over all 18 parks
# (fl_trustmark_hr_factor_check.py). Replaced the OFFICIAL park_factors.pf_hr = 0.624 on 2026-10-07: the raw ratio is
# 0.375-0.39 on 4,766 Trustmark batted balls (66 HR vs ~169 expected; 0.40 in 2025, 0.34 in 2026) and holds in
# same-team home/road splits (Mississippi pitchers 0.39, batters 0.36, visitors 0.31-0.46; no interval reaches 0.624);
# the between-park SD of the true log factor is 0.30, so a raw ratio keeps ~86% of its signal -> 0.446. The official
# table lists Trustmark with n_seasons = 1 and is a compressed version of the raw ratios (slope 0.53 across parks).
# See DECISION_LOG.md "Trustmark HR factor". Previous value 0.624 (half-season shift -0.31, full -0.62).
TRUSTMARK_HR_FACTOR = 0.446
TRUSTMARK_HIT_RATE_DELTA = 0.0073  # shrunk, HR-isolated, cross-park EB estimate (full-season effect)
AIR_HIT_RUN_VALUE = 1.041          # real woba_weights rv_above_out, real FL air-ball hit-type mix
RUNS_PER_HR = 1.4                  # woba_weights HR rv_above_avg ~= 1.464, standard rounding

# TrustmarkPark's own standalone, FIP-based run park factor (park_factors_fip,
# 2025-2026 combined row, 100-centered -- same scale as war_pitchers.ppf).
# Used to build the CORRECT `ppf` input for the MIS/Trustmark scenario
# predictions (see "RA9: MIS-scenario ppf was using the pitcher's real
# current team" in DECISION_LOG.md) -- the baseline RA9 model's `ppf` term
# must reflect the SCENARIO being projected (current team, vs. half/all games
# at TrustmarkPark), not always the pitcher's real 2026 team, or the
# TrustmarkPark shift (which itself assumes a neutral-100 starting point)
# gets added on top of the wrong park context.
TRUSTMARK_PPF = 93.0
LEAGUE_NEUTRAL_PPF = 100.0  # park_factors_fip is centered at 100 by construction


CORRUPT_GAME_MIN_PITCHES = 50  # see pitches_per_pa docstring


def pitches_per_pa(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Per-pitcher (by NAME, matching war_pitchers.player) pitches thrown per
    plate appearance over the given window -- a pitch-economy/efficiency
    signal, distinct from ip_per_g_prior's role/trust signal (see RA9_SPEC
    docstring above).

    Drops corrupted pitcher-game records first: user noticed Derek Salata's
    extreme ppa_resid (the single highest of 797 roster pitchers) and asked
    to investigate -- traced to one game (20260725-WintrustField-1) with 450
    pitches all logged under ONE pa_id and ZERO is_terminal flags, i.e. a
    tracking/export artifact, not a real appearance (dumps pitches into the
    numerator while contributing nothing to the PA denominator, inflating
    the ratio arbitrarily). Excluding it alone dropped Salata's real ratio
    from 4.75 (the league's most extreme value) to 3.82 (slightly BELOW the
    3.91 league average) -- he was never actually an outlier. Checked the
    full 2025+2026 pool directly: exactly 2 pitcher-games total show this
    signature (>=50 pitches, 0 terminal pitches) -- rare, not systemic, but
    real, and this is the only place in the current pipeline that sums raw
    pitch counts this way (ip_per_g_prior/K_pred/BB_pred come from
    war_pitchers' own external aggregates, not vulnerable to this specific
    bug). See DECISION_LOG.md "pitches_per_pa: drop corrupted pitcher-game
    records"."""
    p = fp.filter_period(raw, start, end)
    per_game = p.groupby(["pitcher", "game_id"]).agg(
        n_pitches=("pitch_type", "size"), n_terminal=("is_terminal", "sum")).reset_index()
    corrupt_games = per_game[(per_game["n_pitches"] >= CORRUPT_GAME_MIN_PITCHES) & (per_game["n_terminal"] == 0)]
    if len(corrupt_games):
        print(f"  [pitches_per_pa] dropping {len(corrupt_games)} corrupted pitcher-game record(s): "
              f"{corrupt_games[['pitcher', 'game_id', 'n_pitches']].to_dict('records')}")
        bad_keys = set(zip(corrupt_games["pitcher"], corrupt_games["game_id"]))
        is_bad = list(zip(p["pitcher"], p["game_id"]))
        p = p[[k not in bad_keys for k in is_bad]]

    g = p.groupby("pitcher").agg(n_pitches=("pitch_type", "size"), n_pa=("is_terminal", "sum")).reset_index()
    g = g[g["n_pa"] > 0].copy()
    g["pitches_per_pa"] = g["n_pitches"] / g["n_pa"]
    return g[["pitcher", "pitches_per_pa"]]


CONTINUOUS_REGRESSORS = ["ppf", "ip_per_g_prior", "ppa_resid"]


def clip_to_training_range(df: pd.DataFrame, train_ranges: dict) -> pd.DataFrame:
    """Clip each continuous regressor to the [min, max] actually observed in
    the n=96 training population before prediction. Caught directly (user-
    flagged): without this, a handful of roster pitchers whose real value
    falls outside what the model was ever fit on get the coefficient
    extrapolated well past its validated support -- confirmed as the cause
    of one pitcher's RA9 projection swinging -1.6 runs from
    `pitches_per_pa_prior` alone (value 4.75 vs. a training max of 4.51; only
    6/797 roster pitchers are affected for that regressor, 16/797 for
    ip_per_g_prior, 3/797 for ppf, but the effect is large exactly where it's
    most visible -- the top of the leaderboard). LOOCV validation never
    caught this because every held-out point is drawn from that SAME narrow
    training range by construction; it can't test behavior outside it. See
    DECISION_LOG.md "RA9: continuous regressors clipped to training range"."""
    out = df.copy()
    for col, (lo, hi) in train_ranges.items():
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def fit_ra9_baseline_model(war: pd.DataFrame, raw: pd.DataFrame = None):
    """Fit RA9 ~ K% + BB% + ip_per_g_prior on the one real Trackman-era
    transition (2025 actual -> 2026 actual), BF>=100 both years -- a PARK-NEUTRAL
    baseline (see RA9_SPEC_PROD's note). `ip_per_g_prior` is the PRIOR season's
    EB-shrunk innings per appearance (knowable in advance). Frozen and reused, same convention as every other production
    model in this project. (A pitches-per-PA residual, `ppa_resid`, was the
    fifth regressor until 2026-10-07 -- removed, see RA9_SPEC's note.)
    Returns (model, train_ranges) -- `train_ranges` clips the full roster's
    inputs to the model's validated support before prediction (see
    clip_to_training_range). `raw` is unused and kept only for call
    compatibility."""
    w25 = war[war["year"] == 2025].copy()
    w26 = war[war["year"] == 2026].copy()
    m = w25.merge(w26, on="player", suffixes=("_2025", "_2026"))
    m = m[(m["bf_2025"] >= 100) & (m["bf_2026"] >= 100)].copy()
    m["K_pred"] = m["so_2025"] / m["bf_2025"]
    m["ip_per_g_prior"] = (m["ip_2025"] + N0_IPG * LG_IPG) / (m["g_2025"] + N0_IPG)
    m["BB_pred"] = m["bb_2025"] / m["bf_2025"]
    m["target"] = m["ra9_2026"]

    w = mm.reliability_weight(m["ip_2026"], RA9_N0_EVAL).values
    model = smf.wls(RA9_SPEC_PROD, data=m, weights=w).fit()
    print(f"RA9 baseline model (n={len(m)}): {model.params.round(3).to_dict()}  "
          f"p={model.pvalues.round(3).to_dict()}")
    train_ranges = {col: (float(m[col].min()), float(m[col].max())) for col in RA9_CONTINUOUS}
    print(f"Training ranges (model's validated support): {train_ranges}")
    return model, train_ranges


def _raw_pitch_features(period_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher RAW (unshrunk) mean ivb/frac_high/effective_velo/fourseam_rate
    + pitch count, over whatever window `period_df` already represents."""
    d = period_df.copy()
    d["z_norm"] = (d["plate_z"] - d["sz_bot"]) / (d["sz_top"] - d["sz_bot"])
    d["is_fourseam"] = d["pitch_type"] == "four_seam"
    return d.groupby("pitcher_id").agg(
        n_pitches=("ivb", "size"), ivb=("ivb", "mean"),
        frac_high=("z_norm", lambda s: (s >= 0.67).mean()),
        fourseam_rate=("is_fourseam", "mean"), effective_velo=("effective_velo", "mean"),
    ).reset_index()


def _shrink_air_features(feat: pd.DataFrame) -> pd.DataFrame:
    """EB-shrink each raw feature toward its own league mean with reliability
    n/(n+n0), n0 per AIR_FEATURE_N0 (real method-of-moments estimates -- see
    module constants for the derivation and why one shared pitch-count cutoff
    can't be right for all four features at once). Pitchers with 0 pitches get
    exactly the league mean for every feature (n/(n+n0)=0), which is the
    correct limiting behavior, not a special case."""
    out = feat.copy()
    n = out["n_pitches"].fillna(0)
    for col, n0 in AIR_FEATURE_N0.items():
        raw_val = out[col].fillna(AIR_FEATURE_LEAGUE_MEAN[col])
        w = n / (n + n0)
        out[col] = w * raw_val + (1 - w) * AIR_FEATURE_LEAGUE_MEAN[col]
    return out


AIR_OWN_N0 = 150.0   # BIP pseudo-count shrinking a pitcher's OWN Air% toward the league mean
AIR_SPEC_PROD = AIR_SPEC + " + own_air"
# Own-history Air% was added to the trait model on 2026-10-07 (fl_air_pct_projection_test.py):
# LOOCV R2_w traits-only 0.198 -> 0.234 forward (2025 -> 2026) and 0.225 -> 0.249 reversed, with
# a t of 2.6 / 2.2 on the history term (n0 = 150 BIP; 40..300 all help in both directions). The
# Mississippi translation is proportional to Air%, so this is the main lever on how well it
# separates pitchers.


def _own_air_table() -> pd.DataFrame:
    b = pd.read_csv("output/batted_ball_event_residuals.csv").dropna(subset=["pitcher_id"])
    b["pitcher_id"] = b["pitcher_id"].astype("int64").astype(str)
    b["air"] = (b["angle"] >= 10).astype(float)
    return b.groupby(["pitcher_id", "year"]).agg(bip=("air", "size"), air=("air", "mean")).reset_index()


def _own_air_feature(ids: pd.Series, year: int, tbl: pd.DataFrame) -> pd.DataFrame:
    """Per pitcher: that year's Air%, EB-shrunk toward the league mean by BIP (0 BIP ->
    exactly the league mean, the same limiting behaviour as every other shrunk input)."""
    t = tbl[tbl["year"] == year]
    lg = float(np.average(t["air"], weights=t["bip"]))
    out = pd.DataFrame({"pitcher_id": pd.Series(ids).astype(str).unique()}).merge(t[["pitcher_id", "bip", "air"]], on="pitcher_id", how="left")
    out["bip"] = out["bip"].fillna(0.0); out["air"] = out["air"].fillna(0.0)
    out["own_air"] = (out["bip"] * out["air"] + AIR_OWN_N0 * lg) / (out["bip"] + AIR_OWN_N0)
    return out[["pitcher_id", "own_air"]]


def fit_air_trait_model(raw: pd.DataFrame):
    """Fit the Air% trait model on 2025 pitch characteristics (shrunk toward
    league mean per pitcher, same as at projection time) -> real 2026 Air%
    (angle>=10deg rate, needs >=30 BIP to count as a real target -- a
    separate, OUTCOME-side reliability question from the predictor-side
    shrinkage above). LOOCV R2_w=0.23 validated on the unshrunk version;
    shrinking the predictors too only pulls noisy low-pitch-count pitchers
    toward the mean, it doesn't change what the coefficients mean. Frozen and
    rolled forward in project_air_pct()."""
    r25 = fp.filter_period(raw, bt.START_2025, bt.END_2025)
    feat = _shrink_air_features(_raw_pitch_features(r25))

    bb = pd.read_csv("output/batted_ball_event_residuals.csv").dropna(subset=["pitcher_id"])
    bb["pitcher_id"] = bb["pitcher_id"].astype("int64").astype(str)
    bb["angle_air"] = bb["angle"] >= 10
    a26 = bb[bb["year"] == 2026].groupby("pitcher_id").agg(
        air_pct=("angle_air", "mean"), n_bip=("angle_air", "size")).reset_index()
    a26 = a26[a26["n_bip"] >= 30]

    train = feat.merge(a26, on="pitcher_id", how="inner")
    train = train.merge(_own_air_feature(train["pitcher_id"], 2025, _own_air_table()), on="pitcher_id", how="left")
    model = smf.ols(AIR_SPEC_PROD, data=train).fit()
    print(f"Air% trait model (n={len(train)}): R2={model.rsquared:.3f}  "
          f"{model.params.round(4).to_dict()}")
    return model


def project_air_pct(raw: pd.DataFrame, air_model, roster_ids: pd.Series) -> pd.DataFrame:
    """Roll the frozen Air% trait model forward onto 2026 pitch characteristics
    (shrunk per pitcher, see _shrink_air_features) -> 2027 Air% projection, per
    pitcher_id. Every id in `roster_ids` gets a row -- a pitcher with zero 2026
    pitches gets the model's prediction AT the league-mean feature values
    (not a separate special-cased fallback), everyone else gets smoothly more
    of their own real traits as their 2026 sample grows."""
    r26 = fp.filter_period(raw, bt.START_2026, bt.END_2026)
    feat = _raw_pitch_features(r26)

    out = pd.DataFrame({"pitcher_id": roster_ids.unique()}).merge(feat, on="pitcher_id", how="left")
    out["n_pitches"] = out["n_pitches"].fillna(0)
    out = _shrink_air_features(out)
    out = out.merge(_own_air_feature(out["pitcher_id"], 2026, _own_air_table()), on="pitcher_id", how="left")
    out["proj_air_pct"] = air_model.predict(out).clip(0, 1)
    out["air_pct_source"] = np.where(out["n_pitches"] > 0, "trait_model_shrunk", "league_avg_fallback")
    # Held-out error of this Air% projection: the model's talent-level error
    # (AIR_TALENT_ERR_SD) plus, for pitchers whose input features were mostly
    # shrunk to the league mean for lack of tracked pitches, the explained
    # signal we couldn't observe -- ramps to the full population SD at 0 pitches.
    rel = np.mean([out["n_pitches"] / (out["n_pitches"] + n0) for n0 in AIR_FEATURE_N0.values()], axis=0)
    out["air_pct_sd"] = np.sqrt(AIR_TALENT_ERR_SD ** 2 + (1 - rel) * (AIR_POP_SD ** 2 - AIR_TALENT_ERR_SD ** 2))
    print(f"Air%: {(out['n_pitches']>0).sum()} via shrunk trait model (>=1 2026 pitch), "
          f"{(out['n_pitches']==0).sum()} at pure league average (0 2026 pitches)")
    return out[["pitcher_id", "proj_air_pct", "air_pct_sd", "air_pct_source"]]


def build_replacement_trustmark_shift(pct_home_games: float) -> float:
    """The SAME TrustmarkPark mechanism (build_trustmark_adjustment) applied to
    a generic league-average pitcher profile (K%, BB%, Air% all at their 2026
    league rate) -- i.e. how much would TrustmarkPark shift the REPLACEMENT
    baseline itself, not just the specific pitcher being evaluated. Needed
    because a replacement-level pitcher benefits from the same park effect any
    other pitcher would; comparing an adjusted pitcher against an unadjusted
    replacement baseline overstates the apparent RELATIVE gain."""
    profile = pd.DataFrame({
        "K_pred": [LEAGUE_K_RATE], "BB_pred": [LEAGUE_BB_RATE],
        "proj_air_pct": [LEAGUE_AVG_AIR_PCT],
        "proj_bf": [100.0 * LEAGUE_BF_PER_IP], "proj_ip": [100.0],
    })
    return build_trustmark_adjustment(profile, pct_home_games)["trustmark_ra9_shift"].iloc[0]


def add_replacement_probability(df: pd.DataFrame, ra9_col: str, repl_col: str, sd) -> pd.Series:
    """P(true RA9 < replacement level) -- i.e. probability of being better than
    a readily-available replacement, via the RA9 model's own held-out residual
    SD (RA9_RESID_SD). Normal CDF is a defensible choice specifically for RA9:
    its residuals were checked directly earlier this session and found
    well-behaved (skew=0.28, excess kurtosis=0.28, Shapiro-Wilk p=0.53 -- did
    NOT reject normality), unlike K%'s residuals, which clearly failed that
    same check and needed a bootstrap instead.

    `sd` is a per-pitcher array/Series (see RA9_LUCK_VAR_COEF), no longer one
    constant for everyone."""
    z = (df[repl_col] - df[ra9_col]) / sd
    return norm.cdf(z)


def build_trustmark_adjustment_hr_model(proj: pd.DataFrame, pct_home_games: float = 0.5) -> pd.DataFrame:
    """Given a projections frame with proj_K_rate, proj_BB_rate, proj_air_pct,
    proj_ip, proj_bf, compute the RA9 shift (negative = fewer runs allowed)
    from playing `pct_home_games` of the season at TrustmarkPark."""
    hr_blend = pct_home_games * TRUSTMARK_HR_FACTOR + (1 - pct_home_games) * 1.0
    hit_rate_delta_blend = pct_home_games * TRUSTMARK_HIT_RATE_DELTA

    out = proj.copy()
    contact_pct = 1 - out["K_pred"] - out["BB_pred"]
    hr_pct_baseline = contact_pct * out["proj_air_pct"] * HR_PER_AIR_LEAGUE
    hr_pct_trustmark = hr_pct_baseline * hr_blend
    nonhr_air_pct = contact_pct * out["proj_air_pct"] * (1 - HR_PER_AIR_LEAGUE)

    runs_saved_hr = (hr_pct_baseline - hr_pct_trustmark) * out["proj_bf"] * RUNS_PER_HR
    runs_saved_defense = nonhr_air_pct * out["proj_bf"] * hit_rate_delta_blend * AIR_HIT_RUN_VALUE
    out["trustmark_ra9_shift"] = -(runs_saved_hr + runs_saved_defense) * 9 / out["proj_ip"]
    out["trustmark_runs_saved_hr"] = runs_saved_hr
    out["trustmark_runs_saved_defense"] = runs_saved_defense
    return out


def build_trustmark_adjustment(proj: pd.DataFrame, pct_home_games: float = 0.5) -> pd.DataFrame:
    """PRODUCTION (2026-10-08): the profile translation (fl_pitcher_park: global effect ~0 + K% term, bootstrap-t shrunk) replacing the HR-factor model (kept as build_trustmark_adjustment_hr_model: it counted HR and
    air-ball hit suppression but not the offsetting K% / single / BB effects; contradicted by the published park factor and the pooled fixed-effects estimate: DECISION_LOG 2026-10-08).  `proj` needs `pitcher_id` or `name_clean`
    (an unknown pitcher gets the average-pitcher effect).  Adds trustmark_ra9_shift (negative = fewer runs allowed), trustmark_shift_sd and the legacy runs-saved columns (0)."""
    import fl_pitcher_park as pp
    out = proj.copy(); out["trustmark_ra9_shift"] = pp.shift_r9(out, pct_home_games); out["trustmark_shift_sd"] = pp.shift_sd_r9(out, pct_home_games)
    out["trustmark_runs_saved_hr"] = 0.0; out["trustmark_runs_saved_defense"] = 0.0
    return out


if __name__ == "__main__":
    print("Loading cached FL pitches and war_pitchers...")
    raw = fp.load_fl_pitches_cached()
    import fl_position_player_flag as ppf
    raw = ppf.exclude(raw)     # same exclusion as fl_production_projection.py (air-trait model fit and roster)
    war = read_sql_df("select player, year, bf, ip, so, bb, ra, ra9, rar, ppf, g "
                       "from war_pitchers where game_type='reg_season'")

    ra9_model, ra9_train_ranges = fit_ra9_baseline_model(war)
    import json
    json.dump({"spec": RA9_SPEC_PROD, "params": {k: float(v) for k, v in ra9_model.params.items()}, "train_ranges": ra9_train_ranges, "runs_per_win": 10.706},
              open("output/fl_ra9_model_coefs.json", "w"), indent=1)              # the dashboard's player page shows each input's share of the RA9 projection
    air_model = fit_air_trait_model(raw)

    # league-average fallbacks for ppf/ip_per_g_prior,
    # same neutral-imputation convention as LEAGUE_K_RATE/LEAGUE_BB_RATE above
    # -- used for bridge-only prospects with no real 2026 FL war_pitchers row
    w26_full = war[(war["year"] == 2026) & (war["ip"] >= 20)]
    LEAGUE_IP_PER_G = float(np.average(w26_full["ip"] / w26_full["g"], weights=w26_full["ip"]))
    print(f"League-average fallbacks: ip_per_g={LEAGUE_IP_PER_G:.2f}")

    print("\nLoading 2027 K%/BB% projections (fl_production_projection.py output)...")
    kbb = pd.read_parquet("output/fl_2027_projections.parquet")
    kbb["pitcher_id"] = kbb["pitcher_id"].astype(str)

    # K%/BB%: use the real production projection where it exists; fall back to
    # the other-league bridge rate (fl_k_rate_bridge/fl_bb_rate_bridge, already
    # populated for every roster pitcher) where it doesn't -- a naive but real
    # number instead of a blank cell, same "everyone gets a projection" standard
    # fl_production_projection.py already applies to the bridge lookup itself.
    d_ra9 = kbb.copy()
    n_no_model = d_ra9["proj_K_rate"].isna().sum()
    d_ra9["K_pred"] = d_ra9["proj_K_rate"].fillna(d_ra9["fl_k_rate_bridge"])
    d_ra9["BB_pred"] = d_ra9["proj_BB_rate"].fillna(d_ra9["fl_bb_rate_bridge"])
    print(f"K%/BB%: {len(d_ra9) - n_no_model} via the production model, "
          f"{n_no_model} via other-league bridge fallback")
    d_ra9 = d_ra9.dropna(subset=["K_pred", "BB_pred"]).copy()

    # workload + the new RA9 regressors all come from the SAME real-2026 row --
    # merge once, before predicting. `ppf`/`ip_per_g_prior` use real 2026 values
    # as the best available proxy for 2027 (team/park mix and usage pattern both
    # carry over year to year far more than they reset) -- the exact same
    # "most recent real year stands in for the projection year" convention
    # workload already used below.
    # one pitcher_id (Gilleran, 10103730) is recorded under two spellings in war_pitchers; his workload is the sum of both rows (majority spelling kept)
    w26_real = war[war["year"] == 2026][["player", "bf", "ip", "g"]].assign(player=lambda q: q["player"].replace(PITCHER_NAME_ALIASES)).groupby("player", as_index=False)[["bf", "ip", "g"]].sum().rename(
        columns={"bf": "proj_bf", "ip": "proj_ip"})
    d_ra9 = d_ra9.merge(w26_real, left_on="pitcher", right_on="player", how="left")
    # war_pitchers is keyed by NAME: a name still shared by two genuinely
    # different pitchers (after fl_production_projection's id consolidation,
    # only 'Davis, Tyler') would hand both the SAME blended workload row.
    # Treat those as having no usable war row instead of mis-attributing one.
    shared_name = d_ra9.duplicated("pitcher", keep=False)
    d_ra9.loc[shared_name, ["proj_bf", "proj_ip", "g"]] = np.nan
    print(f"Name still shared by 2+ pitcher_ids (war row withheld): {int(shared_name.sum())} rows")
    no_workload = d_ra9["proj_ip"].isna()
    # 2027 innings, conditional on pitching: regressed 2026 IP (see IP_PROJ_*); a
    # pitcher with a 2025 line but no 2026 innings is a post-gap returner
    # (GAP_RETURNER_IP); one with no FL innings at all is a debut (DEBUT_MEDIAN_IP).
    # NOT the placeholder 100 below, which is rate-invariant for the park shift but
    # would make a no-data pitcher look like a full-workload one.
    ip25 = war[war["year"] == 2025].assign(player=lambda q: q["player"].replace(PITCHER_NAME_ALIASES)).groupby("player")["ip"].sum()
    has26 = d_ra9["proj_ip"].notna() & (d_ra9["proj_ip"] > 0)
    has25 = d_ra9["pitcher"].map(ip25).fillna(0) > 0
    d_ra9["ip_unc"] = np.where(has26, IP_PROJ_A + IP_PROJ_B * d_ra9["proj_ip"].fillna(0),
                               np.where(has25, GAP_RETURNER_IP, DEBUT_MEDIAN_IP))
    d_ra9["ip_unc"] = d_ra9["ip_unc"].clip(lower=UNCERTAINTY_MIN_IP)
    d_ra9["p_return"] = np.where(has26, 1 / (1 + np.exp(-(RETURN_LOGIT_A + RETURN_LOGIT_B * np.log(d_ra9["proj_ip"].clip(lower=0.1).fillna(0.1))))),
                                 np.where(has25, GAP_RETURN_PROB, np.nan))
    print(f"Workload: {(~no_workload).sum()} with real 2026 IP/BF, "
          f"{no_workload.sum()} via league-average-rate placeholder")
    # ip_per_g_prior: EB-shrunk toward the league-wide games-weighted mean
    # (n0=12 games, see RA9_SPEC docstring) -- naturally pulls thin-games
    # pitchers toward league average instead of trusting their own noisy raw
    # ratio.
    d_ra9["ip_per_g_prior"] = (d_ra9["proj_ip"] + N0_IPG * LG_IPG) / (d_ra9["g"] + N0_IPG)

    # `ppf` is a RAW, unshrunk ratio (unlike
    # naive_K_rate/naive_BB_rate, which get proper EB shrinkage, and unlike
    # ip_per_g_prior, now shrunk above) -- the model was trained ONLY on
    # BF>=100 pitchers (fit_ra9_baseline_model's own population filter), so
    # applying it to a near-zero-BF roster pitcher extrapolates wildly on
    # these two inputs. Caught directly: a pitcher with 5 real 2026 PA and
    # pitches_per_pa_prior=7.4 (vs. a training range topping out at ~6.3)
    # produced proj_ra9_baseline=0.88 -- physically impossible. Fix: fall
    # back to league averages below the SAME BF>=100 floor the model was
    # trained under, not just when data is fully missing. Still applied to
    # ip_per_g_prior too as a belt-and-suspenders safety net even though
    # shrinkage already tempers it substantially on its own.
    thin_sample = d_ra9["proj_bf"].isna() | (d_ra9["proj_bf"] < 100)
    print(f"Thin-sample guard: {(~thin_sample).sum()} with real 2026 BF>=100 (model's own training "
          f"floor), {thin_sample.sum()} via league-average ip_per_g_prior")
    d_ra9.loc[thin_sample, "ip_per_g_prior"] = LEAGUE_IP_PER_G
    d_ra9.loc[no_workload, "proj_ip"] = 100.0
    d_ra9.loc[no_workload, "proj_bf"] = 100.0 * LEAGUE_BF_PER_IP
    # clip every continuous regressor to the model's own validated support
    # BEFORE predicting -- see clip_to_training_range's docstring for why
    # (LOOCV can't catch out-of-training-range extrapolation; this caught a
    # real, user-flagged case where it produced a -1.6 run swing for one
    # pitcher from pitches_per_pa_prior alone)
    n_clipped = {col: int(((d_ra9[col] < lo) | (d_ra9[col] > hi)).sum())
                 for col, (lo, hi) in ra9_train_ranges.items()}
    print(f"Clipped to training range: {n_clipped} (pitchers affected per regressor, out of {len(d_ra9)})")
    d_ra9 = clip_to_training_range(d_ra9, ra9_train_ranges)

    d_ra9["proj_ra9_baseline"] = ra9_model.predict(d_ra9)

    air_proj = project_air_pct(raw, air_model, d_ra9["pitcher_id"])
    d_ra9 = d_ra9.merge(air_proj, on="pitcher_id", how="inner")

    half = build_trustmark_adjustment(d_ra9, pct_home_games=0.5)
    full = build_trustmark_adjustment(d_ra9, pct_home_games=1.0)
    final = d_ra9.copy()
    final["k_bb_source"] = np.where(final["proj_K_rate"].isna(), "bridge_fallback",
                                    np.where(final["proj_basis"] == "gap_prior", "gap_prior", "model"))
    final["trustmark_ra9_shift_half"] = half["trustmark_ra9_shift"]
    final["trustmark_ra9_shift_full"] = full["trustmark_ra9_shift"]

    # MISSISSIPPI translation: the baseline (proj_ra9_baseline) is PARK-NEUTRAL, so the
    # Mississippi projection is simply baseline + the explicit TrustmarkPark shift
    # (Air%-driven HR + non-HR air-ball hit suppression; half or all home games). The
    # same arithmetic applies to every pitcher, Mississippi's own included, so a
    # Mississippi pitcher's "Mississippi RA9" and anyone else's are directly
    # comparable, and the park-neutral number is the clean separation between pitcher
    # and park the front office asked for. (Earlier versions also swapped a `ppf` park
    # term to a Trustmark blend and added the shift on top -- the same suppression
    # counted twice; see DECISION_LOG.md "Trustmark park effect was counted twice" and
    # "Park-neutral RA9 + Mississippi translation".)
    final["proj_ra9_trustmark_half"] = final["proj_ra9_baseline"] + final["trustmark_ra9_shift_half"]
    final["proj_ra9_trustmark_full"] = final["proj_ra9_baseline"] + final["trustmark_ra9_shift_full"]

    # replacement level: ONE league-wide constant for everyone (see module
    # constant docs for why the per-pitcher back-out was wrong), and the
    # TrustmarkPark adjustment applied to that SAME baseline via a generic
    # league-average pitcher profile -- so the probability comparison is
    # apples-to-apples (adjusted pitcher vs. adjusted replacement), not
    # adjusted-pitcher-vs-unadjusted-replacement.
    final["ra9_replacement"] = LEAGUE_RA9_REPLACEMENT
    repl_shift_half = build_replacement_trustmark_shift(0.5)
    final["ra9_replacement_trustmark_half"] = LEAGUE_RA9_REPLACEMENT + repl_shift_half
    print(f"\nReplacement level: {LEAGUE_RA9_REPLACEMENT:.3f} league-wide; "
          f"{final['ra9_replacement_trustmark_half'].iloc[0]:.3f} at half-Trustmark "
          f"(shift={repl_shift_half:+.3f})")

    # Per-pitcher SD: single-season luck over the pitcher's expected innings
    # (RA9_LUCK_VAR_COEF / IP), plus -- for the Trustmark scenario only, since
    # Air% enters the baseline model nowhere else -- the propagated held-out
    # error of the Air% projection: shift is proportional to Air%, so an Air%
    # error of air_pct_sd scales the shift by air_pct_sd / proj_air_pct.
    final["ip_unc"] = d_ra9["ip_unc"].values
    luck_var = RA9_LUCK_VAR_COEF / final["ip_unc"]
    air_rel_err = final["air_pct_sd"] / final["proj_air_pct"].clip(lower=0.05)
    final["proj_ra9_sd"] = np.sqrt(luck_var)
    final["trustmark_shift_sd_half"] = half["trustmark_shift_sd"].values
    final["proj_ra9_sd_trustmark_half"] = np.sqrt(luck_var + final["trustmark_shift_sd_half"] ** 2)       # luck + the shift's own bootstrap uncertainty (replaces the Air%-error propagation)
    final["prob_above_replacement"] = add_replacement_probability(
        final, "proj_ra9_baseline", "ra9_replacement", final["proj_ra9_sd"])
    final["prob_above_replacement_trustmark_half"] = add_replacement_probability(
        final, "proj_ra9_trustmark_half", "ra9_replacement_trustmark_half", final["proj_ra9_sd_trustmark_half"])

    # leaderboard columns (user layout 2026-10-08): 2026 innings, RA9- (100 x park-neutral RA9 / league RA9, lower = better), park-neutral WAR over the projected innings
    w26 = war[war["year"] == 2026].groupby("player")["ip"].sum(); dup = final["pitcher"].map(final["pitcher"].value_counts()) > 1
    final["ip_2026"] = final["pitcher"].map(w26).where(~dup)                                    # actual 2026 innings (NaN for pitchers with no 2026 line or a shared name)
    lg_ra9 = 9 * war[war["year"] == 2026]["ra"].sum() / war[war["year"] == 2026]["ip"].sum(); final["lg_ra9"] = lg_ra9
    final["ra9_minus"] = 100 * final["proj_ra9_baseline"] / lg_ra9
    final["proj_war"] = (final["ra9_replacement"] - final["proj_ra9_baseline"]) * final["ip_unc"] / 9 / 10.706      # park-neutral WAR over the projected innings (runs per win 10.706)

    out_cols = ["pitcher_id", "pitcher", "team", "ip_2026", "lg_ra9", "ra9_minus", "proj_war", "K_pred", "BB_pred", "k_bb_source",
                "ip_per_g_prior",
                "proj_air_pct", "air_pct_sd", "air_pct_source", "proj_basis", "ip_unc", "p_return",
                "proj_ra9_sd", "proj_ra9_sd_trustmark_half",
                "proj_ra9_baseline", "trustmark_ra9_shift_half", "trustmark_shift_sd_half", "proj_ra9_trustmark_half",
                "trustmark_ra9_shift_full", "proj_ra9_trustmark_full",
                "ra9_replacement", "ra9_replacement_trustmark_half",
                "prob_above_replacement", "prob_above_replacement_trustmark_half"]
    final[out_cols].to_parquet(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}  (n={len(final)})")
    print(f"\nIP-weighted average Trustmark RA9 shift (half home games): "
          f"{np.average(final['trustmark_ra9_shift_half'], weights=final['proj_ip']):.3f}")
    print("\nSample (highest probability of beating replacement level):")
    print(final.sort_values("prob_above_replacement", ascending=False)
          [["pitcher", "team", "proj_ra9_baseline", "ra9_replacement",
            "prob_above_replacement", "prob_above_replacement_trustmark_half"]]
          .head(8).to_string(index=False))
