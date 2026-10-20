# Decision Log — FLHitting MLB K%/BB% Projection Project

A running record of methodological trade-offs: what was chosen, what the
alternative(s) were, and why — kept so the reasoning behind a choice is never
just "whatever the code happens to do." New entries go at the top.

**Entry format:**
> ### <short title>
> - **Choice:** what we did
> - **Alternative(s):** what else was on the table
> - **Why:** the reasoning
> - **Cost/risk accepted:** what we gave up or are exposed to
> - **Revisit if:** (optional) the condition under which this should be reconsidered

### RA/9 pivot: walk-forward (look-ahead) validation confirms the ~0.05-0.10 R2 estimate, and catches an overclaim on HR%

- **Problem with the prior multi-fold check:** the leave-one-fold-out cross-
  validation done just before this (see "LOOCV reveals..." entry below)
  held out one season-transition at a time, but TRAINED on the other 3 --
  including, for the 2022->2023 test fold, data from 2023->2024,
  2024->2025, and 2025->2026. That uses future information to predict the
  past, which is not a fair test of "will this work for 2027." Caught and
  fixed per the user's explicit point that time-series data needs
  look-ahead (walk-forward) validation, not just leave-one-out.
- **Fix:** walk-forward validation -- for each test transition, train ONLY
  on transitions that happened strictly BEFORE it (expanding window),
  moving forward in time. Three genuinely future-only predictions:
  - train {2022->23} -> test 2023->24: R2_w=0.052
  - train {2022->23, 2023->24} -> test 2024->25: R2_w=0.065
  - train {2022->23, 2023->24, 2024->25} -> test 2025->26: R2_w=0.056
  - pooled: R2_w=0.107
  Remarkably consistent fold-to-fold (0.05-0.07 each) -- a real, stable
  signal, not a one-era fluke. Confirms the leave-one-fold-out estimate
  (0.073-0.085) from before was in the right neighborhood even though its
  methodology had the look-ahead flaw -- fixing the flaw didn't change the
  conclusion, just made it trustworthy for the right reason.
- **Caught and corrected an overclaim on HR% while checking this:** HR%'s
  coefficient was negative and "stable" everywhere it was tested earlier,
  and was described as "clean, real, FIP-consistent signal" when first
  added. Never checked: (1) its own bivariate correlation with the TARGET,
  (2) its statistical significance, (3) whether that significance survives
  pooling across other eras. All three check out badly: bivariate corr with
  future RA9 in the pooled sample is ~0 (-0.02), and its p-value in the
  full pooled spec is 0.187 (not significant) -- even though it WAS
  significant (p=0.03-0.05) in the single 2025->2026 population alone.
  Significance that disappears with MORE data (pooled across eras) rather
  than strengthens is the opposite of what a real, stable effect should
  do. Dropping HR% from the walk-forward spec confirmed this: R2_w goes
  from 0.1049 to 0.1068 -- trivially BETTER without it, not worse.
- **Likely explanation, not a sign-flip bug:** HR% is a well-known noisy,
  luck-driven rate stat. Once `naive_ra9` already captures a pitcher's
  overall level, a pitcher whose HR% looked unlucky that year (while
  naive_ra9/K%/BB% looked normal) is disproportionately likely to have
  been unlucky generally -- and unlucky pitchers regress toward BETTER
  future performance. That produces a real, legitimate negative partial
  coefficient without HR% capturing any genuine "contact quality" signal
  -- the earlier characterization of it as FIP-consistent, stable signal
  was an overclaim that should have been checked for significance and
  bivariate direction at the time, not just coefficient-sign stability.
- **Revised validated spec for 2027 projections:** `naive_ra9 + naive_K% +
  naive_BB%` (pooled N0_NAIVE=16.1, n0_k=58.3, n0_bb=61.3, all fit on the
  much larger n=1,734 pooled cross-section rather than single-year
  estimates), walk-forward R2_w~=0.10. HR%, bridge-projected K%/BB%, park
  adjustment, BABIP, wh_sva_level all remain either dropped (HR%) or
  untested at this rigor (the pitch-level-only features, which can't be
  walk-forward-tested across eras since they only exist for 2025-2026).
- **Revisit if:** more FL seasons/pitch-level data accumulate -- re-run
  this exact walk-forward check with more folds, and re-test whether the
  pitch-level features (bridge K%/BB%, ppf, BABIP, wh_sva_level) survive
  an analogous walk-forward design once more Trackman-era data exists to
  build one from.

---

### RA/9 pivot: multi-year other-league bridge rates don't help M3 either, even for pitchers with a real match

- **Question:** does the multi-year-weighted other-league history already
  built for K%/BB% (`fl_k_rate_bridge`/`fl_bb_rate_bridge` -- blending
  college, Pioneer League, and affiliated MiLB history, per the
  already-validated college-bridge "BB% wants multi-year, K% wants
  single-year" finding) help the RA9 model at M3 (`naive_ra9 +
  K_pct_2025_raw + BB_pct_2025_raw + HR_pct_2025`, FL-only single-year
  inputs), tested under the same LOOCV standard as the milestone comparison
  above (n=96, slightly larger than the n=89 used elsewhere since this
  check doesn't require BABIP_2025/wh_sva_level/K_pred_2026 availability)?
- **Result:** in-sample ticks up (0.2231->0.2384) but LOOCV goes DOWN
  (0.1338->0.1122) -- the same "looks fine in-sample, doesn't survive
  held-out scoring" pattern as BABIP_2025/wh_sva_level above. Checked
  whether this was being diluted by the 38/96 pitchers with no real
  other-league match (filled with a league-average placeholder, pure
  noise for them): restricting to only the 58 pitchers with a genuine
  bridge match, LOOCV still goes DOWN (0.0065->-0.0239) -- not a dilution
  artifact, a real null/negative result even for the players the bridge
  actually describes.
- **Why:** the bridge rates were built and validated to predict a
  pitcher's OWN FL K%/BB%, which are already directly in the M3 model as
  `K_pct_2025_raw`/`BB_pct_2025_raw`. Once the real FL rate is already
  present, the bridge rate is a second, noisier estimate of the same
  underlying signal, not new information about runs allowed specifically --
  it has nothing left to contribute once the thing it approximates is
  already in the equation.
- **Revisit if:** testing other-league info against a spec that DOESN'T
  already include the pitcher's own FL K%/BB% (e.g., for brand-new FL
  debuts with little/no FL track record, where the bridge rate would be
  the ONLY signal available, not a redundant second one).

---

### RA/9 pivot: multi-year FL-own weighting -- pulled real 2022-2024 R/HR data, no confirmed benefit but badly underpowered

- **Gap found:** "multi-year FL weighting" (a genuine Marcel-style weighted
  average of a pitcher's OWN prior FL seasons, as opposed to multi-year
  OTHER-league weighting, already tested and found unhelpful above) had
  never actually been tested, because FL R/HR data pre-2025 had never been
  pulled -- the existing StatsCrew K%/BB% pull (`fl_statscrew_pull.py`,
  2022-2024) only captured IP/H/BB/HBP/SO/BF, not R/HR, so RA9 couldn't be
  computed for those years.
- **Fix:** re-scraped all 48 FRON team-year pages (17 teams active
  2022-2024, statscrew.com/minorbaseball) via browser, this time capturing
  R and HR alongside the existing fields. Confirmed live: BF is
  results-based only (no pitch-level data pre-2025, as expected) and
  missing/fallback-needed for the same years previously identified (742 of
  1,164 team-stints needed the outs+H+BB+HBP fallback formula). Saved to
  `output/fl_statscrew_pitching_r_hr_2022_2024.parquet` (1,121
  pitcher-seasons, 2022-2024).
- **This unlocks a genuinely new, independent held-out check:** predicting
  REAL 2025 FL outcomes (`war_pitchers`, BF>=100) from 2022-2024 FL
  history -- a different transition than the 2025->2026 one used
  everywhere else in this pivot.
- **Raw multi-year RA9 averaging: negative, consistent with the pivot's
  opening finding** (naive RA9 alone needs K%/BB%/HR% decomposition to be
  useful at all): single-year R2_w=-0.25, multi-year recency-weighted
  (Marcel-style 3/2/1 by recency, IP-weighted within year) R2_w=-0.34 --
  multi-year slightly WORSE, not better.
- **Fair test -- multi-year-weighted K%/BB%/HR% (mirroring what M3 actually
  uses), n=34 (pitchers with 2+ of the 3 years AND real 2025 BF>=100):**
  in-sample ticks up (single-year R2_w=0.2500 -> multi-year R2_w=0.2636,
  consistent small improvement on MAE_w/RMSE_w too), but LOOCV essentially
  erases the gap (single-year 0.0573 -> multi-year 0.0639) -- both collapse
  hard from in-sample to LOOCV (same overfitting signature as
  BABIP_2025/wh_sva_level), and the LOOCV difference between them is
  trivial, well inside noise at this n.
- **Why n is this small:** only 34 of 83 matched FL-2025 pitchers have 2+
  years of 2022-2024 FL history at all -- most FL pitchers who reach 2025
  don't have a multi-year FL track record to weight in the first place
  (consistent with the league's known high turnover). This caps how useful
  multi-year FL weighting could ever be here, independent of the noise
  issue.
- **Conclusion:** no confirmed benefit, but this is an underpowered test,
  not a clean negative -- different from the other-league-bridge finding
  above, which had a clearer mechanistic reason (redundancy with existing
  predictors) to expect a null result.
- **Revisit if:** more FL seasons accumulate (2027+) -- the matched-2+-year
  population will only grow, and this exact check can be rerun with more
  power once it does.

---

### RA/9 pivot: LOOCV reveals BABIP_2025 and wh_sva_level don't survive genuine held-out validation -- M4 (projected K%/BB% + HR%) is the real spec, not the full 6-feature stack

- **Problem with every number reported in this RA9 pivot so far:** every
  milestone (park adjustment, BABIP_2025, wh_sva_level, projected K%/BB%,
  the n0 fix) was fit on the full n=89 matched population and scored on
  that SAME n=89 -- in-sample with respect to the regression coefficients,
  even though several predictors (naive_ra9, K_pred_2026, BB_pred_2026)
  reference an earlier year. This is NOT the "fit on the training
  transition, freeze, score on what it never saw" discipline the K%/BB%
  Experiment 1/2 folds use.
- **Choice:** ran leave-one-out cross-validation (n=89 refits -- cheap at
  this n) for a chain of milestones: refit the spec on the other 88 points,
  predict the one held out, aggregate across all 89 truly out-of-fold
  predictions, score R2_w/MAE_w/RMSE_w on those (never on an in-sample fit).
- **Result -- in-sample vs LOOCV R2_w by milestone:**
  - M1 naive_ra9 only: in-sample 0.0385 -> LOOCV **-0.0153** (confirms the
    session's original finding that naive RA9 alone is worse than
    predicting the mean, out-of-sample)
  - M2 + raw K%/BB%: 0.1903 -> 0.1051
  - M3 + HR%: 0.2395 -> 0.1378
  - M4 projected K%/BB% (swap raw->pred) + HR%: 0.2821 -> **0.1891**
  - M5 + BABIP_2025: 0.2894 -> 0.1873 (LOOCV went DOWN slightly)
  - M6 + wh_sva_level (prior "current best"): 0.3111 -> 0.1881 (LOOCV
    essentially flat vs. M4, despite the in-sample number being the
    highest of the whole pivot)
- **Why this happened:** BABIP_2025 and wh_sva_level are both real,
  non-collinear correlates of RA9 (confirmed earlier via correlation checks
  and stable, correctly-signed coefficients) -- this isn't the BIP%-style
  multicollinearity failure. The failure mode here is different: with only
  n=89 and 6+ parameters, the model has enough flexibility to fit in-sample
  noise specific to these 89 pitchers that doesn't transfer to a genuinely
  unseen one. Stable coefficients and non-collinearity are necessary but
  not sufficient for a feature to generalize at this sample size.
- **Choice:** revised the recommended spec DOWN from the full 6-feature
  stack (M6) to M4: `naive_ra9 (ppf-adjusted) + K_pred_2026 + BB_pred_2026 +
  HR_pct_2025`. Honest validated performance: LOOCV R2_w=0.1891,
  MAE_w=1.1696, RMSE_w=1.4408 -- noticeably lower than the in-sample
  numbers reported earlier in this pivot, but trustworthy.
- **Not a rejection of BABIP_2025/wh_sva_level as real signals** -- just an
  honest statement that n=89 can't currently validate them as genuine
  improvements. Revisit once more FL seasons give a larger matched
  population to re-run this same LOOCV check on.
- **Revisit if:** more FL season data becomes available -- re-run this
  exact LOOCV comparison with a larger n before deciding whether
  BABIP_2025/wh_sva_level belong in the shipped spec.

---

### RA/9 pivot: fixed the reliability weighting -- wrong denominator (BF, not IP) and an unvalidated, triple-purposed n0=200

- **Problem found (same shape as the K%/BB% n0_eval issue earlier this
  session):** the RA9 work had been using `reliability_weight(bf_2026, 200)`
  for THREE different roles at once -- `naive_ra9`'s own EB shrinkage of
  2025 raw RA9 toward league average, the WLS fitting weight, and the
  reported R2_w/MAE_w/RMSE_w -- with n0=200 never actually validated for
  RA9, just carried over from whichever value was typed in when `naive_ra9`
  was first built. On top of that, BF was used as the sample-size
  denominator purely out of habit (every K%/BB% weight in this project uses
  PA/BF, since K%/BB% are PA-denominated rates) without checking that RA9
  is actually IP-denominated (runs * 9 / innings pitched), making IP the
  theoretically correct unit here, not BF.
- **Checked the denominator question directly:** fit Var(residual of the
  current-best model) vs 1/n for both BF_2026 and IP_2026 (n=89, the
  matched test population). Both showed a real, statistically significant
  1/n relationship (p=0.037 BF, p=0.020 IP) -- small-sample pitcher-seasons
  really are noisier -- but implied VERY different n0 (552 for BF, 146 for
  IP), and at R2~0.05-0.06 on n=89 this single-transition fit is too noisy
  to fully trust on its own; it was treated as a rough, noisy point estimate
  for the eval/fit role specifically, not as proof either denominator is
  exactly right.
- **Separated the naive-shrinkage n0 from the eval/fit n0** (mirroring the
  K%/BB% "n0 vs n0_eval" split): `naive_ra9`'s own shrinkage question --
  "how much to trust a pitcher's raw 2025 RA9 as an estimate of their true
  talent" -- doesn't need the small n=89 matched population at all; it can
  be fit on the FULL 2025 cross-section (n=244 at IP>=20), the same way
  `mm.estimate_n0` fits K%/BB%'s naive-prior n0 on a full-season pool rather
  than a matched-pair subset. Built the Poisson-on-IP analog of that
  method-of-moments formula (RA9's sampling noise is `9*lg_ra9/IP`, the
  Poisson-count equivalent of K%/BB%'s `p(1-p)/PA` binomial term): result
  N0_NAIVE=22.1, a precise, well-estimated number (n=244) that is WAY below
  the 200 that had been in use -- raw RA9 stabilizes as a talent estimate
  far faster than had been assumed.
- **Choice:** adopted N0_NAIVE=22.1 (IP-based) for `naive_ra9`'s shrinkage
  immediately -- well-estimated, large-n, unambiguous win. Adopted n0_eval=
  146 (IP-based) for the WLS fitting weight AND the reported metrics, with
  the explicit caveat that it's the noisiest number in this whole RA9
  pivot (n=89 ceiling, can't be improved without more matched FL seasons).
- **Result, same current-best spec** (naive_ra9 + K_pred_2026 + BB_pred_2026
  + BABIP_2025 + HR_pct_2025 + wh_sva_level), isolating just the eval/fit
  weight change (both rows already use the corrected N0_NAIVE=22.1 for
  naive_ra9 itself):
  - old (BF, n0=200): R2_w=0.3040  MAE_w=1.0889  RMSE_w=1.3586
  - new (IP, n0_eval=146): R2_w=0.3111  MAE_w=1.0596  RMSE_w=1.3280
  A real, if modest, further improvement -- smaller than the projected-K%/
  BB% or wh_sva_level gains, but free (no new data, just correcting a
  methodological shortcut).
- **Revisit if:** more FL seasons become available -- re-fit n0_eval on a
  larger matched population the same way N0_NAIVE was upgraded from a
  guess to a precise n=244 estimate; right now n0_eval=146 is the best
  available number but not a precise one.

---

### RA/9 pivot: production-projected K%/BB% beat raw rates; stuff traits mostly don't move contact quality; a name-collision merge bug corrupted one check before it was caught

- **Projected vs. raw K%/BB% as RA9 predictors:** had been feeding raw 2025
  K%/BB% (`so_2025/bf_2025`, `bb_2025/bf_2025`) into the RA9 model. Swapped
  in the actual production K%/BB% projection (the `naive+other_league+
  surprise_prior` WLS spec from `fl_halfseason_backtest.py`, fit on the only
  real transition available, 2025->2026, in-sample -- same convention this
  harness already uses when there's no further season to hold out) instead.
  On the same n=89 (7 pitchers dropped for missing bridge features):
  raw R2_w=0.2452/MAE_w=1.1529/RMSE_w=1.4149 -> projected
  R2_w=0.2863/MAE_w=1.1163/RMSE_w=1.3758 -- the single biggest jump found in
  the whole RA9 exploration. Makes sense: raw single-season rates carry real
  sampling noise, and noisy predictors mechanically attenuate a downstream
  regression -- our production model exists specifically to strip that
  noise out, so its output is just a cleaner version of the same signal.
- **Tested keeping BOTH raw and projected K%/BB% together** (since they're
  correlated, r=0.86 for K%, r=0.96 for BB%, not identical): R2_w ticks up
  to 0.3025, but `BB_pred_2026`'s coefficient balloons to 49.2 and
  `BB_pct_2025_raw` flips to -23.1 (nonsensical sign), neither raw term
  significant (p=0.78, p=0.16), design-matrix condition number nearly
  doubles (589->1010) -- the same multicollinearity signature as the
  BIP%-vs-K%/BB% case above. Not adopted; raw stays OUT once projected is in.
- **Other pitcher-trait candidates for the contact-quality side, tested
  against BABIP_2025/xwoba_allowed first:** raw induced vertical break
  (ivb), horizontal break, effective velocity, and the existing validated
  K%/BB% feature-level helpers (`wh_sva_level`, `horiz_var`/`vert_var`,
  `spin_z_norm`, `extension`) were correlated against BABIP_2025 and
  xwoba_allowed (n=369, PA-filtered). ALL correlations were weak
  (|r|<=0.12) -- consistent with classic DIPS theory: a pitcher's stuff/
  movement/command controls K%/BB%/HR% far more than it controls what
  happens to a ball once contact is made. `horz_break_abs` was the best
  BABIP_2025 correlate (r=-0.11) and `wh_sva_level` the best xwoba_allowed
  correlate (r=-0.12) -- both still weak individually.
- **Added the two best candidates to the real RA9 model** (current best =
  naive_ra9 + K_pred_2026 + BB_pred_2026 + BABIP_2025 + HR_pct_2025,
  R2_w=0.2863): `horz_break_abs` added essentially nothing (R2_w=0.2876).
  `wh_sva_level` added a real, non-trivial lift: R2_w=0.3027, MAE_w
  1.1163->1.0883, RMSE_w 1.3758->1.3599, with a sensible, correctly-signed
  coefficient (-15.7) and no instability -- bigger than the park-adjustment
  gain alone. Plausible reason: `wh_sva_level` is a direct, granular stuff
  measurement, while `K_pred_2026` is a shrunk/blended composite that loses
  some of that granularity -- the two aren't fully redundant even though
  both ultimately trace back to whiff-related pitch quality.
- **Bug caught and fixed before trusting the result:** the first attempt at
  this check joined a new per-pitcher feature table onto the RA9 population
  by player name (`"Last, First"` string) alone. 9 names in the raw FL pitch
  data collide across genuinely distinct `pitcher_id`s (e.g. two different
  "Pindel, Buddie" records), so the plain name-merge silently duplicated/
  misattributed rows -- n inflated from the expected 89 to 95, corrupting
  that run's numbers. Caught by checking `m['player'].is_unique` after each
  merge step; fixed by deduping the feature table to one `pitcher_id` (the
  one with the most 2025 pitches) per name before merging, which restored
  n=89 and gave the real, trustworthy numbers above. Any future per-pitcher
  feature merge keyed by name (not `pitcher_id`) needs the same dedup step
  or the same explicit uniqueness assertion.
- **Not yet done:** still the single year-over-year season-level check, not
  the project's real fold structure.
- **Revisit if:** building the RA9 model for real -- current best validated
  spec is naive_ra9 (ppf-adjusted) + K_pred_2026 + BB_pred_2026 (projected,
  not raw) + BABIP_2025 + HR_pct_2025 + wh_sva_level, target ppf-adjusted
  then un-adjusted back for scoring. Move this onto proper held-out folds
  before shipping.

---

### RA/9 pivot: park-adjusting the target (and naive_ra9 predictor) is a real, honest improvement -- use `ppf`, not `pf`

- **Context:** `war_pitchers` already carries two park-factor columns, `pf`
  and `ppf`, built elsewhere (no source code for their construction is in
  this repo -- treated as an external, unverified input and validated
  empirically rather than trusted at face value, per project discipline).
  Both are player-specific (vary row-to-row even within the same team/year,
  implying they're blended across each pitcher's actual mix of home/road
  parks, not a flat per-venue number).
- **Checked which one is actually a runs park factor before using either:**
  correlation with raw observed ra9_2026 is weak for `pf` (r=0.138) and
  meaningfully stronger for `ppf` (r=0.237). Neither is strong -- consistent
  with the classic FIP-vs-ERA gap (park effects on component rates don't map
  cleanly onto actual runs) -- but `ppf` is the more defensible of the two.
- **Choice:** park-adjust the RA9 TARGET with `ppf`: fit the model against
  `ra9_2026 * 100/ppf_2026` (removing park noise from what the model is
  trying to explain), then multiply predictions back by `ppf_2026/100`
  before scoring against the real, unadjusted `ra9_2026` -- never score
  against the adjusted target itself, that would be comparing against a
  different quantity than what we actually care about forecasting.
- **Result (n=96, same BF>=100-both-years population, best spec =
  naive_ra9 + K% + BB% + xwoba_allowed + HR%):**
  - no park adjustment: R2_w=0.1937  MAE_w=1.2350  RMSE_w=1.5613
  - target adjusted by `pf`:  R2_w=0.2021  MAE_w=1.2422  RMSE_w=1.5531
  - target adjusted by `ppf`: R2_w=0.2269  MAE_w=1.2204  RMSE_w=1.5288
  - target adj (`ppf`) + ALSO park-adjusting the `naive_ra9` predictor
    (2025 raw RA9 de-parked by `ppf_2025` before EB shrinkage): R2_w=0.2275
    MAE_w=1.2180  RMSE_w=1.5282 -- a further small gain on top of the
    target-only adjustment.
- **Why this makes sense:** removing park noise from the dependent variable
  before fitting lets the regression explain the PART of RA9 variance that's
  actually skill-driven, rather than spending degrees of freedom partly
  fitting park noise; un-adjusting predictions back before scoring keeps the
  comparison fair against the real baseline's R2_w/MAE_w/RMSE_w (same target
  quantity, not an easier adjusted one).
- **Not yet done:** still the single year-over-year season-level check, not
  the project's real fold structure. Also haven't checked whether `ppf`
  itself is stable/well-estimated at this sample size, or whether it should
  be blended with the simpler team-only `park_factors_fip.park_factor`
  table already used elsewhere in this project for SO%/BB%.
- **Revisit if:** building the RA9 model for real -- adopt `ppf`-adjusted
  target + `ppf`-adjusted naive_ra9 predictor as part of the validated spec,
  but confirm `ppf`'s own reliability (e.g. multi-year stability) before
  leaning on it harder than this single check does.

---

### RA/9 pivot: quality-of-contact-allowed (xwOBA allowed, exit velo, hard-hit%) tested as RA9 predictors -- xwOBA allowed is a real, stable signal; raw exit velo/hard-hit% are not

- **Context:** moving the projection target from K-BB% to RA/9 (K%/BB% carrying
  most of the signal, per the classic FIP-vs-ERA finding -- confirmed earlier
  this session: naive_ra9's own coefficient collapses to ~0 once K%/BB% are
  added). Tested whether "ball in play" outcomes add further signal beyond
  K%/BB%, first via the outcome-rate decomposition (HR%, BIP% = 1-K%-BB%-HBP%),
  then via actual batted-ball quality data once "ball in play" was clarified to
  mean quality of contact, not just the BIP rate.
- **Data found:** `output/batted_ball_event_residuals.csv` already has real,
  tracked (not reconstructed) batted-ball-level `exit_speed`, `angle`, `woba`,
  `xwoba` with `pitcher_id`/`pitcher` for 2025+2026 (80,773 events, 460-487
  pitchers/year) -- no new pull needed. Built pitcher-season aggregates
  (exit_velo_allowed, hard_hit_pct_allowed @ EV>=95, xwoba_allowed) on the same
  BF>=100-both-years population (n=96) used for the earlier RA9 check; median
  153 BIP per pitcher in this population (min 63), so not a tiny-sample metric.
- **Choice:** validated each contact-quality metric honestly before adopting
  any of them, same discipline as the K%/BB%/HR%/BIP% check.
  - baseline (naive_ra9 + K% + BB%): R2_w=0.1596
  - + HR%: R2_w=0.1914 (clean, stable coefficients -- already known good)
  - + exit_velo_allowed: R2_w=0.1623 (essentially no lift)
  - + hard_hit_pct_allowed: R2_w=0.1597 (no lift at all)
  - + xwoba_allowed: R2_w=0.1725 (real, modest lift)
  - + xwoba_allowed + HR%: R2_w=0.1937 (best stable combination)
  - naive_ra9 + xwoba_allowed only (no K%/BB%): R2_w=0.0511 -- confirms xwOBA
    allowed does NOT substitute for K%/BB%, only supplements them.
- **Why xwOBA allowed works and raw exit velo/hard-hit% don't:** xwOBA already
  blends exit velo + launch angle + (implicitly) location into a run-value-
  weighted quality score, closer to "how good was the contact, run-value-wise"
  than a single physical measurement. Raw exit velo and a binary hard-hit cutoff
  throw away the launch-angle information that determines whether hard contact
  is a double or a routine liner to the gap.
- **Collinearity check:** unlike BIP_pct (which is ~1-K%-BB%-HBP% by
  construction and produced unstable, sign-flipped coefficients when added
  alongside K%/BB% -- see the HR%/BIP% entry above), xwoba_allowed's
  correlation with K_pct/BB_pct is modest (-0.23/-0.14), and every model's
  K%/BB%/HR% coefficients stayed stable and sensibly signed after adding it.
  This is a genuinely independent signal, not a collinear restatement of K%/BB%.
- **Not yet done:** this is still a single year-over-year (2025->2026) check
  at the season level, same as the rest of the RA9 work so far -- not yet
  validated on the project's standard half-season fold structure, and not yet
  wired into production.
- **Revisit if:** building the RA9 model for real -- combine HR% + xwoba_allowed
  (both independently validated, low collinearity with each other or with K%/BB%)
  as the "ball in play" component, and move the whole RA9 check onto proper
  held-out folds before shipping anything.

---

### K-BB% CI bounding attempt: deriving from component K%/BB% logit CIs does NOT fix the low-PA blowup -- tried, reverted

- **Problem found:** The dashboard's K-BB% 90% CI (`ci_str(..., bounded=False)`,
  a plain point +/- z*se interval, deliberately unbounded since K-BB% can be
  negative) was producing physically implausible bounds for very-low-PA
  pitchers -- e.g. a 1-PA pitcher's 90% CI reaching 96%+ on the high end and
  -84%+ on the low end, vs. a REAL observed K-BB% range (at PA>=60 in actual
  FL data) of only [-16%, +40%]. Root mechanism: `get_prediction().se_obs`
  divides the residual-variance term by the reliability weight w=PA/(PA+n0),
  so as PA->0, w->0 and SE grows essentially without bound.
- **Choice:** Did NOT ship a fix this round -- reverted the one approach
  tried.
- **Alternative tried:** Derive K-BB%'s CI from the two ALREADY-bounded
  (logit-transform) K%/BB% CIs instead (kbb_lo = k_lo - bb_hi, kbb_hi =
  k_hi - bb_lo), reasoning that since each component is bounded in (0,1)
  their difference should be too.
- **Why rejected:** Checked directly and the reasoning was wrong. Bounding
  a component to (0,1) does not stop its OWN interval from being vacuously
  wide under extreme uncertainty -- for the 1-PA case above, the logit-
  bounded K% CI alone was already [0.2%, 96.8%] (spans nearly the entire
  possible range) and BB% CI was [0.4%, 84.6%]. Subtracting two
  already-vacuous bounded intervals can still span nearly the full
  [-100%, 100%] range. Measured directly: this made things WORSE for the
  exact population it was meant to fix -- the fraction of pitchers whose
  upper CI bound exceeds the population's own max point estimate rose from
  34% (171/498) to 64% (318/498).
- **Real takeaway:** The problem isn't the shape of the transform (logit
  vs. plain) or which metric it's applied to -- it's that se_obs itself
  explodes as PA->0, and nothing downstream of that (any bounded or
  unbounded transform) can turn a vacuous interval into a useful one. A real
  fix needs to target the SE computation directly (e.g. a floor on the
  reliability weight used for the SE specifically, not the point estimate,
  or simply not displaying an individual CI below some PA floor) rather
  than reshaping how an already-enormous SE gets converted into a
  displayed range.
- **Revisit if:** picked back up -- test a reliability-weight floor for the
  SE computation (point estimate's EB shrinkage already handles low PA
  correctly; only the uncertainty formula needs the equivalent treatment)
  or a PA floor below which the player page simply doesn't show a precise
  CI, before trying another transform-only fix.

---

### Hierarchical binomial GLMM for K%: tested as an alternative to WLS/LAD, consistently underperforms -- not pursued further

- **Motivation:** the PA-weighting investigation (several entries below) found the
  n0-based reliability weight under-corrects for PA_test-driven noise, and
  that weighted-LAD fitting improves on WLS. A fully hierarchical model
  (binomial likelihood on raw K/PA_test counts, logit-link regression on the
  same covariates, optional observation-level random effect for residual
  heterogeneity) was proposed as the principled alternative: instead of
  approximating "how much to trust a low-PA point" via an external weight,
  model the sampling noise directly in the likelihood.
- **Choice:** Not adopted. Tested via PyMC (NUTS, clean convergence --
  0 divergences, r_hat=1.0 throughout) across 4 variants, all underperforming
  both WLS (R2_w=0.3034) and LAD (R2_w=0.3159) on the standard held-out fold:
  - With observation-level random effect (eps_i ~ Normal(0,sigma^2)):
    R2_w=0.2993
  - Without the random effect (pure binomial-logistic regression):
    R2_w=0.2904 -- WORSE than with it, ruling out "the random effect is
    stealing signal from the fixed effects" as the explanation.
  - Capped effective binomial n (testing whether unbounded binomial
    precision -- unlike the current weight, which saturates toward 1 as
    PA->infinity -- lets a few high-PA points dominate): CAP=150/250/400 all
    gave R2_w~0.299-0.300, no improvement. Telling detail: training PA_test
    maxes out at 285, so CAP=400 doesn't bind on ANY training point yet
    performs identically to CAP=150 -- directly rules out the
    unbounded-precision hypothesis too.
- **Honest conclusion:** no confirmed explanation for the gap. Best
  remaining guess (not verified) is that the beta~Normal(0,2) priors act as
  a mild ridge-style regularizer not well-matched to this problem, but this
  wasn't tested further given four consecutive negative variants.
- **Revisit if:** a 3rd FL season gives more training data (n=134 is modest
  for a model with ~134 additional per-observation latent parameters in the
  random-effect variant); or if a specific, better-justified prior/variant
  is proposed rather than further undirected search.

### PA-stratified bootstrap + Medium/High risk tier SHIPPED, replacing the parametric K-BB% CI entirely

- **Choice (SHIPPED):** Replaced the dashboard's numeric K-BB% confidence
  interval with a two-tier risk LABEL (Medium/High, no "Low" -- confirmed
  nothing in this population is genuinely low-uncertainty), computed once
  per player during the batch build (`fl_bootstrap_risk_tier.py`,
  `compute_risk_tiers`) and stored as a `risk_tier` column -- never
  recomputed live in the dashboard.
- **Two real bugs caught and fixed during implementation, before shipping:**
  1. NaN handling: `np.where(width >= threshold, "High", "Medium")` silently
     mapped every NaN-width row (pitchers with no 2026 PA at all, hence no
     real projection) to "Medium" (NaN comparisons are False). Fixed to
     explicitly preserve NaN/None for these 299 pitchers rather than
     mislabeling them with a tier that implies a real projection exists.
  2. Metric mismatch: the risk threshold had been validated using K%'s own
     bootstrap width compared against K-BB%'s population SD (0.0715) --
     different metrics. Fixed by running K% and BB% bootstraps TOGETHER
     (same RNG seed/stream for both, so each iteration's case- and donor-
     resampling draws align, preserving the real K/BB correlation) and
     deriving width from the properly combined K_sim - BB_sim distribution.
     This revealed a real, bigger finding: once combined correctly, EVERY
     roster player's width (min 0.078) exceeds the population-SD benchmark
     (0.0715) -- an even stronger version of "nothing is low risk" than
     originally found with K% alone. An absolute threshold can't
     differentiate anyone once K and BB uncertainty are properly combined,
     so the shipped threshold uses the roster's own median width instead
     (a relative split, 249/249) -- the only one that produces a meaningful
     distinction in this population.
- **Also fixed:** the production training set (`d_tr_k`, n=134) used as the
  bootstrap's case-resampling population is almost entirely high-PA players
  (confirmed: only 1-2 donors in the lowest PA buckets), so it can't supply
  the low-PA tail with real residual variability. Fixed by sourcing residual
  donors from a separate, much richer early-2025-cutoff pool (1,352 cases
  spanning the full PA range) while still refitting coefficients on the real
  production training set.
- **Validated** (see entries below for the full detail): coverage lands
  within a few points of nominal at every PA bucket and every confidence
  level tested (50/67/80/90/95%), a genuine improvement over the parametric
  CI's uneven 0.90x-1.30x needed correction by PA bucket.
- **Dashboard changes:** removed the now-dead `ci_str`/`ci_percentile_str`/
  `_logit_ci`/`Z_90` (fully superseded, no remaining callers) from
  projections-dashboard/src/app.py; leaderboard shows a "Risk" column,
  player page shows the tier plus its width relative to the roster median.
- **Revisit if:** a 3rd FL season allows checking whether the median-based
  threshold (relative to THIS roster) stays stable, or needs to be
  re-derived each season as the roster composition shifts.

### Release point as a K% regressor (raw level AND a between-pitch-type "deception" consistency proxy): tested, no signal either way -- not shipped

- **Choice:** Do not add release point (release_pos_z, rel_side_as) to the
  K% equation, in either form tested.
- **Alternative(s) tested, both on top of the already-validated wh_sva_level
  + spin_z_norm + extension spec, same fold-1-fit/fold-2-score discipline:**
  (a) raw per-pitcher mean release_pos_z/rel_side_as -- a simple level, like
  the original spin_rate/extension tests (release point varies modestly by
  pitch type here, ~0.45-0.55ft spans, so some pitch-mix confounding exists
  but far less than spin_rate's ~1200rpm spread). (b) a deception-specific
  proxy: usage-weighted variance of each pitch TYPE's own mean release point
  ACROSS a pitcher's arsenal (how similar does the fastball's release point
  look vs. the breaking ball's) -- the actual mechanism the user raised
  ("may play into deception"), and a genuinely different construction from
  the within-pitch-type release-point variance tested earlier this session
  (found weak for BB%, a command question, not a deception one).
- **Why:** All four variants (2 level + 2 consistency dims, plus a combined
  consistency spec) came back flat: held-out K% R2_w moved by at most +-0.002
  from the 0.3044 baseline, and every new term's own coefficient had
  p>=0.73 -- a sharp contrast to spin_z_norm/extension's more modest but
  real p=0.15-0.23 that still produced genuine held-out gains. Neither the
  raw-level nor the deception-consistency framing of release point carries
  any detectable incremental signal for K% in this data.
- **Cost/risk accepted:** None -- clean negative, not shipped.
- **Revisit if:** A different target (e.g. BB%, or called-strike rate if
  ever modeled directly) might be a more natural fit for a deception
  mechanism than K% -- not tested here.

### Adding spin_rate + extension as direct K% regressors: validated held-out improvement (R2_w 0.2686->0.2979) -- role/workload proxy tested, did NOT survive -- SHIPPED

- **Choice (SHIPPED to production):** Add `spin_z_norm` (pitch-type-
  normalized spin rate -- see the refinement entry below) and `extension`
  (per-pitcher means over each fold's own feature window) as two more linear
  regressors alongside naive_K_rate/fl_k_rate_bridge/wh_sva_level in the K%
  equation. Do NOT add `median_pitches_per_game` (a starter/reliever
  workload proxy) -- tested alongside the other two and found to contribute
  nothing once properly validated.
  - Implementation: `fp.compute_spin_ext_features` (fl_pitcher_features.py)
    + `bt.stuff_levels` (fl_halfseason_backtest.py), wired into
    `assemble_features` the same way wh_sva_level/horiz_var/vert_var already
    are -- the spin-rate pitch-type baseline is recomputed fresh from each
    fold's OWN feature window (not pinned to one fold), for consistency with
    every other feature in this harness; re-verified end to end after
    wiring that this reproduces the validated result (K% R2_w=0.3034 wired
    vs. 0.3044 in the original scratch test, BB% unaffected at 0.3631).
  - `MODEL_SPECS["naive+other_league+surprise_prior"]["K"]` updated;
    `fl_production_projection.py`'s `project_from_college_stats`/
    `save_model_coefs`/`build_2027_projections` output columns and the
    dashboard's `project_prospect` (projections-dashboard/src/app.py, plain-
    arithmetic mirror for the no-FL-history lookup/manual calculator) all
    updated to carry spin_z_norm_mean/extension_mean through. Full pipeline
    re-run, outputs copied to the dashboard's data/ directory, dashboard's
    project_prospect sanity-checked against the new coefficients file.
- **Alternative(s) considered:** (a) Modify wh_sva's internal XGBoost model
  to weight spin_rate/extension more heavily, since both are already among
  its FULL_COLS inputs -- rejected as the first move: a much bigger, riskier
  change (re-deriving wh_sva would ripple through every downstream use of
  it: horiz_var/vert_var comparisons, the stickiness/n0 work, the dashboard)
  for an unproven payoff. Testing them as direct additional linear terms in
  the OUTER regression isolates the same practical question (is there
  leftover incremental signal in these two variables beyond what wh_sva's
  single compressed number captures?) far more cheaply and reversibly.
  (b) Add median_pitches_per_game (role proxy) -- motivated by a strong
  descriptive correlation (same-fold, see the error-analysis entry below)
  but didn't survive proper validation (see below).
- **Why:** This followed directly from the trait-vs-residual error analysis
  (see entry below), which found spin_rate, extension, and role each
  independently correlated with K% over/underperformance in a same-fold
  diagnostic check. Given the milb/none dummy correction's recent failure to
  generalize from an in-fold-looking pattern, each candidate was instead
  fit on fold 1 (full-2025 -> actual 1H2026) and FROZEN before scoring on
  fold 2 (2025+1H2026 -> actual 2H2026) -- the same discipline that caught
  that false positive, applied here to see which of these actually survive.
  - +role alone: held-out K% R2_w 0.2686->0.2683 (flat/slightly worse). Its
    own fold-1 coefficient is ~0 (p=0.976) -- the correlation found in the
    error analysis was fold-2-specific, same trap as the milb/none dummy.
    Dropped.
  - +spin_rate alone: 0.2686->0.2800. +extension alone: 0.2686->0.2817.
    +both together: 0.2686->0.2979 (best). Neither coefficient is
    individually significant when fit on fold 1's smaller training sample
    (n=134; spin_rate p=0.36, extension p=0.28), but unlike the role proxy,
    using them to predict FORWARD onto fold 2 genuinely improves accuracy
    rather than hurting it -- the signature of a real, if noisily-estimated,
    effect rather than overfitting (pure noise added as a regressor would on
    average cost held-out R2, not gain it).
- **Cost/risk accepted:** Individually-insignificant fold-1 coefficients
  mean the exact magnitude of these two terms is uncertain; FL's 2-season
  data limit means this is the same single held-out check every other change
  this session relies on, not a true multi-fold cross-validation. The
  DIRECTION and the held-out gain are real by the only test available, but
  should be re-checked once a 3rd season of real data exists.
- **Revisit if:** A 3rd FL season becomes available (re-validate on a fresh
  fold); or if wh_sva itself is ever re-derived, re-check whether these two
  terms are still needed as separate linear additions or get absorbed.
- **Follow-up refinement (also validated):** raw per-pitcher mean spin_rate
  blends pitch types that differ by over 1000 rpm (splitter ~1170 vs
  sweeper ~2420 in this data) -- a pitcher's crude average is driven as much
  by THEIR PITCH MIX as by how much spin they generate on a given pitch
  (flagged directly by the user). Extension varies far less by type
  (5.15-5.95 across all types), so it didn't need this fix. Built a
  pitch-type-normalized version (each pitch's spin_rate as a z-score
  relative to that SAME pitch type's own league baseline, computed from the
  fold-1 training window only to avoid leakage, then averaged per pitcher)
  and validated it the same way: held-out K% R2_w improves further,
  0.2979->0.3044, and the term's own fold-1 p-value tightens from 0.360 to
  0.149 -- a real, if still not fully "significant," improvement. Checked
  wh_sva_level's own pitcher-level correlation with raw spin_rate/extension
  (0.50 / 0.44) -- moderate, not collinear-to-the-point-of-redundant,
  consistent with these three variables (wh_sva's nonlinear per-pitch use,
  the outer linear per-season terms) carrying genuinely separable signal
  rather than double-counting the same information. Recommend
  pitch-type-normalized spin_z over the raw mean if this gets shipped.

### Error analysis: the full model systematically OVER-projects unmatched ("no bridge source") pitchers -- real, actionable bias found; not yet fixed

- **Finding (not yet a shipped change):** Ran a residual/error analysis on
  the standard Experiment 2 held-out fold using the current full model
  (naive+other_league+surprise_prior, corrected full bridge match). Grouped
  residuals (actual - predicted K-BB%) by `k_bridge_source`:
  - college (n=107): mean resid -0.007, mean z-score -0.03 -- essentially
    unbiased, the model's primary source is well-calibrated.
  - milb (n=54): mean resid +0.013, mean z +0.22 -- mild positive bias
    (slight underprediction).
  - pioneer (n=15): mean resid +0.012, mean z +0.10 -- small n, similar
    direction.
  - **none (n=84, no bridge match of any kind): mean resid -0.039, mean z
    -0.41** -- a real, systematic OVER-projection. These are not just the
    noisiest predictions (already known from lower R2 in earlier subgroup
    checks) -- they are consistently worse than predicted, not merely more
    variable around a correct center.
  - Confirmed this isn't a PA-level artifact in disguise: z-scores are
    flat (~0) across all four PA quartiles, so the model's own uncertainty
    scaling already handles low-PA noise correctly -- the bridge-source bias
    is a separate, distinct effect.
  - Checked the mechanism directly: unmatched pitchers' actual K-BB%
    (+0.0575) sits measurably below both the league average (+0.0835) and
    the model's own prediction for them (+0.0797), even though their OWN
    naive_K_rate input is unremarkable (0.1959, essentially league-average).
    This suggests "has no identifiable prior-league record" is itself a real
    signal of a weaker population (e.g. organizational filler, fringe
    international signings, players from leagues entirely outside this
    project's 3-source universe) that the model currently discards --
    unmatched pitchers get a neutral league-average bridge fallback, which
    this data says is too generous for them specifically.
- **Follow-up (tested, NOT shipped):** Added `is_milb`/`is_none` dummy
  regressors (per-metric, using k_bridge_source for the K% equation and
  bb_bridge_source for BB%) to both specs and fit them on fold 1 (full-2025
  -> actual 1H2026) -- a genuinely earlier, different period than the fold
  that surfaced the bias -- then froze and scored on fold 2 (2025+1H2026 ->
  actual 2H2026, the exact fold the finding came from), avoiding the
  circularity of fitting and testing the fix on the same data.
  - The fitted dummy coefficients came back tiny (is_none_k=-0.0004,
    is_milb_k=-0.0098, is_none_bb=-0.0019, is_milb_bb=+0.0035) because the
    bias barely exists IN-SAMPLE on fold 1: "none" source K% resid there is
    -0.0058 (vs. -0.0392 on fold 2), BB% resid -0.0091 (vs. the fold-2
    pattern). The correction had almost nothing to learn from.
  - Scored on fold 2 anyway: held-out K% R2_w dropped 0.2686->0.2553, BB%
    R2_w dropped 0.3631->0.3531 -- WORSE overall, not just flat. The bias it
    was meant to fix got WORSE too (none: -0.0392->-0.0428; milb:
    +0.0129->+0.0239), not better.
  - Checked why: the "none" population is not stable across folds -- fold 1
    has 51 unmatched pitchers, fold 2 has 84, and only 38 overlap. The
    fold-2 bias looks like it's driven by whichever SPECIFIC players happen
    to be unmatched in that half-season, not a structural property of "has
    no bridge match" as a category that a fixed intercept shift can capture.
- **Cost/risk accepted:** None -- not shipped. This is the right outcome of
  doing the validation properly: a bias that looked real and actionable in
  the original error analysis did not survive being fit on an independent
  period and tested forward, which is exactly why that discipline matters
  here.
- **Revisit if:** A later season's data lets a 3rd independent fold confirm
  whether the "none" population's composition stabilizes year over year, or
  if a within-fold (rather than cross-fold) explanation for WHY fold 2's
  specific unmatched cohort underperformed is found (e.g. checking whether
  they cluster in mid-season arrivals, a population already flagged as hard
  earlier this session).

### Injury/appearance-gap feature: validated against real IL transactions, no detectable performance effect -- not pursued further

- **Choice:** Do not build an injury or appearance-gap feature into the
  model at this time.
- **Alternative(s):** (a) Use naprobaseball.com's Frontier League
  transactions log directly as a live injury feature -- ruled out
  immediately: the page's newest entry is November 3, 2025, with zero 2026
  data (same external-staleness pattern as Pioneer League's StatsCrew
  pages), so it cannot inform current projections, only a historical 2025
  check. (b) Use a within-season GAP between a pitcher's consecutive FL
  appearances (already derivable from existing pitch-level data for any
  year, including 2026) as a proxy for injury, independent of the stale
  transactions feed.
- **Why:** Built the gap proxy and validated it against the real 2025 IL
  transactions (fetched via browser, 473 IL events parsed, 253 distinct
  players) as ground truth:
  - A MID-SEASON gap (pitcher goes quiet, then returns) of 14-21+ days is a
    moderately real injury signal: 44-48% of such gaps have a confirmed real
    IL placement within the surrounding window -- well above chance, useful
    signal in principle.
  - A LATE-SEASON gap (pitcher's last appearance is well before season end,
    never returns that season) is NOT a good injury proxy: only ~6% confirmed
    by a real 60-Day/season-ending IL placement, vs. ~11% confirmed by a
    plain RELEASE -- in an indy league, roster churn (releases) explains
    "stopped appearing" far more often than injury does, and the majority
    (>80%) of late-season disappearances are explained by neither in the
    ground-truth data (other causes -- trades to other leagues, visa/personal
    inactive-list moves -- or name-matching misses).
  - The actual payoff question: does returning from a 14+-day mid-season gap
    show a detectable K%/BB% change in the following 21 days (a "rust after
    injury" effect)? Tested directly on 189 pitchers with enough pre/post PA:
    K% dropped slightly (18.07%->17.02%, the expected direction) but NOT
    significant (paired t-test p=0.247); BB% showed no change at all
    (p=0.963).
- **Cost/risk accepted:** The mid-season gap-then-return proxy genuinely
  correlates with real injury (unlike the late-season version), so this
  isn't a dead end if revisited with a larger sample or a different outcome
  window -- but the one concrete, testable claim (performance drop after
  return) did not clear significance with the data on hand, so it isn't
  worth adding to a production model right now.
- **Revisit if:** More seasons of data accumulate (larger n for the
  pre/post comparison), or naprobaseball.com's transactions page catches up
  to 2026 (would also unblock the Pioneer League same-year bridge gap, a
  separate but similarly-blocked finding).

### Mixed-league ROS/same-year: pulled 2026 MiLB data (Pioneer blocked externally) -- confirms freshness, does NOT close the hardest-population gap

- **Choice:** Extended `pull_milb_pitching.py` to include 2026 (now
  `output/milb_pitching_2018_2026.parquet`, `fl_milb_bridge.py`'s `MILB_PATH`
  updated to match) so `build_full_universe_predictions`'s "most recent
  season on record" lookup can resolve a SAME-YEAR MiLB->FL transition for a
  player who arrives mid-season, instead of only ever matching on a stale
  (>=1-year-old) MiLB record. Did NOT attempt the same for Pioneer League --
  see below.
- **Alternative(s):** Leave both other-league sources at their 2025 ceiling
  (status quo going into this entry).
- **Why:** Confirmed directly (browser fetch of
  statscrew.com/minorbaseball/stats/t-{team}/y-2026 for two Pioneer teams)
  that StatsCrew has NOT populated any 2026 Pioneer League stats at all --
  "Team Record: 0-0", empty roster, even though the real season has ended
  (today is 2026-10-04) and the identical 2025 pages are fully populated.
  This is an external data-availability gap, not a scraping/parsing problem
  on this project's end -- Pioneer League's same-year ROS case is genuinely
  blocked until StatsCrew updates. MiLB, by contrast, is pulled from the
  public MLB Stats API, which already has full 2026 season data -- pulled
  successfully (51,863 total rows, +~8,350 new 2026 pitcher-seasons across
  levels).
  - Confirmed the MiLB pull does what it's supposed to at the population
    level: 4,716 of 13,937 distinct MiLB pitchers (34%) now resolve to a 2026
    "most recent season" instead of a stale prior year.
  - Re-ran the mid-season-arrival held-out check (Experiment 2 fold) that
    motivated this: the 35 mid-season arrivals who already had SOME
    other-league match ticked up slightly (K% R2_w 0.2520->0.2536, BB%
    0.3394->0.3504) -- a small, directionally-consistent improvement
    plausibly from fresher MiLB data for a few of them, though this fold is
    too small to call it precise.
  - The real target -- the 16 mid-season arrivals with NO match anywhere --
    was completely unaffected: checked each of their 16 names directly
    against the new 2026 MiLB data by exact name_clean match, and NONE of
    them appear in it. Their gap is not "their transition year wasn't pulled
    yet" -- it's something else entirely (a league outside this project's
    3-source universe -- American Association, Atlantic League,
    international, a non-D1 college source the FanGraphs pull doesn't cover
    -- or a genuine name-matching failure, e.g. nickname/suffix mismatch).
- **Cost/risk accepted:** Reported the negative result for the actual target
  population honestly rather than crediting the pull with a win it didn't
  produce. The MiLB freshness improvement is real and worth keeping
  (benefits future rosters' name-matching going forward, not just this
  fold), but it does not address the "team needs help predicting guys with
  less prior FL history" problem for this specific hardest subgroup.
- **Revisit if:** StatsCrew populates 2026 Pioneer League data (worth a
  periodic recheck); or if manually identifying even a few of the 16 names'
  true prior league (outside this project's current sources) turns out
  tractable.

### Mixed-league prior-year bridge: TBF-weighted multi-source blending tested, does not beat the existing pick-one-by-validated-R2 rule -- not shipped

- **Choice:** Keep `extend_match_by_other_leagues`'s existing rule for a player
  who matches MORE than one other-league source in the same prior year
  (college + Pioneer, Pioneer + MiLB, etc. -- 64 of 738 current roster
  players, 8.7%): use the single source with the highest already-validated
  real R2 for that specific metric (K%: college > Pioneer 0.096 > MiLB 0.037;
  BB%: college > MiLB 0.198 > Pioneer 0.096), never a blend.
- **Alternative(s):** Pool the multiple matched sources' bridge-predicted
  rates into one TBF-weighted average (weight = each source's own TBF in its
  matched season) -- the natural "a player who split last year across two
  leagues should have both stints inform the prior" construction.
- **Why:** Built the TBF-weighted blend and scored it head-to-head against
  the current pick-one rule on the real Experiment 2 held-out fold, both
  overall and specifically on the n=23 test pitchers where the two approaches
  can actually differ (n_sources>=2): blending was WORSE for K% (R2_w
  0.2499->0.2137 on the multi-source subgroup, 0.2690->0.2586 overall) and a
  wash for BB% (0.4204->0.4247 multi-source, RMSE_w 0.0473->0.0471 -- not a
  real gain). TBF (playing time) and predictive quality are different
  things -- blending by TBF injects a worse-calibrated source's noise into
  what would otherwise be the single best-validated prediction, undoing the
  whole point of the per-metric priority order chosen in the first place.
- **Cost/risk accepted:** n=23 on the affected subgroup is small; this result
  shouldn't be treated as precise, only directionally real (consistently
  non-positive across both the subgroup and the overall population, on both
  metrics).
- **Revisit if:** A future source's bridge R2 improves enough to make a
  quality-weighted (not TBF-weighted) blend plausible -- not tested here.

### Backtest harness only joined the narrow college bridge table -- silently understated every "naive+other_league" result reported this session; fixed

- **Choice:** `fl_halfseason_backtest.py`'s `bridge_features()` now runs the
  SAME three-tier match production ships (narrow bridge-fitting table ->
  name-extended college, any season on record -> Pioneer/MiLB), via a new
  shared `fl_bridge_matching.py` module imported by both
  `fl_production_projection.py` and `fl_halfseason_backtest.py` (previously
  duplicated code was quietly diverging).
- **Alternative(s):** Leave the backtest harness as-is (narrow join only,
  ~152 true transition pairs by exact pitcher_id) since it was the
  originally-approved design.
- **Why:** Discovered while investigating mid-season-arrival performance:
  the harness's `has_college_match` reflected only the bridge's own fitting
  sample, never `extend_match_by_name`/`extend_match_by_other_leagues` or the
  Pioneer/MiLB bridges -- all three of which are live in production. Real
  match rate on the Experiment 2 test set jumped from ~26% to 68% once fixed.
  Re-ran the standard held-out check under the corrected harness: naive-only
  K% R2_w=0.228/BB%=0.312 -> +other_league 0.264/0.334 -> full
  (+surprise_prior) 0.268/0.363 -- the full model's advantage over naive-only
  survives and is still real under the TRUE match rate, just measured
  correctly now instead of on an artificially narrow subset.
- **Cost/risk accepted:** Every "naive+other_league..." number reported
  earlier in this session (coefficient drift, stickiness-vs-K%-takeover,
  platoon/offspeed checks, mid-season-arrival splits before this fix) used
  the narrow-only match and should be treated as a lower bound on what the
  full model can see, not wrong in direction but understated in bridge
  coverage.
- **Revisit if:** A new other-league source is added -- extend
  `fl_bridge_matching.full_match` rather than re-duplicating the chain.

### Park-adjusting K%/BB% targets with Supabase's park_factors_fip: tested, does not improve held-out prediction -- not shipped

- **Choice:** Keep naive_K_rate/naive_BB_rate (and the actual-rate targets they're
  trained against) computed from raw, unadjusted K/BB counts -- do NOT
  park-adjust using Supabase's existing `park_factors_fip` table (per-team
  SO%/BB% factors, already computed from real FL play-by-play, not derived
  by this project).
- **Alternative(s):** Divide each terminal PA's K/BB credit by that game's
  park factor (via `home_team` -> `park_factors_fip`, joined at the per-game
  level so mid-season team-switchers are naturally handled correctly without
  special-casing) before aggregating into a rate, then use that park-neutral
  rate as the naive-prior predictor.
- **Why:** Built `fp.load_park_factors`/`fp.attach_park_factors`/
  `fp.build_period_kbb_park_adjusted` (per-game join, not a season-level
  approximation) and tested the real question directly: does a TRAIN-year
  park-adjusted rate predict the pitcher's actual FUTURE (still raw, since
  that's the real-world quantity being projected) rate better than the raw
  train-year rate, across both Experiment-2 folds?
  - K%: inconsistent -- better in one fold (0.422->0.439), worse in the other
    (0.387->0.386, essentially flat).
  - BB%: WORSE in both folds (0.509->0.505, 0.450->0.445) -- small but
    consistently in the wrong direction.
  - Checked the population where the theoretical case is strongest -- team-
    switchers (different home park mid-season, not just the usual home/away
    blend everyone gets): BB% was STILL worse for switchers in both folds
    (no exception). K% showed one positive result (n=33, the larger/more
    trustworthy switcher sample) and one negative (n=11, too small to read
    much into either way) -- not a repeated, confirming pattern.
  - Plausible reasons this didn't work even though the theory is sound:
    each park's factor is itself estimated from a fairly small sample (one
    team's home games over at most 1.5 seasons, and SO/BB rates are
    inherently noisier to estimate a park effect for than HR/runs, which is
    what park factors are usually built and validated against); and players
    already face a mix of many different parks across a season (their own
    home park plus every opponent's park on the road), which partially
    self-cancels park variance even without explicit adjustment.
- **Cost/risk accepted:** None -- not shipped. The reusable park-factor
  join/adjustment functions are kept in `fl_pitcher_features.py` for future
  use (e.g. if a sturdier park-factor source becomes available, or if this
  is revisited for a different target like HR-dependent metrics where park
  effects are typically much stronger).
- **Revisit if:** A metric more sensitive to park dimensions (e.g. a future
  HR-rate or extra-base-hit model) needs this same machinery -- the
  infrastructure built here should transfer, even though it didn't pay off
  for K%/BB% specifically.

---

### Naive-prior recency-decay sweep (gentler tapers, 3-fold evaluation): reconfirms the already-shipped most-recent-year-only design -- no change

- **Choice:** Keep the production naive K%/BB% prior as-is (uses ONLY the
  single most recent trailing year, already shipped per the "Flat-pool
  window size" entry below) -- do not switch to any intermediate recency
  taper.
- **Alternative(s):** A gentler decay (weight each trailing year by
  `decay**lag` instead of a hard window cutoff) might recover some of the
  2-3-year-old signal flat-pooling or the discarded classic-Marcel weighting
  each lost, per that entry's own "Revisit if."
- **Why:** `fl_recency_weight_test.py` swept decay in {0.0...1.0} (0.0 =
  already-shipped most-recent-year-only, 1.0 = flat pool, i.e. the
  already-rejected alternative) across THREE full-season folds instead of
  the original two -- added a "trailing 2022-23 StatsCrew -> actual 2024
  StatsCrew" fold on top of the existing val2025/test2026 checks, reusing
  already-pulled data, no new scraping, for a more robust/larger-sample
  comparison.
  - **K%**: clean and monotonic on all 3 folds -- more aggressive recency-
    weighting always wins, all the way to the already-shipped extreme (mean
    R2_w across folds: 0.212 flat -> 0.251 at decay=0.0). Confirms the
    existing design is still correct, now on 3 independently-confirming
    folds instead of 2.
  - **BB%**: noisier, non-monotonic -- decay=0.0 scores best in the
    aggregate (0.273) but decay=0.2-0.4 actually UNDERPERFORM flat pooling
    (0.253-0.260 vs 0.263), and decay=0.6-1.0 are statistically
    indistinguishable from each other. Checked for a selection-effect
    confound (decay=0.0 drops pitchers with no most-recent-year data point,
    since their prediction becomes undefined): real but modest, ~7-10% of
    the evaluation sample on the 2026 fold specifically (not the much larger
    ~60% attrition seen in the full historical pool, most of which is
    unrelated players who simply didn't pitch in the target year at all).
  - Net read: no decay value beats the already-shipped most-recent-year-only
    design on either metric across this larger sample -- a genuine
    re-validation, not just "nothing changed because nothing was tested."
- **Cost/risk accepted:** None -- no production change.
- **Revisit if:** FL accumulates enough seasons that most players carry 3+
  real trailing years (the same precondition flagged in "Marcel-weighted-
  with-reliability FL prior" below) -- re-sweep then, since a short-history
  league and a mature one could have genuinely different right answers here.

---

### Recency-weighted multi-year BB% bridge input: helps college, hurts MiLB, inconclusive for Pioneer -- college-only, SHIPPED

- **Choice:** `fl_college_bridge.py`'s `bb_pct_multi` (the TBF-weighted
  multi-year average used as the BB% bridge input) now ALSO weights by
  recency: `weight = tbf * 0.5**lag` (lag = years before the player's own
  most recent matched season), not TBF alone. `fl_pioneer_bridge.py`/
  `fl_milb_bridge.py` are UNCHANGED -- this was tested per-bridge and only
  validated for college.
- **Alternative(s):** Apply the same recency-decay to Pioneer/MiLB too (for
  consistency); use a different decay value; leave all three bridges on
  flat TBF-only averaging.
- **Why:** The existing design treats a 3-year-old season exactly the same
  as last year's, TBF for TBF -- never actually tested whether recency
  should matter on top of sample size. Swept decay in {0.0...1.0} on each
  bridge's own real transition-pair sample (`fl_bridge_recency_test.py`,
  same in-sample WLS-R2 methodology already used to validate TBF-weighting
  itself in "Multi-year college input"), then fine-swept the winner:
  - **College (n=152, 115 with >1 trailing year)**: real, non-trivial peak
    around decay=0.5 -- R2 0.2461 (flat) -> 0.2604 (decay=0.5), corr 0.496 ->
    0.510. NOT monotonic toward single-year-only either -- decay=0.0 (pure
    most-recent-season) was WORSE than flat (R2=0.2138), so this is
    genuinely "moderate recency-weighting," not a disguised single-year
    finding.
  - **MiLB (n=194, 158 with >1 trailing year)**: monotonically WORSE as decay
    decreases -- flat (R2=0.1081) is already optimal, recency-weighting
    actively hurts (down to R2=0.0644 at decay=0.0). Opposite direction from
    college.
  - **Pioneer (n=68, only 14 with >1 trailing year)**: no meaningful
    movement either way (R2 0.0189 -> 0.0206 across the full decay range) --
    too few multi-year players to say anything with confidence. Left
    unchanged rather than acting on noise.
  - Re-ran the full held-out Experiment-2 check with the college change
    applied (`fl_other_league_bridge_test.py`) to confirm it propagates
    cleanly: BB% R2_w improved 0.3274 -> 0.3445 (narrow college-only bridge)
    and 0.3509 -> 0.3631 (widened college+Pioneer+MiLB bridge) -- both
    variants improved by roughly the same margin, no regression anywhere
    downstream.
  - Regenerated the full production pipeline (`fl_production_projection.py`):
    refit college bridge coefficients, 2027 roster projections, dashboard
    coefs JSON.
- **Cost/risk accepted:** The college/MiLB/Pioneer split is a real,
  data-driven difference, not an inconsistency to "clean up" -- resisting the
  urge to apply one decay value everywhere for the sake of uniformity.
- **Revisit if:** More Pioneer years accumulate and enough players cross the
  >1-trailing-year threshold to give that sweep real statistical power; or if
  college/MiLB's own underlying age-of-data distribution shifts enough to
  move either optimum.

---

### Location variance (horiz_var/vert_var) replaces horiz_dist/vert_dist in the BB% model -- genuine held-out win, SHIPPED

- **Choice:** Production BB% model now uses `horiz_var`/`vert_var` (location
  VARIANCE -- command consistency/repeatability, `fl_pitcher_features.
  compute_loc_var`) in place of `horiz_dist`/`vert_dist` (mean distance from
  zone center -- command accuracy/aim). Changed in
  `fl_halfseason_backtest.MODEL_SPECS["naive+other_league+surprise_prior"]`
  (the live production spec) only -- other `MODEL_SPECS` entries (the
  changepoint variants, the plain `naive+surprise_prior` baseline) still use
  `horiz_dist`/`vert_dist` unchanged, since this swap was validated only for
  this one spec.
- **Alternative(s):** Keep `horiz_dist`/`vert_dist`; add `horiz_var`/`vert_var`
  ALONGSIDE them instead of replacing; fold them into the naive-prior
  shrinkage target (the "informed prior" mechanism from the earlier
  K%/wh_sva_level win).
- **Why:** `horiz_var`/`vert_var` beat `horiz_dist`/`vert_dist` on every axis
  checked before touching the model at all: held-out corr with future BB_rate
  (0.274/0.365 vs. 0.214/0.310), comparable-or-better stickiness across three
  different splits (~0.74-0.78 vs. ~0.70-0.77) -- while being highly
  correlated with them (r=0.85/0.87), reading as a cleaner measurement of the
  same underlying command trait, not an independent one. That motivated
  testing a REPLACEMENT rather than an addition.
  - **Four variants tested** (`fl_locvar_test.py`, same held-out Experiment-2
    check as everything else): A (current production, `horiz_dist`+
    `vert_dist`) R2_w=0.3274; B (plain replacement, `horiz_var`+`vert_var`)
    R2_w=**0.3407**; C (informed prior via var, no separate terms)
    R2_w=0.3251 (worse than A); D (informed prior + separate var terms)
    R2_w=0.3289 (better than C, still worse than B).
  - **Plain replacement wins cleanly; informed prior does NOT help here
    either** -- reinforces the earlier "Informed-prior shrinkage" finding
    that BB% has two related-but-distinct command signals, and compressing
    them into one scalar prior-target loses information the outer regression
    can use better on its own, true even with these objectively stronger
    input variables, not just the weaker distance-based ones tested before.
  - Individual coefficients in variant B aren't significant on their own
    (horiz_var p=0.218, vert_var p=0.477, same pattern `fl_bb_rate_bridge`
    already shows) -- the held-out R2_w improvement is the deciding metric
    here, consistent with how every other change this session was judged.
  - Regenerated the full production pipeline (`fl_production_projection.py`):
    frozen BB% coefficients (`horiz_var`=0.0315, `vert_var`=0.0155), 2027
    roster projections, and the dashboard's model-coefs JSON. Dashboard
    (`app.py`) relabeled "Horizontal/Vertical Command Score" ->
    "Horizontal/Vertical Command Consistency" to match what's actually being
    displayed now, and `project_from_college_stats`/`save_model_coefs`
    updated to the new field names throughout.
- **Cost/risk accepted:** `horiz_var`/`vert_var` are a less intuitive display
  metric than a plain "feet from center" distance (variance is in squared
  units, harder to eyeball) -- accepted as a minor UX cost for a genuine
  accuracy win. The two variables are highly collinear with the OLD
  `horiz_dist`/`vert_dist` (r=0.85/0.87), so this is a close substitution,
  not a wholesale change in what the model is "about."
- **Revisit if:** A future location-based candidate beats `horiz_var`/
  `vert_var` the same way these beat `horiz_dist`/`vert_dist` -- worth
  re-running this same replacement-vs-addition-vs-informed-prior comparison
  rather than assuming the current winner is final.

---

### Fastball Zone% as a direct BB% regressor: real standalone signal, redundant once added to the model -- NOT shipped

- **Choice:** Keep the production BB% model as `naive_BB_rate + fl_bb_rate_bridge
  + horiz_dist + vert_dist` -- do NOT add `fb_zone_pct` (fraction of four-seam/
  sinker fastballs thrown in the strike zone, `fl_pitcher_features.
  compute_fb_zone_pct`) as a fifth regressor, despite it looking like a strong
  candidate on its own.
- **Alternative(s):** Add `fb_zone_pct` to `fl_halfseason_backtest.MODEL_SPECS`
  and wire it into production.
- **Why:** Explored via correlation screen first, then the standard held-out
  check, in that order -- each step caught something the previous one missed:
  - **Concurrent corr with BB_rate = -0.60** (same window) -- but this is
    largely circular: Zone% and BB_rate are measured over the SAME pitches in
    the SAME window, so the relationship is partly mechanical, not predictive.
  - **Held-out corr (train-2025 Zone% -> actual 1H2026 BB_rate) = -0.415** --
    survives the circularity check, genuinely predictive of FUTURE BB_rate,
    and stronger than `horiz_dist`/`vert_dist`'s own *concurrent* corr (~0.38).
  - **Stickiness**: self-corr only ~0.39-0.45 across splits (1H/2H2025,
    1H/2H2026, 2025->2026), consistently WEAKER than `BB_rate`'s own
    self-corr (~0.50-0.55) -- a noisier metric than the thing it's predicting.
  - **n0 (empirical-Bayes stabilization) = 157 fastballs (~77 PA-equivalent)**
    -- needs ~35% more playing time than `BB_rate` itself (n0_bb=57 PA) to
    reach the same signal/noise split.
  - **Held-out model test (the real bar)**: adding `fb_zone_pct` to the full
    BB% spec and re-running Experiment 2 -- R2_w 0.3274 -> 0.3216 (WORSE),
    coefficient p=0.380 (not significant). Moderately correlated with
    `horiz_dist`/`vert_dist` (-0.38/-0.46) and more fundamentally redundant
    with `naive_BB_rate` -- a pitcher's own actual walk history already
    captures most of what Zone% describes, and what's left over is mostly
    Zone%'s own extra noise (slower-stabilizing n0) rather than new signal.
- **Cost/risk accepted:** None -- not shipped. Logged so this isn't
  re-explored from scratch later without the context of why the standalone
  correlation didn't survive being added to the full model.
- **Revisit if:** A future model restructuring drops or changes
  `naive_BB_rate`/`horiz_dist`/`vert_dist` in a way that removes the overlap
  this finding is attributed to -- Zone%'s standalone predictive signal is
  real, it just isn't ADDITIVE on top of what's already there.

---

### Pioneer League + affiliated MiLB bridge: widens "other league" coverage 270->474/797, genuine (if asymmetric) held-out win -- SHIPPED

- **Choice:** Extend the "other league prior" bridge beyond college to also
  cover the Pioneer League (`fl_pioneer_bridge.py`) and affiliated MiLB
  (`fl_milb_bridge.py`), wired into `fl_production_projection.py` via a new
  `extend_match_by_other_leagues` step that fills `fl_k_rate_bridge`/
  `fl_bb_rate_bridge` for any roster player the college match (narrow + name
  extension) still misses -- never overwriting a real college match.
- **Alternative(s):** Leave the bridge college-only (270/797 matched); use
  FanGraphs instead of the MLB Stats API for MiLB (user explicitly chose the
  Stats API, already pulled, no new fetch needed); blend all three sources by
  some weighted average rather than a strict per-metric priority order.
- **Why:** Pulled Pioneer League pitching from StatsCrew.com (10 modern-era
  teams, 2023-2025, 819 pitcher-seasons -- same page structure, same BF/HBP
  year-dependent-availability quirk already handled for FRON). MiLB needed no
  new pull (`pull_milb_pitching.py` already had 45,071 rows via the official
  MLB Stats API, 2018-2025). Built REAL Pioneer-year-Y -> FL-year-Y+1 and
  MiLB-year-Y -> FL-year-Y+1 transition pairs by name match, restricted to
  genuine FL DEBUTS (no FL PA in an earlier checked year) -- without that
  restriction a returning FL veteran gets counted again against a stale
  other-league season and the fit collapses to near-zero (confirmed directly:
  unrestricted Pioneer fit was K% R2=0.001, BB% R2=0.034).
  - Each source fit independently via PA-weighted WLS (not plain OLS like the
    college bridge) -- confirmed directly that plain OLS lets tiny-PA FL-debut
    cameos (as low as 1 PA) dominate the fit (unweighted Pioneer corr:
    K%=-0.09, BB%=0.14), while a PA>=50 subset alone showed real correlation
    (K%=0.37, BB%=0.39); PA-weighting is the standard fix since rate-estimate
    noise scales ~1/PA (the same justification `mlb_marcel`'s reliability
    weighting already uses elsewhere in this project).
  - **Real validated R2 per source** (PA-weighted, n=68 Pioneer / n=194 MiLB
    transition pairs): College (unweighted, for reference) K%=0.128,
    BB%=0.246. Pioneer K%=0.096, BB%=0.096. MiLB K%=0.037, BB%=0.198. Pioneer
    and MiLB each turned out WEAKER than college, contrary to the plan's
    "closer caliber = maybe more predictive" hypothesis -- let the real pairs
    decide, they said no.
  - **Each source is better on a DIFFERENT metric** -- MiLB's walk-rate signal
    transfers much better than its strikeout-rate signal; Pioneer is
    mediocre-but-even on both. Combining rule is therefore a PER-METRIC
    priority order by each source's own validated R2, not one "pick the best
    overall source" rule: K% tries college, then Pioneer, then MiLB; BB% tries
    college, then MiLB, then Pioneer.
  - **Match coverage**: 270/797 -> 474/797 roster players (+204, +76%
    relative). MiLB alone added 169 new players, Pioneer 52 (with overlap).
  - **Held-out re-validation (Experiment 2: fit full-2025 -> actual 1H2026,
    freeze, score 2025+1H2026 -> actual 2H2026)**: K% R2_w 0.2580 -> 0.2670;
    BB% R2_w 0.3274 -> 0.3509. Both improve, neither regresses.
  - **Bonus finding**: the earlier-logged matched/unmatched performance gap
    (see below) nearly closes for BB% with the richer bridge (matched
    R2_w=0.348 vs unmatched R2_w=0.349 -- statistically a wash) but persists
    for K% (matched 0.273 vs unmatched 0.194) -- consistent with that gap
    being partly a real coverage-quality issue fixable by widening the bridge
    (which this change does) and partly something else for K% specifically
    (a selection effect in who matches at all, not just how good the matched
    prediction is).
- **Cost/risk accepted:** Pioneer's own contribution is modest (R2~0.10 on
  both metrics) and rests on a smaller, noisier sample (n=68) than college or
  MiLB; the debut-only transition-pair construction is a reasonable but not
  foolproof heuristic (a pitcher who played Pioneer ball, sat out, then
  debuted in FL after multiple years away would still be included -- not
  checked for separately). The per-metric priority combining rule is a
  simple, defensible default, not the only reasonable one -- an R2-weighted
  blend was considered and deferred as unnecessary complexity given the
  validated held-out win already achieved with the simpler rule.
- **Revisit if:** More Pioneer/MiLB years get pulled (would grow n and could
  change the per-source R2 ranking, especially Pioneer's small n=68); a
  future season makes the "genuine debut" heuristic miss more real
  transitions (e.g. a wave of Pioneer veterans debuting in FL after a gap
  year).

---

### Marcel-weighted-with-reliability FL prior: tested on real data, does NOT beat flat pooling -- not shipped

- **Choice:** Keep the current production "naive prior" as-is (flat, equal-weighted
  pooling of whatever trailing years are available, single shrinkage) -- do NOT
  switch to the new Marcel-recency x reliability-weighted prior
  (`mlb_marcel.marcel_reliability_prior`), despite building and validating it as
  requested.
- **Alternative(s):** Ship the new Marcel+reliability prior into
  `fl_production_projection.py`, replacing the flat-pool naive prior.
- **Why:** Pulled real StatsCrew.com box-score K/BB/BF for every FRON team, 2022-2024
  (1,192 team-stint rows -> 1,144 pitcher-seasons after summing mid-season-trade
  stints together, confirmed directly on 45 real traded-pitcher cases, e.g.
  David Gauthier's 2022 across 3 teams). Built the new prior (classic 5/4/3 Marcel
  recency weights, each season's contribution ALSO scaled by its own reliability
  `PA/(PA+n0)`, applied as a weighted AVERAGE OF RATES rather than classic Marcel's
  weighted pooling of raw counts) and validated it honestly:
  - **Validation (trailing 2022-2024 -> actual 2025)**: K% flat-pool R2_w=0.220 vs
    new method R2_w=0.037. BB% flat-pool R2_w=0.257 vs new R2_w=0.027. Flat pooling
    wins by a wide margin on BOTH metrics.
  - **Test (trailing 2023-2024+2025 -> actual 2026, the real held-out check)**: K%
    flat-pool R2_w=0.237 vs new R2_w=0.039. BB% flat-pool R2_w=0.285 vs new
    R2_w=0.004. Same result, confirmed out-of-sample -- not a validation-only fluke.
  - **Root-caused, not just reported as a loss**: re-ran with the reliability
    multiplier REMOVED (recency weights only, classic-Marcel-style) -- K% R2_w=0.054,
    still far below flat-pool's 0.220. The reliability term isn't the problem;
    Marcel's core RECENCY assumption is. Only 19 of 139 matched validation players
    (14%) have all 3 trailing StatsCrew years; most have just one. For the minority
    who DO have multiple years, downweighting the older ones throws away real signal
    that flat-pooling correctly used -- consistent with FL/indy-ball K%/BB% being a
    fairly STABLE year-over-year skill for this population (often older,
    already-established players, not rapidly-developing prospects), unlike the MLB
    population Marcel's recency weighting was originally built and validated for.
- **Cost/risk accepted:** None -- explicitly not shipped, so no production risk.
  The StatsCrew pull itself (`fl_statscrew_pull.py`,
  `output/fl_statscrew_pitching_2022_2024.parquet`) and the reusable
  `mlb_marcel.marcel_reliability_prior` function are kept for future use (e.g. if
  more years accumulate and the 1-trailing-year-only majority shrinks).
- **Revisit if:** More FL seasons pass and most players accumulate 3+ real trailing
  years (not just the current 14%) -- worth re-testing whether recency weighting
  earns its keep once there's enough multi-year population to judge it on, and
  whether a GENTLER recency taper (e.g. all trailing years weighted equally,
  dropping only the oldest) does better than full classic 5/4/3 for this specific
  population.

---

### Flat-pool window size: 1 trailing year clearly beats 2 or 3 -- a real production opportunity, not yet applied

- **Choice (finding, not yet shipped):** Using ONLY the single most recent trailing
  year as the flat-pool naive prior beats pooling 2 or 3 years, consistently, on
  both the K%/BB% metrics and on both the validation (2025) and held-out test
  (2026) years. Production currently pools whatever years are passed in (e.g. the
  2027 projection pools 2025+2026 together, a 2-year window) -- this finding says
  that's worse than using 2026 alone.
- **Alternative(s):** Keep pooling every available trailing year (current
  production default, implicitly a growing window as more seasons accumulate).
- **Why:** Swept window=1/2/3 with `fl_flatpool_tuning.py`, same flat-pool
  arithmetic, same n0/lg_rate, same StatsCrew+Supabase data as the Marcel
  comparison above. Results (R2_w):

  | window | K% val2025 | BB% val2025 | K% test2026 | BB% test2026 |
  |---|---|---|---|---|
  | 1 year | 0.247 | 0.271 | 0.286 | 0.297 |
  | 2 years | 0.216 | 0.234 | 0.236 | 0.279 |
  | 3 years | 0.200 | 0.249 | 0.230 | 0.273 |

  Monotonic in the same direction on all 4 checks -- not a fluke of one metric or
  one year. Consistent with (and sharper than) the Marcel finding above: it's not
  just that RECENCY-WEIGHTING old data hurts, pooling old data AT ALL (even at
  equal weight) hurts. Plausible reading: FL/indy-ball rosters turn over fast and
  a player's role/stuff/control can genuinely shift year to year (unlike the
  "stable skill" reading floated in the Marcel entry above, which this result
  revises) -- so last year's rate is simply a better predictor than blending in
  two-year-old data, even with no extra penalty for age.
- **Status: APPLIED.** `fl_production_projection.py`'s `build_2027_projections`
  now calls `bt.naive_prior(raw, [(bt.START_2026, bt.END_2026)])` for
  `naive_K_rate`/`naive_BB_rate`/`PA` specifically (2026-only), confirmed
  directly in the current code -- this entry previously said "not yet
  applied," which went stale once the switch was made without updating this
  log. Everything else in the feature build (bridge-match population,
  `wh_sva_level`/`horiz_var`/`vert_var`, team lookup) deliberately keeps the
  full 2025+2026 pooled window, since that wasn't what this test covered.
- **Revisit if:** More seasons accumulate -- re-sweep window size periodically,
  since the right window could change as the league's roster-turnover pattern or
  data volume changes.

---

### Folding the college-bridge rate into the SAME flat pool: does not beat keeping it as a separate regressor

- **Choice:** Keep the college/other-league bridge rate as a separate additive
  regressor in the combined projection model (current production design:
  `naive_K_rate + fl_k_rate_bridge + wh_sva_level`) -- do NOT fold it into the
  naive prior's own pooled K/PA counts as a pseudo-season.
- **Alternative(s):** Convert the bridge-predicted rate to pseudo K/PA counts
  (via `fl_college_bridge.bridge_pseudo_pa`, which backs out an "equivalent
  sample size" from the bridge's own R2) and pool those counts directly alongside
  the real trailing-year counts before shrinking -- tested directly.
- **Why:** The bridge's pseudo-PA is small (K=8.1, BB=21.7 -- backed out from its
  modest R2 of 0.13/0.25), so even for players with a college match, the pooled
  rate ends up dominated by shrinkage toward the league average rather than the
  college signal. On the college-matched subset specifically: K% R2_w = 0.070
  (val2025) / 0.120 (test2026), BB% R2_w = 0.146 (val2025) / 0.071 (test2026) --
  clearly worse than the unrestricted FL-years-only baseline (0.20-0.27 across the
  same checks). Folding it into the pool doesn't let the model learn a DIFFERENT
  (and here, apparently more useful) coefficient on the bridge rate than on a
  pitcher's own recent FL rate, which a separate-regressor WLS can do and
  evidently needs to.
- **Cost/risk accepted:** None -- confirms, rather than changes, the existing
  production design.
- **Revisit if:** The bridge's own R2 improves substantially (e.g. from a richer
  college feature set), which would raise its pseudo-PA and could change this
  conclusion.

---

### BF/G as a direct role/usage regressor: adds nothing once wh_sva/bridge/naive are already in the model

- **Choice:** Do not add BF/G (batters faced per game -- high for starters, low
  for relievers) as a regressor in the production K%/BB% model.
- **Alternative(s):** Add it alongside the existing
  `naive+other_league+surprise_prior` regressors, following up on two earlier
  signals that pointed this direction: the error-analysis finding that
  low-training-PA pitchers underperformed their naive-only prediction (Finding 2,
  the "'everyone's college rate looks the same' error analysis" entry), and the
  separate FL precedent that BF/G was the single strongest lever for predicting
  `wh_sva` itself.
- **Why:** Tested directly via the same genuinely held-out check as Experiment 2
  (fit on full-2025 -> actual 1H2026, frozen, scored on 2025+1H2026 -> actual
  2H2026). Adding `bf_per_g` to the already-validated model moved K% R2_w from
  0.2580 to 0.2585 and BB% from 0.3274 to 0.3276 -- noise, not signal -- and its
  coefficient isn't significant in either model (K% p=0.94, BB% p=0.61). Reconciled
  with the two earlier signals rather than contradicting them: the error-analysis
  finding was about the plain NAIVE-only baseline, which doesn't include
  `wh_sva_level`/bridge rate; the BF/G precedent was about predicting `wh_sva`
  itself, not K%/BB% directly. Once `wh_sva_level` is already a regressor (which
  it is, in the model actually used here), it likely already carries most of the
  usage-correlated signal BF/G would have added -- there's no contradiction, just
  two different modeling contexts where the same underlying signal enters through
  a different variable.
- **Cost/risk accepted:** None -- not shipped.
- **Revisit if:** A future model drops `wh_sva_level` (e.g. for a simpler/faster
  variant) -- BF/G might earn its keep there since the usage signal it carries
  wouldn't already be absorbed by the shape-value feature.

---

### Surprise-variable x unreliability interaction: not detectably true with current sample size

- **Choice:** Keep `wh_sva_level`/`horiz_dist`/`vert_dist` as plain additive terms
  with fixed coefficients -- do not add an interaction letting their influence
  grow for pitchers with an unreliable (small-sample) naive prior, despite that
  being the original motivation for calling them "sticky small-sample metrics"
  back at the start of this project.
- **Alternative(s):** Add `surprise_var x (1 - reliability)` interaction terms,
  where reliability = PA/(PA+n0) is the same quantity already shrinking the naive
  prior -- large for a thin-sample pitcher, near 0 for an established one.
- **Why:** User asked directly whether the current (plain, fixed-coefficient)
  model actually optimizes for this. It doesn't -- tested whether it SHOULD, via
  the same held-out Experiment 2 check. Adding the three interaction terms moved
  K% R2_w from 0.2580 to 0.2615 and BB% from 0.3274 to 0.3273 -- negligible, and
  none of the three interaction coefficients are significant (wh_sva p=0.43,
  horiz_dist p=0.68, vert_dist p=0.61). Framed honestly: NOT "proven false" --
  the interaction coefficients' standard errors are wide relative to their point
  estimates (e.g. wh_sva's interaction coef=0.79, SE=1.0), so this is "not
  detectably true with the current ~260-pitcher held-out sample," not a
  confident null.
- **Cost/risk accepted:** None -- not shipped. The added model complexity isn't
  justified without evidence it earns its keep.
- **Revisit if:** The roster population grows substantially (more power to
  estimate a second-order interaction term precisely), or the base model's own
  R2 improves enough that remaining unexplained variance concentrates
  specifically in the low-reliability region, making an interaction more
  detectable.

---

### Informed-prior shrinkage: genuine win for K%, hurts BB% -- asymmetric, not yet shipped

- **Choice (finding, pending user go-ahead to ship):** For K% only, shrink the
  naive prior toward a wh_sva-PREDICTED target (`0.198 + 1.06*wh_sva_level`,
  fit once on the training window) instead of the flat league average, and DROP
  `wh_sva_level` as a separate regressor in the outer model (it's now inside the
  prior). Keep BB%'s structure unchanged (flat-mean-shrunk naive prior,
  `horiz_dist`/`vert_dist` as separate additive terms, as today).
- **Alternative(s):** Fold `horiz_dist`/`vert_dist` into the BB% prior the same
  way (tested, hurts); keep both metrics on the current flat-prior design
  (status quo); fold in AND keep the separate term for either metric (tested,
  also worse than the simple fold-in for K%, worse than status quo for BB%).
- **Why:** Follow-up to the interaction-term test above -- instead of an
  explicit `surprise_var x (1-reliability)` term, shrink toward a surprise-
  variable-informed target directly, so a low-PA pitcher's prediction leans on
  their wh_sva/dist naturally (shrinkage weight `n0/(PA+n0)` is large exactly
  when PA is small) without a bolted-on second-order term. Tested via the same
  held-out Experiment 2 check, three variants compared to the current design:

  | model | K% R2_w | BB% R2_w |
  |---|---|---|
  | A: current (flat prior + bridge + wh_sva/dist separate) | 0.2580 | 0.3274 |
  | B: informed prior, no separate surprise term | 0.2684 | 0.3186 |
  | C: informed prior + separate surprise term (both) | 0.2625 | 0.3160 |

  K%: B clearly wins over A (+0.0104) AND is simpler (one fewer regressor).
  BB%: B and C are both worse than A -- folding `horiz_dist`+`vert_dist` into
  one scalar shrinkage target loses information the outer regression could
  otherwise use directly, plausibly because their relationship to BB% is
  weaker/noisier (~0.38 corr) than wh_sva's to K% (~0.50 corr), so compressing
  two noisier signals into a single fitted target adds more estimation error
  than it saves.
- **Cost/risk accepted:** Not yet applied to production -- asymmetric designs
  (different structure per metric) are a little more code to maintain; worth
  it only if the user confirms they want the K% half shipped.
- **Revisit if:** The BB% surprise variables' own predictive power improves
  (e.g. a cleaner single combined metric instead of two separate ones), which
  might make the same informed-prior trick work there too.

---

### Folding the college bridge into the K% informed prior: no robustness gain, actively worse for matched players

- **Choice:** Keep `fl_k_rate_bridge` as its own separate outer regressor in the
  K% model (variant B above) -- do not also fold it into the wh_sva-informed
  shrinkage target.
- **Alternative(s):** Fit `informed_K_target = a + b*wh_sva + c*bridge_K%` (both
  sources blended into one shrinkage target) and drop the separate bridge term
  entirely (variant D).
- **Why:** User asked directly whether folding the bridge in would make the
  model more robust for the 66% of the roster with no college match. Tested by
  comparing variant B vs. D split by `has_college_match`, same held-out
  Experiment 2 check:

  | | unmatched (n=193) | matched (n=67) |
  |---|---|---|
  | B (bridge separate) | R2_w=0.234 | R2_w=0.339 |
  | D (bridge folded into prior) | R2_w=0.233 | R2_w=0.318 |

  No robustness gain for the unmatched group (0.234 vs 0.233, a wash) and
  clearly worse for the matched group (0.339 vs 0.318). The reasoning this
  confirms: for an unmatched player, `fl_k_rate_bridge` is already a FIXED
  CONSTANT (the league-average fallback) carrying zero player-specific
  information -- restructuring how a constant combines with the rest of the
  model can't create information that isn't there. The only way to genuinely
  help the unmatched population is to shrink it (better name-matching coverage,
  already pursued separately -- see "everyone's college rate looks the same"
  entry above), not to reshuffle how the existing non-signal is wired in. For
  players who DO have real college data, folding it into one compressed scalar
  prior-target loses the flexibility a separate regressor has to weight it
  against the naive prior on its own terms -- the same pattern already found
  for BB%'s horiz_dist/vert_dist.
- **Cost/risk accepted:** None -- not shipped.
- **Revisit if:** The bridge's own match coverage improves substantially
  (fuzzy name matching, more college-source years) -- doesn't change this
  conclusion directly, but worth re ­checking once the matched population is
  large enough to estimate a blended target more precisely.

---

### `build_period_kbb` grouped on (pitcher_id, pitcher name) -- a real aggregation bug, now fixed

- **Choice:** Group `fl_pitcher_features.build_period_kbb` by `pitcher_id` alone,
  never by the composite `(pitcher_id, pitcher)` key it used before.
- **Alternative(s):** Leave it as-is (it had been "working" throughout the whole
  backtest harness up to now).
- **Why:** Caught directly while adding the 2025-vs-2026 K%/BB% split to the
  production dashboard: `fl_2027_projections.parquet` had 946 rows for only 820
  real pitchers. Root cause: a handful of `pitcher_id`s carry more than one
  `pitcher` name value across their rows (confirmed: id `1000248636` is
  'House, Tristan' on 26 appearances and 'McCartney, Seth' on a single
  mislabeled game). The old composite-key groupby silently split that one
  real pitcher's counts across two output rows -- already a correctness bug
  (each row understates that pitcher's true PA) -- and became a visible
  cross-product explosion once two independently-built `build_period_kbb`
  tables were merged together on `pitcher_id` (exactly what the new per-year
  split did). This is a bug in OUR aggregation code (an unsafe groupby key),
  not just the upstream name/id inconsistency that triggered it -- the
  upstream inconsistency is a fact of the data; splitting one real pitcher's
  row in two was an avoidable modeling mistake.
- **Cost/risk accepted:** None found -- the fix only removes rows, it can't
  introduce new ones, and no existing caller of `build_period_kbb` selected its
  `pitcher` column (checked directly across all 4 call sites), so nothing
  downstream relied on the old grouping.
- **Revisit if:** A future caller needs the display name back -- join it in
  separately afterward via a `pitcher_id -> canonical name` lookup (already the
  pattern `fl_production_projection.py` uses), never by re-adding `pitcher` to
  this function's own groupby key.

---

### Production roster: consolidate same-person multi-id pitchers before building 2027 projections

- **Choice:** In `fl_production_projection.py`, remap `pitcher_id` to a
  canonical id (grouping by `(pitcher name, most-common team)`, keeping
  whichever id has the most rows) before building the 2027 roster/feature
  table -- applied only to the roster build, NOT to the frozen Experiment-2
  model fit (`fit_frozen_model` still trains on the original, unmapped ids, so
  the already-validated coefficients stay byte-for-byte unchanged).
- **Alternative(s):** Leave ids unmerged (status quo); or merge by name alone
  with no team check.
- **Why:** User-reported bug: "Brian Williams is missing his 2026 data" on the
  dashboard. Confirmed directly: 'Williams, Brian' has two `pitcher_id`s, both
  on 'Mississippi Mud Monsters' -- id `1000098100` (2025 only) and
  `1000079778` (2025 AND all 978 PA of 2026 data). The Player Page lists
  pitchers by NAME (deduplicated), so `.iloc[0]` on the matching rows silently
  picked whichever id's row sorted first -- which happened to be the
  2025-only id, hiding the real 2026 data that existed under the other id.
  Checked the scope before fixing: 22 distinct `(name, team)` groups across
  the full roster are split this way, including the already-known "Brouwer,
  Adam" case from the college-bridge work -- this is systemic, not a one-off.
  Merging by name ALONE was rejected because this project has also confirmed
  genuine different-real-people name collisions elsewhere (~1% of cases in the
  college bridge); requiring a shared team is a cheap, meaningful guard against
  merging two different people who happen to share a name.
- **Cost/risk accepted:** A real trade (same person, genuinely different teams
  across 2025/2026) would NOT be merged under this rule and could still show
  the same split-row symptom -- not checked for separately; the team-match
  requirement is a deliberately conservative trade-off (a recoverable missed
  merge, not a wrong one). Row count dropped 820 -> 797 after the fix.
- **Revisit if:** Another "missing data" report turns out to be a genuine
  same-person-different-team trade case -- would need a looser (but riskier)
  matching rule, e.g. also checking for non-overlapping date ranges between
  the two ids' name-only matches.

---

### Multi-year college input: helps BB%, not K% (tested directly, now in production)

- **Choice:** College bridge is now asymmetric by metric -- K% uses the
  pitcher's single most recent college season; BB% uses a TBF-weighted average
  across every available college season up to and including their most
  recent one. Also pulled 2026 college pitching data (5,404 rows, via the same
  FanGraphs API pattern as 2023-2025) so it's available as the "most recent
  season" for anyone who pitched college ball in 2026, feeding 2027 FL
  projections.
- **Alternative(s):** Single-year for both metrics (the original design);
  multi-year average for both.
- **Why:** User asked directly: does multi-year college history improve the
  bridge, tested on real 2026 FL data? Fit both forms on the n=152 (deduped)
  FL-rookie bridge pairs, weighted by FL-side reliability, and checked a
  combined model: K% -- single-year R2_w=0.120 vs multi-year R2_w=0.080;
  in the combined model multi-year's term is not significant at all (p=0.93,
  near-zero coefficient) while single-year stays significant (p=0.01).
  BB% -- multi-year R2_w=0.240 vs single-year R2_w=0.214; in the combined
  model multi-year is clearly significant (p=0.005, coef=0.45) while
  single-year drops to barely-significant (p=0.09) once multi-year is
  included. Plausible baseball reason: command/BB% tendency is a more stable,
  slower-changing trait that benefits from averaging out single-season noise,
  while K%/stuff is more directly tied to a pitcher's CURRENT physical
  development, so last year's data is diluted rather than helped by blending
  in two-years-stale numbers.
- **Cost/risk accepted:** The BB% multi-year average is computed from
  whatever years FanGraphs has for that name, with no additional team-level
  collision check beyond what the existing name-normalization already does
  (unlike the FL-side pitcher_id consolidation above, there's no "team" to
  cross-check a college player against an FL team) -- accepted given the
  already-measured ~1% name-collision rate for this specific matching task.
- **Also fixed in the same pass**: the original "College-Based K%/BB% are
  identical for everyone" bug report had two real causes, both addressed --
  (1) the Player Page was silently showing the league-average FALLBACK value
  for any unmatched player as if it were a real prediction (81% of the roster
  was unmatched against the old narrow bridge-fitting sample alone) -- now
  shows "no college match" instead; (2) extended coverage from the narrow
  150-player bridge-fitting-sample match to a broader NAME-based match against
  the full 2023-2026 college universe, raising real matches to 270/797 roster
  players. This broader match is layered on top of, not inside,
  `fl_halfseason_backtest.py`'s own `bridge_features()` -- so Experiment 1-4's
  already-reported backtest numbers are untouched; it only affects the
  production 2027 roster build.
- **Revisit if:** More FL/college seasons accumulate -- worth re-testing
  whether an even-longer BB% window (4+ years) keeps helping or starts adding
  pure noise, and whether a recency-weighted (not flat TBF-weighted) multi-year
  average beats the simple average tested here.

---

### wh_sva/horiz_dist/vert_dist as direct LEVEL priors: a real, held-out-validated win (unlike the changepoint use of the same raw features)

- **Choice:** Add the pitcher's training-window MEAN `wh_sva` (K% models) and mean
  `horiz_dist`/`vert_dist` (BB% models) as direct regressors alongside the naive
  prior -- a new model family, `naive+surprise_prior` (and combined with
  `other_league`), distinct from the `changepoint` use of the same underlying
  quantities.
- **Alternative(s):** Only use these features via changepoint detection (already
  shown not to generalize); or not use them as half-season-level priors at all
  (only via their already-validated CONCURRENT correlation, never tested as a
  forward-looking prior until now).
- **Why:** Reran the full 4-experiment harness with `wh_sva_level`/`horiz_dist`/
  `vert_dist` wired in as direct regressors. In Experiment 2 -- the largest
  genuinely held-out test (n=262, froze coefficients fit on full-2025->1H2026,
  scored on the real 2H2026 actual) -- adding the surprise-prior improved EVERY
  variant it touched: naive K% R2_w 0.2231->0.2457, BB% 0.3119->0.3323;
  naive+other_league K% 0.2397->0.2575, BB% 0.3154->0.3357. The best held-out
  performer across the entire harness is `naive+other_league+surprise_prior`
  (K% R2_w=0.2575, BB% R2_w=0.3357) -- beating every changepoint variant.
  Stacking changepoint ON TOP of this best model still hurts it
  (K% 0.2575->0.2419, BB% 0.3357->0.3212), reconfirming the changepoint finding
  independently rather than contradicting it. The in-sample experiments (3/4)
  show the same direction and are even larger in magnitude (Exp4 BB% R2_w
  0.3445->0.4290), consistent with -- not just an artifact of -- the held-out
  result. Exp1 (the smallest/earliest window, ~54K swing-pitches to fit the
  wh_sva model on) is the one exception: helped K% (0.0566->0.0904) but hurt BB%
  (0.2118->0.1808) -- plausibly too little data in FL's very first half-season to
  fit the two-stage model reliably yet, not evidence against the feature itself.
- **Cost/risk accepted:** Exp1's mixed result means this shouldn't be trusted
  blindly on very small training windows; the feature's value likely scales with
  how much pitch-level data is available to fit the underlying wh_sva/location
  models, same caveat as the changepoint's min-starts threshold.
- **Revisit if:** More FL seasons accumulate to re-check Exp1-scale (very small
  training window) behavior specifically, or if a future hitter/xwOBA build wants
  the same prior-vs-changepoint distinction tested on a different feature set.

---

### Half-season K%/BB% backtest: changepoint feature overfits in-sample, doesn't survive holdout; composed K-BB% beats direct K-BB% exactly where it's genuinely tested

- **Choice:** Treat the `wh_sva`/`horiz_dist`/`vert_dist` within-window changepoint
  feature as unproven for production use, and keep composing K-BB% from separate
  K%/BB% models rather than fitting it directly -- both decided from real held-out
  evidence on FL data (`fl_halfseason_backtest.py`), not carried over from the
  earlier MLB-notebook finding or from in-sample fit stats alone.
- **Alternative(s):** Trust the changepoint feature and/or fit K-BB% directly,
  since both look like wins in-sample.
- **Why:** Ran 4 backtest experiments (Exp1/2 genuinely held-out -- fit on one
  transition, froze coefficients, scored on the next period's actual rate; Exp3/4
  in-sample only, since 2H2026 is the last period FL data exists for, per the
  user's own call that there isn't enough data to hold out there). Results:
  - **Changepoint, held-out (Exp2, naive -> naive+changepoint):** K% R2_w
    0.2231 -> 0.2062 (WORSE), BB% R2_w 0.3119 -> 0.2991 (WORSE). Same feature,
    in-sample (Exp3): K% 0.2503 -> 0.2577 (better), BB% 0.3130 -> 0.3187
    (better). Classic overfit signature -- looks like a real improvement until
    actually held out, then hurts. The full combined model
    (naive+other_league+changepoint) shows the same pattern: best in-sample
    performer in Exp3/4, but WORSE than naive+other_league alone in Exp2's
    held-out test (K% 0.2397 -> 0.2198).
  - **K-BB%, direct vs. composed:** in the two genuinely held-out experiments,
    composed matches or clearly beats direct (Exp2 naive+other_league+changepoint:
    composed R2_w=0.1967 vs. direct=0.1809; every Exp2 variant favors composed or
    ties). In the in-sample-only experiment (Exp3), direct "wins" on every variant
    (e.g. naive+other_league+changepoint: direct=0.2368 vs. composed=0.2291) --
    but that's exactly the in-sample-overfit pattern just confirmed above by the
    changepoint result, not independent evidence for direct. This reinforces,
    with real FL held-out evidence (not just the separate MLB-notebook claim,
    which used different data/grain), that composing is the safer choice.
  - **Other-league prior**: genuinely mixed -- slightly HURT in Exp1's held-out
    test (K% 0.0566->0.0560, BB% 0.2118->0.1863) but helped in every other
    experiment (Exp2 held-out: K% 0.2231->0.2397, BB% 0.3119->0.3154; Exp3/4
    in-sample: consistently positive). Plausibly explained by Exp1's training
    window (1H2025) being FL's very first half-season, before most returning
    pitchers had built up real FL history for the bridge to compete against --
    not retested further here.
- **Cost/risk accepted:** The changepoint construction (per-start `ruptures`
  PELT detection within a window) may still carry real signal with a better
  penalty/min-starts tuning or a larger sample -- this isn't a verdict that
  within-season shape drift is fake, only that THIS construction, at FL's current
  half-season sample sizes, doesn't generalize. Not pursuing further tuning now.
- **Revisit if:** More FL seasons accumulate (more held-out transitions to test
  changepoint tuning against), or the changepoint detector is validated on a
  larger/denser population (e.g. full-season MLB data, where per-pitcher start
  counts are much higher) before reintroducing it here.

---

### Error analysis on Exp2's held-out residuals (fl_backtest_error_analysis.py):
### three real findings, one confirmed upstream data issue

- **Finding 1 -- low-naive-K% pitchers are under-shrunk, not just noisy.** Bucketed
  by training-period naive K% tercile: low tercile mean residual = -0.0183 (model
  predicts ~18pp higher K% than these pitchers actually post in the test period),
  middle/high terciles near zero (-0.003, -0.002). This isn't explained by sample
  size alone (see Finding 2) -- it's specifically the LOW end of the talent
  distribution that keeps declining further than plain reliability-shrinkage
  predicts. Actionable: an asymmetric/nonlinear shrinkage (shrink the low tail
  harder) is a concrete next thing to try, not yet built.
- **Finding 2 -- training-period PA (usage) itself carries signal beyond its role
  as a reliability weight.** Lowest training-PA quartile shows the worst residual
  (-0.0179 K%) even though it's already the most heavily down-weighted bucket in
  the aggregate metric. This matches the same "role/usage is a strong lever"
  pattern already validated earlier in this project's separate FL `wh_sva`
  precedent work (BF/G was the single strongest lever found there) -- suggests
  adding training-period PA/usage as a direct regressor (not just a weight) is a
  more promising next step than further tuning the changepoint feature.
- **Finding 3 -- the changepoint's raw sign barely tracks the error direction,
  sharper evidence than the aggregate R2 drop alone.** Among pitchers with a
  nonzero detected changepoint: K% `wh_sva_cp` same-sign-as-residual rate = 52.8%
  (barely above a coin flip), BB% `horiz_dist_cp` = 57.3%, BB% `vert_dist_cp` =
  43.5% (WORSE than a coin flip). This confirms the DECISION_LOG entry above isn't
  just "the linear term added nothing" -- the feature doesn't reliably point the
  right direction at all at current FL half-season sample sizes.
- **Confirmed upstream issue, concrete scale measured**: of 489 distinct 2026
  pitcher names, 16 (3.3%) are split across >1 `pitcher_id` (33 ids total, ~6.5%
  of all 2026 ids) -- the same issue flagged earlier in this project
  ("Brouwer, Adam"/"Jones, Logan"/"Martzolf, Max") now confirmed to directly
  explain at least 2 of a sampled 13 worst-residual cases (e.g. Casey Bargo's
  2026 PA split nearly 60/40 across two ids with OVERLAPPING date ranges,
  2026-06-09..07-09 and 2026-05-15..09-06 -- not even a clean before/after split).
  Split pitchers are modestly over-represented in the worst-residual tail
  relative to their 6.5%-of-ids base rate. Not fixed here (consistent with the
  earlier precedent of flagging rather than patching this upstream) -- but
  joining on name+team instead of raw `pitcher_id` for noisy small-PA cases is a
  cheap mitigation worth trying before the next backtest run.
- **Neutral finding**: college-bridge match status doesn't differentiate residual
  size either direction (K%: -0.0107 unmatched vs -0.0001 matched mean residual;
  BB%: -0.0020 vs -0.0022) -- no evidence the bridge is working asymmetrically
  well/poorly for the population it actually covers.
- **Per-pitcher confirmation of the aggregate pattern**: the full
  (other_league+changepoint) model improves individual abs-error for only 56.4% of
  K% pitchers (barely a majority) and 44.7% of BB% pitchers (a minority) relative
  to naive alone -- confirms the net BB% R2 decline isn't a few outliers dragging
  the average down, it's a genuinely more-often-than-not worse trade.

---

### College age: tested as a bridge covariate, not added to the model

- **Choice:** Do not add college age as an input to the college->FL K%/BB%
  translation. Keep the raw-rate-only model from the entry above.
- **Alternative(s):** Add age as a second regressor alongside raw college
  K%/BB%.
- **Why:** Merged FanGraphs college age (2023-2025, ~65% coverage) into the
  160-pair bridge on `(name, team, season)` -- 113 pairs matched. Reliability-
  weighted WLS, raw rate alone vs. raw rate + age: K% R2 0.112 -> 0.136 (age
  coef +0.0103/yr, p=0.080); BB% R2 0.197 -> 0.198 (age coef +0.0014/yr,
  p=0.745). K% shows a real-feeling direction (older college pitchers ->
  slightly higher FL K-rate, plausible physical-maturity effect) but doesn't
  clear conventional significance at n=113; BB% shows nothing. This closely
  mirrors the already-established MLB finding: a genuine-looking aging effect
  for K% that isn't large enough to earn a spot in the model once added
  on top of a rate-based prior, and no effect for BB%.
- **Cost/risk accepted:** If a larger bridge sample later pushes the K%
  age term below p=0.05 with a similar effect size, this should be
  revisited -- the direction is consistent with the MLB precedent, so a
  larger n finding significance wouldn't be surprising.
- **Revisit if:** Bridge sample grows substantially (more years or the
  hitter/xwOBA side), or age coverage improves beyond ~65%.

---

### Conference-season-relative K%/BB%: tested on real data, does NOT beat raw rate
- **Choice:** Use RAW college K%/BB% as the bridge input for college->FL
  translation, not conference-season-relative, reversing the design direction
  from several turns of prior discussion.
- **Alternative(s):** Conference-season-relative (player minus PA/TBF-weighted
  conference-season average) -- the originally planned approach.
- **Why:** Built the real bridge on 3 years of actual FanGraphs college
  pitching data (2023-2025) matched against real Supabase FL pitcher-seasons
  (2025-2026), n=160 clean pairs after excluding genuine name collisions.
  Raw rate beat relative on every measure: K% corr 0.365 vs 0.291, BB% corr
  0.467 vs 0.441, K% R2 0.133 vs 0.085, BB% R2 0.218 vs 0.195. In a combined
  model, the relative term added nothing significant beyond the raw rate
  (p=0.075 K%, p=0.121 BB%). This directly confirms, with real data, the
  "zero-sum within a closed conference" hypothesis raised earlier: mean
  college K% (20.01%) and mean FL K% (20.16%) are nearly identical, and mean
  BB% is an EXACT match (12.12% both) -- there isn't much real
  between-conference-level signal in this specific bridge to strip out, so
  subtracting a conference-season average mostly adds estimation noise
  without removing real bias.
- **Cost/risk accepted:** None -- this is strictly better and simpler than the
  originally planned approach. Conference identity was NOT thrown away
  entirely though: it's still the right tool for the minimum-sample-floor
  question and for any FUTURE check of whether a specific conference's talent
  pool has shifted; it just isn't needed as an INPUT to the translation model.
- **Also resolved as moot:** the conference-season minimum-sample-floor
  worry -- every one of the 90 real conference-seasons checked had 33+
  contributing pitchers, well above any reasonable threshold.
- **Revisit if:** A much larger bridge sample (more years, or extended to
  batters/hitters) shows a different pattern -- n=160 is real signal but not
  enormous; also worth retesting once genuine within-player conference-to-
  conference transfers (not just college-to-FL) are available, since that's a
  cleaner test of the same question this decision was based on.

### FanGraphs college data needs real browser automation, not plain HTTP
- **Choice:** Treat FanGraphs' `/api/leaders/college/data` JSON endpoint as
  requiring genuine browser-context access for any production pull, not a
  `requests`-based scraper.
- **Alternative(s):** A plain Python `requests` pull with realistic headers
  (User-Agent, Referer) -- the approach that worked fine for StatsCrew.
- **Why:** Tested directly: the same request that succeeds via `fetch()` inside
  an actual loaded browser page returns full data (confirmed `totalCount:
  5471` for 2025 college pitchers, resolving the earlier "only 121 rows"
  mystery -- that was a rendered-table cap, not a real population limit).
  The identical URL via plain `requests`, even with a realistic User-Agent and
  Referer header, returns 403. This points to fingerprint/challenge-based bot
  protection, not just header-checking.
- **Cost/risk accepted:** A production pull needs real browser automation
  (e.g. Playwright) for this specific source, meaningfully heavier than the
  StatsCrew pull -- a real added dependency/complexity, not a drop-in
  replacement of the same pattern.

### Conference/level effects: measure via bridge pairs, not cross-sectional variance
- **Choice (proposed):** Split the hierarchical model's job in two -- keep the
  player-level random effect for within-conference-season shrinkage, but
  measure the actual conference/level TRANSLATION factor from bridge pairs
  (players who actually moved between conferences, or from college into pro
  ball), mirroring the already-validated MLB<->AAA/AA bridge design, rather
  than trying to extract it from the GLMM's conference-level cross-sectional
  variance component.
- **Alternative(s):** Keep trying to estimate conference effects purely from
  aggregate cross-sectional variation (the original hierarchical-model design).
- **Why:** The "zero-sum within a closed conference" hypothesis specifically
  predicts that CROSS-SECTIONAL conference-average comparisons should be muted
  (matches the simulation's persistent conference-variance underestimation).
  A player who actually changes competitive environment gives a WITHIN-PERSON
  comparison instead -- same talent, different context -- which isolates a
  real level effect even when the cross-sectional signal is weak or hard to
  estimate. This is the identical logic that already worked for the MLB<->MiLB
  bridge; no new methodology needed, just the same construction one career
  stage earlier.
- **Cost/risk accepted:** Narrows what the conference-level model term needs to
  do (mainly a prior for players who haven't yet made a jump to measure
  against), which reduces how much the fixed-vs-random-effect debate for that
  term actually matters for the translation task specifically -- reprioritizes
  rather than fully resolves that open question.
- **Revisit if:** Bridge-pair sample sizes turn out too thin per
  conference-pair to trust (the same sample-starvation problem that dropped
  A+/A/Rookie from the MLB bridge) -- would need to fall back toward pooling
  strategies (e.g. by division, not individual conference) for the bridge
  itself.

### Conference-level term: proposed switch from random effect to fixed dummies
- **Choice (proposed, not yet tested on real data):** Keep player as a random
  effect (partial pooling -- validated via simulation to recover accurately and
  stably regardless of group count), but replace the conference-season random
  intercept with fixed-effect dummy variables instead of a shared-variance
  random effect.
- **Alternative(s):** Keep the full two-level random-intercept GLMM as
  originally specified.
- **Why:** Simulation evidence (n_conf=8/15/30, matching up to the real D1
  count) showed the conference-level variance component persistently
  underestimated and throwing VB non-convergence warnings at every group count
  tested, while the player-level component was accurate and stable throughout.
  Hypothesis (user's, mechanistically coherent): conferences are closed
  competitive systems -- hitters and pitchers within one face each other
  repeatedly, so a strong pitching class and a strong hitting class partially
  cancel in the aggregate rate, plausibly making true between-conference
  variance genuinely small. Small/near-zero variance components are a known
  hard case for MLE/VB estimators (boundary problem), which would explain the
  non-convergence directly rather than it being just a fitting nuisance.
- **Cost/risk accepted:** Loses partial pooling for thin conferences (few
  qualifying players) -- a fixed effect there gets no shrinkage toward a
  sensible average. Mitigated by the already-planned minimum-sample floor for
  conference-season aggregates, rather than requiring partial pooling to solve
  it.
- **Important nuance, not fully resolved by the "zero-sum" argument:** this
  explains why conference-*average rate levels* might cluster together: it does
  NOT by itself establish that conference identity carries zero information
  about how a given *relative* performance translates to pro ball (two
  conferences could have similar aggregate K% while housing very different
  overall talent levels). Worth keeping distinct from the variance-estimation
  question.
- **Revisit if:** Real fitted conference dummy coefficients show a genuinely
  wide, non-noise spread (would undercut the "not much between-conference
  variation" hypothesis and argue for keeping/improving the random-effect
  version instead).
- **Follow-up simulation finding (homogeneous-variance assumption, checked
  directly):** simulated half the conferences with tight true player-talent
  spread (SD=0.10) and half wide (SD=0.40); the single pooled variance the
  homogeneous model can fit (0.269) sits far closer to the wide conferences'
  truth than the tight ones'. Converted to reliability weight, this makes
  tight-conference players get WAY too much trust in their raw small-sample
  rate (correct reliability 0.842 vs. 0.975 under the wrong pooled estimate,
  error +0.133) while wide-conference players are barely affected (error
  -0.014) -- a real, asymmetric practical cost, not just a theoretical
  concern. Also confirmed the earlier non-convergence wasn't an
  insufficient-iterations problem: rerunning with 5000 max iterations and a
  much tighter gradient tolerance produced IDENTICAL conf/player variance
  estimates and still failed to converge -- consistent with a genuine
  small/near-zero-variance boundary problem, not a tunable optimizer setting.

### Hierarchical model's "constant rate within a season" assumption: accepted, by design
- **Choice:** Accept the binomial GLMM's assumption that a player's true K%/BB% is
  constant across their whole conference-season, rather than building in-season
  drift into this model.
- **Alternative(s):** Extend the model with a time-varying rate (e.g. a random walk
  or spline on the logit scale across the season) to capture drift directly.
- **Why:** In-season drift is architecturally a separate concern from the stable
  prior this model exists to produce -- exactly the split already validated for
  MLB, where the Marcel baseline stayed a constant-rate prior and in-season
  updating was handled by a separate `wh_sva`-surprise-vs-baseline mechanism
  (real, validated signal: R2 0.33->0.37 on rest-of-season K%). Baking drift into
  the prior model itself would conflate two different jobs.
- **Cost/risk accepted:** None yet, since the drift-detection piece (intrinsic
  characteristics like `wh_sva`/bat-tracking, for FL/college) doesn't exist yet --
  explicitly deferred, to be built out later.
- **Revisit if:** The drift-detection component is built and its output needs to
  feed back into how the prior model's assumption is framed.

### College/indy priors: conference-SEASON relative K%/BB%, not raw rates
- **Choice:** For college (and eventually indy-league) priors, use each player's K%/BB%
  relative to their own conference's average **for that specific season**, not the
  raw rate and not a conference average pooled across multiple years.
- **Alternative(s):** Raw K%/BB% translated directly (mirroring the MLB↔AAA/AA
  approach); or relative-to-conference using a fixed, all-years-pooled conference
  average as the baseline.
- **Why:** Raw rates conflate the player's own skill with how strong that year's
  competition was. A conference average pooled across years would still miss
  year-to-year swings in a conference's actual strength (a weak SEC year vs. a
  strong one). Conference-*season* relative performance isolates skill from both
  effects, AND lets every conference's players pool into one translation model
  instead of needing (likely sample-starved) per-conference slopes.
- **Cost/risk accepted:** Needs a conference-season aggregate computed from
  `external_season_stats` before any player's relative metric can be computed --
  an extra aggregation step, and one more place a minimum-sample rule is needed
  (see below).
- **Still open:** (1) difference vs. ratio vs. z-score for "relative" -- leaning
  difference for the first pass (simpler, no conference-season variance estimate
  needed) with z-score as an empirically-tested alternative later, matching this
  project's pattern of starting simple and only adding complexity that's proven
  to earn its keep (recall: robustifying edge_dist and flexible age specs both
  failed that test). (2) Result stats beyond K%/BB%: narrowed to power (ISO or
  HR/AB) as the one candidate worth testing -- reasoning being power is
  bat-speed/swing-plane driven rather than BABIP-luck driven, so it may carry
  real skill signal a peripheral-only model would miss, unlike AVG/OBP/SLG
  generally. To be validated empirically (does it add anything beyond relative
  K%/BB% in a held-out check) once real conference-season data exists --
  not assumed. ISO/HR-AB will get the SAME conference-season-relative treatment
  as K%/BB% (decided) -- consistent with the whole point of the relative-metric
  approach, though worth double-checking empirically once real data exists,
  since power's park/altitude sensitivity might not be fully captured by
  conference identity the way general competition strength is.

### AAA Statcast pitch-level pull deprioritized (paused, not abandoned)
- **Choice:** Stopped the in-progress AAA Statcast pull (partial data:
  112+ checkpointed chunks in `output/aaa_statcast_raw_chunks_v2/`, not yet
  assembled into yearly files) to redirect effort to Frontier League 2026
  season-end projections.
- **Alternative(s):** Finish the pull first (it was making progress, just
  slow — recursive splitting means more API calls than a naive pull), then
  pivot.
- **Why:** User priority shift — FL's 2026 season just ended, making FL
  projections the timely thing to work on now; the AAA pitch-level upgrade to
  the MiLB bridge was a nice-to-have refinement on top of an already-shipped,
  working box-score-based bridge, not a blocker for anything.
- **Cost/risk accepted:** None lost — checkpointed chunks remain on disk and
  `pull_aaa_statcast.py` can simply be rerun later to resume from them.
- **Revisit if:** MiLB-bridge accuracy becomes a priority again, or there's
  a specific rookie projection where the box-score-only bridge is clearly
  insufficient.

### Adaptive recursive splitting for the AAA Statcast pull
- **Choice:** Pull a date window; if the result hits (or nears) a 25,000-row
  cap or its earliest returned date is later than requested, split the window
  in half and recurse on each half, until every leaf is proven untruncated.
- **Alternative(s):** Guess a "safe" fixed window size (e.g. always 4 days)
  based on observed daily volume, matching how the original MLB pull just used
  fixed monthly chunks.
- **Why:** A naive month-chunked first attempt at this pull silently lost
  ~80% of every month (each chunk hit exactly 25,000 rows and only contained
  the LAST few days of the requested range — the API truncates from the
  front, not randomly). Daily AAA pitch volume isn't constant enough to trust
  a single fixed "safe" size — doubleheaders, schedule density, and level
  differences could silently blow past a guessed threshold the same way the
  month-sized guess did. Detecting truncation directly and bisecting
  guarantees correctness regardless of the true underlying volume.
- **Cost/risk accepted:** More total API calls (roughly 8-10x the naive
  chunk count) and slower wall-clock time for the same date range.

### AAA Statcast row cap/front-truncation discovered
- **Choice:** Treated `len(result) >= ~25,000` as inherently suspect and
  verified against the returned date range before trusting any pull, rather
  than accepting a 200-OK response with a plausible-looking row count at
  face value.
- **Alternative(s):** Trust the API response since it returned HTTP 200 with
  no error — the response looks like a completed, successful query.
- **Why:** All 18 chunks in the first attempt landed on exactly 25,000 rows —
  suspicious uniformity across windows of different lengths (some 28 days,
  some 31, one a partial month) that real pitch-count variation would never
  produce. A quick check of one chunk's actual date range (June 2024 request
  → only June 25-30 came back) confirmed silent front-truncation.
- **Cost/risk accepted:** None once caught — but this is the kind of bug that
  would have silently corrupted every downstream AAA feature/model without a
  deliberate sanity check; a reminder to distrust round/repeated numbers.

### Only the "base" Marcel input is touched by the MiLB blend
- **Choice:** For thin-reliability or rookie pitchers, replace only the
  `Marcel_K_rate`/`Marcel_BB_rate` value fed into the already-fitted K%/BB%
  regression. The regression itself (age terms, arsenal-shape features,
  coefficients) is identical for every pitcher, established or not.
- **Alternative(s):** Refit a separate model for thin-history pitchers, or
  retrain the whole K%/BB% regression on the MiLB-augmented population.
- **Why:** The K%/BB% regression was already validated via walk-forward
  backtesting on established pitchers; refitting it on a population that
  includes translated MiLB priors would require re-validating the whole
  model, not just the new piece. Swapping only the input keeps the blast
  radius of this change limited to exactly the population it targets.
- **Cost/risk accepted:** Established-pitcher projections are provably
  unchanged (verified: Skenes/Skubal/Wheeler numbers were byte-identical
  before/after), but this also means the model can't learn anything
  MiLB-specific beyond what flows through the single `base` term (e.g., no
  MiLB-specific arsenal-shape interaction is possible with this design).

### MiLB→MLB translation fit with a genuine train/test split
- **Choice:** Fit translation and combo-blend coefficients on bridge pairs
  from seasons ≤2023, validate on 2024-2025 held out, and only refit on all
  years for the final production coefficients after the held-out check
  passed.
- **Alternative(s):** Fit directly on all available bridge pairs (~1,500-1,700
  pairs) and report in-sample R².
- **Why:** The bridge sample is small enough (a few hundred to ~1,700 pairs)
  that in-sample fit statistics would be a weak guarantee against overfitting.
  A genuine held-out check is the only way to know the combo blend (which
  reversed Marcel's negative R² for this population) is real signal, not an
  artifact of fitting the same small sample it's evaluated on.
- **Cost/risk accepted:** Splitting a small sample further concentrated the
  A+/A/Rookie-level bridge pairs below a trustworthy size, so only AA/AAA
  translation factors are used in production; lower levels were reported
  descriptively but not deployed.
- **Revisit if:** More MiLB seasons accumulate (each new year adds bridge
  pairs) — worth periodically refitting and re-checking A+/A coverage.

### Robustifying the edge-distance metric was tested and rejected
- **Choice:** Kept the raw per-pitcher-season mean of signed edge-distance,
  despite it having the worst stability (n0) of any location metric built
  this project, rather than winsorizing/trimming/using the median.
- **Alternative(s):** Winsorize at 1/99 or 5/95 percentile, trim 5% each
  tail, or use the median instead of the mean.
- **Why:** Tested all four alternatives directly. Every one made the BB%
  correlation *worse* (0.307 → 0.303 → 0.289 → 0.263 as more aggression was
  applied), meaning the "outlier" pitches being suppressed were carrying real
  signal about a pitcher's walk-proneness, not contaminating it. The
  instability's actual root cause (confirmed by a follow-up test) was mixing
  heterogeneous count-leverage situations into one average, not outlier
  pitches — so a robustness fix aimed at outliers was fixing the wrong thing.
- **Cost/risk accepted:** The raw-mean metric is genuinely noisier
  (n0≈587 pitches) than it would be if outliers really were the problem —
  addressed instead by splitting the metric by count leverage, which is a
  separate, later entry.

### Age functional form: kept the simplest one that ties
- **Choice:** Kept linear+kink-at-30 for the K%/BB% models' age term.
- **Alternative(s):** Pure linear (no kink), kink at 33, fully flexible
  age-bin dummies (non-parametric).
- **Why:** Backtested all four specs. None beat the current one by more than
  noise (K% R²ʷ ranged 0.3166-0.3181 across all four; BB% was actually
  *better* with no age term at all, 0.1689 vs 0.1684-0.1688). Most of the
  age-related decline is already captured by the Marcel base term (a
  declining pitcher's recent performance already reflects the decline), so
  the residual, non-redundant contribution of age is small enough that its
  exact shape doesn't matter.
- **Cost/risk accepted:** None identified — this was a case where the
  simplest defensible choice and the empirically-best choice coincided.

### Lighter-weight backtest used as a stand-in for later ablations
- **Choice:** After confirming the full-history-fit `wh_sva` backtest
  (mlb_kbb_model.py) tracked the fully rigorous per-fold-refit version
  (mlb_kbb_walkforward.py) within <0.01 R²ʷ, used the lighter version for
  every subsequent feature ablation (arsenal depth, age specs, BRK/OFF
  location, count-leverage command) rather than re-paying the ~10-minute
  rigorous-refit cost each time.
- **Alternative(s):** Re-run the fully rigorous per-fold pipeline for every
  new feature test.
- **Why:** The rigorous version's whole purpose is to rule out a specific
  kind of look-ahead (the `wh_sva` XGBoost stage seeing future-fold data).
  That risk doesn't change based on which UNRELATED feature is being tested
  on top of it — once validated once, re-validating it for every ablation is
  pure repeated cost with no new information.
- **Cost/risk accepted:** If a future feature interacted with the `wh_sva`
  fitting itself (none tested so far have), this shortcut would need
  revisiting for that specific case.

### wh_sva conditioning: validated both, didn't assume the answer
- **Choice:** Built and backtested BOTH swing-conditioned (Whiff%) and
  unconditional (SwStr%) shape-value-added, per pitch-type family, rather
  than committing to Whiff% on the (reasonable-sounding) prior that it
  should win.
- **Alternative(s):** Use only Whiff%-conditioned SVA (the established
  convention from prior FL work), based on it having the higher raw
  correlation with K%.
- **Why:** Explicitly requested rather than assumed. Turned out to matter:
  the two are nearly tied in the walk-forward backtest (swstr slightly ahead,
  0.318 vs 0.316 R²ʷ, but the edge flips fold-to-fold) — a real, reportable
  finding that would have been missed by picking one on priors alone.
- **Cost/risk accepted:** Roughly 2x the model-fitting cost for this feature
  family (6 XGBoost fits instead of 3, per fold).

### Empirical reliability curve built and cross-checked before any use
- **Choice:** Estimated n0 (the stabilization point for reliability
  weighting) two independent ways — single-season method-of-moments and a
  fitted year-over-year correlation curve — and required them to be in the
  same ballpark before trusting either, rather than picking one method or a
  textbook constant (e.g. the commonly-cited "70 PA" for K%).
- **Alternative(s):** Use a fixed textbook stabilization constant, or trust
  a single estimation method without cross-checking it.
- **Why:** The two methods measure related but different things (pure
  sampling noise vs. sampling noise + real year-to-year talent drift), so
  they're expected to differ somewhat (YoY should read as "less reliable") —
  but a *large* divergence signals a bug rather than real drift. This
  standard was then reused later in the project (see the edge-distance
  entries above) to catch a real methodological problem.
- **Cost/risk accepted:** More upfront validation work before any modeling
  could start.

### Trimmed Statcast columns for the 2018-2021 pull
- **Choice:** Pulled only ~38 of the available 119 Statcast columns for the
  2018-2021 backfill (dropped `vx0/vy0/vz0/ax/ay/az` among others) to reduce
  memory/disk pressure after an earlier full-column attempt was killed
  (likely by a laptop sleep event, diagnosed later, but the column trim was
  also a genuine memory-pressure mitigation at the time).
- **Alternative(s):** Pull the full 119-column schema, matching the
  pre-existing 2022-2025 files.
- **Why:** The dropped columns weren't needed for any feature planned at the
  time (Marcel, `wh_sva`, location, BF/G, age all use the retained columns).
- **Cost/risk accepted:** This directly blocked physics-based pitch-tunneling
  reconstruction for 2018-2021 later in the project (that analysis needed the
  full 9-parameter trajectory and had to be scoped to 2022-2025 only, which
  still have the full column set from before this project started trimming).
- **Revisit if:** A future feature needs full trajectory reconstruction for
  the older years — would require re-pulling 2018-2021 with the extra columns.

### BB% bridge as an informative prior (K% kept as a free OLS term)
- **Choice:** In the production BB% model, the other-league bridge rate no
  longer enters as its own free OLS regressor (`naive_BB_rate +
  fl_bb_rate_bridge + horiz_var + vert_var`). It's folded into the naive
  rate itself via a Marcel-style two-source shrinkage blend: `blended_BB_rate
  = (BB_raw + n0_bb*lg_bb + n0_bridge_bb*fl_bb_rate_bridge) / (PA + n0_bb +
  n0_bridge_bb)`, fit as `blended_BB_rate + horiz_var + vert_var`.
  `n0_bridge_bb` (the bridge's pseudo-PA "worth") is fit once, by nonlinear
  least squares against real outcomes, on the exact Experiment 2 training
  transition (full-2025 -> actual 1H2026) and frozen (`n0_bridge_bb=58.0`,
  `mlb_marcel.fit_n0_bridge`/`blended_rate`). The SAME formula, evaluated at
  PA=0, now also replaces the old two-coefficient mixing in
  `project_from_college_stats` (lookup + manual calculator pages) for a
  brand-new prospect with zero real FL PA. K%'s bridge is UNCHANGED (still a
  free OLS term) -- the prior formulation was tested for K% too and didn't
  win there (see below).
- **Alternative(s):** Keep the bridge as a free OLS coefficient for both K%
  and BB% (the prior design); or apply the informative-prior reform to K%
  as well, for architectural symmetry.
- **Why:** Motivated by `fl_ros_coef_drift_test.py` (sweeping in-season
  cutoffs at weeks 2/5/8/11, both 2025 and 2026): the bridge's OLS
  coefficient was consistently the LEAST stable regressor in both the K%
  and BB% specs (CV 0.37 and 0.49, vs. 0.20-0.31 for every other regressor),
  and it drifted in a specific, sensible direction -- high early in the
  season (thin current-season sample, lean on the bridge), decaying to
  near-zero by week 11 (current sample has taken over). A free per-cutoff
  OLS coefficient can only approximate that continuous, PA-dependent
  relationship piecewise; a frozen production coefficient is flat-out wrong
  for most of the season. Tested the fix two ways, genuinely held out by
  year (fit on one year, freeze, score on the other, both directions):
    - **BB%** (`fl_ros_bridge_prior_test.py`): the informative prior won
      BOTH directions (train25->test26: R²=0.274 vs. OLS's 0.263 vs.
      naive-only's 0.235; train26->test25: R²=0.218 vs. OLS's 0.193 --
      OLS didn't even beat naive-only in that direction). It was also
      confirmed on the ACTUAL production transition
      (`scratch_verify_bb_prior_exp2.py`, full-2025->actual 1H2026 frozen,
      scored on 2025+1H2026->actual 2H2026): a dead statistical tie with
      the current OLS approach (R²=0.3623 vs. 0.3631) -- no regression at
      full-season PA, where the two formulations are expected to converge
      (the bridge's weight is small and stable there either way).
    - **K%**: the informative prior LOST both held-out directions
      (train25->test26: R²=0.176 vs. OLS's 0.188; train26->test25: R²=0.185
      vs. OLS's 0.200) -- OLS's extra free parameter (independently
      rescaling the naive term, not just adding the bridge) earns its keep
      for K% in a way it doesn't for BB% at these sample sizes (~220-255
      pitchers per fold). K% was left on its original OLS spec.
    A single frozen `n0_bridge_bb`, with no per-cutoff refitting, was also
    confirmed to reproduce the SAME decaying-influence pattern the unstable
    OLS coefficients were approximating (implied weight 0.6 at week 2's
    median PA down to 0.28 at week 11, for both K% and BB% diagnostics) --
    confirming the original drift diagnosis, even though the fix was only
    adopted for BB%.
- **Cost/risk accepted:** Two different mixing mechanisms now coexist for
  K% (free OLS term) vs. BB% (informative prior) -- a real asymmetry, not a
  bug, but one to remember when extending either spec. `n0_bridge_bb` is a
  new frozen hyperparameter (like `n0_k`/`n0_bb`) that would need refitting
  if the training transition or population changes meaningfully.
- **Revisit if:** A future, genuinely thin-sample production use case (e.g.
  an in-season ROS tool, the original motivation) would be exactly where
  this reform's real edge shows up most -- the full-season tie-test above
  means it's a safe default, not that it's been shown to matter yet in the
  current preseason-only product.

### RA9: park factor + prior-season usage as regressors (error-analysis-driven, SHIPPED)
- **Choice:** Added two regressors to the production RA9 model: `ppf` (real,
  player-season-specific park factor, already in `war_pitchers`, known in
  advance since it follows from roster/schedule, not performance) and
  `ip_per_g_prior` (the PRIOR season's innings-per-appearance -- a usage/
  trust signal). New spec: `RA9 ~ K_pred + BB_pred + ppf + ip_per_g_prior`,
  replacing the walk-forward-simplified `RA9 ~ K_pred + BB_pred` alone.
  `RA9_RESID_SD` updated from 1.6087 to 1.5773 (new spec's LOOCV RMSE_w) --
  this feeds directly into every P(> Repl.) probability on the dashboard.
- **Alternative(s):** Keep the plain K%+BB% spec (the walk-forward-simplified
  "survives everything" baseline); use the TARGET season's own IP/appearance
  instead of the prior season's (tested, rejected -- see below).
- **Why:** Motivated by `fl_error_analysis_v2.py`'s residual breakdown on the
  plain K%+BB% spec: team-level RA9 residuals showed a ~1.7-run spread with
  zero team/park information in the model at all. A combined individual-
  level regression (n=96) of the residual on candidate explanations found
  innings-per-appearance the single strongest, most robust predictor
  (p=0.0015, survives everything else in the model, and explains away an
  earlier-looking `horiz_var`/command-consistency signal that turned out to
  be a confound of it), with `ppf` a smaller but real, independently
  significant contributor (p=0.035 alone).
  - **Critical distinction caught before shipping**: the error analysis used
    the TARGET season's (2026) own IP/appearance, which is NOT a legitimate
    predictive feature -- a pitcher's usage in the season being predicted is
    partly an OUTCOME of that season (how much a coach trusts them), not
    knowable in advance. Tested both: the leaky target-year version gave
    R2_w=0.265 (not deployable); the leak-free prior-year version gives the
    real, usable R2_w=0.140 -- a real but much more modest improvement than
    the raw error-analysis correlation implied.
  - **Validated genuinely held out, BOTH directions** (forward 2025->2026,
    and the non-causal 2026->2025 reversal -- same robustness-check
    convention as the BB% bridge-prior work above): R2_w 0.105->0.140
    forward, 0.002->0.055 reversed. Same feature ordering (ppf helps, prior-
    usage helps more, both together helps most) and monotonic RMSE
    improvement in BOTH directions -- real evidence this isn't a fluke of
    which year happened to be the training transition.
  - **Noted but not resolved**: the REVERSED direction's baseline itself is
    far weaker (R2_w=0.002 vs 0.105 forward) -- checked whether this is a
    sample-size artifact (2025 pitchers having thinner BF/IP than 2026,
    which would make 2025-as-target noisier) and it isn't an obvious story
    (mean BF 139 vs 148, median IP identical at 23.7 across the two years).
    Best guess, unverified: something about 2025 specifically (FL's first
    tracked season) makes it a harder target regardless of predictors --
    flagged as a real asymmetry worth its own investigation, separate from
    whether ppf/usage are real (they are).
- **Production wiring**: for the full 2027 roster, `ppf`/`ip_per_g_prior`
  use each pitcher's REAL 2026 value (best available proxy for 2027, same
  "most recent real year stands in for the projection year" convention
  workload already used); bridge-only prospects with no real 2026 FL row
  get league-average fallbacks (`ppf=99.95`, `ip_per_g=3.45`, IP-weighted
  over ip>=20 2026 pitchers) -- same neutral-imputation convention as
  `LEAGUE_K_RATE`/`LEAGUE_BB_RATE`. The TrustmarkPark shift mechanism itself
  is UNCHANGED (still a separate, additive, Air%-driven adjustment on top of
  `proj_ra9_baseline` -- `ppf` only entered the baseline, deliberately, to
  avoid double-counting park effects between a generic blended `ppf` and the
  specific TrustmarkPark HR/hit-suppression mechanism).
- **Cost/risk accepted:** `ppf` requires a park-factor table source
  (`war_pitchers.ppf`) that's treated as an external, unverified input per
  project discipline -- not re-derived here. `ip_per_g_prior`'s predictive
  value may partly reflect roster/coaching decisions specific to FL's
  current coaching staffs, which could shift if staffs turn over.
- **Revisit if:** more FL seasons accumulate -- re-run the same forward/
  reversed validation with a 3rd real season to see if the ppf/usage effect
  sizes are stable, and to investigate the 2025-as-target asymmetry noted
  above.

### RA9: MIS-scenario `ppf` was using the pitcher's real current team, not Mississippi's (bug fix, SHIPPED)
- **Problem (user-caught):** after adding `ppf` to the RA9 baseline model
  above, `proj_ra9_trustmark_half`/`full` were still computed as
  `proj_ra9_baseline + trustmark_ra9_shift` -- but `proj_ra9_baseline` bakes
  in the pitcher's REAL CURRENT team's `ppf`, not Mississippi's. The
  Trustmark shift itself (the separate Air%-based HR/hit-suppression
  mechanism) was built assuming a NEUTRAL (100-equivalent) starting park
  context for the "non-Trustmark" portion of its blend. Stacking that
  neutral-assumption shift on top of a baseline that already reflects the
  pitcher's real (often non-neutral) park double-counted or mis-specified
  the park effect depending on how far that real team's park sits from
  neutral.
- **Fix:** for the Trustmark scenarios specifically, re-predict the baseline
  with `ppf` swapped to the SCENARIO's actual blend -- `0.5*TRUSTMARK_PPF +
  0.5*100` for half-Trustmark, `TRUSTMARK_PPF` for full -- using
  TrustmarkPark's own standalone park factor (`park_factors_fip`,
  2025-2026 combined, `TRUSTMARK_PPF=93.0`, 100-centered, same scale as
  `ppf`) instead of the pitcher's real current-team value. `K_pred`/
  `BB_pred`/`ip_per_g_prior` are pitcher traits and correctly stay
  unchanged across scenarios; only `ppf` needed to track the scenario.
- **Verified the fix behaves correctly in both directions**: Gateway
  Grizzlies pitchers (real ppf=111.5, a bad/hitter-friendly park) now show a
  BIGGER half-Trustmark benefit than before (~0.6 runs, not just the 0.31-run
  Air%-only shift) -- correct, since they're leaving a worse park behind.
  New Jersey Jackals pitchers (real ppf=95.1, already a good park) now show
  a SMALLER benefit, sometimes a net downgrade in P(> Repl.) terms -- also
  correct, since a 96.5 half-Trustmark blend is actually slightly worse than
  their real, already-pitcher-friendly home park.
- **Scope checked**: `LEAGUE_RA9_REPLACEMENT` and the replacement-level
  Trustmark shift (`build_replacement_trustmark_shift`) do NOT use
  `ra9_model.predict()` or `ppf` at all (the replacement baseline is a fixed
  empirical constant; its own Trustmark shift only uses the Air%-based
  mechanism against a synthetic league-average profile) -- unaffected by
  this bug, no further changes needed there.

### RA9: pitches-per-PA added as a 5th regressor; functional-form sweep for IP/G; thin-sample extrapolation bug caught and fixed (all SHIPPED)
- **`pitches_per_pa_prior` added** (prior-season pitches thrown per plate
  appearance -- a pitch-economy/efficiency signal, distinct from
  `ip_per_g_prior`'s role/trust signal): only modestly correlated with
  `ip_per_g_prior` (r=-0.29) and has a genuinely different distribution
  shape (unimodal/bell-shaped vs. `ip_per_g_prior`'s bimodal SP/RP-cluster
  shape) -- not a restatement. Tested as a straight swap-in for
  `ip_per_g_prior` (slightly worse alone, R2_w=0.135 vs 0.140) and added on
  top (real additional lift, R2_w=0.159) -- genuinely complementary, not
  redundant. Validated held out BOTH directions before shipping: R2_w
  0.140->0.159 forward (2025->2026), 0.055->0.088 reversed (2026->2025).
  New spec: `RA9 ~ K_pred + BB_pred + ppf + ip_per_g_prior +
  pitches_per_pa_prior`; `RA9_RESID_SD` updated 1.5773 -> 1.5598.
- **Functional-form sweep for `ip_per_g_prior`** (prompted by its visibly
  bimodal distribution -- 37 pitchers clustered at 1-2 IP/appearance, 40 at
  4-6, a thin gap between): tested log, sqrt, a binary SP-like dummy
  (>=3 IP/G), a piecewise-linear hinge at the gap, AND a role-relative
  version (IP/G minus that pitcher's OWN role-group mean, roles from prior-
  season `ip_sp`/`ip_rp`, built leave-one-out to avoid leakage). Plain
  linear essentially ties log/sqrt (R2_w differs in the 4th decimal) and
  beats every role-aware alternative outright -- binary R2_w=0.131,
  piecewise R2_w=0.123, role-relative R2_w=0.103 (coefficient not even
  significant, p=0.52) vs. linear's 0.140. **The bimodal shape of the INPUT
  distribution does not imply the X->Y relationship needs to be nonlinear**
  -- those are different questions, and every attempt to route the signal
  through an explicit SP/RP split destroyed value rather than adding it, at
  this sample size (n=96). Kept plain linear.
- **Bug caught before shipping, not after**: the first production run with
  `pitches_per_pa_prior` added produced a physically impossible
  `proj_ra9_baseline=0.88` for a pitcher with only 5 real 2026 PA (whose own
  `pitches_per_pa_prior=7.4` sat outside the training population's range,
  which tops out around 6.3) -- and several >10 RA9 values on the other
  tail. Root cause: `ppf`/`ip_per_g_prior`/`pitches_per_pa_prior` are RAW,
  UNSHRUNK ratios (unlike `naive_K_rate`/`naive_BB_rate`, which get proper
  EB shrinkage) computed directly from however little 2026 data a roster
  pitcher has, fed into a model trained ONLY on BF>=100 pitchers
  (`fit_ra9_baseline_model`'s own population filter) -- applying it below
  that floor extrapolates. Fix: fall back to league-average `ppf`/
  `ip_per_g_prior`/`pitches_per_pa_prior` for any roster pitcher with real
  2026 BF<100 (the model's own training floor), not just when data is
  fully missing (the pre-existing `no_workload` check only caught the
  latter). Verified the fix directly: re-ran after the fix, confirmed the
  extreme low end became a clean, plausible pitcher (Salata, Derek: 22.9%
  K%, 9.2% BB%, RA9=3.71) and only one >12 RA9 value remained across all
  797 roster pitchers.
- **That one remaining extreme value is a SEPARATE, pre-existing issue, not
  from this change**: Brown, Daniel projects to RA9=14.5, but his
  `ppf`/`ip_per_g_prior`/`pitches_per_pa_prior` all correctly sit at
  league-average fallback (he has zero real FL data, is bridge-only) -- the
  real driver is his bridge-projected BB_pred=57.6% (a bad college-stat
  translation), which would have produced an almost-as-extreme RA9 (~13.4)
  under the ORIGINAL 2-feature K%+BB%-only spec too. Flagged, not fixed
  here -- a bridge-matching data-quality issue, out of scope for this
  change.
- **Cost/risk accepted:** the thin-sample guard (BF<100 -> league average)
  is a hard cutoff, same style as the pre-existing `no_workload` check --
  a pitcher just above/below 100 real BF gets a discontinuous jump in their
  `ppf`/`ip_per_g_prior`/`pitches_per_pa_prior` inputs rather than a smooth
  transition. Matches this project's existing convention for this file
  specifically (not the smoother EB-shrinkage convention used for
  naive_K_rate/naive_BB_rate elsewhere) -- acceptable here since it only
  affects three secondary regressors, not the main K%/BB% inputs.
- **Revisit if:** the Brown, Daniel-style bridge-quality issue recurs for
  other players -- worth a dedicated look at why some college->FL bridge
  translations produce extreme BB% estimates, independent of this RA9 work.

### RA9: continuous regressors clipped to training range (user-caught, SHIPPED)
- **Problem (user-caught)**: after the thin-sample-guard fix above, Derek
  Salata still topped the leaderboard by a suspiciously large margin (over a
  run ahead of #2). User pushed back specifically: "there wasn't that
  outsized of a gap before the latest update." Traced directly: reverting
  just the `pitches_per_pa_prior` addition dropped Salata's own predicted
  RA9 by -1.586 runs -- by far the largest shift of anyone near the top
  (next-biggest mover was -0.49). His real `pitches_per_pa_prior` (4.75) sits
  ABOVE the training population's own max (4.51, n=96) -- the model was
  extrapolating a borderline-significant coefficient (-1.894, p=0.079) well
  past where it has any real evidence.
- **Why LOOCV didn't catch this**: every held-out point in the validation is
  drawn from the SAME n=96 training population by construction, so leave-
  one-out can only ever test robustness WITHIN the training range, never
  extrapolation beyond it. The aggregate R2_w/RMSE improvement from adding
  `pitches_per_pa_prior` (validated genuinely held out, both directions) was
  real and is not in question -- but it says nothing about behavior outside
  the n=96 population's observed range, which the full 797-pitcher roster
  does venture into for a handful of players.
- **Checked whether this was unique to `pitches_per_pa_prior`**: no --
  `ip_per_g_prior` has 16/797 roster pitchers outside its own training range
  [1.04, 6.26], `ppf` has 3/797 outside [90.23, 108.02]. A general blind
  spot across all three continuous RA9 regressors, not specific to the
  newest one.
- **Fix**: `fit_ra9_baseline_model` now also returns the training-observed
  [min, max] for each continuous regressor; `clip_to_training_range` clips
  every roster pitcher's `ppf`/`ip_per_g_prior`/`pitches_per_pa_prior` to
  that range before any `ra9_model.predict()` call (baseline AND both
  Trustmark scenarios) -- the model is never asked to extrapolate past its
  validated support again. Affects a small, now-logged number of pitchers
  per regressor (6/16/3 of 797) -- most of the roster is untouched.
- **Verified the fix**: Salata's baseline moved 3.71 -> 4.15 (clipped), and
  the gap to #2 shrank from >1 run to 0.07 runs -- a normal, unremarkable
  margin. His REMAINING ~0.58-run edge specifically in the Trustmark-
  adjusted number is legitimate and untouched by this fix (his real `ppf`,
  102.3, is well within range) -- traced earlier to a genuine personal park-
  context advantage under the half-Trustmark scenario specifically, not an
  extrapolation artifact.
- **Cost/risk accepted**: a hard clip (not a smooth shrinkage) -- a pitcher
  just outside the training range gets capped at the boundary rather than a
  continuously tapering adjustment. Simple, consistent with this file's
  existing thin-sample-guard convention (also a hard cutoff), and the
  affected population is small enough that the discontinuity itself is a
  minor concern relative to the extrapolation it prevents.
- **Revisit if**: more FL seasons widen the training population (more real
  BF>=100-both-years pitcher pairs) -- the clip boundaries should be
  re-derived from the larger population, not left at today's n=96 range
  forever.

### RA9: ip_per_g_prior shrunk, pitches_per_pa_prior deliberately left raw (SHIPPED)
- **Choice:** `ip_per_g_prior` now uses Marcel-style EB shrinkage toward the
  league-wide games-weighted mean (`LG_IPG=2.2887`, `N0_IPG=12.0` games)
  instead of the raw `ip/g` ratio. `pitches_per_pa_prior` stays raw --
  tested the identical treatment and it actively hurts.
- **Why (asymmetric, tested not assumed):** user asked to shrink both after
  the clip-to-training-range fix. Tried the formal method-of-moments
  derivation first (same convention as `mm.estimate_n0`, substituting the
  real per-trial sampling variance for the binomial `p(1-p)/n` term) -- it
  broke for `ip_per_g_prior` specifically, returning a NEGATIVE implied n0
  (cross-sectional variance doesn't shrink with games the way the model
  assumes; plausible since IP/G is bimodal/role-driven, not noise around one
  center). Fell back to empirically grid-searching n0 via the same LOOCV
  discipline used throughout (precedent: the team-random-effect credibility
  constant earlier) -- `ip_per_g_prior` showed a clean, consistently-peaked
  optimum at n0=12 games in two independent sweeps (R2_w 0.159->0.164
  forward), and held (not regressed) in the reversed direction (0.088->0.089
  -- a tie). `pitches_per_pa_prior` showed the opposite: R2_w degrades
  MONOTONICALLY as shrinkage increases (raw/n0=0 best; the formal n0=94
  estimate actively hurts, R2_w 0.159->0.144) -- left raw.
- **Why the asymmetry makes sense:** the LOOCV test population IS the
  BF>=100-both-years population by construction -- these pitchers already
  have enough prior-season data that `pitches_per_pa_prior` (trial unit: PA,
  typically 100s) is already reliable; shrinking it there just adds bias.
  `ip_per_g_prior`'s trial unit is GAMES (often single digits even for an
  established pitcher), so real sampling noise remains even within this
  "established" population -- shrinkage has genuine room to help there
  specifically.
- **Note on what this doesn't test:** LOOCV, by construction, can only
  validate behavior WITHIN the training population's own range -- it cannot
  show whether shrinkage helps pitches_per_pa_prior's suitability for the
  genuinely thin-sample roster pitchers outside that range (the reason
  shrinkage was proposed in the first place). The `clip_to_training_range`
  safety net (previous entry) remains the actual mechanism protecting
  against that specific failure mode for BOTH regressors; shrinkage of
  `ip_per_g_prior` is a separate, independently-validated improvement to the
  core model fit, not a replacement for the clip.
- **Verified in production:** training range for `ip_per_g_prior` narrowed
  as expected (shrinkage compresses the distribution): [1.04, 6.26] raw ->
  [1.30, 4.64] shrunk -- fewer roster pitchers now need the hard clip (16 ->
  7 of 797), since shrinkage tempers most extreme cases on its own before
  clipping would even trigger. Full-roster sanity sweep re-run: no new
  extreme values, Salata's lead over #2 is now 0.18 runs (down from the
  original >1 run bug, consistent with the already-validated, legitimate
  smaller edge from his personal park context).
- **Revisit if:** more FL seasons give enough data to re-attempt the formal
  method-of-moments n0 for `ip_per_g_prior` properly (e.g. using real game-
  level IP variance instead of the cross-sectional proxy) rather than the
  empirically-grid-searched value used here.

### RA9: pitches_per_pa_prior residualized against K%/BB%/ppf (user-caught, SHIPPED)
- **Problem (user-caught):** user asked directly whether the usage/efficiency
  regressors might be proxying for skill rather than adding independent
  signal, and flagged them as possibly too influential. Checked both
  variables directly: `ip_per_g_prior` is fine (only 21% of its own variance
  explained by K_pred+BB_pred+ppf, VIF=1.27, partial relationship with RA9
  STRONGER than its raw bivariate one -- genuine independent signal).
  `pitches_per_pa_prior` was the real problem -- 57.5% of its variance is
  explained by K_pred+BB_pred+ppf (VIF=2.35), and that collinearity was
  cannibalizing K_pred's own coefficient: K_pred collapsed from -8.74
  (p=0.009, no efficiency term) to -4.17 (p=0.317, raw efficiency in the
  model). Confirmed on the real 797-pitcher roster via per-regressor
  contribution-spread: K_pred's spread (0.132) was SMALLER than either usage
  variable's individually -- backward for what should be a skill-driven
  model.
- **Tested whether a K%xBB% interaction explains pitches_per_pa_prior better
  than K%+BB% alone first** (the user's second question): it doesn't --
  R2 0.569->0.570, interaction term p=0.76. The mechanical relationship
  (both strikeouts and walks need more pitches than contact) is purely
  additive, no multiplicative effect worth keeping.
- **Fix:** residualize `pitches_per_pa_prior` against `K_pred + BB_pred +
  ppf` (`PPA_RESID_SPEC`, fit on the training transition, frozen -- same
  "fit once, reuse" discipline as every other coefficient in this project)
  and use the residual (`ppa_resid`) as the regressor instead of the raw
  ratio. By construction this is exactly orthogonal to K_pred/BB_pred/ppf
  (correlation = 0.0000 in training), so it cannot steal their explanatory
  credit regardless of how it's estimated.
- **Verified this is a free fix, not a tradeoff**: LOOCV R2_w/RMSE_w are
  IDENTICAL between raw and residualized pitches_per_pa_prior (0.1638/1.5552
  forward, 0.0885/1.4258 reversed -- both directions byte-for-byte matched).
  This is expected by the Frisch-Waugh-Lovell theorem: `ppa_resid` is just
  `pitches_per_pa_prior` minus a linear combination of regressors ALREADY in
  the model, so the model's span (and therefore its predictions) is
  unchanged -- only the coefficient attribution is. With the residual:
  K_pred=-9.04 (p=0.006, MORE significant than even the original no-
  efficiency 2-feature model), `ppa_resid` itself keeps essentially the same
  coefficient the raw version had (-1.945 vs -1.894, p=0.071 vs 0.079) --
  the real efficiency signal is fully preserved, just no longer
  mis-attributed to the wrong variable.
- **Verified on the real 797-pitcher roster** (not just the n=96 training
  population): contribution-spread now shows K_pred (0.287) as the clear #2
  driver behind BB_pred (0.424), ahead of all three secondary regressors
  (ip_per_g_prior 0.256, ppf 0.199, ppa_resid 0.191) -- the sensible
  ordering for a skill-driven model, confirmed end to end, not just in
  theory.
- **Thin-sample handling**: for roster pitchers below the BF>=100 thin-
  sample guard, `ppa_resid` is set to exactly 0.0 directly (neutral: "no
  information beyond what K%/BB%/ppf already say"), NOT derived from the
  league-average raw `pitches_per_pa_prior` substitute -- that constant
  doesn't match any individual pitcher's own K/BB/ppf-implied expectation,
  so subtracting the aux model's prediction from it would produce a
  spurious nonzero residual.
- **Cost/risk accepted:** none identified -- this is a strict improvement
  over both the raw-pitches_per_pa_prior spec (fixes the collinearity/
  mis-attribution problem at zero accuracy cost) and over simply dropping
  the regressor (preserves its real, validated predictive contribution
  instead of discarding it).
- **Revisit if:** `ip_per_g_prior` is ever found to develop a similar
  collinearity problem as more data accumulates -- the same residualization
  technique would apply directly if so.

### pitches_per_pa: drop corrupted pitcher-game records (user-caught, SHIPPED)
- **Problem (user-caught):** after the residualization fix above, user
  looked at WHY Derek Salata still topped the leaderboard and asked
  pointedly whether the model was really saying his "inefficiency beyond
  K%/BB%" was a good thing. Investigated directly: his `ppa_resid` was the
  single highest of all 797 roster pitchers (99.7th percentile, hit the
  clip boundary). Traced to the raw pitch log: one game
  (`20260725-WintrustField-1`) logged 450 pitches all under a SINGLE
  `pa_id`, with ZERO `is_terminal` flags -- a tracking/export artifact, not
  a real appearance (dumps pitches into the `pitches_per_pa` numerator while
  contributing nothing to the PA denominator, inflating the ratio with no
  ceiling). Confirmed by excluding just that one game: his real ratio drops
  from 4.75 (the league's most extreme value) to 3.82 -- slightly BELOW the
  3.91 league average. He was never actually an outlier; the whole top-of-
  leaderboard story was a data artifact.
- **Checked scope before fixing**: swept the full 2025+2026 pool for the
  same signature (>=50 pitches in one game, 0 terminal pitches) -- found 4
  total pitcher-games across both seasons (Salata 2026; Ferguson/Gonzalez/
  O'Donnell in 2025), not systemic. Only `pitches_per_pa` is vulnerable to
  this specific bug -- `ip_per_g_prior`/`K_pred`/`BB_pred` come from
  `war_pitchers`' own external aggregate stats, a different data source
  entirely, unaffected.
- **Fix:** `pitches_per_pa()` now drops any pitcher-game with >=50 pitches
  and 0 terminal pitches before aggregating (`CORRUPT_GAME_MIN_PITCHES=50`).
  Applied uniformly to both the training-transition build and the full-
  roster build -- the training fit ALSO picked up the 3 previously-
  unnoticed 2025 cases once the filter was in place, refitting
  `ppa_resid`'s coefficient on a cleaner population (-1.945 -> -2.119,
  p=0.071 -> 0.051 -- more significant, not less, now that corrupted
  training data isn't in the mix either).
- **Verified end to end**: Salata's `ppa_resid` moved from +0.584 (max in
  the league) to -0.074 (near-neutral/average); his `proj_ra9_baseline`
  moved from 4.02 (leaderboard #1) to 5.33 (#37 of 797) -- a realistic,
  unremarkable position matching his actual underlying profile. Full-roster
  sanity sweep re-run: no new extreme values introduced, the pre-existing
  Brown/Dima bridge-quality cases (already flagged, unrelated) unchanged.
- **Cost/risk accepted:** `CORRUPT_GAME_MIN_PITCHES=50` is a judgment-call
  threshold, not a formally derived one -- chosen as "comfortably above any
  plausible real single-game pitch count" rather than tuned. A genuine
  (if extreme) complete-game effort could theoretically exceed it and get
  wrongly dropped, but the co-occurring "zero terminal pitches" condition
  makes that essentially impossible for a real appearance (a real outing
  long enough to need 50+ pitches will have faced multiple batters, each
  ending in a real terminal pitch).
- **Impact on model metrics was mixed, not uniformly positive** -- re-ran
  LOOCV in both directions after the fix. Forward (2025-actual ->
  2026-actual, the direction production actually uses): R2 0.1638 -> 0.1685,
  RMSE 1.5552 -> 1.5507 (both improved). Reversed (2026->2025, the
  non-causal direction used only as a sanity check): R2 0.0885 -> 0.0631,
  RMSE 1.4258 -> 1.4455 (both got worse) -- because Salata's corrupted 2026
  game had been acting as a spuriously helpful predictor INPUT in that
  direction. Reporting both honestly rather than only the favorable one.
  `RA9_RESID_SD` updated 1.5552 -> 1.5507 (the corrected forward LOOCV
  RMSE, since forward is what production uses) and re-shipped.
- **Audited the rest of the pitch-level feature pipeline for the same bug**
  (the "revisit if" below, resolved): the corrupted game's 450 rows are
  null on `pitch_type` and `False` on `is_swing`, in addition to
  `is_terminal`. Checked every other feature built from raw pitches --
  `compute_pooled_wh_sva` (-> `wh_sva_level`/Stuff+) filters to
  `is_swing`-only rows; `compute_spin_ext_features` (-> `spin_z_norm`,
  `extension`) and `compute_loc_var` (-> `horiz_var`, `vert_var`) both
  `dropna` on `pitch_type`. All three naturally exclude the corrupted rows
  as a side effect of filters that exist for unrelated reasons -- confirmed
  directly on the real Salata game, not just by reading the code. Only
  `pitches_per_pa()` was vulnerable, because it was the only feature that
  counted raw pitch ROWS without requiring `pitch_type` or `is_swing` to be
  populated. Conclusion: the K%/BB% production model (`fl_production_
  projection.py`) does NOT need to be re-fit or re-shipped; only RA9 was
  contaminated, and it already has been.
- **Revisit if:** this same corruption pattern (real-looking physical
  columns populated, but `pitch_call="undefined"`/null `pitch_type`/`False`
  `is_swing`/`False` `is_terminal`) is found in a FUTURE feature that, like
  `pitches_per_pa()`, counts or aggregates raw pitch rows without filtering
  on `pitch_type` or `is_swing` -- the audit above is a snapshot of the
  current pipeline, not a standing guarantee for code added later.

---

### Preston Johnson: stale cross-league bridge match blocked a far more relevant one

- **Found by the user**, from a real observation: Preston Johnson was
  acquired by a MiLB (AA) team mid-2026 and pitched very well there, but his
  2027 FL projection didn't reflect it at all. Traced to his match
  (`fl_k_rate_bridge=0.217`, `fl_bb_rate_bridge=0.117`, tagged `"college"`)
  not corresponding to ANY row in either the narrow college-bridge
  transition table or the broad name-matched college universe -- it was a
  leftover artifact from a now-overwritten, unrecoverable prior version of
  `fl_college_bridge_clean.parquet` (overwritten during an earlier,
  unresolved rebuild-correlation investigation this session, without a
  backup -- a real mistake, logged here so it isn't repeated). Meanwhile his
  real 2026 AA record (365 BF, K%=29.0%, BB%=9.0% -- a genuinely strong
  year) sat unused in `output/milb_pitching_2018_2026.parquet`, because the
  fixed college>Pioneer>MiLB priority order (`fl_bridge_matching.
  extend_match_by_other_leagues`) only ever tries Pioneer/MiLB when
  `has_college_match` is False -- once ANY college tag is set, true or not,
  stale or not, it's permanent and nothing fresher can override it.
- **Choice:** Replace the fixed-priority chain with
  `fl_bridge_matching.recency_weighted_match` -- for each pitcher, per
  metric (K%/BB% independently), compute weight = (that source's own
  validated held-out R2) x (TBF behind its rate estimate) x
  (`RECENCY_DECAY=0.5` ** years-stale relative to the roster build's most
  recent completed season), across college/Pioneer/MiLB, and use whichever
  single source scores highest. A fixed match is no longer forever --
  college only wins when it's actually the most relevant signal, not
  merely the first one tried.
- **Alternative(s) tried and rejected:** A straight weighted BLEND
  (probability-weighted average of all matching sources' predictions) was
  the original design -- tested first, held out WORSE than the existing
  fixed-priority chain on the project's Experiment-2 harness (K% R2_w 0.287
  vs 0.294, BB% 0.348 vs 0.353): diluting an already-good college match
  with noisier Pioneer/MiLB signal costs more than the added information is
  worth. Picking the single best-weighted source instead of blending fixed
  this.
- **Why (the real validation, `fl_other_league_bridge_test.py`, now a 3-way
  A/B/C comparison):** On the same held-out Experiment-2 check every other
  bridge change in this project has been judged against (n=260 test
  pitchers): single-best-source picking beat the fixed-priority chain on K%
  (R2_w 0.2937 -> 0.2951, RMSE_w 5.784 -> 5.778pp) and landed close-but-
  slightly-behind on BB% (R2_w 0.3530 -> 0.3488, RMSE_w 4.297 -> 4.311pp).
  Net effect on this specific held-out slice is close to a wash -- but this
  test population's coverage barely changes between the OLD and NEW method
  (both land at 479/797 matched on the real 797-pitcher roster, confirmed
  directly), because it's dominated by pitchers who were ALREADY
  well-matched by college; it doesn't contain many cases like Preston
  Johnson's (a real match that's wrong/stale, with a much better one
  available elsewhere), which is exactly the failure mode this change
  targets and the fixed-priority chain structurally could not express.
  Confirmed directly on the real case: Johnson now matches `"milb"` for
  both metrics (`fl_k_rate_bridge=0.217`, `fl_bb_rate_bridge=0.117`,
  `proj_K_rate=30.5%`, `proj_BB_rate=9.3%` -- a strong profile, matching his
  real 2026 form) instead of the old bogus `"college"` tag.
- **Decay rate (0.5):** No real sample of players who match >1 of
  college/Pioneer/MiLB at once exists to grid-search a cross-league decay
  on directly, so this transplants `fl_college_bridge.BB_MULTI_YEAR_DECAY`
  (the one real decay-rate sweep this project has run, on the within-
  college multi-year BB% average) rather than picking an arbitrary number.
  A decay sweep on the single-best-source design (0.0/0.2/0.5/0.8/1.0) found
  K% actually keeps improving as decay->1.0 (less staleness penalty: 0.222 /
  0.279 / 0.287 / 0.290 / 0.302) while BB% is flat (0.347/0.349/0.348/0.347/
  0.347) -- this specific test slice doesn't show decay<1.0 earning its
  keep, consistent with staleness not correlating with worse accuracy in
  the within-college check either (see below). Kept decay=0.5 anyway as the
  more defensible, validated-elsewhere default. **Revisit if** a real
  multi-source-overlap sample becomes available to test this directly.
- **Also fixed in the same pass (real bugs, found while investigating):**
  - **Name-collision data corruption, college**: `fl_college_bridge.
    load_all_college_years` grouped by `(name_clean, season)` alone,
    silently averaging together different real people who share a cleaned
    name -- confirmed directly, 137/11,788 name-season rows span >1 distinct
    FanGraphs `upid` (e.g. two different real "Alex Hernandez"es, SLU vs
    GT, same 2025 season). Now groups by `(upid, season)` first;
    `_latest_and_multi` and `build_college_prospect_projections` (both
    copies -- `fl_production_projection.py` and `fl_bridge_matching.py`
    had quietly diverged into near-duplicates) resolve to the single upid
    whose own history is most recent when a name still collides, instead of
    blending two careers into one fictional composite. Measured impact on
    the headline K%/BB% correlation was negligible (6/249 pairs affected,
    0.256->0.258 / 0.273->0.271) -- a real fix, but not the explanation for
    this bridge's broader correlation gap vs its older (now-unrecoverable)
    version, which remains unexplained (staleness and single-bad-cohort-
    year were also directly tested and ruled out).
  - **Same bug, MiLB**: `fl_milb_bridge.load_all_milb_years` had the
    identical vulnerability, actually worse (368/13,943 lowercase names
    span >1 distinct MLBAM id) -- fixed the same way, keyed on the real
    MLBAM `pitcher` id.
  - **Pioneer League**: same risk exists in principle, but the StatsCrew
    pull has no stable per-player id to resolve it against -- accepted as a
    residual, unfixed risk (small-n source, ~1% rate measured elsewhere).
  - **BB%'s informative prior was silently using only the NARROW bridge
    match**: `fl_production_projection.build_2027_projections` computed
    `blended_BB_rate` (the actual BB% model input) from `fl_bb_rate_bridge`
    BEFORE running the name-extended/Pioneer/MiLB match extension -- every
    pitcher who only matched via the broader chain (not the ~245-person
    narrow transition-pair table) was silently getting the LEAGUE-AVERAGE
    fallback blended in instead of their real bridge value, for BB% only
    (K% was unaffected -- it reads `fl_k_rate_bridge` later, already past
    this point). Fixed by moving the bridge match earlier, before
    `blended_BB_rate` is computed. This was already true before any change
    this session -- not introduced by the new matching logic, just
    surfaced while rewiring that section.
  - **`fl_halfseason_backtest.bridge_features` had drifted from
    production AGAIN**: was still calling the old fixed-priority
    `full_match` (the exact class of divergence this shared module was
    extracted to prevent -- see its own docstring) and a separate
    `Experiment 4` call site (`br4 = bridge_features(naive4["pitcher_id"])`)
    was missing required arguments entirely (a pre-existing, unrelated
    bug -- that code path hadn't been run recently). Both now call
    `recency_weighted_match` with an explicit `current_year` derived from
    each feature window's own end date, so a backtest's bridge features
    can't see a "future" season to itself.
- **Verified end to end:** re-ran `fl_production_projection.py` and
  `fl_ra9_trustmark_projection.py` on the real 797-pitcher roster: total
  bridge-match coverage unchanged at 479/797 (confirmed old and new methods
  land on the identical count -- this change reattributes WHICH source wins
  and WHEN, not how many players get matched); Johnson's case fixed as
  described above; dashboard data refreshed and reloaded with no console
  errors.
- **Revisit if:** a real test population containing more cases like
  Johnson's (stale/wrong primary match, clearly-better alternative
  available) becomes identifiable, to re-check the decay rate and the
  blend-vs-pick-best choice on a sample that actually exercises the failure
  mode this change targets -- the Experiment-2 harness used here mostly
  doesn't.

---

### Bridge age/sample-size follow-ups, and pulling more prior years (college blocked, Pioneer done)

- **College age, re-tested on the larger (bug-fixed) sample -- does NOT
  replicate.** The original "College age: tested as a bridge covariate, not
  added to the model" entry above found a borderline K% effect (R2 0.112->
  0.136, age coef p=0.080, n=113) and flagged "revisit if bridge sample
  grows." Re-tested now on n=175 (merging `fg_college_age_2023/2024/2025.
  parquet` against the current, upid-fixed n=249 transition pairs): K% age
  coef p=0.70 (R2 0.0855->0.0863), BB% age coef p=0.21 (R2 0.0812->0.0895,
  direction flipped negative). More data made the effect WEAKER, not
  stronger -- the original borderline signal doesn't replicate. Age stays
  out of the bridge.
- **Sample size (TBF) before the transition, college and MiLB: a real,
  replicated non-monotonic pattern, not yet explained.** Bucketed real
  transition pairs into terciles by the source-league TBF behind the K%
  input (`tbf_latest`): college K corr by tercile = 0.152 (low, mean
  TBF=61) / 0.484 (mid, TBF=161) / 0.091 (high, TBF=308); MiLB shows the
  SAME shape = 0.125 / 0.371 / 0.160. The middle tercile beats both low AND
  high in both independent sources -- not a simple "more data = better
  bridge" relationship. College BB% (TBF-weighted multi-year input) is
  flat across terciles (0.261/0.255/0.316) -- this pattern is specific to
  the single-year K% input. Plausible but UNCONFIRMED explanation: very
  high-TBF college/MiLB pitchers may be a different population (long-
  tenured organizational depth arms / multi-year college seniors) whose
  translation to FL doesn't follow the same relationship as a typical
  prospect. Bucket sizes (65-84) are thin enough that this could partly be
  noise -- not acted on, just flagged.
- **Sample size (FL PA) after the transition:** college K corr by PA
  tercile = 0.309 (low, mean PA=35) / 0.307 (mid, PA=108) / 0.201 (high,
  PA=274) -- mildly DECREASING, opposite of the naive "more FL PA = more
  reliable outcome = cleaner signal" intuition. Plausible explanation: a
  player's earliest FL performance sits closest in time to their college
  season, so the one-shot translation tracks it most closely; performance
  accumulated over a longer FL stretch reflects real in-season development/
  regression the static bridge input was never going to predict. MiLB's
  same cut is noisier/non-monotonic (0.214/0.143/0.262, n=64-65 per
  bucket) -- no clear story there beyond sample noise.
- **Pulled 2021-2022 Pioneer League data (user explicitly green-lit more
  prior years for college/Pioneer).** StatsCrew team-year pages fetched
  live for all 9 modern-era teams x 2 years (Grand Junction Jackalopes
  didn't exist yet in either year; Glacier Range Riders and Northern
  Colorado Owlz only existed starting 2022 -- confirmed directly, both
  years empty for those two under 2021). 421 real team-stint rows, real BF
  populated throughout (same as 2023-2025), aggregated into 399 new
  pitcher-seasons and concatenated with the existing 819-row 2023-2025
  panel -> `output/fl_pioneer_pitching_2021_2025.parquet` (1,218 pitcher-
  seasons, 2021-2025), built via `pioneer_2021_2022_build.py` (a one-off
  script parsing the live-fetched tables; `fl_pioneer_pull.py` still owns
  the original 2023-2025 pull and is unchanged). `fl_pioneer_bridge.
  PIONEER_PATH` repointed at the combined file.
  - **Real validated improvement**: refit on the richer history, n=68->76
    transition pairs (the extra prior years let a few more FL debutants
    find a multi-year-eligible match), K% R2 0.037->0.056, BB% R2 0.198->
    0.208. Modest but real, in the expected direction -- more history
    helps the TBF-weighted multi-year BB% input more than the single-year
    K% input, consistent with the asymmetric design already used
    throughout this project.
- **College data expansion: blocked, not done.** Attempted the same
  approach for college (FanGraphs currently only covers 2023-2026) --
  FanGraphs' `/leaders/college` page is now behind a Cloudflare Turnstile
  "verify you are human" challenge. Solving CAPTCHAs is a hard line this
  project does not cross regardless of instruction, so this wasn't
  attempted; college stays at 2023-2026 until/unless a legitimate path
  (e.g. the user solving the challenge themselves, or a different data
  source) becomes available.
- **Re-ran the full production pipeline** (`fl_production_projection.py`,
  `fl_ra9_trustmark_projection.py`) on the expanded Pioneer data; also
  fixed four more `bridge_features()` call sites
  (`fl_bootstrap_risk_tier.py`, `fl_low_pa_ci_calibration.py`,
  `fl_ros_early_test.py`, `fl_ros_bridge_prior_test.py`) that broke when
  `current_year` became a required argument earlier in this session --
  all were missed in the first pass since they're test/calibration
  scripts, not part of the main `fl_production_projection.py` call path
  that was actually re-run and checked.
- **Revisit if:** FanGraphs' college leaderboard becomes reachable without
  a CAPTCHA (a site change, or the user provides already-fetched data by
  another means), to backfill 2019-2022 the same way Pioneer was just
  extended.

---

### Dashboard: Player Page's Air% percentile direction was backwards

- **Choice:** Flip the Player Page's "Air%" metric from `higher_better=
  False` to `higher_better=True`, and relabel it "Air% (Trustmark fit)" to
  make clear what the direction actually reflects.
- **Why:** User caught this from two things already established this
  session: (1) "Is Air% related to RA9" found ~zero correlation between a
  pitcher's own Air% and their baseline RA9 (partial corr -0.04, p=0.61 --
  see that entry above) -- so "lower Air% is generally better" was never
  actually true for the player's own skill, just a plausible-sounding
  claim ("no grounder is ever a HR") that didn't survive contact with real
  data. (2) Air% DOES matter, but only through
  `build_trustmark_adjustment`, and in the OPPOSITE direction: both
  `hr_pct_baseline` and `nonhr_air_pct` are directly proportional to
  `proj_air_pct`, and both subtract into `trustmark_ra9_shift` -- so a
  HIGHER Air% mechanically produces a bigger (more negative, better)
  Trustmark shift (confirmed by reading the formula directly, not just
  inferred). TrustmarkPark suppresses HR/hit outcomes
  (`TRUSTMARK_HR_FACTOR=0.624`), so a fly-ball pitcher has more at-risk
  contact to BE suppressed -- the old framing had this exactly backwards.
- **Cost/risk accepted:** None identified -- this was a straightforwardly
  wrong direction, not a judgment call. The relabel to "(Trustmark fit)"
  is a small UX change to stop the metric from reading as a general skill
  percentile, since it only has a real, known direction in the Trustmark-
  specific context.

---

### Opponent quality (who the pitcher faced): tested directly, doesn't survive contact with the existing model

- **Choice:** Do not add an opponent-quality regressor to the K%/BB%
  production model.
- **Alternative(s):** A leave-one-out opponent-quality index -- PA-weighted
  average of each OPPOSING batter's own K%/BB% (excluding that batter's PA
  against the specific pitcher being scored, to avoid mechanical leakage) --
  as an added regressor, mirroring `mlb_opponent_kpct_explained.py`'s
  methodology (that script's own MLB-level run/finding was never logged in
  this project; this is the first real answer for FL specifically). Needed
  a fresh pull: `batter`/`batter_id` exist on the real Supabase `pitches`
  table but are NOT in `fl_pitcher_features.load_fl_pitches_cached()`'s
  column set (only `batter_side`/handedness is) -- pulled separately via
  `fl_opponent_quality_test.py`.
- **Why:** Two-stage test, same discipline as every other feature this
  project has evaluated.
  - **Concurrent association (is it related to anything at all):** real
    but weak -- corr(pitcher's own K_rate, opponent-implied K_rate)=0.098,
    BB_rate=0.154 (n=666 pitchers, full 2025+2026, >=20 PA). Directionally
    sensible (facing batters who strike out/walk more themselves nudges
    the pitcher's own rate the same way) but the opponent-implied index is
    heavily compressed (SD 0.90pp/0.65pp vs the actual rates' 7.02pp/
    6.61pp) -- expected, since it's an average of many noisy individual
    batter rates.
  - **The test that matters -- held out, incremental to the EXISTING
    model** (Experiment 2: fit full-2025->actual-1H2026, freeze, score
    2025+1H2026->actual-2H2026, opponent quality computed ONLY from the
    TRAIN window's own pitches): K% R2_w 0.3006->0.3027 (negligible,
    coefficient p=0.933 -- not significant); BB% R2_w 0.3517->0.3408
    (WORSE) despite the coefficient being significant in-sample (p=0.0064)
    -- the classic overfit signature: looks real on the fitting data,
    doesn't generalize. Once naive_K_rate/naive_BB_rate (the pitcher's own
    recent performance) and the other validated priors are already in the
    model, who they happened to face adds nothing real -- the concurrent
    correlation above is apparently already absorbed by those features,
    not independent signal.
- **Cost/risk accepted:** None -- rejected the regressor, not shipped.
  `fl_opponent_quality_test.py` kept as a reusable, documented negative
  result (and as the template if this is worth retrying on a much larger
  future sample).
- **Revisit if:** FL accumulates enough seasons that the opponent-quality
  index itself gets less noisy (less compression), or if a batter-quality
  metric with better signal (e.g. a shrunk/EB-stabilized rate rather than
  the raw leave-one-out rate used here) is built for the hitting side of
  this project and could be reused here.

---

### Arsenal interactions (five constructions, two sessions of work): consistently null on real FL data

- **Choice:** Do not add any arsenal-interaction feature to the K%/BB%
  production model.
- **Alternative(s) tested, each its own genuinely held-out check (same
  Experiment 2 harness: fit full-2025->actual-1H2026, freeze, score
  2025+1H2026->actual-2H2026, feature computed ONLY from the train
  window):**
  1. **Velocity differential** (`fl_arsenal_interaction_test.py`):
     fastball mean effective_velo minus offspeed mean effective_velo
     (FASTBALL_TYPES = four_seam/sinker vs. everything else). K% R2_w
     0.3006->0.3034 (p=0.657); BB% 0.3517->0.3435 (WORSE, p=0.460).
  2. **Arsenal depth** (same script): # distinct pitch types at >=5% usage
     and >=25 pitches, mirroring `mlb_arsenal_depth_wh_sva_aging.py`'s
     definition. K% 0.3006->0.3032 (p=0.811); BB% 0.3517->0.3438 (WORSE,
     p=0.176). Combined with velo_diff: K% 0.3006->0.3069, still neither
     term significant.
  2.5. **Release-point consistency across arsenal** (already logged above,
     "Release point as a K% regressor") -- a location-based deception
     proxy, found flat (p>=0.73) before this thread started.
  3. **Velocity Diversity+ / Movement Diversity+ / Repertoire+**
     (`fl_repertoire_diversity_test.py`) -- from the user-supplied article
     (medium.com/@mattjcreally, "Quantifying Arsenal Effects Using MLB's
     Play-by-Play Data"): PITCH-LEVEL (not pitch-type-mean) standard
     deviation of perceived velocity, and of Pythagorean combined
     movement sqrt(ivb^2+horz_break^2), pooled across every pitch thrown
     -- a genuinely different construction from (1), since it rewards an
     unpredictable mix across the WHOLE arsenal rather than a two-group
     gap. Weak correlation with wh_sva_level (-0.12 velo, +0.18 move) --
     not simply a Stuff+ proxy, per the article's own caveat, but also not
     independently useful: K% 0.3006->0.2956/0.2928 (BOTH WORSE, p=0.51/
     0.64); BB% 0.3517->0.3538/0.3500 (negligible/worse, p=0.44/0.56).
  4. **Tunneling+** (`fl_tunneling_test.py`, same article) -- the most
     novel and highest-effort construction: real pitch trajectory physics
     (`pos(t) = p0 + v0*t + 0.5*a*t^2`, the same kinematic model Statcast
     itself uses) to find each pitch's actual (X, Z) position 150ms after
     release, then the Pythagorean-combined spatial standard deviation of
     that position across a pitcher's full repertoire -- lower spread =
     pitches look more alike at the batter's decision point = better
     tunneling, per the article's own framing. Needed a fresh pull
     (x0/y0/z0/vx0/vy0/vz0/ax0/ay0/az0 exist on the real `pitches` table
     but aren't in `fl_pitcher_features.load_fl_pitches_cached()`'s column
     set). Values came back physically sensible (mean spread ~0.68ft at
     150ms) and not a Stuff+ proxy (corr with wh_sva_level=-0.10), but
     still null: K% 0.3006->0.3040 (p=0.523); BB% 0.3517->0.3412 (WORSE,
     p=0.838 -- not even close to significant, pure noise added).
- **Why this is a real finding, not just five failed attempts:** five
  independently-constructed mechanisms -- location-based deception
  (release point), speed-based deception (velo differential AND
  pitch-level velo spread), movement-mix unpredictability, pitch-count
  diversity, and genuine decision-point trajectory tunneling -- all come
  back null or mildly negative on the SAME held-out check. That's
  consistent enough across constructions to read as "FL's existing model
  (own recent performance + bridge + wh_sva_level/spin_z_norm/extension +
  location variance) already captures what's extractable from this data,"
  not "we haven't found the right arsenal feature yet."
- **Cost/risk accepted:** None -- all five rejected, none shipped. Every
  test script kept as a reusable, documented negative result.
- **Revisit if:** FL's pitch-level sample grows substantially (more
  seasons), which is the most likely reason any of these null results
  would flip -- several of the underlying std-dev-based metrics are
  probably still fairly noisy at FL's current per-pitcher pitch counts
  (most pitchers are well under 1,000 pitches in a train window).
- **Extended to RA9 (`fl_ra9_arsenal_test.py`), same five candidates, same
  LOOCV discipline RA9_SPEC itself was validated with (leave-one-pitcher-
  out, refit per fold, forward direction) on RA9's own real n=96 training
  population -- even more decisively negative.** Baseline LOOCV R2_w=
  0.1685/RMSE_w=1.5507 beat every single addition: velo_diff 0.1538/
  1.5644, arsenal_depth 0.1431/1.5743, velo_sd 0.1535/1.5647, move_sd
  0.1479/1.5699, tunneling_score 0.1536/1.5647 -- ALL worse on both
  metrics, no exceptions, and every coefficient p>=0.53 (full-sample
  in-sample fit, not even close to significant). Makes sense given n=96
  already fits 5 real regressors plus intercept -- there's no spare
  signal for a 6th term to find, consistent with these features also
  having none to offer. Across both models (K%/BB% and RA9) and five
  constructions, that's ten total held-out checks, zero real wins.
- **Sixth construction, same result: average release arm angle**
  (`fl_arm_angle_test.py`). `arm_angle` exists on the real `pitches` table
  (not in the cached pull) with real missingness (~49% null in the 2025
  window) that's roughly UNIFORM across pitchers (median per-pitcher
  coverage 53%, only 16/465 at 0% -- a per-pitch tracking gap, not a
  per-pitcher one), so a per-pitcher mean over whatever real readings
  exist is still a reasonable trait estimate for nearly everyone (394/465
  pitchers with >=20 readings). Not a Stuff+ proxy (corr with
  wh_sva_level=-0.02). Same two-model check: K% 0.3006->0.3026 (p=0.544);
  BB% 0.3517->0.3233 (WORSE, p=0.425); RA9 LOOCV 0.1685->0.1665 (WORSE,
  p=0.272). No significance anywhere, BB% takes a real hit. Six
  independently-constructed mechanics/arsenal features now, eleven total
  held-out checks across both models, zero real wins -- the same
  conclusion keeps replicating from a different angle each time.
- **Seventh and eighth: non-fastball Zone% and mechanical consistency**
  (`fl_offspeed_zone_mech_consistency_test.py`). Non-fastball Zone% is the
  direct complement of the already-rejected `fb_zone_pct` ("Fastball
  Zone% as a direct BB% regressor" above) -- same construction
  (`fl_pitcher_features.ZONE_HALF_WIDTH`-based in-zone flag), applied to
  OFFSPEED/breaking pitches instead of fastballs, testing a different
  mechanism (ability to throw secondary stuff for strikes, not just bury
  it as a chase pitch). Mechanical consistency = per-pitcher SD of
  `arm_angle` (release-angle repeatability) -- the natural "low spread is
  good" counterpart to Velocity/Movement Diversity+ above (which rewarded
  spread). Both tested on both models:
  - **off_zone_pct**: K% 0.3006->0.2981 (worse, p=0.290); BB% 0.3517->
    0.3478 (worse, p=0.511); RA9 LOOCV 0.1685->0.1506 (worse, p=0.798).
    Same fate as its fastball counterpart -- a plausible mechanism that
    doesn't survive once the existing model is already in place.
  - **arm_angle_sd**: K% 0.3006->0.2987 (worse, p=0.652); BB% 0.3517->
    0.3527 (negligible, within noise, p=0.658); RA9 LOOCV 0.1685->0.1626
    (worse, p=0.302).
  No significance anywhere, no metric improved by more than noise. Eight
  independently-constructed features, sixteen total held-out checks across
  both models, zero real wins.
- **Ninth/tenth: release point variance (fresh, current specs) and
  extension variance** (`fl_release_extension_variance_test.py`). Release
  point variance was previously tested only in pieces, on now-outdated
  specs (raw level + between-pitch-type "deception" variance for K% only
  in "Release point as a K% regressor" above; within-pitch-type variance
  noted elsewhere as "weak for BB%" on the pre-`horiz_var`/`vert_var` BB%
  spec) and never on RA9. Extension variance (release-distance
  consistency) had never been tested in any form -- only extension's MEAN
  is a shipped K% regressor. Re-tested all three (simple pooled per-pitcher
  SD, same construction as `arm_angle_sd`) fresh, on today's specs, all
  three models:
  - **release_z_sd** (vertical release height SD): null everywhere -- K%
    0.3006->0.3000 (p=0.746); BB% 0.3517->0.3522 (negligible, p=0.545);
    RA9 LOOCV 0.1685->0.1549 (worse, p=0.672).
  - **extension_sd**: also null -- K% 0.3006->0.3011 (p=0.331); BB%
    0.3517->0.3505 (worse, p=0.969); RA9 LOOCV 0.1685->0.1546 (worse,
    p=0.581).
  - **rel_side_sd** (horizontal release-side SD): the ONE exception in
    this entire investigation -- improved BOTH K% (0.3006->0.3068, p=
    0.088, the closest to significance of anything tested this thread) AND
    BB% (0.3517->0.3562, p=0.142), though it still hurt RA9 LOOCV (0.1685
    ->0.1590, p=0.845). Not clean enough to ship on its own (neither p
    clears 0.05, and it doesn't generalize to RA9), but genuinely
    different in character from every other null result here -- worth a
    real follow-up (e.g. a decay/stability check, or testing on a future,
    larger sample) rather than filing alongside the clean negatives.
- **Eleventh: first-pitch strike rate** (`fl_first_pitch_strike_test.py`).
  Computed per-PA from the real first pitch thrown -- `pitch_no` in the
  cached pull is a per-GAME cumulative counter, not within-PA (confirmed:
  `pa_id` alone recurs across up to 1,725 different games, so the first
  pitch of a PA is MIN(pitch_no) within each (pitcher_id, game_id, pa_id)
  group, not pitch_no==1). A real teaching case for why the INCREMENTAL
  test matters more than raw correlation: fps_rate's raw corr with
  naive_BB_rate is -0.627 (strong -- getting ahead 0-1 nearly mechanically
  predicts fewer walks), but once naive_BB_rate is already in the model it
  adds NOTHING (BB% 0.3517->0.3523, p=0.740) -- the correlation was never
  independent signal, just a proxy for the walk rate the model already
  has directly. K% 0.3006->0.2977 (worse, p=0.205); RA9 LOOCV 0.1685->
  0.1544 (worse, p=0.589).
- **Running total: eleven independently-constructed mechanics/arsenal/
  sequencing features, twenty-two held-out checks across two models, one
  genuine (if unresolved) maybe (rel_side_sd), zero clean wins.**

---

### OpenCommand-calibrated Command Score replaces horiz_var/vert_var in the BB% model -- genuine held-out win, SHIPPED

- **Choice (SHIPPED to production):** Replace `horiz_var`/`vert_var` as the
  BB% model's own regressors with a single `command_score` -- the same two
  numbers, translated into OpenCommand-equivalent inferred-miss inches via
  a FROZEN, externally-fit MLB regression, rather than two freely-fit FL
  coefficients. `horiz_var`/`vert_var` are kept everywhere else (display,
  `fl_pitcher_features.compute_loc_var`) -- only the BB% model's own two
  free parameters were replaced by one fixed, pre-calibrated one.
- **Where the calibration comes from:** github.com/tomdoyo/open-command --
  real computer-vision-derived distance from the catcher's actual inferred
  target (glove tracking on MLB broadcast video), published for 2024-2026.
  Pulled `command_scores.csv` for 2024/2025 directly from Hugging Face
  (`huggingface.co/datasets/tomdoyo/open-command`, no `huggingface_hub`
  install needed -- plain `curl` against the resolve URL works). Matched
  against `horiz_var`/`vert_var` computed IDENTICALLY to
  `fl_pitcher_features.compute_loc_var` but on this project's own existing
  MLB Statcast parquets (2024+2025, already pulled for other MLB-side
  research -- confirmed genuinely real data, not a placeholder: real 2025
  season date range, 2,506 real games, top-pitch-count leaders are real
  2025 workhorse starters like Logan Webb/Garrett Crochet).
- **Real contamination found and fixed before trusting any correlation:**
  the raw, unfiltered name-match (n=1,060) included real MLB POSITION
  PLAYERS making mop-up pitching cameos -- Amed Rosario (SS), Tucker
  Barnhart (C), Miles Mastrobuoni (UTIL), each with only 17-31 tracked
  pitches, sitting at the extreme "worst command" tail and distorting it.
  Refiltered to a real-workload floor (`n_pitches>=300 AND n_oc>=300`,
  n=769) -- after that, the extremes are exactly what real scouting
  reputation would predict: best command includes Tyler Rogers (literally
  OpenCommand's own README header example) and Aaron Nola (a well-known
  command specialist); worst includes Austin Adams and Felix Bautista,
  both documented for genuine real-world control problems. The filtered
  result (R2=0.68) was essentially unchanged from an earlier, looser
  n>=200 check (R2 0.81 vs 0.82 combined-corr) -- reassuring that the
  earlier number hadn't itself been contaminated, but the contamination
  was real and worth catching before trusting ANY number from the
  unfiltered match.
- **Z-scored regression resolved a real, separate mistake made mid-
  analysis:** raw-scale joint coefficients (horiz_var=3.94, vert_var=4.28)
  looked "nearly equal," which was reported as such -- WRONG. Z-scoring
  the two predictors (on the MLB training population's own weighted mean/
  SD) shows vert_var genuinely matters more: 0.70 inches of predicted miss
  per SD of vert_var vs. 0.45in/SD for horiz_var (~1.6x). The raw
  coefficients only looked similar because vert_var happens to have a
  wider natural range in this data -- a reminder that raw-scale
  coefficients across differently-scaled predictors are not a fair
  "which matters more" comparison, confirmed by the user catching it.
- **The actual held-out test (the real bar), both models:**
  - **BB% (Experiment 2, fit full-2025->actual-1H2026, freeze, score
    2025+1H2026->actual-2H2026):** replacing horiz_var+vert_var with
    command_score: R2_w 0.3517 -> **0.3610** (RMSE_w 4.302 -> 4.271pp),
    coefficient p=0.0478 (significant). Adding command_score ALONGSIDE the
    two originals (not replacing) gives a smaller gain (0.3577) --
    expected, redundant/collinear with its own ingredients. This is a
    real, if modest, win: a single EXTERNALLY-calibrated regressor (fit on
    MLB's much larger n=769 sample) beats two FL-locally-fit coefficients
    (n=134 training transition) -- a classic bias/variance story: fixing
    the combination weights trades a little bias (MLB's true weights may
    not exactly match FL's) for a real reduction in estimation variance on
    FL's much smaller sample, and the variance reduction wins here.
  - **RA9 (LOOCV on its own real n=96 population):** null, consistent with
    every other location-based candidate tested on RA9 this session --
    neither command_score (0.1685->0.1538) nor raw horiz_var+vert_var
    (->0.1587) help, neither coefficient significant (p=0.45, p=0.12/0.63).
    RA9 was NOT changed.
- **Cross-league extrapolation, checked and clipped:** FL pitchers run
  meaningfully higher horiz_var/vert_var than the MLB training population
  (FL mean 0.83/0.92 vs MLB's 0.61/0.71 -- FL's mean already sits past
  MLB's 75th percentile, as expected for a lower level of competition).
  58/757 FL pitchers fall outside MLB's horiz_var training range, 9/757
  outside vert_var's -- both clipped to the MLB-validated support before
  prediction, same discipline as the RA9 model's `clip_to_training_range`
  (`fl_pitcher_features.COMMAND_SCORE_HORIZ_RANGE`/`_VERT_RANGE`).
- **n0 reliability, checked (not assumed):** `mm.estimate_n0`'s binomial
  method-of-moments doesn't apply (command_score is a continuous,
  unbounded score, not a rate) -- used the project's OTHER established
  tool instead, `mm.empirical_reliability_curve`/`fit_n0_from_curve` (real
  split-half pairs, 1H2026 vs 2H2026, n=253 FL pitchers with real pitches
  on both sides). Fitted n0~=126 pitches. Checked whether this threatens
  the BB% result: the Experiment-2 test's actual train population has a
  median of 454 pitches behind command_score (full-season, not half-
  season), and the PA-reliability-weighted median (weighting by the SAME
  `reliability_weight(PA, n0_bb)` the WLS fit and R2_w metric already use)
  is 776 pitches -- 6x n0, ~86% reliable. The thin-sample noise n0=126
  warns about is concentrated in exactly the low-PA fringe pitchers the
  model was already discounting for an unrelated reason (their own BB_rate
  outcome is unreliable at low PA too) -- not a reason to doubt the result.
- **Shipped:** `fl_pitcher_features.compute_command_score` (frozen
  coefficients + MLB training-range clip); `fl_halfseason_backtest.
  loc_var_levels`/`MODEL_SPECS["naive+other_league+surprise_prior"]["BB"]`;
  `fl_production_projection.BB_SPEC_BLENDED`/`fit_frozen_model`/
  `build_2027_projections`/`project_from_college_stats`/
  `save_model_coefs`; `fl_bootstrap_risk_tier.py`'s impute list; the
  dashboard's `project_prospect` (Manual Calculator) and `FEATURE_INFO`
  (new "Command Score (in. from target)" row, lower=better). Re-ran the
  full pipeline (`fl_production_projection.py` -> `fl_ra9_trustmark_
  projection.py`), verified end to end in the live dashboard (Player Page
  shows a real example where horizontal consistency is mediocre at the
  22nd percentile but vertical is excellent at the 99th, and Command Score
  correctly reflects vertical's larger weight at the 96th percentile
  overall -- not just "it ran without an error").
- **Cost/risk accepted:** Rests on one real, unverifiable-from-FL-data
  assumption -- that the MLB-fit relationship between location variance
  and actual (glove-tracked) target distance generalizes to FL's different
  tracking setup and competition level. FL has no broadcast glove-tracking
  data of its own to check this against directly; the held-out BB% win is
  the best available evidence it transfers reasonably, not proof the
  cross-league translation is exactly right.
- **Revisit if:** FL ever gets its own real command-tracking data (glove
  location, target inference) to replace the MLB-borrowed calibration with
  an FL-native one; or if a future, larger OpenCommand pull (2026 data) or
  a longer FL pitch history meaningfully changes either the MLB regression
  or the n0 estimate.


---

## 2026-10-07 methodology review of the projections dashboard

Review requested before front-office use / private Streamlit Community Cloud
deploy. Each entry below lists the problem, the evidence, the choice, the
alternatives, and what would change it. Scripts: `fl_ra9_input_robustness.py`,
`fl_ra9_uncertainty_model.py`, `fl_league_drift_check.py`.

### RA9: Trustmark park effect was counted twice (SHIPPED)
- **Problem:** the MIS scenarios re-predicted the baseline with `ppf` swapped to
  a Trustmark blend (96.5 half / 93 full, FIP-based) AND added the explicit
  Air%-based HR + non-HR air-ball hit shift. The FIP factor already contains
  Trustmark's HR suppression, and the explicit shift is built from that same HR
  suppression.
- **Evidence:** the baseline model's implied `ppf` coefficient is 0.08 runs per
  point (p=0.045), i.e. -0.56 runs/9 for ppf 93 at full season, versus the
  explicit shift's -0.67 -- two independent estimates of the same effect that the
  old code summed (-1.2 full / -0.60 half). 96.5 is also below every real team's
  mean `ppf` (97.8-101.9), near the edge of the fitted range.
- **Choice:** MIS scenarios = baseline re-predicted at a park-NEUTRAL `ppf` (100)
  + the explicit shift, once. `ppf` keeps its original jobs: neutralizing a
  pitcher's observed RA9 context and the "Current team" column. Half-Trustmark
  mean effect is now -0.33 (was -0.60).
- **Alternatives:** (a) drop the explicit shift and rely on `ppf` only -- loses
  the pitcher-specific Air% dependence, the part that is genuinely new;
  (b) decompose the FIP factor into HR and non-HR parts and send only the non-HR
  part through `ppf` -- principled but needs a decomposition we don't have data
  for; (c) shipped option, simplest and consistent.
- **Side effect (finding 2 fixed):** the replacement baseline got only the
  explicit shift (-0.32) while the pitcher got shift + `ppf` swap (-0.60), which
  inflated P(>Repl.) under Trustmark. Now both get the same single mechanism.
- **Revisit if:** a per-park HR/non-HR split of the run park factor becomes
  available (option b).

### P(WAR>0): fixed SD replaced by innings-scaled, calibrated uncertainty (SHIPPED)
- **Problem:** one SD (1.55) for every pitcher. Out-of-sample on 312 pitchers
  (2025 inputs -> 2026 RA9, incl. 2026 debuts) its standardized-error SD was
  1.65 and its nominal-90% interval covered 76% (72% for no-history pitchers).
- **Fit:** Var(error) = a + b/IP + c(1 - r_prior), MLE. All the variance went to
  the luck term: b = 185.2 (90% CI 149-207), a = c = 0 (90% upper bounds 0.22 and
  0.68). New SD = sqrt(185.16 / IP): standardized SD 0.98, coverage 80-93% at
  nominal 90% across prior-information buckets.
- **What this means:** forecast uncertainty is dominated by single-season luck
  over the innings thrown, not by how little we know about the pitcher; the
  talent-level uncertainty is small and NOT separately identifiable at n=312.
  P(WAR>0) is therefore "P(RA9 over his expected IP beats replacement)", and thin
  workloads are pulled toward 50%. IP = 2026 IP, else 2025 IP, else 31 (median
  debut IP) -- NOT the old placeholder of 100, which made no-data pitchers look
  like full-workload ones.
- **Result:** no-history pitchers' mean P fell 0.744 -> 0.62; 0-30 PA pitchers now
  sit near 0.55.
- **Air% error (requested):** the Air% trait model's held-out error is propagated
  into the Trustmark scenario only (Air% enters nowhere else). LOOCV RMSE 0.0645;
  after removing the binomial noise of the 2026 target the talent-level error SD
  is 0.0443 (population SD 0.0723 for a pitcher with no tracked pitches, ramped by
  feature reliability). Because the shift is proportional to Air%, the added RA9
  error is shift x (air_sd / air_pct): at most +0.002 runs/9 on the SD -- included
  as asked, confirmed negligible.
- **Alternatives:** talent-level P with a structural prior (would need the
  unidentified a, c; taking their upper bounds makes the no-info pitcher look
  MORE certain, not less); keep the fixed SD (miscalibrated).
- **Not applied:** no-history pitchers' RA9 forecasts are +0.40 runs too good
  (IP-weighted, se 0.23, ~1.8 sigma; shrinks to +0.04 at 40+ IP). Suggestive
  (debuts are fringe arrivals), not significant -- no correction applied.
- **Revisit if:** a third season doubles the sample and a/c become identifiable.

### RA9 baseline: raw-trained vs projection-trained inputs, out-of-sample (NO CHANGE)
- **Question:** the shipped fit trains on raw prior-season so/bf and bb/bf but is
  scored on shrunk model projections. Risk: attenuated coefficients applied to
  already-shrunk inputs (double shrinkage).
- **Test:** K%/BB% projections cross-fitted (10-fold) so no pitcher's projection
  saw his own outcome; LOOCV of the RA9 spec, n=79, both directions.
  Forward (2025 -> 2026): raw->raw R2_w 0.186; raw->proj (SHIPPED) 0.189, calib
  slope 0.93; proj->proj 0.203, slope 0.82. RMSE_w diff shipped - proj->proj
  +0.013, 90% bootstrap CI [-0.030, +0.054]. Reversed (2026 -> 2025): 0.055 /
  0.054 / 0.051, differences ~0.
- **Choice:** keep shipped (raw-trained). No measurable harm from the mismatch;
  proj->proj is marginally better forward and marginally worse reversed.
- **Observation:** calibration slopes 0.64-0.93 mean RA9 predictions are slightly
  over-dispersed; an extra shrink toward the mean is a candidate, not tested for
  significance.
- **Revisit if:** more transitions allow a decisive comparison.

### Duplicate pitcher names: resolved by evidence, not blanket refusal (SHIPPED)
- **Problem:** 16 names appeared twice in the roster (Player Page and leaderboard
  link keyed on name opened the wrong pitcher; per-year K%/BB% split across ids).
- **Evidence (arsenal distance, handedness, date/game overlap):** 15 of 16 are
  the same person on two ids (sequential dates, same hand, arsenal distance
  0.13-0.51 vs same-person 99th pct 0.83 and different-pitcher median 1.05).
  `Davis, Tyler` is two people (opposite hands, distance 1.26).
- **Choice:** `build_canonical_id_map` keeps the legacy same-team rule and adds a
  cross-team rule: same name + same hand + no shared date/game + arsenal distance
  <= 0.80 (when >= 7 features comparable). Roster 797 -> 782. Dashboard now keys
  the Player Page/leaderboard on `pitcher_id` (team shown for shared names). The
  frozen K%/BB% fit still uses the ORIGINAL ids, as before.
- **Flagged for a manual look (legacy-merged, rule would not have merged them):**
  `Wilson, Bradley` (arsenal 1.01, 118 pitches in 2026) and `Hughes, Grif` (one
  shared date and game).
- **Also:** the RA9 script merges `war_pitchers` by NAME; names still shared after
  consolidation (Davis) now get no war row (league-average workload) instead of
  both getting the same blended row.

### Gap prior for pitchers with a 2025 line but no 2026 PA (SHIPPED, flagged weak)
- **Problem:** 299 of 797 had no model projection (naive prior is 2026 only).
- **Evidence:** returns after a one-year gap are rare (24/758, 3.2%, flat across
  prior-PA buckets). Lag-2 persistence is weaker (K slope 0.56, BB 0.70 vs lag-1
  0.86/0.87); equivalent n0 ~230 K, ~150 BB vs ~60 lag-1. The gap-only sample is
  n=18, so no gap-specific constants are estimable.
- **Choice:** naive rates built from 2025 counts discounted by n0/230 (K) and
  n0/150 (BB), then run through the same frozen model. 2025-derived stuff features
  (wh_sva, spin, extension; command_score) are shrunk toward the mean by 0.65 /
  0.80 (the lag-2/lag-1 ratios) -- an ASSUMPTION extended from the rate
  persistence. Flagged `proj_basis="gap_prior"` and shown with a caution on the
  Player Page. Shown as 0 2026 PA; excluded from the recentering weights.
- **Alternatives:** keep bridge/league mean only (ignores real data; costs little
  since ~3% return); gap-specific fit (impossible at n=18).

### League-level drift: projections recentered to the latest league average (SHIPPED)
- **Evidence:** league BB% 10.4% (2025) -> 11.4% (2026), K% 19.1% -> 19.4%; BB% by
  half 9.8% (2H25) -> 11.7% (1H26) -> 11.1% (2H26). The model's PA-weighted
  projected mean sat +0.58pp (K%) / +0.70pp (BB%) above the 2026 league level.
  Held-out Exp-2: K% bias +0.63pp -> -0.02pp and R2_w 0.301 -> 0.310 when recentered
  to the latest window; BB% 0.3665 -> 0.3703 when recentered to the pooled feature
  window but 0.3404 when recentered to the (noisy) 1H2026 window alone.
- **Choice:** subtract a constant so the PA-weighted mean projection (2026 PA
  weights) equals the full-2026 league rate. Shifts K -0.58pp, BB -0.70pp, stored
  in the coefs JSON and applied by the Manual Calculator too. Assumes 2027 league
  level = 2026 level.
- **Caveat:** one held-out transition; BB% evidence is mixed (a recentering
  target built from one half-season hurt). A full-season target is more stable
  than a half-season one but the choice remains a judgment call (`RECENTER` flag).
- **Revisit if:** 2027 data shows which direction the level moved.

### Dashboard/pipeline consistency fixes (SHIPPED)
- Manual Calculator vs leaderboard: `fl_projection_model_coefs.json` stored league
  rates pooled over 2025+2026 (BB% 0.1092) while the roster pipeline used 2026-only
  (0.1142); now 2026-only everywhere, plus the recentering shift.
- Bootstrap risk tier removed from the production pipeline (user: P(WAR>0)
  deliberately replaced it); `fl_bootstrap_risk_tier.py` left in place, unimported.
- Leaderboard "Training PA" relabeled "2026 PA"; text updated.
- **Housekeeping left alone on purpose (user: keep until completely done):** stray
  `data"` files and unused `plotly` in `projections-dashboard/`. New: backup of the
  pre-review data in `projections-dashboard/_backup_data_before_review/` -- delete
  before deploying.
- **Deployment:** Streamlit Community Cloud deploys from GitHub; use a private repo
  and restrict viewer access to the app.


---

## 2026-10-07 follow-up: Jackson Smith, ppa_resid, and the value of other-league history

Scripts: `fl_ppa_resid_check.py`, `fl_ppa_decomposition.py`,
`fl_called_strike_foul_skills.py`, `fl_bridge_value_deepdive.py`,
`fl_milb_signee_analysis.py`. NO production code changed in this pass -- the
open decisions are listed at the end of each entry.

### ppa_resid drives Jackson Smith's RA9 projection -- and is a weak, partly mis-built regressor (OPEN)
- **Observation:** Smith projects 6.55 RA9 (30th pct) despite average K-BB% and an 84th-pct
  IP/G. Contributions vs the roster mean (runs/9): K% +0.04, BB% -0.04, park +0.04,
  IP/G -0.22, `ppa_resid` +0.51 (4th pct; he throws 3.66 pitches/PA vs ~3.90 implied).
  With `ppa_resid` = 0 he projects 6.03 (64th pct); halved, 6.29 (46th).
- **Stability:** `ppa_resid` YoY corr 0.27 overall, 0.36 same-team, -0.07 for team
  changers (n=25); team ICC ~0.03.
- **Decomposition** (exact identity: pitches/PA - 1 = fouls + non-terminal called strikes +
  swinging strikes + balls, per PA; residualized on K/BB/ppf, components sum to `ppa_resid`):
  balls 50% of variance (YoY 0.10), fouls 34% (0.18), called strikes 11% (0.51), whiffs
  5% (0.56). The stable pieces have no RA9 signal; the unstable pieces (84% of the
  variance) carry the forward gain, with the counter-intuitive sign (more fouls/balls per PA
  -> lower RA9: -4.0, -3.4 runs/9 per +1 pitch/PA).
- **Cleaner skill versions** (location-model called strikes above expected, with and without a
  shrunk catcher effect; foul share of contact; whiffs/swing): YoY 0.35-0.62 but team-changer
  stability poor for called strikes (-0.20; ICC 0.21 -> -0.04 after catcher adjustment).
  RA9 value inconsistent: called-strikes-above-expected helps forward (+0.03 R2, p=0.59) and
  flips sign reversed (p=0.03); foul share and whiff rate add nothing either way.
- **Out-of-sample value of the shipped term (LOOCV, n=79):** forward R2 0.139 (none) -> 0.186
  (shipped); reversed 0.052 -> 0.055. Halved: 0.178 forward / 0.064 reversed.
- **Smith's actual profile:** called strikes above expected 85th pct (77th catcher-adjusted),
  foul share 26th, whiffs per swing 0th, fewer balls (7th) and fewer non-terminal whiffs
  (3rd) per PA than expected -- a contact-heavy, call-getting profile.
- **Possible bug (production-only):** the auxiliary regression behind `ppa_resid` is fit
  against RAW prior-season K/BB (orthogonal by construction in training) but fed the SHRUNK
  projected K/BB at scoring time. On the roster (n=258, BF>=100) `ppa_resid` correlates +0.15
  with projected K% and +0.26 with projected BB% (raw: +0.22/+0.29), so the term
  partially offsets the BB penalty and K benefit it was meant to be independent of. Using
  observed 2026 K/BB in the auxiliary step (as in training) cuts the BB correlation to +0.12 but
  overshoots K to -0.15..-0.20 (the 2025-fit aux coefficients also drift in 2026). Effect of the
  switch on RA9: mean +0.07, SD 0.17 runs/9, rank correlation 0.97; Smith 6.55 -> 6.44.
- **Options:** keep as is; halve; drop; and/or fix the auxiliary input. Recommended: fix the
  auxiliary input AND halve the term. Not applied pending the user's call.

### Value of other-league history (College / Pioneer / MiLB): modest, uneven, mostly BB% (OPEN)
- **Held-out ablation, Experiment 2 (n=260):** removing the bridge terms costs K% R2_w 0.301 ->
  0.278 (SSE +3.1%) and BB% 0.367 -> 0.350 (+2.6%). The benefit sits in bridge-matched pitchers
  (SSE -4.2% K, -4.0% BB) with 31-250 feature-window PA; unmatched pitchers gain ~0, and pitchers
  with <=30 PA (n=27) do not gain.
- **Experiment 1 (n=122):** K% bridge HURTS (R2_w 0.082 without -> 0.001 with; SSE +8.9%), BB%
  flat (+0.7%). The K% evidence is inconsistent across the two held-out checks.
- **For genuine FL debuts (the population the bridge is fit on), leave-one-out vs league mean:**
  K% R2_w 0.03-0.05 in every league (college 0.048, Pioneer 0.032, MiLB 0.029); BB% R2_w college
  0.054, MiLB 0.155, Pioneer 0.175. BB% depends strongly on the sample behind the source rate:
  MiLB with >=150 TBF R2_w 0.223, <150 TBF -0.221; college >=150 TBF 0.094, <150 -0.007.
- **Implication:** BB% bridge is useful when the source sample is real (MiLB/Pioneer >=150 TBF);
  the K% bridge is close to noise. The BB% prior currently weights the bridge by a constant
  n0_bridge (~53 PA) regardless of the source sample, which the debut data suggest is too
  generous for thin source samples. Candidate (untested): scale the bridge's pseudo-PA by source
  TBF reliability, and make the K% bridge term optional.

### Preston Johnson / FL -> affiliated MiLB (informational)
- **Timeline (MLB Stats API transactions + game logs):** released by the Orioles Aug 2025; FL
  (Mississippi) 2026-05-07..13: 39 PA, K 43.6%, BB 0%; signed by the Twins 2026-05-18;
  Wichita AA from 2026-05-22: 365 BF, K 29.0%, BB 9.0%. All the 2026 MiLB stats post-date the
  FL stint.
- **Test (34 FL pitchers who reached affiliated ball after an FL stint; game-log dates used so
  "after FL" really means after the last FL game; 11 name-matched, unverified):** FL K%/BB%
  (shrunk to the FL mean) predicts post-FL MiLB K%/BB%. K% slope 0.42 (90% bootstrap CI 0.16-0.64,
  R2 0.155); BB% slope 0.66 (CI 0.10-1.27, R2 0.11); both stay positive after controlling for
  pre-FL MiLB (K 0.32, BB 0.71) though only ~8 pitchers have pre-FL MiLB history. The FL-only
  model predicts Johnson's MiLB line at K 26.5% / BB 8.9% vs actual 29.0% / 9.0%. So FL dominance
  in a tiny sample is informative once shrunk, and Johnson's result is consistent with it.
- **Direction matters for the dashboard:** the dashboard uses MiLB -> FL. Johnson's 2027 FL K%
  (30.6%) is driven by the 39-PA FL line (+4.4pp via the naive prior) and Stuff+ 150 (+5pp), NOT
  by his MiLB record: the MiLB K% bridge (slope 0.20, fit on mostly released/fringe arrivals,
  R2 ~3%) adds only +0.7pp over a league-average bridge. The bridge's selection (players sent to
  FL) differs from Johnson's (signed away from FL), so it likely understates his MiLB evidence.
- **Caveats:** n=34, clustered pitchers, unverified name matches, MiLB level mix.

### ppa_resid routed through K%/BB% and contact quality (user hypothesis): fails both tests (OPEN -- recommend dropping from RA9)
- **Hypothesis (user):** pitch-count information is believable only if it feeds the K%/BB%
  projections (plate-discipline skill) or reflects quality of contact; a direct RA9 term is
  otherwise suspect. Script: `fl_ppa_kbb_and_contact.py`.
- **Test 1 -- predicts FUTURE K%/BB% beyond the shipped models (held-out, paired bootstrap on
  weighted squared error):** Exp2 K%: +pitches/PA SSE -3.0% [90% CI -5.6, -0.5] (worse), 4
  components -6.3% [-15.1, +1.4]; BB%: -0.7% / -4.7% (worse). Exp1: K% +0.1% (nil) / -4.4%; BB%
  -0.7% / -9.1% (worse). The in-sample t-value for fouls/PA on K% was 3.66 in the Exp2 train fit
  and -0.07 in Exp1's -- an in-sample artifact that does not survive holdout.
- **Test 2 -- contact quality (exit velo, hard-hit%, xwOBAcon, wOBAcon, wOBA-xwOBA gap, air%
  allowed):** same-year partial correlations (metric residualized on K%, BB%) for total ppa_resid
  are -0.01..-0.16 in 2025 and -0.02..-0.06 in 2026 (inconsistent). Predictive (2025 residual ->
  2026 metric, controlling the metric's own 2025 value, n=91): total ppa_resid |t| <= 1.3 on all six.
  Only component-level hits: swinging-strike component (non-terminal whiffs/PA beyond K%/BB%)
  -> next-year hard-hit% t=-2.2 and wOBAcon t=-2.2, and same-year xwOBAcon -0.24/-0.15; fouls
  -> hard-hit% t=+2.1. With 30 tests ~1.5 spurious |t|>2 are expected, and that whiff component is
  ~5% of ppa_resid variance and has no RA9 coefficient (p=0.78).
- **The RA9 gain itself is not statistically distinguishable from zero:** LOOCV RMSE_w (no term)
  minus (shipped) = +0.042 forward, 90% CI [-0.015, +0.100]; +0.002 reversed, CI [-0.051, +0.055].
- **Conclusion:** by the user's own criterion the term has no believable channel (not K%, not BB%,
  not contact quality), the underlying components are unstable (YoY 0.10-0.18 for the 84% that is
  fouls + balls) and the sign is counter-intuitive. Recommend REMOVING `ppa_resid` from the RA9
  baseline (Smith 6.55 -> ~6.03). Cost: -0.047 forward LOOCV R2_w (0.186 -> 0.139), not significant.
  Alternative: keep halved (retains 0.178 forward). Not applied; awaiting the user's decision.

### ppa_resid REMOVED from the RA9 baseline; xwOBAcon-allowed and age tested as replacements (SHIPPED removal)
- **Choice (user decision, 2026-10-07):** RA9 baseline is now `RA9 ~ K% + BB% + ppf + ip_per_g_prior`
  (`RA9_SPEC_PROD`); `ppa_resid`, its auxiliary regression and the Player Page's "Prior-Season
  Pitches/PA" row are gone. Legacy `RA9_SPEC`/`PPA_RESID_SPEC`/`CONTINUOUS_REGRESSORS` names are
  kept only so the experiment scripts that reproduce earlier results still import. Re-ran the
  pipeline: roster RA9 baseline mean unchanged (-0.002), SD of the per-pitcher change 0.19 runs/9,
  rank correlation 0.94; Jackson Smith 6.55 -> 6.03 (64th pct), P(>Repl.) 0.75. Refit
  coefficients: K% -8.88, BB% +12.33, ppf +0.080, IP/G -0.352 (all p<=0.05).
- **Contact-quality-allowed as a replacement channel (`fl_ra9_new_sources_test.py`):** YoY
  stability (BIP>=40, n=104): exit velo 0.27, hard-hit% 0.10, xwOBA on contact 0.07, air% 0.48,
  wOBA-xwOBA gap -0.08. LOOCV RA9 (n=79): base R2_w 0.139 forward / 0.052 reversed; adding
  EB-shrunk xwOBAcon allowed gives 0.112 / 0.026 (n0 = 50, 100, 200 identical); exit velo 0.089 /
  0.056, hard-hit% 0.103 / 0.049, air% 0.109 / 0.034, gap 0.131 / 0.039. None helps, so the
  earlier RA9-pivot "real, modest lift" for xwOBA allowed (R2_w 0.160 -> 0.173, in-sample single
  season pair) does not survive forward/reversed validation. Contact quality allowed is
  essentially not a persistent pitcher skill at FL sample sizes except air%.
- **Age:** `output/pitcher_bio.parquet` is keyed on MLBAM ids; only ~1% of the RA9 sample has a
  birth date, so age is not testable without new data.
- **Context for further additions:** single-season luck dominates RA9 at FL workloads (fitted luck
  variance 185/IP; the expected luck variance over the n=79 fit sample is about as large as the
  total observed variance of RA9, so the achievable R2 ceiling is small and the LOOCV R2 of
  0.14-0.19 is already near it). Gains from more regressors are bounded; larger levers are the
  workload (innings) projection and the park/team treatment.

### 2027 innings projection, validated on five seasons of FL history (SHIPPED)
- **Problem:** P(WAR>0) used the pitcher's LAST-season IP as next season's workload, with a luck law
  (Var = 185/IP) calibrated on one transition. Script: `fl_innings_projection.py` (StatsCrew 2022-24
  + war_pitchers 2025-26 = four transitions, 543 retained pitcher-transitions).
- **Findings:** next-season IP (given he pitches again) regresses hard: OLS slope 0.64/0.63/0.67/0.59
  by transition (stable), pooled `IP_next = 19.3 + 0.628 * IP_prior` (resid SD 29.5, R2 0.33).
  Leave-one-transition-out: RMSE 29.7 vs 32.0 for "next = prior" (R2 0.32 vs 0.21). A second prior
  season adds nothing (RMSE 28.1 vs 28.1, n=139). SP share helps modestly (CV RMSE 30.8 -> 29.7) but
  exists for ONE transition only -- not shipped. Retention is 32-36% every year (logit
  -2.29 + 0.54 ln IP; 16% at <=10 IP, ~55% at 80+); first-season median IP is 18 (n=1,107); pitchers
  returning after a skipped season (n=22) have median IP 24.
- **Luck law re-validated:** K%/BB%-only RA9 model, leave-one-transition-out, broad sample (n=452):
  b = 135 / 174 / 179 / 210 per transition, pooled 184.7 (shipped 185.16), a ~ 0, standardized-error
  SD 0.95-0.99 per transition. With PROJECTED IP in the denominator, b = 185 still gives 50/80/90%
  coverage of 0.50/0.79/0.90 (z-SD 1.13, heavy tails); refit b = 242 over-covers (0.56/0.85/0.92).
- **Choice:** `ip_proj = 19.31 + 0.6278 * IP_2026` for pitchers with 2026 innings; 24 for post-gap
  returners; 18 for debuts; SD = sqrt(185.16 / ip_proj). The dashboard now also shows P(returns).
  Debut/post-gap IP assumptions are NOT covered by the validation (transitions start from pitchers
  who pitched the prior season).

### Park-neutral RA9 + Mississippi translation; Air% projection upgrade (SHIPPED)
- **User direction:** Mississippi is the team; use the Mississippi translation to separate a park-neutral
  projection from the Mississippi projection, with consistency for Mississippi pitchers. Scripts:
  `fl_park_term_test.py`, `fl_air_pct_projection_test.py`.
- **Park term:** `war_pitchers.ppf` disagrees with the FIP park factors the Trustmark adjustment uses
  (team-level corr 0.61; Mississippi pitchers average ppf ~101 although half their games are at a
  factor-93 park); `war_pitchers.pf` tracks the FIP factor at corr 0.99, and so does a game-by-game
  home-park exposure built from the pitch data (Mississippi pitchers 97.7 vs the expected 96.5 at 47%
  home). None predicts RA9 out-of-sample (LOOCV n=79, forward / reversed R2_w: none 0.151 / 0.045,
  ppf 0.139 / 0.052, pf 0.122 / 0.031, own game-mix 0.121 / 0.031).
- **Choice:** RA9 baseline = K% + BB% + IP/appearance (no park term) = PARK-NEUTRAL; Mississippi RA9 =
  neutral + the explicit Air%-driven TrustmarkPark shift (half or all home games), identical arithmetic
  for every pitcher including Mississippi's own. "Current team" columns removed. Replacement and
  pitcher both get the same shift mechanism. Refit coefficients: K% -9.24, BB% +11.53, IP/G -0.378.
- **Air% (drives the shift):** traits-only LOOCV R2_w 0.198 forward / 0.225 reversed -> traits + own
  BIP-shrunk (n0=150) Air% 0.234 / 0.249 (t 2.6 / 2.2); n0 40..300 all help in both directions. Held-out
  talent-level error SD 0.0443 -> 0.0410 (propagated into the Mississippi P(WAR>0)).
- **Not done / caveat:** non-Mississippi parks get no park-specific treatment now (the user's focus is
  Mississippi); a per-park Air% x HR-factor generalisation remains the natural extension.

### Bridge weighting by source sample: BB% no change, K% small consistent gain (OPEN -- recommend K% only)
- **Test (`fl_bridge_weight_test.py`; free parameters chosen on the TRAIN transition; paired bootstrap):**
  BB% pseudo-count scaled by source TBF (n0_i = n0max * tbf/(tbf+k), k 100-800) vs the shipped constant:
  SSE +0.2% [-2.5, +2.8] (Exp2) and -0.1% [-1.0, +0.7] (Exp1) -- no difference (median matched TBF is
  ~270-320, so scaling barely moves anyone). No bridge at all: -2.6% / -0.7% (the bridge helps BB%
  modestly).
- **K%:** bridge term x tbf/(tbf+k): +0.5% [0.0, +1.1] (Exp2) and +3.8% [-0.8, +8.3] (Exp1) vs the
  shipped free term; never worse. Dropping the K% bridge entirely: -3.2% (significantly worse, Exp2)
  but +8.1% (significantly better, Exp1) -- the K% bridge is regime-dependent, and TBF scaling is a
  compromise that is at least as good as shipped in both.
- **Recommendation:** leave BB% as is; adopt K% TBF scaling with k=200 (needs the bridge TBF carried through
  `recency_weighted_match` -- added as additive columns `k_bridge_tbf`/`bb_bridge_tbf` -- the K model
  refit, and a "batters faced" input on the Manual Calculator). Not applied pending the user's call.

### Other-league bridge: recency and league audit; unified input policy tested (OPEN -- recommend adopting for the prospect translation)
- **Scripts:** `fl_bridge_recency_league_study.py` (audit + grids), `fl_bridge_unified_test.py` (unified policy,
  pair LOO + held-out harness). 802 FL debuts (2025+2026); pairs: College 249, Pioneer 73, MiLB 191 (513 total).
- **Audit -- the three bridges are built differently:** seasons College 2023-26 (FanGraphs), Pioneer 2021-25
  (StatsCrew, NO 2026), MiLB 2018-26 (the repo's file comes from the MLB Stats API via `pull_milb_pitching.py`, not
  FanGraphs); K% = latest single season for all (unshrunk; median TBF 126-162); BB% = TBF x 0.5^lag (college) vs flat
  TBF-weighted (Pioneer, MiLB); college fit is plain OLS while Pioneer/MiLB are PA-weighted WLS; name collisions
  resolved by upid (college), MLBAM id (MiLB), not at all (Pioneer); source seasons per pair: gap 1/2/3+ = 179/50/20
  (College), 50/14/9 (Pioneer), 83/40/68 (MiLB: 36% of pairs have a source 3+ years old); multi-year share 62% /
  23% / 81%; MiLB match is by FL id == MLBAM id for 166/191 pairs. The roster pick among leagues weights by each
  league's in-sample R2 (K ~0.04-0.07) x TBF x 0.5^(2026 - source season); the staleness never enters the mapping itself.
- **TBF shrinkage is the dominant lever; decay is second-order.** Shrinking the source rate toward its league mean by
  n0 batters faced (LOO R2_w, one pooled line): K% 0.037 -> 0.055-0.060, BB% 0.112 -> 0.185 (n0=50); per league BB%:
  College 0.054 -> 0.143, Pioneer 0.166 -> 0.194-0.215, MiLB 0.155 -> 0.253. With shrinkage in place the decay choice
  moves R2 by <=0.02-0.04: K% prefers little multi-year (d~0.25; College/Pioneer latest-only, MiLB d~0.25-0.5), BB%
  prefers flat (d=1.0; MiLB), College BB% indifferent.
- **Staleness is real and the shipped mapping ignores it:** under shipped inputs the slope on a source 2+ years old is
  about half (College K 0.37 -> 0.12, BB 0.61 -> 0.14; MiLB K 0.31 -> 0.16, BB 0.47 -> 0.31); pooled slope-change t =
  -2.3 (K) / -2.9 (BB). After shrinkage the K% effect persists (damping rho~0.6-0.8 per extra year lifts pooled R2_w
  0.059 -> 0.070) but the BB% effect disappears (rho=1.0 best).
- **League identity matters little once rates are treated consistently:** pooled LOO R2_w, K% / BB%: one line for all
  leagues 0.046 / 0.115; league intercepts 0.043 / 0.109; league intercepts + slopes (the shipped structure) 0.037 /
  0.105. Residual league shifts under the unified inputs are <= 0.9pp.
- **Unified policy tested** (n0=50, decay K 0.25 / BB 1.0, K damped by 0.7^(gap-1), BB undamped, one pooled mapping,
  pick the source with the highest reliability): debut pairs, own-line LOO R2_w College K 0.048 -> 0.089, BB 0.054 ->
  0.136; Pioneer K 0.033 -> 0.085, BB 0.166 -> 0.194; MiLB K 0.030 -> 0.043, BB 0.155 -> 0.253. Pooled mapping: K%
  FL = 0.195 + 0.49 u (R2_w 0.069), BB% FL = 0.111 + 0.67 u (R2_w 0.196).
- **Held-out harness (vets with FL history; contains almost no true newcomers):** K% Exp2 -1.9% SSE [-2.8, -1.1],
  Exp1 +8.6% [+4.6, +12.4]; BB% Exp2 +0.1% [-2.2, +2.0], Exp1 -1.9% [-4.1, +0.2]: a wash for veterans. The prospect
  translation (Other-League Lookup, Manual Calculator) is a debut-style use, so the pair LOO is the relevant test.
- **Open data questions:** (1) Pioneer 2026 (StatsCrew) is missing, so 2026 Pioneer performers cannot be seen and
  Pioneer's best possible gap is 1 while College/MiLB can be 0; (2) the user described MiLB as FanGraphs -- the repo
  uses the MLB Stats API; confirm which source is intended; (3) the 17-24 pitchers per league pair who appear in two
  leagues are too few to fit a cross-league combination.

### Unified vs by-league bridge: head-to-head (`fl_bridge_unified_vs_byleague.py`)
- **2x3 design on the same 513 debut pairs (LOO, PA-weighted; % reduction in squared error vs each league's own-mean guess;
  RMSE_w pp vs guess 6.00 K% / 5.35 BB%):** K%: shipped inputs + by-league lines 3.9% (5.89pp); shipped + pooled line 3.9%;
  unified inputs + by-league lines 7.0% (5.79); unified + pooled line 7.1%; unified + pooled line + league intercepts 7.8%
  (5.76, best). BB%: shipped by-league 11.2% (5.04pp); unified by-league 19.1%; unified + pooled line 20.2% (4.77, best);
  + league intercepts 19.7%. Inputs (TBF shrinkage, one decay per metric, K% staleness damping) supply most of the gain
  (+3.1 pts K%, +7.9 pts BB%); pooling adds +0.1 / +1.1 and mostly helps Pioneer (K 8.5 -> 12.4, BB 19.4 -> 23.2, n=73).
- **Out-of-year check (parameters chosen on ONE debut year, scored on the other; % SE cut vs the training-year mean):**
  K%: 2025 -> 2026: shipped 1.1 vs unified 4.6; 2026 -> 2025: 3.9 vs 4.4. BB%: 2025 -> 2026: 4.1 vs 21.6; 2026 -> 2025:
  16.8 vs 19.5. Picked parameters are stable for BB% (n0 25-100, flat decay, no damping); K% picks vary but are always
  positive. The fixed policy (n0 50; K d .25 + 0.7 damping; BB flat) scores 5.6 / 6.6 (K) and 23.0 / 20.9 (BB) -- slightly
  optimistic because its parameters were chosen on all pairs.
- **Roster impact (474 matched pitchers):** unified K% bridge values are more compressed (SD 1.17pp vs 1.73pp), BB% similar
  (2.43 vs 2.50); corr with shipped 0.69 / 0.68; median |change| 0.66pp K% / 1.08pp BB%, 90th pct 1.70 / 2.83pp, max 6.9 /
  25.6pp (thin-sample rates now shrunk); the winning league changes for 4% (K%) / 7% (BB%) of matched pitchers. Examples:
  Preston Johnson K 21.7% -> 21.8%, BB 11.7% -> 12.3%; Matt Stil BB 13.9% -> 15.7%; Jackson Smith unchanged.
- **Scale caveat:** the K% bridge stays weak in absolute terms (RMSE 6.00 -> 5.79pp); BB% is the useful one (5.35 -> 4.77pp).
- **MiLB source (user, 2026-10-07):** "MiLB can come from wherever is easiest" -> keep the MLB Stats API pull
  (`pull_milb_pitching.py`, 2018-2026, MLBAM ids; 166/191 debut pairs matched by id). No FanGraphs MiLB pull needed.

### Unified other-league bridge IMPLEMENTED (SHIPPED)
- **User approval:** "let's implement" (2026-10-07). MiLB stays on the MLB Stats API ("wherever is easiest"). Pioneer 2026 is NOT loaded:
  the Pioneer League site export the user supplied covered only 22 pitchers with batters faced (page 1 of an ERA-sorted
  table, 103 zero-IP rows) and the user then retracted it; Baseball Reference has the data but acquisition there has been
  restricted before, so it was not attempted. The Pioneer Lookup page says data runs through 2025.
- **Code:** new `fl_unified_bridge.py` (policy: n0=50 batters, K% decay .25 + 0.7 staleness damping, BB% flat, pooled mapping with
  K% league intercepts, highest-reliability source, MiLB matched by MLBAM id when the FL id is one); `fl_halfseason_backtest.
  bridge_features` now routes through it (`BRIDGE_IMPL = "unified"`, `"legacy"` kept for reproducing earlier results);
  `fl_production_projection.py` fits it, builds the three prospect tables from it, uses it for the roster match, and writes
  `bridge_unified` coefficients to the JSON (the old per-league bridge keys are gone); `fl_bridge_matching.py` /
  `fl_halfseason_backtest.py` gained additive `k_bridge_tbf` / `bb_bridge_tbf`. Dashboard: lookup tables carry gap / batters faced /
  reliability and the page shows them; Manual Calculator takes batters faced (K% season and total) and years-since inputs and
  uses the same arithmetic (`fl_unified_bridge.translate`; the pipeline asserts calculator == lookup).
- **Fit (518 debut pairs: College 249, MiLB 196 [id-matched where possible], Pioneer 73):** K% intercepts college 0.188 / pioneer
  0.195 / milb 0.205, slope 0.510 (LOO R2_w 0.070); BB% = 0.1105 + 0.671 u (LOO R2_w 0.198). Roster re-fit: K% bridge coefficient
  0.280 (was 0.283), BB% n0_bridge 43 (was 53). Roster K%/BB% projections barely move (corr with the previous version 0.997 /
  0.988; SD of change 0.26pp / 0.46pp, max 2.0 / 5.4pp); matched pitchers 478/782. Pre-change code, data and parquet snapshots:
  `FLHitting/_code_backup_before_unified_bridge/`.
- **Verified in the dashboard:** Calculator defaults (K 20%, BB 9%, 250 BF each, gap 1) -> 18.9% / 11.2%; Lookup (MiLB) Kade Anderson
  28.0% K / 7.0% BB with the reliability and batters-faced caption; no console/server errors.

### Pioneer League 2026 loaded from Baseball Reference (SHIPPED, partial coverage)
- **Source:** the user pasted nine Baseball Reference Pioneer team pitching tables (13 pastes; four exact duplicates dropped) after
  the first site export (22 pitchers with batters faced, page 1 of an ERA-sorted table) was retracted. Saved as
  `output/pioneer_2026_raw/pioneer_2026_br.csv`; `fl_pioneer_2026.py` builds `output/fl_pioneer_pitching_2026.parquet`
  (same schema as the 2021-25 file); `fl_pioneer_bridge.load_all_pioneer_years`, the unified bridge and the Pioneer lookup
  history now include 2026. Pioneer roster matches rose from 63 to 82 (K%) / 52 to 68 (BB%); the Pioneer prospect table has 295
  pitchers whose latest season is 2026 (gap 1, so no K% staleness damping). Example: Cole Calnon (42 K / 15 BB in ~416 BF)
  translates to 15.3% K / 6.3% BB.
- **Coverage:** 295 pitchers, ~37,400 batters faced (K% 17.6%, BB% 12.9%) from 9 of the 12 teams. The tables carry no team label; matching
  against the earlier export's team names identifies Long Beach (A), Billings (D), Oakland (E), Missoula (F), Boise (G),
  Yuba-Sutter (H), RedPocket (I); tables B and C are therefore Glacier and Modesto in some order, and the three teams with no table
  are inferred to be Great Falls Voyagers, Idaho Falls Chukars and Ogden Raptors (their pitchers from the earlier export --
  Rogers, Smiley, Stancato, Lane -- appear in no table). Awaiting the user's confirmation.
- **Batters faced are ESTIMATED** (the BF column is blank on those pages): BF = outs + H + BB + HBP. Against six pitchers with true BF it runs
  0-3% high (Mayer 196 vs 195, Calnon 416 vs 415, Rosengard 33 vs 33, Cermenelli 6 vs 6, Dayton 222 vs 218, Romo 241 vs 234); the n0 = 50
  shrinkage absorbs that.
- **Reconciliation:** only tables B and C sum exactly to their printed totals rows; the other seven differ both ways (listed/printed outs
  0.92-1.10), a Baseball Reference quirk -- not row-level errors: every pitcher cross-checked against the independent export matched
  exactly. Printed in the loader as an informational check.
- **Update -- all 12 Pioneer teams now loaded (user pasted the three missing tables):** J = Great Falls Voyagers (Rogers, Gauna), K = Ogden
  Raptors (Stancato, Lane, Valentine, Jordan), L = Idaho Falls Chukars (Smiley) -- confirming the elimination inference. Totals now:
  405 pitchers, ~50,800 estimated BF (K% 17.1%, BB% 13.1%); Pioneer prospect table 405 pitchers with a 2026 season; roster bridge matches
  497/782 (Pioneer is the K% source for 91 pitchers, was 63 before 2026 data). Tables B and C remain Glacier Range Riders / Modesto
  Roadsters in an unknown order (no pitcher in either could be placed); team labels only affect the lookup display. Printed-total
  reconciliation for J/K/L: 0.96 / 0.96 / 1.10 (listed outs / printed). Dashboard text now reads "all 12 teams".
- **Team labels finished (user, 2026-10-07):** table C (Bartkoski/Longshore) = Glacier Range Riders, table B (Alaniz/Serrano) = Modesto Roadsters. All
  12 Pioneer 2026 teams are now labeled in the lookup history; labels are display-only (the bridge keys on name and rates).

### Trustmark HR factor: the shipped 0.624 understates the park's HR suppression (OPEN -- recommend 0.446) -- `fl_trustmark_air_validation.py`, `fl_trustmark_hr_factor_check.py`
- **Measured, all 80,437 batted balls (4,766 at TrustmarkPark, both seasons):** HR per air ball 0.0260 at Trustmark vs 0.0694 elsewhere = ratio 0.375
  (2025: 0.40, 2026: 0.34); non-HR hit rate on air balls 0.320 vs 0.318 (no difference); wOBA on contact 0.349 vs 0.392 (-0.043).
- **Same-team home/road (removes "Mississippi's players are different"):** Mississippi pitchers 0.387 [90% CI 0.28-0.53] (41 vs 87 HR), Mississippi batters 0.361
  [0.25-0.53], visiting pitchers 0.305 [0.22-0.42], visiting batters 0.455 [0.35-0.59]. Every group sits at 0.31-0.46 and no CI reaches 0.624.
- **Empirical Bayes over the 18 parks:** the true between-park SD of the log HR factor is 0.30 (Poisson noise SD 0.08), so a park's raw ratio keeps ~86% of its
  signal: Trustmark 0.390 raw (66 HR vs 169 expected) -> 0.446. The official `park_factors.pf_hr` table is a compressed version of the raw ratios (slope 0.53 across 16
  parks, corr 0.77) and lists Trustmark with n_seasons = 1.0 (heavily regressed), 0.624.
- **Implication for the Mississippi shift (roster mean, runs/9):** HR factor 0.624 (shipped): half-season -0.310, full -0.620; EB 0.446: -0.432 / -0.863; raw 0.390:
  -0.471 / -0.941. Cross-check: the measured wOBA-on-contact gap alone (-0.043 x 23.8 batted balls per 9 IP / the league wOBA scale) is ~-1.0 runs/9 (FL wOBA scale 1.018) for an
  all-home-games pitcher, in line with or above the EB figure. Other (more regressed) estimates sit lower: official pf_runs 0.95 (n_seasons 1), FIP-based park_factor 93 / raw 90.
- **Air% interaction not verifiable:** out-of-sample by 2025 Air% tercile, the HR-per-batted-ball gap at Trustmark is -0.019 / -0.037 / -0.025 (low / mid / high), no
  monotone trend; the groups hold only ~5-15 Trustmark HR each. And the Air%-driven part is small anyway: the Mississippi shift has SD 0.046 runs/9 across the roster
  (10th-90th pct -0.377..-0.265) around -0.32, i.e. 8% of the spread in projected RA9; the Mississippi adjustment is nearly uniform across pitchers.
- **Proposed change (not applied):** `TRUSTMARK_HR_FACTOR` 0.624 -> 0.446 (EB), keeping the 0.0073 non-HR hit term. Moves the Mississippi RA9 level ~0.12 runs/9 lower
  (half-season); pitcher and replacement shift together so P(WAR>0) barely moves, but the absolute Mississippi RA9 column and any runs-saved-by-park valuation do.

### P(WAR>0) calibration and league-level drift (read-only checks; no change) -- `fl_pwar_calibration.py`
- **Calibration, fully out-of-sample 2025 -> 2026, broad sample n=312 (IP>=10), current design (cross-fitted K%/BB%, park-neutral RA9 model, projected innings,
  SD = sqrt(185.16 / IP_proj), replacement 6.963):** overall predicted 0.654 vs actual 0.599. By bin predicted -> actual: 0.57 -> 0.55 (n=49), 0.645 -> 0.564 (211),
  0.74 -> 0.73 (22), 0.84 -> 0.84 (19), 0.94 -> 1.00 (6). IP-WEIGHTED actuals (the weighting that matters for WAR): 0.635 / 0.652 / 0.805 / 0.881 / 1.00, i.e. calibrated or
  slightly under-confident. Sliced by REALIZED workload the unweighted rate swings from 0.40 (10-25 IP) to 0.91 (90+ IP): innings are endogenous (poor performers get cut,
  good ones keep pitching), which the pre-season probability cannot know. Discrimination is modest: Brier skill +2.3% vs the base rate, mean P 0.666 for pitchers who beat
  replacement vs 0.636 for those who did not; logistic recalibration intercept -0.32, slope 1.13 (se 0.35).
- **League level by season (StatsCrew 2022-24 + war_pitchers 2025-26):** K% 21.1 / 20.2 / 19.9 / 19.2 / 19.6; BB% 9.8 / 10.5 / 10.0 / 10.6 / 11.5; K-BB% 11.3 -> 8.1
  (a 3.2pp decline in four years). Year-to-year change SD 0.6pp for both. Forecasting next year's level: "last year" RMSE 0.51pp (K%) / 0.67pp (BB%) beats "mean of
  all prior years" 0.87 / 0.80 and "last 2 years" 0.66 / 0.72 -- supporting recentering to the LATEST season's level (as shipped), not a multi-year average. BB% rose in 3 of 4
  years (+0.65, -0.43, +0.56, +0.92pp); if that trend continues the 2026 level under-forecasts 2027 BB% by ~0.5pp. Only four transitions, so not adopted.
- **SHIPPED (user: "add it", 2026-10-07):** `TRUSTMARK_HR_FACTOR` 0.624 -> 0.446 in `fl_ra9_trustmark_projection.py`; the 0.0073 non-HR hit term is unchanged; module docstring
  and constant comment updated. Re-ran the RA9 stage and refreshed the dashboard data. Roster result (PA>=30): half-season Mississippi shift -0.445 runs/9 (SD 0.064; was -0.319),
  full-season -0.890 (was about -0.64); Mississippi RA9 mean 5.82 vs park-neutral 6.27; the replacement baseline shifts with it (6.521 at half-Trustmark), so P(WAR>0) is unchanged
  (mean 0.630 park-neutral and Mississippi); Jackson Smith 6.05 park-neutral / 5.66 Mississippi, P 0.72 / 0.71. K%/BB% projections are unaffected.

### Innings projection: role (starter share / IP per appearance) NOT added -- keep IP-only (DECIDED, no change)
- **Question (proposal #4):** should `IP_proj = 19.31 + 0.6278 x IP_prior` (and the retention logit on ln IP) also use role -- innings per appearance or GS/G -- so a
  starter with 60 IP and a reliever with 60 IP stop getting the same 2027 innings?
- **Evidence in hand (single transition only, 2025 -> 2026, CV):** next-season IP RMSE 30.8 (IP only) vs 29.5 (+ IP per appearance) vs 29.7 (+ starter share GS/G);
  the starter-share coefficient was +27.6 IP (t = 3.9); the prior-prior season added nothing. ~1.1-1.3 IP of RMSE on a 30-IP error is a small gain, from one fold.
  In war_pitchers 2025-26 about 32% of pitcher-seasons are starter-type and 49% pure relievers, so the role split is real, just not large in IP terms.
- **Prepared but not used:** pulled games / starts for the 2022-24 StatsCrew pages (`pull_statscrew_games_2022_2024.py` ->
  `output/fl_statscrew_pitching_games_2022_2024.parquet`; 1,192 rows, IP reconciles exactly with the earlier pull; GS is blank for 2022, complete for 2023-24) so role
  could have been validated on four leave-one-transition-out folds (starter share on three). The multi-fold test was never run.
- **Decision (user, 2026-10-07: "projecting innings alone will suffice, no need to stress over role"):** keep the shipped IP-only projection and the ln-IP retention
  logit. Why: P(WAR>0) uses innings only through SD = sqrt(185.16 / IP_proj), so the role effect moves the SD slightly and nothing else, and the leaderboard already
  carries IP per appearance through the RA9 baseline.
- **Alternative left open:** IP_next ~ IP + IP/G (+ GS/G), validated leave-one-transition-out with the 2022-24 games file above; the only reason to revisit is if
  starter / reliever innings errors turn out to be visibly lopsided on the Mississippi targets list. The unrun test script was deleted; the data pull script and parquet
  are kept (harmless, reusable).

### Mississippi Targets view added to the dashboard (SHIPPED) -- `projections-dashboard/src/app.py` (`render_targets`, `_target_frame`, `_targets_table`)
- **What it is:** a new sidebar view that ranks Frontier League pitchers as Mississippi acquisition candidates. No model or data change: it reads the same
  `proj_ra9_trustmark_half` / `prob_above_replacement_trustmark_half` / `ip_unc` columns the leaderboard uses, so a candidate and a current Mississippi arm are
  measured on the same Mississippi (half games at TrustmarkPark) basis. Click a row to open the Player Page (by pitcher_id, as on the leaderboard); a collapsed
  table shows the Mississippi staff for reference.
- **Ranking metric (the real choice).** Options: (a) MIS RA9 -- rate only, ignores workload, so a 20-IP reliever can outrank a 100-IP starter; (b) P(WAR>0) -- noisy for
  thin workloads by design (pulled to 50%) and saturates near 90-95% for every good starter; (c) **runs vs replacement = (Mississippi replacement RA9 6.52 - his MIS RA9) x
  projected IP / 9** -- one number carrying both quality and expected workload, in the units WAR is built from. Default sort is (c); (a), (b) and projected K-BB% are selectable.
  Top of the list: Cook (Schaumburg) +25.3 runs, Brothers +22.0, Escobar +20.7. Across the 357 default candidates the metric correlates 0.94 with P(WAR>0), -0.85 with RA9, 0.74 with IP.
- **Innings used = `ip_unc`** (the same regressed-from-2026-workload number behind P(WAR>0)), which assumes he pitches at all; the chance he returns (`p_return`) is NOT multiplied in,
  because a target who signs is assumed to pitch. Role is deliberately not modelled (see the innings-role entry above), so a reliever and a starter with equal 2026 IP get
  equal projected IP.
- **"Rank on MIS staff":** where his MIS RA9 would slot among Mississippi pitchers with >= 30 2026 PA (20 pitchers; staff median MIS RA9 6.11, replacement 6.52, median P(WAR>0) 56%).
  Caveat shown on the page: this is last season's staff, not a 2027 roster, and the 30-PA floor lets thin-sample arms (e.g. Preston Johnson, 39 PA) set the top slots.
- **Defaults and filters:** minimum 2026 PA 30 (matches the leaderboard and the percentile reference floor); Mississippi's own pitchers hidden (checkbox to include); pitchers with no 2026
  appearance (293, 2025-only discounted prior, weak evidence) hidden, and when included they bypass the PA slider (the floor is about 2026 playing time, which they don't have --
  first version had the box doing nothing at the default slider; fixed). Team multiselect and sort selector added. Counts check out: 357 candidates + 20 own = 377; + 293 = 670.
- **Not included (alternative left open):** other-league arms (College / Pioneer / MiLB). The existing lookup page translates them to K%/BB%/K-BB% only; ranking them on Mississippi RA9
  would need the RA9 baseline coefficients exported to `fl_projection_model_coefs.json` and a neutral Air% / IP-per-appearance assumption. Also not in the data: contract status and
  availability, so the page says a listed pitcher may not be gettable.
- **Verification:** streamlit AppTest across all five views, 0 exceptions; every filter and sort exercised; row click opens the right Player Page (Cole Cook, MIS RA9 4.28 matches the table).

### Why P(WAR>0) shows only +2.3% Brier skill, and how much any fix could add (read-only diagnostic; no change) -- `fl_pwar_weakness.py`
- **Question (user: "weak by Brier skill"):** is +2.3% (n=312, 2025 -> 2026, AUC 0.587) a model defect or the nature of the target?
- **Ceiling simulation** (real pitchers, real projections, replacement 6.963, realized innings; outcomes drawn inside the shipped uncertainty law,
  RA9 = projection + talent error N(0, a) + luck N(0, 185.16 / IP); 400 draws): if the shipped P were exactly right (a = 0) it would score
  **+2.1% Brier skill / AUC 0.598 -- the real +2.3% / 0.587 is what the model itself predicts.** An ORACLE that knew every pitcher's true talent would reach
  +3.9% (a = 0.22, the 90% upper bound from the variance fit) to +8.6% (a = 0.68, the loosest bound); AUC 0.63-0.68. So no talent-side improvement can lift single-season
  Brier skill past roughly 4-9%.
- **Why it is capped:** IP-weighted SD of projected RA9 across the pool is only 0.47 runs/9, against a luck SD of 2.2 at the median realized IP (1.36 at 100 IP). Most
  pitchers sit 0.55-0.78 in P, so a binary outcome barely separates them.
- **The ranking still works on the scale that matters (IP-weighted, 2026 actuals):** by P quintile, actual runs vs replacement per 9 IP: 0.41 / 0.61 / 0.73 / 0.89 / 1.43 (monotone);
  IP-weighted share beating replacement 0.62 -> 0.82. RA9 forecast RMSE (IP-weighted): model 2.05 vs league mean 2.09 on all 312; on pitchers with >= 20 IP in 2025 (n=94) model 1.58 vs
  league mean 1.73 vs LAST YEAR'S RA9 2.24 (worse than the league mean); correlation with actual 0.36 vs 0.19.
- **Where P is informative (by projected innings, known pre-season):** < 25 IP (n=205, two thirds of the sample): skill -0.9%, AUC 0.58; 25-45 IP (n=42): -3.9%, AUC 0.47; 45-70 IP (n=43): +5.4%, AUC 0.62;
  70+ IP (n=22): +3.7%, AUC 0.68. The overall figure is dragged down by cup-of-coffee arms whose outcomes are nearly pure luck (small cells; directional).
- **Single-transition worry on the RA9 baseline, partly answered:** raw prior-year K% / BB% slopes by transition: K% -11.1 / -8.5 / -4.5 / -7.3, BB% +4.4 / +14.0 / +19.1 / +15.9 (each with se 3.3-5.8);
  pooled over four transitions (n=314): K% -9.03 (se 1.7), BB% +13.04 (se 2.6). Shipped -9.24 / +11.53 (projected inputs, 2025 -> 2026 only) sit within one se of the pooled values. No change.
- **Options (none applied):** (1) accept -- it is at its ceiling; (2) report the metric the front office uses (runs vs replacement, IP-weighted / restricted to >= 45 projected IP) instead of
  all-pitcher Brier skill; (3) add a multi-season or talent-level probability (lower luck, but the talent-error term is not identifiable, so it cannot be validated and would read as overconfident);
  (4) look for more RA9 signal (prior tests of ppa_resid, age, opponent quality, arsenal interactions, xwOBAcon-allowed were all null).

### HITTERS, step 1: data panel and how predictable each hitter stat is (read-only foundation; no dashboard change yet) -- `fl_hitter_panel.py`, `fl_hitter_baseline_backtest.py`, `pull_statscrew_batting_2022_2024.py`
- **Data assembled (1,817 hitter-seasons, 2022-26):** 2022-24 from the BATTING table on the already-cached StatsCrew team-year pages (1,092 team-stints, no new network pulls);
  2025-26 from Supabase `lineup_stats` (PA/AB/H/TB/BB/HBP/SO/SF/SH, pipeline wOBA, xwOBA), 2B/3B/HR counted from `pitches`, `war_players` (WAR parts) and `advanced_splits` (Trackman skill
  metrics: EV, EV90, barrel%, hard-hit%, chase%, whiff%, zone contact%, pull-air%, xDamage; 97% of hitters). Traded players summed across stints; keyed by cleaned name (cross-team
  duplicate names NOT yet audited for hitters, as was done for pitchers).
- **One wOBA definition for all five seasons:** counts x the project's FL linear weights (`woba_weights`: BB .811, HBP .842, 1B .945, 2B 1.251, 3B 1.534, HR 1.863). Reconciles to the
  pipeline's own 2025-26 wOBA at correlation 0.9994 / 0.9999 (mean diff +0.0007 / +0.0002). Caveat: IBB only exists in the 2022 StatsCrew lines (treated as ordinary BB in 2023-24).
  Alternative (use the pipeline's event-level wOBA where it exists) rejected: it would make 2022-24 and 2025-26 differently measured.
- **League environment moves a lot:** league wOBA .361 / .359 / .346 / .358 / .374 (2022-26), HR/PA 2.8 / 2.6 / 2.1 / 2.1 / 2.6%, K% 21.3 -> 19.6, BB% 9.8 -> 11.5. League K% and BB% match the
  pitcher side exactly (19.2 / 19.6 and 10.6 / 11.5 in 2025 / 2026). Same lesson as pitchers: recenter to the league level, don't project raw rates.
- **Predictability (leave-one-transition-out, 4 transitions, next-year PA >= 100, n=423):** year-over-year correlation of league-relative rates: K% .66, BB% .52, HR/PA .57, ISO .55, HBP% .52,
  BABIP .12, wOBA .22. Regression constant n0 (PA of league-average performance mixed in): K% 72, BB% 145, HR/PA 141, ISO 118, BABIP 864, wOBA 443 -- stable across held-out
  transitions (K% 60-82, BB% 139-152). RMSE vs next year: wOBA league mean .0467, last-year raw .0684 (much WORSE than the mean), regressed .0436; K% .0629 / .0551 / .0444;
  BB% .0365 / .0392 / .0284. A prior-prior season (weight 0.5-0.75) trims RMSE slightly (wOBA .0436 -> .0434); not a big lever.
- **wOBA direct vs from components:** wOBA built from separately regressed K%, BB%, HR/PA and BABIP: RMSE .0405 vs direct regressed .0410 on the same rows (n=302); a 50/50 average .0402.
  Components win modestly, so the plan is to project components and combine, not to regress wOBA itself.
- **League drift cost:** using last year's league level instead of the target year's adds ~1-4% RMSE (wOBA .0436 -> .0446); league wOBA changed -.002 / -.013 / +.012 / +.016 across transitions,
  i.e. larger and less one-directional than K%/BB%. The 2027 league level is a real unknown for hitters (same open issue as pitchers).
- **Trackman skills, 2025 -> 2026 only (n=114, PA >= 50 / >= 100):** correlation with next-year wOBA: xwOBA .41 vs own wOBA .35; barrel% .36, EV90 .29, BB% .28, chase% -.24. 10-fold RMSE: league mean .0429,
  wOBA .0404, **xwOBA .0387**, wOBA + xwOBA .0389; adding K%/BB% or EV90/chase%/zone-contact% did NOT help (.0396). So contact-quality (xwOBA) is a genuine signal over results, but one transition
  and n=114 cannot support a many-skill model -- same single-transition limit as the pitcher Stuff+ features.
- **Non-batting parts of hitter WAR (2025 -> 2026, PA >= 100 both years, n=104):** fielding runs y/y corr **.83** (SD ~6 runs; OAA-based, mostly outfielders), batting runs .41, positional adjustment .30,
  WAR .48. Baserunning runs exist for 2026 only (missing for 2025), so its persistence cannot be tested. Among 168 hitters with PA >= 200 in 2026, 81% had WAR > 0 (mean batting runs +2.9, replacement runs
  12.6 -- hitters' WAR>0 is far easier to clear than pitchers' 60%), so P(WAR>0) will be near 1 for regulars and the useful hitter output is expected runs/WAR, not that probability.
- **Forks surfaced for the user (not yet decided):** WAR definition (bat + position only vs add fielding vs add baserunning); headline metric (P(WAR>0) vs projected WAR/runs above replacement with an interval); v1 scope.

### HITTERS, step 2: 2027 hitter projections BUILT and added to the dashboard (SHIPPED v1) -- `fl_hitter_projection.py`, `fl_hitter_park.py`, `fl_hitter_model_dev.py`; dashboard `hitters_view.py`, `ui_helpers.py`, `app.py`
- **User decisions (2026-10-07):** WAR = bat + position + fielding (no baserunning); headline = projected WAR / runs above replacement with an 80% range, P(WAR>0) secondary; v1 = core first (leaderboard, player
  page, Mississippi targets), other-league hitter bridges / lookup / calculator in a second pass. Dashboard: a Pitchers | Hitters switch at the top (`st.segmented_control`, needs streamlit >= 1.40, requirements bumped);
  each section keeps its own sidebar views. Output: 621 hitters (everyone with 2025 or 2026 PA; 237 have no 2026 PA and are flagged "2025 only").
- **Model (all constants fit leave-one-transition-out on 2022-26, frozen in `output/fl_hitter_model_coefs.json`):** (1) K%, BB%, HR/PA (park-neutral), BABIP, HBP% regressed to the league level with two seasons
  (n0 75 / 148 / 136 / 862 / 153 PA; prior-season weight .75 for K%, .5 for the rest); (2) wOBA = 50/50 blend of the regressed wOBA (n0 444, weight .75) and wOBA rebuilt from the regressed components
  (weights K -.15, BB +.49, HR 1.39, BABIP .38), then + 0.63 x (xwOBA shrunk with n0=200 - blend); (3) everything recentered to the 2026 league level (wOBA .3749). LOTO RMSE of next-year wOBA (n=409): league mean .0467,
  last year raw .0684, regressed .0434; components .0423 vs direct .0427 vs blend .0422. xwOBA add-on: 2025 -> 2026 only, n=111, 10-fold RMSE .0379 -> .0368 (single transition: treat as provisional).
- **Playing time:** PA_next = 158.6 + 0.412 x PA_t if he plays (resid SD 129; prior-prior season adds nothing), retention logit -3.44 + 0.62 ln PA_t (LOTO log-loss .593 vs .658 base rate). Hitters whose last season is 2025:
  PA 166 (median of the 26 of 402 who came back), P(returns) 6.5%, n0 multipliers wOBA x2, BB% x1.7, HR x2, K% x1 (only 19 usable pairs -- very noisy).
- **WAR parts (runs per PA):** batting = (wOBA - league)/1.0175; replacement 0.0362 runs/PA and 11.976 runs per win (2026 values); fielding = own 2026 rate x PA/(PA+200) (2025 -> 2026 corr .74, 10-fold optimum n0 200,
  slope ~1.0, error 6.2 runs/600 PA); position = own rate shrunk (n0 100) toward his primary position's 2026 group rate (1B -8.7, LF/RF -5, C +8.4, SS +6.9, 2B +2.8 runs/600 PA). **The position piece is NOT validated**:
  the 2025 and 2026 pipeline builds differ (2025 `pos_adj_runs` is nearly flat except catchers; `primary_pos` is a static label), so 2025 -> 2026 persistence (.50) is contaminated; 2027 follows the 2026 structure by assumption.
  Baserunning excluded (user), though published WAR includes it for 2026.
- **WAR uncertainty:** Monte Carlo (3,000 draws per hitter): PA ~ N(projected, 129), wOBA error Var = a + b/PA with a = .000266 (SD .016), b = .4545 (fit on 4 held-out transitions: standardised SD 1.00, coverage .50/.80/.91),
  fielding and position error 6.2 / 6.0 runs per 600 PA. WAR is conditional on playing.
- **Whole-chain check, 2025 -> 2026** (123 hitters, PA >= 20 in 2025, >= 100 in 2026; fielding shrink tuned on the same rows, so mildly optimistic): total runs above replacement per 600 PA RMSE 22.6 vs 25.5 for the
  league-average guess (corr .46); batting 22.3 vs 24.1 (corr .39); fielding 6.2 vs 10.1 (corr .73); position 6.0 vs 6.6. Standardised error SD .88, coverage 50/80/90% = .56/.85/.94 (slightly conservative).
  P(WAR>0) given realized PA: mean predicted .80 vs actual .76, Brier .1675 vs .1802 (+7.0% skill), but the 0.5-0.7 bin (n=22) predicted .62 vs actual .46 -- marginal hitters are over-rated. Survivorship: among hitters who reach
  100 PA next year, low-PA_t hitters did BETTER than projected (+.014 to +.018 wOBA), so no downward correction for part-timers was applied (the projection is conditional on playing).
- **Park (TrustmarkPark), same model as the pitcher side** (HR per air ball x 0.446; non-HR air-ball hits -0.0073; 1.4 runs per HR; 1.041 per air-ball hit; air balls = launch angle >= 10 deg): (a) HISTORY is neutralised only for the
  Trustmark share of a hitter's own PA (Mississippi hitters 48%, visitors 3%): Mississippi hitters (PA >= 100, n=27) .353 -> .362 wOBA (+.009 add-back; league .366); (b) the TRANSLATION adds
  -s x [HR/PA x (1-.446) x 1.4 + (air balls/PA - HR/PA) x .0073 x 1.041] x 1.0175 for s = 0.5: roster mean -.0119 wOBA (SD .0045), park fit (his shift minus the average hitter's) SD 1.35 runs per season, range about -6.8 to +2.6.
  Air% shrunk with n0 = 89 batted balls (2025 -> 26 corr .57); HR per air ball persistence corr .72 (n0 23). WAR (MIS) = park-neutral WAR + park fit; WAR is park-adjusted by construction, so the translation moves it only through the fit term.
  **Note vs published WAR:** the pipeline's own hitter park factor for Mississippi is 99.0 (a ~1% effect), versus a measured Trustmark effect of about -.012 wOBA at half exposure; Mississippi hitters' published WAR therefore
  barely credits the park. Other parks get no adjustment (same convention as pitchers).
- **Known gaps / next:** hitters with NO Frontier League history (college / Pioneer / MiLB bridge) are not projected; hitters are keyed by cleaned name (same-name merges not audited); 2022-24 lines have no IBB split (2023-24) and no
  neutralisation; xwOBA and position pieces rest on one transition; 2027 league wOBA level is an open risk (league wOBA moved -.002 / -.013 / +.012 / +.016 across the four transitions).

### HITTERS: is BABIP predictable? Mostly no -- real but small and diluted (read-only study; no model change) -- `fl_hitter_babip_study.py`, `fl_hitter_babip_venue.py`
- **Data:** 87,338 balls in play (HR excluded) by FL hitters, 2025-26 regular season, pitch-level; 94.7% have exit velocity and launch angle. xBABIP = out-of-fold P(hit | EV, LA, spray direction) from a league-wide gradient-boosted model, grouped
  by hitter-season so a hitter's own balls never train the model that scores them.
- **Skill vs luck inside one season (hitters with >= 80 tracked BIP):** league BABIP .327; observed SD across hitters .047 / .044 (2025 / 2026) vs binomial noise SD .0375 / .0372 -> 37% / 28% of the variance is not coin-flip noise.
  But split-half reliability (random halves of each hitter's balls, Spearman-Brown to a full season) is only .33 for BABIP, .36 for xBABIP, .21 for "finishing" (BABIP - xBABIP). Correlation of a hitter's BABIP with his xBABIP the same season: .74.
- **Year to year:** 2025 -> 2026, n=113 (>= 50 tracked BIP both years): BABIP .10, xBABIP .21, finishing -.06 (no repeatable ability to beat the batted-ball quality). xBABIP 2025 -> BABIP 2026 .155 (better than BABIP's own .098). On the
  larger StatsCrew + Trackman history (4 transitions, AB >= 100 both years, n=337): BABIP y/y .19 (by transition .25 / .21 / .27 / .08), implying a STABLE talent SD of about .0175 against an observed SD of .039 -- roughly 20% of the variance.
- **Best predictors of next-year BABIP (RMSE, 2026 target, n=113):** league mean .0431; last year's raw BABIP .0572 (far WORSE than the mean); BABIP shrunk with n0 = 800 balls .0426; xBABIP shrunk n0 = 400 **.0422** (-2%); a fitted BABIP + xBABIP
  blend .0437 (overfits at n=113). Nothing materially beats "league average, nudged".
- **Context is bigger than expected:** the venue explains BABIP the batted ball does not: SD of venue mean residual .018-.020 vs .0075 noise; venue effects are STABLE across years (18 venues, corr .58, true SD about .0146 per ball), from -3.2 pp
  (Slammers Stadium) to +3.5 pp (Campanelli Field). TrustmarkPark is about 0 (-.005 / +.003 by year; Mississippi hitters .323 at home vs .305 on the road, xBABIP .326 vs .313, not significant) -- consistent with the Trustmark effect being HR suppression, not
  fewer non-HR hits (the model's -0.0073 per air ball is about -.004 per ball in play, inside the noise). Because a hitter's balls are spread across venues, removing the venue effect barely moves hitter-level predictability
  (BABIP y/y .098 -> .109; best RMSE .0426 -> .0416). The user's earlier "team defense not predictable" holds at the hitter level, but venue (park + home defense) effects are real.
- **What it does to the shipped hitter model:** BABIP enters only through the component half of the wOBA blend (weight .38, n0 862, i.e. almost the league mean). The SD of its contribution to projected wOBA is about .002 against an SD of .019 for
  projected wOBA (~1% of variance). Swapping in xBABIP would improve BABIP's own RMSE by ~1-2% and wOBA by far less. **Decision: no change** (recorded options: replace the BABIP regression with shrunk xBABIP n0 ~ 400; apply venue BABIP factors to
  neutralise history at non-Trustmark parks; show BABIP luck = BABIP - xBABIP on the hitter page).

### HITTERS: project wOBAcon instead of HR/PA + BABIP? Equal accuracy, not adopted (read-only test) -- `fl_hitter_wobacon_test.py`
- **Idea (user):** wOBA = (wBB*uBB + wHBP*HBP + contact * wOBAcon) / den is an exact identity (verified: max |diff| 0.0000 on all 1,817 hitter-seasons), where contact = AB - SO + SF and wOBAcon = hit weights per ball in play
  (HR included). Project K%, BB%, HBP% and wOBAcon and rebuild wOBA with no fitted component weights.
- **wOBAcon is far more stable than BABIP:** y/y corr (league-relative, 4 transitions) .34 vs BABIP .12, HR/PA .57; n0 = 168 balls in play. Trackman: 2025 xwOBAcon -> 2026 wOBAcon corr .57 vs wOBAcon's own .48; RMSE of next-year wOBAcon
  (n=105): league mean .0596, wOBAcon shrunk .0517, xwOBAcon shrunk (n0 100) **.0475**.
- **But the whole-wOBA result is no better.** Leave-one-transition-out, 379 hitter-transitions, next-year PA >= 100, RMSE of next-year park-neutral wOBA: league mean .0452; direct .0417; shipped components (K, BB, HR, BABIP with fitted weights)
  .0419; **shipped 50/50 blend .0415**; pure identity (each piece regressed separately) **.0432** (worse than the shipped blend by .0018, 90% CI .0009 to .0027); identity averaged into the shipped blend .0420. Why the identity loses: each piece is shrunk
  on its own, so the strikeout penalty (n0 75) is applied nearly in full while the high-strikeout hitters' extra power (wOBAcon, n0 168) is shrunk away -- the pieces are correlated, and the fitted weights handle that (K -.14 vs the identity's
  implied -.37).
- **Hybrid (fitted weights, wOBAcon in place of HR + BABIP):** K, BB, wOBAcon .0422; + HBP .0421; K, BB, HR, wOBAcon .0418; shipped .0419 (all 50/50-with-direct versions .0415-.0418). Weights: K -.12, BB +.43, wOBAcon +.47. No gain.
- **xwOBAcon routed through the contact component** (2025 -> 2026, n=102): regressed wOBAcon -> + g x (shrunk xwOBAcon - regressed) -> hybrid weights: RMSE .0363 (g=0), .0358 (.25), .0354 (.5), .0352 (.75), .0352 (1.0); the shipped
  whole-xwOBA nudge scores .0352 (gamma .47). Same accuracy through a single, more interpretable path.
- **Decision: no change** (accuracy identical; the pure identity is worse). Recorded alternative if simplicity matters: K%, BB%, wOBAcon (fitted weights .-12/.43/.47) with xwOBAcon pulled in at g about .75 replaces the HR/BABIP components
  and the separate xwOBA nudge (3 weights instead of 4 + gamma + a second xwOBA shrink constant). Also worth showing projected wOBAcon / xwOBAcon on the hitter page.

### HITTERS: PullAir%, OppoGB%, Barrel%, Max EV, EV90 as projection inputs -- reliable and descriptive, but no added predictive value (read-only test) -- `fl_hitter_contact_vars_test.py`
- **Definitions (per hitter-season, all tracked batted balls in play incl. HR; EV > 120 mph dropped as tracking error):** PullAir% = share of batted balls with launch angle >= 10 hit more than 15 degrees to the pull side; OppoGB% = share with angle < 10
  hit more than 15 degrees to the opposite field; Barrel% = Trackman barrel flag (EV >= 95.5 at a matching angle); MaxEV; EV90 (90th percentile EV); AvgEV for reference. Spray sign checked on home runs (RHH -12.6 deg, LHH +15.4 deg => pull = beyond -15 R / +15 L).
- **Reliability:** split-half full-season (Spearman-Brown) / y/y 2025 -> 2026 (n=114, >= 50 balls both years): EV90 .92 / .87; AvgEV .84 / .77; Barrel% .80 / .78; PullAir% .76 / .76; OppoGB% .72 / .76; MaxEV .79 / **.49** (a single hardest ball is outlier-driven
  and does not repeat). All are far more stable than BABIP (.33 / .10) and wOBA.
- **What they describe (same season, 404 hitter-seasons, park-neutral outcomes):** correlation with HR/PA / ISO / wOBAcon: Barrel% .83 / .83 / .67; EV90 .70 / .72 / .61; AvgEV .66 / .69 / .58; MaxEV .47 / .48 / .39; PullAir% .43 / .42 / .14 (BABIP -.25);
  OppoGB% -.45 / -.45 / -.19. Barrel% and EV90 are close to direct measures of power; PullAir% matters for HR but hurts BABIP; OppoGB% is mostly a (negative) power marker.
- **Next-year prediction beyond the shipped model (2025 -> 2026, n=97, 10-fold x 20 add-on, variables shrunk by their own reliability):** next-year wOBA RMSE .0337 base (shipped blend + xwOBA nudge): + PullAir% +.0006, OppoGB% +.0002, Barrel% +.0006, MaxEV +.0001, EV90 +.0005, all five +.0005
  (none help; the xwOBA nudge already carries contact quality). Next-year wOBAcon (base .0501): Barrel% -.0011, MaxEV -.0018, EV90 -.0010, PullAir% +.0006. Next-year HR/PA (base .0118): Barrel% -.0008, EV90 -.0007, MaxEV -.0002, PullAir% +.0001.
- **Reverse check (2026 -> 2025):** wOBA changes of +.0001 to +.0007 (no help); HR/PA +.0000 to +.0001 and wOBAcon -.0001 to +.0007 -- the forward gains for HR/PA and wOBAcon do NOT replicate (forward re-run with simple bases: HR/PA Barrel% -.0004, EV90 -.0004; wOBAcon MaxEV -.0013).
- **Decision: none added to the model.** They are good descriptors for the hitter page (Barrel%, EV90, Pull-air%, Hard-hit% already shown; Max EV and OppoGB% are not). Revisit when a second Trackman transition (2026 -> 2027) exists.

### EVALUATION STANDARD (user, 2026-10-07): weight every target by its reliability AND its variance at that sample size -- applied to all hitter tests -- `fl_eval_weights.py`, `fl_hitter_weight_audit.py`
- **Standard:** a target observed over n trials has error variance Var_i = a + b/n_i (a = talent + model-misfit variance that does not shrink with n; b/n = sampling noise). The efficient weight is w_i = 1 / (a + b/n_i). This IS the project's
  reliability weight n/(n + n0_eval) (same shape, n0_eval = b/a, as fitted for K%/BB%/RA9 earlier) and the inverse target variance at that sample size -- one weight, not two multiplied together (multiplying would count n twice; run as a sensitivity
  below). (a, b) are estimated by Gaussian maximum likelihood from a reference model's residuals, then FROZEN so every candidate model is scored with the same weights (no model can pick its own judge). `n0_eval` is per target and per
  denominator (PA for rates, AB for BABIP, balls in play for wOBAcon / BABIP studies) -- never a shared constant.
- **What was wrong:** the first hitter tests (backtest, wOBA components, BABIP, wOBAcon, contact variables) used a placeholder w = n/(n + 150) for both fitting and scoring, and the regression constant itself was fitted with a self-referential weight.
  Fitted laws: wOBA a .00026, b .455 (n0_eval ~1,750 PA); K% a ~0 (weights proportional to PA); BB% n0_eval 246; HR/PA 475; BABIP 1,628 AB (2022-26) or 468 balls in play (2025-26 Trackman); HBP% 1,330; next-year wOBAcon a .0023, b .079 (n0_eval 34: mostly misfit, weights nearly flat).
- **Re-fit of every hitter constant under the standard (4 transitions):** n0 wOBA 448 -> 440, K% 76 -> 79, BB% 147 -> 147, HR/PA 136 -> 137, BABIP 905 -> 852, HBP% 153 -> 160; component weights K -.15 -> -.16, BB .49 -> .50, HR 1.39 -> 1.38, BABIP .38 -> .43.
  **The 2027 projections barely move:** SD of the change in projected wOBA .00014 (max .0004; hitters' SD .019), K% .0005, BB% .00004, HR/PA .00003, runs above replacement per 600 PA .08 (SD across hitters 12). The product (double-counted) weighting gives
  n0 within 4%-14% of these. Scored errors fall (wOBA regressed .0434 -> .0421, K% .0434 -> .0415) because large samples now count for more; the ordering of models is unchanged under all three weightings (direct .0417, components .0414, 50/50 blend .0412 on the
  same 409 rows; legacy .0427 / .0423 / .0422; product .0403 / .0401 / .0399).
- **Conclusions re-checked under the standard (all unchanged):** BABIP: xBABIP shrunk n0 400 .0409 vs league mean .0419 vs raw last year .0555 (2% gain; venue-adjusted .0406 vs .0416); wOBAcon identity .0424 vs shipped blend .0408 (gain -.00158, 90% CI -.0024 to -.0008),
  hybrids .0410-.0415, xwOBAcon routed through contact .0351 vs shipped nudge .0349; contact variables: no help for next-year wOBA (+.0001 to +.0007), wOBAcon Barrel% -.0012 / MaxEV -.0019, HR/PA Barrel% -.0008 / EV90 -.0007, reverse check flat.
- **One constant changed:** the xwOBA nudge gamma 0.63 -> 0.57 (n0x stays 200; 2025 -> 2026 n=111, 10-fold: RMSE .0369 -> .0360). Dashboard hitter data republished.
- **Still on older weights (flag):** the fielding shrink (n0 200, tuned with PA weights) and the whole-chain WAR backtest (PA weights); their conclusions are not sensitive but they should be re-tuned under the standard when more data arrive.

### HITTERS: five contact variables vs the xwOBA nudge, and by prior sample size (read-only; no change) -- `fl_hitter_xwoba_vs_vars_test.py`
- **Setup:** base = shipped regressed wOBA blend (no Trackman). Add-ons cross-validated (10-fold x 20), standard inverse-variance weights: xwOBA nudge (shipped); PullAir%, OppoGB%, Barrel%, MaxEV, EV90 jointly (each shrunk by its own reliability, with intercept); EV90 + Barrel% only;
  xwOBA + the five; and "x (1 - reliability)" interactions (do skills matter more when the hitter's own wOBA is thin). Sample: prior PA >= 20 and >= 30 tracked balls, next-year PA >= 100. Groups by prior PA: 20-99 / 100-249 / 250+.
- **Forward 2025 -> 2026 (n=103; groups n = 11 / 25 / 67):** RMSE change vs base: xwOBA nudge -.0004 (90% CI -.0016 to +.0006); five variables -.0001 (-.0020 to +.0019); EV90 + Barrel% +.0002; xwOBA + five +.0002; interaction versions +.0002 (xwOBA) and +.0024 (five). Five vs xwOBA nudge:
  +.0004 (CI -.0014 to +.0023) -- not better. By prior PA, five-vs-xwOBA: 20-99 -.0021 (CI -.0070 to +.0024, only 11 hitters), 100-249 +.0017, 250+ +.0003 -- nothing distinguishable.
- **Reverse 2026 -> 2025 (n=112; groups 8 / 28 / 76), simple base:** xwOBA nudge **-.0027 (CI -.0048 to -.0005)** with gamma 1.01; five variables +.0008 (CI -.0006 to +.0021); five vs xwOBA +.0035 (CI +.0013 to +.0056, WORSE); xwOBA + five -.0020 (worse than xwOBA alone).
  By prior PA the xwOBA nudge helps MOST when the prior sample is small (-.0061 / -.0044 / -.0017 for 20-99 / 100-249 / 250+); the five variables do the opposite (+.0038 / -.0000 / +.0008) and their reliability interaction is worse (+.0272 in the thinnest group).
- **Read:** the five variables carry no next-year wOBA information beyond what the regressed outcomes and xwOBA already hold; xwOBA (which combines exit speed, launch angle and spray into a run-value) is the better contact-quality summary, and it is the one that helps thin-history hitters in the reverse direction.
  Caveats: forward and reverse disagree on the size of the xwOBA nudge (gamma .47 vs 1.01; shipped .57), the thin-prior cells hold only 8-11 hitters (hitters who reach 100 PA next year from a thin prior are a survivor sample), and fitting six coefficients on ~100 hitters invites overfit. **No change.**

### HITTERS: subsets of the five contact variables (user: "some are redundant, use a subset") -- no subset helps (read-only) -- `fl_hitter_subset_test.py`
- **Redundancy (404 hitter-seasons, >= 80 tracked balls):** Barrel% / EV90 .82, MaxEV / EV90 .68, Barrel% / MaxEV .53 (the "power" cluster); PullAir% / OppoGB% -.58 (the "spray" pair); across clusters |r| <= .41. So effectively two ideas: power (EV90 or Barrel%) and spray (PullAir% or OppoGB%).
- **All 31 non-empty subsets, cross-validated in both directions** (2025 -> 2026 n=103; 2026 -> 2025 n=112; standard inverse-variance weights), change in next-year wOBA RMSE vs the regressed blend. Reference: the shipped xwOBA nudge alone -.0006 forward, -.0024 reverse.
  As a REPLACEMENT for the nudge: best average -.0002 (OppoGB + MaxEV: -.0006 fwd / +.0001 rev; PullAir + MaxEV -.0002 / -.0002); EV90 alone -.0000 / +.0003; Barrel% alone +.0001 / -.0000; **0 of 31 beat the nudge in both directions.**
  As an ADD-ON to the nudge: best average +.0001 (OppoGB + MaxEV -.0000 / +.0003); EV90 +.0006 / +.0004; Barrel% +.0007 / +.0003; **0 of 31 improve it in both directions.** (With 31 looks the best rows are optimistic, and they still do not clear the nudge.)
- **Honest selection (nested CV, subset chosen inside each training fold):** 2025 -> 2026: base .0352, nudge .0346, chosen subset replacing the nudge .0361, chosen on top of it .0355; 2026 -> 2025: base .0408, nudge .0384, replacing .0411, on top .0387. The chosen subset changes from fold to
  fold and direction to direction (forward: OppoGB + Barrel + MaxEV / OppoGB + MaxEV; reverse: PullAir + MaxEV / Barrel) -- another sign there is no stable subset to find.
- **HR/PA (drives the Mississippi park fit), singles and pairs, both directions:** best EV90 -.0004 fwd / .0000 rev (base .0120 / .0111), Barrel% -.0004 / +.0001; changes of 1-3% at most and not replicated in reverse. No subset adopted.
- **Decision: no change; xwOBA remains the single Trackman input.** Recommended descriptive set for the hitter page: EV90 (power, most reliable at .92) and one spray stat.

### HITTERS: do the contact variables beat xwOBA for LOW-HISTORY hitters? Year-to-year cannot say; a within-season PA split says EV90 does (OPEN -- not shipped) -- `fl_hitter_lowhistory_test.py`, `fl_hitter_pa_split_test.py`
- **Year-to-year, thin prior history (history = total FL PA in all seasons through the input year; target needs only 30 PA next year; standard weights):** only 16 hitters (2025 -> 2026) and 13 (2026 -> 2025) fall under 100 PA of history. Change in wOBA RMSE vs the regressed blend:
  xwOBA nudge +.0008 forward / -.0046 reverse (pooled -.0014, 90% interval -.0039 to +.0009); EV90 -.0007 / +.0009; OppoGB + MaxEV -.0007 / +.0023; EV90 + PullAir -.0007 / +.0022; all five +.0006 / +.0023. Pooled, subsets vs the nudge: +.0019 to +.0027 (intervals include 0). 100-299 PA group (n=35 / 43):
  nudge -.0010 / -.0040, subsets about 0 or worse. **Underpowered and inconclusive for the thinnest group.**
- **Within-season PA split (the thin-sample question with enough hitters):** 369 hitter-seasons with >= 150 PA (2025-26); known sample = m random PA (m = 30, 60, 120, 200), target = his wOBA over the remaining PA (avg 271 / 241 / 181 / 137); base = lg + m(wOBA_known - lg)/(m + 440);
  add-ons fit by 10-fold CV grouped by hitter; weights = inverse-variance law fitted to the base's residuals at each m; intervals = bootstrap over hitters (400). Change in RMSE vs the regressed wOBA (90% interval) at m = 30 / 60 / 120 / 200:
  xwOBA nudge -.0000 / -.0004 / -.0006 / -.0003 (mostly includes 0); **EV90 -.0013 / -.0014 / -.0014 / -.0019 (all intervals exclude 0)**; EV90 + PullAir -.0014 / -.0015 / -.0015 / -.0019; all five -.0014 / -.0015 / -.0015 / -.0019; Barrel + PullAir -.0009 to -.0012; OppoGB + MaxEV -.0011;
  nudge + EV90 (prior-weighted) -.0013 / -.0017 / -.0018 / -.0022. EV90 vs the nudge: -.0013 / -.0010 / -.0009 / -.0015 (intervals exclude 0 at all four sizes). Base RMSE .044-.051, so the gain is about 3% on the target.
  The EV90 term is about +.009 to +.012 wOBA per +1 SD of shrunk EV90 (.0116 / .0106 / .0094 / .0085). The prior-weighted form (term scaled by 440/(m + 440)) scores the same as the plain additive form.
- **Read:** in a thin same-season sample the power measure EV90 carries real information about the hitter's remaining PA that the regression to the league mean and the xwOBA nudge do not; the gain is NOT larger for thinner samples (flat from 30 to 200 PA), so it behaves like a hitter-type prior rather than a small-sample patch.
  Caveats: same-season split (no aging, role, park or health changes between input and target), 369 independent hitter-seasons, year-to-year tests (n about 100, coefficient +.003 to +.006 per SD, no RMSE gain) neither confirm nor contradict it. **Not shipped.** Proposed rule: add a prior-weighted EV90 term (about +.008 wOBA per SD, shrunk) if the 2026 -> 2027
  transition shows a positive gain on the thin-history hitters; the xwOBA nudge stays in the meantime.

### HITTERS: is the 90th percentile the right exit-velocity percentile for the "power prior"? A flat plateau from about the 80th to the 95th -- keep EV90 (read-only) -- `fl_hitter_evpct_test.py`
- **Design:** the within-season PA-split test (369 hitter-seasons, known sample 30 / 60 / 120 / 200 PA, hitter-grouped CV, standard inverse-variance weights, hitter-cluster bootstrap), one exit-velocity summary at a time, each shrunk by its own reliability (n0 balls from random-half splits of full seasons).
- **Reliability (split-half r) / 2025 -> 26 correlation / n0 balls:** EV50 .64 / .73 / 49; EV60 .70 / .80 / 37; EV70 .75 / .85 / 29; EV80 .81 / .86 / 20; EV90 .86 / .87 / 14; EV95 .87 / .87 / 13; EV98 .87 / .90 / 14; MaxEV .65 / .49 / 46; hard-hit share >= 95 mph .77 / .81 / 26; >= 100 mph .83 / .86 / 18.
  Same-season correlation with HR/PA peaks at EV75-EV80 (.72) and with wOBAcon at EV60 (.65); EV90 .71 / .61.
- **Change in wOBA RMSE vs the regressed wOBA (negative = better), average of the four sample sizes:** EV50 -.0014; EV60 -.0014; EV70 -.0015; EV75 -.0015; EV80 -.0016; EV85 -.0016; **EV90 -.0016**; EV95 -.0016; EV98 -.0015; MaxEV -.0013; hard-hit >= 95 -.0015; **hard-hit >= 100 -.0017**.
  Versus EV90: the 80th-95th percentiles are indistinguishable (|difference| <= .0001, no interval excludes 0 except EV95 at 200 PA at -.0001); EV50-EV70 are slightly worse in the thinnest sample (+.0002 to +.0004 at 30 PA, interval excludes 0); EV98 +.0001; MaxEV worse at every size (+.0002 to +.0004,
  intervals exclude 0); hard-hit >= 100 mph is a hair better (-.0001 on average, -.0002 at 120 and 200 PA, intervals exclude 0) -- practically a tie.
- **Read:** the information is "how hard does he hit the ball at the upper end", and any summary from about the 80th to the 95th percentile (or share of balls over 100 mph) captures it; the very top (single hardest ball) is too noisy, the middle (50th-60th) carries less of the power signal. **Keep EV90** (best reliability/stability balance, standard
  definition); hard-hit >= 100 mph is an equally good alternative. Selection among near-ties on 369 hitter-seasons is not worth a method change.

### HITTERS: EV90 "power prior" term ADOPTED in the projection (user decision, provisional) -- `fl_hitter_ev.py`, `fl_hitter_ev90_fit.py`, `fl_hitter_projection.py`
- **User decision (2026-10-07): "Use the EV90 term."** Form: informed prior, proj_wOBA = blend + gamma x (shrunk xwOBA - blend) + B x (shrunk EV90 - league EV90, mph) x n0w / (PA_eff + n0w); EV90 = 90th percentile of exit velocity over his tracked batted balls (HR included, > 120 mph dropped),
  pooled 2025-26, shrunk by n0 = 14.26 balls toward the league level (100.1 mph; hitter-seasons with >= 50 tracked balls; SD across hitters 3.2 mph); PA_eff = PA_t + 0.75 PA_{t-1}; n0w = 440 (880 for hitters with no 2026 PA); no EV data (17 hitters) -> term 0.
- **Coefficients (within-season PA-split, 34,650 draws, m = 30 / 60 / 120 / 200, joint fit with the xwOBA nudge, standard weights):** B = **.00377 wOBA per mph** (SE .00080; by m .0041 / .0037 / .0035 / .0036), nudge gamma = **.485** (SE .12; was .57 from the year-to-year fit alone).
  One SD of hitter EV90 (3.2 mph) = +.012 wOBA at zero sample, fading by n0w/(PA_eff + n0w) (x0.37 for a 450 PA hitter with a prior year, x0.8 for a 100 PA hitter).
- **Effect on the 2027 projections:** the term has mean -.0018, SD .0072 (PA >= 150: SD .0065), range -.026 to +.024; the spread of projected wOBA among hitters with 150+ PA rises from .0186 to .0223. Largest lifts: Kotowski (Windy City, EV90 109.0, +.024), Harrison-Dudley (+.020, only 22 balls), Contreras (+.019);
  largest cuts: Roberts (Windy City, EV90 86.5 on 16 balls, -.026), Brady (-.026), Wargo (-.023). Dashboard data republished; the "How these are computed" text mentions the power prior.
- **Provisional:** the year-to-year tests could not confirm the gain (n about 100, coefficient +.003 to +.006 per SD, no RMSE change); revisit with the 2026 -> 2027 transition (does the term improve next-year error for thin-history hitters?). Nothing else changed.

### CORRECTION (2026-10-07): the within-season PA-split gains for EV90 / the five variables were overstated -- unfair reference (intercept) -- `fl_hitter_pa_split_fair.py`
- **Flaw:** in `fl_hitter_pa_split_test.py`, `fl_hitter_lowhistory_test.py` and `fl_hitter_evpct_test.py` the add-on models were cross-validated WITH a fitted intercept (+.002 to +.009 wOBA) while the reference (regressed wOBA) and the one-parameter xwOBA nudge had none. The intercept is not
  information about the hitter: the targets are hitters who reached 150 PA, who are better than the league average the base regresses to, so a constant shift lowers error for any model allowed one. Found while checking why an infield-hit coefficient of +.002 wOBA per SD
  appeared to cut RMSE by .0012.
- **Fair comparison (every model, including the reference, gets the same CV intercept; 369 hitter-seasons; hitter-cluster bootstrap; standard weights):** the intercept alone is worth .0004 (30 PA) to .0010 (200 PA) of RMSE (raw base vs intercept-only +.0004 / +.0003 / +.0005 / +.0010).
  Change vs the intercept-only reference at m = 30 / 60 / 120 / 200 PA: xwOBA nudge -.0002 / -.0007 / -.0008 / -.0007 (intervals exclude 0 from 60 PA); **EV90 (prior-weighted) -.0010 / -.0010 / -.0010 / -.0008 (all exclude 0)**; nudge + EV90 (production form) -.0010 / -.0013 / -.0013 / -.0011;
  nudge + EV90 + PullAir -.0011 / -.0014 / -.0014 / -.0012. **EV90 minus nudge: -.0008 / -.0004 / -.0002 / -.0001** -- EV90 is better than the nudge mainly in the thinnest samples and converges with it as the sample grows (earlier claim of "-.0009 to -.0015 vs the nudge" was an artifact).
- **What stands:** EV90 still adds real, significant information (about .001 wOBA RMSE, ~2% of error, additive to the nudge) in every sample size, and the production coefficients (B = .00377 per mph, nudge gamma .485) were fit WITH an intercept, so they are not biased by the flaw. The EV90 term stays; the size of its benefit is about
  .001, not .0014-.0019, and its edge over xwOBA is small and confined to thin samples. The year-to-year subset results (all within noise) are unaffected in conclusion (there the nudge was the only intercept-free model, which if anything favoured the variable subsets and they still did not win).
- **Related open issue (survivor bias of the regression target):** among next-year >= 100 PA hitters the regressed projection under-predicts by +.004 on average and by +.016 (se .0065, n=34) / +.009 (n=36) / +.004 / +.003 / -.0003 for prior-season PA < 50 / 50-100 / 100-200 / 200-400 / 400+; league wOBA of hitters with >= 100 PA is .004-.007 above the all-PA league level. A constant shift does not improve LOTO RMSE
  (.0421 -> .0423) so nothing was changed, but the projections are "conditional on playing" and thin-history hitters are probably projected about .01 too low on that basis (about 0.5 WAR per 600 PA). Candidate fix: regress thin hitters toward the survivor mean; needs more thin-history data to fit.
- **Lesson / rule:** any model comparison gives every candidate AND the reference the same free parameters (intercept included).

### HITTERS: infield-hit rate as a sprint-speed signal -- a real speed signal, but no value for projecting wOBA (read-only) -- `fl_hitter_ifh_test.py` (superseded), `fl_hitter_ifh_test2.py`
- **Data limit:** fielder identity comes from the iScore play-by-play, which exists for 2026 only (0% of 2025 balls have a fielder). The first version (`fl_hitter_ifh_test.py`) mixed a fielder-based definition (2026) with a distance fallback (2025) and gave 21% vs 9% infield-hit rates; its year-to-year result is invalid.
  Distance alone is a poor proxy (only 44% of singles under 60 ft were fielded by an infielder). So no 2025 -> 2026 test is possible; the earliest is 2026 -> 2027.
- **Definition (2026):** infield hit = single fielded by 1B/2B/3B/SS/P/C; population = tracked ground balls (angle < 10 or a bunt) with a recorded fielder (18,467; 1,590 infield hits = 8.6%; LHH 7.8%, RHH 9.3%; SS 35% / 3B 23% / 2B 22% / P 11% / 1B 9% of infield hits). Speed signal = actual minus expected
  infield hits given the ball (EV, angle, spray, side; league GBM, out-of-fold by hitter).
- **Reliability:** raw IFH per ground ball: random-half r .36 (full-season .53), first half vs second half .36; residual: .30 (.46), .28. n0 for the residual about 65 ground balls. Hitter-level SD of the residual .103 per ground ball. **The venue matters:** true venue SD .021 per ground ball (odd/even game split corr .71, 18 venues) -- turf and infield surface.
- **Validity (137 hitters, >= 60 ground balls, >= 120 PA):** IFH residual vs stolen bases per PA +.44 (raw rate +.46); published baserunning runs per PA +.30; GIDP over expected -.20 (fast hitters ground into fewer); triples per PA +.20. It is a genuine, moderate speed signal.
- **Value for projecting wOBA (within-season PA split, 191 hitter-seasons with >= 150 PA, on top of the current model, fair intercept-only reference):** change in RMSE -.00002 / -.00003 / +.00005 / +.00007 at 30 / 60 / 120 / 200 PA (intervals include 0 or are +.00001 to +.00012 -- i.e. nothing); coefficient +.0019 / +.0021 / +.0008 / +.0006 wOBA per SD.
  Speed's wOBA effect through infield hits is tiny (about 1-2 hits per 100 ground balls = +.001 wOBA per SD). **Not added.** It would be the natural input for a baserunning / speed-prior if baserunning is ever added to WAR (user excluded baserunning); revisit with 2027 data for year-to-year stability.

### HITTERS: model ladder -- prior year at face value vs naive shrinkage vs the current model (read-only) -- `fl_hitter_model_ladder.py`
- **Rungs (park-neutral wOBA, next-year PA >= 100, league level of the target year treated as known for all, standard inverse-variance weights frozen from the current model, leave-one-transition-out; same rows for every rung):** 0 league average; 1a prior year at face value; 1b the same shifted for the
  league-level change; 2 naive shrinkage (one season, one n0 = 434 PA fit on the other transitions: a 300 PA hitter keeps 41% of his deviation, a 100 PA hitter 19%); 3 current model without Trackman (two seasons, 50/50 blend of regressed wOBA and wOBA rebuilt from regressed components);
  4 + xwOBA nudge (.485) + EV90 term (.00377/mph) -- Trackman exists only for 2025 -> 2026.
- **LEVEL 1, four transitions, n = 379:** RMSE / R2 vs league average: league average .0448 / 0; **face value .0672 / -1.25** (shifted for league level .0668 / -1.22); **naive shrinkage .0412 / +.152**; current model without Trackman **.0408 / +.169**.
  Paired bootstrap (90%): shrinkage vs face value -.0257 [-.0327, -.0193]; **current model vs naive shrinkage -.00043 [-.00096, +.00008] (includes 0)**. An out-of-sample constant shift changes nothing in the ordering (.0437 / .0667 / .0671 / .0413 / .0407).
- **By prior-season PA (RMSE; mean error):** < 100 PA (n=61): league average .0453 (+.013), face value .128, naive .0465 (+.015), current .0472 (+.015) -- nothing beats the league average for thin priors (survivor bias, see the correction entry); 100-249 (n=97): .0447 / .0572 / .0432 / .0421; 250+ (n=221): .0447 / .0444 / .0389 / .0384.
  Face value is only competitive with the league average at 250+ PA (.0444 vs .0447) and is 2-3x worse below 100.
- **LEVEL 2, 2025 -> 2026, n = 110, same rows:** league average .0392; face value .0639 (R2 -1.66); naive .0363 (+.139); current no Trackman **.0346 (+.218)**, vs naive -.00168 [-.00263, -.00070]; + xwOBA nudge .0340 (+.247), vs no-Trackman -.00067 [-.00172, +.00035]; + EV90 term .0341 (+.241),
  vs nudge **+.00014 [-.00092, +.00121]**; full model vs naive -.00223 [-.00380, -.00063]. The EV90 term (SD .0069) and the nudge (SD .0063) are small next to the blend (SD .0178) and the actual SD of next-year wOBA (.0395). (Mild overlap: the nudge/EV90 constants were fit on within-season splits of 2025-26 PA.)
- **Read:** (1) last year's wOBA at face value is a bad forecast -- worse than the league average in every sample-size bucket below 250 PA; (2) one-line naive shrinkage captures most of the available gain (R2 .15 of the model's .17 over four transitions); (3) the extra structure (second season, components) is worth about .0004 over four transitions (not distinguishable from 0)
  but .0017 on the most recent transition; (4) Trackman terms are not distinguishable from zero in year-to-year data (their support is the within-season splits); (5) for thin-prior hitters no model beats the league average -- the survivor-prior problem.

### HITTERS: does more than one prior season help? Yes for K% (clearly), a little for BB%, not detectably for wOBA or HR/PA; nothing beyond two seasons (read-only) -- `fl_hitter_history_depth_test.py`
- **Setup:** lg_next + sum_j d^j n_j (x_j - lg_j) / (sum_j d^j n_j + n0) using the last K = 1, 2, 3, 4 seasons (d in {.25, .5, .75, 1} and n0 fit on the other three transitions; standard weights frozen; park-neutral wOBA; next-year PA >= 100; n = 423 hitter-transitions). K = 1 is naive shrinkage.
  Hitters with fewer than K seasons use what exists. **Most hitters have little history:** seasons available 1 / 2 / 3 / 4 = 272 / 96 / 39 / 16, so the all-rows comparison dilutes any effect (the production 2027 run has 2026 + 2025 for most and older seasons for few).
- **All hitters, wOBA RMSE:** league average .0458; K=1 .0423; K=2 .0422; K=3 .0423; K=4 .0423. 2 vs 1 -.00009 [-.00038, +.00016]; 4 vs 1 -.00000. By last-season PA (< 100 / 100-249 / 250+): K=1 .0470 / .0448 / .0399 vs K=2 .0467 / .0446 / .0399 (league average .0461 / .0457 / .0458).
  Fitted (d, n0) on all transitions: K=1 n0 434; K=2 d .5, n0 427; K=3 d .75, n0 433; K=4 d .75, n0 436.
- **Components, all hitters:** K% .0425 -> .0417 (2 seasons) -> .0416 -> .0416 (2 vs 1 **-.00077 [-.00134, -.00021]**); BB% .0282 -> .0280 -> .0280 -> .0280 (2 vs 1 -.00023 [-.00065, +.00030]); HR/PA .0133 at every K (no change).
- **Only hitters with deeper history (the rows where K can matter):** with 2+ seasons (n=151; mean last-season PA 304, effective PA with older seasons 461): wOBA K=1 .0381 -> K=2 .0378 (-.00025 [-.00106, +.00051]); K% .0381 -> .0358 (**-.00229 [-.00401, -.00061]**), 4 vs 1 -.00265; BB% .0259 -> .0253 (-.00066 [-.00195, +.00085]); HR/PA .0131 -> .0130.
  With 3+ seasons (n=55): wOBA .0406 -> .0395 (-.00112 [-.00259, +.00031]); K% .0341 -> .0294 (**-.00466 [-.00840, -.00143]**); BB% .0246 -> .0237 (-.00095 [-.00204, +.00010]); HR/PA .0145 -> .0144.
- **Read:** a second season measurably improves K% (a stable skill that keeps adding information), probably helps BB%, and does not detectably help wOBA or HR/PA (outcome-noise dominated: a 304 PA season already carries ~41% weight; a second season lifts it to ~52%, a small change to the prediction). A third or fourth season adds nothing
  measurable for any stat. The production setup (last two seasons, d = .75 wOBA/K%, .5 others) is about right; no change. Power is limited: only 55 hitters have 3+ seasons.

### HITTERS: the K% model reviewed and UPDATED with a Whiff% contact-skill prior (user: "update the K% model"; provisional) -- `fl_hitter_k_study.py`, `fl_hitter_k_study2.py`, `fl_hitter_whiff_fit.py`, `fl_hitter_whiff.py`, `fl_hitter_projection.py`
- **The K% model before the update:** regress K% to the league (2026 level 19.6%) over the last two seasons: lg + (n_t dev_t + .75 n_p dev_p) / (n_t + .75 n_p + 79.3), n = PA; also feeds the wOBA components (weight -.157 per unit) and the Mississippi air-ball calculation. Benchmarks (4 transitions, n=423, next-year PA >= 100, standard
  weights): y/y correlation .66 (hitter SD .0655); RMSE league mean .0608, last year raw .0531, one-season shrinkage .0425, two seasons **.0415**; a 50 PA hitter keeps 39% of his deviation, 100 PA 56%, 200 PA 72%, 400 PA 83%. K% history weight check: d = .25 / .5 / .75 / 1.0 -> RMSE .0417 / .0416 / .0415 / .0416 (.75 kept).
  A second season clearly helps K% (see the history-depth entry); a third adds nothing.
- **Plate-discipline measures from 530k pitches (2025-26; 134,636 PA, 19.1% strikeouts; 375 hitter-seasons with >= 150 PA):** full-season reliability / 2025 -> 26 correlation: SwStr% .91 / .86; Whiff% (whiffs per swing) .90 / .85; Chase% .88 / .85; ZSwing% .90 / .80; ZContact% .86 / .78; OContact% .80 / .78; Swing% .92 / .85; Take-K% .90 / .83;
  FPSwing% .90 / .75; K% itself .83 / .76. Same-season correlation with K%: Whiff% +.90, ZContact% -.86, SwStr% +.79, OContact% -.77; Chase% .08, ZSwing% .13, Swing% .10. The contact measures are more reliable than K% per PA because every PA carries several pitches (Whiff% n0 = 52.5 swings).
- **Within-season PA split (fair intercept in every model):** K% RMSE change vs the intercept-only reference (reference RMSE .0509 / .0456 / .0422 / .0414 at m = 30 / 60 / 120 / 200 PA): Whiff% -.0038 / -.0036 / -.0020 / -.0008; ZContact% -.0034 / -.0036 / -.0020 / -.0009; SwStr% -.0031 / -.0029 / -.0015 / -.0005;
  OContact% -.0020 / -.0020 / -.0013 / -.0005; Chase% 0 (alone), Swing% 0; pair SwStr% + Chase% -.0045 / -.0042 / -.0023 / -.0010; all nine -.0048 / -.0044 / -.0025 / -.0013 -- all intervals exclude 0. Chase% only helps jointly with a contact measure (adjusts for how much he swings). Faded and unfaded forms score the same.
  **Subset chosen: Whiff% alone** (-.00257 average over m vs -.00298 for SwStr% + Chase% and -.00323 for all nine; parsimony, near-ties).
- **Year to year did NOT confirm (n=125 forward / 120 reverse, production two-season base, fair intercept):** 2025 -> 2026: Whiff% faded -.0003 [-.0015, +.0012], unfaded -.0012 [-.0028, +.0003]; 2026 -> 2025: faded +.0008 [+.0000, +.0017] (slightly WORSE), unfaded +.0002. Correlation of last year's measure with next-year K% (PA >= 100, n=105): Whiff% .81 vs K% .76 forward, .71 vs .76 reverse;
  hitters with 30-100 prior PA (n = 15 / 13): Whiff% .55 / .68 vs K% .10 / .62 -- directionally supportive but tiny. The within-season gain is concentrated in thin samples, where the year-to-year test has almost no hitters.
- **Fit and adoption (provisional, same precedent as EV90):** B = **0.605** K% per unit of shrunk Whiff% deviation (SE .046; by m .62 / .63 / .58 / .52); one SD of hitter Whiff% (.069) = +4.2 K% points at zero sample. Form: proj_K = regressed K% + B x (shrunk Whiff% - league Whiff%) x 79.3 / (PA_eff + 79.3), Whiff% pooled over 2025-26 swings (n0 52.5), PA_eff = PA_t + .75 PA_{t-1}, clipped to [.03, .55].
  Effect on 2027: term SD .0070 for hitters with 150+ PA (projected K% SD .0478 -> .0539), SD .021 for hitters with < 60 PA (n=115; range -.031 to +.069, e.g. a 25-swing hitter with a 4% whiff rate drops 3 points, a 22-swing hitter at 64% rises 6). The wOBA components use the updated K% (weight -.157; weights were fit before the term, a small mismatch).
  **Switch-off:** set B_WHIFF = 0 in `fl_hitter_projection.py`. Revisit with 2026 -> 2027: does the term lower next-year K% error for thin-history hitters? Dashboard text updated.

### HITTERS: does Swing% work as an informative prior for BB%? Yes, modestly and more consistently than Whiff% did for K% (read-only; NOT yet implemented) -- `fl_hitter_bb_study.py`
- **Current BB% model:** regress to the league over two seasons (n0 146.6 PA, prior-season weight .5; BB% year-to-year .71, full-season reliability only .70 -- the least reliable of the three rates). Candidates from the same 530k pitches (134,636 PA, 10.7% walks; 375 hitter-seasons with >= 150 PA).
- **Reliability / 2025 -> 26 correlation / same-season correlation with BB% / last year's measure vs NEXT year's BB%:** Swing% .92 / .85 / -.79 / **-.75**; Chase% .89 / .85 / -.79 / **-.76**; ZSwing% .90 / .80 / -.57 / -.60; FPSwing% .91 / .75 / -.51 / -.59; Take-K% (called strikes per pitch) .90 / .83 / +.49 / +.58; BB% itself .70 / .71 / 1 / **+.71**.
  A hitter's Swing% last year predicts his BB% this year (|r| .75) better than his BB% does (.71), and Swing% is 6x less noisy per PA (n0 92 pitches vs 147 PA). Swing% is uncorrelated with K% (+.10).
- **Within-season PA split (fair CV intercept in every model; reference RMSE .0330 / .0317 / .0315 / .0324 at m = 30 / 60 / 120 / 200 PA):** change in BB% RMSE: Swing% -.0027 / -.0025 / -.0016 / -.0011; Chase% -.0026 / -.0026 / -.0017 / -.0012; ZSwing% -.0019 / -.0019 / -.0013 / -.0009;
  Swing% + Chase% -.0029 / -.0028 / -.0018 / -.0013; ZSwing% + Chase% -.0034 / -.0031 / -.0019 / -.0014; all five -.0039 / -.0036 / -.0022 / -.0015 -- every interval excludes 0 (8% of error at 30 PA for Swing% alone).
- **Year to year (production two-season base, fair intercept):** 2025 -> 2026 (n=125, base RMSE .0254): Swing% faded -.0009 [-.0018, +.0000]; Chase% -.0009 [-.0018, -.0000]; ZSwing% -.0010; Swing% + Chase% -.0008; all five **-.0018 [-.0033, -.0003]**. Reverse 2026 -> 2025 (n=120, base .0292): Swing% +.00005 [-.0003, +.0004], Chase% -.0001, ZSwing% +.0002 (slightly worse),
  all five +.0010 [-.0001, +.0022]. So: forward gains that match the size expected from the within-season test, reverse neutral (not harmful for Swing% or Chase%, unlike Whiff% for K%).
- **Size (within-season fit, faded form, raw units):** Swing% B = **-0.387** BB% per unit of shrunk Swing% deviation (SE .030): one SD of hitter Swing% (.058) = **-2.25 BB points** at zero sample (x0.33 at 300 PA of evidence); Chase% B = -0.407 (one SD .062 = -2.51 points). Intercepts +.0009 / +.0007.
- **Read / recommendation:** Swing% (or Chase%, near-identical) works as an informative prior for BB%, with the same caveat as the K% term (strongest evidence within-season, year-to-year gains at about the expected size forward and none in reverse). Swing% alone is the parsimonious choice; ZSwing% is the weakest (and worse in reverse),
  the five-variable version is best in sample but the reverse check turns slightly negative. Not implemented pending the user's go-ahead.

### Clean forward validation of the pitch-level terms: coefficients fit on 2025 ONLY, two-stage bootstrap (user: use 2025 -> 2026 rather than the possibly leaky reverse) -- `fl_hitter_forward_validation.py`
- **Why:** the reverse direction predicts the past from the future, and the earlier forward tests used coefficients fit on within-season draws of both years (including 2026 outcomes). **Design:** Stage 1 fits each coefficient on 2025 within-season PA splits only (m = 30/60/120/200, pooled, fair intercept, production fade forms; variable league levels from 2025); Stage 2 applies the frozen
  coefficients to 2025 -> 2026 on the production bases (K%: n0 79.3, d .75; BB%: n0 146.6, d .5; wOBA: shipped blend) and scores against 2026; 5,000 bootstrap replicates resample BOTH the 2025 hitter-seasons used to fit the coefficient (300 refits) and the hitters being forecast; the model and the reference each get a constant fit on the same resample.
- **Stage-1 coefficients (2025 only, 182 hitter-seasons) vs the 2025-26 production fits:** Whiff% -> K% B = .596 (SE .085) vs .605; Swing% -> BB% -.365 (SE .043) vs -.387; Chase% -.376 (.042) vs -.407; nudge gamma .620 (.207) vs .485; EV90 .00344/mph (.00114) vs .00377. Stable.
- **Forward result (change in RMSE vs the production base; negative = helps; 90% interval; share of replicates in which the term helps):**
  BB% Swing% (n=123) **-.00108 [-.00201, -.00008], P(helps) .96** (95% interval [-.0022, +.0001]); BB% Chase% -.00098 [-.00195, +.00006] P .94; BB% Swing% + Chase% -.00109 [-.00204, -.00002] P .95.
  K% Whiff% (n=123) -.00078 [-.00215, +.00073] P .82. wOBA EV90 + nudge (n=110) -.00040 [-.00195, +.00177] P .62; EV90 alone -.00045 [-.00134, +.00061] P .77; nudge alone -.00058 [-.00174, +.00082] P .77.
  Thin history (prior PA 10-149, next PA >= 50; n=29): K% Whiff% +.00079 [-.00391, +.00577] P .43; BB% Swing% -.00116 [-.00386, +.00185] P .76; Chase% -.00057 P .63; both -.00091 P .69 -- too few hitters to say anything.
- **Read:** the BB% Swing% (or Chase%) term has the strongest clean forward support (about 4% of the base BB% error; the forward gain matches what the within-season fit implies at 200-300 PA); the K% Whiff% term is leaning positive but not established (P .82); the wOBA terms (EV90, xwOBA nudge) are small and statistically
  indistinguishable from zero in a clean forward test (P .6-.8) -- their support remains the within-season splits. Thin-history cells cannot be evaluated until 2027 data exist.

### HITTERS: rest-of-season (ROS) model, step 1 -- rate update with pitch-level terms, chronological walk-forward backtest (user: "Trackman will be more useful for a ROS model" -- confirmed) -- `fl_hitter_ros.py`, `fl_hitter_ros_backtest.py`, `fl_hitter_ros_seasonal.py`
- **User decisions (2026-10-07):** hitters first; outputs = rates + ROS WAR (playing time and WAR still to build); pipeline + backtest now, dashboard view later. **Design:** at an as-of date the current season is the NEWEST partial season in the production regression (in-season PA weight nu = 1, last season .75 / .5, then decayed; same n0s),
  wOBA = 50/50 of regressed wOBA and wOBA rebuilt from regressed K%/BB%/HR/BABIP, plus the production pitch-level terms computed on this season's pitches pooled with last season's where they exist (xwOBA nudge .485, EV90 .00377/mph, Whiff% .605 -> K%, Swing% -.387 -> BB%), all faded by n0/(PA_eff + n0).
  Inputs: dated PA / pitch tables from Supabase (`output/hitter_pa_ros_2025_2026.parquet`, `hitter_pitches_dated_2025_2026.parquet`); 2025 seasons start 2025-05-08, 2026 2026-05-07 (data to 2026-09-06).
- **Backtest:** 2025 and 2026, cutoffs at 15 / 30 / 45 / 60 / 75% of the season by date; every hitter with >= 30 PA after the cutoff (2,476 hitter-cutoffs, 206-289 per cutoff); preseason history from earlier seasons only (2025: 2022-24 box scores, no Trackman history; 2026: 2025 and earlier); target = ROS wOBA / K% / BB%; standard inverse-variance weights
  (law fit on the ROS-update residuals, frozen: wOBA a .00002 b .437; K% .00047 / .333; BB% .00042 / .149); production constants FROZEN (no refit); hitter-cluster bootstrap (500).
- **wOBA RMSE (all cutoffs; by 15 / 30 / 45 / 60 / 75%):** league average now .0574; preseason only .0556; **YTD face value .0730 (worse than the league average at every cutoff)**; YTD naive shrinkage .0554; **ROS update .0544** (vs preseason -.00124 [-.00177, -.00069]; vs naive -.00110 [-.00169, -.00056]);
  **+ pitch-level terms .0531** (vs ROS update -.00131 [-.00189, -.00071]); total -.0025 vs preseason (4.5%). By the hitter's YTD PA (< 30 / 30-99 / 100+): preseason .0640 / .0512 / .0547 -> ROS update .0638 / .0501 / .0528 -> + terms **.0621 / .0493 / .0513**.
- **K%:** league .0692; preseason .0605; face value .0673; naive .0566; **ROS update .0531** (-.0074 vs preseason [-.0094, -.0054]; -.0035 vs naive); **+ Whiff% .0519** (-.0012 [-.0020, -.0005]); by YTD PA < 30 / 30-99 / 100+: .0719 / .0582 / .0560 (preseason) -> .0699 / .0485 / .0433.
  **BB%:** league .0438; preseason .0404; face value .0525; naive .0405; **ROS update .0389** (-.0015 [-.0023, -.0009]); **+ Swing% .0381** (-.0008 [-.0012, -.0003]).
- **Forward refit check (fit nu and the Trackman scale on 2025 cutoffs, score 2026):** fitted nu 1.0 / 1.5 / 1.0 and scale 1.0 / 0.5 / 0.5 (wOBA / K% / BB%); the refit does not beat the production constants on 2026 (wOBA .0495 = .0495; K% .0500 vs .0493; BB% .0379 vs .0377) -> **no refit; production constants stand.** Reverse (2026 -> 2025) agrees. These are chronological same-season tests, the setting the earlier random-split Trackman evidence only approximated.
- **Seasonal level drift (new finding):** league wOBA peaks in June and declines (ROS league level minus YTD league level: 2025 +.0072 / +.0042 / -.0046 / -.0094 / -.0127, 2026 +.0038 / -.0046 / -.0113 / -.0116 / -.0115 at the five cutoffs); K% is 2.0-2.2 points higher in the first weeks (ROS minus YTD -.022 / -.012 / -.008 / -.006 / +.001 in 2025; -.018 / -.009 / -.006 / 0 / +.002 in 2026);
  BB% runs 0.2-1.2 points lower later in the season. Weighted mean error of the wOBA forecasts is -.0115 (forecasts too high). A seasonal level correction from the OTHER season: wOBA **-.00040 [-.00068, -.00011]** forward (2025 -> 2026) and -.0014 reverse (helps both ways, but leaves most of the .0115 bias, because the size of the decline differs by season);
  K% +.0013 forward (hurts; the K% forecasts are already unbiased because the YTD data and the prior-season history offset), BB% +.0004 forward / -.0010 reverse (inconsistent). **Recommendation: apply the seasonal level correction to wOBA only (average over available seasons), K% and BB% not;** provisional.
- **Next:** ROS playing time (PA per team game x games remaining, availability), ROS WAR (batting + fielding + position + replacement over remaining PA) with an 80% range, a production function `ros_projection(as_of)`, then the dashboard view. Fielding / position runs are season totals only (no by-date split exists), so ROS WAR will carry them as preseason-rate constants.

### PITCHERS: rest-of-season (ROS) model rebuilt on the current framework -- K% / BB% update and pitch-level terms, chronological backtest (user: "update the pitcher ROS model") -- `fl_pitcher_ros.py`, `fl_pitcher_ros_backtest.py`, `fl_pitcher_ros_terms.py`
- **What was replaced:** `fl_ros_early_test.py` / `fl_ros_coef_drift_test.py` / `fl_ros_bridge_prior_test.py` refit the OLD spec (naive + bridge + stuff) in-sample at each cutoff with the old n/(n+n0) weights and a single ~3-week cutoff or four weekly cutoffs. The new pitcher ROS mirrors the hitter ROS: the current season to date is the newest partial season in a Marcel-style regression of K% and BB%
  per batter faced, constants fit only on transitions BEFORE the scored season and frozen, five chronological cutoffs (15 / 30 / 45 / 60 / 75% of the season), >= 30 BF after the cutoff (2,801 pitcher-cutoffs; 1,335 in 2025, 1,466 in 2026), standard inverse-variance weights, pitcher-cluster bootstrap.
  History: 2022-24 StatsCrew box scores (name-keyed) + 2025-26 war_pitchers. **Not carried into this first version:** the other-league bridge, wh_sva_level / spin / extension, command_score and the BB% informative-prior blend of the production model (YTD data supersedes them for pitchers with in-season BF; they still drive the preseason projection).
- **Constants (LOTO, standard weights; prior-season weight d, n0 in BF):** scoring 2026 (transitions through 2024->25): K% d .25, n0 113 (y/y .55; LOTO RMSE .0487 vs league .0583 vs last year .0581); BB% d .25, n0 103 (y/y .52). Scoring 2025 (2 transitions): K% d .5, n0 117; BB% d .25, n0 118. Pitchers carry far less from the past than hitters (hitters: K% d .75, n0 79; BB% d .5, n0 147).
- **Step 1, ROS update (all cutoffs; 15 / 30 / 45 / 60 / 75%):** K% RMSE: league average now .0659; preseason only .0634; YTD face value **.0767 (worse than the league average at every cutoff before 75%)**; YTD naive shrinkage .0606; **ROS update .0594** (vs preseason **-.0040 [-.0051, -.0029]**; vs naive -.0012 [-.0017, -.0007]).
  BB%: .0477 / .0449 / .0579 / .0425 / **.0415** (vs preseason **-.0034 [-.0044, -.0025]**; vs naive -.0010 [-.0013, -.0008]). By BF so far this season (< 30 / 30-99 / 100+): K% preseason .0677 / .0628 / .0594 -> ROS update .0667 / .0578 / .0531; BB% .0518 / .0444 / .0384 -> .0509 / .0393 / .0337.
- **Step 2, pitch-level terms (coefficient fit on the 2025 chronological ROS rows only, frozen, scored on 2026; two-stage bootstrap over pitchers; fair constant; reliability n0 in pitches from random-half splits: SwStr% 289, Whiff% 136 swings, CSW% 399, Chase% 227, Zone% 275, FPS% 263):**
  K% (ROS update RMSE .0584): SwStr% -.00043 [-.0011, +.0002] P(helps) .89; Whiff% -.00045 P .89; CSW% -.00020 P .76; Chase% +.00014 P .30; Zone% +.00004 P .51; SwStr% + Chase% -.00042 P .84 -- leaning positive, **not established**.
  BB% (ROS update .0431): **Zone% -.00096 [-.0016, -.0004] P .996; FPS% -.00081 [-.0014, -.0003] P .99**; Chase% -.00030 P .95 (interval touches 0); CSW% +.00004; SwStr% +.00004; **Zone% + FPS% -.00116 [-.0018, -.0005] P .999; Zone% + Chase% -.00118 [-.0019, -.0006] P 1.00** (about 2.7% of BB% error).
  Reverse check (fit 2026 -> score 2025): nothing distinguishable from zero (BB% Zone% -.00015 P .62, Zone% + FPS% -.00013 P .62; K% terms |change| <= .00018, P .28-.69); the signs agree for the BB% terms and the 2026-fit coefficients are larger (Zone% -1.10 vs -.62).
- **Read / recommendation:** adopt the ROS update for K% and BB% (clear gains over preseason and over naive shrinkage); adopt a BB% Zone% + FPS% term (provisional: strong forward, non-contradicting reverse); the K% whiff terms are NOT adopted (not established; unlike hitters, pitchers' whiff measures are noisier per BF, reliability .54-.66 vs .8-.9).
  Still to build: ROS RA9 (K%/BB% -> RA9 baseline + Mississippi shift), ROS innings / appearances, ROS WAR and P(WAR>0); the within-season league-level drift check for pitchers; the production function and dashboard view.

### PITCHERS: the production skill measures as ROS informative priors -- wh_sva_level -> K%, command_score -> BB% (user suggestion; supersedes the simple whiff / zone terms) -- `fl_pitcher_ros_stuff.py`
- **Measures (rebuilt at every as-of date from last season's pitches + this season's pitches BEFORE the date; the whiff model is refit on that pool, so nothing from the ROS window enters):** wh_sva_level = mean over a pitcher's swings against him of (whiff - expected whiff given the pitch's characteristics), `fp.compute_pooled_wh_sva`; command_score / horiz_var / vert_var = `fp.compute_command_score` (OpenCommand-calibrated
  inferred-miss inches from location variance). **Reliability (random-half r; n0):** wh_sva_level **.95, n0 = 10 swings** (far more reliable than any simple whiff measure: .54-.66); command_score .82 (97 pitches); horiz_var .86 (73); vert_var .89 (57). Each shrunk toward the league by its own n0; form: B x shrunk deviation x n0stat / (PA_eff + n0stat) (n0stat = the stat's ROS regression constant, 113 K% / 103 BB%).
- **Two-stage chronological test (coefficient fit on the 2025 ROS rows only, five cutoffs; scored on 2026; two-stage pitcher bootstrap; fair constant; ROS-update RMSE on the scored season K% .0584, BB% .0431):**
  **BB% command_score: -.00182 [-.00247, -.00113], P(helps) 1.000** (4.2% of error); horiz_var + vert_var -.00167 [-.0024, -.0008] P .999; command_score + both -.00168. K% **wh_sva_level -.00084 [-.00168, +.00011], P .93** (1.4%).
  **Reverse check (fit 2026, score 2025; ROS-update RMSE K% .0605, BB% .0399):** command_score **-.00134 [-.00245, -.00011] P .96**; horiz_var + vert_var -.00113 [-.0022, +.0002] P .91; wh_sva_level -.00090 [-.00197, +.00029] P .90. **Same sign and a nearly identical coefficient in both directions** (K% .946 vs .955; BB% command_score .029 vs .037).
  Pooled 2025-26 fit: K% B = **0.958** per unit of shrunk wh_sva deviation (SE .226; one SD of the shrunk deviation, .019, = +1.8 K points at zero sample before the fade); BB% command_score B = **0.0345** per inch (SE .0039; one SD, .675 in, = +2.3 BB points before the fade); horiz_var .130 / vert_var .120 per unit.
- **The simple measures add nothing once these are in:** K% wh_sva + SwStr% -.00092 (P .90) vs wh_sva alone -.00084; wh_sva + Whiff% -.00091; BB% command_score + Zone% -.00178, + FPS% -.00182, + Zone% + FPS% -.00179 vs command_score alone -.00182 (reverse: all slightly worse, +.0002 to +.0003 vs command_score alone). The earlier Zone% + FPS% result (-.00116) was a weaker stand-in for command.
- **Recommendation:** adopt command_score for BB% (strong, replicated both ways) and wh_sva_level for K% (consistent in both directions, interval touches zero: provisional, same status as the hitter K% / BB% terms). horiz_var + vert_var perform about the same as command_score; command_score is simpler and slightly better. Not yet wired into a production function (pitcher ROS still needs RA9 / innings / WAR).

### ROS (hitters and pitchers): corrections, workload model, production module -- `fl_ros_workload.py`, `fl_ros_production.py`, `fl_ros_build.py`, dashboard `ros_view.py`
- **Correction (constants leak, pitchers):** the pitcher ROS backtests scored a season with constants fit on transitions whose target year included that season (2026 scored with 2025 -> 2026). Fixed: 2025 uses transitions through 2023 -> 24, 2026 through 2024 -> 25. Re-run: K% ROS update .0592 (was .0594; vs preseason -.0041 [-.0052, -.0031]); BB% .0417 (was .0415; -.0034); constants 2026: K% d .5, n0 117; BB% d .25, n0 118; 2025: K% d .25, n0 130; BB% d .25, n0 136.
  Stuff / command priors re-run under the corrected constants: command_score forward **-.00205 [-.0028, -.0013] P 1.00**, reverse -.00155 P .975; wh_sva_level forward -.00072 P .915, reverse -.00101 P .90 (slightly stronger than before; same conclusions).
- **Workload model (PA for hitters, BF for pitchers):** rate per TEAM game since the player first appeared (blended with last season's rate; blend constant k = 5 hitters / 10 pitchers) x team games remaining, then a RECENT-ACTIVITY rule: players who appeared in the last 21 days get c x the rate-based projection, everyone else a small constant
  (fit on the other season; hitters c .75-.79, inactive 5-9 PA; pitchers c .68, inactive 4-7 BF). 2025 -> 2026 / 2026 -> 2025: hitters RMSE 79.9 / 73.9 -> **70.9 / 62.5**, bias -34.5 / -35.9 -> +5.6 / -5.0; pitchers 87.7 / 85.6 -> **67.7 / 69.1**, bias -47.5 / -45.1 -> -0.4 / +0.4 (league-rate RMSE 106.6 / 107.7).
  About 20% of players who had appeared by a cutoff have not appeared in the last 21 days and average ~5 PA / BF afterward (vs 140 / 106 for the rest). The scale is uniform across workload quintiles (pitchers .65-.76, hitters .66-.82): a systematic fall-off in how much players play late in the season, not a starter/reliever artefact; individual misses remain large.
- **Play-by-play runs (2026 only):** `fl_pitcher_runs_dated.py` builds dated outs and runs charged per pitcher from `pbp_events` (runs_charged_to lists one pitcher per run): IP corr .987 (ratio .991), runs corr .948 (ratio 1.019), league RA9 6.36 vs published 6.25. 2025 has no play-by-play, so ROS RA9 / innings can be scored for 2026 only.
- **Production module:** hitters: ROS rates (production regression + pitch-level terms) + wOBA seasonal level correction from the other season, PA workload, WAR = bat + fielding (prior-season rate, n0 200) + position (group rates) + replacement (0.0362/PA, 11.976 runs per win), 80% range from simulation (PA error 70, luck law .000266 + .4545/PA, fielding / position error 6 runs per 600 PA).
  Pitchers: ROS K% / BB% with wh_sva / command priors (2025-only coefficients 0.929 / 0.030 for 2026 snapshots; pooled 0.958 / 0.0345 live), RA9 = frozen park-neutral baseline on the ROS K% / BB% / IP per appearance (this season's approximate IP pooled with last season's), Mississippi version via the Trustmark shift (trait-model Air%, display-only), innings = ROS BF / 4.512,
  WAR = (6.963 - RA9) x IP / 9 / 10.706 with P(WAR>0) from the luck law 185.16 / IP. Snapshots at eight 2026 as-of dates use only pre-date information (constants, workload and stuff coefficients fit on 2025; seasonal correction from 2025); display-only static inputs: position group rates, trait-model Air%.
- **Dashboard:** new "Rest of Season" view in both the Hitters and Pitchers sections (as-of selector, recently-active filter, optional "what actually happened" columns).
- **Pitcher ROS RA9 and innings, scored on the 2026 play-by-play (951 pitcher-cutoffs with >= 10 ROS IP; five cutoffs; inverse-variance weights from the ROS model's residuals, law a ~ 0, b = 161.7):** RMSE of ROS RA9: league RA9 so far 2.378; YTD RA9 at face value **3.937**; YTD RA9 shrunk to the league (40 IP) 2.353; preseason K% / BB% -> the frozen RA9 baseline **2.230**;
  **ROS K% / BB% (with stuff + command priors) -> the baseline 2.139** (vs preseason inputs **-0.091 [-0.129, -0.055]**; vs YTD shrunk **-0.212 [-0.287, -0.137]**); by cutoff (15 / 30 / 45 / 60 / 75%): preseason 1.950 / 2.091 / 2.238 / 2.412 / 2.808 -> ROS 1.905 / 1.998 / 2.126 / 2.303 / 2.693. A pitcher's results-to-date RA9 is a far worse predictor than his K% / BB% run through the baseline.
  Innings (everyone who had appeared by the cutoff; zeros included): mean actual 18.3 IP, RMSE 16.7, MAE 12.0, bias -0.2, corr .68; actual ROS BF per IP 4.541 vs the 4.512 used. Caveat: the RA9 baseline's own 3 coefficients were fit on 2025 -> 2026 transitions (including the scored 2026 season), so the RA9 check is mildly optimistic; 2025 cannot be scored (no play-by-play).

### ROS robustness and training windows -- `fl_ros_robust.py`, `fl_ros_training_windows.py`, `fl_ros_bb_nu.py` (2026-10-07; nothing in production changed)
- **Cutoffs (16 as-of dates, 10-85% of the season; 32 season x cutoff cells; cutoff-block and player bootstrap):** every headline gain replicates. Hitters, ROS update vs preseason: wOBA -.0012 (32/32 cells better), K% -.0075 (32/32), BB% -.0016 (31/32); vs naive YTD shrinkage: wOBA -.0010 (26/32), K% -.0035 (30/32), BB% -.0017 (32/32). Pitch-level terms on top: wOBA -.0015 (31/32), K% -.0012 (31/32), BB% -.0007 (30/32). Pitchers update vs preseason K% -.0042 / BB% -.0035 (32/32 each); stuff/command terms K% -.0009 (16/16), BB% -.0017 (14/16). Both seasons agree in sign everywhere; the pitch-level wOBA term is stronger in 2026 (-.0023) than 2025 (-.0009).
- **Horizon (next 14 / 28 / 56 days vs rest of season):** the update helps at every horizon (hitter K% -7.7% of error at 14 days, -12% at ROS); the wOBA gain is small at short horizons (-1%) and grows with horizon (-2.2%). The wOBA/K% terms are significant at every horizon; hitter BB% term is marginal at 14 days (-.0004 [-.0007, -.0001]).
- **Recency (half-life 90/45/21 d; trailing window 90/60/30 d; select on one season, score on the other):** nothing for wOBA, K%, or pitchers (selected variant = full season to date or change <= .0001, intervals straddle 0). **Hitter BB%: half-life 21 d wins in both directions (-.00047 [-.00082, -.00013] forward, -.00056 reverse).**
- **In-season weight nu (1 = production):** wOBA flat (best 2.0, differences .00004), K% best at 0.75-1.0, pitchers ~0.75-1.0. **Hitter BB% prefers the lowest tried (0.5, 100% of draws).**
- **Follow-up (hitter BB% only; nu grid down to 0.1 x half-life):** nu alone .3 -> -.00049 [-.0008, -.0002]; half-life 21 alone -.00054 [-.00076, -.00032]; both (21 d, .5) -.00065. Forward selection: choose on 2025 -> (hl 21, nu .75) scores 2026 at -.00061 [-.0010, -.0002]; choose on 2026 -> (nu .2, no recency) scores 2025 at -.00017 [-.0006, +.0003] (not distinguishable). Interpretation: in-season BB% is less informative per PA than its preseason n0 implies (nu and recency both shrink the effective in-season n); it is a ~1.5% BB% error gain, replicated forward in one direction and sign-consistent in the other. **Not adopted yet** (needs the same constants-refit treatment as other in-season parameters); candidate: hitter BB% nu = .5, optionally hl 21 d.
- **Training windows, hitter preseason constants (n0; prior-season weights fixed):** all-earlier vs last-two vs last-one vs production (which includes the scored season): wOBA identical (+.00000 to +.00002), BB% production-vs-leak-free +.00002 (no practical leak), K% shorter windows slightly better (last-two -.00008 [-.00014, -.00002], last-one -.00018 [-.00034, -.00002]) -- tiny, and the last-one n0 swings (2025: 67 vs 2026: 120) so not adopted. **The known leak in the hitter ROS backtests (production SPEC includes 2025->26) is quantified: <= .00005 on any stat. Closed.**
- **Pitcher constants windows:** K% all-earlier / last-two / last-one indistinguishable; BB% "all four" (leaky) is -.00023 better [-.00035, -.00013] -> the earlier leak correction removed a genuine ~.0002 of optimism; the numbers reported after the correction stand.
- **Term-coefficient training windows:** hitters, scale on production coefficients: chosen scale 1.0 in 7 of 12 train->test pairs (0.5 or 1.5 otherwise), and wherever it differs from 1.0 it does WORSE than production on the held-out slice (+.00014 to +.00048); terms vs none help in every split (wOBA .0008-.0024, K% .0004-.0015, BB% .0006-.0009). Pitchers: wh_sva_level -> K% coefficient .83-1.08 across splits (pooled .958), P(helps) .86-.98; command_score -> BB% .028-.037 (pooled .0345), P(helps) .98-1.00 in all four splits (by season and early/late). **Coefficients are stable across training windows; no change.**
- **History depth in the update:** hitters 3 earlier seasons vs 0 / 1 / 2: wOBA +.00045 / +.00012 / +.00007 worse, K% +.0030 / +.0009 / +.0002, BB% +.0013 / +.0003 / +.0001 -> keep 3. Pitchers: 1+ earlier seasons suffice (K%: 0 seasons +.0013 worse; 1-4 identical; BB% same); keep 4 (free).
- **Bottom line:** the ROS models are robust to cutoff, horizon, and training window. Only candidate change: hitter BB% in-season weight (nu ~ .5 and/or 21-day half-life), worth ~.0005 RMSE.

### ROS by days into the season (clarification of "training time windows" = as-of DAY, not prior seasons) -- `fl_ros_by_day.py` (2026-10-07; nothing in production changed)
- **Design:** as-of days 7, 14, ..., 105 after the first game (15 dates x 2 seasons); target = rest of season; hitters wOBA/K%/BB% (preseason / update / update + terms), pitchers K%/BB% (update). Then the in-season weight nu, preseason-prior scale n0_scale and (hitters) pitch-term scale are tuned in four day bins (<=28, 35-56, 63-84, >=91) on one season and scored on the other vs the single production setting (player bootstrap on the pooled change).
- **Shape of the gain:** the update's gain over preseason grows with the day, then flattens: hitter K% -.001 (day 7) -> -.012 (day 84-98); wOBA ~0 -> -.003 (2025) / -.0012 (2026); BB% -.0001 -> -.004 / -.002; pitchers K% -.0007 -> -.008, BB% -.001 -> -.004 to -.006 (peak ~day 50 in 2025). Day 7 is not worth much (a week of PA is mostly noise: K% gain only -.001). The pitch-level terms help from day ~14 on and stay roughly constant (hitter K% -.001 to -.002 at every day; wOBA -.0005 to -.003, strongest day 35-70 in 2026; BB% -.0005 to -.001); at day 105 only ~132 hitters in 2025 qualify (>= 30 ROS PA), so late-day numbers are noisier.
- **Do the best parameters change with the day?** Hitters: no. wOBA nu / n0_scale / term-scale selected per bin vs production: +.0000 [-.0001, +.0001], -.0001 [-.0002, +.0000], -.0001 [-.0002, +.0001]; K%: nu +.00015 and term scale +.00030 [+.0001, +.0005] (selecting per bin is WORSE); n0_scale +.0000. The per-bin choices flip between seasons, i.e. no stable day-dependence. Hitter BB% prefers a lower nu in EVERY bin (best .5 in 7 of 8 cells; selected-by-bin -.00040 [-.00055, -.00023]) -- the same constant-across-days effect found in the robustness study, not a day effect.
- **Pitchers (new finding): the update is under-weighted on in-season data.** n0_scale = .5 (half the preseason-prior strength) is best in nearly every bin in both directions: K% -.00036 [-.00069, +.00001] (all 8 cells <= -.00002; larger later in the season: -.0003 early to -.0007 late), BB% -.00052 [-.00079, -.00027] (6 of 8 cells; best-on-train was 1.0-1.5 in the late bins, where the gain is zero or slightly negative). nu = 1.5-2 for K% in late bins (-.0004 to -.0008) says the same thing. Candidate: pitcher n0_scale ~.5-.75 (check that the stuff/command fade `n0/(pa_eff+n0)` should keep the unscaled n0, and refit pitcher constants from in-season targets rather than season-to-season transitions). **Not adopted; needs a leak-free refit.**
- **Interpretation / asymmetry:** hitter BB% wants MORE shrinkage toward the prior in-season (nu .5, n0_scale 2), pitchers want LESS (n0_scale .5). Season-to-season transition constants were fit with full-season samples; applying them to in-season partial samples is an assumption that holds for hitter wOBA and K%, and not for hitter BB% or pitcher K%/BB%.

### In-season strength refit (leak-free, on ROS targets) -- `fl_ros_inseason_refit.py` (2026-10-07)
- **Method:** one free parameter per stat fit on ONE season's as-of days (7 ... 105 by 7), standard inverse-variance weights frozen from the production predictions, scored on the OTHER season (forward 2025 -> 2026; reverse as a check); two-stage bootstrap (stage 1 resamples players in the fitting season and re-picks the parameter; stage 2 resamples players in the scored season). Alternative considered: also free nu / half-life jointly (rejected: one parameter per stat, and nu / n0_scale are nearly collinear in-season).
- **Pitchers: NOT ADOPTED (reverses the by-day suggestion).** Update-only, a weaker prior does help (K% n0 x .5: 2026 .05900 -> .05868; BB% x .65: .04314 -> .04262; reverse similar). But once the wh_sva / command_score terms are in (B refit on the fitting season; fade built from the scaled or the unscaled n0), the gain is gone: K% -.00001 [-.0005, +.0007] forward / -.00011 reverse (P(better) .55-.57); BB% is WORSE: +.00019 forward (P(better) .05), +.00068 reverse (P .00). The lower n0 had been standing in for the stuff / command information (it pulls the update toward the pitcher's own in-season results, which the terms now supply). Keep n0 x 1 and the unscaled-n0 fade. The by-day pitcher result was update-only; a reminder to run in-season parameter checks on the full model, not the base.
- **Hitters, BB%: ADOPTED nu_bbr = 0.4** (`fl_hitter_ros.PROD["nu_bbr"]`; BB% in-season weight 0.4 instead of 1, including in the Swing% term's fade; explicit `nu=` still overrides for all stats). Forward: fit on 2025 picks nu .4 (stage-1 bootstrap spread .1-.8), scored on 2026 **-.00063 [-.00107, -.00022], P(better) 1.00**; reverse (fit on 2026 picks .2) scored on 2025 -.00021 [-.00064, +.00031] P .73 (same sign, not distinguishable from zero). Error curve is flat for nu in .2-.5. Swing% term scale 1.0 is still best at nu .3 and .5 (x0.5 and x1.5 both worse), so no coefficient refit. The BB% deviation also enters the wOBA component blend: effect on wOBA -.00002 (2026 -.00004, 2025 0), K% unchanged. Recency half-life 21 d was equally good alone (-.0005) but is a second mechanism for the same thing; not added.
- **Rebuilt:** `output/ros_hitters_2026.parquet` (+ dashboard copy; pre-change copy kept as `ros_hitters_2026_before_nubbr.parquet`): ROS BB% moved by .004 on average (max .017), ROS wOBA / WAR by rounding-level amounts; pitcher snapshots unchanged.
- **Caveat:** nu .4 was chosen on 2025 and the 2026 snapshots are scored on the 2026 forward test, so the snapshot values are leak-free; live 2027 should refit pooled (2025 + 2026 rows).

### Live (pooled-coefficient) ROS build -- `fl_ros_live.py` (2026-10-07)
- **What:** a year-general entry point for a season that is still being played (2027): workload constants, seasonal wOBA level correction and pitcher regression constants fit on 2025 + 2026 (pitcher n0: K% 113, BB% 103, delta .25 each), stuff / command coefficients pooled (.958 / .0345), hitter BB% nu .4. Writes `output/ros_{hitters,pitchers}_live_{year}.parquet`, does NOT overwrite the dashboard's 2026 files; refuses to run if the dated PA table has no rows for the year (needs the data refresh first).
- **Trade-off:** pooled coefficients use more data (the rule for live use) but cannot be validated out of sample until the season has run; the clean (fit-on-2025) 2026 snapshots remain the backtest-honest versions. Alternative (keep using the 2025-only coefficients live) rejected: it throws away a full season of information.
- **Check (in-sample, --year 2026, 677 hitter / 865 pitcher snapshot rows at two dates):** pooled vs clean snapshot: hitter ROS wOBA mean |diff| .0021 (corr .995), K% / BB% identical, ROS PA mean |diff| 3.4, ROS WAR mean |diff| .009 (corr .999); pitcher K% / BB% mean |diff| .0021, RA9 .035 (corr .998), IP .12, WAR .004. The pooled coefficients move forecasts only slightly, as expected.
- **Still unvalidated (unchanged):** hitter fielding / position rates are last season's (annual data only, no dated split to test against); RA9 baseline coefficients include the 2026 transition; the dashboard ROS view reads the 2026 files only.

### ROS pitchers: published preseason projection as the prior -- `fl_pitcher_pre2026.py`, `fl_ros_published_prior.py`, live path in `fl_ros_live.py` (2026-10-07)
- **Question (user):** should the ROS update start from the published preseason K% / BB% (production projection model: bridges, arsenal / stuff features, 2025-26 frozen coefficients) instead of the ROS model's own Marcel history? Only 2026 can test it.
- **Construction:** 2026-style preseason projection = the frozen Experiment-2 model (fit full-2025 -> actual 1H2026) on full-2025 features (`output/pitcher_pre2026_published.parquet`, 465 pitchers with a 2025 FL record; rookies have none). **Leak:** the coefficients were fit on 2026 targets up to 2026-07-07, so the test uses as-of days 63-105 (>= 2026-07-09: targets are 2H-only, out of sample for the coefficients). That leaves 7 as-of dates x ~240 pitchers, one season, one half-season: thin evidence.
- **Update:** dev0 = pre - BF-weighted mean(pre) (recentred so the CURRENT league level anchors, as the production projection recentres); posterior = lg_now + (n_p dev0 + nu n_ytd (x_ytd - lg_now)) / (n_p + nu n_ytd), n_p = phi (W_hist + n0), i.e. the Marcel update's own history weight, phi = 1 -> zero fitted parameters (phi .5-2 as sensitivity). Alternatives considered: fitting n_p freely (too little clean data), averaging the Marcel and published estimates (extra free weight, no principled value).
- **Stuff / command terms double count** with a published prior that already contains last season's wh_sva_level / command_score: scales 0 / .25 / .5 / 1 compared.
- **Results (weighted RMSE; change vs production Marcel + full terms; pitcher bootstrap):** K% production .06044; published prior phi 1 with terms x.5 .05992, **-.00052 [-.00117, +.00011] P(better) .90**; terms x1 -.00045 (P .79), terms x0 -.00030 (P .78). BB% .04169 -> .04125 at x.5, **-.00049 [-.00112, +.00019] P .88**; x1 -.00042. phi .5 / 1.5 / 2 all within .0001 of phi 1. The gain is concentrated where the published projection exists: K% .05874 -> .05734, BB% .03958 -> .03843 on those pitchers; rich-history pitchers improve most (K% .0527 -> .0504), thin-history ones barely (.0651 -> .0645). Only 41% of the 2026 pitcher-dates have a published projection (no bridge for 2026 rookies in this test).
- **Decision:** adopted in the LIVE path only (`fl_ros_live.py`: published projection as prior, phi 1, stuff / command terms x .5; for 2027 it reads `fl_2027_projections.parquet`). Evidence is suggestive (P ~.9, interval includes 0), consistent in sign across K% and BB% and all phi / term-scale settings, and the principle (don't discard the model's other information) favours it. The 2026 dashboard snapshots are NOT changed: the 2026 published projection saw 1H2026 data through its coefficients, so a 2026 snapshot built on it before July 7 would not be honest.
- **Not done:** hitters (the published hitter projection is hard-wired to 2027 in `fl_hitter_projection.build_projections`; a leak-free 2026 version needs that generalised and the transitions capped at 2024 -> 2025; and the hitter history part of the ROS model is already the same regression, so the expected difference is smaller); bridged rookies / prospects (no clean test in 2026). Re-test pitchers when the 2027 ROS targets exist.

### Position players in the pitcher tables -- `fl_position_player_flag.py` (2026-10-08)
- **Trigger:** Gavin Roy (2B, 113 hitter PA, 9 BF in two mop-up games, only 2 of 18 pitches tracked) had a projected K% of -3.7% in the published 2027 table. 70 of 782 rows in `fl_2027_projections.parquet` had >= 100 hitter PA (2025-26) and <= 60 pitching PA.
- **Rules (either flags):** A = >= 100 hitter PA (name-matched to `fl_hitter_panel`) and <= 60 BF; V (user request: pitch characteristics, outlier lows) = >= 5 tracked fastballs (four-seam / sinker / cutter, effective velocity) and robust z <= -5 vs non-rule-A pitchers (median 88.5 mph, robust SD 2.47 -> cut ~76.2 mph). The velocity distribution has a clean gap: position players 48-71 mph, slowest real pitchers 77-81 mph (Sechrist, Pierson, Ordorica ...).
- **Velocity arbitrates rule A (decision):** with >= 5 tracked fastballs and normal velocity (z > -3), the hitter-record match is NOT trusted and the pitcher is not flagged (24 pitchers at 81-93 mph, e.g. Spillane 92.1, Stevens 93.1, O'Brien 89.1: real two-way players or pitchers sharing a name with a hitter); they are recorded as `possible_two_way` for review. Alternative (rule A alone) rejected: it flagged 25 pitchers who throw 87-93 mph. Rule A stands when velocity is unverifiable (< 5 tracked fastballs; 25 pitchers, incl. Roy) or in the low tail (z <= -3; Alvarez, 79.9 mph).
- **Result:** 51 of 820 pitchers flagged (22 hitter record + low velocity, 3 low velocity only [Harlan 59.3 mph / 61 BF, Blaum 57.5 / 65 BF, Garrison 63.4 / 5 BF: all missed by the BF <= 60 / hitter PA >= 100 rule], 1 below-normal velocity, 25 unverifiable); 50 appear in the 2027 table. Total flagged volume: 647 BF of 133,391 (0.5%).
- **Applied:** dashboard Pitcher views hide flagged pitchers by default (sidebar checkbox "Include position players who pitched (50)"), including the ROS pitcher view; hitter views untouched. `output/position_player_pitchers.parquet` copied to the dashboard data folder.
- **NOT done (open):** the published projection table, league averages, and the frozen model's fitting sample still include them (0.5% of BF: negligible for league levels; the model-fit effect is untested -- excluding them changes the published numbers slightly and needs a refit); deep link `?pitcher=` to a flagged pitcher shows "not found" unless the checkbox is on; the `possible_two_way` list is unreviewed.

### Refit and regenerate without position players (2026-10-08) -- `fl_production_projection.py`, `fl_ra9_trustmark_projection.py`, check `fl_posplayer_refit_check.py`
- **Done:** `fl_position_player_flag.exclude(raw)` drops the 51 flagged pitchers (2,099 pitches) right after the pitch data is loaded in BOTH scripts, so they are out of the frozen-model fit, the league levels / recentering, the unified bridge fit and the projections. Re-ran `fl_production_projection.py` (frozen Experiment-2 fit, 2027 projections, prospect tables, coefficient JSON) and `fl_ra9_trustmark_projection.py` (air-trait model, RA9 baseline projection). Backups of every replaced file in `output/_backup_before_posplayer_refit/` (the dashboard copies were byte-identical to the old outputs before they were overwritten). Dashboard data folder updated with the nine regenerated files; the "include position players" checkbox was removed (the tables no longer contain them); the ROS pitcher view filters them.
- **Fit sample:** 134 -> 129 pitchers (5 position players had 1H2026 targets). Coefficient changes: K% intercept -.030 -> -.012 (level; the production output is recentred), fl_k_rate_bridge .282 -> .258, wh_sva_level .503 -> .520, spin_z .0108 -> .0085, extension .0132 -> .0108; BB%: blended_BB_rate .970 -> .824, command_score .0060 -> .0123, n0_bridge_bb 43.4 -> 47.6; K-BB residual rho -.149 -> -.249. Bridge: 518 -> 512 pairs, slopes unchanged to the 3rd decimal. **The BB% coefficients moved a lot for 5 rows removed: the fit sample is small (n ~ 130), so they are noisy.**
- **Held-out check (fold 2: features 2025 + 1H2026 -> actual 2H2026; 256 non-flagged pitchers; level-free; pitcher bootstrap on the scored pitchers only):** K% old fit .05458, new .05485, **new worse by .00026 [+.00011, +.00043]**; BB% old .03793, new .03741, **new better by .00052 [-.00106, +.00001], P(better) .95**. Mixed and small; the bootstrap does not include the uncertainty of re-fitting on n ~ 130, so treat both as within noise. Decision: keep the refit (principled: non-pitchers do not belong in the sample) rather than cherry-pick per stat.
- **Effect on projections (732 pitchers, was 782):** mean |change| K% .0026, BB% .0041, K-BB .0049; for pitchers with >= 100 PA (251) max change .011 K%, .014 BB%. Largest movers are 0-PA gap-prior pitchers whose projection leans on the bridge (e.g. K% .158 -> .195). RA9 baseline mean |change| .053 (max .33), P(WAR>0) mean |change| .008 (max .043); replacement RA9 unchanged at 6.963. No negative K%: minimum projected K% is now .133.
- **Still open:** hitter views untouched; the 24 `possible_two_way` pitchers were kept (normal velocity); ROS 2026 snapshots were not rebuilt (they use their own Marcel / stuff pipeline, not these coefficients; only the live path reads the published table).

### HITTERS: survivorship correction for thin-history hitters -- `fl_hitter_survivor_explore.py`, `fl_hitter_survivor_test.py` (2026-10-08); DECISION: no correction, revisit with 2027 data
- **Question:** thin-history hitters were thought to be projected ~.01 wOBA low because the regression is fit on hitters who reach 100 PA next year (selection on the outcome).
- **Target decision (user):** the projection estimates the rate over the PA he actually gets next year, PA-weighted (consistent with the WAR simulation multiplying rate by expected PA). Alternative rejected: the rate conditional on a regular role (>= 100 PA). The size of the "bias" depends on this definition: mean error by prior-season PA <50 / 50-100 / 100-200 / 200-400 / 400+: **>= 100-PA target +.016 / +.009 / +.004 / +.003 / .000; PA-weighted any-PA target +.011 / +.003 / +.002 / +.002 / +.001**; unweighted any-PA target: thin hitters are 33 points WORSE than projected (those who play only a few PA mostly hit badly: the same selection working in reverse). Retention (any PA next year): 15% / 27% / 36% / 56% / 64%.
- **Estimate (all 4 transitions, PA-weighted, hitter-cluster bootstrap, n = 524 hitter-transitions of which 103 have effective history < 100 PA):** regulars +.0021 [-.0020, +.0057]; thin +.0067 [-.0025, +.0159]; thin minus regulars **+.0046 [-.0053, +.0142]**. By transition the thin-minus-regulars gap is +.0027 / +.0097 / +.0023 / +.0042 (same sign in all four). A shift for everyone is unstable by year (-.0032 to +.0083; the league level moves independently) so no global shift.
- **Forward / leave-one-transition-out tests of correction shapes** (reference = intercept only; every model gets the same free intercept; n0 refit on training transitions only; coefficients fit on training transitions only; PA-weighted RMSE .0432 forward, .0451 LOTO): fade-share S1, thin step (<100) S2, 1-P(retain) S3, two bins S4: held-out change **<= .00007 in every case, intervals include 0 (P(better) .35-.69)**. PA weighting gives thin hitters almost no weight, so a thin-hitter correction cannot be detected on accuracy.
- **Size if applied:** +.0046 wOBA for effective history < 100 PA ~ 0.06 WAR for a typical thin hitter (about 160 expected PA).
- **Decision (user):** no correction now; re-test when the 2026 -> 2027 transition exists (5th transition). Alternative offered: apply +.0046 step for history < 100 PA (sign-consistent, symmetric tiny cost). Caveat that stands in the open-issues list: projections for thin-history hitters are conditional on playing, and are slightly low on that basis.

### HITTERS: BB% Swing% term added to the PRESEASON projection -- `fl_hitter_swing.py`, `fl_hitter_projection.py`, check `fl_hitter_swing_preseason_check.py` (2026-10-08)
- **Change (user go-ahead):** proj_BB% += B_SWING x (shrunk Swing% - league) x n0bb / (PA_eff + n0bb), B = -0.387 BB% per unit (fit within-season, SE .030), Swing% = swings per pitch seen pooled over 2025-26 and shrunk by n0 = 92.1 pitches, n0bb = 146.6 (the BB% regression constant), PA_eff = PA_t + .5 PA_prev, result clipped to [0, .4]. Same form as the Whiff% -> K% term. 617 of 621 hitters get a term (mean |term| .006 BB% points, max .033).
- **Implementation check (production path emulated forward: features 2025 + 2024 history, term from 2025 pitches only, production constants, fair intercept in both, 125 hitters with >= 100 PA in 2026):** BB% RMSE .02501 -> .02397, **-.00104 [-.00211, -.00008], P(better) .96** -- matches the earlier study (-.00108).
- **Design choice -- feed the wOBA components?** The term enters proj_bbr BEFORE the component wOBA is built, like the Whiff term for K%, so the displayed wOBA is consistent with the displayed BB%. Test: wOBA blend RMSE .03602 (term not feeding) vs .03600 (feeding), **-.00002 [-.00029, +.00022]**: accuracy-neutral. Chosen for coherence (alternative: keep wOBA untouched, which is how the ROS model does it; the ROS blend uses the regressed rates without terms).
- **Effect on the 2027 hitter table (621 hitters):** proj_BB% mean |change| .0068 (max .033), proj_wOBA mean |change| .0009 (max .0042), WAR mean |change| .014 (max .06); K% unchanged. Backup of the previous outputs in `output/_backup_before_swing_term/`; published to the dashboard (hitter views run without errors).

### ROS hitters: published preseason projection as the prior -- NOT adopted -- `fl_hitter_pre2026.py`, `fl_ros_hitter_published_prior.py` (2026-10-08)
- **Question (user, item 5):** should the HITTER ROS update start from the published preseason projection (as adopted for pitchers in the live path)?
- **Construction (leak-free for every 2026 date):** the production hitter path (`fl_hitter_projection.build_projections`) run on a panel of real seasons <= 2025 only, years shifted +1 so that the real 2025 plays the latest season, constants refit through 2024 -> 2025, EV90 / Whiff% / Swing% from 2025 pitches only (`output/hitter_pre2026_published.parquet`, 617 hitters, 232 "gap" hitters whose last season is 2024). Update identical to the pitcher test: est = lg_now + (n_p dev0 + nu n_ytd (x_ytd - lg_now)) / (n_p + nu n_ytd), n_p = Marcel history weight + n0 (phi = 1), dev0 = published - the 2025 league level (hitters' preseason numbers are not recentred on their own mean; recentring was tried first and removed the returner premium -> worse still), plus tau x the ROS model's own pitch-level increment, tau in {0, .5, 1}.
- **Result (2026, as-of days 7-105, 3,709 hitter-dates, 50% with a published projection):** production ROS is BETTER than every published-prior variant: wOBA .04968 vs .05083 (tau .5), **+.0011 [+.0005, +.0018] worse, P(better) .00**; K% .05088 vs .05123, +.0004 [-.0000, +.0008] (P .07); BB% .03707 vs .03749, +.0004 [+.0001, +.0008] (P .02). Not a park artifact (Mississippi hitters and all others both worse), present for hitters who played the latest season (non-gap n = 1,776: wOBA .04360 vs .04586); gap hitters (n = 73) mixed (K%, BB% slightly better with published, wOBA worse).
- **Why (partly traced):** at the start of a season the published prior and the ROS history prior are nearly the same number (day-7 correlation .98-.99, RMS difference .0045 wOBA): the hitter preseason projection IS the same regression the ROS model runs. Preseason-only (no in-season data, level-free): ROS history with 2 seasons ~ published (wOBA .03944 vs .03949; K% .04349 vs .04319; BB% .02813 vs .02875), **3 seasons slightly better (wOBA .03929, K% .04314, BB% .02789)**. The remaining gap in the combined ROS test (the ROS model's levels, 3-season history, constants refit on each fold's own earlier data) is not fully traced.
- **Decision:** keep the ROS hitter prior as is (history regression + pitch terms). Unlike pitchers, the published hitter projection carries no information the ROS history does not already contain (no bridge / arsenal features for hitters).
- **Follow-up flagged (not done): a THIRD history season in the hitter PRESEASON projection** (currently latest + previous only; the ROS model uses three earlier seasons). ROS evidence: three earlier seasons beat two by .0001 (wOBA, significant), .0002 (K%), .0001 (BB%) in-season, and by ~.0002 / .0000 / .0009 preseason-only (level-free; one season, 541 hitter-dates). A forward test on the production transitions needs 2 transitions with a 3-season history (2024 -> 25, 2025 -> 26) -- thin.

### Open-list decisions, name / id audit (2026-10-08)
- **Two-way list (user decision):** the 24 pitchers with a hitter record but normal fastball velocity (81-93 mph) stay in the pitcher tables with no marker; list kept in `output/position_player_pitchers.parquet` (`possible_two_way`).
- **Baseball Reference Pioneer 2026 data (user):** the user downloaded the Baseball Reference Pioneer stats by hand (no scraping); kept in the Other-League bridge / lookup for the private dashboard. Source: manual download of Baseball Reference Pioneer League 2026 pitching (`output/fl_pioneer_pitching_2026.parquet`).
- **Name / id audit (`fl_name_id_audit.py`; hitters key on the cleaned name, pitchers on id with a canonical-id map):** hitters 645 ids / 618 names: 31 names carry > 1 id, of which 2 are certain collisions (same name, two ids in one game: Graham Brown 5 pitches, Taylor Wright 10 pitches: stray ids, immaterial); the other 29 are non-concurrent re-issued ids (e.g. Christian Fedko: one id in 2025, another in 2026, same team) so merging by name is right. 4 ids carry two spellings: Taeg / Teag Gollert (376 / 57 pitches: history split by 57 pitches), Jacob / Jake Houtz (12 / 7), a stray 12 pitches of "Blake Robertson" on Sam Linscott's id, Dario Borrero / Drew Barragan (81 / 19, one id). Pitchers 820 ids / 782 names: 39 names with > 1 id, 2 same-game collisions (Eli Majick 1 stray pitch, Grif Hughes 112 vs 287 pitches in 2025: two real people, both Evansville), 3 split ids (Seth McCartney / Tristan House 43 / 649, James / Jimmy Gilleran 119 / 1,272, Eli Majick / Pavin Parks 1 / 450). **Not material to any projection (a handful of pitches); no change made.** Alias map for Gollert / Gilleran / Houtz available if wanted.

### P(WAR>0) re-evaluated as a calibration / uncertainty statement, not a classifier -- `fl_pwar_calibration_lens.py` (2026-10-08); display unchanged (user decision)
- **User clarification:** P(WAR>0) is meant to express prediction uncertainty (point estimate + spread, scaled by the projected-innings luck law) in roster-decision terms, not to classify WAR > 0. Brier skill / AUC (fl_pwar_weakness.py) are classifier metrics and are the wrong lens; they stay in the log only as a ceiling analysis. Display decision (user): keep as is, no caption / dash.
- **Calibration lens (2025 -> 2026, out of sample, n = 312 with >= 10 IP in 2026; predictive distribution RA9 ~ Normal(pred, 185.16 / IP_proj)):** coverage of the central 50 / 80 / 90% intervals = **.609 / .833 / .901** (sd of the standardised error .93: spread slightly conservative); logit calibration slope **1.13 (se .35)**, intercept -.32 (se .25). By projected IP: **45+ IP (n = 65): predicted .732 vs observed .723, coverage .554 / .815 / .892, slope 1.13 (se .48), every P bin's predicted value inside its 90% interval -> calibrated where the roster decisions are made**; 30-45 IP (n = 30): .606 vs .667; 22-30 (n = 20): .663 vs .450 (small); **< 22 projected IP (n = 197): predicted .635 vs observed .563, bin 0.6-0.7: .643 vs .580 [.52, .64] -- optimistic by about 7 points, mean standardised error +.31 (actual RA9 above predicted)**.
- **Reading:** the spread is right (slightly wide), the probabilities are calibrated for arms projected 45+ IP, and for the thinnest arms P is about 7 points too high, which is a LEVEL miss (2026 RA9 ran above the 2025-based prediction; mean z +.27 overall, +.13 at 45+) more than a spread problem. PIT deciles are thin in the lowest two (.029 / .042 vs .10): realized RA9 was rarely far below the forecast.
- **Open (not changed):** the low-IP optimism is a candidate level correction (RA9 baseline level / league drift for cup-of-coffee arms); production recentres K% / BB% to the latest league level, the backtest does not, so part of the +.27 is a backtest artifact. Revisit with 2027 outcomes.

### HITTERS: calibration of the WAR distribution and P(WAR>0), by playing time -- `fl_hitter_war_calibration.py`, `fl_hitter_pre2026.build_shifted` (2026-10-08); no change
- **Test:** production hitter path on 2025-and-earlier data only (constants, PA model, pitch terms from 2025 pitches) -> production WAR simulation (3,000 draws per hitter, conditional on playing) -> realised 2026 WAR on the projection's own definition (park-neutral wOBA vs the realised 2026 league, realised fielding and position runs, realised PA, the 2025 replacement and runs-per-win constants). n = 146 hitters with a projection and >= 20 PA in 2026 (one transition).
- **Coverage of the simulated WAR distribution (nominal 50 / 80 / 90%):** **.459 / .753 / .904** (se about .041 / .033 / .024: the 50 and 80% shortfalls are 1.0 and 1.4 se); PIT deciles roughly uniform (.10 / .12 / .08 / .12 / .06 / .06 / .12 / .10 / .09 / .14); mean realised minus projected WAR +.05. By projected PA: 100-200 (n = 33) .455 / .667 / .939; 200-300 (n = 88) .489 / .807 / .909; 300+ (n = 25) .360 / .680 / .840 (small cell).
- **Reliability of P(WAR>0):** mean predicted .756 vs observed .740; bins .4-.6 (n = 20): .514 vs .600; .6-.8 (n = 62): .714 vs .677; .8-.95 (n = 53): .874 vs .830; .95-1 (n = 9): .969 vs .889; all inside their 90% Wilson intervals (two cells of 2 and 4 hitters are not). By projected PA the means are .668 vs .727 (100-200), .766 vs .727 (200-300), .836 vs .800 (300+).
- **Decomposition (realised minus projected):** batting -4.8 runs per 600 PA on average, SD of the standardised batting error .87 (the wOBA luck law is slightly conservative); fielding SD of error 7.1 vs the model's 6.2 runs per 600 PA, position 7.5 vs 6.0; playing time: RMSE 139 vs the model's 124, mean +44 PA (hitters who played >= 20 PA, so partly selection). **Widening those components to the realised SDs moves coverage by < .02 (50 / 80 / 90: .473 / .774 / .904)**, so the small shortfall is not driven by one layer; it is within sampling error.
- **Decision:** no change. The hitter WAR range and P(WAR>0) are calibrated within the resolution one transition and 146 hitters give. Fielding / position residual SDs (6.2 / 6.0) could be raised to about 7 / 7.5 if more fielding data confirm; only 2025-26 carry fielding runs, so it cannot be separated from a year effect. Re-test with 2027 outcomes.

### HITTERS: a THIRD history season in the preseason regression -- ADOPTED -- `fl_hitter_depth3_test.py`, `fl_hitter_projection.py` (2026-10-08)
- **Model:** pred = lg_next + (n_t dev_t + d n_p dev_p + d^2 n_pp dev_pp) / (n_t + d n_p + d^2 n_pp + n0), the third season at weight delta^2 (geometric, no new parameter), n0 refit for the depth. A missing middle season (e.g. played 2026 and 2024, not 2025) now counts: the older season is the "third" season at delta^2.
- **Test (transitions 2022->23 ... 2025->26; a 3-season history exists for 59 of the 219 hitter-transitions in 2024->25 and 2025->26; n0 refit per depth on training transitions only):** forward (train on earlier, score 2024->25 and 2025->26): wOBA RMSE .04064 -> .04035, **-.00030 [-.00058, +.00001], P(better) .94** (-.00105 on the rows that have a 3rd season); K% -.00001 (P .55); BB% -.00008 (P .75); HR/PA -.00005 (P .72); BABIP -.00007 (P .91). Leave-one-transition-out: wOBA -.00031 [-.00065, -.00001] P .96, the rest the same. **Production blend (0.5 direct + 0.5 components), forward 2025->26 (n = 125): .03624 -> .03571, -.00053 [-.00086, -.00023], P(better) 1.00.** Consistent with the ROS finding (3 earlier seasons beat 2).
- **Decision:** adopt for all six stats (K% neutral; the others small and same sign). The pitch-level terms' fades (Whiff%, Swing%, EV90) keep their validated 2-season PA_eff. n0: wOBA 440 -> 433, K% 79.3 -> 79.7, BB% 146.6 -> 146.7, HR/PA 136.8 -> 139.8, BABIP 852 -> 834; component weights K -.157 -> -.162, BB .496 -> .501, HR 1.378 -> 1.395, BABIP .426 -> .438.
- **Effect on the 2027 table (621 hitters):** wOBA mean |change| .0003 (max .009), K% .0014 (max .073: hitters with a thin recent season and a big older season now use the older season, e.g. 19 PA in 2026 + 261 PA in 2024), BB% .0004, WAR mean |change| .006 (max .15). Backup in `output/_backup_before_depth3/`; published to the dashboard.

### HITTERS: validation of the position adjustment and the fielding input -- `fl_hitter_position_test.py` (2026-10-08): CANNOT be validated out of sample; coverage finding, no change
- **Attempt:** predict next-season position runs per PA from (a) zero, (b) the group mean of the predictor-season position, (c) own rate at face value, (d) the production blend (n0 = 100), against the oracle (group mean of the target-season position), forward 2025 -> 2026 and reverse. Result looked bad for every model (RMSE 6.6-7.6 runs per 600 PA; the "zero" model won forward) -- and the oracle at 2.3-3.8 exposed why.
- **Cause (data):** the 2025 WAR table has no position information for 78% of hitters (`pos_innings` = 0 and all PA booked as DH; `primary_pos` present for 22%, and among hitters with >= 100 PA only catchers). So the 2025 position adjustment is a DH-level constant (about -2.8 runs per 600 PA for 2B / 3B / CF / SS) except catchers (+19.5), while 2026 has real position values (C +8.0, SS +7.1, 2B +2.7, 3B +1.2, CF +1.8, LF -5.0, RF -4.9, 1B -8.8 per 600 PA). The two seasons are on a different basis, so a 2025 -> 2026 test cannot say anything about the production formula, and the production formula (own latest-season rate shrunk to the latest-season group rate, n0 = 100) only ever uses the 2026 basis (for 2027).
- **Fielding has the same limit:** 2025 fielding runs exist only for catchers (framing) and outfielders (OAA / outfield runs); infield fielding is exactly 0 in 2025 (no infield data), non-zero for 41% of hitters in 2025 vs 68% in 2026. The 2025 -> 2026 fielding correlation of .74 (earlier whole-chain check, corr .73, RMSE 6.2 vs 10.1) is carried by outfielders (r .95-.97, n = 41) and catchers (.46, n = 15); infielders cannot be tested. The fielding shrink (n0 = 200) was tuned on that outfield / catcher-dominated sample.
- **What the data do support:** within 2026 the SD of hitters' position runs around their group mean is only 2.6-5.7 runs per 600 PA (about 3.7 pooled), versus the 6.0 residual the WAR simulation uses; the simulation value presumably carries position-change risk (not measurable: the only year with positions is 2026, and 2025 positions were copied from 2026 for the 38% in the panel).
- **Consequence for the user's goal:** the position / fielding layers of hitter WAR are the least validated component; the batting layer (wOBA / K% / BB%) is validated on 4 transitions, playing time on 4. Re-test fielding and position with 2027 data (two fully tracked seasons). Until then the infield fielding and position pieces rest on one season. Options (not applied): flag infielders' WAR range as low-confidence; widen the fielding residual for infielders (no data to size it); drop the fielding / position terms for hitters without a 2026 season (gap hitters currently get group position only, zero fielding).
- **Decision (user, 2026-10-08):** leave the hitter position / fielding layers as they are for now. Later task (user): try to fill the infielder fielding gaps (2025 has no infield fielding / position data; options to explore then: reconstruct 2025 infield fielding from play-by-play or pitch-tracking data, or position innings from lineups) so the position and fielding layers can be validated on two seasons.

### REPORTING STANDARD (user, 2026-10-08): every model change shows weighted RMSE AND weighted MAE (old vs new, frozen standard weights, cluster bootstrap) -- `fl_eval_report.py`; recent changes re-reported in `fl_recent_changes_report.py`
Weighted RMSE / weighted MAE, old -> new (change; 90% interval; P(better)):
1. **Third history season, hitter preseason (forward 2024, 2025 folds):** wOBA .04064 -> .04035 (-.00030 [-.00061, +.00002], .93) / MAE .03163 -> .03135 (-.00028 [-.00063, +.00007], .91); K% .03860 -> .03859 / .03008 -> .02992; BB% .02651 -> .02643 / .02102 -> .02098; HR/PA .01320 -> .01315 / .01010 -> .01002 (MAE -.00008, P .91); BABIP .04248 -> .04241 / .03241 -> .03239. **Production wOBA blend 2025 -> 26 (n = 125): .03624 -> .03571 (-.00053 [-.00086, -.00022], 1.00) / MAE .02889 -> .02844 (-.00045 [-.00073, -.00015], .995).**
2. **Swing% term, hitter preseason BB% (forward 2025 -> 26, n = 125):** BB% .02501 -> .02397 (-.00105 [-.00201, +.00004], .94) / MAE .01991 -> .01914 (-.00077 [-.00186, +.00037], .88); feeding the wOBA components: wOBA RMSE .03603 -> .03601 (-.00002, .56) but **MAE .02858 -> .02872 (+.00014 [-.00018, +.00043], P(better) .26): neutral to slightly worse on typical error**; kept for coherence with the displayed BB% (flag: reconsider if a later test makes it worse).
3. **Hitter ROS BB% in-season weight nu = .4 (as-of days 7-105):** scored 2026 (nu chosen on 2025): .03776 -> .03713 (-.00063 [-.00097, -.00030], .998) / MAE .02905 -> .02850 (-.00055 [-.00087, -.00020], .998); reverse, scored 2025: .03911 -> .03884 (-.00027 [-.00058, +.00004], .94) / MAE .03097 -> .03078 (-.00019 [-.00048, +.00012], .87).
4. **Pitcher K% / BB% refit without position players (fold 2, 256 pitchers, level-free):** K% .05458 -> .05485 (+.00026 [+.00011, +.00043], .00) / MAE .04364 -> .04389 (+.00025, .01): slightly WORSE; BB% .03793 -> .03741 (-.00052 [-.00105, -.00000], .95) / MAE .02821 -> .02786 (-.00036 [-.00089, +.00015], .85).
5. **Pitcher ROS with the published preseason projection as the prior (live path; 2026 days 63-105, 1,681 pitcher-dates):** K% .06044 -> .05992 (-.00052 [-.00117, +.00015], .91) / MAE .04866 -> .04823 (-.00043 [-.00106, +.00014], .89); BB% .04169 -> .04125 (-.00045 [-.00113, +.00028], .86) / MAE .03147 -> .03121 (-.00026 [-.00079, +.00028], .79).
- **Reading:** MAE tells the same story as RMSE for every change (same sign, similar significance), so none of the gains is driven by a few outliers; the exception to note is the Swing% -> wOBA feed (RMSE flat, MAE slightly worse) and the K% refit without position players (worse on both).

### HITTERS: is 50/50 the best blend of direct and component wOBA? -- `fl_hitter_blend_weight_test.py` (2026-10-08); no change
- **Test:** blend(lam) = lam x direct + (1 - lam) x components, lam = 0 .. 1 by .1; all out of sample (n0 per stat on the training transitions, component weights on the training transitions only, three-season regression, no Trackman terms); target next-year park-neutral wOBA, >= 100 PA, standard weights; weighted RMSE AND MAE; hitter-cluster bootstrap vs lam = .5.
- **Leave-one-transition-out (n = 409):** RMSE / MAE at lam = 0, .3, .5, .7, 1: .04162 / .03281, .04135 / .03269, **.04128 / .03266**, .04131 / .03267, .04152 / .03289. **Forward (2024->25, 2025->26; n = 211):** .03952 / .03084, .03935 / .03073, **.03932 / .03070**, .03937 / .03076, .03958 / .03097. Nothing between .3 and .7 differs from .5 by more than .0001 (RMSE or MAE); all-component (0) is +.0003 / +.0002 (P(better) .08 / .27), all-direct (1) is +.0002 to +.0003 (P .15-.21); no interval excludes 0.
- **No stable alternative:** the best lam by held-out fold is 0.6 / 0.5 / 0.9 / 0.0 (LOTO) and 0.9 / 0.0 (forward); component weights are stable across folds (K -.14 to -.18, BB .41-.57, HR 1.32-1.52, BABIP .34-.61). By history: < 200 PA (n = 111 LOTO / 54 forward) prefers direct (best .8-1.0; RMSE .04677 vs .04717 at .5 in LOTO, .04372 vs .04375 forward), 200-500 PA prefers components (.2-.3; .04077 vs .04080), 500+ PA prefers .5-.6. A history-dependent weight would add parameters for gains of at most .0004 on 54-111 hitters: not adopted.
- **Decision:** keep lam = .5. The error surface is flat around it and it is the (weak) optimum in both schemes.

### HITTERS: how well does each regressed component predict, and do the components need more shrinkage? -- `fl_hitter_component_check.py` (2026-10-08); read-only, no change
- **Leave-one-transition-out, n0 per stat refit on the other transitions, three-season regression, standard weights (n = 423; BABIP 409):** weighted RMSE / MAE, league mean -> last season at face value -> regressed; skill = 1 - RMSE^2 / RMSE_league^2; slope of (next - league) on (predicted - league):
  wOBA .0458 / .0359 -> .0672 / .0485 -> **.0420 / .0332** (skill 16%; slope .98 [.76, 1.19]); K% .0608 / .0488 -> .0531 / .0386 -> **.0415 / .0320** (53%; .99 [.92, 1.06]); BB% .0365 / .0294 -> .0392 / .0270 -> **.0279 / .0217** (42%; 1.05 [.95, 1.14]);
  HR/PA .0169 / .0131 -> .0162 / .0121 -> **.0132 / .0099** (39%; .99 [.86, 1.12]); **BABIP .0408 / .0317 -> .0677 / .0488 -> .0398 / .0306 (skill 4.6%; slope .92 [.56, 1.25])**; HBP/PA .0155 / .0112 -> .0193 / .0123 -> **.0122 / .0091** (37%; .99 [.84, 1.12]).
- **Reading:** every stat's own regression is calibrated (slopes .92-1.05, intervals contain 1): the per-stat n0 already shrinks the right amount. BABIP is barely predictable (4.6% of variance; last year at face value is 66% WORSE than the league mean), so its n0 = 833 is large and its slope is the most uncertain; it enters wOBA only through the component half with weight .438.
- **Extra shrinkage on the component wOBA?** Held-out (component weights fit on the training transitions only; n = 409): wOBA RMSE / MAE league .0453 / .0356; direct .0415 / .0329 (slope .97 [.74, 1.19]); components .0416 / .0328 (slope **.88 [.67, 1.06]**: mildly over-dispersed, interval contains 1); **50/50 blend .0413 / .0327 (slope .99 [.77, 1.16])**. The component half alone would want about 12% more shrinkage; averaging it with the direct regression (which carries the correct amount) already brings the blend to slope 1, so no additional shrinkage factor is applied. (Consistent with the earlier finding that the pure run-value identity, with no fitted weights, loses by .0018.)

### HITTERS: is the BABIP component helping the wOBA projection? -- `fl_hitter_babip_ablation.py` (2026-10-08); finding: no; removal NOT yet applied (awaiting user)
- **Ablation (old = K%, BB%, HR/PA, BABIP components; new = BABIP removed, weights refit on the training transitions; held out; weighted RMSE / MAE; hitter bootstrap):**
  leave-one-transition-out (n = 409): components alone .04162 -> .04167 (+.00005 [-.00033, +.00038], P(better) .42) / MAE .03281 -> .03276 (-.00005, .60); **50/50 blend .04128 -> .04108 (-.00020 [-.00039, -.00001], .96) / MAE .03266 -> .03241 (-.00025 [-.00046, -.00001], .96)**.
  Forward (2024->25, 2025->26; n = 211): components alone .03952 -> .03970 (+.00018, .29) / MAE .03084 -> .03068 (-.00016, .68); **blend .03932 -> .03918 (-.00014 [-.00040, +.00013], .80) / MAE .03070 -> .03043 (-.00027 [-.00058, +.00001], .94)**.
- **For context, dropping the other components HURTS:** HR/PA blend +.00091 / +.00107 RMSE (P(better) .01), BB% +.00061 / +.00070 (.03-.04), K% +.00013 / +.00014 (not significant).
- **Reading:** BABIP adds nothing the other three components and the direct regression do not already capture (its own skill is 4.6% of variance; its fitted weight swings .34-.61 across folds); removing it is neutral on the components and slightly better in the blend on both metrics. Not applied yet: user asked the question only. Candidate change: drop BABIP from the wOBA components (keep the projected BABIP for display); simpler model, one fewer fitted weight.

### HITTERS: BABIP dropped from the wOBA components -- APPLIED (user go-ahead 2026-10-08) -- `fl_hitter_projection.py`, `fl_hitter_ros.py`
- **Change:** component half of the blend now uses K%, BB%, HR/PA only; refit weights K -.177, BB .516, HR 1.414 (were -.162 / .501 / 1.395 with BABIP .438). The projected BABIP is still computed and shown, but no longer feeds wOBA. Same change in the ROS blend (BETA).
- **Before / after, held out (same rows, from `fl_hitter_babip_ablation.py`):** 50/50 blend RMSE .04128 -> .04108 (-.00020 [-.00039, -.00001], P(better) .96), MAE .03266 -> .03241 (-.00025 [-.00046, -.00001], .96) leave-one-transition-out; forward .03932 -> .03918 (-.00014, .80) / MAE .03070 -> .03043 (-.00027, .94). Components alone: RMSE +.00005 / +.00018 (neutral), MAE -.00005 / -.00016.
- **ROS wOBA (final, as-of days 7-105, `fl_ros_babip_drop_test.py`):** scored 2025 (n = 3,564) RMSE .05966 -> .05968 (+.00002 [-.00002, +.00006], P .25), MAE .04642 -> .04644 (+.00002); scored 2026 (n = 3,709) .04968 -> .04966 (-.00002 [-.00007, +.00003], .73), MAE .03775 -> .03776 (+.00001): neutral, applied for consistency / simplicity.
- **Production effect (621 hitters):** proj_wOBA mean |change| .0007 (max .0037); component wOBA mean |change| .0026 (max .0142); WAR mean |change| .013 (max .09). Fit-sample note: the production fit's in-sample component rows rose from 409 to 423 (BABIP needs AB data), so its LOTO RMSE line is not comparable to the old one (blend .0411 on 409 rows -> .0414 on 423). Backups in `output/_backup_before_babip_drop/`; published; ROS snapshots rebuilt; dashboard hitter views run without errors.

### HITTERS: is Air% related to BABIP? -- `fl_hitter_air_babip_test.py` (2026-10-08); read-only finding, no change
- **Same season (372 hitter-seasons >= 150 PA, >= 60 batted balls, 2025-26):** corr(Air%, BABIP) = **-.19** (GB% +.19); Air% and GB% are almost the same variable here (corr -.997), so their partial effects cannot be separated. HR/PA and K% are positively related to Air% (+.39, +.28) and not to BABIP (-.01, +.05). Direction: more air balls = fewer balls on the ground = lower BABIP (HR are excluded from BABIP).
- **Reliability and persistence (2025 -> 2026, n = 89):** Air% .70, GB% .71, BABIP .18; **last year's Air% predicts next year's BABIP at r = -.30, stronger than last year's BABIP does (.18)**.
- **Forward 2025 -> 2026 (n = 119, Air% shrunk by n0 = 89 batted balls, 2025 balls only; reference = same BABIP regression with a fitted intercept, 10-fold CV intercept in both):** coefficient -.185 BABIP per unit of shrunk deviation (one SD = .046 -> -.0086 BABIP); RMSE .04132 -> .04055 (-.00076 [-.00184, +.00040], P(better) .85); MAE .03222 -> .03139 (-.00083 [-.00199, +.00046], .87): suggestive, not significant; one transition only (Trackman covers 2025-26).
- **Implication (not acted on):** BABIP has a real batted-ball-type signal that the BABIP regression does not use. BABIP was dropped from the wOBA components because the BABIP projection adds nothing; with an Air% prior its skill might rise (4.6% of variance today) enough to earn a place again. Open question for the user.

### HITTERS: the wOBA - xwOBA residual and BABIP -- `fl_hitter_resid_babip_test.py` (2026-10-08); read-only finding, no change
- **Same season (374 hitter-seasons >= 150 PA, Trackman):** corr(wOBA - xwOBA, BABIP) = **+.65** (residual mean +.001, SD .027); with K% -.02, BB% .00, HR/PA +.09, Air% -.07. The residual is mostly "did the balls in play fall in".
- **Persistence 2025 -> 2026 (n = 89):** **residual r = -.01 (no persistence)**, BABIP +.18, wOBA +.41, xwOBA +.63. Last year's residual vs next year's BABIP -.03; xwOBA vs next BABIP -.04.
- **Forward 2025 -> 2026 (n = 119), next-year BABIP, base = production BABIP regression, intercept in both:** residual (shrunk, n0 = 150 / 300 / 600 PA): RMSE .04132 -> .04112 / .04104 / .04100 (-.00020 / -.00027 / -.00031, P(better) .62-.73, intervals include 0), **MAE .03222 -> .03236 / .03239 / .03241 (+.00014 to +.00020, P(better) .37-.41: not better)**; coefficients -.44 / -.62 / -.94 (mean reversion of the residual, not distinguishable from 0). Residual (n0 300) + Air%: RMSE -.00102 [-.00260, +.00046] (P .88) / MAE -.00064 (.78) vs Air% alone -.00076 / -.00083.
- **Reading:** the wOBA - xwOBA residual is BABIP luck: it correlates .65 with this year's BABIP, does not repeat (r = -.01) and does not predict next year's BABIP, so it adds nothing as a BABIP signal. That is consistent with the xwOBA nudge in the wOBA projection (gamma = .485), which pulls a hitter toward his persistent expected wOBA and away from the unstable residual. Air% (r = -.30 with next year's BABIP) remains the only batted-ball-type candidate for BABIP (suggestive, one transition).

### HITTERS: is BABIP related to BIP% (balls in play per PA)? -- `fl_hitter_bipct_babip_test.py` (2026-10-08); no: read-only finding, no change
- **Definition:** BIP% = (AB - K - HR + SF) / PA, the share of PA ending on a ball in play with HR excluded (the BABIP denominator over PA). 863 hitter-seasons >= 150 PA, 2022-26 (counting stats exist for all five seasons); mean BIP% .647 (SD .082), BABIP .322.
- **Same season:** corr(BABIP, BIP%) = **-.02** (year-demeaned; raw -.03); by year +.01 / -.07 / -.05 / -.06 / +.05. BABIP is also unrelated to K% (.00), BB% (+.04), HR/PA (+.05) once the year level is removed.
- **Persistence (2022->23 ... 2025->26, n = 64-89):** BIP% .70 / .78 / .82 / .80 (a stable contact-profile skill) vs BABIP .31 / .29 / .24 / .18. Last year's BIP% vs next year's BABIP: -.17 / -.26 / -.05 / +.04 (no stable sign).
- **As a BABIP predictor (BABIP regression + b x shrunk BIP% deviation, n0 = 150 PA; coefficients by fold -.05 to -.14; reference has the same fitted intercept; weighted RMSE / MAE):** leave-one-transition-out (n = 409) .03983 -> .03980 (-.00003 [-.00036, +.00034], P(better) .53) / MAE .03066 -> .03064; forward (n = 211) .04244 -> .04277 (+.00033 [-.00031, +.00103], .21) / MAE .03250 -> .03310 (+.00060 [-.00002, +.00127], .06). No help (forward slightly worse).
- **Reading:** BIP% is a persistent hitter trait but does not determine BABIP; BABIP's real structure is in the type of ball in play (ground vs air: Air% r = -.19 same season, -.30 to next year's BABIP), not in how often the ball is in play.

### HITTERS: Pop-up rate and Oppo% vs BABIP -- `fl_hitter_popup_babip_test.py`, `fl_hitter_oppo_babip_test.py` (2026-10-08); read-only findings, no change
- **Pop-up rate** (tracked batted balls with launch angle > 50; league 9.1%, hitter SD 3.7 points): same season (n = 372 hitter-seasons >= 150 PA and >= 60 tracked balls) corr with BABIP **-.40** (-.11 with BABIP on non-pop-up balls), slope -.0048 BABIP per +1 point of pop-up rate, corr with Air% +.67. Reliability r = .38 at 82 balls per half (n0 = 135 balls); 2025 -> 2026 persistence .54 (BABIP .18, Air% .70); **PU%_2025 vs BABIP_2026 r = -.38** (Air% -.30). Forward 2025 -> 2026 (n = 119, fitted intercept in both): alone RMSE .04132 -> .04093 (-.00039 [-.00143, +.00075], P(better) .69) / MAE .03222 -> .03205 (-.00017, .59).
- **Oppo%** (share of tracked balls with spray direction > 15 deg toward the opposite field; sign convention verified: RHB pull = negative direction, LHB pull = positive; league 18.0%, hitter SD 5.3 points): same season corr with BABIP **+.22** (slope +.0019 per point), with Pull% -.78, Air% -.43, Pop-up% -.39, K% -.22. Reliability r = .49 at 81 balls per half (n0 = 86); persistence .72; **Oppo%_2025 vs BABIP_2026 +.31**. Forward (n = 119): alone RMSE .04132 -> .04067 (-.00065 [-.00192, +.00075], .78) / MAE .03222 -> .03139 (-.00082 [-.00217, +.00062], .81).
- **Combined (forward, n = 119):** Air% alone .04050 / .03130 (-.00082 / -.00092, P .87); Oppo% + Air% .04064 / .03142; Oppo% + Air% + Pop-up% .04101 / .03175 (-.00031 / -.00047): **the three measures are collinear views of one batted-ball profile (ground / oppo vs air / pull / pop-up), together no better than Air% alone.**
- **Reading:** BABIP has a real, persistent batted-ball-profile signal (all three correlate .2-.4 with next year's BABIP, 2-3x BABIP's own persistence) but one transition (n = 119, intervals include 0, P(better) .6-.9) cannot establish it or pick the best measure; Air% (the one already computed for the park model) is the simplest candidate. Not implemented.

### HITTERS: Air% as a prior in the BABIP component of the wOBA blend -- `fl_hitter_air_babip_component_test.py` (2026-10-08); result: does not earn BABIP back; no change
- **Versions of the component half (blend always 0.5 direct + 0.5 components):** V0 K%, BB%, HR/PA (current); V1 + BABIP regressed as before (old production); V2 + BABIP with the Air% prior (b fit on training folds). Forward 2025 -> 2026, n = 119 hitters, production three-season regressions (n0 from 2022-24), Air% from 2025 balls only (n0 = 89); Air% coefficient and component weights fit by hitter-grouped 10-fold CV x 15 repeats on this one transition, same free intercept in every version; weighted RMSE and MAE.
- **BABIP prediction itself:** Air% prior RMSE .04132 -> .04054 (-.00077 [-.00187, +.00040], P(better) .85); MAE .03222 -> .03137 (-.00085 [-.00201, +.00050], .87).
- **wOBA blend:** V0 (no BABIP) .03521 / MAE .02774; V1 (BABIP, no Air%) .03547 / .02797; V2 (BABIP + Air%) .03537 / .02790. **V2 vs V0 (current): +.00016 RMSE [-.00012, +.00048], P(better) .20; MAE +.00016 [-.00017, +.00050], .20: not better. V2 vs V1: -.00009 [-.00028, +.00008], .81 / MAE -.00007 (.72). V1 vs V0: +.00025 [+.00009, +.00041], P(V1 better) .004 / MAE +.00023 (.004): the old BABIP component is clearly worse on this sample, confirming the drop.**
- **Reading:** the Air% prior improves the BABIP projection itself (suggestively) but only recovers about a third of the damage the BABIP component does to wOBA; the improved component still does not beat leaving BABIP out. Decision: BABIP stays out of the wOBA components; Air% for BABIP is not implemented. Re-test with 2027 data (two Trackman transitions).

### Player page: why projected WAR looks low next to "Published WAR" (user question 2026-10-08); no model change, page caption + WAR/600 column added
- **Comparison (119 hitters with >= 300 PA in 2026):** mean published 2026 WAR **1.59** vs projected 2027 park-neutral WAR **1.00**. Per 600 PA the gap is 6.2 runs of 28.4 (published) vs 22.2 (projected): batting **+6.8 vs +1.3** (5.5 runs: regression to the mean), baserunning +0.8 vs 0 (not in the projection by design), position -1.0 vs -1.0, fielding +0.2 vs +0.2, replacement 21.7 vs 21.7, park ~0. PA: published at the realised 390 PA, projected 319 (-18%). Decomposition of the .59 WAR gap: rate (batting) ~.32, baserunning ~.05, PA ~.22 (at the projected rate; small interaction).
- **Is the projection too conservative? Backtest (2025-and-earlier data -> 2026, 116 hitters with >= 150 PA in 2026):** WAR per 600 projected vs realised by projected quartile: .38 / .75, 1.60 / 1.43, 2.53 / 2.01, **4.41 / 3.23** (top quarter projected +4.4, produced +3.2); slope of realised on projected = **.61** (1 = right amount of regression; < 1 = not regressed enough), intercept .49. The same hitters' 2025 published WAR / 600 was 2.70, the model's 2026 projection 2.41, what happened 1.89 (n = 89). So the projection sits between last year's published number and the outcome, closer to the outcome, and is if anything slightly high at the top.
- **PA model check (returners, all four transitions):** mean PA next year vs the model by prior PA: <150 190 / 188, 150-300 251 / 254, 300-400 306 / 304, 400-450 328 / 330: calibrated. **But the PA model is season-length blind**: regular season 2025 = 96 games, 2026 = 102 (user), 2022-24 about 96 (max games played by a hitter 96); PA per hitter with >= 100 PA: 275 / 272 / 264 / 261 / **289** (2022-26, +10.7% in 2026 on +6.25% games). If 2027 is 102 games the PA projections (fit mostly on ~96-game seasons) run low by roughly 6%. Candidate change (not made): model PA per team game, project at the 2027 game count; needs the 2027 schedule length.
- **Page change:** hitter player page history table gains "Published WAR / 600 PA" and the caption now explains regression to the mean, regressed PA, baserunning, and the backtest result.

### HITTERS: season-length-aware playing time, 2027 = 102 games -- APPLIED -- `fl_hitter_pa_games_test.py`, `fl_hitter_projection.py` (2026-10-08)
- **Inputs (user):** regular season 2025 = 96 games, 2026 = 102, 2027 = 102; 2022-24 taken as 96 (the maximum games played by a hitter on a team was 96 in each year; the median of the team maxima, 91.5-94, matches 2025's 91.5 in the tracked data).
- **Model:** PA per team game, PA_next/G_next = 1.622 + 0.404 PA_t/G_t (returners, 524 hitter-transitions, resid SD 1.31 per game = 133.8 PA at 102 games, was 129.0); projected PA = 102 x (1.622 + .404 PA_t / G_latest), floor 20; gap hitters keep the 166-PA constant (it came from 2026, a 102-game season). The WAR simulation's PA SD is now 133.8.
- **Held-out PA (returners, equal weights, leave-one-transition-out), RMSE / MAE old -> new:** all four transitions 129.7 / 111.3 -> **128.6 / 110.6** (-1.11 [-1.76, -0.38], P(better) .998; MAE -0.68 [-1.34, -0.00], .95); 2025 -> 2026 only (the 102-game season) 141.8 / 125.2 -> **139.1 / 121.6** (-2.70 [-4.91, -0.74], .98; MAE -3.56, .998; bias +33.2 -> +17.6 PA); the three 96 -> 96 transitions 124.6 / 105.9 -> 124.2 / 106.3 (RMSE -0.41, MAE +0.45: neutral, as it should be).
- **Effect on the 2027 table:** the old model already contained the 2026 transition in its intercept, so the correction is small: mean projected PA +3.3 (max +6.8), WAR +.008 (max +.04); the 119 regulars (>= 300 PA, non-gap): PA 319 -> 323, WAR 1.00 -> 1.01, 80% range width 3.10 -> 3.14. **The season-length fix does NOT explain the gap to published WAR; that gap is regression to the mean (see the player-page entry).**
- Backups in `output/_backup_before_pa_games/`; published.

### DASHBOARD: "chance of playing next season" removed (user, 2026-10-08)
- Removed the hitter Player Page metric "Chance he plays in the league again" (the PA / Mississippi wOBA metrics now sit in two columns), the pitcher Player Page sentence "Chance he pitches in the league again at all" and the two methodology notes pointing to it, and `p_return` from the pitcher merge. The p_return columns stay in the data files (not displayed anywhere). Hitter and pitcher views run without errors.

### HITTERS: quality-aware playing time (top-end WAR calibration) -- APPLIED -- `fl_hitter_pa_quality_test.py`, `fl_hitter_war_topend_check.py`, `fl_hitter_projection.py` (2026-10-08)
- **Trigger (user):** wOBA / wRC+ projections look calibrated but projected WAR does not: the best projection was ~2.6 WAR while many hitters finished > 4 and several > 5. Backtest (production path on 2025-and-earlier data -> 2026, 146 hitters >= 20 PA): top 10 by projected WAR projected 2.46 vs realised 3.13 WAR, PA projected 298 vs realised 388; the 8 hitters who finished >= 4 WAR were projected at 230-320 PA and played 405-481 (PIT .84-.99). Batting rates were fine (per-600 slope .61); **the miss was playing time: the PA model used only last year's PA, but better hitters play more.**
- **Model:** PA/game_next = 1.732 + 0.344 PA_t/G_t + 14.8 q, projected PA = 102 x that (floor 20), q = the hitter's projected wOBA deviation (direct three-season regression); resid SD 132 PA. +.010 wOBA = +15 PA at 102 games (470 PA last year: 323 PA at q = -.010, 338 at 0, 353 at +.010, 368 at +.020).
- **Held-out PA (LOTO, 524 returners, equal weights) RMSE / MAE old (per-game, no quality) -> new:** 128.6 / 110.6 -> **126.8 / 108.0** (-1.81 [-3.18, -0.36], P(better) .98; MAE -2.69 [-4.05, -1.17], 1.00). Top decile of projected quality (n = 53): actual 344 PA, old projects 303 (bias +40), new 342 (+2). By quality quintile (actual / old / new): Q1 215 / 248 / 219, Q3 224 / 233 / 234, Q5 320 / 295 / 324.
- **Backtest of projected WAR (146 hitters; old = previous per-game PA model, new = production; equal weights):** WAR RMSE 1.221 -> 1.211 (-.010 [-.036, +.013], P .76), MAE .991 -> .993 (+.0025, .43); top 40 by projected WAR RMSE 1.581 -> 1.541 (-.040, .84), MAE 1.348 -> 1.339. **Top 10: projected 2.58 -> 2.88 vs realised 3.15-3.17; top 20: 2.27 -> 2.49 vs 2.43-2.57; top 30: 2.08 -> 2.22 vs 1.95-1.99; max projected 3.01 -> 3.48.** Expected count of hitters at >= 4 / >= 5 / >= 3 WAR under the new simulated distributions: **6.0 / 2.3 / 14.7 vs realised 8 / 2 / 10**; coverage of the 50 / 80 / 90% intervals .445 / .788 / .918 (above p90 11%, below p10 10%); the top 10 by projected WAR have mean PIT .53 (calibrated). The 8 realised >= 4 WAR hitters still sit high in their own distributions (PIT .74-.99): the realised stars are the upper tail of a calibrated distribution, not evidence of a shifted one.
- **Effect on the 2027 table:** projected PA mean |change| 9.7 (max +89 / -63), WAR mean |change| .026 (max +.73 / -.17); top projection (park-neutral, non-gap): Czech 3.51 WAR (428 PA, 80% range 1.3-5.8), Crowl 3.33, Bender 2.70, Wright 2.70. Backup in `output/_backup_before_pa_quality/`; published.

### Hitter history table: wRC+ columns (user request 2026-10-08)
- Added "wRC+ (park-neutral)" for every season (100 x (1 + (park-neutral wOBA - league) / (1.0175 x league runs per PA)); league runs per PA 2022 .1446, 2023 .1399, 2024 .1320 (StatsCrew league runs / PA), 2025 .1452 (backed out of the published wRC+), 2026 .1529 (war_context)) and "Published wRC+" for 2025-26 (the pipeline's park factor). Regulars (>= 300 PA) average 113.4 / 113.4 (2025) and 107.4 / 107.4 (2026) on the two bases, so the park-neutral column is consistent with the published one.

### HITTERS: is the playing-time model any good, and can it be improved? -- `fl_hitter_pa_model_search.py` (2026-10-08); no change
- **Target:** PA next year at 102 games for hitters who return (524 hitter-transitions, mean 266, SD 145), leave-one-transition-out, equal weights. **RMSE / MAE / R^2 vs a constant:** league mean 139.2 / 120.3 / .00; last year's PA at face value 151.4 / 114.6 / -.18; PA_t/G only 128.6 / 110.6 / .15; **current (+ projected wOBA quality) 126.8 / 108.0 / .17**.
- **Candidates that do not beat it (RMSE / MAE; change vs current [90%]; P(RMSE better)):** + PA two seasons ago 127.1 / 108.4 (+0.28, .19); + x^2 + x*q 126.8 / 108.0 (+0.05, .46); + all of those 127.2 / 108.3 (+0.40, .21); regular / part-timer logit split 127.9 / 109.8 (+1.10, .06); gradient boosting 130.8 / 109.7 (+4.06, .00); random forest 129.0 / 109.0 (+2.17, .05).
- **Error by last year's PA (current model):** <100 PA (n = 114) RMSE 145, bias +6; 100-200 (89) 130, -10; 200-300 (122) 119, +3; 300-400 (146) 112, +7; 400+ (53) 133, -15: no systematic bias at any level.
- **Why it is poor in absolute terms:** the outcome is bimodal and driven by things the data does not contain (roster moves, injuries, role, age, contract / team context): of last year's regulars (>= 300 PA pace) 63% are regulars again; 37% of last year's part-timers become regulars. The model beats every available baseline by 9-16% in RMSE; no feature in the data adds to it.
- **Where it matters:** the WAR range already carries this error (PA SD 132 enters the simulation), so the interval is the honest statement; the point WAR is "if he plays a typical role for his profile". Possible product change (not made): an editable projected-PA input on the Player Page that rescales his WAR (rate per PA x the PA the front office expects).

### Dashboard: hitter WAR rate shown per 400 PA, not 600 (user, 2026-10-08)
- Leaderboard column "WAR/400", Mississippi Targets "WAR/400 (MIS)", the sort option, staff-median metric, the rank-on-staff basis, the Player Page runs build ("Runs / 400 PA", "WAR / 400 PA"), the history table ("Published WAR / 400 PA") and the explanatory text all use 400 (a constant `PER` in `hitters_view.py`); the data files still store per-600 rates (`war_per600`, `rar_per600`) and the dashboard rescales linearly (x 400 / 600). Backtest caption rescaled (top quarter projected +2.9 WAR per 400 PA, produced +2.2). Hitter views run without errors.

### PITCHERS: does a blend of the full model with prior history help (as the 50/50 blend does for hitter wOBA)? -- `fl_pitcher_blend_test.py` (2026-10-08); no: no change
- **Test:** blend(lam) = lam x model + (1 - lam) x prior history, lam 0 .. 1 by .1, level-free (same fitted intercept), standard inverse-variance weights, pitcher-cluster bootstrap, weighted RMSE and MAE. K% / BB% / K-BB%: fold 2 (features 2025 + 1H2026, targets actual 2H2026, 256 pitchers; frozen production model without position players; prior history = the naive regressed own rate, no stuff / bridge / command terms). RA9: 2025 -> 2026 out-of-sample build (312 pitchers, IP-weighted), prior history = own 2025 RA9 shrunk toward the league (n0 = 50 / 100 / 200 IP).
- **K%:** model alone RMSE .05488 / MAE .04392; **50/50 .05495 / .04377 (+.00007 RMSE, P(better) .44; MAE -.00014)**; best lam .8 .05476 / .04373 (-.00012 [-.00050, +.00022], P .71); prior history alone .05619 (+.00131, .11). Flat from lam .4 to 1.0.
- **BB%:** model alone .03746 / .02790; **50/50 .03832 / .02855 (+.00086 [+.00029, +.00144], P(better) .006; MAE +.00065)**; best lam 1.0; history alone .04008 (+.00262, .000).
- **K-BB%:** model alone .07205 / .05568; **50/50 .07326 / .05634 (+.00122 [+.00022, +.00226], P .016)**; best lam 1.0; history alone .07591.
- **RA9 (n0 = 100 IP; 50 / 200 similar):** model alone 2.0335 / 1.5363; 50/50 2.0368 / 1.5419 (+.0033, P(better) .36); best lam .8 2.0319 / 1.5366 (-.0017 [-.0091, +.0064], .68); own 2025 RA9 alone 2.0647 (+.031, .10).
- **Reading:** unlike the hitter blend (two different model structures: direct regression and components), the pitcher model already contains prior history as its strongest regressor (the naive rate has coefficient ~.51 in the K% model and the blended rate .82 in the BB% model), so averaging it back in just undoes the stuff / bridge / command information: neutral for K% and RA9, clearly worse for BB% and K-BB%. The analogous structural blend for pitchers (direct K-BB% vs composed from K% and BB%) was tested earlier (composed beats direct, see the half-season backtest entries).

### PARK: the data-driven TrustmarkPark effect vs the current Mississippi translation (user: Air% -> Trustmark profiles; MIS-translated projection; handedness) -- `fl_park_woba_fe.py`, `fl_park_profile_fe.py`, `fl_park_profile_oos.py` (2026-10-08); findings only, NO MODEL CHANGE YET
- **Published park factors (war_players.park_factor, team level, 2025-26 only):** 96.8-103.7 across the 18 teams; Mississippi 99.7 (2025) / 99.3 (2026), raw 100.1 / 99.7: Trustmark is ~neutral for wOBA. No park factors exist for 2022-24 (no home / road data), so a wRC+ target with a team wOBA factor can only be built for 2025-26 unless the 2025-26 factors are carried back.
- **Own estimate (135,382 PA, 2025-26; hitter + pitcher + year + park fixed effects; park effect vs the league-average park; SE by bootstrap over games):** TrustmarkPark wOBA **+.0007 (SE .0083) = park-factor scale 100.2**; all parks range -.027 (Hinchliffe) to +.028 (Thomas More), SEs .007-.010 (individual park estimates are noisy). **Outcome decomposition at Trustmark (per PA, % of the league rate): HR -1.33 points (-55%), K% -2.42 points (-12%), BB% +0.70 (+6%), singles +1.55 (+9%), 2B+3B +0.22 (+5%), wOBA +.0007: the HR suppression (HR factor .446 confirmed) is offset by fewer strikeouts and more singles / walks.**
- **Profile (Trustmark x hitter profile; profile measured on his NON-Trustmark PA / balls; cluster-robust SEs by game; n = 131,688 PA), per +1 SD:** power (shrunk HR/PA) wOBA -.0199 (t -2.8; HR/PA -.0101, t -3.9); Air% wOBA +.0271 (t 3.3; 2B+3B +.0091 t 2.5; singles +.0102 t 1.6); left-handed vs average +.0173 (t 1.2; singles +.0257 t 2.0, BB% -.0211 t -2.0). By side (own-league park baseline): LHB wOBA +.0108 (HR -.0114, singles +.0314), RHB -.0070 (HR -.0146, singles +.0051, BB% +.0152). Air% and power profiles correlate +.42 (so air-ball hitters without power gain, power hitters lose).
- **Out-of-sample (fit on 2025, score on 2026; 44 hitters with >= 20 Trustmark PA = 2,132 PA; observed effect = Trustmark wOBA minus a baseline from his 2026 non-Trustmark PA; very noisy, hitter-level RMSE ~.105):** observed mean effect +.0075 (full park); predicted mean: current translation **-.0198**, global-only +.0161 (2025 fit; pooled 2025-26 is +.0007), profile model +.0148. RMSE / MAE vs the current translation: no adjustment .1065 / .0775 (vs .1095 / .0860; MAE P(better) .99), global only .1066 / .0754 (.97), **profile model .1040 / .0761 (dRMSE -.0055 [-.0152, +.0070], P .77; dMAE -.0099, P .92)**, power only .1062 / .0758, handedness only .1064 / .0753. Calibration slope of observed on predicted: current .60, profile model **.99**, handedness only 1.04. Profile vs global-only: dRMSE -.0025 [-.0069, +.0025], P .81; MAE +.0007, P .45: the profile adds little beyond the global effect at this sample size.
- **Reading:** (1) the current Mississippi translation (average -.012 wOBA for a half-time Trustmark hitter, about -.020 full park) is contradicted by the published factor, by the fixed-effects estimate and by the 2026 out-of-sample outcomes (it is the worst model on MAE, P(alternatives better) .92-.99): it counts HR suppression (and fewer hits per air ball) but not the offsets (K%, singles, BB%). The same HR-factor logic drives the pitcher-side Mississippi RA9 shift (-0.435 runs/9 for half the games): likely overstated too, not yet tested. (2) Profile-specific effects exist in sign and size (power -, Air% +, LHB +; in-sample t 1.2-3.3) and the profile model is the best calibrated out of sample, but cannot be told apart from a flat global effect with 44 test hitters. (3) Nothing has been changed; the dashboard's MIS columns still use the old translation.

### HITTERS: Mississippi (TrustmarkPark) translation REPLACED by the profile model -- APPLIED (user decision 2026-10-08) -- `fl_park_profile_final.py`, `fl_hitter_park.py`, `fl_hitter_projection.py`
- **Model:** full-park Trustmark wOBA effect per PA = global + power + Air% + handedness terms, pooled 2025-26 (131,688 PA, 7,005 at Trustmark; hitter + pitcher + year + park fixed effects; profile measured away from Trustmark; cluster-robust SEs by game), each coefficient shrunk by the positive-part James-Stein factor max(0, 1 - 1/t^2): **global +.0005 (SE .0083, t .06) -> 0; Air% +.0271/SD (t 3.3) -> +.0246; power -.0199/SD (t -2.8) -> -.0174; left-handed +.0173 (t 1.2) -> +.0058** (z's standardised on the PA-weighted pool; left = share of his PA from the left side, so switch hitters enter at their share). Files: `output/trustmark_profile_coefs.json`, `output/hitter_park_profile.parquet` (621 hitters). The shift for a Mississippi hitter is half of this; unknown profile = the average hitter (0). History: park-neutral wOBA = observed wOBA - (share of his PA at Trustmark) x his effect (2025-26 only: 2022-24 carry no Trustmark share). HR/PA neutralisation (HR factor .446) is unchanged. The old translation is kept as `neutralise_hr_model` / `translate_hr_model`.
- **Effect on the 2027 hitter table (621 hitters):** Mississippi wOBA shift mean **-.0114 -> +.0001** (SD .0031 -> .0095; new range -.030 to +.031); park-neutral projected wOBA mean |change| .0005 (max .0027); park-neutral WAR mean |change| .011 (max .07); Mississippi WAR mean +.009 (mean |change| .13, max .78); park fit (runs per season vs the average hitter) SD .87 -> 2.30; Mississippi hitters' neutral wOBA .3605 -> .3596. Examples: Booker (left, 2.8% HR, 42% air) -5.4 runs; Furr (right, 55% air, power-light) +2.8; Nicklaus (left, 56% air) +2.6. Dashboard text for MIS wOBA / Park fit rewritten; hitter views run without errors. Backup in `output/_backup_before_park_profile/`.
- **Not changed (flag):** the pitcher-side Mississippi RA9 shift (-0.435 runs/9 for half the games) is built on the same HR-factor logic and is likely overstated for the same reason; K% / BB% history is not neutralised for Trustmark's K% (-2.4 points) / BB% (+0.7) effects.

### HITTERS: park-adjusted wRC+ as the model target (user request 2026-10-08) -- `fl_hitter_wrcplus_target_test.py`; result: marginally better, within noise; NOT adopted (awaiting user)
- **Park adjustment (published team wOBA park factor):** wOBA_adj = wOBA - k (PF/100 - 1), k = .15 (calibrated on the published 2025-26 wRC+: the published adjustment is -k (PF-100)/100 + a constant, k .138 / .157, corr -.95 / -.995); PF per team = 2025-26 average of war_players.park_factor (96.8 / 97.4 Washington ... 103.7 Florence), carried back to 2022-24 (assumption: no home / road data); Empire State and New England (no current factor) get 0; recentred so each season's league wOBA is unchanged. wRC+ = 100 (1 + (wOBA_adj - league) / (1.0175 x league runs per PA)) with the history-table run environments. **Check: constructed vs published wRC+, 2025-26, 455 hitter-seasons >= 100 PA: corr .999, mean diff -0.03, RMS 1.33 points.** The park adjustment is small: SD .0028 wOBA (about 0.8 wRC+ points).
- **Models (three-season regression, delta .75, n0 refit on training transitions; target = next-year park-adjusted wRC+, >= 100 PA; standard weights fit on the wRC+ errors; weighted RMSE / MAE in wRC+ points):** A current (park-neutral wOBA in wOBA units, prediction converted to wRC+); B park-adjusted wOBA in wOBA units; C park-adjusted wRC+ in wRC+ units (the proposal). **Leave-one-transition-out (n = 423):** A 28.756 / 22.691; B 28.759 / 22.672 (+.004 RMSE [-.080, +.075], P(better) .52; MAE -.019, .66); **C 28.723 / 22.646 (-.033 [-.140, +.081], P .65; MAE -.046, .71)**; C vs B -.037 (P .81). **Forward (2024->25, 2025->26; n = 219):** A 26.298 / 20.480; B 26.273 / 20.401 (-.025, .65; MAE -.079, .83); **C 26.192 / 20.345 (-.106 [-.272, +.061], P .85; MAE -.134, .91)**; C vs B -.081 (P .82).
- **Reading:** both the team park adjustment and the wRC+ units are directionally better (about 0.1 wRC+ point on an RMSE of 26) but no interval excludes zero; with a park-adjustment SD of only 0.8 wRC+ points there is little for them to fix. Not adopted: the gain is not distinguishable from zero, 3 of 5 seasons would rest on carried-back park factors, and the existing model already displays wRC+ (park-neutral). If wRC+ is preferred for presentation, the history table can show park-adjusted wRC+ (published for 2025-26) without changing the model.

### PITCHERS: the matching TrustmarkPark test (user 2026-10-08) -- `fl_park_pitcher_fe.py`; findings only, NO CHANGE YET
- **Global effect on the pitcher's side is the same estimate as on the hitter's side:** wOBA allowed at Trustmark **+.0007 (SE .0083) = +0.03 runs/9 for a full-time Trustmark pitcher** (x 4.512 BF/IP x 9 / 1.0175), against the current pitcher translation's about **-0.87 runs/9 at the league air-ball rate (-0.435 at half the games)**: 2.6 SEs apart (SE .0083 wOBA = +-0.33 runs/9). Outcomes: HR -1.33 points, K% -2.42, BB% +0.70, singles +1.55, 2B+3B +0.22 per PA (same table as the hitters).
- **Trustmark x pitcher profile (131,767 PA; per +1 SD, wOBA points allowed per PA, positive = worse for the pitcher; cluster-robust SEs by game):** HR-prone (shrunk HR/PA allowed): HR/PA **-.0114 (t -6.1)**, singles +.0176 (t 3.0), wOBA -.0075 (t -1.05: the HR relief is offset by singles); K% allowed: **wOBA +.0192 (t 2.7)** (K% -.0277, t -4.8; singles +.0181): strikeout pitchers lose the K advantage the park removes; Air% allowed: wOBA -.0024 (t -0.3), BB% -.0094 (t -1.9); left-handed pitchers: wOBA +.0126 (t 0.85), K% -.0243 (t -2.1). Air% and HR-prone profiles correlate +.37.
- **Out of sample (fit on 2025, score on 2026): only 11 pitchers with >= 30 Trustmark BF (1,250 PA), so it cannot adjudicate.** The 2025 fit gave a POSITIVE global effect (+.0166); 2026 observed -.0335 wOBA allowed (-1.34 runs/9; pitchers did better at Trustmark than their baseline). RMSE / MAE: current translation **.0415 / .0294 (best)**, no adjustment .0517 / .0399, global (2025) .0637 / .0528, global + profile .0604 / .0510 (all worse than current, P(better) .00-.04); slopes -1.7 / 1.5. Year-to-year sign flips of the Trustmark effect (+.017 in 2025 vs -.034 for this 2026 subset) show how noisy single-season, few-pitcher estimates are; the pooled two-season estimate (+.0007) is the better-supported number but is itself +-.0083.
- **Reading:** the pooled evidence (published park factor 99.5, fixed-effects +.0007, identical park outcomes) says the pitcher-side average shift should be near 0 as on the hitter side, with profile terms (HR-prone pitchers gain on HR but lose singles; high-K pitchers lose); the tiny 2026 pitcher test favours the old translation but rests on 11 pitchers whose 2025 -> 2026 sign flipped. Not changed. The pitcher Mississippi WAR also uses a Trustmark-shifted replacement level (ra9_replacement_trustmark_half), so a global shift partly cancels in runs vs replacement but not in the displayed Mississippi RA9.

### PARK EFFECTS RE-ESTIMATED WITH VISITING PLAYERS ONLY + Mississippi matched pair; PITCHER translation REPLACED (user 2026-10-08) -- `fl_park_visitors_fe.py`, `fl_park_matched_pair_ms.py`, `fl_park_final_visitors.py`, `fl_pitcher_park.py`, `fl_ra9_trustmark_projection.py`
- **User point:** park effects should be measured with visiting players only (home players can be built / selected for the park) and the Mud Monsters' home-vs-road should be checked as a matched pair. **Design:** wOBA = year + hitter + pitcher + [hitter x home] (hitter side) or [pitcher x home] (pitcher side) + park; the per-player home effects absorb every home participant's home PA, so the park effect is identified from VISITING hitters (hitter side) or VISITING pitchers (pitcher side). Game-cluster bootstrap (B = 100-200, fixed effects refit each draw).
- **Global Trustmark effect (wOBA vs the league-average park):** visiting hitters **-.0126 (bootstrap SE .0118, 90% [-.030, +.009])**, visiting pitchers (wOBA allowed) **+.0127 (SE .0123, [-.005, +.034])**; opposite signs because each side faces a home staff / lineup whose fit to the park contaminates the estimate (home pitchers do better at home, home hitters do better at home). **Average of the two = +.0027 (bootstrap SE .0086, 90% [-.012, +.015], t .31): the park is neutral on wOBA (= 0.0 runs/9), as in the all-PA estimate (+.0007).** Other parks' visitor-only estimates in `fl_park_visitors_fe.py` (SEs .01-.02 each; ranking broadly unchanged: Hinchliffe / Wild Things lowest, Thomas More / Wintrust highest).
- **Matched pair, Mud Monsters (same players, home vs road, beyond the league-wide home advantage; cluster-robust SEs):** hitters **+.0193 wOBA (SE .0121, t 1.6)** extra at home, pitchers **-.0192 (SE .0114, t -1.7)** extra wOBA allowed at home (both better at home); average gain **+.0192 (SE about .008, t 2.3)** = a team home-field fit worth about +0.77 runs/9-equivalent. League context (point estimates, 18 teams): hitters' extra home effect SD .021, pitchers' SD .016; Mississippi's hitters rank 14th of 18, its pitchers 4th best. So the park is neutral for visitors while Mississippi's roster plays better at home than the average team's does: consistent with a roster built for Trustmark (or luck), which is exactly what the all-PA estimates would have mixed into the park / profile terms.
- **Profile terms, visitors only (bootstrap B = 200; t from the bootstrap SE; positive-part James-Stein shrink; final = shrunk):** HITTERS per +1 SD: Air% **+.0323 (t 3.5, 90% [+.018, +.048], same sign in 100% of draws) -> +.0296**; power **-.0157 (t -1.9, [-.030, -.004], 98%) -> -.0112**; left-handed +.0187 (t 1.05) -> +.0016; global 0. (All-PA had Air% +.0246, power -.0174, left +.0058.) PITCHERS per +1 SD (wOBA allowed): **K% allowed +.0169 (t 1.8, [+.003, +.037], 98%) -> +.0115 (= +0.46 runs/9 per SD at full exposure)**; HR-prone -.0105 (t -1.1) -> -.0016; Air% allowed -.0104 (t -1.1) -> -.0025; left-handed +.0150 (t 0.7) -> 0; global 0. (All-PA K% +.0201 -> +.0172.) The profile findings are robust to the visitors-only design.
- **Applied to production:** hitters (file `trustmark_profile_coefs.json`; previous all-PA files kept as `*_allpa.json`) and **pitchers: `fl_ra9_trustmark_projection.build_trustmark_adjustment` now = `fl_pitcher_park` (profile model: z_hr, z_k, z_air capped at +-3 SD, lefty; global 0), the HR-factor version kept as `build_trustmark_adjustment_hr_model`; the Mississippi replacement level is no longer shifted (6.521 -> 6.963); the Mississippi P(WAR>0) SD now adds the shift's own bootstrap uncertainty (mean .35 runs/9) instead of the Air%-error term.** Result for the 2027 pitcher table (732): Mississippi RA9 shift (half games) mean -.456 -> -.001 (IP-weighted -.435 -> +.001), SD .13 -> .22, range -.60 to +.73 (K-heavy pitchers worse at Trustmark, contact pitchers slightly better); P(WAR>0 MIS) mean .630 -> about .626. Hitters: Mississippi shift mean -.0004 (SD .0107, range -.037 to +.035), park fit SD 2.6 runs. ROS snapshots rebuilt; dashboard text (hitters and pitchers) rewritten and the pitcher page's "Air% (Trustmark fit)" metric replaced by "Trustmark fit (runs/9, half games)" with its uncertainty. All hitter and pitcher views run without errors.
- **Caveats:** (1) visitor-only samples are half the size (SEs .009-.012 on the global effect), so the global effect is "neutral within +-.015 wOBA", not "exactly zero"; (2) the 2026 out-of-sample pitcher test (11 pitchers) favoured the old translation and the 2025 and 2026 single-season estimates have opposite signs, so the pooled estimate carries the decision; (3) the profile terms were estimated on a mixture of visitors facing a Trustmark-tailored home side; (4) the Mississippi roster-fit gain (+.019 wOBA) is not in the projections: a Mississippi hitter is projected with the park effect only, not with any extra "built for Trustmark" premium.

### PARK: why do air-ball hitters gain at Trustmark, and are Air% and Power confounded? -- `fl_park_airball_mechanism.py`, `fl_park_air_power_confound.py` (2026-10-08); read-only, no change
- **Mechanism (visiting hitters, per +1 SD of Air%, holding power fixed; 3,613 visitor PA at Trustmark, 131,688 PA in the fixed-effects model):** wOBA **+.0331 (SE .0095, t 3.5)**. Per PA: singles **+.0156 (t 2.2)**, 2B+3B **+.0098 (t 2.3)**, HR +.0020 (t 1.1), K% -.0066 (t -1.0), BB+HBP +.0017, balls in play +.0029 (ns). Contribution to the wOBA effect (coefficient x FL linear weight): singles +.0147, extra-base hits +.0125, HR +.0037, BB +.0014: **about 83% comes from non-HR hits (BABIP-type), HR from air-ball hitters are not suppressed more**. Per ball in play (non-HR): hit rate **+.0345 (t 3.0)** (league .328); **on air balls (>= 10 deg) +.0534 (t 3.4)** (league .351) vs on ground balls +.0215 (t 1.4); the share of balls that are air balls is unchanged (-.0065, t -0.6); extra-base share of hits +.0137 (t 0.8). **Yes: the air-ball gain is BABIP, concentrated on hits on air balls (non-HR balls in play fall in more often in Trustmark's outfield, converting would-be outs / HR trade-offs into singles and doubles).**
- **Confounding of Air% and Power:** the two profiles correlate +.42 across PA (+.42 among visitor PA at Trustmark, +.36 across hitters); including both inflates each variance only 1.22x (SE x 1.10), so the estimates are separable. **But power is a suppressor: Power ALONE -.0025 (t -0.3), Air% ALONE +.0259 (t 2.9), together Air% +.0331 (t 3.5) and Power -.0154 (t -1.8).** Orthogonalising: Air% residual after removing power +.0331 (same), power residual after removing Air% -.0154 (same); Air% orthogonal to power -.0016 / power orthogonal to air +.0266 columns are the cross terms (the part of Air% explained by power has no Trustmark effect of its own). 2 x 2 cells (median splits; visitor PA 932 / 674 / 639 / 1,368): air effect holding power low **+.0710 (SE .0250, t 2.8)**, holding power high +.0135 (t 0.5); power effect holding air low +.0094 (t 0.4), holding air high **-.0481 (t -1.9)**. **Air% x Power interaction term: +.0004 (SE .0082, t 0.0): the additive model is adequate** (stylised hitters, additive vs with interaction: air-ball low-power +.049 vs +.048, air-ball high-power +.018 vs +.018, ground-ball low-power -.018 vs -.017, ground-ball high-power -.049 vs -.049).
- **Reading:** power looks irrelevant when taken alone because air-ball hitters (who gain) and power hitters (who lose HR to the park) overlap; holding Air% fixed the power penalty appears (power hitters with a given air-ball rate lose about .015 per SD), and holding power fixed the air-ball bonus appears (+.033 per SD). Production keeps both terms (shrunk: Air% +.030, power -.011).

### Dashboard: leaderboard layouts (user 2026-10-08)
- **Hitters:** Hitter, Team, 2026 PA, Proj. PA, Proj. wRC+ (colour), Proj. MIS wOBA (colour), MIS wOBA Effect (colour, scale centred on 0), Proj. K% (colour, reversed), Proj. BB% (colour), Proj. WAR/400 (colour), P(WAR>0) (colour); WAR/400 and P(WAR>0) park-neutral.
- **Pitchers:** Pitcher, Team, 2026 IP, Proj. IP, Proj. MIS RA9, Proj. RA9-, MIS RA9 Effect, Proj. K%, Proj. BB%, Proj. K-BB%, Proj. WAR, P(WAR>0), all but the first four coloured; sorted by Proj. WAR; the PA floor slider became an innings floor (default 8 IP). New data columns in `fl_ra9_trustmark_projection.py`: `ip_2026` (actual 2026 innings; NaN with no 2026 line or a shared name), `ra9_minus` = 100 x park-neutral RA9 / league 2026 RA9 (6.25, IP-weighted; lower = better; interpretation of "RA9-" as an ERA-/wRC+-style index: tell me if a plain park-neutral RA9 was meant), `proj_war` = (replacement RA9 6.963 - park-neutral RA9) x projected IP / 9 / 10.706 (park-neutral; WAR (MIS) is on the Targets page as runs vs replacement); Proj. IP = ip_unc (19.31 + .6278 x 2026 IP, floor 18).

### MODEL AUDIT (user 2026-10-09: "dive deeper into potential issues") -- new tests: `fl_pitcher_ip_quality_test.py` + ad hoc movers / sensitivity / data checks; no model change
- **Pitcher top-end WAR (2025 -> 2026 backtest, 312 pitchers with >= 10 IP): top 15 by projected WAR projected 1.63 vs realised 2.22; top quintile .88 vs 1.05 (projected IP 59 vs real 70); slope of realised on projected WAR 1.24; max projected 2.35 vs realised 4.35.** Cause tested: do better pitchers get more innings (the hitter PA finding)? Returning pitchers, 543 transitions, leave-one-transition-out, equal weights: IP_next = 19.31 + .628 IP_t RMSE 29.66 / MAE 23.67 (R^2 .322); per team game 29.55 / 23.59 (-.108, P .89); + shrunk K-BB quality 29.51 / 23.60 (-.148 [-.349, +.061], P .86; R^2 .329); + prior-prior IP 29.60 / 23.68; + starter share 29.58 / 23.65. **Bias of the current model by transition for pitchers with >= 80 IP: -3.3 / +1.1 / -1.2 / -0.8 IP (n = 20-30 each): no top-end IP bias in any transition; quality adds ~nothing (corr .16 with next-year IP per game).** The 2026 top-end miss is mostly RA9 luck among the top 15 (realised RA9 4.4 vs projected 5.0 for the same arms) and a modest 2026 workload rise (top-15 realised IP ~90 vs projected ~83), not a modelling gap. No change.
- **Park carry-over for hitters who changed teams (2025 -> 2026; 32 movers of 125 hitters): next-year error vs the change in the published team park factor = -.00005 wOBA per PF point (bootstrap 90% [-.0030, +.0037]; expected +.0015 if the published adjustment were right).** Inconclusive (no power); consistent with the wRC+ target test (team park adjustment adds ~0.1 wRC+ point of accuracy).
- **Parameter sensitivity (hitters): all n0 x 0.7 or x 1.4 change projected wOBA by .0007-.0009 on average (regulars .0007, max .0037), WAR/600 by .04 (max .18), rank correlation .998-.999.** The projections are not sensitive to the regression constants within +-40%.
- **Data check (needs the user):** the tracked 2026 regular-season data has 939 distinct games (18 teams: about 104.3 games per team; every team 102-107, median 104) vs the stated 102-game schedule (918 games expected); 2025 has 848 games (94.2 per team vs 96). The workload scaling uses 96 / 102 as stated; if the extra 2026 games are doubleheader / make-up / mis-tagged games this is a ~2% workload and PA-scaling question (and the tagged playoffs are separate: 23 games in 2026). The dated PA table (72,281 PA in 2026) is within 1% of the panel's reg-season PA (72,969).
- **Issues carried forward (not new, not fixed):** (1) position / fielding layers validated on one season (infield 2025 data missing); (2) no aging term (age not in the FL tables; hitters or pitchers); (3) thin-history / gap hitters (n = 19 pairs behind the gap multipliers; survivorship +.005 wOBA unapplied); (4) pitch-level terms (Whiff% weakest; EV90, Swing%, wh_sva, command score each rest on 2025-26); (5) Trustmark: neutral for visitors within +-.015 wOBA, Mississippi roster-fit +.019 wOBA not in projections; the 2026 out-of-sample pitcher test (11 pitchers) disagreed; (6) the league environment is anchored at 2026 (league wOBA .375 vs .359 in 2025, .346 in 2024; replacement level 6.963 RA9 / .0362 runs per PA, runs per win from 2026): relative measures (wRC+, RA9-, WAR vs league) are insulated but replacement-based WAR and P(WAR>0) are not; (7) pitcher P(WAR>0) is ~7 points optimistic for arms projected under 22 IP (level miss in 2026); (8) coverage: no other-league (college / Pioneer / MiLB) hitter bridge (pitchers have one), and rookies with no FL history have no hitter projection; (9) hitter ROS WAR carries preseason fielding / position rates; (10) all backtests have 4 transitions (2 with Trackman), so intervals are wide.

## 2026-10-09 -- Pitcher P(WAR>0) optimism for arms projected < 22 IP: selection / informative playing time (user: "some survivorship here, maybe?")
Script: `fl_pwar_selection_test.py` (backtest 2025 -> 2026, 197 low-projected arms; `fl_pwar_calibration.RETURN_ALL` flag added to expose the unfiltered frame).
- The calibration sample is floored at 10 realised 2026 IP (`MIN_IP` in `fl_ra9_uncertainty_model.py`), so the floor can only be RAISED in this data; arms with < 10 IP are not in the data build.
- Gap (predicted - observed P) for arms projected < 22 IP: all (>= 10 IP) +.071; >= 20 realised IP -.012; >= 30 IP -.057.  By realised IP: 10-20 IP (n 55) predicted .632 vs observed .345 (gap +.287, 90% [+.18, +.40]); 20-30 IP (n 40) +.101; 30+ IP (n 102) -.057 (n.s.).  Innings-weighted gap is -.014 (calibrated).
- Mechanism = INFORMATIVE PLAYING TIME, not classical survivorship: corr(RA9 miss vs prediction, log realised IP) = -0.40 (< 22 projected IP) and -0.39 (>= 22): arms who pitch worse than predicted pitch fewer innings.  The arms who end with 10-20 IP are mostly arms that were pulled after bad results.  Classic survivorship (dropping arms with < 10 IP) would, if anything, flatter the observed rate, so the true per-arm optimism is not smaller than measured.
- Consequence: per-arm P(WAR>0) is optimistic for arms who end up with little usage, calibrated innings-weighted (the dollar/roster-relevant view).  No model change made (the user earlier chose to keep the P(WAR>0) uncertainty as-is).  Options left open: a dashboard note, or a short-arm discount; revisit with 2027 data.

## 2026-10-09 -- HITTER OTHER-LEAGUE BRIDGE (MiLB -> FL), rookies / thin history (user: "need to make a bridge for hitters"; "thin-history also means rookies")
Data: `pull_milb_hitting.py` -> output/milb_hitting_2018_2026.parquet (MLB Stats API, AAA/AA/A+/A/ROK, 43,893 hitter-seasons, same endpoint/playerPool=ALL as the pitching pull).  NOT available for hitters: college (FanGraphs hitting) and Pioneer hitting -- only pitching was ever pulled; flagged to the user (see below).
Pairs: 687 FL debut hitters 2023-26 (first panel season, >= 20 PA); matched by normalised name to MiLB seasons strictly before the debut year (names with > 1 MLBAM id dropped) -> 200 pairs (29%).  Caveat: the panel starts in 2022, so someone with pre-2022 FL history can look like a debut; and the pairs are SELECTED (hitters who actually signed in FL with >= 20 PA).
Design (`fl_hitter_bridge_test.py`, `fl_hitter_bridge.py`): per stat (wOBA proxy, K%, BB%, HR/PA) FL deviation from that year's FL league mean = a + b*u; u = PA x decay^lag weighted deviation from the same LEVEL-SEASON MiLB mean, shrunk by n0 PA, summed over level groups; pooled slope; n0/decay by (nested) leave-one-year-out.  Relative measures throughout (league environment anchored at FL 2026).
Results (leave-one-debut-class-out, nested tuning, inverse-variance weights from the debut-class-mean residuals, player-cluster bootstrap, vs debut-class mean):
  K%     RMSE .07131 -> .06090 (-.0104, 90% [-.0148, -.0067], P 1.00); MAE .05464 -> .04605 (P 1.00)
  BB%    RMSE .03770 -> .03417 (-.0035, [-.0055, -.0015], P .998);       MAE .02867 -> .02452 (P 1.00)
  HR/PA  RMSE .01841 -> .01692 (-.0015, [-.0022, -.0008], P 1.00);        MAE .01477 -> .01312 (P 1.00)
  wOBA direct RMSE .05941 -> .05770 (-.0017, [-.0042, +.0006], P .88); MAE .04559 -> .04275 (P .99).  Component-built wOBA RMSE .05641 (-.0030, P 1.00), MAE .04208; 50/50 blend RMSE .05672, MAE .04172 (blend vs component: P .27 / .81, indistinguishable).
  Level-group slopes (3 groups) did not beat the pooled slope for the components (M3 ~ M2); pooled used everywhere.  Debut-class mean is ~ league mean (intercepts within noise).
  OOF calibration inside the training range is good (top-20% wOBA predicted +.0174 vs actual +.0158; slopes .74 wOBA, 1.00 K%, .77 BB%, .86 HR).
Decisions: (1) wOBA = 0.5 direct + 0.5 component-built, same as the FL production model (blend ~ component on RMSE, better MAE).  (2) Feature u clipped to the training range; prospects beyond the 5th-95th percentile of training u flagged `beyond_validated_range` (22% of the table) -- the training population is hitters who signed in FL, so elite affiliated prospects are extrapolation.  (3) Output: output/fl_hitter_bridge_coefs.json, output/fl_hitter_milb_prospects.parquet (4,198 MiLB hitters with PA, 81 already in the FL panel).  wRC+ and WAR/400 use the production constants (no fielding/position adjustment: role unknown).  No Mississippi translation (no Trackman profile for these hitters; global effect ~ 0).
THIN-HISTORY PRIOR TEST (`fl_hitter_bridge_prior_test.py`; 44 second-season hitters, 2023-25 classes): shrinking the debut season toward the bridge prediction instead of the league mean: K% RMSE .04818 -> .04363 (P .99), HR/PA .01318 -> .01237 (P .96), BB% neutral (P .55), wOBA .04144 -> .04205 (P .26; direct-only P .24, component-only P .17).  NOT ADOPTED in the production projection: no gain on the target that matters (wOBA) and n = 44 is underpowered; revisit when the 2026 class has a second season (2027 data).
Dashboard: new Hitters view "Rookie / MiLB Lookup" (`hitters_view.render_hitter_lookup`, route in `app.py`); data fl_hitter_milb_prospects / fl_hitter_milb_history / fl_hitter_bridge_coefs copied to projections-dashboard/data (`fl_hitter_bridge.py --publish`).  AppTest clean.  Testing rule saved to memory: "thin history" test groups also include no-history hitters (rookies); the rookie row of the evidence is the 200-pair bridge test above.

## 2026-10-09 -- Trustmark outfield OAA, fly-ball hits and the Air% profile (user: "our OFs had a huge OAA at home and so did opposing outfielders at Trustmark"; main effects = sanity check, profiles matter)
Scripts: `fl_park_oaa_outfield.py`, `fl_park_airball_by_fielder.py`, `fl_park_airball_contact.py`, `fl_park_ev_by_angle.py`, `fl_park_ev_tracking_check.py` (data: 2025_2026_pitches.parquet; OAA exists only for outfield-fielded balls, 11,858 of 77,904 non-HR balls in play).
- OAA: Trustmark has the highest mean OAA per outfield ball (+.045; other parks -.004; diff +.048, bootstrap SE .011).  Home fielders +.046, visiting fielders +.043 -> a PARK property, not roster quality.
- It is a real hit-rate effect: cross-fitted traits-only expected-hit model (EV, angle, spray, distance, hang time, position, side; Brier .0945), outfield balls at Trustmark are hit .0418 less than expected (SE .011, home fielders -.044, visiting -.040); across parks OAA and hit-minus-expected correlate -.87.  Infield / other balls: -.0007 (nothing).  About 15% of balls in play are outfield-fielded -> roughly -0.6 points of overall BABIP.  All three OF positions show it (LF -.044, CF -.035, RF -.037).
- Air% x Trustmark is NOT a fielding / hit-given-contact effect: with batted-ball traits fixed the residual hit rate has no Air% interaction (outfield balls -.009, infield/other +.007, all +.003; |t| < 1).  It lives in the CONTACT: per +1 SD Air%, visiting hitters at Trustmark: exit speed +1.14 mph (t 3.3), xwOBA on contact +.027 (t 3.5), wOBA on contact +.041 (t 3.1); launch angle and air-ball share unchanged (t .5 / -.3).  Power x Trustmark stays negative (wOBA on contact -.026, t -2.3).
- Tracking artefact check: given recorded EV + angle, airborne balls at Trustmark travel +4.0 ft FARTHER than predicted (highest of 18 parks; ~84 F) and hang +.023 s; hitter-adjusted EV is ordinary (+.26 mph, JL Bruno higher).  An over-read EV would show shorter carry, so there is no sign of an EV artefact.  (An EV-by-angle-bin cut was uninformative: its main effects are tangled with the park fixed effect.)
- Consequences: no change to production (profile terms are already shrunk; visitors-only).  The global park effect is the sanity-check main effect: ours raw +.0007 (FE), visitors average +.0027 (SE .0086) -> shrunk ~0, published team wOBA PF for Trustmark 99.3-99.7 (about -.002 wOBA): consistent within noise.  Possible follow-up (not run): does a pitcher's outfield-fielded-ball share (fly-ball tendency) interact with Trustmark's outfield suppression?  The pitcher air term was ~0 in the existing profile model.
- Game counts (user question 2): tracked 2025 = 847 game ids (2025-05-08 to 2025-08-31), 2026 = 939 (2026-05-07 to 2026-09-06).  Counting only games with >= 40 tracked PA, the per-team maximum is exactly 96 (2025) and 102 (2026) = the schedule; the extras are stub ids (0-39 PA: 19 in 2025, 49 in 2026: rainouts / suspended / untracked) and 112 same-day extra ids (doubleheaders).  Coverage: 2026 teams have 87-102 full tracked games (mean 98.9).  GAMES = 96 / 102 constants are right; nothing to change.
- Data plan: user will provide 2026 Pioneer hitters; college hitters from FanGraphs; pre-2026 Pioneer hitters from StatsCrew (extends the hitter bridge to college / Pioneer; not started).
- Dashboard reopened on port 8503 with the Rookie / MiLB Lookup (user's 8501 untouched).

## 2026-10-09 -- PLAYING THE PARK: pitcher fly-ball follow-up, ball-level park effects, outfield alignment, Mud Monsters' home edge (user: "run the follow-up, I'm generally curious about how to best play at our park")
Scripts: `fl_park_play_the_park.py`, `fl_park_ms_approach.py`, `fl_park_ms_kbb.py`, `fl_park_ms_discipline.py` (logs in output/*_log.txt).  EXPLORATORY: two seasons, one park, many cuts (no multiplicity correction); nothing here enters the projections.
Ball level (80,349 tracked balls in play incl. HR; park-blind cross-fitted expectation from EV, angle, field direction, side; 4,808 at Trustmark): overall Trustmark effect per ball in play -.031 wOBA (HR -.017/ball, non-HR hits +.002).  By type x spray (relative to the hitter), wOBA/ball vs a same-quality ball at the average park: pulled fly ball -.184 (HR -.124, t -5.6), oppo fly ball -.170 (t -4.2), middle fly ball -.070 (t -3.9), oppo line drive -.107 (t -3.3), middle line drive -.050, pull line drive -.045; ground balls ~0 to +.02 (n.s.), pop-ups ~0 (oppo -.030).  Airborne balls by field sector: biggest losses in the RF gap (RHH -.117, LHH -.103; LHH HR -.089/ball) and LF gap (-.055 / -.075); CF -.058 / -.033; the lines ~0 (n 36-103, imprecise).
Outfield alignment: at Trustmark both defences play deeper (CF 342 / 338 ft vs 312; LF 273-279 vs 256; RF 278-282 vs 256) and wider (LF/RF ~12-15 ft further from centre).  Inside Trustmark the hit residual on outfield balls moves +.008 per +10 ft depth (SE .010; shallow balls < 250 ft +.029 per 10 ft, n 189; deep balls >= 300 ft -.009): the usual trade (deeper = fewer deep hits, more shallow ones) but not significant -- we cannot say the current alignment is right or wrong.
PITCHER FOLLOW-UP (fly-ball pitchers and the outfield suppression): a pitcher's predicted Trustmark shift from his batted-ball mix (type x spray) has mean -.030 per ball in play, SD .0047 across 166 pitchers (fly-ball-type tercile -.034 vs ground-ball tercile -.026: .008 per ball ~ .005 per batter faced).  That is ~3.6x smaller than the K%-allowed profile (+.0115/SD) already in the model -> batted-ball mix is second order; most of the suppression is park-wide, not pitcher-specific.  The validation regression (62 visiting pitchers, >= 8 balls at Trustmark) has no power: slope -4.7, SE 2.9 (expected signal SD .005 vs per-pitcher noise ~.05).  NOT added to the projections.
MUD MONSTERS' HOME EDGE (difference-in-differences vs other teams at Trustmark): balls in play are NOT where it comes from -- MS hitters' ball residual home-road -.028 vs others' -.031 (DiD +.003, t 0.2).  It is in PLATE DISCIPLINE: MS hitters K% 14.1% home vs 19.6% road; DiD -4.7 pts (t -4.0), BB% +1.9 (t 1.6), wOBA DiD +.040 (t 2.6, raw PA design).  Pitch level with hitter + pitcher + year + park fixed effects: K per PA -3.9 pts (t -3.0), BB +3.0 pts (t 2.7), whiff per swing -2.3 pts (t -2.0), swing -1.4 (t -1.6), chase -0.9 (n.s.), zone swing -0.7, zone contact +1.6 (n.s.); the pitchers' side of the same pitches is unchanged (zone share -1.4, speed -0.03 mph, fastball share -1.3; all n.s.).  Other teams' hitters get no such K% change at Trustmark (-1.0, generic home advantage).  (The 'others' pitchers at Trustmark K% -5.7' figure is those pitchers facing MS hitters, not the park.)  MS PITCHERS: ground-ball share -5.2 pts vs baseline (t -2.4), pop-up share +3.4 (t 2.5), ground-ball residual -.047 (t -2.2) -- exploratory.
Tension left open: the ball-level result says a given airborne ball is worth LESS at Trustmark, while the visitors-only profile model says high-Air% hitters gain relative to the Trustmark average (+.033/SD) via harder contact (+1.1 mph, xwOBA on contact +.027).  Both can hold (per-ball penalty vs contact-quality change), but the cause of the harder contact is unexplained; two seasons only.

## 2026-10-09 -- PITCHER PROFILE SCREEN at Trustmark, controlling for K% and BB% allowed (user: "when you control for K% and BB% is there a pitcher profile that works better?")
Scripts: `fl_park_pitcher_profile_screen.py`, `fl_park_pitcher_profile_replicate.py` (outputs/pitcher_profile_screen*.csv, logs).  Design = the production visitors-only pitcher model (year + hitter + pitcher + pitcher x home + park FE; interaction on VISITING pitchers at Trustmark; 131,767 PA, 810 pitchers, 3,389 visiting-pitcher PA at Trustmark; game-cluster SEs).  15 candidate profiles measured on non-Trustmark data, shrunk, z-scored.
- Controls: Trustmark x K% allowed +.0162 (t 2.1; high-K pitchers do relatively WORSE), x BB% allowed -.0031 (t -0.4).
- One at a time with K% and BB% controlled (wOBA per +1 SD, negative = better for the pitcher): fastball induced vertical break -.0202 (t -2.4), fly-ball share -.0186 (t -2.3), HR/PA allowed -.0169 (t -2.2), exit speed allowed -.0170 (t -2.0), ground-ball share +.0138 (t +1.9), pop-up share -.0141 (t -1.8), release height -.0131; velocity, spin, extension, repertoire shares, handedness, line-drive share ~0.  3 of 15 have |t| > 2 (0.7 expected by chance); the contact-type ones (fly-ball, HR, exit speed, ground-ball) are mutually correlated, so this is one finding, not four: pitchers who allow air balls / hard contact do relatively BETTER at Trustmark, consistent with the ball-level result (airborne balls are worth less here).  Fastball IVB is the one stuff signal (more ride = better), correlated .15 with K%.
- Joint model with all 17 terms: nothing individually stable (exit speed allowed t -1.8, bootstrap); collinear (type shares sum to 1; cluster SEs blow up) -- not informative.
- Replication (composite fixed from the screen's signs = mean of z fly-ball share, HR/PA, exit speed allowed, -z ground-ball share; so POST HOC): pooled -.0229 per SD (bootstrap t -2.8; -0.91 runs/9 per SD at full exposure; with K% +.0140 (t 2.0), BB% -.008 (t -0.9)); adding HR/PA and left: -.0255 (t -2.0), HR/PA +.005, left +.005 (nothing).  By year (same sign both years, SE large): composite 2025 -.0096 (SE .012), 2026 -.0350 (SE .009), difference t 1.7; fly-ball -.005 / -.027, HR/PA -.009 / -.027, IVB -.010 / -.032, exit speed -.017 / -.019; ground-ball share does NOT replicate (-.005 then +.030, t -2.4 between years).
- Caveats: (1) every visiting pitcher at Trustmark faces Mississippi's hitters, whose K% falls ~4-5 points at home (fl_park_ms_discipline), so the K% x Trustmark term (+.014/SD) is partly roster fit (high-K arms lose proportionally more strikeouts), roughly a quarter by a back-of-envelope scaling; (2) composite is post hoc after a 15-way screen (Bonferroni on 15 ~ p .075); (3) 2025 alone is not significant.
- DECISION: NOT added to production.  Production keeps K%-allowed (+.0115 shrunk) and nothing else.  Candidate for pre-registration: test the contact-prone composite on 2027 data (true out-of-sample) before adopting; by the production rule (positive-part James-Stein on bootstrap t) its shrink would be ~.87.

## 2026-10-09 -- FLY-BALL SHARE vs GROUND-BALL SHARE of pitchers at Trustmark, K% and BB% controlled (user: "Fly-ball share and ground-ball share are the ones I'm most interested in")
Script: `fl_park_pitcher_gbfb_deep.py` (log output/pitcher_gbfb_deep_log.txt).  Same visitors-only design as the screen; shares from non-Trustmark balls (GB < 10 deg, LD 10-25, FB 25-40, PU > 40), shrunk n0 89.
- Scale: league mean GB .469, FB .162; +1 SD (shrunk, PA-weighted pool) = 4.3 pts of balls in play for GB (6.2 raw across pitchers with >= 150 balls), 2.1 pts for FB (2.9 raw); FB>=25 incl. pop-ups mean .311, SD 4.0.
- ONE DIMENSION: GB alone +.0138 (t 1.9), FB alone -.0186 (t -2.3), FB>=25 -.0181 (t -2.4), Air>=10 (the production term) -.0138 (t -1.9) = the exact mirror of GB (Air = 1 - GB).  Together GB +.0033 (t .3), FB -.0164 (t -1.4); with LD in the model nothing separates (GB +.012, FB -.008, LD +.011).  Ground-ball and fly-ball share are not separately identified at this sample size; FB keeps the larger coefficient.
- Shape: FB share terciles (K, BB controlled): middle vs lowest -.0226, highest vs lowest -.0506 (t -2.6) -- steeper than linear.  GB share: middle +.020, highest +.0275 (t 1.6), noisy.
- By year (GB + FB model): FB -.0178 (2025) / -.0140 (2026) -> REPLICATES; GB -.0168 (2025) / +.0210 (2026), difference t -1.8 -> does NOT replicate.
- Outcomes (per PA per +1 SD): FB share -> HR/PA -.0076 (t -3.1; league HR/PA .0238), singles+doubles+triples 0.0000, K -.003, BB -.002; GB share -> K per PA -.0170 (t -2.0), HR -.0024, hits +.0067 (n.s.).  The FB effect is HR suppression, the sensible mechanism.
- Magnitude check FAILS: the ball-level park effects (fl_park_play_the_park) imply about -.001 to -.002 wOBA per batter faced per +1 SD of FB share (and ~ +.002 for GB); the regression is -.016 to -.019, i.e. ~10x the structural size; the HR effect rests on 28 home runs allowed by visiting pitchers at Trustmark (3,389 PA, HR/PA .0083 vs league .0238).  Selection from a 15-way screen (winner's curse) is the likely reason.  With the structural size as an informative prior (SD ~ .004) the combined estimate would be about -.003 (FB) and +.005 (GB) wOBA per SD (~ -0.1 / +0.2 runs/9), not -0.7 / +0.5.
- DECISION: not adopted.  FB share is first in line for the pre-registered 2027 test (HR/PA allowed at Trustmark by visiting pitchers' FB share, sign negative, size ~ structural).  GB share and Air% give nothing separate from it.

## 2026-10-09 -- IS FLY-BALL SHARE (25-40) BETTER THAN FB+LD / Air%?  HOW STICKY IS IT?  CAN TRAITS PREDICT IT?  (user questions)
Scripts: `fl_pitcher_fbshare_sticky.py` (Parts 1-3), `fl_park_pitcher_fb_proxy.py` (logs output/pitcher_fbshare_sticky_log.txt, pitcher_fb_proxy_log.txt).
PART 1 definitions at Trustmark (K%, BB% controlled; wOBA per +1 SD): FB 25-40 -.0186 (t -2.3); FB+LD 10-40 -.0078 (t -1.0); FB incl. pop-ups >= 25 -.0181 (t -2.4); Air% >= 10 -.0138 (t -1.9) = mirror of GB +.0138; LD alone +.0077 (t 1.1; reliability ~0, adds only noise); pop-up > 40 -.0141 (t -1.8).  Out-of-season test (fit on one season, apply to the other's Trustmark PA): only FB 25-40 removes any squared error and only in one direction (2025 -> 2026 +0.08%, P .998; 2026 -> 2025 -0.12%, P .30); FB+LD, FB>=25, Air, GB, pop-up all make it WORSE out of season (-0.03% to -0.57%).  FB alone >= FB+LD, but nothing predicts the other season in a practical way.
PART 2 stickiness: split-half reliability (full-length) / implied n0 / 2025->2026 r (85 pitchers >= 60 balls both seasons): FB 25-40 .15 / 773 balls / .23 (least sticky); LD .02 / ~6000 / -.03 (pure noise); FB+LD .22 / 483 / .23; FB>=25 .59 / 98 / .47; Air% .58 / 101 / .55; GB .55 / 115 / .55; pop-up .51 / 135 / .54; K% allowed .76 / 67 PA / .64; BB% allowed .69 / 92 PA / .68.  The persistent batted-ball trait is ground-ball vs air (and pop-ups), not fly ball vs line drive: the 25-40 degree band is the transition band and mostly measurement noise at ~140 balls per season.
PART 3 predictability: same-season traits (fastball vertical approach angle, velocity, IVB, release height, extension, spin, pitch height, mix, handedness) -> FB share, 5-fold by pitcher, 644 pitcher-seasons: weighted RMSE .0385 -> .0353 (P 1.00), MAE .0298 -> .0265, weighted R^2 .16; strongest standardised effects (FB-share points per +1 SD): vertical approach angle +1.7, fastball velocity -1.7, release height +1.1, fastball height -0.8.  Forward 2025 -> 2026 (111 pitchers): league mean RMSE .0369 / MAE .0289; last year's FB share shrunk (n0 500) .0361 / .0283 (P .76); last year's traits .0369 / .0275 (MAE P .79); both .0374 / .0277 -- nothing clearly better (corr 2025 vs 2026 share .18; trait prediction vs 2026 .24).
FOLLOW-UP at Trustmark: proper shrinkage does not change it (FB n0 770: -.0199, t -2.5; GB n0 115 +.0139; Air% n0 100 -.0138; pop-up n0 135 -.0145).  BUT BY SEASON the single-term estimates are 2025 / 2026: FB -.008 (SE .013) / -.027 (SE .010); GB -.005 / +.029; Air% +.005 / -.030; pop-up +.005 / -.033 -> it is a 2026 pattern, with 2025 ~ 0 (this corrects the earlier GB+FB joint-model by-year read of FB -.018 / -.014, which came from an unstable joint model).  A stable trait-based proxy of fly-ball tendency (cross-fitted ridge prediction; corr .35 with the share, -.55 with GB) shows -.0067 (t -0.9; 2025 +.009, 2026 -.022): no support.
Conclusion: fly-ball-share and its relatives at Trustmark are NOT adopted; the signal is a one-season (2026) pattern in a noisy, low-reliability measure with a magnitude 10x the ball-level mechanism.  Pre-registration for 2027 stays (FB share and the GB/Air mirror, HR/PA outcome), expectations lowered.

## 2026-10-09 -- DOES FLY BALL + POP-UP SHARE (>= 25 deg) HELP THE TRUSTMARK PITCHER MODEL?  (user: "Does it help the model?")
Script: `fl_park_pitcher_fb25_model_test.py` (log output/pitcher_fb25_model_test_log.txt).  Unit = visiting pitcher-season at Trustmark (205; 3,389 PA); target = mean FE-residualised wOBA allowed over his Trustmark PA; prediction = the fitted interaction part; coefficients shrunk by the production rule max(0, 1 - 1/t^2) with the training-sample t; weights 1/(a + b/n) from model B's residuals, frozen (a ~ 0: Trustmark samples are so small that variance is pure sampling noise, weights ~ n); pitcher-cluster bootstrap.  Weighted RMSE / MAE:
5-fold by pitcher (x5):   A none .10712/.07949 | B production (HR-prone, K%, Air%, left) .10726/.07961 | C B + FB+PU .10730/.07917 | D K%-only .10683/.07940 | E K% + FB+PU .10623/.07809 | F K% + BB% + FB+PU .10624/.07805.
  C vs B: dRMSE +.00004 (P .52), dMAE -.00043 (P .79).  E vs B: dRMSE -.00104 [-.00205, .00000] (P .95), dMAE -.00151 [-.00259, -.00037] (P .99).  E vs D (K%-only, what production effectively keeps): dRMSE -.00061 (P .78), dMAE -.00130 (P .93).
Cross-season (fit one season, predict the other): A .10712/.07949 | B .10958/.08273 | C .11013/.08381 | D .10712/.07998 | E .10909/.08150 | F .10910/.08150.
  C vs B: dRMSE +.00056 (P .32), dMAE +.00108 (P .19).  E vs B: -.00049 (P .78) / -.00123 (P .96).  E vs D: dRMSE +.00197 (P(better) .06), dMAE +.00152 (P .16).  B vs D: +.00246 (P(better) .016) -> the extra production terms (HR-prone, Air%, left) HURT out of season.
Reading: FB+PU share does not help the model.  It helps only in the pitcher-fold scheme, which mixes seasons (it can learn a 2026-specific pattern from other pitchers in the same season) and not at all across seasons, where it is worse than K%-only.  Side finding: even the K% term does not beat 'no profile' out of sample (D vs A: RMSE .10683 vs .10712 by pitcher fold, identical across seasons, MAE slightly worse across seasons) and the other three production terms make it worse; the profile layer's justification is the in-sample bootstrap (t ~ 2), and the test has almost no power (per pitcher-season ~16 Trustmark PA, wOBA SD ~.11 against a K%-term signal SD ~.012).  NOT adopted.  Open for the user: keep the K%-allowed term (current), or fall back to global-only for pitchers.

## 2026-10-09 -- PREDICTING FLY BALL + POP-UP SHARE (>= 25 deg) FROM PITCHER TRAITS (user question)
Script: `fl_pitcher_fbpu_traits.py` (log output/pitcher_fbpu_traits_log.txt).  Same design as the FB (25-40) trait test: 644 pitcher-seasons (533 pitchers, >= 30 balls, >= 150 pitches tracked), ridge on 19 traits (fastball VAA, velocity, IVB, horizontal break, spin, release height, extension, pitch height, fastball height, upper-zone share, 8 pitch-mix shares, lefty), pitcher-grouped 5-fold, inverse-variance weights from the league-mean residuals (frozen), pitcher-cluster bootstrap.
FB+PU same season: weighted RMSE .0707 -> .0554, MAE .0566 -> .0423 (dRMSE -.0153 [-.0181, -.0125], P 1.00; dMAE -.0143, P 1.00), weighted R^2 .39 (pop-up .32, Air%/GB .36, FB 25-40 only .16).  Drivers (share points per +1 SD of trait): fastball vertical approach angle +3.7 (flatter = more), fastball velocity -3.6, release height +2.4, fastball IVB +1.5, fastball height -1.5, pitch height +1.3.  Approach angle + pitch height alone: R^2 .11.
Forward 2025 -> 2026 (111 pitchers): league mean RMSE .0678 / MAE .0546 | 2025 share shrunk (n0 92) .0605 / .0467 (dRMSE -.0073, P .99; dMAE -.0079, P 1.00) | 2025 traits (ridge) .0631 / .0485 (dRMSE -.0048, P .93; dMAE -.0061, P .97) | share + traits .0614 / .0475 (no better than the share alone).  corr(2025 share, 2026 share) .39; corr(traits prediction, 2026 share) .40.
Reading: FB+PU (like GB/Air) is largely arsenal / delivery geometry; traits predict it as well as last year's share does, but adding them to last year's share does not help at n = 111.  Possible use (not built): a trait-based prior for batted-ball shares of rookies / thin-history pitchers.

## 2026-10-09 -- DO Air% OR FB+PU MATTER AT TRUSTMARK AFTER K% AND BB%?  (summary answer; `fl_park_pitcher_fbpu_byyear.py`)
Pooled (wOBA per +1 SD, visiting pitchers, K% and BB% controlled): Air% -.0138 (SE .0070, t -2.0, -0.55 runs/9); FB+PU -.0181 (SE .0074, t -2.5, -0.72 runs/9).  By season: Air% +.0049 (SE .011) in 2025 vs -.0295 (SE .010) in 2026, difference t 2.3; FB+PU +.0026 (SE .0105) vs -.0361 (SE .0112), difference t 2.5.  Out-of-season (fit one season, predict the other's Trustmark PA): both make squared error worse (Air -0.12% / -0.45%; FB+PU -0.06% / -0.53%).  Park-model test: adding FB+PU to production gives no gain (RMSE +.00004 by pitcher fold, +.00056 across seasons).  Ball-level mechanism implies ~10x smaller effect.  Verdict: not established; a 2026-only pattern.

## 2026-10-09 -- THE TRUSTMARK HOME-RUN SUPPRESSION EXPLORED; TRIPLES OFFSET; DOES THE AIR PROFILE COVER BOTH?  (user: "hugely suppressed HR park factor ... explore this"; "the triples factor is ridiculously high ... does that offset anything?"; "whether the Air-focused profile covers both factors")
Scripts: `fl_park_hr_explore.py`, `fl_park_triples_offset.py` (logs output/hr_explore_log.txt, triples_offset_log.txt, triples_offset_partb*_log.txt).  80,767 tracked balls in play, 2025-26; park-blind expectations from a cross-fitted traits model (EV, angle, field direction, side) and from the tracking data's own 'HR in how many of 17 MLB parks' count (isotonic-mapped on the other 17 FL parks).
HR FACTOR: Trustmark 0.475 (SE .042; traits) / 0.402 (MLB count), lowest of 18 parks (next Ottawa .71); 69 HR vs 145 expected (traits) / 172 (MLB count) in 4,822 tracked balls -> 76 / 103 HR lost = 0.75 / 1.01 per tracked game (102 games).  Highest: Grizzlies 1.88, Thomas More 1.31, Bosse 1.24.
WHERE: by sector x batter side (HR actual / expected; lost): RHH LF line 6/8.9, LF gap 26/41.5 (.63), CF 3/11.0 (.27), RF gap 0/13.3 (0.00), RF line 1/1.0; LHH LF gap 2/7.1 (.28), CF 5/9.5 (.53), RF gap 16/40.5 (.40, 24.5 lost = the biggest bucket), RF line 9/10.6 (.85).  Foul-pole corners are about normal; the gaps and CF lose the HR.  By contact: ratio ~.4-.5 for 95-105 mph balls at 15-35 degrees, but 105+ mph at 25-35 degrees still leaves (ratio 1.0-1.16): the park kills medium-hard air balls, not barrels.  By projected distance (airborne balls): HR share 360-380 ft 7% vs 31% elsewhere, 380-400 29% vs 58%, 400+ 65% vs 91%.
EFFECTIVE FENCE (50% HR distance for EV >= 92, 15-45 degree balls): Trustmark deeper than the other parks by LF line +19.7 ft (SE 2.9), LF gap +16.9 (2.5), RF gap +39.3 (4.0), CF +23.6 (SE 43, unstable), RF line +2.4 (3.0).  Would-be HR (>= 70% of MLB parks): 42% HR at Trustmark vs 85% elsewhere; 36% become doubles/triples (11%), 21% outs (3%): -.47 wOBA per such ball.
MUD MONSTERS: HR / expected: MS hitters at home .459 (SE .069), visiting hitters at Trustmark .488 (.055), MS hitters on the road 1.22 (.12; 64 HR vs 52 expected), MS pitchers on the road 1.07, all other balls 1.03.  Mean launch angle at home 9.7 vs 10.4 for visitors, exit speed 83.0 vs 83.7, share >= 25 deg .307 vs .316: no sign of an adapted batted-ball approach; MS power plays on the road and is simply lost at home.  Weather: wind readings mostly calm / 0 mph (43% coverage), temperature coverage 32% (mean 84 F): not informative.
TRIPLES: Trustmark triple factor 1.65 (41 vs 24.9 expected), 2nd highest (Skylands 1.89); also doubles factor .86 (251 vs 291).  Triples by sector: RF gap 17 (factor 2.0), RF line 8 (2.0), CF 10 (1.23), LF gap 6 (1.56) -- the same gaps that lose the most HR (RF gap HR factor .30).  Converted to wOBA with FL weights, Trustmark minus expected over 4,822 balls: HR -74.9 x 1.863 = -139.6; triples +16.1 x 1.534 = +24.7; doubles -40.2 x 1.251 = -50.3; singles +42.1 x .945 = +39.8; net -125.3 wOBA points = -.026 wOBA per ball in play (bootstrap SE .006).  Triples offset 18% of the HR loss; all non-HR outcomes together offset 10% (doubles lost cancel most of triples + singles gained).  The park-wide wOBA effect in the fixed-effects model is ~0 because strikeouts fall and walks rise outside the ball-in-play measure (and contact is not worse for hitters who hit it hard); the full reconciliation was not re-derived here.
AIR PROFILE COVERAGE (visiting hitters, production design, per +1 SD per PA, 131,688 PA): Air% -> HR +.0020 (t 1.1), triples +.0038 (t 2.0), doubles +.0060 (t 1.7), singles +.0156 (t 2.2), K -.0066, BB +.0010; wOBA +.0331 (t 3.5), split by outcome: singles 45%, doubles 23%, triples 18%, HR 11%.  Power -> HR -.0074 (t -2.5) = -.0138 wOBA, singles +.0020, BB -.0083; wOBA -.0154 (t -1.8).  So the Air% term carries the OFFSET side (extra-base hits and singles, incl. triples), NOT the HR loss; the HR loss is carried by the power term.  Both terms are needed; an Air-only profile misses the HR side.  Speed proxy (stolen bases per PA, 2022-24 rows only -> 53,274 PA): triples +.0048 per SD (t 1.5), no significant effect on other outcomes.  PITCHERS (visiting, production profile, 131,767 PA): HR-prone -> HR/PA -.0121 (t -5.9) but wOBA only -.0116 (t -1.4) because singles +.0152 (t 2.2) and walks offset; Air% -> HR +.0029 (t 1.5), singles -.0135 (t -2.0), BB -.0102, K +.0173 (t 2.3), wOBA -.0104 (t -1.3): the pitcher Air% term does not capture the HR side either.

## 2026-10-10 -- BATTED-BALL PROFILES JUDGED ON BALLS IN PLAY ONLY (user: K% and BB% effects are closer to global; Air% is only a function of batted balls)
Script: `fl_park_bip_profiles.py` (log output/bip_profiles_log.txt).  Rows = balls in play (88k); outcomes per ball in play (wOBA on contact, HR, 3B, 2B, 1B); no K%/BB% terms or controls; production visitors-only FE design; per +1 SD.  The per-PA 'Air% -> K / BB' coefficients shown earlier (pitcher Air% -> K +.0173) are not meaningful for a batted-ball descriptor and are dropped from the argument.  The ball-level offset numbers (-.026 wOBA per ball in play; triples offset 18% of the HR loss, all non-HR outcomes 10%) were already per ball and stand.
HITTERS (2,428 visiting-hitter balls in play at Trustmark): Air% alone +.0323 wOBA-on-contact (t 2.7); Air% + power (HR/PA) +.0429 (t 3.3) and -.0237 (t -2.1); Air% + power per ball in play +.0418 (t 3.2) and -.0194 (t -1.7).  By outcome (Air% + power per BIP): Air% -> HR +.0017 (t .6), triple +.0052 (t 1.9), double +.0076 (t 1.4), single +.0223 (t 2.2); power -> HR -.0116 (t -2.4), others ~0.  Air%'s wOBA gain splits ~ singles 48%, doubles 23%, triples 20%, HR 8%.  By season the Air% term is identical (+.0424 in 2025, +.0412 in 2026; difference t .05) and power -.0127 / -.0243 (t .5).  => the two factors map to two terms: Air% = the non-HR hit / triple boost, power = the HR loss; an Air-only profile does not cover the HR side; the batted-ball-only Air% effect is stable across seasons (the per-PA estimate was noisier).
PITCHERS (2,416 visiting-pitcher balls in play): Air% allowed -.0059 (t -.6); FB+PU share -.0102 (t -1.0) -- through HR per ball -.0064 (t -2.5); production-style joint (Air%, HR per BIP, exit speed, left): Air% -.0017 (t -.2), HR per BIP allowed -.0165 (t -1.4; HR -.0173 per ball, t -5.6, offset by singles +.0279, t 2.9), exit speed allowed -.0190 (t -1.8), left +.0026.  No batted-ball profile on the pitcher side clearly moves wOBA on contact; HR-prone pitchers lose home runs at Trustmark but gain singles.
IMPLICATION (open for the user): the production PITCHER profile's only surviving term is K% allowed (+.0115/SD), a non-batted-ball term; with K/BB treated as global, with the roster-fit contamination (every visiting pitcher faces our low-strikeout hitters), and with no out-of-sample benefit (fl_park_pitcher_fb25_model_test.py: K%-only RMSE .10683 vs none .10712), dropping it (pitcher shift = global ~0) is the consistent choice; recommended, awaiting the user's call.

## 2026-10-10 -- PITCHER MISSISSIPPI SHIFT: PROFILE MODEL -> GLOBAL-ONLY (user: "Yes, let's shift to global-only and focus MIS effect modeling on hitters")
Change: `fl_pitcher_park.py` now uses PROFILE_TERMS = ("global",): every pitcher's park effect = the global Trustmark effect (bootstrap-t shrunk -> 0.000; raw +.0027, SE .0086, t .31), SD from the bootstrap SE of the global effect (0.172 runs/9 at half exposure).  The old profile model is reproducible with PROFILE_TERMS = all five names; backups in output/_backup_before_pitcher_global_only (coefs json, ra9 parquet, old module).  `fl_ra9_trustmark_projection.py` re-run (park-neutral RA9 and park-neutral P(WAR>0) unchanged; Mississippi RA9 = park-neutral RA9; Mississippi P(>replacement) moved by 0.026 on average, 0.141 at most; old shift range -0.60 to +0.73 runs/9, SD 0.216), parquet copied to the dashboard; app text updated (leaderboard help, player-page Trustmark-fit metric).
MODEL-CHANGE REPORT (fl_pitcher_global_only_test.py; 205 visiting pitcher-seasons, target = FE-residualised wOBA allowed at Trustmark, frozen inverse-variance weights from OLD-CV, pitcher-cluster bootstrap): weighted RMSE / MAE
  5-fold by pitcher:  OLD-final (shipped coefficients, in-sample) .10605 / .07869 | OLD-CV (4 terms refit + shrunk) .10726 / .07961 | NEW global-only .10712 / .07949.  new vs OLD-CV: dRMSE -.00015 [-.00120, +.00104] (P better .56), dMAE -.00011 [-.00141, +.00127] (P .53); new vs OLD-final: dRMSE +.00106 [+.00011, +.00196] (P better .03), dMAE +.00081 (P .11).
  cross-season:       OLD-final .10605 / .07869 | OLD-CV .10958 / .08273 | NEW .10712 / .07949.  new vs OLD-CV: dRMSE -.00246 [-.00452, -.00034] (P better .97), dMAE -.00324 [-.00547, -.00086] (P .99); new vs OLD-final: dRMSE +.00106 (P .03), dMAE +.00081 (P .12).
Reading: the shipped profile fits its own training plate appearances slightly better (as it must: it was fitted on them); out of sample the new model is tied by pitcher and clearly better across seasons.  Cost of the switch in the RA9 outputs: pitchers with high K% lose their +0.3 to +0.7 runs/9 Mississippi penalty, low-K pitchers their small bonus.
Consequences: pitcher leaderboard 'Proj. MIS RA9' equals the park-neutral RA9 and 'MIS RA9 Effect' is 0.00 for every pitcher (columns kept as the user specified; candidates to hide); Mississippi Targets pitcher ranking = park-neutral ranking.  MIS effect modelling now focuses on HITTERS (Air% and power terms on balls in play hold up across seasons).

## 2026-10-10 -- HITTER PROFILES AT TRUSTMARK ON BALLS IN PLAY, INCLUDING HANDEDNESS (user: "Let's dive into different hitter profiles, including handedness")
Scripts: `fl_hitter_profile_screen_bip.py`, `fl_hitter_platoon_confound.py`, `fl_hitter_platoon_excess.py` (logs output/hitter_profile_screen_bip_log.txt).  88,309 balls in play (2,428 visiting-hitter balls at Trustmark); production park design; baseline = Air% (+.0416, t 3.2) + power as HR per ball in play (-.0192, t -1.6), wOBA on contact per +1 SD.
SCREEN (each added to the baseline): fly-ball share -.024 (t -1.5), pop-up +.022 (1.2), line-drive -.002, fly+pop-up -.003, EV90 -.013 (-.7), hard-hit -.010 (-.6), mean EV -.005, barrel -.008, pull share +.001, oppo share -.008, pull-air -.011 (-.7), oppo-air -.001: nothing adds to Air% + power (|t| < 1.6); ground-ball share is an exact mirror of Air% (collinear, row uninformative).  Out-of-season SSE gains are ~0 for all of them.
HANDEDNESS: (1) left-handed (side used in the PA) +.0436 (t 1.8; 2025 +.085, 2026 +.014; out-of-season -0.37% / +0.11%, P .16 / .98): LHH balls at Trustmark average .391 wOBA on contact vs .392 league, RHH .347 vs .384 (-.037).  (2) by-side slopes: Air% LHH +.063 (t 3.2) vs RHH +.027 (t 1.6), difference t 1.4; power LHH -.019 vs RHH -.023 (same).  (3) pull-air share x left +.040 (t 1.9): LHH +.018 (t .8), RHH -.023 (t -1.4); not stable out of season.
SAME-HAND MATCHUP: my first screen reported -.0588 (t -2.9) -- WRONG as an excess: the model had no ordinary platoon term, so it measured the full Trustmark platoon gap.  Corrected (`fl_hitter_platoon_excess.py`): ordinary same-hand effect -.0255 wOBAcon (t -5.7; HR -.0091, t -6.4) in all parks; Trustmark EXCESS -.0374 (t -1.9; HR -.0054, t -1.1; non-HR hits -.0264, t -1.4); by season -.029 (SE .028) / -.043 (SE .028), difference t .36; out-of-season SSE gain +0.17% (P .89) and +0.05% (P .59).  Group check (platoon gap, wOBAcon, same minus opposite): visiting hitters at Trustmark -.060 (SE .019); MS pitchers on the road -.032 (.022); MS hitters at home -.044 (.028); MS hitters on the road -.019 (.028); all other games -.0255 (.005): the extra gap shows when MS is the batting AND the pitching team at Trustmark, not on the road -> a park (or home) effect, not a Mississippi-staff trait.  The Trustmark platoon gap is ~2.5x normal.
LEFT vs SAME-HAND are confounded: RHH face a same-hand pitcher 68% of the time, LHH 24%.  Together: left +.0306 (t 1.3), same-hand excess -.0256 (t -1.3): cannot be separated at this sample size; both point the same way (right-handed hitters do worse at Trustmark on contact).  Rough size if real: LHH minus RHH ~ +.04 wOBA per ball in play ~ +.028 per PA at full exposure ~ 8 runs per 600 PA at half exposure (before shrinkage); current production has no handedness effect (per-PA left +.015, t .8 -> shrunk to 0).
DECISION: no change yet.  The only handedness candidate near conventional significance (t ~ 1.8-1.9) is unstable by season (left) or confounded (same-hand); baseline Air% + power stay.  Options for the user: add a left term shrunk by the production rule (t 1.8 -> keep ~69%), or pre-register handedness / same-hand for 2027.

## 2026-10-10 -- DOES THE HANDEDNESS EFFECT CHANGE WHEN CONTROLLING FOR POWER AS EV90?  (user follow-up; `fl_hitter_left_ev90.py`)
Left-handed (side used in the PA) effect on wOBA on contact per ball in play at Trustmark, visiting hitters: no controls +.0485 (t 2.0); EV90 +.0468 (t 2.0); Air% +.0423 (t 1.8); EV90 + Air% +.0392 (t 1.7); HR per BIP + Air% (earlier baseline) +.0436 (t 1.8); EV90 + HR/BIP + Air% +.0419 (t 1.7).  HR part +.0055 to +.0090 (t 1.0-1.6), non-HR hit part +.0204 to +.0306 (t 1.0-1.4).  Handedness is unrelated to EV90 across hitters (corr +.01; left-sided rows mean z -0.05, right-sided +0.04), so controlling for it barely moves the effect.  EV90 itself: -.013 to -.019 per SD (t -1.1 to -1.5), same by side (LHH -.022, RHH -.015); left x EV90 -.0066 (t -.3).  By season (EV90 + Air%): 2025 +.078 (SE .036), 2026 +.011 (SE .030), difference t 1.4: the same instability as before.  Conclusion unchanged: borderline (t ~ 1.7-2.0), not explained by power, not stable by season; decision still open (add shrunk left term vs pre-register for 2027).

## 2026-10-10 -- OVERALL BATTED-BALL PERFORMANCE vs xwOBA AT TRUSTMARK, BY EV x LA (user question)
Scripts: `fl_park_xwoba_surface.py` (surface), `fl_park_xwoba_decomp.py` (within-hitter), `fl_park_xwoba_by_hand.py`, `fl_park_hard_grounders_by_hand.py`; logs output/xwoba_surface_log.txt, xwoba_decomp_log.txt.  80,349 balls in play with wOBA and xwOBA (xwOBA ~ f(EV, LA); within-cell SD .024; league mean .387 vs actual .389).
OVERALL: Trustmark wOBA .349 vs xwOBA .377 (residual -.028); other parks .392 vs .388 (+.004): the park costs ~.032 wOBA per ball in play relative to expectation (all hitters).  Within hitter, visiting hitters (all balls in play, hitter + pitcher + year + hitter x home FE): wOBAcon -.0206 (t -1.2) = contact quality xwOBA +.0016 (t .1) + park residual -.0221 (t -1.6).
SURFACE (residual TM minus other, wOBA per ball): by LA band: < 0 +.011; 0-10 -.008; 10-20 -.039 (t -2.2); 20-30 -.087 (t -4.1); 30-40 -.137 (t -6.9); > 40 -.016.  By EV band: < 80 +.007, 80-90 -.009, 90-95 -.081 (t -5.0), 95-100 -.078 (t -3.4), 100-105 -.067 (t -2.2), 105+ -.126 (t -2.9).  Cells: EV 90-100 x LA 20-30 -.17 / -.22; EV 90-100 x LA 30-40 -.25 / -.23; EV 105+ x LA 10-20 -.29 (n 32); EV 100-105 x LA 30-40 -.78 (n 19); but EV 100-105+ x LA 20-30 ~0 (those still leave).  The park taxes hard-hit (>= 90 mph) air balls at 10-40 degrees, nothing below 90 mph, nothing on grounders (< 10 deg) or pop-ups.
HANDEDNESS: all hitters residual RHH -.041 (t -5.1) vs LHH -.018 (t -1.7); xwOBA TM - other RHH -.025, LHH +.012.  Within hitter (visiting): RHH wOBAcon -.056 (t -2.4) = quality -.015 (t -1.1) + park -.041 (t -2.1); LHH +.029 (t 1.1) = quality +.025 (t 1.5) + park +.004 (t .2): the LHH - RHH gap (~.085) is about half contact quality, half park penalty.  By zone, hard-hit AIR balls are penalised about equally on both sides (EV >= 90: LA 10-20 RHH -.076 / LHH -.055; 20-30 -.119 / -.186; 30-40 -.323 / -.218) although RHH contact quality is lower there (xwOBA -.031 / -.057 / -.134 vs +.006 / -.004 / -.047); the side difference sits in hard-hit GROUNDERS / low liners (EV >= 90, LA < 10): RHH -.054 vs LHH +.051 (difference ~.105, t ~3), mostly up the middle / opposite field (LHH middle hit rate +.083, SE .042; RHH middle -.038, SE .035; pull ~0 both); fielder position is not tracked for infield plays, so infield defence vs surface cannot be separated.  Exploratory.

## 2026-10-10 -- HITTER MISSISSIPPI TRANSLATION: BATTED-BALL RESIDUAL SURFACE, POOLED WITH MISSISSIPPI HITTERS (user: "this is the basis we should use for our park effects; it's the best control on competition level" and "it's important to consider the mud monsters hitters too, because they play against a bunch of different defenses")
Scripts: `fl_hitter_park_surface.py` (surface + hitter shift), `fl_hitter_park_surface_test.py`, `fl_hitter_park_ms_check.py` (pooling check + validation), `fl_hitter_quality_by_group.py`, `fl_hitter_park_surface_final.py` (final fit + bootstrap), `fl_hitter_park.py` (PARK_MODEL = "surface" | "profile"); backups output/_backup_before_hitter_surface.
DEFINITION: park effect = what Trustmark does to a ball of a given exit speed / launch angle / batting side, i.e. effect(x) = f_TM(x) - f_away(x) on hitter-adjusted residuals res_adj = (wOBA - xwOBA) - the hitter's own away mean (shrunk n0 150), depth-3 gradient boosting (40 balls per leaf).  Contact-QUALITY shifts (xwOBA) are deliberately not part of the park effect.  Hitter shift per ball in play = mean effect over his own non-Trustmark balls, shrunk toward the league mean by n/(n+60); per PA = 0.671 x (shift - league mean) + global total (the fixed-effects global effect, shrunk to 0.000); K%, BB% and everything not on a batted ball are global (same for every hitter).  Bootstrap (100 game-cluster refits) gives a per-hitter SD (median .0077 per ball in play).
FORKS AND CHOICES: (1) Trustmark training balls = VISITING hitters + MISSISSIPPI hitters at home (reverses the visitors-only rule for the hitter side, per the user's randomisation argument; away reference = visitors elsewhere + Mississippi on the road).  (2) Features: exit speed + launch angle + batting side; adding spray direction did not help out of sample (evla_side_dir worse), batting side did.  (3) Smoother = regularised gradient boosting (alternative: a 6 x 6 cell table by side).  (4) Shrinkage n0 = 60 balls.  (5) Per-PA conversion uses a constant 0.671 balls in play per PA.
POOLING CHECK (hitter-adjusted residual, Trustmark minus away baseline, visiting hitters facing the MS defence vs Mississippi hitters at home facing 17 defences): all balls -.026 vs -.025 (difference +.002, SE .012); LA 20-30 -.086 vs -.103; LA 30-40 -.151 vs -.114; EV 90-100 -.081 vs -.068; EV >= 90 & LA 10-40 -.122 vs -.152; LHH -.024 vs +.000; RHH -.028 vs -.039: no cell differs by more than noise -> the effect is the park, not one team's defence.
CONTACT QUALITY vs PARK (fl_hitter_quality_by_group.py): xwOBA (contact quality) per SD of Air%: visiting hitters +.0126 (t 1.9), Mississippi hitters at home -.0086 (t -1.0); residual (park effect) per SD of Air%: visitors +.0159 (t 2.0), Mississippi -.0019 (t -.2); per SD of power: visitors -.0117 (t -1.6), Mississippi -.0407 (t -2.9).  So the old 'air-ball hitters gain' result appears only against the Mississippi staff/defence (competition), not for Mississippi's own hitters facing varied defences; the power penalty appears in both groups.
VALIDATION (held-out season; fit on one season, score the other; units = hitter-seasons with >= 8 Trustmark balls in play; T1 = park effect on same-quality contact, T2 = total within-hitter change in wOBA on contact; frozen inverse-variance weights from the level-only model; hitter-cluster bootstrap): weighted RMSE / MAE
  ALL units (170): T1 NONE .09927/.07772 | OLD profile .09935/.07746 | NEW EV+LA .09870/.07611 | NEW EV+LA+side .09846/.07601 | NEW +spray .09934/.07663; NEW(side) vs OLD dRMSE -.00090 (P .74), dMAE -.00146 (P .87).  T2: NONE .12315/.09305 | OLD .12417/.09404 | NEW EV+LA .12360/.09277 | NEW +side .12260/.09174 (dRMSE -.00157, P .80; dMAE -.00230, P .85).
  MISSISSIPPI hitters at home (35 units): T1 NONE .07370/.05927 | OLD .07342/.05842 | NEW EV+LA .06896/.05483 | NEW +side .06791/.05462 | NEW +spray .06883/.05553; NEW(side) vs OLD dRMSE -.00552 (P .986), dMAE -.00380 (P .95).  T2: NONE .08464/.06132 | OLD .08529/.06286 | NEW +side .08161/.05860 (dRMSE -.00368, P .78; dMAE -.00426, P .84).
  VISITING hitters (135): T1 NONE .11287/.08976 | OLD .11317/.08994 | NEW +side .11416/.09009 (dRMSE +.00099, P(better) .27; dMAE +.00015, P .46): a tie.
  (Before pooling, visitors-only training gave NEW worse than OLD for visitors: T1 RMSE .11823 vs .11484.)  Caution: only 35 Mississippi units; every edge is on the Mississippi side.
EFFECT ON THE PROJECTIONS: Mississippi shift per PA: old SD .0107 (range -.037 to +.035) -> new SD .0038 (-.018 to +.013); mean abs change in Mississippi WAR .166, max 1.02; park-neutral wOBA moved by .0002 on average (max .004, Mississippi hitters, through the neutralisation of observed seasons); correlation of old and new shifts -.17 among hitters with >= 250 projected PA; new shift correlates -.67 with projected HR/PA and -.39 with Air% (old: +.89 with Air%).  Dashboard text updated; fl_2027_hitter_projections / fl_hitter_history / model coefs republished.
OPEN: (a) MIS hitters' K% is ~4-5 points lower at home (fl_park_ms_discipline) and is NOT in the batted-ball surface; whether that is a Trustmark effect any hitter gets after adapting (eye / sight lines) or a Mississippi-selection effect is unresolved -- the user treats K/BB as global.  (b) Handedness enters only through the side-specific surface.  (c) The per-PA constant (0.671) and shrink n0 were not tuned.

## 2026-10-10 -- DOES THE RESIDUAL-SURFACE APPROACH WORK FOR PITCHERS?  (user: "quality of contact is so random but at least Air% seems predictable from pitcher traits")
Scripts: `fl_pitcher_park_surface_test.py`, `fl_pitcher_park_surface_drivers.py`, `fl_pitcher_park_surface_air.py` (logs output/pitcher_park_surface_test_log.txt).  Same surface as the hitters (the Trustmark ball set is identical for both sides: visiting pitchers' balls allowed = Mississippi hitters' balls; Mississippi pitchers at home = visiting hitters' balls); a pitcher's shift = the surface averaged over the balls he allows.
PART 1 (160 pitchers with >= 150 non-Trustmark balls): SD of the shift .0063 per ball in play (hitters .0117) = .0042 wOBA per batter faced = 0.084 runs/9 at half exposure (~1.4 runs, ~0.13 WAR for a 150-IP pitcher).  Split-half reliability of the mean effect .31 (random halves; Spearman-Brown .40, implied n0 ~300 balls); 2025 -> 2026 r .35 (78 pitchers); traits (pitcher-grouped ridge) predict it with r .20, weighted R^2 .07.  Drivers: shift correlates -.65 with Air% (reliability .66), -.64 with fly+pop-up share, -.51 with hard-hit share, -.54 with mean EV allowed, -.73 with the hard-hit AIR share (EV >= 90 at 10-40 degrees; reliability only .26): Air% is predictable but is only a proxy; the penalty falls on hard-hit air balls, whose share is noisy.  Slope of shift on Air%: -.0044 (2025 train) / -.0079 (2026 train) wOBA per ball per SD of Air%.
PART 2 held-out season (unit = pitcher-season with >= 8 balls allowed at Trustmark, target = hitter-adjusted residual allowed at Trustmark minus own away mean; NONE = level only; HIST = surface over his own balls; TRAITS = ridge-predicted shift; HIST->TRAITS; AIR = shift from his shrunk Air%): weighted RMSE / MAE: ALL (163): NONE .08919/.06196, HIST .08920/.06228 (dRMSE +.00001, P better .47), TRAITS .08941/.06192 (+.00021, P .15), HIST->TRAITS .08925/.06214, AIR .08927/.06214 (+.00008, P .37).  Mississippi pitchers at home (39): NONE .03888/.02969, HIST .03999/.03060 (P .06), TRAITS .03893/.02935, AIR .03935/.02981.  Visiting pitchers (124): NONE .12624/.10046, HIST .12588/.10017 (-.00035, P .81), TRAITS .12644/.10058, AIR .12612/.10061 (-.00011, P .65).  Nothing beats a single level; per-pitcher noise (RMSE ~.09-.13) is ~15-20x the signal SD (.006), so the test has no power either way.
CONCLUSION: the approach is coherent (same surface, same pooling) and the direction is right (more hard-hit air balls allowed -> bigger penalty), but pitcher differences are tiny, only ~40% reliable, weakly predictable from traits, and not detectable out of season.  NOT adopted: the pitcher Mississippi shift stays global-only (user decision earlier today).  If wanted for display, the shift could be shown as a note (SD ~0.08 runs/9 at half exposure).

## 2026-10-10 -- DOES PITCHER HANDEDNESS MATTER UNDER THE RESIDUAL-SURFACE APPROACH?  (user question; `fl_pitcher_hand_park.py`, log output/pitcher_hand_park_log.txt)
77,656 balls; 28% from left-handed pitchers; 4,441 at Trustmark.
A PROFILE ALLOWED (non-Trustmark, 66 LHP / 178 RHP with >= 100 balls), LHP - RHP: Air% -.003 (t -.3); mean launch angle -0.45 deg (t -.6); fly+pop-up share -.011 (t -1.2); hard-hit share (>= 90 mph) -.019 (t -2.9); hard-hit AIR share -.006 (t -1.1); mean exit speed -0.75 mph (t -3.3); share of balls from left-handed batters 35% vs 46% (t -13.8).  Lefties allow the same Air% and launch angle but softer contact.
B SURFACE SHIFT (wOBA per ball in play; negative = more penalty): LHP -.0292 vs RHP -.0300, difference +.0009 (t .7) = +0.01 runs/9 at half exposure.  Split: contact channel (batter side at the league mix) +.0033 (t 2.9; lefties' softer contact is taxed less), batter-side channel -.0024 (t -9.5; lefties face more right-handed batters, whose balls are taxed more); the two nearly cancel.
C DIRECT TEST (hitter-adjusted residual wOBA - xwOBA, all Trustmark balls vs away): all -.028 (t -4.2); LHP -.019 (SE .012), RHP -.032 (SE .008); LHP - RHP +.013 (SE ~.014).  By matchup: LHP vs LHH (same hand) +.005 (SE .022, n 448); LHP vs RHH -.032 (t -2.1); RHP vs RHH (same hand) -.040 (t -4.3); RHP vs LHH -.021 (t -1.8).  Hard-air share of the Trustmark balls vs away: LHP vs LHH .165 vs .189; RHP vs LHH .255 vs .224 (left-handed batters hit more hard air balls against righties at Trustmark: a contact-quality shift, excluded by definition).
CONCLUSION: pitcher hand does not change the park effect in any detectable way once ball quality and batter side are conditioned on; its two channels (softer contact; different batter-side mix) offset.  The only visible pattern is in the same-hand cells (LHP vs LHH about no penalty, RHP vs RHH -.040), on small samples and inconsistent with the hitter-side same-hand excess.  Not added; pitcher shift stays global-only.

## 2026-10-10 -- PITCHER HANDEDNESS LEFT OUT; MIS RA9 COLUMNS REMOVED FROM THE PITCHER LEADERBOARD (user: "let's leave it out of the pitcher model. Then remove the Proj. MIS RA9 and MIS RA9 Effect columns")
Pitcher park model unchanged (global-only; handedness not added).  Pitcher leaderboard columns now: Pitcher, Team, 2026 IP, Proj. IP, Proj. RA9-, Proj. K%, Proj. BB%, Proj. K-BB%, Proj. WAR, P(WAR>0); 'How to read' bullet replaced by a Park note (no pitcher park adjustment; park-neutral numbers).  The Mississippi Targets page still shows 'MIS RA9' (it equals the park-neutral RA9; used as the staff-comparison basis) with its help text corrected; the player-page 'Trustmark fit' metric (0.00 +- 0.17 runs/9) is unchanged.  Data columns (proj_ra9_trustmark_half etc.) remain in the parquet.  Backup of the previous app.py: output/_backup_before_hitter_surface/app.py.bak.
Follow-up (user: "Remove it from the targets page also, and the Trustmark fit column"): Mississippi Targets page no longer shows the MIS RA9 column, its colour scale or the 'Staff median MIS RA9' tile; the sort option 'Mississippi RA9 (rate only)' became 'Rank on MIS staff (rate only)' (same ordering: the rank is by projected RA9); help text and caption reworded (park-neutral basis).  Player page: the 'Trustmark fit (runs/9, half games)' metric and its +-0.17 caption removed.  Still on the player page (identical to the park-neutral values now): the 'Mississippi (half games at TrustmarkPark)' RA9 tile and P(beating replacement) tile -- candidates for removal.  AppTests clean.
Player page: the two 'Mississippi (half games at TrustmarkPark)' tiles (projected RA9, P(beating replacement)) removed (identical to park-neutral).  AppTests clean.
BASERUNNING IN HITTER WAR (user question, checked 2026-10-10): the PROJECTED hitter WAR is bat + fielding + position + replacement, NO baserunning (user decision 2026-10-07; fl_hitter_projection.py docstring).  PUBLISHED WAR (war_players, shown in the history table and used in the WAR calibrations) INCLUDES baserunning (bsr_runs).  2026 published baserunning, hitters >= 250 PA (n 141): mean +0.5, SD 7.3 runs per 600 PA (range -17.7 to +21.8), i.e. SD ~0.6 WAR per 600 PA; the 2025 WAR rows have no baserunning, so its year-to-year persistence cannot be measured yet.  (The 'bsr' key read in read_league_rpa is the league runs total used for wRC+, not baserunning.)

## 2026-10-10 -- VALUE OF THE PROJECTION MODELS vs JUST LAST YEAR'S STATS, IN WINS  (user question; `fl_projection_value_test.py`, log output/projection_value_log.txt)
Out-of-sample 2025 -> 2026, WAR on the projection's own definition.  Pools: 139 hitters (>= 20 PA in 2026, complete last-year inputs; 144 before the NaN drop) and 113 pitchers (2025 record, >= 10 IP in 2026).  NAIVE-1 = last year's park-neutral wOBA / fielding / position per PA (hitters) or RA9 (pitchers) x last year's playing time x 102/96, through the same WAR formula; NAIVE-2 = last year's rates x the model's playing time; MODEL-R x NAIVE-PA = the model's rates x last year's playing time.  (Naive hitter inputs lack 2025 infield fielding / position data, as the 2025 table does.)
ACCURACY (projected vs realised 2026 WAR): hitters RMSE 1.232 vs 1.449 naive (dRMSE -.216, 90% [-.326, -.109], P 1.00), MAE .991 vs 1.167 (P 1.00); corr .468 vs .459, Spearman .333 vs .337, slope of realised on projected .96 vs .47.  Pitchers RMSE 1.134 vs 1.407 (-.273, [-.442, -.117], P 1.00), MAE .868 vs 1.059; corr .517 vs .398, Spearman .410 vs .358, slope 1.31 vs .45.  NAIVE-2 is no better than NAIVE-1 (1.476 / 1.337) and MODEL rates x naive playing time ~ MODEL (hitters 1.238, pitchers 1.124): nearly all the accuracy gain comes from the RATE model (regression of last year's rates); the playing-time model adds ~0 to RMSE.  Naive WAR overstates player-to-player differences about 2x.
SELECTION (wins per season = realised-WAR difference of the K best by MODEL minus the K best by NAIVE-1; players bootstrapped, B = 300): hitters K=5 from the whole pool +4.13 [-2.73, +8.14], K=9 +2.47 [-6.04, +9.51]; from random candidate pools: K=5 of 40 +0.28 [-1.42, +2.13], of 25 +0.29; K=9 of 40 +0.56 [-1.79, +2.76], of 25 +0.03.  Pitchers K=3 whole pool +4.13 [-0.66, +8.07], K=5 +6.46 [-3.61, +11.55]; K=3 of 40 +0.92, of 25 +0.63; K=5 of 40 +0.62 [-1.72, +2.87], of 25 +0.84 [-0.68, +2.12].  Decomposition (K=9 hitters whole pool): rate model +7.10, playing-time model -4.63.  COMBINED REALISTIC SCENARIO (9 hitters chosen from 40 candidates + 5 pitchers from 30): +1.20 wins per season (hitters +.45, pitchers +.75), 90% interval [-1.28, +4.17], P(> 0) .77.
CAVEATS: one transition (2025 -> 2026), small pools; survivor filter (players who played in 2026) identical for all methods; excludes rookies / other-league bridge, Mississippi park fit, prices (a market that already regresses last year's stats leaves less to gain; one that pays for last year's stats leaves more through avoided overpays, which a pure selection test does not count); 1 WAR ~ 1 win.

## 2026-10-10 -- VALUE OF THE ROOKIE BRIDGES IN WINS  (user: "what about on rookies?"; `fl_rookie_value_test.py`, log output/rookie_value_log.txt)
FL DEBUT players with other-league history; out-of-sample (hitters: nested leave-one-debut-class-out, 200 debuts 2023-26; pitchers: leave-one-out WLS on the unified-bridge debut pairs, 459 pitchers one pair each).  MODEL = bridge; RAW = rank by the raw latest-season other-league line (hitters: raw MiLB wOBA proxy; pitchers: raw K% minus raw multi-year BB%), unadjusted and unshrunk; RANDOM = no information.  Realised = first FL season (hitters: wOBA deviation from the FL league -> batting WAR per 400 PA = x 32.8; pitchers: first-season K% / BB% turned into runs per 9 with the RA9 baseline slopes -> WAR per 100 IP = x 1.04; K/BB-skill only, no luck, no fielding / baserunning).
ACCURACY (weighted, frozen weights, bootstrap): hitters league-mean guess RMSE .0594 (1.95 WAR/400 PA) / MAE .0456 (1.50) -> MODEL .0567 (1.86) / .0417 (1.37): dRMSE -.0027 [-.0047, -.0007] P .98, dMAE -.0039 P 1.00; correlation with realised .23 (RAW .14), Spearman .23 (RAW .10).  Pitchers .8454 runs/9 (0.88 WAR/100 IP) / .6043 (0.63) -> MODEL .8161 (0.85) / .5778 (0.60): dRMSE -.0293 [-.0521, -.0062] P .98, dMAE -.0265 P .96; correlation .17 (RAW .14), Spearman .27 (RAW .28).
SELECTION (wins per season for K picks together, each hitter = 400 PA, each pitcher = 100 IP; players bootstrapped): hitters MODEL vs RANDOM: K=3 of 20 +1.68 [+.25, +3.33], of 40 +2.71 [+.21, +5.12], whole pool +5.51; K=5 of 20 +2.56 [+.46, +4.98], of 40 +2.69 [-.21, +5.97]; MODEL vs RAW stats: K=3 of 20 +0.20 [-1.14, +2.09], of 40 +1.12 [-1.74, +3.83]; K=5 of 20 +0.73, of 40 +0.23 (all intervals include 0).  Pitchers MODEL vs RANDOM: K=3 of 20 +1.41 [+.76, +2.02], of 40 +1.61 [+.62, +2.74]; K=5 of 20 +1.74 [+.86, +2.64], of 40 +2.59 [+1.28, +3.97]; MODEL vs RAW stats: -0.06 to -0.50 wins (all intervals include 0): the pitcher bridge does NOT beat ranking by the raw K% - BB% line; its value is the level / shrinkage calibration, not the ranking.
CAVEATS: first-season skill only (no playing-time or role projection for rookies, no fielding / position / baserunning); other-league pools are the debut hitters / pitchers who actually signed and played in the FL (selected); K=3-5 picks per season is the scale a team operates at.

## 2026-10-10 -- BASERUNNING STICKINESS (2026 split-half), user: "We are not able to do baserunning WAR for 2025. Can we check 2026 reliability / split-half stickiness?"  (`fl_baserunning_reliability.py`, log output/baserunning_reliability_log.txt, output/baserunning_reliability_2026.csv)
Source: pbp_events (2026 only: why baserunning WAR exists only for 2026): 290,569 events; 17,064 steal opportunities (runner on 1B with 2B open, one per runner x PA), 2,685 attempts; 8,400 extra-base opportunities (single / double with a runner on 1B / 2B); 4,202 GDP opportunities.  Run values approximate (SB +.20, CS -.45, extra base +.27, out on bases -.65, GDP -.44): they weight the composite only.  A placeholder runner "__PLACED__" (the automatic extra-innings runner; 'out' on every hit by construction, 29 opportunities) was inflating the outs-on-bases reliability to .91 and is excluded; impossible out counts (>= 3) also excluded.
REBUILD vs PUBLISHED 2026 (223 hitters >= 100 PA): steals SB r .97, CS .96, wSB-lite vs wsb_runs .93; extra bases (UBR-lite) vs ubr_runs .76; GDP-lite vs wgdp_runs .79; composite vs bsr_runs .84 (rebuilt composite SD 4.97 runs per 600 PA vs published 8.59: the lite version captures a part of the published scale).
SPLIT-HALF RELIABILITY (alternating games | first vs second half; split-half r -> full-season = 2r/(1+r)): steal ATTEMPT rate .78 / .78 -> .88 / .88; steal runs per opportunity .21 / .30 -> .35 / .46; steal SUCCESS rate (SB / attempts) .13 / .13 -> .23 / .23; extra bases taken above expected .30 / .39 -> .46 / .56; outs on the bases above expected .06 / .03 -> .11 / .07 (noise); GDP above expected .25 / .20 -> .40 / .33; COMPOSITE baserunning runs per PA .45 / .43 -> .62 / .60.
Implied talent: composite full-season reliability ~ .6 (n0 ~ 400 PA at 600 PA); with the published scale (SD 8.6 runs per 600 PA, ~0.7 WAR) the true-talent SD is ~ 0.55 WAR per 600 PA.  Caveats: within-season split halves mix talent with role, lineup slot and team steal policy (attempt rate is partly the manager's green light); there is no second season to test persistence across seasons (2025 has no baserunning); rebuilt, not the published formula.

## 2026-10-10 -- PIONEER LEAGUE 2026 HITTING INGESTED (user paste of 13 Baseball Reference team tables; provisional hitter bridge)  (`fl_pioneer_hitting_2026.py`, `fl_hitter_bridge_pioneer.py`; raw output/pioneer_2026_raw/pioneer_2026_br_hitting.csv)
13 pasted tables = 11 distinct teams (two pasted twice, removed) -> one of the 12 Pioneer teams was not supplied; no team names in the paste.  271 player-team rows, 265 distinct players (BR ids), 6 players on two teams (rows summed: Trimble, Winland, Feliciano, Sim, Valdes, Yuran); 261 with PA > 0, 116 with >= 150 PA, 32 already in the FL data, 17 also in the MiLB table.  Pioneer 2026 league means (PA-weighted): wOBA proxy .412, K% .175, BB% .133, HR/PA .038 (an extreme offensive environment: FL 2026 .374 / .196 / .115 / .026).
DATA CHECK: team totals rows reconcile exactly with the player rows in 4 of 11 tables (4, 5, 7, 9) and not in 7 (PA gaps of +306, +164, -390, +100, -68, -6, -365).  The player rows were summed by hand straight from the paste for four of the failing tables (1, 2, 3, 6: 4,425 / 4,402 / 3,924 / 4,411 PA) and equal the file, so the gaps are in the source tables (players missing from a page, a player listed without being in the team total), not transcription.  Rows used as pasted.
PROVISIONAL PROJECTIONS: the MiLB-fitted pooled mapping (fl_hitter_bridge_coefs.json) applied to deviations from the Pioneer 2026 mean (u = dev x PA/(PA+n0), clipped to the MiLB training range, 'beyond validated range' flag 26% of hitters with >= 150 PA).  No Pioneer hitting before 2026 exists in the project (pio_overlap_batting holds only 255 Pioneer seasons of players who also played in the FL, no league means), so there are no Pioneer -> FL debut pairs to fit or check a Pioneer-specific mapping.  Projected wOBA among >= 150 PA: mean .378, SD .014, wRC+ 82-127.  Dashboard: Hitters > Rookie / MiLB Lookup now has a 'Source league' toggle with a provisional warning.  STILL NEEDED to fit the real Pioneer bridge: pre-2026 Pioneer hitting for the whole league (user: StatsCrew) and college hitting (FanGraphs).

## 2026-10-10 -- SHRUNK BASERUNNING TERM ADDED TO PROJECTED HITTER WAR (user: "I think we can add the shrinkage term")  (`fl_baserunning_term_test.py`, log output/baserunning_term_test_log.txt; `fl_hitter_projection.py`)
Term: bsr_rate = mu + (own 2026 published bsr_runs per PA - mu) x PA / (PA + 300), mu = the 2026 league mean (about 0), no 2026 baserunning data -> mu; added to the runs above replacement per PA (rar_rate = bat + fielding + position + baserunning + replacement).  WAR simulation: baserunning drawn with sd = sqrt(PA^2 tau^2 n0/(PA+n0) + PA tau^2 n0), tau = .0094 runs per PA (published scale, from the SD of 7.3 runs per 600 PA at ~.6 reliability).  The backtest frames (data <= 2025) have no baserunning, so the term is 0 there and the earlier calibrations are unchanged.
MODEL-CHANGE REPORT (within 2026, held-out half; rebuilt composite per PA; n0 chosen on the alternating-games split = 300 PA; frozen weights; runner-cluster bootstrap): first half -> second half (190 runners) RMSE none .00878 -> shrunk .00791 (-.00087, 90% [-.00123, -.00052], P 1.00), MAE .00693 -> .00616 (P 1.00); second -> first .00931 -> .00836 (P 1.00), MAE .00712 -> .00655 (P .998); alternating games (257) odd -> even .00968 -> .00904, even -> odd .00954 -> .00889, MAE .00743 -> .00689 / .00737 -> .00667, all P >= .999.  Slope of actual on predicted .1.48 (n0 = 300 is, if anything, slightly strong).  Caveat: one season only (no cross-season persistence available), rebuilt not published components, team steal policy and role are inside 'baserunning skill'.
EFFECT: 621 projected hitters; 384 with 2026 baserunning; term per 600 PA mean -0.05, SD 2.45, range -14.0 to +10.3 runs (SD ~ 0.2 WAR); proj WAR change mean abs 0.051, max 0.47 (largest: -0.39 / +0.45); park-neutral wOBA and projected PA unchanged; WAR per 600 PA SD .976 -> 1.022; 80% range width 2.39 -> 2.46; mean abs change in P(WAR>0) .016.  Dashboard: Proj. WAR / WAR-400 text updated, 'Where the runs come from' table has a 'Baserunning (shrunk, 2026)' row; published WAR still includes baserunning (unshrunk).


## 2026-10-12 -- DASHBOARD REVIEW FOLLOW-UPS (user review answers 1-9)  (backups of every replaced code / data file: _code_backup_before_review_fixes_20261012/)
REVIEW (read-only, 2026-10-12): all 11 views ran clean; findings: (1) wOBA blend halves treat Trustmark differently (component half uses HR/PA neutralised with the 0.446 HR factor, direct half uses the net-zero surface; Mississippi hitters' component - direct gap +.0078 vs others, t 1.6); (2) infield fielding is 0 for 90% of SS / 73% 2B / 71% 3B / 50% 1B regulars even in 2026; (3) hitter park fit unstable across defensible models (old profile vs surface shifts corr -.17; the surface model TIES / is slightly worse for visiting hitters held out: T1 RMSE .11416 vs .11317 profile / .11287 none; its edge is the 35 Mississippi units); (4)-(5) thin hitters / short arms; (6) ROS pitcher 'ROS RA9 (MIS)' still carried the old K%-profile shift (SD .22 runs/9), shown by default; (7)-(9) stale / long dashboard text, player pages showing non-inputs (Stuff+, horiz/vert var) and missing the headline outputs, hidden sort key, inconsistent floors; deploy risk: use_container_width deprecated + unpinned streamlit.
- **Item 1 (user: "a" = neutralise K% and BB% for Trustmark as well):** `fl_park_kbb_neutral.py` (visiting-hitter design: year + hitter + pitcher + hitter x home + park FE; game-cluster bootstrap B = 100; production shrink): **K% raw -.0082 (SE .0099, t -.83) -> 0; BB% raw -.0035 (SE .0080, t -.44) -> 0** (all-PA FE had K -.0242 / BB +.0070: that was Mississippi hitters' own home K drop, DiD -4.7 points t -4.0, not the park). Neutralising K / BB by the park therefore changes nothing; NOT implemented (would be a zero term). The mismatch is the HR neutralisation (component half restores the HR the park takes away, nothing offsets it). User asked instead for component park factors with generic home-field advantage controlled, related to each hitter, plus the Mississippi extra beyond that: `fl_park_component_factors.py` (results in the next entry).
- **Item 2 (fielding):** user first said drop fielding; model-change report `fl_fielding_drop_report.py` (2025 -> 2026 backtest, 146 hitters, realised-PA weights, hitter bootstrap): vs realised WAR INCLUDING fielding RMSE 1.2790 -> 1.3887 (+.110 [+.055, +.168], P(better) .00), MAE 1.0551 -> 1.1265 (+.071, P .01); OF + C RMSE 1.31 -> 1.52, IF 1.250 -> 1.246 (their term is already 0); vs realised WAR excluding fielding .0159 better (P .68). **User decision: keep OF / C fielding, flag infielders.** Implemented in `fl_hitter_projection.py`: `fld_measured` = primary position LF / CF / RF / C; unmeasured hitters (infield, unknown position, gap hitters) keep 0 fielding and get the realised season spread of measured outfielders (2026 OF regulars, ~12 runs / 600 PA) as their fielding SD in the WAR simulation instead of the 6.2 residual. Dashboard: Pos shown with * (targets), player-page caption "fielding not measured".  80% range width: unmeasured 2.37 -> 2.65 WAR, measured 2.66 -> 2.79 (the latter from item 4). Open (not acted on): CF fielding averages +6.2 runs / 600 PA vs SS +0.2 -- check whether OF fielding runs are relative to all outfielders (then CF's +1.9 position adjustment partly double-counts).
- **Item 4 (thin hitters; user: correlated playing time only):** `fl_informative_pt_test.py` (2025 -> 2026, 156 hitters with >= 1 PA; joint simulation, one parameter: PA residual = rho x z + sqrt(1 - rho^2) x e, z = standardised season wOBA error at the realised PA; rho 2-fold cross-fitted). **rho = +.223 (SD across folds .074).** Per-player scoring: Brier of P(WAR>0) .1877 -> .1857 (90% [-.0036, -.0004], P(better) .98), log loss .5652 -> .5568, 80% coverage .789 -> .801; PA-weighted WAR RMSE 1.2010 -> 1.1878, MAE .9794 -> .9704. By history (pred vs observed P(WAR>0)): < 50 prior PA (n 23) .687 -> .682 vs .652; 50-149 (22) .725 -> .715 vs .591; 150+ (111) .799 -> .790 vs .802. Per-player mean z by history -.04 / -.40 / -.15 (the < 50 group was NOT over-projected per player in this transition; the 4-transition PA-weighted vs unweighted gap stands). **Choice:** the simulation MEAN rises +.07 WAR with rho (cov(PA, rate)) and that made the per-player WAR bias worse in the backtest (-.087 -> -.156; < 50 PA +.147 -> +.068, 150+ -.127 -> -.194), so the published point WAR = projected rate x projected PA ("if he plays his projected role"; mean change -.010, i.e. unchanged) and the correlated simulation drives only the 80% range and P(WAR>0) (`RHO_PT` in `fl_hitter_projection.py`; the simulation mean is kept as `proj_war_sim_mean`). Alternative rejected: use the simulation mean (better PA-weighted, worse per player). Effect on 621 hitters: P(WAR>0) mean -.014 (max .048), p90 +.23, p10 -.003. Lowering the thin-history prior itself was offered and declined (conflicts with the 10-08 PA-weighted target).
- **Item 5 (pitchers projected < 22 IP):** (a) correlated innings (same joint model, lognormal IP; rho -.25): per-arm Brier .2346 -> .2378 (P(better) .01) -- WORSE, NOT adopted (the >= 10 IP data floor distorts the innings marginal; the miss is a level miss). (b) `fl_pwar_short_arm_test.py` (2-fold by pitcher x 50, 312 arms >= 10 IP; mean RA9 miss +.80 all, +.985 short, +.48 others): global level shift Brier +.0037 (P(better) .27); short-arm level shift -.0011 (P .58; group calibration .635 -> .515 vs observed .564, IP-weighted Brier worse .1983 -> .2081); global + short +.0050 (P .24); widen short-arm SD x2.19 -.0014 (P .69). **Nothing fixes it out of sample; no model change.** These arms' P(WAR>0) carries little resolution (near coin flips); revisit with 2027 outcomes.
- **Item 6 (ROS):** `fl_ros_build.py` re-run: pitcher ROS Mississippi shift now 0 (global-only park model; max |shift| was .73 runs/9); hitter ROS unchanged. Dashboard: the 'Show the Mississippi version of RA9' checkbox and column removed. ROS hitter WAR keeps no baserunning (no pre-date data in 2026 snapshots).
- **Items 7-9 / C (dashboard):** explanatory text cut to short 'Column notes' expanders; pitcher player page = park-neutral outputs (K%, BB%, K-BB%, RA9, RA9-, IP, WAR, P(WAR>0)) + the exact model inputs (own K% shrunk, other-league K%, whiffs vs expected, spin z, extension, own BB% with prior, command score, IP per appearance) + the seasons behind them (Stuff+ and horizontal / vertical command no longer shown: not model inputs; the stale 'Air% at league average' warning removed); hitter player page = outputs, Mississippi block, model inputs with what each Trackman prior adds, WAR parts, history (non-input Trackman stats removed); hitter leaderboard shows Proj. WAR (its sort key); pitcher pages use one IP floor (8) and an 'include no-2026 pitchers' box on the leaderboard too; targets park-fit colour centred on 0; the uninformative 'Mississippi wOBA shift (avg)' tile removed; MiLB lookup hides 'beyond validated range' by default and labels the PA slider by league; `use_container_width` -> `width="stretch"`; requirements pinned (streamlit 1.59.1, pandas 3.0.3, numpy 2.4.6; plotly kept per the keep-until-done rule). AppTest: all 11 views + gap / infielder / outfielder / gap-pitcher deep links, 0 exceptions. (Checked and correct: the pitcher Pioneer caption 'all 12 teams'.)

## 2026-10-12 -- TRUSTMARK COMPONENT PARK FACTORS, HOME-FIELD ADVANTAGE CONTROLLED, RELATED TO EACH HITTER; MISSISSIPPI EXTRA  (user: "control for home field advantage in general and then see if there is any extra effects beyond that for Mississippi hitters ... use the individual component park factors and relate them to each hitter"; `fl_park_component_factors.py`, log output/trustmark_component_factors_log.txt, table output/trustmark_component_factors.csv)
PART A (2025-26, 135,382 PA, game-cluster bootstrap B = 40; per PA): visiting-hitter design (hitter x home FE) -- 1B +.0116 (SE .0088, t 1.3; shrunk +.0050, factor 1.03), 2B .0000, **3B +.0041 (t 2.8; factor 1.95), HR -.0141 (t -4.6; factor .44)**, BB -.0035 (t -.4) -> 0, HBP -.0009 -> 0, K -.0082 (t -.9) -> 0.  Generic-HFA design (one league-wide home dummy + Mississippi-hitters-at-home dummy + park; assumes Mississippi pitchers have no extra home effect): same pattern (1B +.0142, 3B +.0039, HR -.0152, BB -.0064, K -.0123).  Park total in wOBA per PA for visiting hitters: raw -.0126, shrunk -.0149, HFA design -.0165 (bootstrap SE .0125: not distinguishable from 0; production's global effect is 0 = the average of the visiting-hitter and visiting-pitcher estimates).  **Mississippi hitters' extra home effect beyond park + generic home advantage: +.0384 wOBA per PA (SE .0195, t ~2.0): BB +2.95 points (t 2.8), HBP +1.07 (t 2.3), K -2.65 (t -1.8), 1B +.3, HR +.4, 3B -.35 (t -1.6)** -- plate discipline, as in fl_park_ms_kbb.
PART B (held out by season; 120 hitter-seasons with >= 20 Trustmark PA; target = wOBA per PA at Trustmark minus own away baseline; frozen inverse-variance weights; hitter bootstrap vs NONE+MS): ALL -- NONE (one level) RMSE .08995 / MAE .06594; NONE+MS (separate Mississippi level) .09235 / .06755 (NONE better, P .98); SURFACE .09215 / .06737; COMPONENT (shrunk factors x his own away component rates) .09347 / .06896 (worse than NONE+MS, P(better) .06 / MAE .04); COMPONENT raw .09394 / .06934 (P .04).  Mississippi at home (33): NONE .05842 / .04616, NONE+MS .06113, SURFACE .05994 (vs NONE+MS P .99), COMPONENT .06130.  Visitors (87): NONE .11695, NONE+MS .11914, SURFACE .11961, COMPONENT .12085 (P(better) .03).  Fold factors differ a lot by season (HR -.12 shrunk / -.27 raw fit on 2025 vs -.68 / -.72 on 2026; 1B +.10 vs .00).
READING: (1) the component factors are real at the park level (HR x .44, triples x 1.95, singles slightly up) but relating them to each hitter's component rates ADDS NOISE out of season (as the user anticipated for 1B / 2B / 3B); (2) Mississippi hitters' extra home edge (+.038, mostly walks / HBP / fewer K) is significant pooled but does NOT carry from one season to the next (a Mississippi-specific level loses to one common level held out), so it is not projected; (3) no personalisation beats a single level held out: the surface model's per-hitter shift only helps relative to the (worse) Mississippi-level baseline.  NO PRODUCTION CHANGE.  The item-1 mismatch (components restore the HR the park takes, nothing restores the triples / singles it adds) remains open for the user: options are raw HR/PA in the components (net-zero basis, same as the direct half) or keep.
Also checked (2026 published WAR, per 600 PA by position): fielding runs are NOT centred within the outfield positions -- CF +9.8, LF -4.8, RF -3.0 (C -0.3, infield ~0) -- while the position adjustment adds CF +1.8, LF -4.9, RF -5.1: outfield fielding is measured against all outfielders and the position adjustment is applied on top, so CF's difficulty is counted twice (CF - LF ~21 runs / 600 PA).  The projection inherits it from the published WAR.  Open for the user (centre outfield fielding within position in the projection, or keep the published convention).

## 2026-10-13 -- wh_sva TARGET AND COUNT CONDITIONING: whiff | swing vs swinging strike vs strike, each with / without count  (user question; `fl_wh_sva_variants_test.py`, log output/wh_sva_variants_log.txt)
Variants = P(y | stuff + location [+ count]) - P(y | location [+ count]); y = whiff given swing (production) / swinging strike per pitch / strike per pitch (strike = not a ball: called and swinging strikes, fouls, balls in play; HBP not a strike); count = pre-pitch balls-strikes rebuilt from each PA's pitch sequence (matches the tracking file's own count distribution within .004 per cell) and added to BOTH models.  Same XGBoost / 5-fold GroupKFold OOF estimator; the frozen K% model REFIT with each variant (full 2025 -> 1H2026), scored on fold 2 (2025 + 1H2026 -> 2H2026), position players excluded, level-free, frozen inverse-variance weights from production's residuals, pitcher bootstrap.  Batter handedness not added (follow-up option).
K% (n = 256; production reproduces the earlier fold-2 numbers exactly, RMSE .05488 / MAE .04392): whiff + count .05521 / .04420 (dRMSE +.00034 [+.00005, +.00064], P(better) .02 -- WORSE); swstr .05487 / .04383 (P .52, tie); swstr + count .05489 / .04372 (P .49); strike .05450 / .04321 (-.00037 [-.00147, +.00071], P .70; MAE P .89); strike + count .05448 / .04319 (P .69).  In-sample coefficient / t in the K% model: whiff .52 (t 2.4), whiff + count .57 (2.5), swstr .65 (1.6), swstr + count .85 (1.7), strike -.01 (t -.03), strike + count .04 (.08) -- the strike variants "win" only by being ignored by the K% model (it behaves like a model with no stuff term), within noise.
BB% (production BB% model + the variant as an extra regressor): whiff -.00004 (P .59), whiff + count -.00007 (P .69), swstr +.00036 (P .11), swstr + count +.00021, strike +.00131 (P .004, WORSE), strike + count +.00058 (P .02): no variant helps BB%.
Reliability (alternating games 2025-26, 448 pitchers, ~526 pitches per half): split-half r / implied n0 (pitches thrown) / same-period corr with K%: whiff .906 / 54 / .517; whiff + count .914 / 50 / .517; swstr .883 / 69 / .494; swstr + count .872 / 77 / .439; strike .700 / 225 / .156; strike + count .735 / 190 / .049.
DECISION (user confirmed 2026-10-13: "stick with what we had"): no change (production whiff | swing without count is the most reliable, most K%-related, and is not beaten held out; count makes the whiff version significantly worse).  Open: the strike result hints that the stuff term adds little held out on this fold (a K% model without any wh_sva term was not run here) -- an ablation is the natural follow-up.

## 2026-10-13 -- ERROR ANALYSIS OF THE HITTING AND PITCHING PROJECTIONS: traits correlated with the errors, their stickiness, and whether they improve the models (user: "deep dive error analysis ... (1) tendencies / traits / locations / circumstances correlated with the errors (2) sticky and stable (3) improve the model, overall and for rookies / low-history")
Scripts: `fl_error_analysis_common.py` (design, pre-specified), `fl_error_analysis_hitters.py`, `fl_error_analysis_pitchers.py` (stickiness + screen; logs output/error_analysis_{hitters,pitchers}_log.txt, CSVs output/error_analysis_*_{screen,stickiness}.csv), `fl_error_analysis_select.py` (single-trait tests + forward selection + reporting standard; logs output/error_analysis_select_{h,p}_log.txt, CSVs output/error_analysis_{h,p}_{single_trait,selected}_tests.csv), `fl_pitcher_velo_k_test.py` (production-harness refit).
SAMPLES (out of sample for the outcome; level-free residuals; frozen standard inverse-variance weights): hitters PRE (preseason path on <= 2025 -> 2026, 144 hitters >= 30 PA), ROS25 / ROS26 (production rest-of-season model as of mid-season 2025-07-04 / 2026-07-07 -> rest of season, 240 / 244 hitters, 137 / 113 rookies); pitchers PRE (frozen K% / BB% model, fold 2 -> 2H2026, 229), RA9 (2025 -> 2026, 312), ROS25 / ROS26 (204 / 226 for K%, BB%; RA9 for 2026 only, 193).  29 hitter traits and 33 pitcher traits from the tracking data (plate discipline, contact quality, batted-ball type / direction, bat speed, pitch mix seen, platoon exposure; pitcher arsenal shape / velocity / release, usage, zone / chase / called strikes, batted balls allowed, role, times through the order, outfield defence behind him), each measured only in the sample's feature window and shrunk by its own split-half n0.
SCREEN RULE: residual ~ z(trait) + centred prediction + log history, player bootstrap; shortlist = BH q <= .10 on the inverse-variance pooled z of the two disjoint seasons (ROS25, ROS26), same sign in both, split-half r >= .40, disattenuated y/y >= .50.  PRE was NOT used for shortlisting (an independent check).
(1)+(2) FINDINGS:
- HITTER wOBA: 14 traits shortlisted, one correlated cluster: plate discipline (whiff% t -2.0 / -2.3, swing% -2.5 / -1.4, chase% -2.3 / -1.4, zone swing%, two-strike whiff%: more swing-and-miss / swinging = worse than projected; called-strike% +1.6 / +2.2), batted-ball type (ground-ball% +2.2 / +2.8, launch angle -1.8 / -2.6, pull-air% -2.2 / -1.3, xwOBA on contact -2.7 / -0.9: ground-ball / low-angle hitters beat the projection, which leans on xwOBA), pitch mix seen (breaking balls seen -1.8 / -2.4) and platoon exposure (same-hand share -1.9 / -1.5).  In PRE the same traits are weak (|t| <= 1.5).  All are sticky (split-half .45-.84; y/y disattenuated .84-1.05) except LD%, sweet-spot%, fly-ball%, pop-up%, LA SD, foul rate, velocity faced, tracked BABIP (split-half .11-.39).
- HITTER K%: nothing (the Whiff% prior already carries it).  HITTER BB%: power (barrel% +2.7 / +4.0, PRE +1.5; xwOBA on contact, hard-hit%, HR per air ball, exit velocity): power hitters walk more than projected (pitched around).
- PITCHER K%: fastball velocity +4.6 / +3.4, PRE +3.0 (pooled t 5.6), extension +1.0 / +3.7 / +2.4, fastball spin +2.2 / +1.6 / +0.2.  Called-strike-over-expected on takes -1.2 / -2.8 / -2.5 but not sticky (split-half .14).  PITCHER BB% and RA9: nothing shortlisted.  Arsenal / release traits are extremely sticky (.86-.99); batted-ball-allowed traits are not (EV / hard-hit / barrel / xwOBA-on-contact / BABIP / HR per air ball allowed: split-half .05-.23).
(3) IMPROVEMENT (b fit on one season, scored on the other; transfer ROS25 -> PRE; CV inside PRE; weighted RMSE / MAE, frozen weights, player bootstrap):
- HITTER wOBA, forward selection (constant form): whiff% (-), same-hand share (-), called-strike% (+), ground-ball% (+).  ROS25 -> ROS26 RMSE .04958 -> .04835 (-.00124 [-.00250, -.00012], P(better) .96) / MAE .03852 -> .03691 (P .98); ROS26 -> ROS25 .05696 -> .05582 (-.00113, P .96) / MAE P .91; low-history -.00064 (P .83) / -.00218 (P 1.00); rookies -.00082 (P .78) / -.00158 (P .99) (MAE for low-history / rookies worse in ROS25 -> ROS26).  **PRESEASON: worse** -- ROS25 -> PRE +.00050 (P .29), CV within PRE +.00072 [+.00005, +.00146] (P .04), MAE worse (P .03).  Faded form: same pattern, PRE worse (P .002).
- HITTER BB%: barrel% (+ HR per air ball, marginal): ROS25 -> ROS26 .03685 -> .03582 (-.00103 [-.00175, -.00035], P .997) / MAE P 1.00; ROS26 -> ROS25 -.00044 (P .77) / MAE P .87; rookies -.00115 (P .99) / -.00064 (P .81).  **PRESEASON: worse** (ROS25 -> PRE +.00053, P .23; CV +.00029, P .10).
- PITCHER K%: fastball velocity (constant form; nothing added after it): ROS25 -> ROS26 .05366 -> .05314 (-.00052, P .66) / MAE -.00161 (P .92); ROS26 -> ROS25 .05673 -> .05476 (-.00196 [-.00299, -.00082], P .998) / MAE P .99; ROS25 -> PRE .05269 -> .05158 (-.00112 [-.00283, +.00059], P .88) / MAE -.00138 (P .90); CV within PRE -.00118 (P .95) / MAE P .97; low-history and rookies improve in every test (P .56-.99).  Coefficient +.0042 to +.0073 K% per mph of shrunk fastball velocity.  BUT the production-harness REFIT (fl_pitcher_velo_k_test.py: velocity as a regressor in the frozen K% model fit on full 2025 -> 1H2026, n = 129) gives velocity -.0008 (t -.32) and fold 2 RMSE .05488 -> .05515 (+.00027 [+.00012, +.00044], P(better) .003; MAE P .02); replacing spin + extension with velocity is worse (+.00127).  The velocity signal shows in ROS25, ROS26 and the fold-2 residuals but not in the training transition (small n, collinear with wh_sva); only a FIXED external coefficient (fit in-season) helps the preseason projection.
READING: the process traits help the IN-SEASON model and do not help the PRESEASON model.  They stabilise in tens of pitches / balls, so at mid-season they carry information the half-season outcome stats do not yet have; after a full season the regressed outcome stats already contain it (the preseason coefficients are ~0: e.g. wOBA whiff% -.011 vs -.09 in-season).  Caveat: the forward-selection numbers ROS25 <-> ROS26 were also used to select, so they are optimistic; the independent checks are PRE (hitters: no gain; pitchers velocity: gain only as a fixed term) and, for the ROS additions, a future season (2027).
NO PRODUCTION CHANGE (awaiting the user).  Candidates: (a) ROS hitter model + the wOBA discipline / ground-ball set and the BB% barrel term (in-season only); (b) ROS pitcher model + fastball velocity for K%; (c) preseason pitcher K% + fastball velocity as a FIXED prior coefficient (~+.005 K% per mph, provisional like EV90 / Whiff%), or pre-register it for 2027.  Nothing for hitter K%, pitcher BB%, RA9 or the preseason hitter model.

## 2026-10-13 -- NESTED VALIDATION OF THE ERROR-ANALYSIS FINDINGS: they do NOT hold up (user: "Are you confident these variables hold up to cross validation / bootstrapping?")  (`fl_error_analysis_nested.py`; logs output/error_analysis_nested_{h,p}_log.txt; CSVs output/error_analysis_nested[_boot]_{h,p}.csv)
CORRECTIONS to the previous entry: (1) the shortlist and the forward selection used BOTH in-season samples, and the reported cross-season tests were the same two directions -> those gains were optimistic (selection on the test data); (2) the pitcher PRE fold (2H2026 targets) shares its outcome period with ROS26 (after 2026-07-07), so PRE was NOT an independent check for pitchers (hitters: half overlap).  The statement "PRE was not used for shortlisting (an independent check)" was wrong for pitchers.
DESIGN: the whole pipeline (screen with HC1 SEs + BH q <= .10 inside the training season + trait stickiness rules; forward selection by 10-fold-by-player CV inside the training season; fit) uses ONE season; tested on outcomes it never saw (ROS25 -> ROS26 and PRE; ROS26 -> ROS25); PIPELINE BOOTSTRAP: 200 resamples of the training season, whole pipeline rerun each time, scored on the fixed test sample (includes selection uncertainty).
RESULTS (dRMSE new - old; P(better) from the pipeline bootstrap unless stated):
- HITTER wOBA: inside ONE season no trait passes the screen (the cluster appeared only when both seasons were pooled); bootstrap selection is unstable (top trait chase% .37 of resamples trained on 2025, breaking-balls-seen .31 trained on 2026; different leaders by season).  Held-out: ROS25 -> ROS26 mean +.00018 (P .36), ROS26 -> ROS25 -.00011 (P .56), ROS25 -> PRE +.00293 (P .01, worse).
- HITTER BB%: the most consistent signal (power): barrel% selected in .75 of resamples trained on 2026 (single pipeline ROS26 -> ROS25 -.00043, player-bootstrap P .80, MAE P .91); trained on 2025 the pipeline picks hard-hit% + zone contact + zone swing (ROS25 -> ROS26 +.00044, P .23) or, faded, xwOBA on contact (-.00059, P .94).  Pipeline bootstrap: ROS25 -> ROS26 +.00062 (P .18) constant / +.00003 (P .48) faded; ROS26 -> ROS25 -.00002 (P .59) / +.00023 (P .39); ROS25 -> PRE +.00254 (P .00, worse).
- HITTER K%: nothing passes in either season; forcing a selection makes it worse (P(better) .00-.06).
- PITCHER K%: fastball velocity is consistently SELECTED (.90 of resamples trained on 2025, .61 trained on 2026) but the selected sets do not improve held-out K%: single pipelines ROS25 -> ROS26 (velocity + fastball usage) +.00095 (P .26), ROS25 -> PRE -.00011 (P .56), ROS26 -> ROS25 (extension + velocity) -.00105 (P .86); pipeline bootstrap +.00188 (P .09), +.00094 (P .22), -.00003 (P .54).
- PITCHER BB%: nothing passes in either season.
CONCLUSION: the error analysis found residual correlations that replicate in SIGN across the two seasons (pitcher fastball velocity -> K%, hitter power -> BB%, and the hitter discipline / ground-ball cluster -> in-season wOBA), but with ~200-240 players per season they are too small relative to noise for a selected model to improve held-out accuracy reliably, and the selection itself is unstable.  The earlier recommendations (add the sets to the ROS models; velocity prior) are WITHDRAWN.  NO CHANGE.  Pre-registration for 2027 (fixed now, so 2027 is a clean test): (a) pitcher K% + fastball velocity, constant form, single term; (b) hitter BB% + barrel%, constant form, single term; coefficients fit on ROS25 + ROS26 pooled, applied unchanged to the 2027 in-season and 2026 -> 2027 preseason projections, success = held-out weighted RMSE and MAE both better with P(better) >= .90.

## 2026-10-13 -- ROOKIES / LOW-HISTORY / BRIDGE PROJECTIONS vs NAIVE BASELINES (user: "is our model for rookies/low-history/bridge players better than a naive baseline?")  (`fl_lowhistory_baseline_test.py`, output/lowhistory_baseline_test.csv; no selection: existing out-of-sample predictions vs naive predictions on identical rows; level-free, frozen standard weights, player bootstrap; dRMSE = model - baseline)
- HITTER ROOKIES, MiLB bridge (176-200 debuts, nested out-of-fold): beats every baseline on wOBA / K% / BB%: vs league mean wOBA .0594 -> .0566 (P .99; MAE P 1.00), K% .0708 -> .0605 (1.00), BB% .0370 -> .0334 (.999); vs raw MiLB at face value or level-adjusted: wOBA .081-.082 -> .057, K% .098-.100 -> .061, BB% .067-.068 -> .033 (all P 1.00): raw minor-league lines are worse than guessing the league average.
- PITCHER ROOKIES, unified bridge (459, leave-one-out): K% .0631 -> .0605 vs league mean (P .999), BB% .0523 -> .0467 (1.00); vs raw source rates .077 / .070 (P 1.00).  (Earlier finding stands: for RANKING rookies the pitcher bridge does not beat the raw K% - BB% order; its gain is calibration.)
- HITTERS WITH THIN FL HISTORY (< 150 PA, 87 hitter-transitions, history regression LOTO): K% beats league mean (.0641 -> .0546, P .998) and last season (P 1.00), ties one-season shrinkage (P .73; MAE P .98); BB% beats league mean (P .84; MAE .97), ties shrinkage (P .81); **wOBA: no better than the league mean (.0476 -> .0469, P .75; MAE worse, P .24) and identical to one-season shrinkage (P .57)** -- for thin hitters the model is essentially the league average, by design (heavy shrinkage), and nothing in the data beats that.  Last season at face value is far worse (.1030).  Regulars (>= 150 PA): the model beats every baseline (wOBA vs league mean P 1.00, vs shrinkage P .92).
- PITCHERS WITH THIN HISTORY (fold 2): thin (< 150 BF, 144) K% .0595 vs league mean .0715 / face value .0800 / naive shrunk rate .0618 (P 1.00 / 1.00 / .96); BB% .0398 vs .0501 / .0566 / .0440 (all P 1.00); 2026 rookies (149) the same pattern (K% vs naive shrunk P .83, BB% P 1.00).  Gains over simple shrinkage are largest for thin pitchers (bridge / stuff / command terms carry them); for regulars the model ~ shrinkage.
- PITCHER RA9 2025 -> 2026: thin (< 30 IP in 2025, incl. none; 228) 2.254 vs league 2.264 (P .70), vs shrunk own RA9 2.263 (P .69), vs face value 2.631 (P .997); no 2025 record (194): 2.313 vs 2.334 (P .91); >= 30 IP (84): 1.569 vs league 1.725 (P .99), vs shrunk 1.679 (P .97).  For thin arms the RA9 model barely beats the league average (RA9 over few innings is mostly noise).

## 2026-10-13 -- PRODUCTION vs THE BRIDGE AS IS for thin-history players with an other-league record (user: "is it better than just using the bridge as is?")  (`fl_bridge_asis_test.py`, output/bridge_asis_test.csv; level-free, frozen standard weights, player bootstrap)
- HITTERS (44 FL debut hitters 2023-25 with MiLB history and >= 20 PA in their second FL season; production-like = debut season shrunk to the league with the production n0): production beats the bridge as is on wOBA .04570 -> .04130 (P .99; MAE P .99), BB% .03461 -> .02719 (P .996; MAE 1.00), K% .05735 -> .04735 (P .89; MAE .64).  vs shrinking the debut season toward the bridge (the 10-09 candidate): wOBA production better (P .85), BB% production better (P .83), **K% the bridge-prior version is better (.04337 vs .04735, P(production better) .01; MAE .004)** -- the same sample and result as 10-09 (not new evidence), n = 44.
- PITCHERS with a bridge match (fold 2, 179; production already combines the FL rate with the bridge): production beats the bridge as is for K% (< 150 FL BF: .06570 -> .05738, P 1.00; 150+: P .95) and BB% (< 150: .04343 -> .03871, P .99; 150+: P .98).
Reading: once a player has any FL data, the bridge alone is worse than the production projection, for hitters and pitchers.  Open candidate (not applied): a K%-only bridge prior for thin-history hitters (P .99 on 44 hitters; wOBA / BB% not helped); would need a fresh test (2026 debut class's second season, 2027) before adopting.

## 2026-10-13 -- EXHAUSTIVE VALIDATION of the hitter K% BRIDGE PRIOR (user: "Big RMSE drop. Can we do a more exhaustive validation for it?")  (`fl_hitter_k_bridge_prior_validation.py`, output/hitter_k_bridge_prior_validation.csv)
Candidate: hitters with a MiLB record before their FL debut have their K% regression shrink toward their out-of-fold bridge prediction (leave-one-debut-class-out) instead of the league mean; same three-season regression, n0 = production K% n0 refit leave-one-transition-out (nothing tuned on the evaluation rows).  Level-free, frozen standard weights, hitter-cluster bootstrap.
- All FL hitter-transitions of bridge-matched hitters 2023->24 ... 2025->26 (56 transitions, 45 hitters; median FL history 295 PA; mean prior weight .23): RMSE .04636 -> .04274 (-.00362 [-.00581, -.00131], P(better) .998), MAE .03654 -> .03313 (P .996).
- By transition: 2023->24 (14) -.00754 (P .996); 2024->25 (20) -.00184 (P .79; MAE P .57); 2025->26 (22) -.00337 (P .999).  By FL history: 150-399 PA (36) -.00131 (P .84; MAE P .94); the < 150 and 400+ cells have < 15 rows.
- Next-season PA floor >= 1: -.00379 (P .998); >= 100: -.00313 (P .98).  Prior weight n0 x .5: -.00234 (P .995); x 2: -.00532 (P .993) (sensitivity only, not chosen).
- PERMUTATION (bridge predictions shuffled among hitters within a transition, 500 times): shuffled mean +.00170 (worse than production); share of shuffles at least as good as the real bridges .002 -> the gain is the hitter's own MiLB information, not extra shrinkage.
- FULL PRODUCTION PATH 2025 -> 2026 (Whiff% prior term from 2025 pitches added to both arms, 22 hitters): .03883 -> .03594 (-.00289, P .993); MAE P .998.
- Size: K% mean |change| .0086 (max .046) for matched hitters; implied wOBA change through the component half (0.5 x -.177) mean .0008 (max .0041).
Caveats: 45 hitters, and these rows include the 44 of the 10-09 test that raised the idea (this validates the same data more thoroughly; it is not an independent sample); K% was one of four stats tried on 10-09 (permutation p .002 survives a x4 correction).  Clean confirmation = the 2026 debut class's second seasons (2027).
Status: candidate, strongly supported; not applied (awaiting the user).

## 2026-10-13 -- HITTER K% BRIDGE PRIOR -- APPLIED (user: "Let's do it")  (`fl_hitter_projection.py`: K_BRIDGE_PRIOR, k_bridge_dev(); validation in the previous entry)
Change: for hitters who debuted in the FL in 2023+ with a MiLB record before the debut, the K% regression shrinks toward the MiLB bridge prediction (published K% bridge coefficients, fl_hitter_bridge_coefs.json; MiLB seasons before the FL debut) instead of the league mean: proj_K += n0_k x bridge_dev / (history weight + n0_k), applied before the Whiff% term.  Same form and constants as validated (RMSE .04636 -> .04274, P .998; MAE P .996; permutation p .002; full production path P .993).  Not applied in the shifted backtest builds (fl_hitter_pre2026.build_shifted sets K_BRIDGE_PRIOR = False: shifted years do not match real debut years) and not in the ROS model (not validated there).
Effect on the 2027 table (621 hitters): 116 get the prior; K% mean |change| .0105 (max .077), 76 up / 40 down; proj wOBA mean |change| .0001 (max .0035), WAR .0014 (max .05), P(WAR>0) .0004 (max .018).  Largest: Valerio .161 -> .084 (32 FL PA, MiLB K% 10.8 points below the league), Wright .136 -> .086, Abbott .300 -> .346.  New output columns k_bridge_dev, k_bridge_term; proj_k_reg stays the history regression alone.  Backups: output/_backup_before_k_bridge_prior/, _code_backup_before_review_fixes_20261012/*_before_k_bridge.py.  Published.
Dashboard: hitter player page K% inputs gain "MiLB bridge K% (vs league)" with what it adds; ui_helpers.percentile_of now drops missing values from the reference group (players without an input no longer deflate percentiles -- affects any input with gaps).  AppTest clean.
Pre-registered check: 2026 debut class's second seasons (2027 outcomes).

## 2026-10-13 -- HITTER MISSISSIPPI PARK EFFECT APPLIED POST-PROJECTION, SURFACE LEVEL NOT CENTRED, HR/PA BY AN HR SURFACE, K% / BB% FROM THE COMPONENT TABLE (user: "it should be applied post-projection to capture the MIS park effects, but with handedness and the xwOBA surface ... for hitters only"; answers: history neutralised with the surface incl. HR/PA, "the only other thing to do would be to apply the K% and BB% effects from the table"; level = the surface's own; handedness = inside the surface only)
Not a model target (the park effect cannot be validated as a projection target, user); the model is fit on park-neutral history and the Trustmark effect is added AFTER the projection for half of a hitter's PA.
- wOBA: full-park effect per PA = 0.671 x his surface shift (exit speed x launch angle x batting side; NOT centred: league mean -.0307 per ball in play = -.0206 per PA) + the wOBA consequence of the K% / BB% effects (+.0018 = BB_effect x .811 + (-K_effect - BB_effect) x .397 away wOBA per ball in play).  Average -.019 per PA at a full park, -.009 for half the games.
- HR/PA: NEW HR surface (`fl_hitter_park_hr_surface.py`; is_HR - xHR, same pooled design): Trustmark actual / expected HR .467; league mean -.0213 HR per ball in play (-56%) = -.0143 per PA (visiting-hitter fixed effects gave -.0141); hitter shift SD .0059 per ball in play.  Replaces the 0.446 HR factor for neutralising HR/PA history.
- K% / BB%: global constants from the visiting-hitter component table (`fl_park_kbb_neutral.py`): K -.0082, BB -.0035 per PA at a full park.  NOTE: t -0.9 / -0.4 -- the production shrink rule would set both to 0; applied unshrunk at the user's request.
- History neutralisation (fl_hitter_model_dev.load): wOBA (surface, now with the level), HR/PA (HR surface), K% and BB% (constants) for the Trustmark share of each season's PA.  Mississippi regulars 2026: K% .1575 -> .1617, BB% .1132 -> .1150, HR/PA .0131 -> .0193, wOBA .3621 -> .3696.  The history table still displays observed K% / BB% / HR (renamed raw columns).
- The two wOBA halves now agree: direct half adds back ~.019 per PA (full park), component half (K, BB, HR weights) ~.021 -- the 10-12 item-1 mismatch (component half restoring HR the park takes with no offset) is closed.  Overlap of the two surfaces: corr(wOBA shift, HR shift) .74 across 587 hitters (HR explains 55% of the wOBA shift; the rest is non-HR outcomes); the HR surface is used only for HR/PA, never added to the wOBA adjustment.
- Effect on the 2027 table: MIS wOBA shift mean -.0094 (SD .0038, range -.028 to +.004); park fit runs unchanged (corr 1.00 with before: relative to the average hitter); park-neutral wOBA mean +.0006 (Mississippi hitters +.0013, max .0027); K% Mississippi +.0022; WAR mean -.009 (park-neutral league wOBA .3737 -> .3747 raises the bar slightly), Mississippi hitters +.010.  New columns proj_k_mis / proj_bbr_mis / proj_hrr_mis.  WAR (MIS) = park-neutral WAR + park fit (relative), unchanged definition: the level is common to all hitters and to replacement.
- Code: fl_hitter_park.py (effect, effect_hr, kbb_woba, neutralise_rates, CENTRED flag for the old behaviour), fl_hitter_model_dev.py, fl_hitter_projection.py.  Backups: output/_backup_before_post_projection_park/, _code_backup_before_review_fixes_20261012/*_before_post_projection_park.py.  ROS snapshots rebuilt; published.  Dashboard: MIS wOBA Effect colours centred on the average hitter (not 0); column note mentions the ~.009 level.  AppTest: all views clean.
- Pitchers unchanged (global-only, 0).

## 2026-10-13 -- PRODUCTION READINESS (user: "what else before production?" -> items 1-12)
- **WAR = batting + positional adjustment + replacement (user, item 5).**  Fielding and baserunning computed and stored (fld_rate, bsr_rate) but out of projected WAR (flags WAR_INCLUDE_FIELDING / WAR_INCLUDE_BASERUNNING in fl_hitter_projection.py; same in the ROS hitter WAR).  2027 table: WAR mean change -.002, mean |change| .10, max 1.39 (strong OF / C defenders lose fielding credit); 80% range width 2.69 -> 2.44.  Dashboard: fielding / baserunning rows, the Pos * marker and the "fielding not measured" caption removed; WAR notes reworded.
- **Trustmark K% / BB% effects SHRUNK to 0 (user, item 6)** by the production rule (raw -.0082 / -.0035, t -0.9 / -0.4): park_K_EFFECT / BB_EFFECT now read the shrunk values; the K% / BB% wOBA term is 0; Mississippi shift mean -.0094 -> -.0103.
- **Name aliases (item 7):** fl_statscrew_pull.NAME_ALIASES (teag -> taeg gollert, jake -> jacob houtz, james -> jimmy gilleran; majority spellings).  fl_hitter_panel now aggregates by (name, season) before the joins (the aliases had created 4 x duplicated rows); panel 1,817 -> 1,815 rows, only the two merged hitters change.
- **One-command rebuild (item 1):** run_all.py (10 steps in dependency order, logs in output/run_logs/, publish MANIFEST of 20 files, then the dashboard tests).  First full run: all 10 steps OK (about 9 min), 42 tests passed; reproducible (pitcher outputs identical; hitter outputs identical except Mississippi shift |change| <= .0023 from the surface refit on alias-merged names).
- **Tests (item 4):** projections-dashboard/tests (test_data.py: files present, unique keys, rate ranges, WAR = bat + position + replacement, no position players in pitcher tables, ROS pitchers global-only; test_app.py: every view + edge-case player pages).  pytest installed.
- **Version control (item 1):** local git repos, branch main: projections-dashboard (app, data, tests, README with private Streamlit Cloud steps) and FLHitting (code, DECISION_LOG, run_all, config, prereg; outputs / inputs / backups / logs / secrets.toml excluded; credentials checked: none hard-coded).  Not pushed (needs the user's GitHub).
- **Season config (item 2):** fl_season_config.py (LATEST_SEASON 2026, TARGET_YEAR 2027, GAMES incl. 2027 = 102) wired into the hitter projection and hitter bridge; ROLL_FORWARD.md lists the remaining dated windows (pitcher pipeline ~140 references, panel queries, ROS dates, file names).
- **Pre-registration (item 12):** fl_2027_preregistered_checks.py + prereg/preregistered_2027.json, frozen 2026-10-13: C1 pitcher K% + fastball velocity (b = +.00543 K% per mph of shrunk velocity, fit ROS25 + ROS26), C2 hitter BB% + barrel% (b = +.2833 per unit barrel rate), C3 K% bridge prior confirmation, C4 thin-hitter survivorship, C5 pitcher P(WAR>0) calibration by projected IP, C6 hitter WAR range coverage, C7 baserunning persistence, C8 Trustmark surfaces, C9 fielding / position with two tracked seasons.
- **Data (items 8, 9):** FanGraphs college API is behind a Cloudflare challenge (403 to scripts) -> not scraped; user exported one CSV (qualified hitters, one season, no season column) saved to input/fangraphs_college/; need one export per season 2022-2026 with minimum PA 0.  Pioneer 2026 hitting: the missing team is the BOISE HAWKS (11 pasted tables matched one-to-one to the other 11 teams via players in the team-labelled pitching table).

## 2026-10-13 -- SHOULD THE PRIOR LEVEL DEPEND ON PLAYING TIME? (user: "in which places is a different (probably lower) level better? The players with the most playing time are usually the best and can bias the average")  (`fl_pt_prior_test.py`, output/pt_prior_test.csv)
Test: one extra parameter per model, prior mean = league + gamma x (log(1 + history) - mean), fit out of sample (hitters: leave-one-transition-out over 2022->23 ... 2025->26, next-season PA >= 20, n0 refit; pitchers K% / BB%: frozen model refit + log(1 + BF), fold 2; RA9 2025 -> 2026: 10-fold cross-fit), scored level-free under PRODUCTION weights (grow with the target's playing time) and PER PLAYER (equal weights, gamma refit with equal weights).
- HITTER wOBA: the sign depends on the weighting.  Production weights: gamma -.0033 per log unit in every fold (thin hitters' prior HIGHER, ~+.01 wOBA for a 20-PA history): RMSE -.00006 (P .71), MAE -.00027 (P .99); history < 150: RMSE -.00033 (P .89), MAE -.00069 (P .99).  Per player: gamma +.005 to +.009 (thin prior LOWER) but held out it is WORSE for thin hitters (RMSE +.00092, P(better) .03; MAE P .02).  This is the 10-08 survivorship finding again (PA-weighted +.011, unweighted -.033): thin hitters who go on to play a lot hit well.
- HITTER K%, HR/PA: no effect (all |dRMSE| <= .0001).  HITTER BB%: production-weighted worse for thin hitters (P(better) .02).
- PITCHER K%, BB%: z coefficients t +0.7 / -1.2; nothing significant (per-player K% P .92 for a -.0002 change).  RA9: gamma -.08 / -.12 runs/9 per log BF (thin arms worse, the known short-arm pattern) but no held-out gain (P .66 / .59; 150-399 group worse).
READING: there is no place where a LOWER prior level for thin players improves the projections out of sample.  The league priors are playing-time weighted and so are the targets the WAR uses; thin players who keep playing are the good ones (selection on the outcome), which offsets the "regulars inflate the average" effect.  For rookies the bridges already use the debut-class mean, not the league.  The one consistent signal (thin hitters' wOBA prior slightly HIGHER under the production target) is the survivorship candidate already pre-registered for 2027 (C4).  NO CHANGE.

## 2026-10-13 -- COLLEGE -> FL HITTER BRIDGE (user supplied FanGraphs college batting exports 2022-2026, minimum PA 0: input/fangraphs_college/)  (`fl_hitter_bridge_college.py`; validation log output/hitter_college_bridge_validation_log.txt, CSV output/hitter_college_bridge_validation.csv)
Data: 26,096 college hitter-seasons (about 5,400 per season); names with > 1 FanGraphs PlayerId dropped; conference from the college pitching table's team map (100% matched).  Pairs: 274 FL debut hitters 2023-26 with college seasons before the debut (31 / 67 / 85 / 91 by class; 237 college only, 37 also MiLB).
Design: mirrors the MiLB hitter bridge; fork NOT decided a priori: basis raw (vs all-college season mean) / conf (vs conference-season mean) / conf+ (conf + conference-strength feature), x n0 x decay chosen by leave-one-year-out INSIDE each leave-one-debut-class-out training set (honest out-of-fold predictions).  Chosen: K% mostly raw, BB% mostly conf, HR/PA mostly conf+, wOBA unstable (conf+ / raw).
Validation (level-free, frozen weights, player bootstrap) vs the debut-class mean / vs the raw latest college line:  K% .0667 -> .0554 (P 1.00) / P 1.00;  BB% .0392 -> .0365 (P 1.00) / P 1.00;  HR/PA .0168 -> .0155 (P .997) / P 1.00;  wOBA (0.5 direct + 0.5 built) .0537 -> .0528 (P .87) but MAE .0400 -> .0405 (P(better) .25) / vs raw P 1.00;  wOBA direct only P .83 (MAE .21); built only P .68 (MAE .28).  College-only hitters: same pattern (K% / BB% / HR P .96-1.00; wOBA P .74, MAE .15).
Decision: build with the 50/50 wOBA (as the MiLB bridge and the main model); the dashboard states that college wOBA is close to the league average (no better than average on MAE) and that the signal is in K% / BB% / HR/PA.  Final fit on all pairs: wOBA conf+ n0 200 decay .5; K% raw n0 50 decay .5 (slope .84); BB% conf n0 50 decay 1; HR/PA conf+ n0 100 decay 1.
Output: output/fl_hitter_college_prospects.parquet (7,346 college hitters with a 2025 or 2026 season; projected wOBA mean .356 vs FL league .375; 10% beyond validated range; 112 already in FL; 401 also MiLB), output/fl_hitter_college_history.parquet, "college" section in fl_hitter_bridge_coefs.json.  Dashboard: Hitters > Rookie / MiLB Lookup has a College source; tests 47 passed.  run_all.py: new step 10 (college bridge) and two MANIFEST files.
NOT done (candidates): the college bridge as the K% prior for thin-history FL hitters without a MiLB record (like the MiLB K% prior) -- needs its own exhaustive validation; a combined MiLB + college prior for hitters with both.

## 2026-10-13 -- wOBA MODEL-CHANGE REPORT FOR THE K% BRIDGE PRIOR, components rebuilt from the UPDATED K% (user: "make sure it uses the updated K% and BB% projections ... when comparing to the previous model")  (`fl_hitter_k_bridge_woba_check.py`, output/hitter_k_bridge_woba_check.csv)
Production check: the published woba_comp equals lg + sum(beta x (final proj - lg)) built from the FINAL K% (bridge prior + Whiff%), BB% (+ Swing%) and HR/PA (max difference 6e-17).
Comparison on the 56 validation hitter-transitions, OLD = 50/50 wOBA with components from the league-prior K%, NEW = components from the bridge-prior K% (direct half, BB%, HR/PA unchanged; all regressions LOTO; production component weights): RMSE .04022 -> .04010 (-.00011 [-.00026, +.00004], P(better) .89), MAE .03021 -> .03014 (P .68); by transition P .66 / .94 / .68.  FULL production path 2025 -> 2026 (Whiff% / Swing% / xwOBA / EV90 terms on both sides): .03617 -> .03598 (-.00019, P .90), MAE P .76.  wOBA change mean .0008 (max .0041).  Reading: the K% prior's wOBA consequence is small and slightly favourable; the change stands.
Rule recorded (user): every future wOBA comparison against a previous model rebuilds the component half from the updated K% / BB% / HR/PA projections of the new model.

## 2026-10-14 -- PIONEER HITTING: Boise Hawks added (all 12 teams) and 2021-25 from StatsCrew; the Pioneer bridge VALIDATED (user: Boise table pasted; "Use Statscrew for pre-2026 Pioneer hitting")
- 2026: Boise Hawks pasted 2026-10-13 -> block 12 in output/pioneer_2026_raw/pioneer_2026_br_hitting.csv; blocks 1-11 named via players in the team-labelled pitching table (one-to-one); 292 hitters (166 with >= 100 PA).  Boise's own totals row (4,183 PA) is 208 PA below the sum of its 27 player rows (4,391) -- a source mismatch like 7 of the other 11 tables; rows used as pasted.  Pioneer 2026 league: wOBA proxy .411, K% .174, BB% .133, HR/PA .036.
- 2021-25: `pull_statscrew_pioneer_batting.py` (league pages l-PION/y-{year} -> team pages; robots.txt Crawl-Delay 5 respected; cached in output/statscrew_pages_pioneer/): 1,341 team stints, 8 / 10 / 10 / 12 / 12 teams, PA complete; 1,503 Pioneer hitter-seasons 2021-26 after summing stints.  Independent era only (MLB rookie league before 2021).
- VALIDATION (`fl_hitter_bridge_pioneer_fit.py`, 81 FL debut hitters 2023-26 with Pioneer history, leave-one-debut-class-out, n0 / decay chosen inside the folds, level-free, frozen weights, bootstrap; wOBA components rebuilt from each arm's own K% / BB% / HR): a Pioneer-SPECIFIC fit beats the league mean (wOBA .0503 -> .0478 P .93; K% P .98; BB% P .96; HR P .92) and the raw Pioneer line (P .99-1.00) but does NOT beat the existing method (the MiLB mapping applied to Pioneer deviations): K% P .48, BB% .50, HR .37, wOBA .07 (MiLB mapping .0474 vs fit .0478).  DECISION: keep the MiLB mapping (fit on 200 pairs vs 81), now applied to all Pioneer seasons 2021-26 (was 2026 only) -- exactly the validated arm; "provisional" removed.
- Output: 454 Pioneer hitters with a 2025 or 2026 season (241 with >= 150 PA; projected wOBA mean .378, wRC+ 83-131; 22% beyond validated range); history file now multi-season with teams.  Dashboard source renamed "Pioneer League" with a validation note; 47 tests passed.

## 2026-10-14 -- 2026 INFIELD FIELDING METRIC ON THE SIS PART FRAMEWORK; WAR rule NOT met (user: https://library.fangraphs.com/defensive-runs-saved-2020-update/ "Can we use this framework to build a 2026 fielding metric for infielders?" + "if reliability / stickiness are reasonable ... add a shrunken component to WAR")  (`fl_fielding_part_2026.py`; outputs output/fielding_part_2026_{plays,players,teams}.parquet, summary json)
Data reality first: 2025 has NO fielding data for anyone but catchers / pitchers (war_playing_time positions C / P only; no fielded-by, no fielder names / positions in tracking; StatsCrew pages have no fielding tables) -> 2025 infield fielding and positions cannot be rebuilt from our sources.  2026 has: innings at every position, pbp_fielded_by (47,269 balls), fielder names + starting coordinates on 95-96% of pitches.
PART mapping: A (ball only) and B (ball + each infielder's starting angle / depth) feasible; C (fielder's touch time / throw) NOT -> Range + Throwing combined; Air (infield-range balls, LA >= 10, <= 180 ft) separate; multiple fielders credited / debited on every play (all starting spots known); positioning (B - A) credited to the TEAM (user default).  Join pbp_fielded_by -> tracking by game / inning / half / batter / order: 90% of balls in play; outcomes agree.  Models: gradient boosting per position x set, 5-fold cross-fitted BY GAME; a pitcher / catcher out term so B closes; runs = plays x (non-out value - out value), FL run values (.96 GSL, .97 AIR); centred within position.  Batter-speed proxy (shrunk SB attempt rate): no gain in held-out log loss (.4648 vs .4648) -> dropped.
RESULTS: per-position runs sum to 0; regular leaders / laggards plausible (Craig SS +12.8 in 225 opportunities; Schield SS -14.9).  RELIABILITY (alternating games, runs per opportunity, >= 15 opportunities per half; n0 from split halves; evaluated at a full-time season = 90th percentile of opportunities at the position, the baserunning convention): all infielders split-half r .163 (90% -.003 to .324), n0 228, full-time reliability .30; by position SS r .365 (.09-.60) -> .62; 2B .144 -> .26; 3B .091 -> .16; 1B .039 -> .07.  Note: the first printout applied Spearman-Brown at the average (part-time) fielder (.28); the full-time figure is the correct comparison with the baserunning precedent.
PRE-STATED RULE (reliability >= .40 for infielders) NOT MET -> fielding stays OUT of WAR.  Shortstop alone clears it but on 28 fielders with a wide interval and chosen after seeing the result -> not adopted (forking).  Team positioning still biased low (mean about -13 runs per team even with the P / C term: B undershoots A) -> team numbers are diagnostic only.
NEXT: this exact metric (frozen spec) added to the 2027 pre-registration (C9): two seasons double the opportunities and allow the year-to-year test; if it passes, a shrunk term enters WAR.  Optional: show as a labelled scouting stat (not in WAR) -- user's call.

### 2026-10-14 -- Infield RANGE-only component: stability check (exploratory; user: "How stable would just a Range component be for infielders?")
SPLIT (fl_fielding_range_split_2026.py): pbp fielded_by credits infield hits to the infielder who reached the ball (e.g. 441 SS singles), so RANGE = reached the grounder (fielded_by == position) vs P(reach | ball + his starting spot), cross-fitted by game; THROW = out given reached vs expected.  Same reliability method as the PART metric (alternating games, >= 15 opportunities per half, full-time = 90th percentile).
RESULTS (all infielders, full-time reliability): current metric (range+throw, grounders + infield air) .30 (reproduced); range+throw grounders only (UZR's rule) .30; RANGE ONLY .37 (split-half r .21, 90% .05-.37, n0 162 vs full-time 94); THROW ONLY .17.  By position, range: SS .68, 3B .60, 2B .17, 1B 0 (13-28 fielders each; wide intervals).
DECISION: not adopted for WAR -- below .40 and chosen after seeing the data (forking).  The pattern (range steadier than throwing, which mostly adds noise) is a candidate pre-registered alternative for 2027: two seasons of range at n0 162 would give about .54.

### 2026-10-14 -- Hitter WAR fielding = positional adjustment only; baserunning by position and by height (fl_baserunning_position_height.py)
WAR: user "let's have fielding WAR be entirely just the position adjustment" -> already the production setting (WAR_INCLUDE_FIELDING = False; WAR = batting + position + replacement).  No code change.
DATA LIMITS: baserunning runs exist for 2026 only (2025 column is all zeros = placeholder); SB / CS only 2022-24 in the panel; positions only 2025-26 (the steal-by-position check uses players' later position: 78 players).  Height / weight / age from pbp_players (fallback rosters), 95% of 2026 PA matched.
POSITION (2026 runs per 600 PA vs league, 90% player bootstrap): CF +5.6 (3.7 to 7.2), SS +2.3, RF +1.0, 2B 0.0, LF -0.9, 3B -1.9, 1B -1.9, C -3.3 (-5.2 to -1.6); permutation p .0004.  Independent check, 2022-24 steal attempts by later position: same ordering (CF +23, 2B +12 attempts per 600 PA; C -21, 1B -17), p .0002.  -> Baserunning DOES differ by position.
HEIGHT: unadjusted, shorter players run better (2026 runs -0.35 per 600 PA per inch, 90% -0.78 to +0.09, permutation p .12; steal attempts -4.0 per inch, p .001).  But it vanishes once weight (and age) are in: height +0.24 (-0.34 to +0.76), while weight -1.0 runs per 600 PA per 10 lb (-1.6 to -0.4); with position too, weight -0.6 (-1.2 to -0.02).  -> Height itself does not matter beyond weight; heavier players run worse, partly through position.
NOT YET A MODEL CHANGE (baserunning stays out of WAR); options put to the user.
REVISION (same day, user: "use bsr_runs from the war_players table for this analysis. that is all I care about."): rerun on war_players alone (reg_season 2026, its own pa / bsr_runs / primary_pos;
steal-attempt checks removed; 24 hitters with no primary_pos excluded from the position test).  POSITION: CF +5.6 (3.7 to 7.4), SS +2.7 (0.1 to 5.3), RF +0.8, 2B -0.2, LF -0.8, 1B -2.0, 3B -2.0,
C -3.3 (-5.1 to -1.4) runs per 600 PA; permutation p .0002.  HEIGHT: alone -0.35 per inch (-0.78 to +0.06, p .10); with weight + age +0.23 (-0.28 to +0.75) while weight -1.0 per 10 lb
(-1.5 to -0.4); with position as well, weight -0.56 (-1.15 to +0.02).  Conclusions unchanged: position matters; height does not beyond weight.

### 2026-10-14 -- Baserunning proxy for hitter WAR (position + DH + height + weight): built and validated, NOT yet switched on (fl_bsr_proxy.py, fl_bsr_proxy_validation.py)
REQUEST: "use position, height, and weight as the proxies for baserunning WAR"; then "DHs probably have worse baserunning. It's important to include them".
DH: war_players.primary_pos never says DH (regular DHs are filed at 1B / RF / LF, cameo DHs have no position).  2026 baserunning by share of PA at DH: <= 10% +0.8, 10-30% -0.2,
30-50% +0.8, 50-70% -5.3, > 70% -6.3 runs per 600 PA -> DH = >= 50% of PA at DH (dh_pa; 2026 only -- 2025 dh_pa and pos_innings are broken, pbp_events starts 2026).
PROXY: WLS of 2026 bsr_runs per PA (weights PA) on 9 position levels incl. DH + height + weight; 377 hitters.  Effects per 600 PA: CF +5.5, SS +2.4, RF +0.7, 2B +0.2, 1B 0.0,
LF -0.5, 3B -1.7, C -3.2, DH -5.3; height +0.04 per inch, weight -0.28 per 10 lb (1B's earlier -1.4 was mostly DHs filed at 1B).  Residual true-talent SD 3.95 runs / 600 PA.
(a) 2026 bsr per PA, 10-fold CV by player x 20 repeats, frozen inverse-variance weights (n0_eval 378 PA), runs per 600 PA:
    league mean -> position:                 RMSE 9.549 -> 9.291 (-0.258, 90% [-0.497, -0.015], P .96); MAE 6.671 -> 6.519 (-0.152, [-0.411, +0.121], P .83)
    league mean -> height + weight:          RMSE -0.052 (P .73); MAE -0.042 (P .68)
    league mean -> position + height + weight: RMSE -0.196 (P .92); MAE -0.103 (P .74)
    position -> position + height + weight:  RMSE +0.062 (90% [+0.016, +0.117], P(better) .01); MAE +0.049 ([+0.008, +0.097], P .02)  -> height / weight HURT out of sample once position (incl. DH) is known.
(b) WAR backtest -> 2026 (146 hitters, realised WAR incl. baserunning, realised-PA weights; cross-fitted proxy):
    position + height + weight: RMSE 1.3740 -> 1.3494 (-0.0246, [-0.0458, -0.0054], P .99); MAE 1.1244 -> 1.0994 (-0.0250, [-0.0447, -0.0055], P .98)
    position only:              RMSE 1.3740 -> 1.3476 (-0.0264, [-0.0465, -0.0069], P .99); MAE 1.1244 -> 1.0983 (-0.0261, [-0.0452, -0.0066], P .99)
    DH mean error +0.81 -> +0.56 WAR, CF -0.38 -> -0.13, C +0.74 -> +0.63.
CAVEAT: the backtest's "2025" positions are the 2026 positions copied back (lineup_stats has no real 2025 positions), so role is target-season in old and new alike.
DECISION PENDING (user): baserunning proxy helps WAR either way; position-only (incl. DH) is as good or better than adding height + weight, which overfit.
DECISION (user 2026-10-14, chose "Position + DH only (Recommended)"): hitter WAR = batting + position adjustment + BASERUNNING PROXY + replacement.  bsr_rate = the 2026 position
average incl. DH (fl_bsr_proxy.PRODUCTION_FEATURES = ("pos",)); a hitter's own baserunning record is not used.  Position for the proxy: 2026 role (DH if >= 50% of PA at DH),
else the projection's primary_pos; no position -> PA-weighted average position (about 0).  WAR simulation: baserunning runs = rate x PA + sqrt(PA^2 tau2 + PA noise) x z, with
tau2 / noise the residual variance law around the proxy (true-talent spread it leaves + one season's noise).  ROS WAR uses the same term (the proxy is fit on full-season 2026,
so in-season 2026 ROS snapshots see it -- display only).  Dashboard: WAR parts table gains a Baserunning row; WAR notes updated.  Replaces the unused shrunk-own-bsr code
(BSR_N0 / BSR_TAU removed).

### 2026-10-14 -- Baserunning proxy weighted by POSITION PLAYING TIME? (user: "Does weighting by position playing time help?") -> NO, keep the primary position (fl_bsr_proxy_ptweight_test.py)
CANDIDATE: share_DH = dh_pa / pa, the rest split by 2026 innings per position (war_playing_time, pitcher innings dropped); bsr / PA = sum share x effect (WLS, PA weights).  327 of 377
hitters (99% of PA) have time at 2+ spots incl. DH; median main-position share .79.  Mix effects are more extreme (CF +7.4, RF +4.4, DH -7.1 vs primary +5.8, +0.6, -5.8).
PRE-STATED RULE: adopt only if held-out bsr RMSE improves vs the primary position with P(better) >= .8 AND the WAR backtest is not worse.
(a) held-out 2026 bsr (10-fold x 20, frozen inverse-variance weights), runs per 600 PA: RMSE 9.291 -> 9.300 (+0.009, 90% [-0.116, +0.135], P(better) .44); MAE 6.519 -> 6.527 (+0.008, P .45).
(b) WAR backtest (146): RMSE 1.3476 -> 1.3415 (-0.0062, [-0.0150, +0.0021], P .89); MAE 1.0983 -> 1.0957 (-0.0026, [-0.0114, +0.0061], P .68).
DECISION: rule NOT met at (a) -> production stays on the primary position (incl. DH >= 50%).  The small WAR gain in (b) is within noise.
SHIPPED (same day): rebuilt (run_all --from 7) and published.  Projected WAR effect of the baserunning term over projected PA (mean, range): CF +0.20 (+0.14 to +0.32), SS +0.10, 2B / RF +0.02,
1B / LF -0.01 to -0.02, 3B -0.06, C -0.10, DH -0.17 (-0.30 to -0.12); hitters with no 2026 position 0.  Dashboard player pages (user: "update the player pages with all meaningful components of their projections"): a "How it adds up" table on both pages.
  Hitters: K% (regressed + Whiff% + MiLB bridge), BB% (regressed + Swing%), HR/PA, wOBA (direct, from components, average, + xwOBA, + EV90, + Mississippi), PA and chance he
  plays; WAR parts in runs / WAR / WAR per 400 (batting, position, baserunning with the position used, replacement, total, Mississippi park fit).  The hitter file now keeps
  proj_woba_xw so the wOBA steps sum.
  Pitchers: K% / BB% / RA9 each from the average pitcher plus coefficient x (his input - average); WAR from replacement RA9, runs saved over projected IP.  RA9 coefficients
  are exported (output/fl_ra9_model_coefs.json, in the publish MANIFEST).  tests/test_data.py checks every step sums to the projection (49 tests pass).

### 2026-10-14 -- Position adjustment from 2026 only, DH as its own group; 2025 published WAR hidden (fl_pos_dh_group_test.py)
FINDING: war_players 2025 position adjustments are broken (dh_pa puts 89% of PA at DH, pos_innings ~0.22 per PA vs 1.77 in 2026; pos_adj_runs per 600 PA: most hitters -2.8,
a few +21; only 83 rows with a position).  Production never read them (2027 projections use 2026), but the backtest harness (build_shifted: real 2025 = latest season) did,
and the dashboard showed 2025 published WAR.
USER: "Don't show 2025 Published WAR" / "Use 2026 only for adjustments" / "DHs?".
CHANGES: (1) dashboard history: published WAR shown for 2026 only.  (2) Position adjustment = own 2026 war_players pos_adj_runs per PA shrunk (POS_N0 100) toward his
2026 group; production, ROS and the backtests all read this one table (fl_bsr_proxy.positions(2026)), so backtests now see 2026 roles (accepted).  (3) DH = its own group
(>= 50% of 2026 PA at DH, the baserunning rule; POS_DH_GROUP); DH-only cameos, previously 0, now get the DH rate (about -11 runs / 600 PA).
REPORT (WAR backtest -> 2026, 146 hitters, both with the 2026 table): listed-position group -> DH group: RMSE 1.3188 -> 1.3186 (-0.0002, 90% [-0.0035, +0.0037], P .56);
MAE 1.0643 -> 1.0637 (-0.0006, [-0.0043, +0.0033], P .60); DH mean error +0.21 -> +0.11 WAR.  Neutral overall, less DH bias -> kept.  Backtest level with 2026 adjustments
1.319 / 1.064 vs 1.348 / 1.098 with the broken 2025 ones (partly role leak: the target season's positions).
"Fielding runs position adjustment": the positional adjustment is (and stayed) in WAR; only individual fielding runs are out.

### 2026-10-14 -- How the position adjustment is built (reverse-engineered) and pre-registration amendments
POSITION RUNS (war_players, built upstream; no builder code in this repo): fitted on 2026 rows, R2 .995, max miss 1.6 runs: FanGraphs constants x innings_scaled / 1458
(C +12.7, SS +7.6, 2B +2.6, 3B +2.5, CF +2.4, LF -7.7, RF -7.5, 1B -12.9; scale 1.016) + DH -17.2 per 600 DH PA + a uniform +2.8 runs / 600 PA shift so the league sums to 0.
innings_scaled corrects the ~8% tracking-coverage gap of non-catcher innings.  Constants are MLB conventions, not FL estimates (no infield fielding data to estimate them).
NO MODEL CHANGE NEEDED from this: baserunning (position average) and position runs measure different things; replacement per PA + zero-sum position runs is the FanGraphs frame.
STILL UNVALIDATED (known since 2026-10-08): position shrinkage n0 = 100 PA and the 6.0 runs / 600 PA residual -- one season of real positions.
PRE-REGISTRATION AMENDED (before any 2027 data; frozen coefficients untouched): C6 WAR definition now includes the baserunning proxy; C7 rewritten to confirm the proxy
(vs league mean; vs own 2026 rate shrunk toward the proxy, n0 300); C9 gains the position-runs n0 / residual test.  prereg/preregistered_2027.json carries an 'amendments' entry.

### 2026-10-14 -- Position-runs shrinkage estimated within 2026 (fl_pos_shrink_2026.py) -- decision pending
METHOD: per-game position runs rebuilt from pbp_events (outs credited to each moment's C / 1B / 2B / 3B / SS / LF / CF / RF; DH PA = games never in his team's alignment),
FanGraphs constants; check vs war_players.pos_adj_runs (266 hitters >= 50 PA): scale 1.021, DH -15.8 per 600, R2 .985, max miss 1.9 runs.  First-half own rate / group
(DH >= 50% PA) predict second-half position runs per PA (>= 50 PA each half); frozen inverse-variance weights; n0 grid; player bootstrap.
CHRONOLOGICAL halves (split 2026-07-12, 178 hitters; group changed for 22%): RMSE per 600 PA n0 0 2.77, 25 2.81, 50 2.86, 100 2.95, 200 3.09, group only 3.58; best n0 0
(90% 0-20).  n0 100 -> 0: RMSE 2.954 -> 2.770 (-0.184, [-0.318, -0.051], P .99); MAE 2.003 -> 1.913 (-0.090, [-0.197, +0.016], P .92).  Role-change spread 2.7 runs / 600 PA
(simulation uses 6.0).
ALTERNATING games (236; pure sampling): best n0 20 (10-30); n0 100 -> 20 RMSE 1.600 -> 1.370, MAE 1.064 -> 0.880 (P 1.0); spread 0.6.
CAVEAT: between-season role change (signings, trades, aging) is larger than within-season, so these n0 / spread values are LOWER bounds for a season-ahead projection.
IMPACT on 2027 projections: n0 100 -> 0: mean |dWAR| .033 (max .27); -> 25: .015 (max .10); -> 50: .008 (max .05).
DECISION (user 2026-10-14): POS_N0 100 -> 25; residual 6.0 kept.  Rest of season now uses the same own-rate blend (was group only).  WAR backtest (146; own role = target season,
flatters low n0): RMSE 1.3186 -> 1.3179 (-0.0007, [-0.0029, +0.0013], P .71); MAE 1.0637 -> 1.0634 (-0.0003, [-0.0024, +0.0015], P .62).  C9 text updated to n0 = 25.
Rebuilt (run_all --from 7), published, 49 tests pass.

### 2026-10-20 -- Stale tracking file refreshed; park surfaces / hitter ROS rebuilt; infield fielding metric RE-RUN on the full season
FINDING: 2025_2026_pitches.parquet (hitting tracking, last pulled 2026-08-20) ended 2026-08-16 (757 games); the database has 2026 through 09-26 (regular season ends 09-06, 939 games; playoffs after).
About 200 regular-season games (08-17 to 09-06) were missing.  Re-pulled with pull_2025_2026.py: 978 games / 294,877 rows (= database).  The file has no game_type column and includes
playoff (23 games, 2.4% of 2026 pitches) and spring-training games, as it always has (2025 already contained playoffs) -- left as is; restricting the park surfaces to regular season
would be a model change (open question to the user).
NOT STALE (verified row-for-row vs the database): output/fl_raw_pitches_flagged.parquet (pitcher cache; reg_season only, so it correctly ends 09-06), output/hitter_bbe_2025_2026.parquet
(EV90), output/hitter_pitches_2025_2026.parquet (Whiff% / Swing%).  Import trace: run_all steps 1-4 (panel, position flags, pitcher K/BB, RA9) never reach the refreshed file.
Readers that matter: park surfaces (steps 5-6, via fl_hitter_park_surface.py), fl_fielding_part_2026.py and the ROS build (via fl_error_analysis_common).  An earlier statement that the
hitter Whiff / Swing / EV90 and pitcher inputs were affected was wrong.
REBUILD (run_all --from 5; old files in output/stale_data_backup_20261020/): pitcher outputs not rebuilt (inputs unchanged).  Hitter 2027 projections vs before: K% and BB% identical
(park K / BB effects fixed at 0); park-neutral wOBA mean |change| .00008 (max .0008), HR/PA max .001, WAR mean |change| .0016 (max .012), proj PA max 2 (history is neutralised with the
HR surface, so a refit surface nudges park-neutral numbers); Mississippi: wOBA shift mean change +.0014, park-fit runs mean |change| .14 (max .95, Spearman .95), MIS WAR mean |change| .011 (max .082).
ROS snapshots changed slightly and evenly across as-of dates (ros_ra9 mean |change| about .012, wRC+ .003); the build is deterministic (run twice, identical) so the change is from the data;
cause not isolated.  Published to the local dashboard data folder, 49 tests pass.
FIELDING RE-RUN (same frozen spec, full data: 21,712 balls in play joined vs about 14.9k grounders + air before): all-infielder split-half r .163 -> .236 (90% .09-.38); full-time reliability .297 -> .410
(rule >= .40: NOT met -> MET, by .01); by position 1B .41, 2B .20, 3B .46, SS .56 (n 22-33 each).  Range-only .367 -> .493 (r .305, 90% .15-.46; lower bound about .28); range + throw grounders only .398;
throw only .05.  Team positioning bias unchanged (mean -14.3 runs per team).  The earlier "not met" and "range-only .37" came from incomplete data.
DECISION: fielding stays OUT of WAR (user 2026-10-13: keep fielding out; the pre-registered decision point is 2026 -> 2027 persistence, C9).  The rule is met only narrowly and a within-season split half
does not test year-to-year stability or validity (team positioning bias unresolved).  Range-only is now the stronger candidate (.49).  Options put to the user.
