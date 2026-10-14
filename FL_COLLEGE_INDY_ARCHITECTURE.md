# Frontier League / College / Indy-League Projection — Architecture

Living design doc for the FL pitcher K-BB% and hitter xwOBA projection system,
covering how college and other-indy-league data feeds in as priors. Companion
to `DECISION_LOG.md` (which records *why* each choice here was made, and what
alternatives were considered) — this file is the current-state shape of the
system, that file is the reasoning trail that produced it.

Status: design only. Nothing in this document has been implemented yet.

## 1. Data sources

| Source | Covers | Status |
|---|---|---|
| Supabase `pitches` table | FL pitch-level Trackman, 2025-2026 | Live, in production. Includes an iScore/play-by-play fallback (`row_source='pbp_synth'`) for games without full tracking, already reconciled via `pbp_games`/`pbp_join_state`. |
| `pio_overlap_batting` | FL <-> Pioneer League batting bridge, 2021-2025 | Live. Batting only; built by an external tool not in this repo. Team codes (`bm10448`-style) match StatsCrew's own scheme, so it likely already sources from there. |
| StatsCrew.com (`/minorbaseball/`) | FL/Pioneer box scores back to 1993 (full BF/IP/BB/SO for pitchers, PA/BB/SO for batters); a per-player page giving a player's COMPLETE pro career across both independent AND affiliated MiLB levels in one table; bio (DOB, height, weight, bats/throws) | Validated: permissive robots.txt (5s crawl delay, only `/ads/` blocked), no findable Terms of Service. Not yet pulled. |
| FanGraphs `/leaders/college` | Conference-filterable college leaderboards: `K%`/`BB%`/`K-BB%` (pitching), `BB%`/`K%`/`wOBA`/`wRC+` (batting) | Validated: permissive robots.txt, ToS silent on scraping (unlike Baseball-Reference), `pybaseball` precedent for scraping FanGraphs generally. Open issue: "qualified"-only default appears to hide most of the population (121 pitcher rows returned even with `qual=0` -- needs the right query params, unresolved). |
| NCAA's own stats site (stats.ncaa.org) | — | Ruled out: robots.txt disallows ALL crawling, blanket. |
| Baseball-Reference | — | Ruled out: ToS explicitly prohibits use "for purposes of training, fine-tuning, prompting, or instructing artificial intelligence models" -- directly on point for this project. |
| Pioneer League's own site (pioneerleague.com) | — | Unreachable from this sandbox (CloudFront WAF 403, even via a real browser engine) -- not ruled out, just untested; may work from a different network. |

## 2. Schema (target: same Supabase database as `pitches`, for direct joins)

```sql
CREATE TABLE external_season_stats (
    id              bigserial PRIMARY KEY,
    player_id       text NOT NULL,       -- canonical id, from player_identity_xwalk
    league          text NOT NULL,       -- 'Frontier League', 'Pioneer League', 'NCAA', ...
    level           text NOT NULL,       -- 'Independent', 'College-D1', 'College-D2', ...
    year            int  NOT NULL,
    team            text,
    team_code       text,                -- source's own team id (audit/re-fetch)
    league_division text,                -- FL/Pioneer's OWN internal scheduling division
                                          -- (e.g. "Midwest Conference") -- NOT a quality signal,
                                          -- kept distinct from ncaa_conference on purpose
    ncaa_conference text REFERENCES ncaa_conferences(conference),  -- the quality-tier axis for college
    role            text NOT NULL,       -- 'batter' | 'pitcher'
    source          text NOT NULL,       -- 'statscrew', 'fangraphs_college', 'supabase_trackman', ...
    source_url      text,
    pulled_at       timestamptz NOT NULL DEFAULT now(),

    pa int, ab int, bb int, so int, ip numeric, bf int, h int, r int, er int, hr int,
    extra           jsonb,               -- everything else the source offers

    UNIQUE (player_id, league, year, team, role, source)
);

CREATE TABLE ncaa_conferences (
    conference      text PRIMARY KEY,
    division        text NOT NULL,       -- 'D1' | 'D2' | 'D3' | 'NAIA' | 'JUCO'
    notes           text
);

CREATE TABLE player_bio (
    player_id       text PRIMARY KEY REFERENCES player_identity_xwalk(player_id),
    birth_date      date,
    height_in       numeric,
    weight_lb       numeric,
    bats            text,
    throws          text
);

CREATE TABLE player_identity_xwalk (
    player_id       text PRIMARY KEY,
    display_name    text NOT NULL,
    aliases         jsonb NOT NULL      -- [{"source": "statscrew", "name": "...", "name_clean": "..."}, ...]
);
```

## 3. Identity resolution

1. Normalize (strip accents/punctuation, lowercase, drop Jr./Sr./II/III, reconcile
   "Last, First" vs. "First Last" ordering across sources).
2. Exact match on normalized name + a plausibility check (overlapping/adjacent
   years, same role) -> auto-accepted into the crosswalk.
3. Near-misses scored via `rapidfuzz` (token-sort / Jaro-Winkler); above a high
   threshold -> flagged `needs_review`, never auto-merged. Below it -> left
   unmatched.
4. Conservative bias, deliberately: an unmatched player is a recoverable loss of
   signal; a wrongly-merged pair silently corrupts a translation factor.

## 4. Ad hoc free-agent workflow (no pre-loading required)

- **Player with any pro history** (indy or affiliated MiLB): StatsCrew name
  search (Last/First, 2 requests: search -> per-player page) returns bio +
  entire cross-league, cross-level career table in one shot. Disambiguation
  (common names) via team/age/position context if the search returns multiple
  candidates.
- **True college-only free agent** (no pro history at all): FanGraphs college
  leaderboard/player-page path. Validated as a data source; the specific
  single-player ad hoc lookup mechanics are NOT yet designed -- least-finished
  piece of this workflow.

## 5. Feature engineering: conference-season-relative metrics

For each `(ncaa_conference, year, role)`, compute a PA/BF-weighted average
K%/BB% (min-sample floor TBD empirically). Each player's primary signal:

```
relative_metric = player_rate - conference_season_weighted_average
```

(plain difference, not ratio/z-score -- starting simple, per this project's
pattern of only adding complexity that's proven to earn its keep). Applied
identically to K%, BB%, and -- pending empirical validation against a held-out
check -- ISO/HR-AB, the one "result" stat judged likely to carry real
skill signal (bat-speed/swing-plane driven, not BABIP-luck driven). Raw
AVG/OBP/SLG-type results stats are demoted to secondary at most.

## 6. Reliability: hierarchical GLMM (scoped to this piece)

Two-level binomial logistic mixed model:

```
k_i ~ Binomial(n_i, p_i),   logit(p_i) = mu + alpha_{c(i)} + epsilon_i
alpha_c ~ N(0, tau_c^2)     (conference-season effect)
epsilon_i ~ N(0, tau_p^2)   (player effect within conference-season)
```

fit jointly (all variance components estimated simultaneously, not in a
separate method-of-moments pass) via `statsmodels.BinomialBayesMixedGLM`
(variational Bayes) as a starting point.

**Relationship to the rest of this project's reliability methodology**: under a
normal approximation this reduces exactly to the existing
`n/(n+n0)` shrinkage already used everywhere else (K%, BB%, `wh_sva`, the MiLB
bridge), with `n0 = sigma_within^2 / tau_p^2` -- the identical ratio
`fit_shrinkage_params` already computes. This is a generalization/unification,
not a competing paradigm: it estimates that same ratio jointly with everything
else, and gets a second level of shrinkage (the conference-season average's own
reliability) for free in the same fit, instead of needing a separate hand-tuned
minimum-sample rule.

**Assumptions** (see `DECISION_LOG.md` for the full enumeration and which ones
are checked vs. accepted-by-design):
- Constant true rate within a season -- **accepted by design**; in-season drift
  deliberately deferred to a separate future component (intrinsic-characteristic
  tracking, e.g. `wh_sva`/bat-tracking, mirroring the validated MLB
  Marcel + `wh_sva`-surprise split).
- Independent trials, normally-distributed random effects, additive
  (no cross-level interaction) -- standard GLMM assumptions, not yet stress-tested
  against real data.
- Homogeneous player-level variance across conferences -- **untested**, plausibly
  false (a stronger conference could have a tighter talent band).
- Enough conference-season groups to trust the top-level variance estimate --
  **untested** against real data; a synthetic check with only 8 groups showed
  variational Bayes visibly underestimating that variance (recovered SD 0.068
  vs. true 0.15), a known VB property that gets worse with fewer groups.
- The VB approximation itself (mean-field factorized posterior) is a distinct,
  additional assumption layered on top of the model's own -- worth validating
  against a full MCMC fit before trusting VB posteriors in production.

## 7. Not yet started

- Pioneer League pitching bridge (StatsCrew's per-player pages may make a
  dedicated bridge table unnecessary -- open question).
- FanGraphs "qualified-only" pagination fix.
- Real fuzzy-matching thresholds (proposed, untested).
- Any actual pull/scraper code for StatsCrew or FanGraphs.
- VB-vs-MCMC validation for the hierarchical model.
- The in-season intrinsic-characteristic drift component.
