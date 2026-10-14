"""
fl_hitter_park.py -- TrustmarkPark treatment for HITTERS.
SINCE 2026-10-13 (user: the park is applied POST-PROJECTION; history neutralised with the surface for wOBA and HR/PA; the surface's OWN level, not centred; handedness inside the
surface; K% / BB% Trustmark effects from the component table as global constants): per-PA full-park effects
  wOBA  = BIP_PER_PA x shift_bip_i (wOBA surface, NOT centred: the average hitter loses ~.021 at a full park) + the wOBA consequence of the K% / BB% effects
  HR/PA = BIP_PER_PA x shift_hr_bip_i (HR surface, fl_hitter_park_hr_surface.py: ~-56% of HR, -.0143 HR per PA on average)
  K%    = K_EFFECT (-.0082, visiting hitters, t -0.9: applied unshrunk at the user's request), BB% = BB_EFFECT (-.0035, t -0.4), same for every hitter (fl_park_kbb_neutral.py)
neutralise* remove share x effect from observed history; translate* add share x effect to a park-neutral projection (share 0.5 = a Mississippi hitter).
2026-10-10 .. 10-13 the surface was CENTRED (global level 0 from averaging the visiting-hitter and visiting-pitcher fixed-effects estimates; the pitcher half is contaminated by
Mississippi hitters' home K / BB edge, fl_park_component_factors.py) and HR/PA used the 0.446 HR factor.  SINCE 2026-10-10 the production translation is the batted-ball RESIDUAL SURFACE (PARK_MODEL = "surface": what the park does to a ball of a given exit speed / launch angle / side, wOBA - xwOBA, from visiting hitters + Mississippi hitters at home, averaged over the hitter's own batted-ball mix; contact-quality shifts are not park effects by definition; strikeouts / walks global).  From 2026-10-08 to 2026-10-10 it was the PROFILE model below (global Trustmark wOBA effect ~0 + hitter-profile terms: power, Air%, handedness;
fl_park_profile_final.py, output/trustmark_profile_coefs.json + output/hitter_park_profile.parquet); the original HR-factor translation is kept as neutralise_hr_model / translate_hr_model (reference; it counted HR suppression
and fewer hits per air ball but not the offsetting K% / single / BB effects, and was contradicted by the published park factor, a fixed-effects estimate and the 2026 out-of-sample outcomes: DECISION_LOG 2026-10-08).

ORIGINAL docstring (HR-factor model), built from the same constants as the pitcher translation
(fl_ra9_trustmark_projection): HR per air ball x TRUSTMARK_HR_FACTOR (0.446), non-HR air-ball hits 0.0073 lower,
both scaled by how many air balls the hitter hits.

Two directions, one park model:
  neutralise(df)   remove the park from a hitter's OBSERVED wOBA, using the share of his PA actually taken at Trustmark
                   (Mississippi hitters ~48%, visitors ~3%) and his observed air-ball / HR counts.
  translate(df, s) add the park to a PROJECTED park-neutral wOBA for a hitter who takes share s of his PA at Trustmark
                   (s = 0.5 for a Mississippi hitter), from his projected HR/PA and air balls per PA.
Everything is expressed in runs first (RUNS_PER_HR, AIR_HIT_RUN_VALUE, as on the pitcher side) and converted to wOBA
with the FL wOBA scale (woba_weights.woba_scale = 1.0175).

Other parks get NO treatment (same convention as the pitcher side: the focus is Mississippi).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_ra9_trustmark_projection as ra9

WOBA_SCALE = 1.0175
F = ra9.TRUSTMARK_HR_FACTOR            # HR per air ball, Trustmark vs league
DELTA = ra9.TRUSTMARK_HIT_RATE_DELTA   # non-HR air-ball hit rate, Trustmark minus league (full exposure)
RV_HR = ra9.RUNS_PER_HR
RV_AIR_HIT = ra9.AIR_HIT_RUN_VALUE


def neutralise_hr_model(df: pd.DataFrame) -> pd.Series:
    """Per-PA wOBA to ADD to an observed season to remove Trustmark. Needs pa, air_n, air_hr, pa_trustmark/pa_pitch (share)."""
    s = (df["pa_trustmark"] / df["pa_pitch"]).where(df["pa_pitch"] > 0, 0.0).fillna(0.0)
    keep = 1.0 - s * (1.0 - F)                                   # observed HR = neutral HR x keep
    extra_hr = df["air_hr"].fillna(0) * (1.0 / keep - 1.0)
    extra_runs = extra_hr * RV_HR + (df["air_n"].fillna(0) - df["air_hr"].fillna(0)) * s * DELTA * RV_AIR_HIT
    return (extra_runs / df["pa"].replace(0, np.nan) * WOBA_SCALE).fillna(0.0)


def translate_hr_model(hr_per_pa: pd.Series, air_per_pa: pd.Series, share: float = 0.5) -> pd.Series:
    """wOBA shift (negative) for a hitter taking `share` of his PA at Trustmark, from park-neutral projected rates."""
    runs = share * (hr_per_pa * (1.0 - F) * RV_HR + (air_per_pa - hr_per_pa).clip(lower=0) * DELTA * RV_AIR_HIT)
    return -runs * WOBA_SCALE


# ---------------------------------------------------------------------------------------------------------------------------------------
# PROFILE MODEL (production)
# ---------------------------------------------------------------------------------------------------------------------------------------
import json as _json
import os as _os

_COEF_PATH, _PROF_PATH = "output/trustmark_profile_coefs.json", "output/hitter_park_profile.parquet"
_cache = {}


def _load():
    if not _cache:
        base = _os.path.dirname(_os.path.abspath(__file__))
        js = _json.load(open(_os.path.join(base, _COEF_PATH))); prof = pd.read_parquet(_os.path.join(base, _PROF_PATH))
        _cache.update({"c": {k: v["final"] for k, v in js["coefs"].items()}, "ml": js["standardisation"]["mean_left"], "prof": prof})
    return _cache


PARK_MODEL = "surface"        # "surface" (production since 2026-10-10: batted-ball residual surface, fl_hitter_park_surface_final.py) or "profile" (the 2026-10-08 Air% / power / left model, kept reproducible)
_SURF = {}


def _load_surface():
    if not _SURF:
        base = _os.path.dirname(_os.path.abspath(__file__)); t = pd.read_parquet(_os.path.join(base, "output/hitter_park_surface_effect.parquet")).drop_duplicates("name_clean").set_index("name_clean"); js = _json.load(open(_os.path.join(base, "output/hitter_park_surface_coefs.json")))
        _SURF.update({"t": t, "mu": js["league_mean_effect_per_bip"], "k": js["bip_per_pa"], "g": js["global_total_effect_per_pa"]})
    return _SURF


def effect_sd(name_clean: pd.Series) -> pd.Series:
    """Standard deviation of the per-PA full-park effect (surface model: bootstrap SD of the hitter's shift; unknown hitters get the median)."""
    S = _load_surface(); sd = name_clean.map(S["t"]["shift_bip_sd"]); return (S["k"] * sd.fillna(S["t"]["shift_bip_sd"].median())).astype(float)


_HR = {}
_KBB = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "output/trustmark_kbb_effects.json"))) if _os.path.exists(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "output/trustmark_kbb_effects.json")) else {"k": {"raw": 0.0}, "bbr": {"raw": 0.0}}
K_EFFECT = float(_KBB["k"]["final"]); BB_EFFECT = float(_KBB["bbr"]["final"])      # per PA at a full park: visiting-hitter estimates SHRUNK by the production rule (user 2026-10-13; raw -.0082 / -.0035, t -0.9 / -0.4 -> 0)
CENTRED = False                                                                       # True = the 10-10 .. 10-13 behaviour (level 0)


def _load_hr():
    if not _HR:
        base = _os.path.dirname(_os.path.abspath(__file__)); t = pd.read_parquet(_os.path.join(base, "output/hitter_park_hr_surface_effect.parquet")).drop_duplicates("name_clean").set_index("name_clean")
        js = _json.load(open(_os.path.join(base, "output/hitter_park_hr_surface_coefs.json"))); _HR.update({"t": t, "mu": js["league_mean_hr_effect_per_bip"], "k": js["bip_per_pa"], "wcon": js["wobacon_away"], "wbb": js["woba_weight_bb"]})
    return _HR


def kbb_woba():
    """wOBA consequence (per PA, full park) of the global K% / BB% effects: fewer walks lose the walk weight, the PA they free up and the strikeouts avoided become balls in play
    worth the away league wOBA per ball in play (the surface then applies to those balls like any other)."""
    H = _load_hr(); return BB_EFFECT * H["wbb"] + (-K_EFFECT - BB_EFFECT) * H["wcon"]


def effect_hr(name_clean: pd.Series) -> pd.Series:
    """HR per PA at a full park (negative): balls in play per PA x his HR surface shift (unknown hitter = league mean)."""
    H = _load_hr(); return (H["k"] * name_clean.map(H["t"]["shift_hr_bip"]).fillna(H["mu"])).astype(float)


def effect(name_clean: pd.Series) -> pd.Series:
    """Full-park TrustmarkPark effect on a hitter's wOBA (points per PA taken at Trustmark).  SURFACE model: global total + balls-in-play-per-PA x (his mean effect over his own batted-ball mix, shrunk, minus the league mean); unknown hitter = the global effect.
    PROFILE model (PARK_MODEL = "profile"): global + Air% + power + left-handed terms (shrunk coefficients); unknown profile = average hitter."""
    if PARK_MODEL == "surface":
        S = _load_surface(); sh = name_clean.map(S["t"]["shift_bip"]).fillna(S["mu"])
        if CENTRED:
            return (S["g"] + S["k"] * (sh - S["mu"])).astype(float)
        return (S["k"] * sh + kbb_woba()).astype(float)
    L = _load(); c, ml, prof = L["c"], L["ml"], L["prof"]
    z_air = name_clean.map(prof["z_air"]).fillna(0.0); z_pow = name_clean.map(prof["z_power"]).fillna(0.0); left = name_clean.map(prof["left_share"]).fillna(ml)
    return c["global"] + c["air"] * z_air + c["power"] * z_pow + c["left"] * (left - ml)


def share(df: pd.DataFrame) -> pd.Series:
    return (df["pa_trustmark"] / df["pa_pitch"]).where(df["pa_pitch"] > 0, 0.0).fillna(0.0)


def neutralise_rates(df: pd.DataFrame) -> dict:
    """Additions that remove Trustmark from observed HR/PA, K%, BB% (share of his PA at Trustmark x minus the full-park effect)."""
    s = share(df); return {"hrr": (-s * effect_hr(df["name_clean"])).fillna(0.0), "k": -s * K_EFFECT, "bbr": -s * BB_EFFECT}


def neutralise(df: pd.DataFrame) -> pd.Series:
    """Per-PA wOBA to ADD to an observed season to remove Trustmark: -(share of his PA at Trustmark) x his profile effect.  Needs name_clean, pa_trustmark, pa_pitch."""
    s = (df["pa_trustmark"] / df["pa_pitch"]).where(df["pa_pitch"] > 0, 0.0).fillna(0.0)
    return (-s * effect(df["name_clean"])).fillna(0.0)


def translate(name_clean: pd.Series, share: float = 0.5) -> pd.Series:
    """wOBA shift for a hitter taking `share` of his PA at Trustmark (0.5 = a Mississippi hitter): share x his profile effect (negative = hurts)."""
    return share * effect(name_clean)
