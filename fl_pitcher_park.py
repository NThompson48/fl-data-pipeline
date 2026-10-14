"""
fl_pitcher_park.py -- TrustmarkPark translation for PITCHERS.  PRODUCTION SINCE 2026-10-10: GLOBAL-ONLY (user decision): every pitcher gets the same park effect, the global wOBA-allowed effect at Trustmark (visitors-only estimate, bootstrap-t
shrunk, ~0), converted to runs per 9 (x 4.512 BF per IP x 9 / 1.0175).  Each pitcher's shift has a standard deviation from the bootstrap SE of the global effect (so Mississippi intervals still carry the park uncertainty).
Why: the profile terms (HR-prone, K%, Air%, left; fl_park_pitcher_final.py) did not beat 'no profile' out of sample (fl_park_pitcher_fb25_model_test.py), K% is a non-batted-ball term that is closer to global and is contaminated by our
hitters' lower strikeout rate at home, and no batted-ball profile of pitchers clearly moves wOBA on contact (fl_park_bip_profiles.py).  The previous profile model (K% term +.0115/SD, runs/9 per SD +0.46 at half exposure) is reproducible with
PROFILE_TERMS = all five names; backups in output/_backup_before_pitcher_global_only.  MIS effect modelling now focuses on HITTERS (fl_hitter_park.py).
Keys: `pitcher_id` (str) or `name_clean`.
"""
import json
import os

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__)); _cache = {}
PROFILE_TERMS = ("global",)            # which terms of ["global", "hr_prone", "k_rate", "air", "left"] are active; all five = the pre-2026-10-10 profile model


def _load():
    if not _cache:
        js = json.load(open(os.path.join(_BASE, "output/trustmark_pitcher_coefs.json"))); prof = pd.read_parquet(os.path.join(_BASE, "output/pitcher_park_profile.parquet"))
        prof["pitcher_id"] = prof["pitcher_id"].astype(str)
        _cache.update({"names": js["names"], "final": np.array([js["coefs"][k]["final"] for k in js["names"]]), "cov": np.array(js["covariance_raw"]), "ml": js["mean_lefty"], "to_r9": js["to_r9"],
                       "by_id": prof.drop_duplicates("pitcher_id").set_index("pitcher_id"), "by_name": prof.dropna(subset=["name_clean"]).drop_duplicates("name_clean").set_index("name_clean")})
    return _cache


def _design(df: pd.DataFrame) -> np.ndarray:
    """(n x 5) design: [1, z_hr, z_k, z_air, lefty - mean] from the profile table (unknown = the average pitcher)."""
    L = _load()
    if "pitcher_id" in df.columns:
        prof = L["by_id"].reindex(df["pitcher_id"].astype(str).to_numpy())
    elif "name_clean" in df.columns:
        prof = L["by_name"].reindex(df["name_clean"].to_numpy())
    else:
        prof = L["by_name"].reindex(df.index.to_numpy())
    z = lambda c: np.clip(prof[c].fillna(0.0).to_numpy(), -3.0, 3.0); left = prof["lefty"].fillna(L["ml"]).to_numpy()          # profile z's capped at +-3 SD (5 of 731 pitchers reach +3; no extrapolation of the linear terms)
    X = np.column_stack([np.ones(len(df)), z("z_hr"), z("z_k"), z("z_air"), left - L["ml"]]); mask = np.array([n in PROFILE_TERMS for n in L["names"]], float); return X * mask


def effect_woba(df: pd.DataFrame) -> np.ndarray:
    """wOBA-allowed effect per PA at Trustmark (full park); positive = worse for the pitcher."""
    return _design(df) @ _load()["final"]


def shift_r9(df: pd.DataFrame, share: float) -> np.ndarray:
    """Runs/9 shift for a pitcher throwing `share` of his games at Trustmark (positive = more runs allowed)."""
    return share * effect_woba(df) * _load()["to_r9"]


def shift_sd_r9(df: pd.DataFrame, share: float) -> np.ndarray:
    """Standard deviation of that shift from the bootstrap covariance of the raw coefficients (runs/9)."""
    X = _design(df); return share * _load()["to_r9"] * np.sqrt(np.einsum("ij,jk,ik->i", X, _load()["cov"], X))
