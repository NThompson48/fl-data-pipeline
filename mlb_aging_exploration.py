"""
mlb_aging_exploration.py -- does K%/BB% aging need more than one uniform curve?

The production model (mlb_kbb_project_2026.py) imposes a single linear+post-30-kink
age_c/age_decline shape, shared across every pitcher, for BOTH K% and BB%. This
script checks that assumption empirically, mirroring k26_projection.py's delta-
method aging_curve() (pair each pitcher's year Y -> Y+1 change in STABILIZED rate,
average by age, weighted by playing time) but:
  1. built separately for K% and BB% (they may age differently)
  2. built separately by ROLE (starter vs reliever, split on BF_per_G) and by
     ARSENAL TYPE (fastball-reliant vs breaking-ball-reliant, split on which
     swstr_sva family the pitcher leans on) to see if a single curve is hiding
     real heterogeneity
  3. checked against the current linear+kink fit AND a more flexible age-bin
     regression, to see whether the extra flexibility earns its keep out-of-sample
     (not just in-sample -- matching this project's standing practice of never
     trusting a flexible fit without a holdout check)

Usage:
    python mlb_aging_exploration.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlb_marcel as mm
from mlb_kbb_model import MIN_PA, PANEL_PATH

pd.set_option("display.width", 160, "display.max_columns", 40)


def _wmean_ci(y, w):
    y, w = np.asarray(y), np.asarray(w)
    yb = np.average(y, weights=w)
    neff = w.sum() ** 2 / np.sum(w ** 2)
    sd = np.sqrt(np.average((y - yb) ** 2, weights=w))
    return yb, 1.96 * sd / np.sqrt(neff)


def build_couplets(panel: pd.DataFrame, n0_k: float, n0_bb: float) -> pd.DataFrame:
    """Single-season stabilized K_rate/BB_rate (shrunk toward that YEAR's league
    rate by n0, k26_projection.py's exact k_stab construction -- NOT the 3-year
    Marcel rate, since the delta method needs each season's own noise damped, not
    3 years of history blended in), then consecutive-season pairs per pitcher."""
    o = panel.copy()
    lg_k = o.groupby("game_year").apply(lambda g: np.average(g["K_rate"], weights=g["PA"])).rename("lg_k")
    lg_bb = o.groupby("game_year").apply(lambda g: np.average(g["BB_rate"], weights=g["PA"])).rename("lg_bb")
    o = o.join(lg_k, on="game_year").join(lg_bb, on="game_year")
    o["k_stab"] = (o["K"] + n0_k * o["lg_k"]) / (o["PA"] + n0_k)
    o["bb_stab"] = (o["BB"] + n0_bb * o["lg_bb"]) / (o["PA"] + n0_bb)

    b = o.copy()
    b["game_year"] = b["game_year"] - 1
    cp = o.merge(b, on=["pitcher", "game_year"], suffixes=("1", "2"))
    cp["w"] = 2 / (1 / cp["PA1"] + 1 / cp["PA2"])
    cp["age1_int"] = cp["age1"].round().astype(int)
    cp["dK_stab"] = cp["k_stab2"] - cp["k_stab1"]
    cp["dBB_stab"] = cp["bb_stab2"] - cp["bb_stab1"]
    cp["dK_raw"] = cp["K_rate2"] - cp["K_rate1"]
    cp["dBB_raw"] = cp["BB_rate2"] - cp["BB_rate1"]
    return cp


def empirical_age_curve(cp: pd.DataFrame, dcol: str, min_n: int = 25) -> pd.DataFrame:
    rows = []
    for age, g in cp.groupby("age1_int"):
        if len(g) < min_n:
            continue
        m, ci = _wmean_ci(g[dcol].values, g["w"].values)
        rows.append((age, m, ci, len(g)))
    curve = pd.DataFrame(rows, columns=["age", "mean_delta", "ci95", "n"])
    if len(curve):
        off = np.average(curve.loc[curve.age <= 25, "mean_delta"],
                          weights=curve.loc[curve.age <= 25, "n"]) if (curve.age <= 25).any() else 0.0
        curve["mean_delta_z"] = curve["mean_delta"] - off
        curve["cum_delta"] = curve["mean_delta_z"].cumsum() - curve["mean_delta_z"].cumsum().max()
    return curve


def linear_kink_predict(age, slope_lin, slope_extra_30):
    age_c = age - 29
    age_decline = np.maximum(age - 30, 0)
    return slope_lin * age_c + slope_extra_30 * age_decline


if __name__ == "__main__":
    panel = pd.read_parquet(PANEL_PATH)
    elig = panel[panel["PA"] >= MIN_PA]
    n0_k, _ = mm.estimate_n0(elig["K_rate"], elig["PA"])
    n0_bb, _ = mm.estimate_n0(elig["BB_rate"], elig["PA"])
    print(f"n0_K={n0_k:.1f}  n0_BB={n0_bb:.1f}")

    cp = build_couplets(elig, n0_k, n0_bb)
    print(f"couplets: {len(cp)}")

    print("\n=== Overall empirical aging curve: K% (stabilized delta method) ===")
    curve_k = empirical_age_curve(cp, "dK_stab")
    print(curve_k.round(4).to_string(index=False))

    print("\n=== Overall empirical aging curve: BB% (stabilized delta method) ===")
    curve_bb = empirical_age_curve(cp, "dBB_stab")
    print(curve_bb.round(4).to_string(index=False))

    # -------------------------------------------------------------- #
    #  Role split: starter vs reliever, via BF_per_G1 (year-Y role)
    # -------------------------------------------------------------- #
    role_cut = cp["BF_per_G1"].median()
    print(f"\nBF_per_G median split at {role_cut:.2f} (role at year Y)")
    cp["role"] = np.where(cp["BF_per_G1"] >= role_cut, "starter-ish (high BF/G)", "reliever-ish (low BF/G)")
    for role, g in cp.groupby("role"):
        print(f"\n--- K% aging, {role} (n_couplets={len(g)}) ---")
        print(empirical_age_curve(g, "dK_stab").round(4).to_string(index=False))
    for role, g in cp.groupby("role"):
        print(f"\n--- BB% aging, {role} (n_couplets={len(g)}) ---")
        print(empirical_age_curve(g, "dBB_stab").round(4).to_string(index=False))

    # -------------------------------------------------------------- #
    #  Arsenal-type split: fastball-reliant vs breaking-ball-reliant,
    #  via relative swstr_sva_fb vs swstr_sva_brk magnitude (which family's
    #  shape is actually carrying the pitcher's whiff profile at year Y)
    # -------------------------------------------------------------- #
    cp["fb_reliant"] = cp["swstr_sva_fb1"].abs() >= cp["swstr_sva_brk1"].abs()
    cp["arsenal"] = np.where(cp["fb_reliant"], "fastball-reliant", "breaking-ball-reliant")
    for arsenal, g in cp.groupby("arsenal"):
        print(f"\n--- K% aging, {arsenal} (n_couplets={len(g)}) ---")
        print(empirical_age_curve(g, "dK_stab").round(4).to_string(index=False))
