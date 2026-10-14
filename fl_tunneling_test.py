"""fl_tunneling_test.py -- tests "Tunneling+" from
https://medium.com/@mattjcreally/quantifying-arsenal-effects-using-mlbs-play-by-play-data
(pasted into this session by the user): a pitcher's standard deviation of
pitch spatial position (X, Z) 150ms after release, using the real release
position/velocity/acceleration (x0/y0/z0, vx0/vy0/vz0, ax0/ay0/az0 -- the
same kinematic trajectory model MLB Statcast uses: pos(t) = p0 + v0*t +
0.5*a*t^2). Lower spread = pitches look more alike at the batter's decision
point = better tunneling (article's own scale: higher Tunneling+ = better,
i.e. LOWER raw spatial std dev).

These fields exist on the real Supabase `pitches` table but are NOT in
fl_pitcher_features.load_fl_pitches_cached()'s column set -- pulled fresh
here, restricted to the TRAIN window only (full 2025) to keep the query
light, consistent with every other held-out feature test this session.

Usage:
    python fl_tunneling_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from fl_supabase import read_sql_df
import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

DECISION_T = 0.15  # seconds after release, per the article
MIN_PITCHES = 50


def load_trajectory_train_window() -> pd.DataFrame:
    q = f"""
        SELECT pitcher_id, x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0
        FROM pitches
        WHERE game_type = 'reg_season' AND pitcher_id IS NOT NULL
          AND date >= '{bt.START_2025}' AND date <= '{bt.END_2025}'
          AND x0 IS NOT NULL AND vx0 IS NOT NULL AND ax0 IS NOT NULL
    """
    return read_sql_df(q)


def compute_tunneling(df: pd.DataFrame, t: float = DECISION_T) -> pd.DataFrame:
    d = df.copy()
    d["x_t"] = d["x0"] + d["vx0"] * t + 0.5 * d["ax0"] * t ** 2
    d["z_t"] = d["z0"] + d["vz0"] * t + 0.5 * d["az0"] * t ** 2
    out = d.groupby("pitcher_id").agg(
        n_pitches=("x_t", "size"), x_sd=("x_t", "std"), z_sd=("z_t", "std")).reset_index()
    out = out[out["n_pitches"] >= MIN_PITCHES].copy()
    out["spatial_sd"] = np.sqrt(out["x_sd"] ** 2 + out["z_sd"] ** 2)  # Pythagorean combine, per the article
    return out


if __name__ == "__main__":
    print("Pulling release-point trajectory fields for the TRAIN window (full 2025)...")
    traj = load_trajectory_train_window()
    print(f"  {len(traj):,} pitches with real trajectory data")

    tun = compute_tunneling(traj)
    tun["pitcher_id"] = tun["pitcher_id"].astype(str)
    print(f"  n={len(tun)} pitchers with >= {MIN_PITCHES} pitches")
    print(f"  position @ {DECISION_T*1000:.0f}ms spatial_sd (ft) describe:\n{tun['spatial_sd'].describe()}")

    print("\nLoading cached FL pitches, assembling Experiment-2 panels...")
    raw = fp.load_fl_pitches_cached()
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    lg_spatial_sd = tun["spatial_sd"].median()
    for df in (df_tr, df_te):
        df["pitcher_id"] = df["pitcher_id"].astype(str)
        df["spatial_sd"] = df["pitcher_id"].map(tun.set_index("pitcher_id")["spatial_sd"]).fillna(lg_spatial_sd)
        # tunneling_plus: article's convention, higher = better (= LOWER raw spread) -- use negative
        # spread directly as the regressor so higher_better=True reads naturally in the harness below.
        df["tunneling_score"] = -df["spatial_sd"]

    print(f"\ncorr(spatial_sd, wh_sva_level) = {df_tr['spatial_sd'].corr(df_tr['wh_sva_level']):.4f}  "
          f"(is this just a Stuff+ proxy?)")

    SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]
    results = []
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"], df_te, "K_rate_actual", n0_k, "K% baseline"))
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["tunneling_score"], df_te, "K_rate_actual",
                                   n0_k, "K% + tunneling"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"], df_te, "BB_rate_actual", n0_bb, "BB% baseline"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + ["tunneling_score"], df_te, "BB_rate_actual",
                                   n0_bb, "BB% + tunneling"))
    print()
    print(pd.DataFrame(results).to_string(index=False))

    w_k = mm.reliability_weight(df_tr["PA"], n0_k)
    m_k = smf.wls("K_rate_actual ~ " + " + ".join(SPEC["K"] + ["tunneling_score"]), data=df_tr, weights=w_k).fit()
    print(f"\nK% model: tunneling_score coef={m_k.params['tunneling_score']:.5f} "
          f"p={m_k.pvalues['tunneling_score']:.4f}")

    w_bb = mm.reliability_weight(df_tr["PA"], n0_bb)
    m_bb = smf.wls("BB_rate_actual ~ " + " + ".join(SPEC["BB"] + ["tunneling_score"]), data=df_tr, weights=w_bb).fit()
    print(f"BB% model: tunneling_score coef={m_bb.params['tunneling_score']:.5f} "
          f"p={m_bb.pvalues['tunneling_score']:.4f}")
