"""
fl_trait_residual_analysis.py -- does the current full model's held-out
residual (actual - predicted K%/BB%/K-BB%) correlate with any DESCRIPTIVE
pitcher trait not already in the model -- handedness, bio (height/weight),
average stuff (velo/movement/spin/extension), pitch-mix composition, or
role (starter vs reliever, proxied from usage pattern)? A real correlation
here would point at a candidate feature the model is currently missing;
tested with RELIABILITY-WEIGHTED correlations/group means throughout (same
w = PA/(PA+n0) weighting used for every R2/RMSE this entire project), so a
trait's relationship isn't driven by a handful of thin-PA pitchers.

Usage:
    python fl_trait_residual_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_supabase as fs
import mlb_marcel as mm
from sqlalchemy import text

SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]


def weighted_corr(x, y, w):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return cov / np.sqrt(vx * vy)


def build_held_out_residuals():
    raw = fp.load_fl_pitches_cached()
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    d_tr_k = df_tr.dropna(subset=["K_rate_actual"] + SPEC["K"])
    w_tr_k = mm.reliability_weight(d_tr_k["PA_test"], n0_k).values
    m_k = bt.fit_weighted(d_tr_k, "K_rate_actual", SPEC["K"], w_tr_k)

    d_tr_bb = df_tr.dropna(subset=["BB_rate_actual"] + SPEC["BB"])
    w_tr_bb = mm.reliability_weight(d_tr_bb["PA_test"], n0_bb).values
    m_bb = bt.fit_weighted(d_tr_bb, "BB_rate_actual", SPEC["BB"], w_tr_bb)

    d_te = df_te.dropna(subset=["K_rate_actual", "BB_rate_actual"] + SPEC["K"] + SPEC["BB"]).copy()
    d_te["w_k"] = mm.reliability_weight(d_te["PA_test"], n0_k).values
    d_te["w_bb"] = mm.reliability_weight(d_te["PA_test"], n0_bb).values
    d_te["pred_K"] = m_k.predict(d_te)
    d_te["pred_BB"] = m_bb.predict(d_te)
    d_te["resid_K"] = d_te["K_rate_actual"] - d_te["pred_K"]
    d_te["resid_BB"] = d_te["BB_rate_actual"] - d_te["pred_BB"]
    d_te["resid_KBB"] = d_te["resid_K"] - d_te["resid_BB"]
    d_te["se_K"] = m_k.get_prediction(d_te, weights=d_te["w_k"]).se_obs
    d_te["se_BB"] = m_bb.get_prediction(d_te, weights=d_te["w_bb"]).se_obs
    d_te["z_K"] = d_te["resid_K"] / d_te["se_K"]
    d_te["z_BB"] = d_te["resid_BB"] / d_te["se_BB"]
    d_te["z_KBB"] = d_te["resid_KBB"] / np.sqrt(d_te["se_K"] ** 2 + d_te["se_BB"] ** 2)
    return raw, d_te


def build_pitch_level_traits(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Per-pitcher-ID average stuff/movement, pitch-mix shares, and a simple
    starter/reliever role proxy, over the SAME 2025+2026 window the model's
    own features are built from."""
    pool = fp.filter_period(raw, start, end)
    agg = pool.groupby("pitcher_id").agg(
        velo=("effective_velo", "mean"), ivb=("ivb", "mean"), horz_break=("horz_break", "mean"),
        spin_rate=("spin_rate", "mean"), extension=("extension", "mean"),
        throws=("pitcher_throws", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
    ).reset_index()

    # pitch mix shares (of all pitches thrown)
    FB = {"four_seam", "sinker", "cutter"}
    OFFSPEED = {"changeup", "splitter"}
    BREAKING = {"slider", "curveball", "sweeper", "knuckleball"}
    mix = pool.groupby("pitcher_id")["pitch_type"].value_counts(normalize=True).unstack(fill_value=0)
    mix_share = pd.DataFrame({
        "pitcher_id": mix.index,
        "fb_pct": mix[[c for c in FB if c in mix.columns]].sum(axis=1).values,
        "offspeed_pct": mix[[c for c in OFFSPEED if c in mix.columns]].sum(axis=1).values,
        "breaking_pct": mix[[c for c in BREAKING if c in mix.columns]].sum(axis=1).values,
        "n_pitch_types": (mix >= 0.05).sum(axis=1).values,  # distinct pitches thrown >=5% of the time
    })

    # role proxy: median pitches thrown per game appearance (starters throw
    # far more per outing than relievers)
    per_game = pool.groupby(["pitcher_id", "date"]).size().reset_index(name="pitches_in_game")
    role = per_game.groupby("pitcher_id")["pitches_in_game"].median().reset_index(name="median_pitches_per_game")

    out = agg.merge(mix_share, on="pitcher_id", how="left").merge(role, on="pitcher_id", how="left")
    return out


def build_bio_traits() -> pd.DataFrame:
    df = fs.read_sql_df(text("SELECT name, bats, throws, height_in, weight FROM rosters WHERE position='Pitcher'"))
    return df.rename(columns={"name": "pitcher"}).drop_duplicates("pitcher")


if __name__ == "__main__":
    print("Building held-out residuals (standard Experiment 2 fold, baseline production model)...")
    raw, d_te = build_held_out_residuals()

    print("Building pitch-level descriptive traits (velo/movement/mix/role)...")
    traits = build_pitch_level_traits(raw, bt.START_2025, bt.END_2026)
    d_te = d_te.merge(traits, on="pitcher_id", how="left")

    print("Attaching bio traits from Supabase rosters (bats/height/weight)...")
    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    d_te = d_te.merge(names, on="pitcher_id", how="left")
    bio = build_bio_traits()
    d_te = d_te.merge(bio, on="pitcher", how="left")
    print(f"  bio match rate: {d_te['height_in'].notna().mean():.1%}")

    print("\n" + "=" * 90)
    print("Reliability-WEIGHTED correlation of each trait with held-out residuals (n=%d)" % len(d_te))
    print("=" * 90)
    continuous_traits = ["velo", "ivb", "horz_break", "spin_rate", "extension",
                          "fb_pct", "offspeed_pct", "breaking_pct", "n_pitch_types",
                          "median_pitches_per_game", "height_in", "weight"]
    targets = ["resid_K", "resid_BB", "resid_KBB", "z_K", "z_BB", "z_KBB"]

    rows = []
    for trait in continuous_traits:
        sub = d_te.dropna(subset=[trait])
        row = {"trait": trait, "n": len(sub)}
        for tgt in targets:
            row[tgt] = weighted_corr(sub[trait], sub[tgt], sub["w_k"] if "K" in tgt and "KBB" not in tgt else
                                      (sub["w_bb"] if "BB" in tgt and "KBB" not in tgt else
                                       np.sqrt(sub["w_k"] * sub["w_bb"])))
        rows.append(row)
    corr_df = pd.DataFrame(rows).set_index("trait")
    pd.set_option("display.width", 160)
    print(corr_df.round(3))

    print("\n--- Weighted mean residual by handedness (categorical) ---")
    for col, label in [("throws_x", "pitch-data throws"), ("throws_y", "roster bio throws"), ("bats", "bats")]:
        if col not in d_te.columns:
            continue
        sub = d_te.dropna(subset=[col])
        g = sub.groupby(col).apply(lambda s: pd.Series({
            "n": len(s), "w_mean_resid_K": np.average(s["resid_K"], weights=s["w_k"]),
            "w_mean_resid_BB": np.average(s["resid_BB"], weights=s["w_bb"]),
            "w_mean_z_KBB": np.average(s["z_KBB"], weights=np.sqrt(s["w_k"] * s["w_bb"])),
        }), include_groups=False)
        print(f"\n{label}:")
        print(g.round(4))

    d_te.to_parquet("output/_scratch_trait_residuals.parquet")
