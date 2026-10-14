"""fl_repertoire_diversity_test.py -- tests the "Repertoire+" concept from
https://medium.com/@mattjcreally/quantifying-arsenal-effects-using-mlbs-play-by-play-data
(pasted into this session by the user) on real FL data: a pitcher's
PITCH-LEVEL standard deviation of perceived velocity (velo_sd) and of
Pythagorean combined movement sqrt(ivb^2 + horz_break^2) (move_sd), pooled
across every pitch thrown (not a between-pitch-type mean difference like
the velo_diff tested in fl_arsenal_interaction_test.py -- a genuinely
different construction: this rewards an unpredictable MIX across the whole
arsenal, not just a two-group gap).

The article flags that these may just proxy for Stuff+ -- checked directly
here via (a) their raw correlation with wh_sva_level, and (b) whether they
add anything to a model that ALREADY has wh_sva_level in it, same held-out
Experiment 2 harness as everything else in this project.

Usage:
    python fl_repertoire_diversity_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

MIN_PITCHES = 50  # need a real sample to trust a per-pitcher std dev


def compute_diversity(period_df: pd.DataFrame) -> pd.DataFrame:
    d = period_df.dropna(subset=["effective_velo", "ivb", "horz_break"]).copy()
    d["move_mag"] = np.sqrt(d["ivb"] ** 2 + d["horz_break"] ** 2)
    out = d.groupby("pitcher_id").agg(
        n_pitches=("effective_velo", "size"),
        velo_sd=("effective_velo", "std"),
        move_sd=("move_mag", "std"),
    ).reset_index()
    return out[out["n_pitches"] >= MIN_PITCHES]


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("Computing Velocity/Movement Diversity over the TRAIN window (full 2025)...")
    train_period = fp.filter_period(raw, bt.START_2025, bt.END_2025)
    div = compute_diversity(train_period)
    div["pitcher_id"] = div["pitcher_id"].astype(str)
    print(f"  n={len(div)} pitchers with >= {MIN_PITCHES} pitches")
    print(div[["velo_sd", "move_sd"]].describe())

    print("\nAssembling Experiment-2 train/test panels...")
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    lg_velo_sd = div["velo_sd"].median()
    lg_move_sd = div["move_sd"].median()
    for df in (df_tr, df_te):
        df["pitcher_id"] = df["pitcher_id"].astype(str)
        df["velo_sd"] = df["pitcher_id"].map(div.set_index("pitcher_id")["velo_sd"]).fillna(lg_velo_sd)
        df["move_sd"] = df["pitcher_id"].map(div.set_index("pitcher_id")["move_sd"]).fillna(lg_move_sd)
        df["repertoire_sum"] = df["velo_sd"] + df["move_sd"]  # proxy for "average of the two +scores"

    print(f"\ncorr(velo_sd, wh_sva_level) = {df_tr['velo_sd'].corr(df_tr['wh_sva_level']):.4f}")
    print(f"corr(move_sd, wh_sva_level) = {df_tr['move_sd'].corr(df_tr['wh_sva_level']):.4f}")
    print("(the article's own caveat: check these aren't just proxying for Stuff+/wh_sva)")

    SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]
    results = []
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"], df_te, "K_rate_actual", n0_k, "K% baseline"))
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["velo_sd"], df_te, "K_rate_actual", n0_k,
                                   "K% + velo_sd"))
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["move_sd"], df_te, "K_rate_actual", n0_k,
                                   "K% + move_sd"))
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["velo_sd", "move_sd"], df_te, "K_rate_actual",
                                   n0_k, "K% + both"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"], df_te, "BB_rate_actual", n0_bb, "BB% baseline"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + ["velo_sd"], df_te, "BB_rate_actual", n0_bb,
                                   "BB% + velo_sd"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + ["move_sd"], df_te, "BB_rate_actual", n0_bb,
                                   "BB% + move_sd"))
    print()
    print(pd.DataFrame(results).to_string(index=False))

    w_k = mm.reliability_weight(df_tr["PA"], n0_k)
    m_k = smf.wls("K_rate_actual ~ " + " + ".join(SPEC["K"] + ["velo_sd", "move_sd"]), data=df_tr, weights=w_k).fit()
    print(f"\nK% combined: velo_sd coef={m_k.params['velo_sd']:.5f} p={m_k.pvalues['velo_sd']:.4f}  "
          f"move_sd coef={m_k.params['move_sd']:.5f} p={m_k.pvalues['move_sd']:.4f}")

    w_bb = mm.reliability_weight(df_tr["PA"], n0_bb)
    m_bb = smf.wls("BB_rate_actual ~ " + " + ".join(SPEC["BB"] + ["velo_sd", "move_sd"]), data=df_tr, weights=w_bb).fit()
    print(f"BB% combined: velo_sd coef={m_bb.params['velo_sd']:.5f} p={m_bb.pvalues['velo_sd']:.4f}  "
          f"move_sd coef={m_bb.params['move_sd']:.5f} p={m_bb.pvalues['move_sd']:.4f}")
