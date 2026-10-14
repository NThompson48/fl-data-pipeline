"""fl_opponent_quality_test.py -- does the quality of batters a pitcher faced
explain anything about their K%/BB%, beyond what the existing model already
knows? Mirrors mlb_opponent_kpct_explained.py's leave-one-out methodology
exactly (that script's own MLB-level finding was never logged/rerun here --
this is the FL-specific, real-data answer).

batter_id/batter are NOT in fl_pitcher_features.load_fl_pitches_cached()'s
column set (only batter_side/handedness is) -- pulled fresh here via a lean
direct query.

Leave-one-out: each batter's own K%/BB% is computed EXCLUDING their PA
against the specific pitcher being scored, to avoid the mechanical leakage
of a batter's rate already containing the very outcome being explained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from fl_supabase import read_sql_df
import fl_halfseason_backtest as bt
import mlb_marcel as mm

MIN_PA = 20  # pitcher-side eligibility floor, consistent with this project's usual thresholds


def load_pitches_with_batter() -> pd.DataFrame:
    q = """
        SELECT pitcher, pitcher_id, batter, batter_id, date, year, game_id,
               is_terminal_pitch, k_or_bb
        FROM pitches
        WHERE game_type = 'reg_season' AND pitcher_id IS NOT NULL AND batter_id IS NOT NULL
    """
    return read_sql_df(q)


def opponent_quality_index(term: pd.DataFrame, min_other_pa: int = 10) -> pd.DataFrame:
    """term = terminal pitches only, one row per PA. Returns one row per
    pitcher with expected_K_rate/expected_BB_rate (PA-weighted average of
    each opposing batter's own leave-one-out K%/BB%) and PA_covered."""
    term = term.copy()
    term["is_k"] = term["k_or_bb"] == "strikeout"
    term["is_bb"] = term["k_or_bb"] == "walk"

    batter_tot = term.groupby("batter_id").agg(
        PA_bat=("is_k", "size"), K_bat=("is_k", "sum"), BB_bat=("is_bb", "sum")).reset_index()

    matchup = term.groupby(["pitcher_id", "batter_id"]).agg(
        PA_mu=("is_k", "size"), K_mu=("is_k", "sum"), BB_mu=("is_bb", "sum")).reset_index()
    matchup = matchup.merge(batter_tot, on="batter_id", how="left")

    matchup["PA_loo"] = matchup["PA_bat"] - matchup["PA_mu"]
    matchup["K_loo"] = matchup["K_bat"] - matchup["K_mu"]
    matchup["BB_loo"] = matchup["BB_bat"] - matchup["BB_mu"]
    matchup = matchup[matchup["PA_loo"] >= min_other_pa].copy()
    matchup["K_rate_loo"] = matchup["K_loo"] / matchup["PA_loo"]
    matchup["BB_rate_loo"] = matchup["BB_loo"] / matchup["PA_loo"]

    out = matchup.groupby("pitcher_id").apply(lambda g: pd.Series({
        "expected_K_rate": np.average(g["K_rate_loo"], weights=g["PA_mu"]),
        "expected_BB_rate": np.average(g["BB_rate_loo"], weights=g["PA_mu"]),
        "PA_covered": g["PA_mu"].sum(),
    }), include_groups=False).reset_index()
    return out


if __name__ == "__main__":
    print("Pulling pitches with batter identity (fresh, not the cached pull)...")
    raw = load_pitches_with_batter()
    print(f"  {len(raw):,} pitches")
    is_term = raw["is_terminal_pitch"].astype(str).isin(["1", "True", "true"])
    term = raw[is_term].copy()
    print(f"  {len(term):,} terminal (PA-ending) pitches")

    print("\n=== Full 2025+2026 window, in-sample (is opponent quality related to actual rate AT ALL) ===")
    opp = opponent_quality_index(term)
    pitcher_tot = term.groupby("pitcher_id").agg(
        PA=("is_terminal_pitch", "size"),
        K_rate=("k_or_bb", lambda s: (s == "strikeout").mean()),
        BB_rate=("k_or_bb", lambda s: (s == "walk").mean())).reset_index()
    d = pitcher_tot.merge(opp, on="pitcher_id", how="inner")
    d["coverage"] = d["PA_covered"] / d["PA"]
    print(f"median PA coverage (batters with >=10 other PA): {d['coverage'].median():.3f}")
    d = d[d["PA"] >= MIN_PA]
    print(f"n pitchers = {len(d)}")

    n0_k, _ = mm.estimate_n0(d["K_rate"], d["PA"])
    w = mm.reliability_weight(d["PA"], n0_k).values
    for metric, exp_col in [("K_rate", "expected_K_rate"), ("BB_rate", "expected_BB_rate")]:
        corr = d[exp_col].corr(d[metric])
        r2_w = mm.weighted_r2(d[metric].values, d[exp_col].values, w)
        print(f"  {metric}: corr(actual, opponent-implied)={corr:.4f}  "
              f"R2(opponent-implied AS the prediction, weighted)={r2_w:.4f}  "
              f"SD(actual)={d[metric].std()*100:.2f}pp  SD(opponent-implied)={d[exp_col].std()*100:.2f}pp")

    # ---------------------------------------------------------------- #
    # Real test: genuinely held out, incremental to the EXISTING model.
    # Opponent quality computed ONLY from the TRAIN window's own pitches
    # (full 2025) -- who a pitcher faced during the period the model is fit
    # on -- then checked as an added regressor predicting the ACTUAL
    # 2H2026 rate, same Experiment-2 harness every other feature in this
    # project has been judged against.
    # ---------------------------------------------------------------- #
    print("\n=== Experiment 2 held-out: does opponent quality (from the TRAIN window) add anything? ===")
    import fl_pitcher_features as fp
    raw_train_period = raw[(raw["date"] >= bt.START_2025) & (raw["date"] <= bt.END_2025)]
    is_term_tr = raw_train_period["is_terminal_pitch"].astype(str).isin(["1", "True", "true"])
    term_tr = raw_train_period[is_term_tr].copy()
    opp_tr = opponent_quality_index(term_tr)
    opp_tr["pitcher_id"] = opp_tr["pitcher_id"].astype(str)

    raw_fp = fp.load_fl_pitches_cached()
    df_tr, n0_k2, n0_bb2 = bt.assemble(raw_fp, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                        with_changepoints=False)
    df_te, _, _ = bt.assemble(raw_fp, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    lg_k = term_tr[term_tr["k_or_bb"] == "strikeout"].shape[0] / len(term_tr)
    lg_bb = term_tr[term_tr["k_or_bb"] == "walk"].shape[0] / len(term_tr)
    for df in (df_tr, df_te):
        df["pitcher_id"] = df["pitcher_id"].astype(str)
        df.loc[:, "expected_K_rate"] = df["pitcher_id"].map(opp_tr.set_index("pitcher_id")["expected_K_rate"]).fillna(lg_k)
        df.loc[:, "expected_BB_rate"] = df["pitcher_id"].map(opp_tr.set_index("pitcher_id")["expected_BB_rate"]).fillna(lg_bb)

    SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]
    k_base = bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"], df_te, "K_rate_actual", n0_k2, "K% baseline")
    k_opp = bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["expected_K_rate"], df_te, "K_rate_actual", n0_k2,
                            "K% + opponent quality")
    bb_base = bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"], df_te, "BB_rate_actual", n0_bb2, "BB% baseline")
    bb_opp = bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + ["expected_BB_rate"], df_te, "BB_rate_actual", n0_bb2,
                             "BB% + opponent quality")
    print(pd.DataFrame([k_base, k_opp, bb_base, bb_opp]).to_string(index=False))

    m_k = smf.wls("K_rate_actual ~ " + " + ".join(SPEC["K"] + ["expected_K_rate"]), data=df_tr,
                   weights=mm.reliability_weight(df_tr["PA"], n0_k2)).fit()
    print(f"\nexpected_K_rate coef={m_k.params['expected_K_rate']:.4f}  p={m_k.pvalues['expected_K_rate']:.4f}")
    m_bb = smf.wls("BB_rate_actual ~ " + " + ".join(SPEC["BB"] + ["expected_BB_rate"]), data=df_tr,
                    weights=mm.reliability_weight(df_tr["PA"], n0_bb2)).fit()
    print(f"expected_BB_rate coef={m_bb.params['expected_BB_rate']:.4f}  p={m_bb.pvalues['expected_BB_rate']:.4f}")
