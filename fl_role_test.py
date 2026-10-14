"""
fl_role_test.py -- tests adding BF/G (batters faced per game, a usage/role proxy --
starters high, relievers low) as a direct regressor to the best validated model
(naive+other_league+surprise_prior), via the SAME genuinely held-out check as
Experiment 2 in fl_halfseason_backtest.py: fit on full-2025 -> actual 1H2026, freeze,
roll forward to 2025+1H2026 -> actual 2H2026.

Follows up on the error-analysis finding (training-period usage carried signal
beyond its role as a reliability weight) and the earlier FL precedent (BF/G was the
single strongest lever for wh_sva) -- neither has tested BF/G as a direct K%/BB%
regressor until now.

Usage:
    python fl_role_test.py
"""
from __future__ import annotations

import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

BASE_SPEC = {"K": ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level"],
             "BB": ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_dist", "vert_dist"]}
ROLE_SPEC = {"K": BASE_SPEC["K"] + ["bf_per_g"], "BB": BASE_SPEC["BB"] + ["bf_per_g"]}


def with_role(df: pd.DataFrame, raw: pd.DataFrame, windows: list[tuple[str, str]]) -> pd.DataFrame:
    pool = pd.concat([fp.filter_period(raw, s, e) for s, e in windows], ignore_index=True)
    role = fp.compute_bf_per_g(pool)
    out = df.merge(role, on="pitcher_id", how="left")
    out["bf_per_g"] = out["bf_per_g"].fillna(out["bf_per_g"].mean())
    return out


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("\nAssembling Experiment-2 train (full-2025 -> actual 1H2026) and "
          "test (2025+1H2026 -> actual 2H2026) panels...")
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                      [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    df_tr = with_role(df_tr, raw, [(bt.START_2025, bt.END_2025)])
    df_te = with_role(df_te, raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)])

    print(f"\nBF/G distribution (train window): "
          f"{df_tr['bf_per_g'].describe()[['min', '25%', '50%', '75%', 'max']].to_dict()}")

    print("\n" + "=" * 70)
    print("Held-out test: base model vs. base model + BF/G")
    print("=" * 70)
    results = []
    for name, spec in [("base (naive+other_league+surprise_prior)", BASE_SPEC),
                        ("base + bf_per_g", ROLE_SPEC)]:
        k_res = bt.run_heldout(df_tr, "K_rate_actual", spec["K"], df_te, "K_rate_actual",
                                n0_k, f"{name} [K%]")
        bb_res = bt.run_heldout(df_tr, "BB_rate_actual", spec["BB"], df_te, "BB_rate_actual",
                                 n0_bb, f"{name} [BB%]")
        results += [k_res, bb_res]
    print(pd.DataFrame(results).to_string(index=False))

    print("\n--- Does bf_per_g have a significant coefficient? (fit on train window) ---")
    for metric, target, spec in [("K%", "K_rate_actual", ROLE_SPEC["K"]), ("BB%", "BB_rate_actual", ROLE_SPEC["BB"])]:
        d = df_tr.dropna(subset=[target] + spec)
        w = mm.reliability_weight(d["PA_test"], n0_k if metric == "K%" else n0_bb).values
        m = bt.fit_weighted(d, target, spec, w)
        print(f"\n{metric} model:")
        print(m.params)
        print(f"bf_per_g p-value: {m.pvalues['bf_per_g']:.4f}")
