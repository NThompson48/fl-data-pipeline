"""
fl_interaction_test.py -- tests whether wh_sva/horiz_dist/vert_dist should matter
MORE for pitchers with an unreliable (small-sample) naive prior, rather than
contributing the same fixed marginal weight regardless of how much track record a
pitcher has (the current model's assumption, never actually tested).

Operationalization: interact each surprise variable with (1 - reliability), where
reliability = PA/(PA+n0) is the SAME quantity already used to shrink the naive
prior. This is large (~1) for a low-PA pitcher and near 0 for a well-established
one -- exactly "how much should the surprise variable matter because we don't
trust the track record yet." Added ALONGSIDE the existing plain term (not
replacing it), so the data decides via its own significance whether the
interaction adds anything beyond the already-validated plain effect.

Tested via the same genuinely held-out check as Experiment 2: fit on full-2025 ->
actual 1H2026, frozen, scored on 2025+1H2026 -> actual 2H2026.

Usage:
    python fl_interaction_test.py
"""
from __future__ import annotations

import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

BASE_SPEC = {"K": ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level"],
             "BB": ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_dist", "vert_dist"]}
INTERACT_SPEC = {"K": BASE_SPEC["K"] + ["wh_sva_x_unreliable"],
                  "BB": BASE_SPEC["BB"] + ["horiz_dist_x_unreliable", "vert_dist_x_unreliable"]}


def add_interactions(df: pd.DataFrame, n0_k: float, n0_bb: float) -> pd.DataFrame:
    df = df.copy()
    unreliable_k = 1 - mm.reliability_weight(df["PA"], n0_k)
    unreliable_bb = 1 - mm.reliability_weight(df["PA"], n0_bb)
    df["wh_sva_x_unreliable"] = df["wh_sva_level"] * unreliable_k
    df["horiz_dist_x_unreliable"] = df["horiz_dist"] * unreliable_bb
    df["vert_dist_x_unreliable"] = df["vert_dist"] * unreliable_bb
    return df


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("\nAssembling Experiment-2 train/test panels...")
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                      [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    df_tr = add_interactions(df_tr, n0_k, n0_bb)
    df_te = add_interactions(df_te, n0_k, n0_bb)

    print(f"\nreliability distribution (train PA, K%): "
          f"{mm.reliability_weight(df_tr['PA'], n0_k).describe()[['min','25%','50%','75%','max']].to_dict()}")

    print("\n" + "=" * 70)
    print("Held-out test: base model vs. base + (surprise var x unreliability) interaction")
    print("=" * 70)
    results = []
    for name, spec in [("base", BASE_SPEC), ("base + interaction", INTERACT_SPEC)]:
        k_res = bt.run_heldout(df_tr, "K_rate_actual", spec["K"], df_te, "K_rate_actual",
                                n0_k, f"{name} [K%]")
        bb_res = bt.run_heldout(df_tr, "BB_rate_actual", spec["BB"], df_te, "BB_rate_actual",
                                 n0_bb, f"{name} [BB%]")
        results += [k_res, bb_res]
    print(pd.DataFrame(results).to_string(index=False))

    print("\n--- Significance of the interaction term (fit on train window) ---")
    for metric, target, spec in [("K%", "K_rate_actual", INTERACT_SPEC["K"]),
                                  ("BB%", "BB_rate_actual", INTERACT_SPEC["BB"])]:
        d = df_tr.dropna(subset=[target] + spec)
        w = mm.reliability_weight(d["PA"], n0_k if metric == "K%" else n0_bb).values
        m = bt.fit_weighted(d, target, spec, w)
        print(f"\n{metric} model:")
        print(m.summary().tables[1])
