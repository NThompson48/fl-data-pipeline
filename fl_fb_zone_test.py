"""
fl_fb_zone_test.py -- tests Fastball Zone% (fraction of four-seam/sinker
fastballs thrown in the strike zone) as a direct BB% regressor, via the SAME
genuinely held-out check as Experiment 2: fit on full-2025 -> actual 1H2026,
freeze, roll forward to 2025+1H2026 -> actual 2H2026.

Follows up on the correlation screen (concurrent corr=-0.60, but that's
largely circular -- Zone% and BB_rate are measured in the SAME window and are
mechanically linked; held-out corr train-2025 -> actual 1H2026 = -0.415,
genuinely predictive) and the stickiness/n0 check (self-corr ~0.39-0.45,
n0~157 fastballs ~77 PA-equivalent -- noisier than the model's existing
regressors but real signal, not noise). K% was NOT tested here -- its
concurrent corr with Fastball Zone% was 0.047, negligible.

Usage:
    python fl_fb_zone_test.py
"""
from __future__ import annotations

import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

BASE_SPEC = {"BB": ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_dist", "vert_dist"]}
FB_ZONE_SPEC = {"BB": BASE_SPEC["BB"] + ["fb_zone_pct"]}


def with_fb_zone(df: pd.DataFrame, raw: pd.DataFrame, windows: list[tuple[str, str]]) -> pd.DataFrame:
    pool = pd.concat([fp.filter_period(raw, s, e) for s, e in windows], ignore_index=True)
    fbz = fp.compute_fb_zone_pct(pool)[["pitcher_id", "fb_zone_pct"]]
    out = df.merge(fbz, on="pitcher_id", how="left")
    # same neutral-imputation convention bt.assemble_features already uses for
    # wh_sva_level/horiz_dist/vert_dist -- missing = training population's own mean
    out["fb_zone_pct"] = out["fb_zone_pct"].fillna(out["fb_zone_pct"].mean())
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

    df_tr = with_fb_zone(df_tr, raw, [(bt.START_2025, bt.END_2025)])
    df_te = with_fb_zone(df_te, raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)])

    print(f"\nFastball Zone%% distribution (train window): "
          f"{df_tr['fb_zone_pct'].describe()[['min', '25%', '50%', '75%', 'max']].to_dict()}")

    print("\n" + "=" * 70)
    print("Held-out test: base BB% model vs. base + Fastball Zone%")
    print("=" * 70)
    results = []
    for name, spec in [("base (naive+other_league+dist)", BASE_SPEC), ("base + fb_zone_pct", FB_ZONE_SPEC)]:
        bb_res = bt.run_heldout(df_tr, "BB_rate_actual", spec["BB"], df_te, "BB_rate_actual",
                                 n0_bb, f"{name} [BB%]")
        results.append(bb_res)
    print(pd.DataFrame(results).to_string(index=False))

    print("\n--- Does fb_zone_pct have a significant coefficient? (fit on train window) ---")
    d = df_tr.dropna(subset=["BB_rate_actual"] + FB_ZONE_SPEC["BB"])
    w = mm.reliability_weight(d["PA_test"], n0_bb).values
    m = bt.fit_weighted(d, "BB_rate_actual", FB_ZONE_SPEC["BB"], w)
    print(m.summary().tables[1])
