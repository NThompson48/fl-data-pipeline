"""
fl_other_league_bridge_test.py -- the plan's required verification step: does
widening the "other league prior" bridge (narrow college match -> + college
name-match extension -> + Pioneer League -> + affiliated MiLB, per
fl_production_projection.extend_match_by_name/extend_match_by_other_leagues)
actually improve held-out K%/BB% prediction, or at least not hurt it?

Reuses the EXACT Experiment 2 held-out check (fit on full-2025 -> actual
1H2026, freeze, roll forward to 2025+1H2026 -> actual 2H2026) that every other
model-structure change this session has been judged against. The only thing
that changes between variants is which rows get a real (non-fallback)
fl_k_rate_bridge/fl_bb_rate_bridge value -- the outer naive+other_league+
surprise_prior model spec and its fitting discipline are untouched.

Usage:
    python fl_other_league_bridge_test.py
"""
from __future__ import annotations

import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_college_bridge as bridge
import fl_pioneer_bridge as pbridge
import fl_milb_bridge as mbridge
import fl_production_projection as prod

SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]

TRAIN_WINDOWS = [(bt.START_2025, bt.END_2025)]
TEST_LABEL_WINDOWS = [(bt._DAY_AFTER_2026_CUT, bt.END_2026)]


def enrich_bridge(df: pd.DataFrame, raw: pd.DataFrame, prospect_proj, pioneer_full, milb_full) -> pd.DataFrame:
    """Starting from a bt.assemble_features frame (narrow college-only bridge,
    already fallback-filled), overwrite fl_k_rate_bridge/fl_bb_rate_bridge for
    any row that gets a real match via the college name extension and/or
    Pioneer/MiLB -- the SAME widening fl_production_projection.py's
    build_2027_projections applies to the real roster."""
    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    out = df.merge(names, on="pitcher_id", how="left")
    out = prod.extend_match_by_name(out, prospect_proj)
    out = prod.extend_match_by_other_leagues(out, pioneer_full, milb_full)
    return out.drop(columns=["pitcher"])


def enrich_bridge_weighted(df: pd.DataFrame, raw: pd.DataFrame, college_full, pioneer_full,
                            milb_full, current_year: int) -> pd.DataFrame:
    """Variant C: replace fl_k_rate_bridge/fl_bb_rate_bridge/has_college_match
    entirely with bm.recency_weighted_match's blended output, instead of
    overwriting only rows the narrow match missed -- the new design can
    override even a real college match if Pioneer/MiLB has a more recent
    and/or larger-sample record for the same player (the gap the fixed
    priority order couldn't express; see DECISION_LOG.md)."""
    import fl_bridge_matching as bm
    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    ids_df = df[["pitcher_id"]].merge(names, on="pitcher_id", how="left")
    matched = bm.recency_weighted_match(ids_df["pitcher_id"], ids_df["pitcher"],
                                         college_full, pioneer_full, milb_full, current_year)

    # df's own fl_k_rate_bridge/fl_bb_rate_bridge are already fallback-filled
    # (bt.assemble_features fills unmatched rows with a single lg_k_rate/
    # lg_bb_rate constant) -- capture that exact constant from an unmatched
    # row before dropping, so a player the new match ALSO misses keeps the
    # same real league-average fallback, not a re-derived (and wrong, since
    # every row would already be non-null) stand-in.
    unmatched = ~df["has_college_match"]
    fallback_k = df.loc[unmatched, "fl_k_rate_bridge"].iloc[0] if unmatched.any() else df["fl_k_rate_bridge"].median()
    fallback_bb = df.loc[unmatched, "fl_bb_rate_bridge"].iloc[0] if unmatched.any() else df["fl_bb_rate_bridge"].median()

    out = df.drop(columns=["fl_k_rate_bridge", "fl_bb_rate_bridge", "has_college_match"]).merge(
        matched[["pitcher_id", "fl_k_rate_bridge", "fl_bb_rate_bridge", "has_college_match"]],
        on="pitcher_id", how="left")
    out["fl_k_rate_bridge"] = out["fl_k_rate_bridge"].fillna(fallback_k)
    out["fl_bb_rate_bridge"] = out["fl_bb_rate_bridge"].fillna(fallback_bb)
    out["has_college_match"] = out["has_college_match"].fillna(False)
    return out


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("\nFitting college bridge (narrow) + building full-universe college prospect table...")
    college_model = bridge.fit_bridge()
    prospect_proj = prod.build_college_prospect_projections(college_model)

    print("\nFitting Pioneer League bridge + full-universe table...")
    pioneer_model = pbridge.fit_bridge(raw)
    pioneer_full = pbridge.build_full_universe_predictions(pioneer_model)

    print("\nFitting affiliated MiLB bridge + full-universe table...")
    milb_model = mbridge.fit_bridge(raw)
    milb_full = mbridge.build_full_universe_predictions(milb_model)

    print("\nAssembling Experiment-2 train/test panels (narrow college-only bridge, as today)...")
    df_tr, n0_k, n0_bb = bt.assemble(raw, TRAIN_WINDOWS, [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               TEST_LABEL_WINDOWS, with_changepoints=False)

    print(f"  train match coverage (narrow): {int(df_tr['has_college_match'].sum())}/{len(df_tr)}")
    print(f"  test match coverage (narrow): {int(df_te['has_college_match'].sum())}/{len(df_te)}")

    df_tr_rich = enrich_bridge(df_tr, raw, prospect_proj, pioneer_full, milb_full)
    df_te_rich = enrich_bridge(df_te, raw, prospect_proj, pioneer_full, milb_full)

    print("\nBuilding variant C (recency+reliability+sample-size weighted blend)...")
    df_tr_weighted = enrich_bridge_weighted(df_tr, raw, prospect_proj, pioneer_full, milb_full, current_year=2026)
    df_te_weighted = enrich_bridge_weighted(df_te, raw, prospect_proj, pioneer_full, milb_full, current_year=2026)

    print("\n" + "=" * 70)
    print("Held-out test: narrow college bridge vs. widened bridge vs. recency-weighted blend")
    print("=" * 70)
    results = []
    for name, (tr, te) in [("A: narrow college-only bridge (current production)", (df_tr, df_te)),
                            ("B: college(narrow+name) + Pioneer + MiLB, fixed priority", (df_tr_rich, df_te_rich)),
                            ("C: recency+reliability+TBF weighted blend", (df_tr_weighted, df_te_weighted))]:
        k_res = bt.run_heldout(tr, "K_rate_actual", SPEC["K"], te, "K_rate_actual", n0_k, f"{name} [K%]")
        bb_res = bt.run_heldout(tr, "BB_rate_actual", SPEC["BB"], te, "BB_rate_actual", n0_bb, f"{name} [BB%]")
        results += [k_res, bb_res]
    print(pd.DataFrame(results).to_string(index=False))

    print("\n--- Subgroup check: matched vs. unmatched, widened-bridge test set ---")
    for matched, label in [(True, "matched"), (False, "unmatched")]:
        sub_te = df_te_rich[df_te_rich["has_college_match"] == matched]
        k_res = bt.run_heldout(df_tr_rich, "K_rate_actual", SPEC["K"], sub_te, "K_rate_actual",
                                n0_k, f"{label} [K%]")
        bb_res = bt.run_heldout(df_tr_rich, "BB_rate_actual", SPEC["BB"], sub_te, "BB_rate_actual",
                                 n0_bb, f"{label} [BB%]")
        print(f"  {label}: n={len(sub_te)}  K%: {k_res}  BB%: {bb_res}")
