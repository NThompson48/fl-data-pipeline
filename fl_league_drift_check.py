"""
fl_league_drift_check.py -- does the frozen K%/BB% model carry a league-LEVEL
bias that a simple recentering to a known league average would fix?

The regression is fit on one transition (full-2025 features -> 1H2026
actuals), so its intercept bakes in that transition's league level. If the
league-wide K%/BB% level moves between seasons (ball, umpires, talent pool,
roster churn), every projection inherits a common offset that no per-pitcher
feature can see.

Part 1: league PA-weighted K%/BB% by half-season -- how big is drift really?
Part 2: the shipped Exp-2 held-out transition (train 2025 -> 1H2026; roll to
        2025+1H2026 -> actual 2H2026): PA-weighted mean bias of the frozen
        model, and what happens to held-out R2_w/RMSE_w/bias if projections
        are RECENTERED so their PA-weighted mean equals the league average
        observed in the feature window (the only level knowable at projection
        time). Variants: (a) additive shift, (b) shift weighted by feature-
        window PA only for pitchers with real data.
Part 3: the same question for the shipped 2027 projections themselves.

Usage:  python fl_league_drift_check.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
from fl_production_projection import SPEC, BB_SPEC_BLENDED


def league_rates(raw, s, e):
    t = fp.filter_period(raw, s, e)
    t = t[t["is_terminal"]]
    return len(t), (t["k_or_bb"] == "strikeout").mean(), (t["k_or_bb"] == "walk").mean()


def fit_frozen(raw):
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                     with_changepoints=False)
    term = fp.filter_period(raw, bt.START_2025, bt.END_2025)
    lg_bb_tr = (term[term["is_terminal"]]["k_or_bb"] == "walk").mean()
    d_k = df_tr.dropna(subset=["K_rate_actual"] + SPEC["K"])
    d_bb = df_tr.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score"]).copy()
    w_k = mm.reliability_weight(d_k["PA_test"], n0_k).values
    w_bb = mm.reliability_weight(d_bb["PA_test"], n0_bb).values
    m_k = bt.fit_weighted(d_k, "K_rate_actual", SPEC["K"], w_k)
    d_bb["BB_raw"] = d_bb["naive_BB_rate"] * (d_bb["PA"] + n0_bb) - n0_bb * lg_bb_tr
    n0b = mm.fit_n0_bridge(d_bb["BB_raw"], d_bb["PA"], n0_bb, lg_bb_tr, d_bb["fl_bb_rate_bridge"],
                           d_bb["BB_rate_actual"], w_bb)
    d_bb["blended_BB_rate"] = mm.blended_rate(d_bb["BB_raw"], d_bb["PA"], n0_bb, lg_bb_tr, n0b, d_bb["fl_bb_rate_bridge"])
    m_bb = bt.fit_weighted(d_bb, "BB_rate_actual", BB_SPEC_BLENDED, w_bb)
    return m_k, m_bb, n0b


def report(label, y, p, w_eval, w_level):
    """w_eval: reliability weight (as the project scores); w_level: PA weights
    for league-level bias (what 'league average' means)."""
    return {"variant": label,
            "R2_w": round(mm.weighted_r2(y, p, w_eval), 4),
            "RMSE_w_pp": round(mm.weighted_rmse(y, p, w_eval) * 100, 3),
            "PA-wtd bias_pp (pred-actual)": round((np.average(p, weights=w_level) - np.average(y, weights=w_level)) * 100, 3)}


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()

    print("PART 1 -- league PA-weighted rates by window")
    rows = []
    for lab, (s, e) in {"1H2025": (bt.START_2025, bt.CUTOFF_2025), "2H2025": ("2025-07-05", bt.END_2025),
                        "1H2026": (bt.START_2026, bt.CUTOFF_2026), "2H2026": ("2026-07-08", bt.END_2026),
                        "FULL2025": (bt.START_2025, bt.END_2025), "FULL2026": (bt.START_2026, bt.END_2026)}.items():
        n, k, b = league_rates(raw, s, e)
        rows.append({"window": lab, "PA": n, "K%": round(k, 4), "BB%": round(b, 4), "K-BB%": round(k - b, 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nPART 2 -- Exp-2 held-out transition: bias and recentering")
    m_k, m_bb, n0b = fit_frozen(raw)
    te, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                                  [("2026-07-08", bt.END_2026)], with_changepoints=False)
    feat_pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                           fp.filter_period(raw, bt.START_2026, bt.CUTOFF_2026)], ignore_index=True)
    term = feat_pool[feat_pool["is_terminal"]]
    lg_k_feat, lg_bb_feat = (term["k_or_bb"] == "strikeout").mean(), (term["k_or_bb"] == "walk").mean()
    # league level in the MOST RECENT window only (what the production recenter would use)
    _, lg_k_recent, lg_bb_recent = league_rates(raw, bt.START_2026, bt.CUTOFF_2026)

    te["BB_raw"] = te["naive_BB_rate"] * (te["PA"] + n0_bb) - n0_bb * lg_bb_feat
    te["blended_BB_rate"] = mm.blended_rate(te["BB_raw"], te["PA"], n0_bb, lg_bb_feat, n0b, te["fl_bb_rate_bridge"])
    out = []
    for tgt, spec, model, lg_feat, lg_recent in [("K_rate_actual", SPEC["K"], m_k, lg_k_feat, lg_k_recent),
                                                  ("BB_rate_actual", BB_SPEC_BLENDED, m_bb, lg_bb_feat, lg_bb_recent)]:
        d = te.dropna(subset=[tgt] + spec).copy()
        y = d[tgt].values
        w_eval = mm.reliability_weight(d["PA_test"], n0_k if tgt.startswith("K") else n0_bb).values
        w_level = d["PA_test"].values
        p0 = model.predict(d).values
        # recentering uses ONLY information knowable at projection time: feature-window PA weights
        w_feat = d["PA"].values
        rows = [report("as shipped (no recentering)", y, p0, w_eval, w_level)]
        for nm, lg in [("recenter -> pooled feature-window league", lg_feat),
                       ("recenter -> most recent window league", lg_recent)]:
            shift = lg - np.average(p0, weights=w_feat)
            rows.append(report(f"{nm} (shift {shift*100:+.2f}pp)", y, p0 + shift, w_eval, w_level))
        print(f"\n{tgt}  n={len(d)}  actual PA-wtd league level in test window = {np.average(y, weights=w_level):.4f}")
        print(pd.DataFrame(rows).to_string(index=False))

    print("\nPART 3 -- shipped 2027 projections: PA-weighted projected level vs 2026 league level")
    import os
    p = pd.read_parquet(os.path.join("..", "projections-dashboard", "data", "fl_2027_projections.parquet"))
    q = p.dropna(subset=["proj_K_rate", "proj_BB_rate"])
    _, k26, b26 = league_rates(raw, bt.START_2026, bt.END_2026)
    for c, lg in [("proj_K_rate", k26), ("proj_BB_rate", b26)]:
        w = q["PA"].fillna(0).clip(lower=1)
        print(f"  {c}: PA-weighted mean {np.average(q[c], weights=w):.4f} vs 2026 league {lg:.4f} "
              f"(offset {(np.average(q[c], weights=w) - lg) * 100:+.2f}pp)")
