"""
fl_pitcher_pre2026.py -- the PUBLISHED-style preseason K% / BB% for 2026 (what the production projection model would have said before the season): the frozen Experiment-2 model
(fit full-2025 -> actual 1H2026) applied to full-2025 features only, exactly the way fl_production_projection.build_2027_projections applies it to 2025+2026 features for 2027.

Leak note (honest): the frozen coefficients were fit on 1H2026 outcomes (2026-05-07 .. 2026-07-07), so these projections are out of sample ONLY for targets after 2026-07-07
(the same split fl_halfseason_backtest scores as fold 2).  The ROS test that uses them is therefore restricted to as-of dates on or after 2026-07-08.
Output: output/pitcher_pre2026_published.parquet  (pitcher_id, name_clean, pre_k, pre_bb, pa_model)
"""
import numpy as np
import pandas as pd

import fl_halfseason_backtest as bt
import fl_innings_projection as ip
import fl_pitcher_features as fp
import fl_production_projection as pp
import mlb_marcel as mm


def build():
    raw = fp.load_fl_pitches_cached()
    m_k, m_bb, n0_k, n0_bb, n0_bridge_bb, rho = pp.fit_frozen_model(raw)
    feat, n0_k, n0_bb = bt.assemble_features(raw, [(bt.START_2025, bt.END_2025)], with_changepoints=False)
    term = fp.filter_period(raw, bt.START_2025, bt.END_2025); term = term[term["is_terminal"]]
    lg_bb = (term["k_or_bb"] == "walk").mean()
    feat["BB_raw"] = feat["naive_BB_rate"] * (feat["PA"] + n0_bb) - n0_bb * lg_bb
    feat["blended_BB_rate"] = mm.blended_rate(feat["BB_raw"], feat["PA"], n0_bb, lg_bb, n0_bridge_bb, feat["fl_bb_rate_bridge"])
    d_k = feat.dropna(subset=pp.SPEC["K"]).copy(); d_bb = feat.dropna(subset=pp.BB_SPEC_BLENDED).copy()
    d_k["pre_k"] = m_k.predict(d_k); d_bb["pre_bb"] = m_bb.predict(d_bb)
    out = feat[["pitcher_id", "PA"]].merge(d_k[["pitcher_id", "pre_k"]], on="pitcher_id", how="left").merge(d_bb[["pitcher_id", "pre_bb"]], on="pitcher_id", how="left")
    import fl_pitcher_ros_stuff as pst
    out["name_clean"] = out["pitcher_id"].map(pst.idmap(raw))
    out = out.rename(columns={"PA": "pa_model"})
    out.to_parquet("output/pitcher_pre2026_published.parquet", index=False)
    print(out.describe().round(4).to_string()); print(len(out), "pitchers;", out["pre_k"].notna().sum(), "with K;", out["pre_bb"].notna().sum(), "with BB")
    return out


if __name__ == "__main__":
    build()
