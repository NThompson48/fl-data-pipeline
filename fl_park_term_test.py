"""
fl_park_term_test.py -- which PARK term should the RA9 baseline use, and does it
agree with how Mississippi's TrustmarkPark is treated?

The RA9 baseline uses war_pitchers.ppf (source code for its construction is not
in this project). On the roster it disagrees with the FIP-based park factors the
Trustmark adjustment is built from: team-level corr(ppf, FIP factor) = 0.61, and
Mississippi pitchers average ppf ~101 although half their games are at a park
whose FIP factor is 93. war_pitchers.pf, by contrast, tracks the FIP factor
almost exactly (corr 0.99).

Candidates for the park regressor (all added to K% + BB% + IP/G, forward and
reversed LOOCV, same frame as the other RA9 checks):
  ppf        war_pitchers.ppf (shipped)
  pf         war_pitchers.pf
  own_mix    the pitcher's OWN game-by-game exposure: BF-weighted mean of the
             FIP-based park factor of each game's home park (from the pitch
             data's home_team) -- built from the same park_factors_fip row the
             Trustmark adjustment uses, so Mississippi pitchers are handled
             consistently
  none

Usage:  python fl_park_term_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import text

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_ra9_input_robustness as rob
from fl_supabase import read_sql_df


def own_mix(raw, start, end, pf_by_team):
    p = fp.filter_period(raw, start, end)
    t = p[p["is_terminal"]].copy()
    t["park_f"] = t["home_team"].map(pf_by_team)
    g = t.groupby("pitcher_id").apply(lambda d: pd.Series({
        "own_mix": np.average(d["park_f"].dropna(), weights=np.ones(d["park_f"].notna().sum())) if d["park_f"].notna().any() else np.nan,
        "home_share": (d["pitcher_team"] == d["home_team"]).mean()}), include_groups=False).reset_index()
    g["pitcher_id"] = g["pitcher_id"].astype(str)
    return g


def loocv_park(m, park_col):
    n = len(m); p = np.full(n, np.nan)
    spec = "target ~ K_pred + BB_pred + ip_per_g_prior" + (f" + {park_col}" if park_col else "")
    for i in range(n):
        tr = m.drop(index=i); te = m.loc[[i]].copy()
        mod = smf.wls(spec, data=tr, weights=tr["w"]).fit()
        for c in ("ip_per_g_prior", park_col):
            if c:
                te[c] = te[c].clip(tr[c].min(), tr[c].max())
        p[i] = float(mod.predict(te).iloc[0])
    return p


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached(); raw["date"] = pd.to_datetime(raw["date"])
    war = read_sql_df(text("select player, year, team, bf, ip, so, bb, ra9, ppf, pf, g from war_pitchers where game_type='reg_season'"))
    pff = read_sql_df(text("select team, park_factor from park_factors_fip where year='2025-2026'")).set_index("team")["park_factor"]
    print("home_team values covered by park table:", raw["home_team"].isin(pff.index).mean().round(3))
    ids = pd.DataFrame({"pitcher_id": raw["pitcher_id"].astype(str).unique(), "K_oof": 0.0, "BB_oof": 0.0})
    mixes = {2025: own_mix(raw, bt.START_2025, bt.END_2025, pff.to_dict()), 2026: own_mix(raw, bt.START_2026, bt.END_2026, pff.to_dict())}
    for lab, y_in, y_tgt, win in (("FORWARD 2025 -> 2026", 2025, 2026, (bt.START_2025, bt.END_2025)),
                                  ("REVERSED 2026 -> 2025", 2026, 2025, (bt.START_2026, bt.END_2026))):
        m = rob.build_frame(war, raw, ids, y_in, y_tgt, win)
        m["pitcher_id"] = m["pitcher_id"].astype(str)
        pfw = war[war["year"] == y_tgt][["player", "pf"]]
        m = m.merge(pfw, on="player", how="left").merge(mixes[y_tgt][["pitcher_id", "own_mix", "home_share"]], on="pitcher_id", how="left")
        m = m.dropna(subset=["pf", "own_mix"]).reset_index(drop=True)
        m["K_pred"], m["BB_pred"] = m["K_raw"], m["BB_raw"]
        print(f"\n=== {lab} (n={len(m)}) -- park term is the TARGET season's exposure (known in advance from the roster spot) ===")
        print("corr among park measures: ppf~pf %.2f  ppf~own_mix %.2f  pf~own_mix %.2f" % (m.ppf.corr(m.pf), m.ppf.corr(m.own_mix), m.pf.corr(m.own_mix)))
        res = []
        for nm, col in (("none", None), ("ppf (shipped)", "ppf"), ("pf", "pf"), ("own game-mix FIP factor", "own_mix")):
            r = rob.summarize(m, loocv_park(m, col), nm)
            full = smf.wls("target ~ K_pred + BB_pred + ip_per_g_prior" + (f" + {col}" if col else ""), data=m, weights=m["w"]).fit()
            res.append({"park term": nm, "R2_w": r["R2_w"], "RMSE_w": r["RMSE_w"], "coef (runs/9 per pt)": "" if not col else f"{full.params[col]:+.3f} (t={full.tvalues[col]:+.1f})"})
        print(pd.DataFrame(res).to_string(index=False))
    # Mississippi's own pitchers in 2026
    mi = mixes[2026].merge(raw[["pitcher_id", "pitcher_team"]].drop_duplicates("pitcher_id").assign(pitcher_id=lambda d: d["pitcher_id"].astype(str)), on="pitcher_id")
    ms = mi[mi["pitcher_team"] == "Mississippi Mud Monsters"]
    print(f"\nMississippi pitchers 2026 (n={len(ms)}): mean own game-mix FIP factor {ms['own_mix'].mean():.1f} (expect ~{0.5*pff['Mississippi Mud Monsters']+0.5*100:.1f} at 50% home), home share {ms['home_share'].mean():.2f}")
    print("war ppf for the same team (BF>=100 pitchers):", war[(war.year == 2026) & (war.team == "Mississippi Mud Monsters") & (war.bf >= 100)]["ppf"].mean().round(1))
