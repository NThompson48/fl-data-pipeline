"""
fl_trustmark_hr_factor_check.py -- how much of the raw Trustmark HR suppression is real?

The Mississippi translation uses the OFFICIAL park_factors.pf_hr = 0.624 for the
HR-per-air-ball ratio, deliberately not the raw 0.39 ("unregressed"). On all 80k batted
balls the raw ratio is ~0.37 (66 HR vs ~177 expected). This checks:

 A. SAME-TEAM home/road comparison for Mississippi (removes "Mississippi's players are
    different"): Mississippi pitchers at Trustmark vs on the road, and Mississippi batters
    at Trustmark vs on the road, plus the same for visitors vs their own other games.
 B. The raw HR/air-ball ratio of every park, its Poisson noise, and an empirical-Bayes
    estimate of the true park spread (method of moments on log ratios) -> how far a park's
    raw ratio should be shrunk toward 1, and where Trustmark lands.
 C. How the official pf_hr table relates to the raw park ratios across parks.
 D. What each candidate HR factor implies for the Mississippi shift (full / half season).

Usage:  python fl_trustmark_hr_factor_check.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from fl_supabase import read_sql_df
import fl_trustmark_air_validation as v

MIS = "Mississippi Mud Monsters"


def ratio_ci(h1, e1, h0, e0):
    """HR/air ball ratio (park vs comparison) with a Poisson-based 90% interval."""
    r = (h1 / e1) / (h0 / e0)
    se = r * np.sqrt(1 / max(h1, 1) + 1 / max(h0, 1))
    return r, r * np.exp(-1.645 * np.sqrt(1 / max(h1, 1) + 1 / max(h0, 1))), r * np.exp(1.645 * np.sqrt(1 / max(h1, 1) + 1 / max(h0, 1)))


def main():
    b = v.load()
    air = b[b["air"]].copy()
    print(f"air balls: {len(air):,}; HR on air balls {int(air['hr'].sum())}")
    print("\nA. SAME-TEAM HOME/ROAD (HR per air ball)")
    rows = []
    for who, col in (("Mississippi PITCHERS", "pitcher_team"), ("Mississippi BATTERS", "batter_team")):
        g = air[air[col] == MIS]
        h, r_ = g[g["is_T"]], g[~g["is_T"]]
        r, lo, hi = ratio_ci(h["hr"].sum(), len(h), r_["hr"].sum(), len(r_))
        rows.append({"group": who, "air balls at T / road": f"{len(h)} / {len(r_)}", "HR at T / road": f"{int(h['hr'].sum())} / {int(r_['hr'].sum())}",
                     "HR rate at T": h["hr"].mean(), "road": r_["hr"].mean(), "ratio": r, "90% CI": f"[{lo:.2f}, {hi:.2f}]"})
    # visitors: batters/pitchers NOT Mississippi, at T vs elsewhere (their own other games)
    for who, col in (("VISITING pitchers (non-MIS)", "pitcher_team"), ("VISITING batters (non-MIS)", "batter_team")):
        g = air[air[col] != MIS]
        h, r_ = g[g["is_T"]], g[~g["is_T"]]
        r, lo, hi = ratio_ci(h["hr"].sum(), len(h), r_["hr"].sum(), len(r_))
        rows.append({"group": who, "air balls at T / road": f"{len(h)} / {len(r_)}", "HR at T / road": f"{int(h['hr'].sum())} / {int(r_['hr'].sum())}",
                     "HR rate at T": h["hr"].mean(), "road": r_["hr"].mean(), "ratio": r, "90% CI": f"[{lo:.2f}, {hi:.2f}]"})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nB. EVERY PARK: raw HR/air-ball ratio vs the rest of the league, noise, empirical-Bayes shrinkage")
    lg_rate = air["hr"].mean()
    ps = air.groupby("venue").agg(air_balls=("hr", "size"), hr=("hr", "sum")).reset_index()
    ps["expected"] = ps["air_balls"] * lg_rate
    ps["raw_ratio"] = ps["hr"] / ps["expected"]
    ps["log_ratio"] = np.log(ps["raw_ratio"].clip(lower=0.05))
    ps["noise_var"] = 1.0 / ps["hr"].clip(lower=1)         # Poisson: Var(log ratio) ~ 1/HR
    tau2 = max(ps["log_ratio"].var(ddof=1) - ps["noise_var"].mean(), 1e-4)
    ps["shrink_w"] = tau2 / (tau2 + ps["noise_var"])
    ps["eb_ratio"] = np.exp(ps["shrink_w"] * ps["log_ratio"])
    print(f"   between-park SD of true log HR factor (method of moments): {np.sqrt(tau2):.3f}  "
          f"(observed SD of log ratios {ps['log_ratio'].std(ddof=1):.3f}, mean Poisson noise SD {np.sqrt(ps['noise_var'].mean()):.3f})")
    pf = read_sql_df(text("select team_name, pf_runs, pf_hr, n_seasons, league from park_factors")).copy()
    print("   park_factors table leagues:", pf["league"].value_counts().to_dict())
    pf_fl = pf[pf["league"].str.contains("Frontier|FL|fron", case=False, na=False)]
    teams = read_sql_df(text("select team, venue from park_factors_fip where year='2025-2026'"))
    ps = ps.merge(teams.rename(columns={"venue": "venue"}), on="venue", how="left")
    ps = ps.merge(pf_fl[["team_name", "pf_hr", "pf_runs", "n_seasons"]], left_on="team", right_on="team_name", how="left")
    print(ps.sort_values("raw_ratio")[["team", "venue", "air_balls", "hr", "raw_ratio", "shrink_w", "eb_ratio", "pf_hr", "pf_runs", "n_seasons"]].round(3).to_string(index=False))
    t = ps[ps["venue"] == v.TRUST].iloc[0]
    print(f"\n   Trustmark: raw {t['raw_ratio']:.3f} ({int(t['hr'])} HR vs {t['expected']:.0f} expected), shrink weight {t['shrink_w']:.2f} -> empirical-Bayes {t['eb_ratio']:.3f}; "
          f"official pf_hr {t['pf_hr']:.3f}")
    ok = ps.dropna(subset=["pf_hr"])
    print(f"\nC. official pf_hr vs raw park ratio across {len(ok)} parks: corr {ok['raw_ratio'].corr(ok['pf_hr']):.2f}, "
          f"slope of pf_hr on raw ratio {np.polyfit(ok['raw_ratio'], ok['pf_hr'], 1)[0]:.2f}; SD of pf_hr {ok['pf_hr'].std():.3f} vs SD of raw ratio {ok['raw_ratio'].std():.3f} "
          f"vs SD of EB ratio {ok['eb_ratio'].std():.3f}")

    print("\nD. WHAT EACH HR FACTOR IMPLIES FOR THE MISSISSIPPI SHIFT (roster mean; runs/9, negative = fewer runs)")
    import fl_ra9_trustmark_projection as ra9
    r = pd.read_parquet("../projections-dashboard/data/fl_ra9_trustmark_projection.parquet")
    p = pd.read_parquet("../projections-dashboard/data/fl_2027_projections.parquet")[["pitcher_id", "PA"]]
    d = r.merge(p, on="pitcher_id"); d = d[d["PA"] >= 30].copy()
    d["proj_bf"] = 100 * ra9.LEAGUE_BF_PER_IP; d["proj_ip"] = 100.0
    rows = []
    for nm, f in (("official pf_hr (shipped)", ra9.TRUSTMARK_HR_FACTOR), ("empirical-Bayes", float(t["eb_ratio"])), ("raw all-pitcher ratio", float(t["raw_ratio"]))):
        old = ra9.TRUSTMARK_HR_FACTOR
        ra9.TRUSTMARK_HR_FACTOR = f
        half = ra9.build_trustmark_adjustment(d, 0.5)["trustmark_ra9_shift"].mean()
        full = ra9.build_trustmark_adjustment(d, 1.0)["trustmark_ra9_shift"].mean()
        ra9.TRUSTMARK_HR_FACTOR = old
        rows.append({"HR factor": nm, "value": round(f, 3), "half-season shift": round(half, 3), "full-season shift": round(full, 3)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
