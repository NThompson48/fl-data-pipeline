"""
fl_hitter_wrcplus_target_test.py -- should the hitter model target be park-adjusted wRC+ (user request 2026-10-08: "park-adjust it too with just the wOBA factor")?
Park adjustment = the published team wOBA park factor: wOBA_adj = wOBA - k (PF/100 - 1), k = .15 calibrated on the published 2025-26 wRC+ (adjustment per PF point; the published adjustments are -k (PF-100)/100 + const with k .138 / .157,
corr -.95 / -.995), recentred so the league level is unchanged.  PF per TEAM (2025-26 average of war_players.park_factor), carried back to 2022-24 (no home / road data exist for those years: an assumption).
wRC+_i,y = 100 (1 + (wOBA_adj - league_y) / (1.0175 x league runs per PA_y)); league runs per PA by year as in the history table (.1446 / .1399 / .1320 / .1452 / .1529).
Models, all the production three-season regression (delta .75, n0 refit on the training transitions), target = NEXT-YEAR PARK-ADJUSTED wRC+ (>= 100 PA), standard inverse-variance weights fit on the wRC+ errors:
  A  current: regress park-neutral wOBA (Trustmark profile neutralisation; essentially raw) in wOBA units, convert the prediction to wRC+ with next year's league level and run environment
  B  regress park-ADJUSTED wOBA in wOBA units, convert
  C  regress park-adjusted wRC+ directly in wRC+ units (the proposal)
Leave-one-transition-out and forward (2024->25, 2025->26); weighted RMSE and MAE in wRC+ points; hitter-cluster bootstrap.  Also checks the constructed wRC+ against the published wRC+ for 2025-26.
"""
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import text

import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_baseline_backtest as bt
import fl_hitter_depth3_test as d3
import fl_hitter_model_dev as dev
import fl_innings_projection as ip
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
SC = 1.0175; K = 0.15; RPA = {2022: 0.1446, 2023: 0.1399, 2024: 0.1320, 2025: 0.1452, 2026: 0.1529}; DELTA = 0.75


def build_panel():
    p = dev.load(); pf = read_sql_df(text("select team, year, park_factor from war_players where game_type='reg_season' and pa>=50")).groupby("team")["park_factor"].mean()
    code = {"eo11469": "Evansville Otters", "fy11571": "Florence Y'alls", "gg11692": "Gateway Grizzlies", "js12269": "Joliet Slammers", "lc12432": "Lake Erie Crushers", "nb13412": "New York Boulders", "nj13281": "New Jersey Jackals",
            "ot13547": "Ottawa Titans", "qc13973": "Quebec Capitales", "sb14460": "Schaumburg Boomers", "sm14839": "Sussex County Miners", "ta15559": "Trois-Rivieres Aigles", "tv15042": "Tri-City ValleyCats",
            "wt15238": "Washington Wild Things", "wt15425": "Windy City ThunderBolts"}          # StatsCrew team codes (2022-24); Empire State Greys and New England Knockouts have no current park factor
    p["team_name"] = p["team"].map(lambda t: code.get(t, t)); miss = sorted(set(p["team_name"].dropna().unique()) - set(pf.index)); print("teams without a park factor (adjustment 0):", miss)
    p["pf"] = p["team_name"].map(pf); p["adj"] = (-K * (p["pf"] / 100 - 1)).fillna(0.0)
    for y, g in p.groupby("year"):
        gg = g.dropna(subset=["woba_raw"]); p.loc[g.index, "adj"] = g["adj"] - np.average(gg["adj"], weights=gg["pa"])             # league level unchanged
    p["woba_pa"] = p["woba_raw"] + p["adj"]
    lg = {y: np.average(g.dropna(subset=["woba_pa"])["woba_pa"], weights=g.dropna(subset=["woba_pa"])["pa"]) for y, g in p.groupby("year")}
    p["wrc"] = [100 * (1 + (w - lg[int(y)]) / (SC * RPA[int(y)])) if pd.notna(w) else np.nan for w, y in zip(p["woba_pa"], p["year"])]
    return p, lg


def main():
    p, lg = build_panel()
    pub = read_sql_df(text("select player, year, wrc_plus from war_players where game_type='reg_season' and year in (2025, 2026)")); pub["name_clean"] = pub["player"].apply(ip.lastfirst_to_clean); pub = pub.rename(columns={"wrc_plus": "wrc_pub"})
    j = p.merge(pub.drop_duplicates(["name_clean", "year"])[["name_clean", "year", "wrc_pub"]], on=["name_clean", "year"]); j = j[j["pa"] >= 100]
    print(f"constructed vs published wRC+ (2025-26, >= 100 PA, n = {len(j)}): corr {j['wrc'].corr(j['wrc_pub']):.3f}, mean diff {np.average(j['wrc'] - j['wrc_pub'], weights=j['pa']):+.2f}, RMS diff {np.sqrt(np.average((j['wrc'] - j['wrc_pub']) ** 2, weights=j['pa'])):.2f} wRC+ points; adjustment SD {p['adj'].std():.4f} wOBA")
    frames = {}
    for tag, col in (("A", "woba_n"), ("B", "woba_pa"), ("C", "wrc")):
        q = p.copy(); q["woba"] = q[col]; frames[tag] = d3.frame(q, "woba")
    key = ["name_clean", "t"]; base = frames["C"][key + ["x_next", "w", "lg_next", "n_next"]].copy()
    for tag in ("A", "B"):
        f = frames[tag][key + ["x_next", "lg_next"]].rename(columns={"x_next": f"xn_{tag}", "lg_next": f"lg_{tag}"}); base = base.merge(f, on=key, how="inner")
    base = base.merge(frames["C"][key + ["x_t", "x_prev", "n_t", "n_prev", "lg_t", "lg_prev", "x_pp", "n_pp", "lg_pp"]], on=key); print(f"{len(base)} hitter-transitions common to all three")
    out = {}
    for scheme, plan in (("LEAVE-ONE-TRANSITION-OUT", [(tuple(t for t in (2022, 2023, 2024, 2025) if t != T), T) for T in (2022, 2023, 2024, 2025)]), ("FORWARD (tests 2024->25, 2025->26)", [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)])):
        P = {k: [] for k in "ABC"}; Y, I = [], []
        for tr, te in plan:
            for tag in "ABC":
                d = frames[tag]; n0 = d3.fit_n0(d[d["t"].isin(tr)], DELTA, 3); dte = d[d["t"] == te].merge(base[key], on=key, how="inner"); dte = dte.set_index(key).loc[base[base["t"] == te].set_index(key).index].reset_index()
                pr = d3.pred(dte, n0, DELTA, 3).to_numpy()
                if tag in "AB":                                                      # convert the wOBA prediction to next year's wRC+ scale
                    pr = 100 * (1 + (pr - dte["lg_next"].to_numpy()) / (SC * RPA[te + 1]))
                P[tag].append(pr)
            bt_ = base[base["t"] == te]; Y.append(bt_["x_next"].to_numpy()); I.append(bt_["name_clean"].to_numpy())
        y, ids = np.concatenate(Y), np.concatenate(I); PP = {k: np.concatenate(v) for k, v in P.items()}; w0 = base[base["t"].isin([t for _, t in plan])]
        w0 = pd.concat([base[base["t"] == te] for _, te in plan])["n_next"].to_numpy(float); a, b = ew.fit_variance_law(y - PP["C"], w0); w = ew.inv_var_weights(w0, a, b)
        print(f"\n{'=' * 100}\n{scheme}: n = {len(y)} hitter-transitions; target = next-year park-adjusted wRC+ (SD {y.std():.1f}); weights 1/({a:.1f} + {b:.0f}/PA)\n{'=' * 100}")
        rows = []
        for tag, lab in (("A", "A current (park-neutral wOBA, wOBA units)"), ("B", "B park-ADJUSTED wOBA, wOBA units"), ("C", "C park-adjusted wRC+, wRC+ units (proposal)")):
            r, m = er.wmetrics(y, PP[tag], w); row = {"model": lab, "RMSE": r, "MAE": m}
            if tag != "A":
                rep = er.compare(y, PP["A"], PP[tag], w, ids, n=600); row.update({"dRMSE vs A": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(better)": rep["P(RMSE better)"], "dMAE vs A": rep["dMAE"], "P(MAE)": rep["P(MAE better)"]})
            rows.append(row)
        print(er.table(rows, 3)); rep = er.compare(y, PP["B"], PP["C"], w, ids, n=600); print(f"  C vs B (same park adjustment, wRC+ units vs wOBA units): dRMSE {rep['dRMSE']:+.3f} {rep['RMSE 90%']}, P(better) {rep['P(RMSE better)']:.2f}; dMAE {rep['dMAE']:+.3f}, P {rep['P(MAE better)']:.2f}")


if __name__ == "__main__":
    main()
