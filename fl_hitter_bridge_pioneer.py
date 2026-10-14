"""
fl_hitter_bridge_pioneer.py -- PROVISIONAL Pioneer League (2026) -> Frontier League hitter projections for 2027 (user data 2026-10-10; fl_pioneer_hitting_2026.py).
PROVISIONAL because the mapping is the one FITTED ON AFFILIATED MiLB (fl_hitter_bridge.py: pooled slope per stat, n0, intercept; the mapping is level-free, u is a deviation from the source league's own mean), applied to deviations from the PIONEER 2026 league mean (PA-weighted, 11 of 12 teams).
There is no pre-2026 Pioneer hitting in the project, so there are no Pioneer -> FL debut pairs to fit or validate a Pioneer-specific mapping; whether a Pioneer deviation is worth the same as a MiLB deviation is an assumption.  Same safeguards as the MiLB table: u clipped to the MiLB training range, 'beyond_validated_range' flag.
Source rows: one season (2026), lag 0 for a 2027 projection; u_s = dev_s x PA / (PA + n0_s); dev_FL = intercept + slope x u; wOBA = 0.5 direct + 0.5 component-built (K -.177, BB +.516, HR +1.414); rates = FL 2026 league mean + deviation; WAR / 400 with the production constants and no fielding / position.
Outputs: output/fl_hitter_pioneer_prospects.parquet, output/fl_hitter_pioneer_history.parquet (same schema as the MiLB files; copied to the dashboard data folder with --publish).
"""
import json
import shutil
import sys

import numpy as np
import pandas as pd

BETA = {"k": -0.177, "bbr": 0.516, "hrr": 1.414}


def main(publish=False):
    P = pd.read_parquet("output/pioneer_hitting_2026.parquet"); P = P[P["PA"] > 0].copy(); J = json.load(open("output/fl_hitter_bridge_coefs.json")); co, ct = J["coefs"], J["constants"]
    lg = {s: float(np.average(P.loc[P[s].notna(), s], weights=P.loc[P[s].notna(), "PA"])) for s in ("woba", "k", "bbr", "hrr")}; dev = {}; beyond = np.zeros(len(P), bool)
    for s in ("woba", "k", "bbr", "hrr"):
        c = co[s]; u = (P[s].fillna(lg[s]) - lg[s]) * P["PA"] / (P["PA"] + c["n0"]); beyond |= ((u > c["u_p95"]) | (u < c["u_p5"])).to_numpy() if s == "woba" else False; dev[s] = c["intercept"] + c["slope"] * u.clip(c["u_min"], c["u_max"])
    L = ct["lg"]; sc = ct["woba_scale"]; comp = sum(BETA[s] * dev[s] for s in BETA)
    out = pd.DataFrame({"nm": P["name_clean"], "name": P["name"], "batter": P["br_id"], "latest_season": 2026, "latest_level": "Pioneer", "latest_team": "Pioneer League", "bat": P["bat"], "age": P["age"], "src_pa": P["PA"]})
    out["proj_k"] = L["k"] + dev["k"]; out["proj_bbr"] = L["bbr"] + dev["bbr"]; out["proj_hrr"] = L["hrr"] + dev["hrr"]; out["proj_woba_direct"] = L["woba"] + dev["woba"]; out["proj_woba_comp"] = L["woba"] + comp; out["proj_woba"] = 0.5 * out["proj_woba_direct"] + 0.5 * out["proj_woba_comp"]
    out["proj_wrc_plus"] = 100 * (1 + (out["proj_woba"] - L["woba"]) / (sc * ct["lg_runs_per_pa"])); out["proj_war_per400"] = 400 * ((out["proj_woba"] - L["woba"]) / sc + ct["repl_per_pa"]) / ct["rpw"]
    cw = co["woba"]; sd = float(np.sqrt(cw["law_a"] + cw["law_b"] / 400)); out["woba_sd_400pa"] = sd; out["woba_80_lo"] = out["proj_woba"] - 1.2816 * sd; out["woba_80_hi"] = out["proj_woba"] + 1.2816 * sd; out["beyond_validated_range"] = beyond
    pan = pd.read_parquet("output/fl_hitter_panel.parquet"); out["has_fl_history"] = out["nm"].isin(set(pan["name_clean"])); mp = pd.read_parquet("output/fl_hitter_milb_prospects.parquet"); out["also_in_milb"] = out["nm"].isin(set(mp["nm"]))
    out = out.sort_values("proj_woba", ascending=False).reset_index(drop=True); out.to_parquet("output/fl_hitter_pioneer_prospects.parquet", index=False)
    hist = pd.DataFrame({"nm": P["name_clean"], "player_name": P["name"], "season": 2026, "level": "Pioneer", "team": "Pioneer League", "pa": P["PA"], "k": P["k"], "bbr": P["bbr"], "hrr": P["hrr"], "woba": P["woba"]}); hist.to_parquet("output/fl_hitter_pioneer_history.parquet", index=False)
    print(f"Pioneer 2026 league means: wOBA proxy {lg['woba']:.4f}, K% {lg['k']:.4f}, BB% {lg['bbr']:.4f}, HR/PA {lg['hrr']:.4f}  ->  FL 2026: wOBA {L['woba']:.4f}, K% {L['k']:.4f}, BB% {L['bbr']:.4f}, HR/PA {L['hrr']:.4f}")
    q = out[out["src_pa"] >= 150]; print(f"{len(out)} hitters ({len(q)} with >= 150 PA, {int(out['has_fl_history'].sum())} already in the FL data, {int(out['also_in_milb'].sum())} also in the MiLB table); projected wOBA among >= 150 PA: mean {q['proj_woba'].mean():.4f}, SD {q['proj_woba'].std():.4f}, wRC+ range {q['proj_wrc_plus'].min():.0f}-{q['proj_wrc_plus'].max():.0f}; beyond validated range {q['beyond_validated_range'].mean():.0%}")
    print(q.head(8)[["name", "bat", "age", "src_pa", "proj_woba", "proj_wrc_plus", "proj_k", "proj_bbr", "proj_hrr", "proj_war_per400"]].round(3).to_string(index=False))
    if publish:
        for f in ("fl_hitter_pioneer_prospects.parquet", "fl_hitter_pioneer_history.parquet"):
            shutil.copy(f"output/{f}", f"../projections-dashboard/data/{f}")
        print("published to the dashboard data folder")


if __name__ == "__main__":
    main(publish="--publish" in sys.argv)
