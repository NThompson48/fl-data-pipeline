"""
fl_hitter_bridge_pioneer.py -- Pioneer League -> Frontier League hitter projections (VALIDATED 2026-10-14).
The mapping FITTED ON AFFILIATED MiLB (fl_hitter_bridge.py: n0, decay, intercept, slope, clip per stat) applied to each hitter's Pioneer seasons 2021-2026
(StatsCrew 2021-25 + the user's Baseball Reference 2026 tables, all 12 teams; fl_hitter_bridge_pioneer_fit.pioneer_table), deviations from each Pioneer season's mean.
Validation (fl_hitter_bridge_pioneer_fit.py, 81 FL debut hitters 2023-26 with Pioneer history, leave-one-debut-class-out): this method beats the league mean
(wOBA .0503 -> .0474) and the raw Pioneer line (P 1.00), and a Pioneer-specific fit does NOT beat it (K% / BB% tie, wOBA slightly worse, P(fit better) .07)
-> the MiLB mapping is kept (fit on 200 pairs; the Pioneer-only fit has 81).  Before 2026-10-14 this used 2026 rates only (provisional).
wOBA = 0.5 direct + 0.5 component-built (K -.177, BB +.516, HR +1.414); rates = FL league mean + deviation; WAR / 400 with production constants, no position adjustment.
Outputs: output/fl_hitter_pioneer_prospects.parquet, output/fl_hitter_pioneer_history.parquet (copied to the dashboard data folder with --publish).
"""
import json
import shutil
import sys

import numpy as np
import pandas as pd

BETA = {"k": -0.177, "bbr": 0.516, "hrr": 1.414}


def main(publish=False):
    import fl_hitter_bridge_college as hc
    from fl_hitter_bridge_pioneer_fit import pioneer_table
    t = pioneer_table(); J = json.load(open("output/fl_hitter_bridge_coefs.json")); co, ct = J["coefs"], J["constants"]; tgt = ct["proj_year"]
    ids = t[t["season"] >= tgt - 2].sort_values(["season", "pa"]).groupby("nm").tail(1)[["nm", "name", "team", "season"]].rename(columns={"team": "latest_team", "season": "latest_season"}).reset_index(drop=True)
    ids["year"] = tgt; dev = {}; beyond = np.zeros(len(ids), bool); lg = {s: float(t.loc[t["season"] == 2026, f"lg_{s}"].iloc[0]) for s in ("woba", "k", "bbr", "hrr")}
    for s in ("woba", "k", "bbr", "hrr"):
        c = co[s]; X, _ = hc.features(ids, t, s, "raw", c["n0"], c["decay"]); u = X[:, 0]
        if s == "woba":
            beyond = (u > c["u_p95"]) | (u < c["u_p5"])
        dev[s] = c["intercept"] + c["slope"] * np.clip(u, c["u_min"], c["u_max"])
    br = pd.read_parquet("output/pioneer_hitting_2026.parquet").set_index("name_clean")
    P = ids.assign(name_clean=ids["nm"], br_id=ids["nm"].map(br["br_id"]), bat=ids["nm"].map(br["bat"]), age=ids["nm"].map(br["age"]), PA=ids["nm"].map(t.groupby("nm")["pa"].sum()))
    L = ct["lg"]; sc = ct["woba_scale"]; comp = sum(BETA[s] * dev[s] for s in BETA)
    out = pd.DataFrame({"nm": P["name_clean"], "name": P["name"], "batter": P["br_id"], "latest_season": P["latest_season"], "latest_level": "Pioneer", "latest_team": P["latest_team"], "bat": P["bat"], "age": P["age"], "src_pa": P["PA"]})
    out["proj_k"] = L["k"] + dev["k"]; out["proj_bbr"] = L["bbr"] + dev["bbr"]; out["proj_hrr"] = L["hrr"] + dev["hrr"]; out["proj_woba_direct"] = L["woba"] + dev["woba"]; out["proj_woba_comp"] = L["woba"] + comp; out["proj_woba"] = 0.5 * out["proj_woba_direct"] + 0.5 * out["proj_woba_comp"]
    out["proj_wrc_plus"] = 100 * (1 + (out["proj_woba"] - L["woba"]) / (sc * ct["lg_runs_per_pa"])); out["proj_war_per400"] = 400 * ((out["proj_woba"] - L["woba"]) / sc + ct["repl_per_pa"]) / ct["rpw"]
    cw = co["woba"]; sd = float(np.sqrt(cw["law_a"] + cw["law_b"] / 400)); out["woba_sd_400pa"] = sd; out["woba_80_lo"] = out["proj_woba"] - 1.2816 * sd; out["woba_80_hi"] = out["proj_woba"] + 1.2816 * sd; out["beyond_validated_range"] = beyond
    pan = pd.read_parquet("output/fl_hitter_panel.parquet"); out["has_fl_history"] = out["nm"].isin(set(pan["name_clean"])); mp = pd.read_parquet("output/fl_hitter_milb_prospects.parquet"); out["also_in_milb"] = out["nm"].isin(set(mp["nm"]))
    out = out.sort_values("proj_woba", ascending=False).reset_index(drop=True); out.to_parquet("output/fl_hitter_pioneer_prospects.parquet", index=False)
    hist = t[t["nm"].isin(set(out["nm"]))][["nm", "name", "season", "team", "pa", "k", "bbr", "hrr", "woba"]].rename(columns={"name": "player_name"}).assign(level="Pioneer")
    hist.sort_values(["nm", "season"], ascending=[True, False]).to_parquet("output/fl_hitter_pioneer_history.parquet", index=False)
    print(f"Pioneer seasons 2021-26 ({len(t):,} hitter-seasons); 2026 league means: wOBA proxy {lg['woba']:.4f}, K% {lg['k']:.4f}, BB% {lg['bbr']:.4f}, HR/PA {lg['hrr']:.4f}  ->  FL 2026: wOBA {L['woba']:.4f}, K% {L['k']:.4f}, BB% {L['bbr']:.4f}, HR/PA {L['hrr']:.4f}")
    q = out[out["src_pa"] >= 150]; print(f"{len(out)} hitters ({len(q)} with >= 150 PA, {int(out['has_fl_history'].sum())} already in the FL data, {int(out['also_in_milb'].sum())} also in the MiLB table); projected wOBA among >= 150 PA: mean {q['proj_woba'].mean():.4f}, SD {q['proj_woba'].std():.4f}, wRC+ range {q['proj_wrc_plus'].min():.0f}-{q['proj_wrc_plus'].max():.0f}; beyond validated range {q['beyond_validated_range'].mean():.0%}")
    print(q.head(8)[["name", "bat", "age", "src_pa", "proj_woba", "proj_wrc_plus", "proj_k", "proj_bbr", "proj_hrr", "proj_war_per400"]].round(3).to_string(index=False))
    if publish:
        for f in ("fl_hitter_pioneer_prospects.parquet", "fl_hitter_pioneer_history.parquet"):
            shutil.copy(f"output/{f}", f"../projections-dashboard/data/{f}")
        print("published to the dashboard data folder")


if __name__ == "__main__":
    main(publish="--publish" in sys.argv)
