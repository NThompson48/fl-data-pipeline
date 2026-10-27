"""
fl_2027_preregistered_checks.py -- checks FIXED BEFORE 2027 outcomes exist (pre-registered 2026-10-13, user: "yep").  Nothing here may be re-tuned on 2027 data.

  python fl_2027_preregistered_checks.py --freeze     (run ONCE, now: fits the two fixed coefficients on 2025 + 2026 and writes output/preregistered_2027.json)
  python fl_2027_preregistered_checks.py --run        (after the 2027 season: evaluates every check against the frozen file; not runnable until 2027 data are loaded)

Success rule for every model-change check (the user's reporting standard): held-out weighted RMSE AND weighted MAE both better, frozen standard inverse-variance weights,
player bootstrap P(better) >= .90.  Calibration checks: stated below.
"""
import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
OUT = "output/preregistered_2027.json"

CHECKS = {
    "C1 pitcher K% + fastball velocity": {"type": "model change (candidate, NOT in production)", "form": "proj_K += b x (fb_velo_shrunk - population mean); constant; single term",
        "trait": "mean effective velocity of four-seam / sinker in the feature window, shrunk by n / (n + 8.4) fastballs", "coefficient": "frozen, fit on ROS 2025 + ROS 2026 pooled (see 'frozen')",
        "samples": ["2026 -> 2027 preseason (production K%)", "2027 in-season, as of mid-season -> rest of 2027"], "success": "RMSE and MAE both better, P(better) >= .90, in both samples"},
    "C2 hitter BB% + barrel%": {"type": "model change (candidate, NOT in production)", "form": "proj_BB += b x (barrel_pct_shrunk - population mean); constant; single term",
        "trait": "barrels per tracked ball in play in the feature window, shrunk by the split-half n0", "coefficient": "frozen, fit on ROS 2025 + ROS 2026 pooled",
        "samples": ["2026 -> 2027 preseason", "2027 in-season, as of mid-season"], "success": "RMSE and MAE both better, P(better) >= .90, in both samples"},
    "C3 hitter K% bridge prior (IN production since 2026-10-13)": {"type": "confirmation", "sample": "the 2026 FL debut class with a MiLB record: second-season (2027) K%",
        "compare": "production (bridge prior) vs the same regression shrinking to the league mean", "success": "RMSE and MAE better, P(better) >= .90; if worse with P(worse) >= .90, remove"},
    "C4 thin-history hitter survivorship": {"type": "decision", "sample": "2026 -> 2027 hitters with effective history < 100 PA, PA-weighted target",
        "test": "mean error (actual - projected wOBA); pooled with the four earlier transitions", "success": "apply the +.0046 step if the 5-transition interval excludes 0"},
    "C5 pitcher P(WAR>0) calibration by projected IP": {"type": "calibration", "groups": ["< 22 projected IP", "22-45", "45+"],
        "success": "predicted mean inside each group's 90% Wilson interval; the < 22 group was ~7 points optimistic in 2026"},
    "C6 hitter WAR range coverage": {"type": "calibration", "test": "share of realised 2027 WAR (batting + position + baserunning proxy + replacement) inside the 50 / 80 / 90% ranges", "success": "each within 2 binomial SE of nominal", "amended": "2026-10-14 (before any 2027 data): WAR definition now includes the baserunning proxy (user decision 2026-10-14)"},
    "C7 baserunning persistence": {"type": "confirmation (baserunning proxy IN production since 2026-10-14)", "test": "held-out 2027 baserunning runs per PA (war_players.bsr_runs): (a) production proxy = 2026 position average incl. DH (fl_bsr_proxy, frozen 2026 effects) vs (b) league mean vs (c) own 2026 rate shrunk toward the proxy (n0 300 PA); plus the 2026 -> 2027 correlation of runs per PA (>= 150 PA both years)", "success": "keep the proxy if it beats (b) on RMSE and MAE with P(better) >= .90; switch to (c) if (c) beats the proxy on both with P(better) >= .90", "amended": "2026-10-14 (before any 2027 data): was 'decides whether baserunning returns to WAR'; baserunning entered WAR as a position proxy"},
    "C8 Trustmark surfaces": {"type": "validation", "test": "2027 balls at Trustmark (visitors + Mississippi at home): observed (wOBA - xwOBA) and HR vs the 2025-26 surface predictions, per hitter and overall",
        "success": "calibration slope interval contains 1; level inside the 2025-26 bootstrap range"},
    "C9 fielding / position with two tracked seasons": {"type": "data", "test": "2026 -> 2027 persistence of fielding runs by position (infield if the data exist) and of position runs",
        "success": "decides whether fielding returns to WAR (currently out of production)", "metric_spec": "fl_fielding_part_2026.py as of 2026-10-14 (PART-style infield runs: per-position x set boosted out models on ball + starting position, cross-fitted by game, multi-fielder credit, P/C term, centred within position, no speed proxy); success = 2026->2027 correlation of runs per opportunity and pooled full-time reliability >= .40 -> shrunk term enters WAR",
        "test_position": "POSITION RUNS (in production): own 2026 rate shrunk toward the 2026 group (DH = >= 50% of PA at DH) with n0 = 25 PA (within-2026 estimate, fl_pos_shrink_2026.py) and a 6.0 runs / 600 PA residual in the WAR simulation, neither validated season-ahead.  Test on 2027 position runs per PA: n0 grid (0, 50, 100, 200, group only) and the realised residual SD; success = adopt the held-out-best n0 / residual if it beats 25 / 6.0 on RMSE and MAE with P(better) >= .90", "amended": "2026-10-14 (before any 2027 data): added the position-runs shrinkage / residual test"},
    "C10 pitcher K% stuff term: network swstr_ct vs XGBoost wh_sva (IN production since 2026-10-25)": {"type": "confirmation (user 2026-10-26: 'err on the side of stability unless one is convincingly best')",
        "production": "additive K% model [naive_K_rate, fl_k_rate_bridge, wh_sva_level, spin_z_norm, extension] with wh_sva_level = per-pitch swinging-strike stuff level with pitch type and count (network, 3 seeds, cross-fitted by pitcher, fl_wh_sva_swstr_worker.py)",
        "counterfactual": "the same regression with the XGBoost swing-based wh_sva_level (production until 2026-10-25), refit on the same training transition; its 2027 preseason projections are FROZEN now in prereg/c10_frozen_projections.json (together with production's)",
        "samples": ["2026 -> 2027 preseason K%: the frozen projections scored on actual 2027 K% of pitchers with >= 40 PA in 2027, standard inverse-variance weights from the counterfactual's residuals, level-free, weighted RMSE and MAE, pitcher bootstrap (1,000)",
                    "2027 in-season, as of mid-season -> rest of 2027: pitcher ROS K% power prior with stuff estimator swstr_ct (B = 1.896 K% per unit) vs XGBoost wh_sva (B = .958), frozen coefficients"],
        "success": "KEEP swstr_ct if RMSE and MAE are both better than the counterfactual with P(better) >= .90 in the preseason sample; REVERT to the XGBoost stuff term if both are worse with P(worse) >= .90 in the preseason sample; otherwise inconclusive, keep and decide from the in-season sample and the stability report",
        "report_only": "2026 -> 2027 year-to-year correlation of the stuff level on pitchers with >= 300 pitches in both seasons, fl_y2y_heldout design (frozen model trained on one season, held-out pitchers): swstr_ct (network) vs XGBoost wh_sva; at 2026-10-26 these were .78-.79 vs .84 (difference not significant)",
        "added": "2026-10-26 (before any 2027 data)"},
}


def freeze():
    import pickle
    import fl_error_analysis_common as ea
    from fl_error_analysis_select import load
    frozen = {}
    for kind, tg, trait in (("p", "k", "fb_velo"), ("h", "bb", "barrel_pct")):
        st, S, n0t, _ = load(kind); e, w, x = [], [], []
        for s in ("ROS25", "ROS26"):
            F = S[s]["targets"][tg]; v, n = S[s]["traits"][trait]; xs = ea.shrink(v, n, st.loc[trait, "n0"]).reindex(F.index)
            mu = np.average(xs.dropna(), weights=F.loc[xs.notna(), "w"]); xs = xs.fillna(mu)
            e.append(F["e"].to_numpy()); w.append(F["w"].to_numpy()); x.append(xs.to_numpy() - mu)
        e, w, x = map(np.concatenate, (e, w, x)); b = float(np.sum(w * x * e) / np.sum(w * x * x))
        frozen[f"{kind}:{tg}:{trait}"] = {"b": b, "n0_trait": float(st.loc[trait, "n0"]), "n_rows": int(len(e))}
        print(f"{kind} {tg} {trait}: b = {b:+.5f} per unit of the shrunk trait (n = {len(e)}; trait n0 {st.loc[trait, 'n0']:.1f})")
    json.dump({"frozen_on": "2026-10-13", "checks": CHECKS, "frozen": frozen}, open(OUT, "w"), indent=1); print("saved", OUT)


def freeze_c10():
    """C10 (added 2026-10-26, before any 2027 data): freeze the production (network swstr_ct) and counterfactual (XGBoost wh_sva) 2027 preseason pitcher K% projections and coefficients, and add the check and an amendment
    to the pre-registration files (output/ and the tracked copy in prereg/)."""
    import fl_pitcher_features as fp
    import fl_position_player_flag as ppf
    import fl_production_projection as pp
    import fl_ros_production as rp
    import fl_unified_bridge as ub
    import pandas as pd
    raw = ppf.exclude(fp.load_fl_pitches_cached()); id_map = pp.build_canonical_id_map(raw); raw_roster = raw.copy(); raw_roster["pitcher_id"] = raw_roster["pitcher_id"].replace(id_map)
    ubridge = ub.fit(raw); run_values = fp.load_run_values(); lg_k, lg_bb = pp.league_level_2026(raw); res = {}
    for est in ("swstr_ct", "xgb"):
        fp.WH_SVA_ESTIMATOR = est; m_k, m_bb, n0_k, n0_bb, n0b, rho = pp.fit_frozen_model(raw)
        proj = pp.build_2027_projections(raw_roster, m_k, m_bb, n0b, rho, run_values, ubridge); proj, sk, sb = pp.recenter_to_league(proj, lg_k, lg_bb)
        res[est] = {"k_model": {k: float(v) for k, v in m_k.params.items()}, "level_shift_k": float(sk), "proj": proj[["pitcher_id", "pitcher", "PA", "proj_K_rate"]].assign(pitcher_id=lambda d: d["pitcher_id"].astype(str))}
    pub = pd.read_parquet("output/fl_2027_projections.parquet"); chk = res["swstr_ct"]["proj"].merge(pub[["pitcher_id", "proj_K_rate"]].assign(pitcher_id=lambda d: d["pitcher_id"].astype(str)), on="pitcher_id", suffixes=("", "_pub"))
    diff = float((chk["proj_K_rate"] - chk["proj_K_rate_pub"]).abs().max()); print(f"frozen swstr_ct projections vs the published file: max |diff| {diff:.2e} over {len(chk)} pitchers")
    assert diff < 1e-6, "the frozen production projections must equal the published ones"
    snap = {est: {"k_model": v["k_model"], "level_shift_k": v["level_shift_k"], "projections": v["proj"].to_dict("records")} for est, v in res.items()}
    snap["frozen_on"] = "2026-10-26"; snap["note"] = "2027 preseason pitcher K% projections (feature window 2025 + 2026): 'swstr_ct' = production, 'xgb' = counterfactual (XGBoost wh_sva stuff term, refit on the same training transition)"
    json.dump(snap, open("prereg/c10_frozen_projections.json", "w")); frozen = {"p:k:c10": {"swstr_ct_k_model": res["swstr_ct"]["k_model"], "xgb_k_model": res["xgb"]["k_model"], "ros_B_swstr_ct": rp.STUFF_B["pooled"][0], "ros_B_xgb": 0.958, "n_pitchers": len(chk), "projections_file": "prereg/c10_frozen_projections.json"}}
    for path in (OUT, "prereg/preregistered_2027.json"):
        spec = json.load(open(path)); spec["checks"].update({k: v for k, v in CHECKS.items() if k.startswith("C10")}); spec["frozen"].update(frozen)
        if not any("C10" in a.get("note", "") for a in spec.setdefault("amendments", [])):
            spec["amendments"].append({"date": "2026-10-26", "note": "C10 added before any 2027 data: network swstr_ct vs XGBoost wh_sva as the pitcher K% stuff term (production since 2026-10-25); production and counterfactual 2027 projections frozen"})
        json.dump(spec, open(path, "w"), indent=1)
    print("C10 frozen: coefficients", {e: {k: round(v, 4) for k, v in res[e]["k_model"].items()} for e in res})


def run():
    spec = json.load(open(OUT))
    raise SystemExit("2027 outcomes are not loaded yet. When they are: implement each check strictly as written in " + OUT + " (frozen " + spec["frozen_on"] + ").")


if __name__ == "__main__":
    freeze_c10() if "--freeze-c10" in sys.argv else freeze() if "--freeze" in sys.argv else run()
