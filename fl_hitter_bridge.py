"""
fl_hitter_bridge.py -- hitter other-league bridge (MiLB -> Frontier League), the hitter analogue of fl_unified_bridge.py (user 2026-10-09: "need to make a bridge for hitters"; thin-history includes rookies).
Validation and design choices: fl_hitter_bridge_test.py (nested leave-one-debut-year-out, weighted RMSE and MAE), fl_hitter_bridge_wobamix.py (wOBA = 0.5 direct bridge + 0.5 component-built, as in production); DECISION_LOG.md.
  per stat s in {wOBA, K%, BB%, HR/PA}:   FL deviation (vs that year's FL league mean) = a_s + b_s * u_s,
     u_s = sum_rows PA x decay^lag x (rate - same level-season MiLB mean) / (sum PA x decay^lag + n0_s)  summed over level groups (POOLED slope; level-group slopes did not beat pooled for the components)
  wOBA = 0.5 x direct + 0.5 x component-built (K -.177, BB +.516, HR +1.414 on the bridged component deviations);  rates = FL league 2026 mean + deviation.
Hyper-parameters (n0, decay) per stat: chosen by leave-one-year-out over the four debut classes 2023-26.  Coefficient uncertainty: player bootstrap (B = 500).  Residual spread at n PA from the variance law a + b/n of the out-of-fold residuals.
Outputs: output/fl_hitter_bridge_coefs.json, output/fl_hitter_milb_prospects.parquet (+ copy to the dashboard data folder with --publish).
"""
import json
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_bridge_test as bt
from fl_hitter_projection import read_league_rpa

warnings.filterwarnings("ignore")
BETA = {"k": -0.177, "bbr": 0.516, "hrr": 1.414}
GRID = {"woba": (100, 200, 400, 800), "k": (50, 100, 200, 400), "bbr": (50, 100, 200, 400), "hrr": (100, 200, 400, 800)}
import fl_season_config as _cfg
PROJ_YEAR = _cfg.TARGET_YEAR
B_BOOT = 500


def fit_stat(dd, m, s):
    """Choose (n0, decay) by leave-one-year-out, then fit the pooled mapping on all pairs.  Returns (n0, decay, intercept, slope, a_law, b_law)."""
    y = dd[f"y_{s}"].to_numpy(); n = dd["pa"].to_numpy(float); yr = dd["year"].to_numpy()
    base = np.array([np.average(y[yr != t], weights=n[yr != t]) for t in yr]); a, b = ew.fit_variance_law(y - base, n); w = ew.inv_var_weights(n, a, b)
    best = None
    for n0 in GRID[s]:
        for dc in (1.0, 0.5):
            X = bt.design(bt.features(dd, m, n0, dc, s)[0], "pooled"); err = 0.0
            for t in sorted(set(yr)):
                tr, te = yr != t, yr == t; p, _ = bt.fit_pred(X[tr], y[tr], w[tr], X[te]); err += np.sum(w[te] * (y[te] - p) ** 2)
            if best is None or err < best[0]:
                best = (err, n0, dc)
    _, n0, dc = best; X = bt.design(bt.features(dd, m, n0, dc, s)[0], "pooled"); p, co = bt.fit_pred(X, y, w, X)
    oof = np.zeros(len(y))
    for t in sorted(set(yr)):
        tr, te = yr != t, yr == t; oof[te], _ = bt.fit_pred(X[tr], y[tr], w[tr], X[te])
    a2, b2 = ew.fit_variance_law(y - oof, n)
    return {"n0": int(n0), "decay": float(dc), "intercept": float(co[0]), "slope": float(co[1]), "law_a": a2, "law_b": b2, "n_pairs": int(len(y))}, X, y, w


def main(publish=False):
    m = bt.milb_table(); d = bt.fl_debuts(); coefs = {}; boots = {}
    _, has = bt.features(d, m, 200, 0.5, "woba"); dd = d[has].reset_index(drop=True); dd = dd[dd[["y_woba", "y_k", "y_bbr", "y_hrr"]].notna().all(axis=1)].reset_index(drop=True)
    print(f"{len(dd)} FL debut hitters (2023-26, >= 20 PA) with MiLB history")
    rng = np.random.default_rng(11)
    for s, lab in bt.STATS.items():
        c, X, y, w = fit_stat(dd, m, s); coefs[s] = c; sl = []
        ut = X[:, 0]; c["u_min"], c["u_max"], c["u_p5"], c["u_p95"] = float(ut.min()), float(ut.max()), float(np.percentile(ut, 5)), float(np.percentile(ut, 95))
        for _ in range(B_BOOT):
            ii = rng.integers(0, len(y), len(y)); _, cb = bt.fit_pred(X[ii], y[ii], w[ii], X[:1]); sl.append(cb)
        sl = np.array(sl); c["slope_se"] = float(sl[:, 1].std()); c["intercept_se"] = float(sl[:, 0].std()); boots[s] = sl
        print(f"{lab:6s}: n0 {c['n0']}, decay {c['decay']}, intercept {c['intercept']:+.4f} (SE {c['intercept_se']:.4f}), slope {c['slope']:.3f} (SE {c['slope_se']:.3f}); residual law a {c['law_a']:.5f} b {c['law_b']:.3f}")
    # league levels for the projection year basis (FL 2026, park-neutral wOBA from the production projection)
    proj = pd.read_parquet("output/fl_2027_hitter_projections.parquet"); lg_woba = float(proj["lg_woba"].iloc[0]); lg_rpa = float(read_league_rpa()); rpw = float(proj["rpw"].iloc[0]); repl = float(proj["repl_per_pa"].iloc[0])
    h = pd.read_parquet("output/fl_hitter_panel.parquet"); h26 = h[h["year"] == 2026]
    lg = {"k": float(np.average(h26["so"] / h26["pa"], weights=h26["pa"])), "bbr": float(np.average(h26["bb"] / h26["pa"], weights=h26["pa"])), "hrr": float(np.average(h26["hr"] / h26["pa"], weights=h26["pa"])), "woba": lg_woba}
    consts = {"lg": lg, "lg_runs_per_pa": lg_rpa, "rpw": rpw, "repl_per_pa": repl, "woba_scale": 1.0175, "proj_year": PROJ_YEAR, "beta": BETA}
    # prospects: every MiLB hitter with PA in 2025-26, projected as of PROJ_YEAR
    cur = m[m["season"] >= 2025]; ids = cur.groupby("nm").agg(name=("player_name", "last"), batter=("batter", "last")).reset_index()
    last = m.sort_values(["season", "pa"]).groupby("nm").tail(1).set_index("nm"); ids["latest_season"] = ids["nm"].map(last["season"]); ids["latest_level"] = ids["nm"].map(last["level"]); ids["latest_team"] = ids["nm"].map(last["team"])
    ids["year"] = PROJ_YEAR; dev = {}; ids["beyond_validated_range"] = False
    for s in bt.STATS:
        c = coefs[s]; U, hs = bt.features(ids, m, c["n0"], c["decay"], s); u = U.sum(axis=1)
        if s == "woba":
            ids["beyond_validated_range"] = (u > c["u_p95"]) | (u < c["u_p5"])
        dev[s] = c["intercept"] + c["slope"] * np.clip(u, c["u_min"], c["u_max"])
    ids["src_pa"] = [m[(m["nm"] == nm) & (m["season"] < PROJ_YEAR)]["pa"].sum() for nm in ids["nm"]]
    comp = sum(BETA[s] * dev[s] for s in BETA)
    ids["proj_k"] = lg["k"] + dev["k"]; ids["proj_bbr"] = lg["bbr"] + dev["bbr"]; ids["proj_hrr"] = lg["hrr"] + dev["hrr"]
    ids["proj_woba_direct"] = lg_woba + dev["woba"]; ids["proj_woba_comp"] = lg_woba + comp; ids["proj_woba"] = 0.5 * ids["proj_woba_direct"] + 0.5 * ids["proj_woba_comp"]
    sc = consts["woba_scale"]; ids["proj_wrc_plus"] = 100 * (1 + (ids["proj_woba"] - lg_woba) / (sc * lg_rpa)); ids["proj_war_per400"] = 400 * ((ids["proj_woba"] - lg_woba) / sc + repl) / rpw
    cw = coefs["woba"]; ids["woba_sd_400pa"] = float(np.sqrt(cw["law_a"] + cw["law_b"] / 400))
    ids["woba_80_lo"] = ids["proj_woba"] - 1.2816 * ids["woba_sd_400pa"]; ids["woba_80_hi"] = ids["proj_woba"] + 1.2816 * ids["woba_sd_400pa"]
    fl_names = set(h["name_clean"].map(bt.norm)); ids["has_fl_history"] = ids["nm"].isin(fl_names)
    ids = ids[ids["src_pa"] > 0].sort_values("proj_woba", ascending=False).reset_index(drop=True)
    hist = m[m["nm"].isin(set(ids["nm"]))][["nm", "player_name", "season", "level", "team", "pa", "k", "bbr", "hrr", "woba"]].sort_values(["nm", "season"], ascending=[True, False]); hist.to_parquet("output/fl_hitter_milb_history.parquet", index=False)
    json.dump({"coefs": coefs, "constants": consts}, open("output/fl_hitter_bridge_coefs.json", "w"), indent=1); ids.to_parquet("output/fl_hitter_milb_prospects.parquet", index=False)
    print(f"\nprospect table: {len(ids)} MiLB hitters ({int(ids['has_fl_history'].sum())} already have FL history); projected wOBA mean {ids['proj_woba'].mean():.4f} (FL league {lg_woba:.4f}), SD {ids['proj_woba'].std():.4f}; wRC+ range {ids['proj_wrc_plus'].min():.0f}-{ids['proj_wrc_plus'].max():.0f}")
    print(ids.head(8)[["name", "latest_level", "latest_season", "src_pa", "proj_woba", "proj_wrc_plus", "proj_k", "proj_bbr", "proj_hrr", "proj_war_per400"]].round(3).to_string(index=False))
    if publish:
        for f in ("fl_hitter_bridge_coefs.json", "fl_hitter_milb_prospects.parquet", "fl_hitter_milb_history.parquet"):
            shutil.copy(f"output/{f}", f"../projections-dashboard/data/{f}")
        print("published to the dashboard data folder")


if __name__ == "__main__":
    main(publish="--publish" in sys.argv)
