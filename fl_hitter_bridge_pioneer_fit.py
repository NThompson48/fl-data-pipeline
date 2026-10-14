"""
fl_hitter_bridge_pioneer_fit.py -- a REAL Pioneer League -> FL hitter bridge, replacing the provisional one (MiLB mapping applied to Pioneer 2026 rates).
Data: Pioneer batting 2021-2025 from StatsCrew (pull_statscrew_pioneer_batting.py; independent era) + 2026 from the user's Baseball Reference pastes (all 12 teams).
Machinery: fl_hitter_bridge_college (same features / inner selection / fit); Pioneer is one league, so the basis is the league-season mean ("raw").
  pairs     FL debut hitters 2023-26 with Pioneer seasons strictly before the debut year (normalised name match; StatsCrew has no player id, so same-name players can collide)
  feature   u = sum_s PA_s decay^(year - s - 1) (rate - Pioneer season mean) / (sum + n0); n0 and decay chosen by leave-one-year-out INSIDE each training set
VALIDATION (leave-one-debut-class-out; level-free; frozen weights; player bootstrap): vs the debut-class mean, vs the raw latest Pioneer season, and vs the PROVISIONAL
  method (MiLB coefficients and shrinkage applied to the Pioneer deviations).  wOBA = 0.5 direct + 0.5 built from K% / BB% / HR/PA, components from THIS bridge's
  projections (and the provisional arm from its own), per the comparison rule.
BUILD (--build): fit on all pairs; prospect table for Pioneer hitters with a 2025 or 2026 season -> output/fl_hitter_pioneer_prospects.parquet (replaces the provisional file),
  output/fl_hitter_pioneer_history.parquet; "pioneer" section in fl_hitter_bridge_coefs.json.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

import fl_error_analysis_common  # noqa: F401  (keeps import order consistent with other bridge scripts)
import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_bridge_college as hc
import fl_hitter_bridge_test as hb

warnings.filterwarnings("ignore")
hc.BASES = ("raw",)
STATS = hc.STATS; BETA = hc.BETA


def pioneer_table():
    sc = pd.read_parquet("output/pioneer_batting_2021_2025.parquet").rename(columns={"year": "season", "player": "name"})
    br = pd.read_parquet("output/pioneer_hitting_2026.parquet").rename(columns=str.lower).rename(columns={"2b": "x2b", "3b": "x3b"}); br["season"] = 2026
    cols = ["name", "season", "team", "pa", "ab", "h", "x2b", "x3b", "hr", "bb", "so", "hbp", "sf", "ibb"]
    t = pd.concat([sc[[c for c in cols if c in sc]], br[[c for c in cols if c in br]]], ignore_index=True)
    for c in ("pa", "ab", "h", "x2b", "x3b", "hr", "bb", "so", "hbp", "sf", "ibb"):
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0.0)
    t["nm"] = t["name"].map(hb.norm)
    t = t.groupby(["nm", "season"], as_index=False).agg(name=("name", "first"), team=("team", lambda x: "/".join(sorted(set(map(str, x))))), **{c: (c, "sum") for c in ("pa", "ab", "h", "x2b", "x3b", "hr", "bb", "so", "hbp", "sf", "ibb")})
    t = t[t["pa"] > 0].copy(); ubb = t["bb"] - t["ibb"]; den = t["ab"] + ubb + t["sf"] + t["hbp"]; x1 = t["h"] - t["x2b"] - t["x3b"] - t["hr"]
    t["woba"] = (0.69 * ubb + 0.72 * t["hbp"] + 0.88 * x1 + 1.25 * t["x2b"] + 1.58 * t["x3b"] + 2.03 * t["hr"]) / den.replace(0, np.nan)
    t["k"] = t["so"] / t["pa"]; t["bbr"] = t["bb"] / t["pa"]; t["hrr"] = t["hr"] / t["pa"]
    for s in STATS:
        ok = t[s].notna(); lg = t[ok].groupby("season").apply(lambda g: np.average(g[s], weights=g["pa"]))
        t[f"lg_{s}"] = t["season"].map(lg); t[f"dev_raw_{s}"] = t[s] - t[f"lg_{s}"]; t[f"str_{s}"] = 0.0
    return t


def provisional(dd, t, s):
    """the method being replaced: MiLB n0 / decay / intercept / slope / clip applied to the Pioneer deviations."""
    c = json.load(open("output/fl_hitter_bridge_coefs.json"))["coefs"][s]
    X, _ = hc.features(dd, t, s, "raw", c["n0"], c["decay"]); return c["intercept"] + c["slope"] * np.clip(X[:, 0], c["u_min"], c["u_max"])


def validate():
    t = pioneer_table(); dd = hc.pairs(t); yr = dd["year"].to_numpy(); n = dd["pa"].to_numpy(float)
    print(f"Pioneer table: {len(t):,} hitter-seasons, by season {t.groupby('season').size().to_dict()}; league wOBA proxy by season {t.groupby('season')['lg_woba'].first().round(3).to_dict()}")
    print(f"pairs: {len(dd)} FL debut hitters 2023-26 with Pioneer history (by class {pd.Series(yr).value_counts().sort_index().to_dict()})")
    P, PV, base, raw, W = {}, {}, {}, {}, {}
    by = {k: g for k, g in t.groupby("nm")}
    for s in STATS:
        y = dd[f"y_{s}"].to_numpy(); b0 = np.array([np.average(y[yr != q], weights=n[yr != q]) for q in yr]); a, bb = ew.fit_variance_law(y - b0, n); w = ew.inv_var_weights(n, a, bb)
        oof = np.zeros(len(y)); ch = []
        for q in sorted(set(yr)):
            tr, te = yr != q, yr == q; basis, n0, dc = hc.choose(dd[tr].reset_index(drop=True), t, s, y[tr], w[tr], yr[tr]); ch.append((q, n0, dc))
            X, _ = hc.features(dd, t, s, basis, n0, dc); oof[te], _ = hc.fit_pred(X[tr], y[tr], w[tr], X[te])
        P[s], PV[s], base[s], W[s] = oof, provisional(dd, t, s), b0, w
        raw[s] = np.array([np.average(g[g["season"] == g["season"].max()][s], weights=g[g["season"] == g["season"].max()]["pa"]) for g in (by[nm][by[nm]["season"] < yy] for nm, yy in zip(dd["nm"], dd["year"]))])
        print(f"  {STATS[s]:6s} chosen (held-out class, n0, decay): " + "; ".join(f"{q}: {a_} {b_}" for q, a_, b_ in ch))
    for A in (P, PV):
        A["woba_blend"] = 0.5 * A["woba"] + 0.5 * sum(BETA[s] * A[s] for s in BETA)
    rows = []
    for s, lab, pk in (("woba", "wOBA (0.5 direct + 0.5 built)", "woba_blend"), ("k", "K%", "k"), ("bbr", "BB%", "bbr"), ("hrr", "HR/PA", "hrr")):
        y = dd[f"y_{s}"].to_numpy(); w = W[s]; c_ = lambda p: p + np.average(y - p, weights=w)
        for bl, bv in (("league mean", base[s]), ("raw Pioneer, face value", raw[s]), ("PROVISIONAL (MiLB mapping)", PV[pk])):
            r = er.compare(y, c_(bv), c_(P[pk]), w, dd["nm"].to_numpy(), n=1000)
            rows.append({"stat": lab, "n": len(y), "vs": bl, "RMSE base": r["RMSE old"], "RMSE bridge": r["RMSE new"], "dRMSE": r["dRMSE"], "RMSE 90%": r["RMSE 90%"], "P(bridge better)": r["P(RMSE better)"],
                         "MAE base": r["MAE old"], "MAE bridge": r["MAE new"], "dMAE": r["dMAE"], "P(MAE better)": r["P(MAE better)"]})
    pd.set_option("display.width", 250); R = pd.DataFrame(rows); print("\nPIONEER BRIDGE, leave-one-debut-class-out (level-free; dRMSE < 0 = new Pioneer bridge better)\n" + R.round(5).to_string(index=False))
    R.to_csv("output/hitter_pioneer_bridge_validation.csv", index=False)


def build():
    t = pioneer_table(); dd = hc.pairs(t); yr = dd["year"].to_numpy(); n = dd["pa"].to_numpy(float)
    J = json.load(open("output/fl_hitter_bridge_coefs.json")); ct = J["constants"]; lg = ct["lg"]; tgt = ct["proj_year"]
    ids = t[t["season"] >= tgt - 2].sort_values(["season", "pa"]).groupby("nm").tail(1)[["nm", "name", "team", "season"]].rename(columns={"team": "latest_team", "season": "latest_season"})
    ids["year"] = tgt; dev, co = {}, {}
    for s in STATS:
        y = dd[f"y_{s}"].to_numpy(); b0 = np.array([np.average(y[yr != q], weights=n[yr != q]) for q in yr]); a, bb = ew.fit_variance_law(y - b0, n); w = ew.inv_var_weights(n, a, bb)
        basis, n0, dc = hc.choose(dd, t, s, y, w, yr); X, _ = hc.features(dd, t, s, basis, n0, dc); p, b = hc.fit_pred(X, y, w, X); a2, b2 = ew.fit_variance_law(y - p, n)
        co[s] = {"n0": n0, "decay": dc, "coef": [float(v) for v in b], "u_min": float(X[:, 0].min()), "u_max": float(X[:, 0].max()), "u_p5": float(np.percentile(X[:, 0], 5)), "u_p95": float(np.percentile(X[:, 0], 95)),
                 "law_a": float(a2), "law_b": float(b2), "n_pairs": int(len(y))}
        Xp, _ = hc.features(ids, t, s, basis, n0, dc); u = Xp[:, 0]
        if s == "woba":
            ids["beyond_validated_range"] = (u > co[s]["u_p95"]) | (u < co[s]["u_p5"])
        dev[s] = b[0] + b[1] * np.clip(u, co[s]["u_min"], co[s]["u_max"]); print(f"{STATS[s]:6s}: n0 {n0}, decay {dc}, intercept {b[0]:+.4f}, slope {b[1]:.3f}")
    lw = lg["woba"]; sc = ct["woba_scale"]; comp = sum(BETA[s] * dev[s] for s in BETA)
    ids["proj_k"] = lg["k"] + dev["k"]; ids["proj_bbr"] = lg["bbr"] + dev["bbr"]; ids["proj_hrr"] = lg["hrr"] + dev["hrr"]
    ids["proj_woba_direct"] = lw + dev["woba"]; ids["proj_woba_comp"] = lw + comp; ids["proj_woba"] = 0.5 * ids["proj_woba_direct"] + 0.5 * ids["proj_woba_comp"]
    ids["proj_wrc_plus"] = 100 * (1 + (ids["proj_woba"] - lw) / (sc * ct["lg_runs_per_pa"])); ids["proj_war_per400"] = 400 * ((ids["proj_woba"] - lw) / sc + ct["repl_per_pa"]) / ct["rpw"]
    cw = co["woba"]; ids["woba_sd_400pa"] = float(np.sqrt(cw["law_a"] + cw["law_b"] / 400)); ids["woba_80_lo"] = ids["proj_woba"] - 1.2816 * ids["woba_sd_400pa"]; ids["woba_80_hi"] = ids["proj_woba"] + 1.2816 * ids["woba_sd_400pa"]
    ids["src_pa"] = ids["nm"].map(t.groupby("nm")["pa"].sum()); ids["latest_level"] = "Pioneer"
    pan = pd.read_parquet("output/fl_hitter_panel.parquet"); ids["has_fl_history"] = ids["nm"].isin(set(pan["name_clean"].map(hb.norm)))
    mp = pd.read_parquet("output/fl_hitter_milb_prospects.parquet"); ids["also_in_milb"] = ids["nm"].isin(set(mp["nm"]))
    ids = ids.sort_values("proj_woba", ascending=False).reset_index(drop=True); ids.to_parquet("output/fl_hitter_pioneer_prospects.parquet", index=False)
    hist = t[t["nm"].isin(set(ids["nm"]))][["nm", "name", "season", "team", "pa", "k", "bbr", "hrr", "woba"]].rename(columns={"name": "player_name"}); hist["level"] = "Pioneer"
    hist.sort_values(["nm", "season"], ascending=[True, False]).to_parquet("output/fl_hitter_pioneer_history.parquet", index=False)
    J["pioneer"] = {"coefs": co, "source": "StatsCrew Pioneer batting 2021-25 + Baseball Reference 2026 (user paste, 12 teams)"}; json.dump(J, open("output/fl_hitter_bridge_coefs.json", "w"), indent=1)
    print(f"prospects: {len(ids):,} Pioneer hitters (latest season >= {tgt - 2}); projected wOBA mean {ids['proj_woba'].mean():.4f} (FL league {lw:.4f}) SD {ids['proj_woba'].std():.4f}; beyond range {ids['beyond_validated_range'].mean():.0%}; in FL {int(ids['has_fl_history'].sum())}")
    if "--publish" in sys.argv:
        import shutil
        for f in ("fl_hitter_pioneer_prospects.parquet", "fl_hitter_pioneer_history.parquet", "fl_hitter_bridge_coefs.json"):
            shutil.copy(f"output/{f}", f"../projections-dashboard/data/{f}")
        print("published")


if __name__ == "__main__":
    build() if "--build" in sys.argv else validate()
