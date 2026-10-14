"""
fl_hitter_bridge_college.py -- COLLEGE -> FL hitter bridge (user 2026-10-13: FanGraphs college batting exports, one per season 2022-2026, minimum PA 0, in input/fangraphs_college/).
Mirrors the MiLB hitter bridge (fl_hitter_bridge_test.py / fl_hitter_bridge.py):
  pairs     FL debut hitters 2023-26 (first FL season, >= 20 PA; fl_hitter_bridge_test.fl_debuts) matched by normalised name to their college seasons STRICTLY before the debut year
            (names carried by > 1 FanGraphs PlayerId are dropped as ambiguous).  Target = first-FL-season rate minus that season's FL league mean.
  stats     wOBA proxy (same linear weights as the MiLB proxy), K%, BB%, HR/PA; wOBA reported = 0.5 direct + 0.5 built from K% / BB% / HR/PA (production weights), as for MiLB.
  feature   u = sum_s PA_s decay^(year - s - 1) dev_s / (sum_s PA_s decay^(...) + n0), with dev measured three ways (a fork the user did not decide; chosen by data INSIDE the folds):
              raw   = rate - all-college season mean
              conf  = rate - his conference-season mean
              conf+ = conf, plus a second feature: his PA-weighted conference strength (conference-season mean - all-college season mean)
            (the college PITCHING bridge found raw beats conference-relative; not assumed here)
  mapping   FL dev = a + b u (+ c strength), weighted least squares, inverse-variance weights.
VALIDATION: leave-one-debut-class-out (outer); inside each training set, basis x n0 x decay chosen by leave-one-year-out (inner) -> honest out-of-fold predictions.  Compared
  with the debut-class mean (league average) and the latest college season's rate at face value (raw); weighted RMSE and MAE, level-free, frozen weights, player bootstrap;
  overall, college-only hitters (no MiLB record before the debut) and hitters who also have MiLB history.
BUILD (--build): fit on all pairs with the basis / n0 / decay chosen on all pairs; prospect table = college hitters with a 2025 or 2026 season (projected as of the target year),
  output/fl_hitter_college_prospects.parquet, output/fl_hitter_college_history.parquet, coefficients appended to output/fl_hitter_bridge_coefs.json under "college".
"""
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_bridge_test as hb

warnings.filterwarnings("ignore")
SRC = "input/fangraphs_college/fangraphs-college-leaderboard-*.csv"
STATS = {"woba": "wOBA", "k": "K%", "bbr": "BB%", "hrr": "HR/PA"}
BETA = {"k": -0.177, "bbr": 0.516, "hrr": 1.414}
GRID_N0 = {"woba": (100, 200, 400, 800), "k": (50, 100, 200, 400), "bbr": (50, 100, 200, 400), "hrr": (100, 200, 400, 800)}
BASES = ("raw", "conf", "conf+")


def college_table():
    fr = []
    for f in sorted(glob.glob(SRC)):
        m = re.search(r"(\d{4})\.csv$", f)
        if not m:
            continue
        d = pd.read_csv(f); d["season"] = int(m.group(1)); fr.append(d)
    c = pd.concat(fr, ignore_index=True); c = c[c["PA"] > 0].copy()
    c["nm"] = c["NameASCII"].fillna(c["Name"]).map(hb.norm)
    amb = c.groupby("nm")["PlayerId"].nunique(); c = c[~c["nm"].isin(amb[amb > 1].index)].copy()
    den = c["AB"] + c["BB"] + c["SF"].fillna(0) + c["HBP"].fillna(0)
    c["woba"] = (0.69 * c["BB"] + 0.72 * c["HBP"] + 0.88 * c["1B"] + 1.25 * c["2B"] + 1.58 * c["3B"] + 2.03 * c["HR"]) / den.replace(0, np.nan)
    c["k"] = c["SO"] / c["PA"]; c["bbr"] = c["BB"] / c["PA"]; c["hrr"] = c["HR"] / c["PA"]; c["pa"] = c["PA"].astype(float)
    conf = pd.read_parquet("output/fg_college_pit_with_conf.parquet").dropna(subset=["conference"]).groupby("team")["conference"].agg(lambda s: s.value_counts().index[0])
    c["conference"] = c["Team"].map(conf).fillna("UNKNOWN")
    for s in STATS:
        ok = c[s].notna()
        lg = c[ok].groupby("season").apply(lambda g: np.average(g[s], weights=g["pa"])); cf = c[ok].groupby(["season", "conference"]).apply(lambda g: np.average(g[s], weights=g["pa"]))
        c[f"lg_{s}"] = c["season"].map(lg); c[f"cf_{s}"] = [cf.get((y, k), np.nan) for y, k in zip(c["season"], c["conference"])]
        c[f"dev_raw_{s}"] = c[s] - c[f"lg_{s}"]; c[f"dev_conf_{s}"] = c[s] - c[f"cf_{s}"]; c[f"str_{s}"] = c[f"cf_{s}"] - c[f"lg_{s}"]
    return c


_LONG = {}


def _long(d, c, s, col):
    """(row index, season gap, pa, dev, strength) for every college season of each row's player strictly before his row year; cached per (rows, stat, column)."""
    key = (hash(tuple(zip(d["nm"], d["year"]))), s, col)
    if key not in _LONG:
        L = d[["nm", "year"]].reset_index(drop=True).reset_index().merge(c[["nm", "season", "pa", col, f"str_{s}"]], on="nm")
        L = L[(L["season"] < L["year"]) & L[col].notna()]
        _LONG[key] = (L["index"].to_numpy(), (L["year"] - L["season"] - 1).to_numpy(float), L["pa"].to_numpy(float), L[col].to_numpy(float), L[f"str_{s}"].to_numpy(float))
    return _LONG[key]


def features(d, c, s, basis, n0, decay):
    """X (n x 1 or 2) for rows of d (columns nm, year) from college seasons strictly before d.year; has-history flag.  Vectorised (same arithmetic as the per-row loop)."""
    col = f"dev_{'raw' if basis == 'raw' else 'conf'}_{s}"
    i, gap, pa, dev, st = _long(d, c, s, col); w = pa * decay ** gap; n = len(d)
    sw = np.bincount(i, weights=w, minlength=n); swd = np.bincount(i, weights=w * dev, minlength=n); sws = np.bincount(i, weights=w * st, minlength=n)
    has = sw > 0; X = np.zeros((n, 2)); X[has, 0] = swd[has] / (sw[has] + n0); X[has, 1] = sws[has] / sw[has]
    return (X if basis == "conf+" else X[:, :1]), has


def fit_pred(Xtr, ytr, wtr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr]); b = np.linalg.solve(A.T @ (A * wtr[:, None]) + 1e-9 * np.eye(A.shape[1]), A.T @ (wtr * ytr))
    return np.column_stack([np.ones(len(Xte)), Xte]) @ b, b


def choose(dd, c, s, y, w, yr):
    """basis x n0 x decay by leave-one-year-out inside the given rows."""
    best = None
    for basis in BASES:
        for n0 in GRID_N0[s]:
            for dc in (1.0, 0.5):
                X, _ = features(dd, c, s, basis, n0, dc); err = 0.0
                for t in sorted(set(yr)):
                    tr, te = yr != t, yr == t
                    if tr.sum() < 20 or te.sum() == 0:
                        continue
                    p, _ = fit_pred(X[tr], y[tr], w[tr], X[te]); err += np.sum(w[te] * (y[te] - p) ** 2)
                if best is None or err < best[0]:
                    best = (err, basis, n0, dc)
    return best[1:]


def pairs(c):
    d = hb.fl_debuts(); d = d[d["year"] >= 2023].copy()
    _, has = features(d, c, "woba", "raw", 100, 1.0); dd = d[has].reset_index(drop=True)
    return dd[dd[[f"y_{s}" for s in STATS]].notna().all(axis=1)].reset_index(drop=True)


def validate():
    c = college_table(); dd = pairs(c); yr = dd["year"].to_numpy(); n = dd["pa"].to_numpy(float)
    m = hb.milb_table(); _, has_milb = hb.features(dd, m, 200, 0.5, "woba"); dd["has_milb"] = has_milb
    print(f"college table: {len(c):,} hitter-seasons {sorted(c['season'].unique())}; conference known for {np.mean(c['conference'] != 'UNKNOWN'):.0%}")
    print(f"pairs: {len(dd)} FL debut hitters 2023-26 with college history (by class {pd.Series(yr).value_counts().sort_index().to_dict()}); also MiLB history {int(has_milb.sum())}")
    P, base, chosen = {}, {}, {}
    for s in STATS:
        y = dd[f"y_{s}"].to_numpy(); b0 = np.array([np.average(y[yr != t], weights=n[yr != t]) for t in yr]); a, bb = ew.fit_variance_law(y - b0, n); w = ew.inv_var_weights(n, a, bb)
        oof = np.zeros(len(y)); chosen[s] = []
        for t in sorted(set(yr)):
            tr, te = yr != t, yr == t; basis, n0, dc = choose(dd[tr].reset_index(drop=True), c, s, y[tr], w[tr], yr[tr]); chosen[s].append((t, basis, n0, dc))
            X, _ = features(dd, c, s, basis, n0, dc); oof[te], _ = fit_pred(X[tr], y[tr], w[tr], X[te])
        P[s] = oof; base[s] = b0; dd[f"w_{s}"] = w
    comp = sum(BETA[s] * P[s] for s in BETA); P["woba_blend"] = 0.5 * P["woba"] + 0.5 * comp; P["woba_comp"] = comp
    raw = {}
    by = {k: g for k, g in c.groupby("nm")}
    for s in STATS:
        v = []
        for nm, yy in zip(dd["nm"], dd["year"]):
            g = by[nm]; g = g[g["season"] < yy]; ls = g["season"].max(); g = g[g["season"] == ls]; v.append(np.average(g[s], weights=g["pa"]))
        raw[s] = np.array(v)
    rows = []
    for s, lab, pk in (("woba", "wOBA (0.5 direct + 0.5 built)", "woba_blend"), ("woba", "wOBA direct only", "woba"), ("woba", "wOBA built from K% / BB% / HR/PA only", "woba_comp"), ("k", "K%", "k"), ("bbr", "BB%", "bbr"), ("hrr", "HR/PA", "hrr")):
        y = dd[f"y_{s}"].to_numpy(); w = dd[f"w_{s}"].to_numpy(); c_ = lambda p, msk: p[msk] + np.average(y[msk] - p[msk], weights=w[msk])
        for grp, msk in (("all", np.ones(len(dd), bool)), ("college only (no MiLB)", ~dd["has_milb"].to_numpy()), ("also MiLB", dd["has_milb"].to_numpy())):
            if msk.sum() < 15:
                continue
            for bl, bv in (("league mean", base[s]), ("raw college, face value", raw[s])):
                r = er.compare(y[msk], c_(bv, msk), c_(P[pk], msk), w[msk], dd["nm"].to_numpy()[msk], n=1000)
                rows.append({"stat": lab, "group": grp, "n": int(msk.sum()), "vs": bl, "RMSE base": r["RMSE old"], "RMSE bridge": r["RMSE new"], "dRMSE": r["dRMSE"], "RMSE 90%": r["RMSE 90%"],
                             "P(bridge better)": r["P(RMSE better)"], "MAE base": r["MAE old"], "MAE bridge": r["MAE new"], "dMAE": r["dMAE"], "P(MAE better)": r["P(MAE better)"]})
    pd.set_option("display.width", 250)
    print("\nchoices inside the folds (held-out class: basis, n0, decay):")
    for s in STATS:
        print(f"  {STATS[s]:6s} " + "; ".join(f"{t}: {b} {n0} {dc}" for t, b, n0, dc in chosen[s]))
    R = pd.DataFrame(rows); print("\nCOLLEGE BRIDGE, leave-one-debut-class-out (level-free; dRMSE < 0 = bridge better)\n" + R.round(5).to_string(index=False)); R.to_csv("output/hitter_college_bridge_validation.csv", index=False)


def build():
    c = college_table(); dd = pairs(c); yr = dd["year"].to_numpy(); n = dd["pa"].to_numpy(float); co = {}
    J = json.load(open("output/fl_hitter_bridge_coefs.json")); ct = J["constants"]; lg = ct["lg"]; tgt = ct["proj_year"]
    cur = c[c["season"] >= tgt - 2]; ids = cur.sort_values(["season", "pa"]).groupby("nm").tail(1)[["nm", "Name", "Team", "season", "conference"]].rename(columns={"Name": "name", "Team": "latest_team", "season": "latest_season"})
    ids["year"] = tgt; dev = {}
    for s in STATS:
        y = dd[f"y_{s}"].to_numpy(); b0 = np.array([np.average(y[yr != t], weights=n[yr != t]) for t in yr]); a, bb = ew.fit_variance_law(y - b0, n); w = ew.inv_var_weights(n, a, bb)
        basis, n0, dc = choose(dd, c, s, y, w, yr); X, _ = features(dd, c, s, basis, n0, dc); p, b = fit_pred(X, y, w, X)
        e = y - p; a2, b2 = ew.fit_variance_law(e, n)
        co[s] = {"basis": basis, "n0": n0, "decay": dc, "coef": [float(v) for v in b], "u_min": float(X[:, 0].min()), "u_max": float(X[:, 0].max()), "u_p5": float(np.percentile(X[:, 0], 5)),
                 "u_p95": float(np.percentile(X[:, 0], 95)), "law_a": float(a2), "law_b": float(b2), "n_pairs": int(len(y))}
        Xp, hasp = features(ids, c, s, basis, n0, dc); Xp[:, 0] = np.clip(Xp[:, 0], co[s]["u_min"], co[s]["u_max"])
        dev[s] = np.column_stack([np.ones(len(Xp)), Xp]) @ np.array(co[s]["coef"])
        if s == "woba":
            ids["beyond_validated_range"] = (Xp[:, 0] > co[s]["u_p95"]) | (Xp[:, 0] < co[s]["u_p5"])
        print(f"{STATS[s]:6s}: basis {basis}, n0 {n0}, decay {dc}, coef {np.round(b, 4).tolist()}")
    comp = sum(BETA[s] * dev[s] for s in BETA); lw = lg["woba"]
    ids["proj_k"] = lg["k"] + dev["k"]; ids["proj_bbr"] = lg["bbr"] + dev["bbr"]; ids["proj_hrr"] = lg["hrr"] + dev["hrr"]
    ids["proj_woba_direct"] = lw + dev["woba"]; ids["proj_woba_comp"] = lw + comp; ids["proj_woba"] = 0.5 * ids["proj_woba_direct"] + 0.5 * ids["proj_woba_comp"]
    sc = ct["woba_scale"]; ids["proj_wrc_plus"] = 100 * (1 + (ids["proj_woba"] - lw) / (sc * ct["lg_runs_per_pa"])); ids["proj_war_per400"] = 400 * ((ids["proj_woba"] - lw) / sc + ct["repl_per_pa"]) / ct["rpw"]
    cw = co["woba"]; ids["woba_sd_400pa"] = float(np.sqrt(cw["law_a"] + cw["law_b"] / 400)); ids["woba_80_lo"] = ids["proj_woba"] - 1.2816 * ids["woba_sd_400pa"]; ids["woba_80_hi"] = ids["proj_woba"] + 1.2816 * ids["woba_sd_400pa"]
    ids["src_pa"] = ids["nm"].map(c.groupby("nm")["pa"].sum()); ids["latest_level"] = ids["conference"]
    pan = pd.read_parquet("output/fl_hitter_panel.parquet"); ids["has_fl_history"] = ids["nm"].isin(set(pan["name_clean"].map(hb.norm)))
    mp = pd.read_parquet("output/fl_hitter_milb_prospects.parquet"); ids["also_in_milb"] = ids["nm"].isin(set(mp["nm"]))
    ids = ids.sort_values("proj_woba", ascending=False).reset_index(drop=True); ids.to_parquet("output/fl_hitter_college_prospects.parquet", index=False)
    hist = c[c["nm"].isin(set(ids["nm"]))][["nm", "Name", "season", "conference", "Team", "pa", "k", "bbr", "hrr", "woba"]].rename(columns={"Name": "player_name", "conference": "level", "Team": "team"})
    hist.sort_values(["nm", "season"], ascending=[True, False]).to_parquet("output/fl_hitter_college_history.parquet", index=False)
    J["college"] = {"coefs": co, "source": "FanGraphs college batting exports 2022-2026 (user, minimum PA 0)"}; json.dump(J, open("output/fl_hitter_bridge_coefs.json", "w"), indent=1)
    print(f"prospects: {len(ids):,} college hitters (latest season >= {tgt - 2}); projected wOBA mean {ids['proj_woba'].mean():.4f} (FL league {lw:.4f}) SD {ids['proj_woba'].std():.4f}; "
          f"beyond validated range {ids['beyond_validated_range'].mean():.0%}; already in FL {int(ids['has_fl_history'].sum())}; also MiLB {int(ids['also_in_milb'].sum())}")
    if "--publish" in sys.argv:
        import shutil
        for f in ("fl_hitter_college_prospects.parquet", "fl_hitter_college_history.parquet", "fl_hitter_bridge_coefs.json"):
            shutil.copy(f"output/{f}", f"../projections-dashboard/data/{f}")
        print("published")


if __name__ == "__main__":
    build() if "--build" in sys.argv else validate()
