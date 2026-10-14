"""
fl_park_component_factors.py -- COMPONENT park factors for TrustmarkPark, general home-field advantage controlled, related to each hitter's own component rates; plus Mississippi
hitters' extra home effect beyond park + home-field advantage.  (user 2026-10-12, item 1: "control for home field advantage in general and then see if there is any extra effects
beyond that for Mississippi hitters ... use the individual component park factors and relate them to each hitter. That may or may not involve introducing more noise ... 1B/2B/3B")

PART A (2025-26 pooled, per PA; game-cluster bootstrap B = 40):
  A1  visiting-hitter design (fl_park_visitors_fe: year + hitter + pitcher + hitter x home + park) -> Trustmark effect per outcome (1B, 2B, 3B, HR, BB, HBP, K).
  A2  generic-HFA design: year + hitter + pitcher + HOME (one league-wide home-advantage dummy) + MS_HOME (Mississippi hitters at home) + park.  Identification: Trustmark from
      visiting hitters' PA (assumes Mississippi PITCHERS have no extra home effect beyond the generic one; their K% home - road is -0.3 points, so plausible for K); MS_HOME =
      Mississippi hitters' extra home effect beyond park + generic home advantage.
  Each effect -> wOBA with the FL linear weights; shrunk by max(0, 1 - 1/t^2) (production rule); multiplicative factor f = 1 + shrunk effect / league rate.
PART B (held-out season: fit on one season, score the other; per hitter-season with >= 20 PA at Trustmark, visitors + Mississippi hitters at home):
  target = his wOBA per PA at Trustmark minus his own away wOBA per PA (shrunk 100 PA toward the away league mean).
  models (all fitted on the TRAINING season only):  NONE = one level;  NONE+MS = separate levels for Mississippi-at-home and visitors (the MS extra as a level);
  SURFACE = NONE+MS + 0.671 x (production surface shift - its mean), surface fit on the training season's balls (pooled design);
  COMPONENT = NONE+MS + sum_o weight_o x (his away rate_o - league rate_o) x (f_o - 1), factors from the training season (A1 design), shrunk with the single-season t;
  COMPONENT raw = same, unshrunk factors.  Inverse-variance weights (fit_variance_law on the NONE residuals), weighted RMSE / MAE, hitter bootstrap vs NONE+MS.
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_park_ms_check as msc
import fl_hitter_park_surface as ps
from fl_hitter_panel import woba_weights
from fl_park_visitors_fe import load, park_effects
from fl_park_woba_fe import fe_fit

warnings.filterwarnings("ignore")
MS = "Mississippi Mud Monsters"
OUTS = {"1B": "is_1b", "2B": "is_2b", "3B": "is_3b", "HR": "is_hr", "BB": "is_bb", "HBP": "is_hbp", "K": "is_k"}
N0_RATE = 150.0
MIN_TM_PA = 20


def prep(pa):
    k = pa["kind"].astype(str)
    pa["is_2b"] = (k == "double").astype(float); pa["is_3b"] = (k == "triple").astype(float); pa["is_hbp"] = (k == "hbp").astype(float)
    pa["ms_home"] = ((pa["batter_team"] == MS) & (pa["b_home"] == 1)).astype(int)
    return pa


def lw_of(w):
    return {"1B": w["1B"], "2B": w["2B"], "3B": w["3B"], "HR": w["HR"], "BB": w["BB"], "HBP": w["HBP"], "K": 0.0}


def hfa_design(pa, col, iters=30):
    g = [pd.factorize(pa["year"])[0], pd.factorize(pa["name_clean"])[0], pd.factorize(pa["pid"])[0], pa["b_home"].to_numpy(int), pa["ms_home"].to_numpy(int), pd.factorize(pa["park"])[0]]
    mu, eff = fe_fit(pa[col].to_numpy(float), g, iters=iters); parks = list(pd.factorize(pa["park"])[1]); pe = pd.Series(eff[-1], index=parks)
    w = np.bincount(g[-1], minlength=len(parks)); pe = pe - np.average(pe, weights=np.maximum(w, 1e-9))
    ms_extra = eff[4][1] - eff[4][0] if len(eff[4]) > 1 else np.nan
    return float(pe["TrustmarkPark"]), float(ms_extra)


def part_a(pa, lw, B=40):
    est = {o: {"A1": park_effects(pa, "hitter", c)["TrustmarkPark"], "A2": hfa_design(pa, c)} for o, c in OUTS.items()}
    rng = np.random.default_rng(21); games = pa["game_id"].to_numpy(); ug = pd.unique(games); by_game = pd.Series(np.arange(len(pa))).groupby(games).apply(np.array)
    boots = {o: [] for o in OUTS}
    for b in range(B):
        sub = pa.iloc[np.concatenate([by_game[g] for g in rng.choice(ug, len(ug))])].reset_index(drop=True)
        for o, c in OUTS.items():
            a1 = park_effects(sub, "hitter", c, iters=20).get("TrustmarkPark", np.nan); a2, ms = hfa_design(sub, c, iters=20); boots[o].append((a1, a2, ms))
        print(f"  bootstrap {b + 1}/{B}", flush=True) if (b + 1) % 10 == 0 else None
    rows = []
    for o, c in OUTS.items():
        bb = np.array(boots[o]); se1, se2, se3 = np.nanstd(bb, axis=0); lg = pa[c].mean(); a1 = est[o]["A1"]; a2, ms = est[o]["A2"]
        t1 = a1 / se1; sh = max(0.0, 1 - 1 / t1 ** 2)
        rows.append({"outcome": o, "league rate": lg, "A1 visitors": a1, "SE": se1, "t": t1, "shrunk": a1 * sh, "factor": 1 + a1 * sh / lg,
                     "A2 HFA design": a2, "SE A2": se2, "MS extra (beyond park + HFA)": ms, "SE MS": se3, "t MS": ms / se3, "wOBA weight": lw[o]})
    T = pd.DataFrame(rows).set_index("outcome")
    print("\nPART A.  TrustmarkPark effect per PA by outcome (visiting hitters; generic home-field advantage controlled) and Mississippi hitters' extra home effect\n" + T.round(4).to_string())
    for lab, col in (("A1 park, raw", "A1 visitors"), ("A1 park, shrunk", "shrunk"), ("A2 park (HFA design)", "A2 HFA design"), ("MS extra", "MS extra (beyond park + HFA)")):
        print(f"  {lab:22s} in wOBA per PA: {np.sum(T[col] * T['wOBA weight']):+.4f}")
    ms_se = np.nanstd([np.sum([boots[o][b][2] * lw[o] for o in OUTS]) for b in range(B)]); a1_se = np.nanstd([np.sum([boots[o][b][0] * lw[o] for o in OUTS]) for b in range(B)])
    print(f"  bootstrap SE of the wOBA sums: park (A1) {a1_se:.4f}, MS extra {ms_se:.4f}")
    T.to_csv("output/trustmark_component_factors.csv")
    return T, {o: float(np.nanstd(np.array(boots[o])[:, 0])) for o in OUTS}


def part_b(pa, lw, se_full):
    away = pa[pa["tm"] == 0]; lgw = away["woba"].mean(); g = away.groupby("name_clean")["woba"].agg(["sum", "size"]); base = (g["sum"] + 100 * lgw) / (g["size"] + 100)
    balls = msc.prep()
    allu = []
    for tr_y, te_y in ((2025, 2026), (2026, 2025)):
        tr = pa[pa["year"] == tr_y]

        def units(season):
            v = pa[(pa["tm"] == 1) & ((pa["b_home"] == 0) | (pa["ms_home"] == 1)) & (pa["year"] == season)]
            u = v.groupby("name_clean").agg(n=("woba", "size"), w=("woba", "mean"), ms=("ms_home", "max")); u = u[u["n"] >= MIN_TM_PA].copy(); u["y"] = u["w"] - u.index.map(base).to_numpy(); return u
        Utr, Ute = units(tr_y), units(te_y)
        lev = np.average(Utr["y"], weights=Utr["n"]); lev_ms = {m: np.average(Utr.loc[Utr["ms"] == m, "y"], weights=Utr.loc[Utr["ms"] == m, "n"]) for m in (0, 1)}
        out = Ute.copy(); out["NONE"] = lev; out["NONE+MS"] = out["ms"].map(lev_ms)
        # surface (production design) fit on the training season's balls
        eff = msc.fit_effect_pooled(balls[balls["year"] == tr_y], "evla_side"); sh, mu = ps.hitter_shift(balls[balls["year"] == tr_y], eff, "evla_side")
        out["SURFACE"] = out["NONE+MS"] + 0.671 * (out.index.map(sh).to_numpy() - mu); out["SURFACE"] = out["SURFACE"].fillna(out["NONE+MS"])
        # component factors from the training season, visiting-hitter design; single-season SE ~ full SE x sqrt(2)
        fac_raw, fac_sh = {}, {}
        for o, c in OUTS.items():
            e = park_effects(tr, "hitter", c)["TrustmarkPark"]; lg = tr[c].mean(); t = e / (se_full[o] * np.sqrt(2)); fac_raw[o] = e / lg; fac_sh[o] = e * max(0.0, 1 - 1 / t ** 2) / lg
        aw = tr[tr["tm"] == 0]; rates = aw.groupby("name_clean")[[c for c in OUTS.values()]].agg(["sum", "size"])
        for lab, fac in (("COMPONENT", fac_sh), ("COMPONENT raw", fac_raw)):
            shift = 0.0
            for o, c in OUTS.items():
                lg = aw[c].mean(); s = rates[(c, "sum")]; n = rates[(c, "size")]; r = (s + N0_RATE * lg) / (n + N0_RATE)
                shift = shift + lw[o] * (r - lg) * fac[o]
            out[lab] = out["NONE+MS"] + out.index.map(shift).fillna(0.0).to_numpy()
        out["fold"] = te_y; allu.append(out)
        print(f"\nfold train {tr_y} -> test {te_y}: factors (shrunk) " + ", ".join(f"{o} {fac_sh[o]:+.2f}" for o in OUTS) + " | raw " + ", ".join(f"{o} {fac_raw[o]:+.2f}" for o in OUTS))
    U = pd.concat(allu)
    for lab, sel in (("ALL units", np.ones(len(U), bool)), ("MISSISSIPPI hitters at home", (U["ms"] == 1).to_numpy()), ("VISITING hitters", (U["ms"] == 0).to_numpy())):
        Us = U[sel]; y = Us["y"].to_numpy(float); n = Us["n"].to_numpy(float); a, b_ = ew.fit_variance_law(y - Us["NONE"].to_numpy(float), n); w = ew.inv_var_weights(n, a, b_)
        ids = (Us.index.astype(str) + "_" + Us["fold"].astype(str)).to_numpy(); ref = Us["NONE+MS"].to_numpy(float); rows = []
        for m in ("NONE", "NONE+MS", "SURFACE", "COMPONENT", "COMPONENT raw"):
            v = Us[m].to_numpy(float); rm, ma = er.wmetrics(y, v, w); row = {"model": m, "RMSE": rm, "MAE": ma}
            if m != "NONE+MS":
                rep = er.compare(y, ref, v, w, Us.index.to_numpy(), n=1000); row.update({"dRMSE vs NONE+MS": rep["dRMSE"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
            rows.append(row)
        print(f"\nPART B.  {lab}: {len(Us)} hitter-seasons (>= {MIN_TM_PA} Trustmark PA), target = wOBA per PA at Trustmark minus own away baseline\n" + er.table(rows, 5))
    U.to_parquet("output/trustmark_component_validation_units.parquet")


def main():
    pa = prep(load()); lw = lw_of(woba_weights())
    print("FL wOBA weights:", {k: round(v, 3) for k, v in lw.items()}, "| PA:", len(pa))
    T, se = part_a(pa, lw)
    part_b(pa, lw, se)


if __name__ == "__main__":
    main()
