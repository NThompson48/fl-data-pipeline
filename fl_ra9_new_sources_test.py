"""
fl_ra9_new_sources_test.py -- two candidate "other sources of variation" for the
RA9 baseline (K% + BB% + park + IP/G, now WITHOUT ppa_resid):

  A. xwOBA-on-contact ALLOWED (EB-shrunk by balls in play), plus exit velo,
     hard-hit%, air% and the wOBA-xwOBA gap as controls. The RA9-pivot work
     found xwOBA allowed a real modest lift (R2_w 0.160 -> 0.173) but never
     validated it on forward/reversed folds nor wired it into production.
  B. Pitcher AGE (output/pitcher_bio.parquet birth dates; coverage is partial --
     the file is keyed on MLBAM ids, so FL-only arms are missing).

For each: YoY stability, then LOOCV RA9 forward (2025 -> 2026) and reversed
(2026 -> 2025) adding the term to the base spec, with K%/BB% as the shipped
raw prior-season rates (the same frame the other robustness checks use).

Usage:  python fl_ra9_new_sources_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import text

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_ra9_input_robustness as rob
import fl_ra9_trustmark_projection as ra9
from fl_supabase import read_sql_df

BASE = "target ~ K_pred + BB_pred + ppf + ip_per_g_prior"


def contact(year):
    b = pd.read_csv("output/batted_ball_event_residuals.csv").dropna(subset=["pitcher_id", "exit_speed"])
    b = b[b["year"] == year].copy()
    b["pitcher_id"] = b["pitcher_id"].astype("int64").astype(str)
    b["hh"] = (b["exit_speed"] >= 95).astype(float); b["air"] = (b["angle"] >= 10).astype(float)
    b["gap"] = b["woba"] - b["xwoba"]
    g = b.groupby("pitcher_id").agg(bip=("hh", "size"), ev=("exit_speed", "mean"), hh=("hh", "mean"),
                                    xw=("xwoba", "mean"), air=("air", "mean"), gap=("gap", "mean")).reset_index()
    return g


def shrink_col(g, col, n0):
    lg = np.average(g[col], weights=g["bip"])
    return (g["bip"] * g[col] + n0 * lg) / (g["bip"] + n0) - lg


def loocv(m, extra):
    n = len(m); p = np.full(n, np.nan)
    spec = BASE + "".join(f" + {c}" for c in extra)
    for i in range(n):
        tr = m.drop(index=i); te = m.loc[[i]].copy()
        mod = smf.wls(spec, data=tr, weights=tr["w"]).fit()
        for c in ("ppf", "ip_per_g_prior"):
            te[c] = te[c].clip(tr[c].min(), tr[c].max())
        p[i] = float(mod.predict(te).iloc[0])
    return p


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    war = read_sql_df(text("select player, year, team, bf, ip, so, bb, ra9, ppf, g from war_pitchers where game_type='reg_season'"))
    bio = pd.read_parquet("output/pitcher_bio.parquet")
    bio["pitcher_id"] = bio["pitcher_id"].astype(str)
    bio["birth_date"] = pd.to_datetime(bio["birth_date"])
    ids = pd.DataFrame({"pitcher_id": raw["pitcher_id"].astype(str).unique(), "K_oof": 0.0, "BB_oof": 0.0})

    c25, c26 = contact(2025), contact(2026)
    print("STABILITY (YoY corr, BIP>=40 both years) of contact-quality-allowed metrics:")
    j = c25.merge(c26, on="pitcher_id", suffixes=("_25", "_26")); j = j[(j.bip_25 >= 40) & (j.bip_26 >= 40)]
    print({c: round(j[c + "_25"].corr(j[c + "_26"]), 3) for c in ["ev", "hh", "xw", "air", "gap"]}, f"n={len(j)}")

    for lab, y_in, y_tgt, win, cin in (("FORWARD 2025 -> 2026", 2025, 2026, (bt.START_2025, bt.END_2025), c25),
                                       ("REVERSED 2026 -> 2025", 2026, 2025, (bt.START_2026, bt.END_2026), c26)):
        m = rob.build_frame(war, raw, ids, y_in, y_tgt, win)
        m["pitcher_id"] = m["pitcher_id"].astype(str)
        m = m.merge(cin, on="pitcher_id", how="inner")
        m["K_pred"], m["BB_pred"] = m["K_raw"], m["BB_raw"]
        m = m.merge(bio[["pitcher_id", "birth_date"]], on="pitcher_id", how="left")
        ref_date = pd.Timestamp(f"{y_tgt}-07-01")
        m["age"] = (ref_date - m["birth_date"]).dt.days / 365.25
        m = m.reset_index(drop=True)
        print(f"\n=== {lab}: n={len(m)}, age coverage {m['age'].notna().mean():.0%} ===")
        for n0 in (50, 100, 200):
            m[f"xw_s{n0}"] = shrink_col(m, "xw", n0)
        res = []
        variants = [("base (K, BB, park, IP/G)", []), ("+ xwOBAcon allowed (n0=50)", ["xw_s50"]),
                    ("+ xwOBAcon allowed (n0=100)", ["xw_s100"]), ("+ xwOBAcon allowed (n0=200)", ["xw_s200"]),
                    ("+ exit velo", ["ev"]), ("+ hard-hit%", ["hh"]), ("+ air%", ["air"]),
                    ("+ wOBA-xwOBA gap", ["gap"])]
        for nm, ex in variants:
            res.append({"variant": nm, **{k: v for k, v in rob.summarize(m, loocv(m, ex), nm).items() if k in ("R2_w", "RMSE_w", "calib_slope")}})
        print(pd.DataFrame(res).to_string(index=False))
        full = smf.wls(BASE + " + xw_s100", data=m, weights=m["w"]).fit()
        print(f"   xwOBAcon(n0=100) coefficient {full.params['xw_s100']:+.1f} runs/9 per 1.00 xwOBA (t={full.tvalues['xw_s100']:+.2f}); "
              f"corr(xwOBAcon, K%)={m['xw_s100'].corr(m['K_pred']):+.2f}, corr(xwOBAcon, BB%)={m['xw_s100'].corr(m['BB_pred']):+.2f}")
        # age (subset with known birth date)
        a = m.dropna(subset=["age"]).reset_index(drop=True)
        if len(a) >= 25:
            a["age_c"] = a["age"] - a["age"].mean()
            ra = [{"variant": nm, **{k: v for k, v in rob.summarize(a, loocv(a, ex), nm).items() if k in ("R2_w", "RMSE_w")}}
                  for nm, ex in (("age subset: base", []), ("age subset: + age", ["age_c"]))]
            f2 = smf.wls(BASE + " + age_c", data=a, weights=a["w"]).fit()
            print(f"   AGE (n={len(a)} with birth date): age coefficient {f2.params['age_c']:+.3f} runs/9 per year (t={f2.tvalues['age_c']:+.2f}); "
                  f"mean age {a['age'].mean():.1f}, sd {a['age'].std():.1f}")
            print(pd.DataFrame(ra).to_string(index=False))
