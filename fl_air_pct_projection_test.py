"""
fl_air_pct_projection_test.py -- the Mississippi (TrustmarkPark) translation is
proportional to a pitcher's projected Air% (angle >= 10 deg share of batted
balls), so Air% projection quality is the main driver of how well the
Mississippi adjustment separates pitchers.

Shipped: a pitch-trait regression (IVB, frac_high, effective velo, 4-seam rate;
inputs EB-shrunk by pitch count), LOOCV R2_w ~ 0.23, fit on 2025 traits -> 2026
Air%. The log rejected "just use last year's Air%" (LOOCV R2_w = -0.45,
reliability-weighted) -- but that used the RAW rate. A batted-ball-count-
shrunk own-history rate has YoY stability 0.48 (BIP>=40), so combining it with
the traits may beat either alone.

Compared by LOOCV (target = 2026 Air%, >=30 BIP, weights = BIP/(BIP+60), the
same weighting as the shipped fit), 2025 -> 2026 AND reversed 2026 -> 2025:
    traits only (shipped) | own Air% shrunk (n0 grid) | traits + own Air% shrunk
Usage:  python fl_air_pct_projection_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_ra9_trustmark_projection as ra9


def air_counts():
    b = pd.read_csv("output/batted_ball_event_residuals.csv").dropna(subset=["pitcher_id"])
    b["pitcher_id"] = b["pitcher_id"].astype("int64").astype(str)
    b["air"] = (b["angle"] >= 10).astype(float)
    return b.groupby(["pitcher_id", "year"]).agg(bip=("air", "size"), air=("air", "mean")).reset_index()


def traits(raw, win):
    f = ra9._shrink_air_features(ra9._raw_pitch_features(fp.filter_period(raw, *win)))
    f["pitcher_id"] = f["pitcher_id"].astype(str)
    return f


def run(raw, ac, y_in, y_tgt, win_in, label):
    ft = traits(raw, win_in)
    a_in = ac[ac["year"] == y_in][["pitcher_id", "bip", "air"]].rename(columns={"bip": "bip_in", "air": "air_in"})
    a_tg = ac[(ac["year"] == y_tgt) & (ac["bip"] >= 30)][["pitcher_id", "bip", "air"]].rename(columns={"bip": "bip_t", "air": "air_pct"})
    d = ft.merge(a_tg, on="pitcher_id", how="inner").merge(a_in, on="pitcher_id", how="left")
    d["bip_in"] = d["bip_in"].fillna(0); d["air_in"] = d["air_in"].fillna(0)
    lg = np.average(ac[ac["year"] == y_in]["air"], weights=ac[ac["year"] == y_in]["bip"])
    d["w"] = d["bip_t"] / (d["bip_t"] + 60)
    print(f"\n=== {label}: n={len(d)}; own-history coverage {int((d['bip_in']>0).sum())}/{len(d)} (median BIP {d.loc[d.bip_in>0,'bip_in'].median():.0f}) ===")
    d = d.reset_index(drop=True)
    for n0 in (40, 80, 150, 300):
        d[f"own_{n0}"] = (d["bip_in"] * d["air_in"] + n0 * lg) / (d["bip_in"] + n0)
    specs = {"traits only (shipped)": ra9.AIR_SPEC,
             "league mean": "air_pct ~ 1"}
    for n0 in (40, 80, 150, 300):
        specs[f"own Air% shrunk n0={n0}"] = f"air_pct ~ own_{n0}"
        specs[f"traits + own Air% n0={n0}"] = ra9.AIR_SPEC + f" + own_{n0}"
    rows = []
    for nm, f in specs.items():
        p = np.zeros(len(d))
        for i in range(len(d)):
            m = smf.wls(f, data=d.drop(index=i), weights=d.drop(index=i)["w"]).fit()
            p[i] = float(m.predict(d.loc[[i]]).iloc[0])
        rows.append({"model": nm, "R2_w": round(mm.weighted_r2(d["air_pct"], p, d["w"]), 3),
                     "RMSE_w (pp)": round(mm.weighted_rmse(d["air_pct"], p, d["w"]) * 100, 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    # the best combined spec: coefficient view
    best = smf.wls(ra9.AIR_SPEC + " + own_150", data=d, weights=d["w"]).fit()
    print("   traits + own(n0=150) full-sample coefs:", {k: round(v, 3) for k, v in best.params.items()},
          "| t(own) =", round(best.tvalues["own_150"], 2))


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    ac = air_counts()
    run(raw, ac, 2025, 2026, (bt.START_2025, bt.END_2025), "FORWARD 2025 traits/history -> 2026 Air%")
    run(raw, ac, 2026, 2025, (bt.START_2026, bt.END_2026), "REVERSED 2026 -> 2025 Air%")
