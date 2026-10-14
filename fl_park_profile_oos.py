"""
fl_park_profile_oos.py -- does a Trustmark translation built from hitter PROFILE (Air%, power, handedness) predict out of sample, and how does it compare with the current translation (a hitter's HR / air-ball
rates through the HR factor .446 and the air-ball hit delta .0073) and with no park adjustment?
Design (forward in time): fit on 2025 Trustmark data, score on 2026.
  Observed 2026 Trustmark effect per hitter = mean over his 2026 Trustmark PA of [wOBA - baseline], baseline = year + hitter + pitcher fixed effects estimated on the 2026 NON-Trustmark PA only (so the park cannot
  explain itself), hitters with >= 20 Trustmark PA and non-Trustmark PA in 2026.
  Predictions for the same hitters (profile from his 2025 non-Trustmark PA / balls only):
   M0 no adjustment (0) | M1 CURRENT translation (full park, share = 1) | M2 global Trustmark effect from 2025 (one number) | M3 global + profile (Air%, power, left-handed) from the 2025 interaction fit |
   M4 global + power only | M5 global + handedness only.
  Scores: PA-weighted RMSE and MAE of (observed - predicted), the calibration slope of observed on predicted, hitter-cluster bootstrap for differences (reporting standard).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_ev as hev
import fl_hitter_park as park
import fl_hitter_ros as hr
import fl_hitter_whiff as hwf
import fl_innings_projection as ip
from fl_park_profile_fe import cluster_ols, partial_out
from fl_park_woba_fe import fe_fit

warnings.filterwarnings("ignore")


def build():
    T = hr.Tables(); pa = T.pa.copy(); pit = pd.read_parquet("output/pitcher_pa_ros_2025_2026.parquet"); assert len(pit) == len(pa)
    pa["pitcher"] = pit["pitcher_id"].astype(object).where(pit["pitcher_id"].notna(), "unknown").astype(str).values; pa["park"] = pa["game_id"].str.split("-").str[1]; pa["name_clean"] = pa["name_clean"].fillna("unknown")
    pa = pa[pa["woba"].notna() & pa["park"].notna()].reset_index(drop=True); pa["tm"] = (pa["park"] == "TrustmarkPark").astype(int)
    pi = hwf.load(); pi = pi[pi["year"].isin([2025, 2026]) & pi["pa_id"].notna() & pi["batter_side"].isin(["left", "right"])].copy(); pi["name_clean"] = pi["batter"].apply(ip.lastfirst_to_clean)
    side = pi.sort_values("pitch_no").groupby(["game_id", "pa_id", "name_clean"])["batter_side"].first().rename("side").reset_index(); pa = pa.merge(side, on=["game_id", "pa_id", "name_clean"], how="left"); pa["left"] = np.where(pa["side"] == "left", 1.0, np.where(pa["side"] == "right", 0.0, np.nan))
    kind = pa["kind"].astype(str); pa["is_hr"] = (kind == "home_run").astype(float)
    return pa


def profiles_2025(pa):
    non = pa[(pa["year"] == 2025) & (pa["tm"] == 0)]; g = non.groupby("name_clean").agg(n=("woba", "size"), hr=("is_hr", "sum")); lg = g["hr"].sum() / g["n"].sum(); g["hrr"] = g["hr"] / g["n"]; g["power"] = (g["hrr"] - lg) * g["n"] / (g["n"] + 137.0)
    bb = hev.load_bbe(); bb = bb[bb["year"] == 2025]; bb["air"] = (bb["angle"] >= 10).astype(int); ga = bb.groupby("name_clean").agg(n=("air", "size"), air=("air", "sum")); lga = ga["air"].sum() / ga["n"].sum(); ga["air_dev"] = (ga["air"] / ga["n"] - lga) * ga["n"] / (ga["n"] + 89.0); ga["air_pct"] = ga["air"] / ga["n"]
    return g.join(ga[["air_dev", "air_pct", "n"]].rename(columns={"n": "n_bbe"}), how="inner")


def main():
    pa = build(); prof = profiles_2025(pa); pa["power_z"] = pa["name_clean"].map(prof["power"]); pa["air_z"] = pa["name_clean"].map(prof["air_dev"])
    ps, asd = prof["power"].std(), prof["air_dev"].std(); pa["power_z"] /= ps; pa["air_z"] /= asd; pm, am = (prof["power"] / ps).mean(), (prof["air_dev"] / asd).mean(); pa["power_z"] -= pm; pa["air_z"] -= am
    # ---- fit on 2025
    tr = pa[(pa["year"] == 2025) & pa["power_z"].notna() & pa["air_z"].notna() & pa["left"].notna()].reset_index(drop=True); g_tr = [pd.factorize(tr[c])[0] for c in ("name_clean", "pitcher", "park")]; tmv = tr["tm"].to_numpy(float); lm = tr["left"].mean()
    y = tr["woba"].to_numpy(float); mu, eff = fe_fit(y, g_tr, iters=30); pe = pd.Series(eff[2] - np.average(eff[2], weights=np.bincount(g_tr[2])), index=pd.factorize(tr["park"])[1]); delta0 = float(pe["TrustmarkPark"])
    X = np.column_stack([tmv * tr["air_z"], tmv * tr["power_z"], tmv * (tr["left"] - lm)]); Xr = partial_out(X, g_tr); yr = partial_out(y[:, None], g_tr)[:, 0]; b, se = cluster_ols(Xr, yr, pd.factorize(tr["game_id"])[0])
    print(f"2025 fit ({len(tr):,} PA, {int(tmv.sum()):,} at Trustmark): global Trustmark wOBA effect {delta0:+.4f}; per +1 SD Air% {b[0]:+.4f} ({se[0]:.4f}), per +1 SD power {b[1]:+.4f} ({se[1]:.4f}), left-handed {b[2]:+.4f} ({se[2]:.4f})")
    # ---- observed 2026 Trustmark effect per hitter
    te = pa[pa["year"] == 2026].reset_index(drop=True); non = te[te["tm"] == 0]; tmx = te[te["tm"] == 1]; g_non = [pd.factorize(non[c])[0] for c in ("name_clean", "pitcher", "park")]
    mu2, eff2 = fe_fit(non["woba"].to_numpy(float), g_non, iters=30); hit = pd.Series(eff2[0], index=pd.factorize(non["name_clean"])[1]); pit_e = pd.Series(eff2[1], index=pd.factorize(non["pitcher"])[1]); pk_mean = np.average(eff2[2], weights=np.bincount(g_non[2]))
    tmx = tmx[tmx["name_clean"].isin(hit.index)].copy(); tmx["base"] = mu2 + tmx["name_clean"].map(hit) + tmx["pitcher"].map(pit_e).fillna(0.0) + pk_mean; tmx["excess"] = tmx["woba"] - tmx["base"]
    h = tmx.groupby("name_clean").agg(n_tm=("woba", "size"), excess=("excess", "mean"), left=("left", "mean")); h = h[h["n_tm"] >= 20].join(prof[["power", "air_dev", "hrr", "air_pct"]], how="inner").dropna()
    h["power_z"] = h["power"] / ps - pm; h["air_z"] = h["air_dev"] / asd - am
    print(f"2026 test: {len(h)} hitters with >= 20 Trustmark PA (total {int(h['n_tm'].sum()):,} PA); observed mean Trustmark effect {np.average(h['excess'], weights=h['n_tm']):+.4f}")
    air_pa = (h["air_pct"] * (1 - 0.196 - 0.112)).clip(lower=0); cur = park.translate_hr_model(h["hrr"].clip(lower=0), air_pa, share=1.0)           # the current translation, full park (share = 1)
    left_c = h["left"].round() - lm
    preds = {"M0 no adjustment": np.zeros(len(h)), "M1 current translation (HR factor .446 + air-ball hit delta)": cur.to_numpy(), "M2 global effect only": np.full(len(h), delta0),
             "M3 global + Air% + power + handedness": delta0 + b[0] * h["air_z"] + b[1] * h["power_z"] + b[2] * left_c, "M4 global + power only": delta0 + b[1] * h["power_z"], "M5 global + handedness only": delta0 + b[2] * left_c}
    w = h["n_tm"].to_numpy(float); obs = h["excess"].to_numpy(); ids = h.index.to_numpy(); rows = []
    print(f"  mean prediction (PA-weighted): " + ", ".join(f"{k.split()[0]} {np.average(np.asarray(v), weights=w):+.4f}" for k, v in preds.items()))
    for k, v in preds.items():
        v = np.asarray(v, float); r_, m_ = er.wmetrics(obs, v, w); sl = np.polyfit(v, obs, 1, w=np.sqrt(w))[0] if np.std(v) > 1e-9 else np.nan; row = {"model": k, "RMSE": r_, "MAE": m_, "slope of observed on predicted": sl}
        if not k.startswith("M1"):
            rep = er.compare(obs, np.asarray(preds["M1 current translation (HR factor .446 + air-ball hit delta)"], float), v, w, ids, n=800); row.update({"dRMSE vs current": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE vs current": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
        rows.append(row)
    print("\nOUT OF SAMPLE: observed 2026 Trustmark wOBA effect per hitter vs predictions fit on 2025 (full-park effect, points of wOBA)\n" + er.table(rows, 4))
    rep = er.compare(obs, np.asarray(preds["M2 global effect only"], float), np.asarray(preds["M3 global + Air% + power + handedness"], float), w, ids, n=800); print(f"\nM3 vs M2 (does the PROFILE add to the global effect?): dRMSE {rep['dRMSE']:+.4f} {rep['RMSE 90%']}, P(better) {rep['P(RMSE better)']:.2f}; dMAE {rep['dMAE']:+.4f}, P {rep['P(MAE better)']:.2f}")


if __name__ == "__main__":
    main()
