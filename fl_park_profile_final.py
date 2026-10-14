"""
fl_park_profile_final.py -- final TrustmarkPark wOBA translation for hitters: GLOBAL effect + PROFILE terms, pooled 2025-26, shrunk (user decision 2026-10-08 replacing the HR-factor translation).
  effect_i (full park, wOBA points per PA at Trustmark) = g0 + g_air x z_air,i + g_power x z_power,i + g_left x (left_i - mean_left)
  g0       the Trustmark park fixed effect (hitter + pitcher + year + park FE), z_air / z_power / left the hitter profile measured on his NON-Trustmark PA and balls (Air% shrunk n0 = 89 balls; HR/PA shrunk n0 = 137 PA;
           left = share of his PA taken from the left side); z's standardised on the PA-weighted pool of the estimation sample.
  Shrinkage: positive-part James-Stein on each coefficient, factor max(0, 1 - 1/t^2), t = coefficient / cluster-robust SE (by game): a coefficient that is not distinguishable from zero is shrunk toward 0 (the global
  effect, t ~ 0, goes to exactly 0; Air% t 3.3 keeps ~91%, power t -2.8 ~87%, left-handed t 1.2 ~34%).
Outputs: output/trustmark_profile_coefs.json (raw, se, t, shrink, final coefficients and the standardisation constants) and output/hitter_park_profile.parquet (name_clean, z_air, z_power, left_share, n_pa_nontm, n_bbe).
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_hitter_ev as hev
import fl_hitter_ros as hr
import fl_hitter_whiff as hwf
import fl_innings_projection as ip
from fl_park_profile_fe import cluster_ols, partial_out
from fl_park_woba_fe import fe_fit

warnings.filterwarnings("ignore")
N0_POWER, N0_AIR = 137.0, 89.0


def load_pa():
    T = hr.Tables(); pa = T.pa.copy(); pit = pd.read_parquet("output/pitcher_pa_ros_2025_2026.parquet"); assert len(pit) == len(pa)
    pa["pitcher"] = pit["pitcher_id"].astype(object).where(pit["pitcher_id"].notna(), "unknown").astype(str).values; pa["park"] = pa["game_id"].str.split("-").str[1]; pa["name_clean"] = pa["name_clean"].fillna("unknown")
    pa = pa[pa["woba"].notna() & pa["park"].notna()].reset_index(drop=True); pa["tm"] = (pa["park"] == "TrustmarkPark").astype(int)
    pi = hwf.load(); pi = pi[pi["year"].isin([2025, 2026]) & pi["pa_id"].notna() & pi["batter_side"].isin(["left", "right"])].copy(); pi["name_clean"] = pi["batter"].apply(ip.lastfirst_to_clean)
    side = pi.sort_values("pitch_no").groupby(["game_id", "pa_id", "name_clean"])["batter_side"].first().rename("side").reset_index(); pa = pa.merge(side, on=["game_id", "pa_id", "name_clean"], how="left")
    pa["left"] = np.where(pa["side"] == "left", 1.0, np.where(pa["side"] == "right", 0.0, np.nan)); pa["is_hr"] = (pa["kind"].astype(str) == "home_run").astype(float)
    return pa


def main():
    pa = load_pa(); non = pa[pa["tm"] == 0]; g = non.groupby("name_clean").agg(n=("woba", "size"), hr=("is_hr", "sum")); lg_hr = g["hr"].sum() / g["n"].sum(); g["power"] = (g["hr"] / g["n"] - lg_hr) * g["n"] / (g["n"] + N0_POWER)
    bb = hev.load_bbe(); bb["air"] = (bb["angle"] >= 10).astype(int); bb = bb[bb["year"].isin([2025, 2026])]
    # Air% uses all tracked balls (the ball table carries no park; Trustmark is about 5% of the plate appearances, a small contamination)
    ga = bb.groupby("name_clean").agg(n=("air", "size"), air=("air", "sum")); lg_air = ga["air"].sum() / ga["n"].sum(); ga["air_dev"] = (ga["air"] / ga["n"] - lg_air) * ga["n"] / (ga["n"] + N0_AIR)
    left_share = pa[pa["left"].notna()].groupby("name_clean")["left"].mean()
    pa["power_d"] = pa["name_clean"].map(g["power"]); pa["air_d"] = pa["name_clean"].map(ga["air_dev"]); est = pa.dropna(subset=["power_d", "air_d", "left"]).reset_index(drop=True)
    sd_p, mu_p = est["power_d"].std(), est["power_d"].mean(); sd_a, mu_a = est["air_d"].std(), est["air_d"].mean(); mu_l = est["left"].mean()
    est["z_power"] = (est["power_d"] - mu_p) / sd_p; est["z_air"] = (est["air_d"] - mu_a) / sd_a
    groups = [pd.factorize(est[c])[0] for c in ("year", "name_clean", "pitcher", "park")]; parks = pd.factorize(est["park"])[1]; cl = pd.factorize(est["game_id"])[0]; tmv = est["tm"].to_numpy(float)
    y = est["woba"].to_numpy(float); mu, eff = fe_fit(y, groups, iters=40); pe = pd.Series(eff[3] - np.average(eff[3], weights=np.bincount(groups[3])), index=parks); g0 = float(pe["TrustmarkPark"])
    X = np.column_stack([tmv * est["z_air"], tmv * est["z_power"], tmv * (est["left"] - mu_l)]); Xr = partial_out(X, groups, iters=40); yr = partial_out(y[:, None], groups, iters=40)[:, 0]; b, se = cluster_ols(Xr, yr, cl)
    # SE of the global effect: cluster bootstrap over games (as in fl_park_woba_fe) is slow; use the cluster-robust SE of the Trustmark dummy in the same FE model
    se_g = float(pd.read_csv("output/park_woba_fe_2025_26.csv", index_col=0).loc["TrustmarkPark", "se"])                  # game-bootstrap SE of the Trustmark park effect (fl_park_woba_fe.py); the park dummy is collinear with the park FE
    coef = {"global": (g0, se_g), "air": (float(b[0]), float(se[0])), "power": (float(b[1]), float(se[1])), "left": (float(b[2]), float(se[2]))}; out = {}
    for k, (v, s) in coef.items():
        t = v / s; sh = max(0.0, 1 - 1 / t ** 2) if abs(t) > 1 else 0.0; out[k] = {"raw": v, "se": s, "t": t, "shrink": sh, "final": v * sh}
    prof = pd.DataFrame({"z_power": ((g["power"] - mu_p) / sd_p), "n_pa_nontm": g["n"]}).join(pd.DataFrame({"z_air": (ga["air_dev"] - mu_a) / sd_a, "n_bbe": ga["n"]}), how="outer").join(left_share.rename("left_share"), how="outer")
    prof.index.name = "name_clean"; prof.to_parquet("output/hitter_park_profile.parquet")
    js = {"coefs": out, "standardisation": {"power_sd": sd_p, "power_mean": mu_p, "air_sd": sd_a, "air_mean": mu_a, "mean_left": mu_l, "n0_power_pa": N0_POWER, "n0_air_balls": N0_AIR}, "estimation": {"pa": int(len(est)), "pa_trustmark": int(tmv.sum()), "years": [2025, 2026]}}
    json.dump(js, open("output/trustmark_profile_coefs.json", "w"), indent=1)
    print(f"estimation sample {len(est):,} PA ({int(tmv.sum()):,} at Trustmark); hitters in the profile table {len(prof)}")
    print(pd.DataFrame(out).T.round(4).to_string())
    print(f"example: effect for a +1 SD power / -1 SD Air% RHB = {out['global']['final'] + out['air']['final'] * -1 + out['power']['final'] * 1 - 0:+.4f}; for a +1 SD Air% / -1 SD power LHB = {out['global']['final'] + out['air']['final'] + out['power']['final'] * -1 + out['left']['final'] * (1 - mu_l):+.4f}")


if __name__ == "__main__":
    main()
