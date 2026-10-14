"""
fl_park_pitcher_final.py -- final TrustmarkPark translation for PITCHERS (user decision 2026-10-08, mirroring the hitter side): wOBA-allowed effect per PA at Trustmark (full park) =
    g0 + g_hr x z_hr,i + g_k x z_k,i + g_air x z_air,i + g_left x (lefty_i - mean)
pooled 2025-26 (131,767 PA, hitter + pitcher + year + park fixed effects; profile from the pitcher's NON-Trustmark PA / balls: HR/PA allowed shrunk n0 137, K% allowed shrunk n0 113, Air% allowed shrunk n0 89 balls;
z's standardised on the PA-weighted estimation pool), each coefficient shrunk by the positive-part James-Stein factor max(0, 1 - 1/t^2) with t from the GAME-CLUSTER BOOTSTRAP standard error (B = 200 resamples of games,
refitting the fixed effects every time; the cluster-robust analytic SE is reported alongside).  The bootstrap also gives percentile intervals, sign stability, and the covariance of the raw coefficients, which the projection uses
to give every pitcher's Mississippi shift its own standard deviation.
Conversion to runs per 9: x (4.512 BF per IP x 9 / 1.0175 wOBA scale) = x 39.9.
Outputs: output/trustmark_pitcher_coefs.json, output/pitcher_park_profile.parquet (pitcher_id, name_clean, z_hr, z_k, z_air, lefty).
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_innings_projection as ip
import fl_pitcher_features as fp
import fl_pitcher_ros_stuff as pst
from fl_park_pitcher_fe import pitcher_bbe
from fl_park_profile_fe import cluster_ols, partial_out
from fl_park_woba_fe import fe_fit

warnings.filterwarnings("ignore")
TO_R9 = 4.512 * 9 / 1.0175; B_BOOT = 200; N0_HR, N0_K, N0_AIR = 137.0, 113.0, 89.0


def build():
    import fl_hitter_ros as hr
    pa = hr.Tables().pa.copy(); pit = pd.read_parquet("output/pitcher_pa_ros_2025_2026.parquet"); assert len(pit) == len(pa)
    pa["pid"] = pit["pitcher_id"].astype(object).where(pit["pitcher_id"].notna(), "unknown").astype(str).values; pa["park"] = pa["game_id"].str.split("-").str[1]; pa["name_clean"] = pa["name_clean"].fillna("unknown")
    pa = pa[pa["woba"].notna() & pa["park"].notna()].reset_index(drop=True); pa["tm"] = (pa["park"] == "TrustmarkPark").astype(int); kind = pa["kind"].astype(str); pa["is_hr"] = (kind == "home_run").astype(float); pa["is_k"] = (kind == "strikeout").astype(float)
    raw = fp.load_fl_pitches_cached(); hand = raw.groupby("pitcher_id")["p_throws_clean"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan); hand.index = hand.index.astype(str)
    pa["lefty"] = pa["pid"].map(hand).map({"L": 1.0, "R": 0.0, "left": 1.0, "right": 0.0})
    bb = pitcher_bbe(); bb["pid"] = bb["pitcher_id"].astype(str); bb["air"] = (bb["angle"] >= 10).astype(int)
    non = pa[pa["tm"] == 0]; g = non.groupby("pid").agg(n=("woba", "size"), hr=("is_hr", "sum"), k=("is_k", "sum")); lg_hr, lg_k = g["hr"].sum() / g["n"].sum(), g["k"].sum() / g["n"].sum()
    g["hr_d"] = (g["hr"] / g["n"] - lg_hr) * g["n"] / (g["n"] + N0_HR); g["k_d"] = (g["k"] / g["n"] - lg_k) * g["n"] / (g["n"] + N0_K)
    ga = bb.groupby("pid").agg(nb=("air", "size"), air=("air", "sum")); lg_a = ga["air"].sum() / ga["nb"].sum(); ga["air_d"] = (ga["air"] / ga["nb"] - lg_a) * ga["nb"] / (ga["nb"] + N0_AIR)
    for c, s_ in (("hr_d", g["hr_d"]), ("k_d", g["k_d"]), ("air_d", ga["air_d"])):
        pa[c] = pa["pid"].map(s_)
    est = pa.dropna(subset=["hr_d", "k_d", "air_d", "lefty"]).reset_index(drop=True); stdz = {c: (est[c].mean(), est[c].std()) for c in ("hr_d", "k_d", "air_d")}
    for c in ("hr_d", "k_d", "air_d"):
        est["z_" + c[:-2]] = (est[c] - stdz[c][0]) / stdz[c][1]
    prof = pd.DataFrame({"z_hr": (g["hr_d"] - stdz["hr_d"][0]) / stdz["hr_d"][1], "z_k": (g["k_d"] - stdz["k_d"][0]) / stdz["k_d"][1]}).join(((ga["air_d"] - stdz["air_d"][0]) / stdz["air_d"][1]).rename("z_air"), how="outer").join(hand.map({"L": 1.0, "R": 0.0, "left": 1.0, "right": 0.0}).rename("lefty"), how="outer")
    return est, prof, stdz


def fit_once(est, iters=40):
    groups = [pd.factorize(est[c])[0] for c in ("year", "name_clean", "pid", "park")]; parks = list(pd.factorize(est["park"])[1]); tmv = est["tm"].to_numpy(float); lm = est["lefty"].mean()
    y = est["woba"].to_numpy(float); mu, eff = fe_fit(y, groups, iters=iters); pe = eff[3] - np.average(eff[3], weights=np.bincount(groups[3])); g0 = float(pe[parks.index("TrustmarkPark")])
    X = np.column_stack([tmv * est["z_hr"], tmv * est["z_k"], tmv * est["z_air"], tmv * (est["lefty"] - lm)]); Xr = partial_out(X, groups, iters=iters); yr = partial_out(y[:, None], groups, iters=iters)[:, 0]
    return g0, Xr, yr, lm


def main():
    est, prof, stdz = build(); print(f"estimation sample {len(est):,} PA ({int(est['tm'].sum()):,} at Trustmark), {est['pid'].nunique()} pitchers")
    g0, Xr, yr, lm = fit_once(est); cl = pd.factorize(est["game_id"])[0]; b, se = cluster_ols(Xr, yr, cl); raw = np.concatenate([[g0], b]); names = ["global", "hr_prone", "k_rate", "air", "left"]
    rng = np.random.default_rng(5); games = est["game_id"].to_numpy(); ug = pd.unique(games); by_game = pd.Series(np.arange(len(est))).groupby(games).apply(np.array); draws = []
    for i in range(B_BOOT):
        ii = np.concatenate([by_game[g] for g in rng.choice(ug, len(ug))]); sub = est.iloc[ii].reset_index(drop=True)
        try:
            g_b, Xb, yb, _ = fit_once(sub, iters=20); bb_, _ = cluster_ols(Xb, yb, pd.factorize(sub["game_id"])[0]); draws.append(np.concatenate([[g_b], bb_]))
        except Exception:
            continue
        if (i + 1) % 25 == 0:
            print(f"  bootstrap {i + 1}/{B_BOOT}", flush=True)
    D = np.array(draws); se_b = D.std(axis=0); cov = np.cov(D.T); out = {}
    for j, k in enumerate(names):
        t = raw[j] / se_b[j]; sh = max(0.0, 1 - 1 / t ** 2) if abs(t) > 1 else 0.0; lo, hi = np.percentile(D[:, j], [5, 95])
        out[k] = {"raw": float(raw[j]), "se_cluster_robust": float(se[j - 1]) if j else None, "se_bootstrap": float(se_b[j]), "t_bootstrap": float(t), "ci90": [float(lo), float(hi)], "same_sign_share": float((np.sign(D[:, j]) == np.sign(raw[j])).mean()), "shrink": sh, "final": float(raw[j] * sh)}
    prof.index.name = "pitcher_id"; prof = prof.reset_index(); mp = pst.idmap(fp.load_fl_pitches_cached()); mp.index = mp.index.astype(str); prof["name_clean"] = prof["pitcher_id"].astype(str).map(mp); prof.to_parquet("output/pitcher_park_profile.parquet", index=False)
    json.dump({"coefs": out, "names": names, "covariance_raw": cov.tolist(), "shrink": [out[k]["shrink"] for k in names], "mean_lefty": lm, "to_r9": TO_R9, "standardisation": {k: {"mean": v[0], "sd": v[1]} for k, v in stdz.items()}, "bootstrap_draws": int(len(D)),
               "estimation": {"pa": int(len(est)), "pa_trustmark": int(est["tm"].sum())}}, open("output/trustmark_pitcher_coefs.json", "w"), indent=1)
    t = pd.DataFrame(out).T[["raw", "se_cluster_robust", "se_bootstrap", "t_bootstrap", "ci90", "same_sign_share", "shrink", "final"]]; t["final runs/9 per +1 SD"] = t["final"] * TO_R9
    print(f"\n{len(D)} bootstrap draws (games resampled, fixed effects refit each time)\n" + t.round(4).to_string())


if __name__ == "__main__":
    main()
