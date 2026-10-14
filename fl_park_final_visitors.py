"""
fl_park_final_visitors.py -- the Trustmark translations (hitters and pitchers) re-estimated with VISITING PLAYERS ONLY (user 2026-10-08), game-cluster bootstrap (B = 200), positive-part James-Stein shrinkage on the
bootstrap t.  Replaces the all-PA coefficient files (kept as *_allpa.json).
  HITTERS : wOBA = year + hitter + pitcher + [hitter x home] + park; Trustmark x profile applies to VISITING hitters only (home hitters' home PA are absorbed by their own home effect), profile = z_air, z_power, left;
            global = the Trustmark park effect (from visiting hitters).
  PITCHERS: wOBA allowed = year + hitter + pitcher + [pitcher x home] + park; Trustmark x profile for VISITING pitchers only, profile = z_hr, z_k, z_air, lefty.
  The park effect on the home participants' side (Mississippi's staff / lineup) is a roster-fit effect (see the matched pair in fl_park_visitors_fe.py), not a property of the park, so it is excluded by design.
  Global effect used in production = the average of the visiting-hitter and visiting-pitcher estimates of the same park (they have opposite-signed home-team-fit contamination); each side's profile terms come from its own visitors.
Outputs (same schema as before): output/trustmark_profile_coefs.json (hitters), output/trustmark_pitcher_coefs.json (pitchers); profile tables unchanged.
"""
import json
import shutil
import warnings

import numpy as np
import pandas as pd

from fl_park_pitcher_final import build as pitcher_build, TO_R9
from fl_park_profile_final import load_pa
from fl_park_profile_fe import cluster_ols, partial_out
from fl_park_woba_fe import fe_fit

warnings.filterwarnings("ignore")
B_BOOT = 200


def hitter_frame():
    pa = load_pa(); prof = pd.read_parquet("output/hitter_park_profile.parquet"); js = json.load(open("output/trustmark_profile_coefs.json")); ml = js["standardisation"]["mean_left"]
    pa["z_air"] = pa["name_clean"].map(prof["z_air"]); pa["z_power"] = pa["name_clean"].map(prof["z_power"]); est = pa.dropna(subset=["z_air", "z_power", "left"]).reset_index(drop=True)
    est["b_home"] = (est["batter_team"] == est["home_team"]).astype(int); return est, ml


def fit_side(est, side, iters=40):
    """Returns (global park effect from visitors, Xr, yr) for the Trustmark x profile regression on visitors."""
    home_batter = (est["b_home"] == 1)
    if side == "hitter":
        g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pitcher"])[0], pd.factorize(np.where(home_batter, est["name_clean"], "AWAY"))[0], pd.factorize(est["park"])[0]]; vis = (~home_batter).to_numpy()
    else:
        g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pid"])[0], pd.factorize(np.where(~home_batter, est["pid"], "AWAY"))[0], pd.factorize(est["park"])[0]]; vis = home_batter.to_numpy()
    y = est["woba"].to_numpy(float); mu, eff = fe_fit(y, g, iters=iters); parks = list(pd.factorize(est["park"])[1]); w = np.bincount(g[-1][vis], minlength=len(parks)); pe = eff[-1] - np.average(eff[-1], weights=np.maximum(w, 1e-9)); g0 = float(pe[parks.index("TrustmarkPark")])
    tmv = est["tm"].to_numpy(float) * vis
    if side == "hitter":
        X = np.column_stack([tmv * est["z_air"], tmv * est["z_power"], tmv * (est["left"] - est["left"].mean())])
    else:
        X = np.column_stack([tmv * est["z_hr"], tmv * est["z_k"], tmv * est["z_air"], tmv * (est["lefty"] - est["lefty"].mean())])
    Xr = partial_out(X, g, iters=iters); yr = partial_out(y[:, None], g, iters=iters)[:, 0]; return g0, Xr, yr


def run(est, side, names, tag):
    g0, Xr, yr = fit_side(est, side); cl = pd.factorize(est["game_id"])[0]; b, se = cluster_ols(Xr, yr, cl); raw = np.concatenate([[g0], b]); rng = np.random.default_rng(21); games = est["game_id"].to_numpy(); ug = pd.unique(games)
    by_game = pd.Series(np.arange(len(est))).groupby(games).apply(np.array); draws = []
    for i in range(B_BOOT):
        ii = np.concatenate([by_game[g] for g in rng.choice(ug, len(ug))]); sub = est.iloc[ii].reset_index(drop=True)
        try:
            g_b, Xb, yb = fit_side(sub, side, iters=20); bb, _ = cluster_ols(Xb, yb, pd.factorize(sub["game_id"])[0]); draws.append(np.concatenate([[g_b], bb]))
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  {tag}: bootstrap {i + 1}/{B_BOOT}", flush=True)
    D = np.array(draws); se_b = D.std(axis=0); out = {}
    for j, k in enumerate(names):
        t = raw[j] / se_b[j]; sh = max(0.0, 1 - 1 / t ** 2) if abs(t) > 1 else 0.0; lo, hi = np.percentile(D[:, j], [5, 95])
        out[k] = {"raw": float(raw[j]), "se_cluster_robust": float(se[j - 1]) if j else None, "se_bootstrap": float(se_b[j]), "t_bootstrap": float(t), "ci90": [float(lo), float(hi)], "same_sign_share": float((np.sign(D[:, j]) == np.sign(raw[j])).mean()), "shrink": sh, "final": float(raw[j] * sh)}
    return out, D


def main():
    for f in ("trustmark_profile_coefs.json", "trustmark_pitcher_coefs.json"):
        shutil.copy(f"output/{f}", f"output/{f.replace('.json', '_allpa.json')}")
    # ---------------- hitters
    est_h, ml = hitter_frame(); print(f"HITTERS: {len(est_h):,} PA, {int(est_h['tm'].sum()):,} at Trustmark, of which visiting {int(((est_h['b_home'] == 0) & (est_h['tm'] == 1)).sum()):,}")
    nh = ["global", "air", "power", "left"]; out_h, Dh = run(est_h, "hitter", nh, "hitters")
    # ---------------- pitchers
    est_p, prof_p, stdz = pitcher_build(); est_p["b_home"] = (est_p["batter_team"] == est_p["home_team"]).astype(int); mlp = est_p["lefty"].mean(); print(f"PITCHERS: {len(est_p):,} PA, visiting pitchers at Trustmark {int(((est_p['b_home'] == 1) & (est_p['tm'] == 1)).sum()):,}")
    npt = ["global", "hr_prone", "k_rate", "air", "left"]; out_p, Dp = run(est_p, "pitcher", npt, "pitchers")
    # ---------------- combined global effect (average of the two visitor estimates; their home-team-fit contamination has opposite sign)
    n = min(len(Dh), len(Dp)); gcomb = (Dh[:n, 0] + Dp[:n, 0]) / 2; g_raw = (out_h["global"]["raw"] + out_p["global"]["raw"]) / 2; g_se = float(gcomb.std()); t_g = g_raw / g_se; sh_g = max(0.0, 1 - 1 / t_g ** 2) if abs(t_g) > 1 else 0.0
    glob = {"raw": g_raw, "se_bootstrap": g_se, "t_bootstrap": t_g, "ci90": [float(np.percentile(gcomb, 5)), float(np.percentile(gcomb, 95))], "shrink": sh_g, "final": g_raw * sh_g, "note": "average of the visiting-hitter and visiting-pitcher estimates of the same park"}
    out_h["global"].update({"visiting_hitters_raw": out_h["global"]["raw"], "raw": g_raw, "se_bootstrap": g_se, "t_bootstrap": t_g, "ci90": glob["ci90"], "shrink": sh_g, "final": g_raw * sh_g}); out_p["global"].update({"visiting_pitchers_raw": out_p["global"]["raw"], "raw": g_raw, "se_bootstrap": g_se, "t_bootstrap": t_g, "ci90": glob["ci90"], "shrink": sh_g, "final": g_raw * sh_g})
    # hitters JSON (schema as before)
    js = json.load(open("output/trustmark_profile_coefs_allpa.json")); js["coefs"] = {k: {kk: vv for kk, vv in v.items()} for k, v in out_h.items()}; js["estimation"] = {"design": "visiting hitters only (hitter x home effects absorb home hitters)", "pa": int(len(est_h)), "bootstrap_draws": int(len(Dh))}
    json.dump(js, open("output/trustmark_profile_coefs.json", "w"), indent=1)
    # pitchers JSON: covariance of the raw coefficients with the combined global (own variance + the pitcher-side covariances kept)
    cov = np.cov(Dp.T); cov[0, 0] = g_se ** 2
    jp = json.load(open("output/trustmark_pitcher_coefs_allpa.json")); jp["coefs"] = out_p; jp["covariance_raw"] = cov.tolist(); jp["shrink"] = [out_p[k]["shrink"] for k in npt]; jp["bootstrap_draws"] = int(len(Dp)); jp["mean_lefty"] = float(mlp); jp["estimation"] = {"design": "visiting pitchers only (pitcher x home effects absorb home pitchers)", "pa": int(len(est_p))}
    json.dump(jp, open("output/trustmark_pitcher_coefs.json", "w"), indent=1)
    show = lambda o, nm: pd.DataFrame(o).T[["raw", "se_cluster_robust", "se_bootstrap", "t_bootstrap", "ci90", "same_sign_share", "shrink", "final"]].round(4)
    print("\nHITTERS (visiting hitters only; per +1 SD / left vs average; global = average of the two visitor estimates)\n" + show(out_h, "h").to_string()); print("\nPITCHERS (visiting pitchers only; wOBA allowed per +1 SD, positive = worse for the pitcher)\n" + show(out_p, "p").to_string())
    print(f"\nGLOBAL (both sides): raw {g_raw:+.4f}, bootstrap SE {g_se:.4f}, 90% [{glob['ci90'][0]:+.4f}, {glob['ci90'][1]:+.4f}], t {t_g:.2f}, shrink {sh_g:.2f}, final {glob['final']:+.4f}")


if __name__ == "__main__":
    main()
