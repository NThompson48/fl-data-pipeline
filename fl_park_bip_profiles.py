"""
fl_park_bip_profiles.py -- BATTED-BALL profiles at Trustmark judged on BALLS IN PLAY only  (user 2026-10-10: K% and BB% effects are closer to global, and Air% is only a function of batted balls)
Rows = balls in play (single, double, triple, home run, out, fielder's choice, error); outcomes per ball in play: wOBA on contact (HR included), HR, triple, double, single.  No K% / BB% terms or controls: non-batted-ball outcomes are not part of this question.
Design as in the production park model (visitors-only; year + hitter + pitcher + [player x home] + park fixed effects; game-cluster SEs), interaction = Trustmark x profile for VISITING hitters / pitchers.
HITTERS : Air% (share of the hitter's balls >= 10 deg, shrunk n0 89), power as production (HR/PA profile) and as HR per BALL IN PLAY (shrunk n0 100), both z-scored.
PITCHERS: Air% allowed, GB (mirror), fly ball + pop-up share (>= 25 deg), HR per ball in play allowed (shrunk n0 100), exit speed allowed, left-handed.
For each: coefficient (per +1 SD, per ball in play), t, contribution split by outcome (coefficient x FL wOBA weight), and by season (2025 / 2026).
"""
import warnings

import numpy as np
import pandas as pd

import fl_park_final_visitors as fv
import fl_park_pitcher_final as pf
import fl_pitcher_fbshare_sticky as st
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
BIP = ["single", "double", "triple", "home_run", "out", "fielders_choice", "error"]; W = {"o_hr": 1.863, "o_3b": 1.534, "o_2b": 1.251, "o_1b": 0.945}


def outcomes(df):
    k = df["kind"].astype(str); df["o_hr"] = (k == "home_run").astype(float); df["o_3b"] = (k == "triple").astype(float); df["o_2b"] = (k == "double").astype(float); df["o_1b"] = (k == "single").astype(float); df["y_woba"] = df["woba"]; return df


def shrunk_z(series_by_pid, n_by_pid, n0, pool_pids):
    lg = np.average(series_by_pid, weights=n_by_pid); d = (series_by_pid - lg) * n_by_pid / (n_by_pid + n0); v = pool_pids.map(d); return (d - v.mean()) / v.std()


def fit_block(est, groups, tmv, terms, labels, title):
    cl = pd.factorize(est["game_id"])[0]; Xr = partial_out(np.column_stack([tmv * est[t].to_numpy(float) for t in terms]), groups, iters=30); rows = []; contrib = {t: 0.0 for t in terms}
    for lab, col in (("wOBA on contact", "y_woba"), ("HR", "o_hr"), ("triple", "o_3b"), ("double", "o_2b"), ("single", "o_1b")):
        yr = partial_out(est[col].to_numpy(float)[:, None], groups, iters=30)[:, 0]; b, se = cluster_ols(Xr, yr, cl); row = {"outcome (per ball in play)": lab, "mean": est[col].mean()}
        for i, t in enumerate(terms):
            row[labels[t]] = f"{b[i]:+.4f} (t {b[i] / se[i]:+.1f})"
            if col in ("o_hr", "o_3b", "o_2b", "o_1b"):
                contrib[t] += b[i] * W[col]
        rows.append(row)
    print(f"\n{title} ({len(est):,} balls in play)\n" + pd.DataFrame(rows).to_string(index=False) + "\n   sum of outcome parts in wOBA points per +1 SD: " + "; ".join(f"{labels[t]} {contrib[t]:+.4f}" for t in terms))
    return Xr


def by_year(est, groups, tmv, terms, labels):
    cl = pd.factorize(est["game_id"])[0]; yr = partial_out(est["y_woba"].to_numpy(float)[:, None], groups, iters=30)[:, 0]; rows = []
    cols = []
    for t in terms:
        for y in (2025, 2026):
            cols.append(partial_out((tmv * (est["year"] == y).to_numpy(float) * est[t].to_numpy(float))[:, None], groups, iters=30)[:, 0])
    b, se = cluster_ols(np.column_stack(cols), yr, cl)
    for i, t in enumerate(terms):
        rows.append({"profile": labels[t], "2025": f"{b[2 * i]:+.4f} ({se[2 * i]:.4f})", "2026": f"{b[2 * i + 1]:+.4f} ({se[2 * i + 1]:.4f})", "t of difference": (b[2 * i] - b[2 * i + 1]) / np.hypot(se[2 * i], se[2 * i + 1])})
    print("   BY SEASON, wOBA on contact (joint terms):\n" + pd.DataFrame(rows).round(4).to_string(index=False))


def hitters():
    est, ml = fv.hitter_frame(); est = est[est["kind"].astype(str).isin(BIP)].reset_index(drop=True); est = outcomes(est); est = est.dropna(subset=["y_woba", "z_air", "z_power"]).reset_index(drop=True)
    non = est[est["tm"] == 0]; g = non.groupby("name_clean").agg(n=("o_hr", "size"), hr=("o_hr", "sum")); lg = g["hr"].sum() / g["n"].sum(); d = (g["hr"] / g["n"] - lg) * g["n"] / (g["n"] + 100); v = est["name_clean"].map(d); est["z_hrbip"] = est["name_clean"].map((d - v.mean()) / v.std())
    est = est.dropna(subset=["z_hrbip"]).reset_index(drop=True); est["z_air_c"] = est["z_air"]
    groups = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pitcher"])[0], pd.factorize(np.where(est["b_home"] == 1, est["name_clean"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    tmv = est["tm"].to_numpy(float) * (est["b_home"] == 0).to_numpy(float); print(f"HITTERS: visiting-hitter balls in play at Trustmark {int(tmv.sum()):,}; corr(Air%, HR/PA power) {est.drop_duplicates('name_clean')[['z_air', 'z_power']].corr().iloc[0, 1]:+.2f}, corr(Air%, HR per ball) {est.drop_duplicates('name_clean')[['z_air', 'z_hrbip']].corr().iloc[0, 1]:+.2f}")
    labels = {"z_air": "Air%", "z_power": "power (HR/PA)", "z_hrbip": "power (HR per BIP)"}
    fit_block(est, groups, tmv, ["z_air"], labels, "H1. Air% alone"); fit_block(est, groups, tmv, ["z_air", "z_power"], labels, "H2. Air% + power (HR/PA), as production"); fit_block(est, groups, tmv, ["z_air", "z_hrbip"], labels, "H3. Air% + power measured per BALL IN PLAY")
    by_year(est, groups, tmv, ["z_air", "z_hrbip"], labels)


def pitchers():
    est, prof, stdz = pf.build(); est = est[est["kind"].astype(str).isin(BIP)].reset_index(drop=True); est = outcomes(est); est["b_home"] = (est["batter_team"] == est["home_team"]).astype(int)
    bb = pf.pitcher_bbe(); bb["pid"] = bb["pitcher_id"].astype(str); bb["park"] = bb["game_id"].str.split("-").str[1]; bb = bb[bb["park"] != "TrustmarkPark"]
    est["z_fbpu"] = est["pid"].map(st.share_z(bb, 25, 91, est, 92)); non = est[est["tm"] == 0]; g = non.groupby("pid").agg(n=("o_hr", "size"), hr=("o_hr", "sum")); est["z_hrbip"] = est["pid"].map(shrunk_z(g["hr"] / g["n"], g["n"], 100, est["pid"]))
    pit = pd.read_parquet("2025_2026_pitches.parquet", columns=["pitcher_id", "stadium", "exit_speed"]); pit = pit[(pit["stadium"] != "TrustmarkPark") & pit["exit_speed"].notna()]; pit["pid"] = pit["pitcher_id"].astype(str); s = pit.groupby("pid")["exit_speed"]; est["z_ev"] = est["pid"].map(st.share_z.__globals__["np"].nan if False else None) if False else np.nan
    m, n = s.mean(), s.size(); lgm = np.average(m, weights=n); dd = (m - lgm) * n / (n + 60); vv = est["pid"].map(dd); est["z_ev"] = est["pid"].map((dd - vv.mean()) / vv.std()); est["z_left"] = (est["lefty"] - est["lefty"].mean()) / est["lefty"].std()
    est = est.dropna(subset=["y_woba", "z_air", "z_fbpu", "z_hrbip", "z_ev", "z_left"]).reset_index(drop=True)
    groups = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pid"])[0], pd.factorize(np.where(est["b_home"] == 0, est["pid"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    tmv = est["tm"].to_numpy(float) * (est["b_home"] == 1).to_numpy(float); labels = {"z_air": "Air% allowed", "z_fbpu": "FB+PU share", "z_hrbip": "HR per BIP allowed", "z_ev": "exit speed allowed", "z_left": "left-handed"}
    print(f"\nPITCHERS: visiting-pitcher balls in play at Trustmark {int(tmv.sum()):,}")
    fit_block(est, groups, tmv, ["z_air"], labels, "P1. Air% allowed alone"); fit_block(est, groups, tmv, ["z_fbpu"], labels, "P2. fly ball + pop-up share alone")
    fit_block(est, groups, tmv, ["z_air", "z_hrbip", "z_ev", "z_left"], labels, "P3. production-style batted-ball profile (Air%, HR per BIP, exit speed, left)")
    by_year(est, groups, tmv, ["z_air", "z_fbpu", "z_hrbip"], labels)


if __name__ == "__main__":
    hitters(); pitchers()
