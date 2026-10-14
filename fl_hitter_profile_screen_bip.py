"""
fl_hitter_profile_screen_bip.py -- WHICH HITTER PROFILES MATTER AT TRUSTMARK, on balls in play, incl. handedness  (user 2026-10-10: "Let's dive into different hitter profiles, including handedness")
Rows = balls in play of VISITING hitters (+ the fixed effects' other rows): production park design (year + hitter + pitcher + [hitter x home] + park effects, game-cluster SEs).  Outcomes per ball in play: wOBA on contact (primary), HR, and non-HR hits.
Baseline model = Air% + power (HR per ball in play).  Candidates, each measured on the hitter's NON-Trustmark balls, shrunk, z-scored:
  batted-ball type  ground / line-drive / fly / pop-up / fly+pop-up share           exit speed       mean EV (n0 60), EV90 (n0 14), hard-hit >= 95 mph share (n0 60), barrel share (n0 120)
  spray             pull share, opposite-field share, pull-air share, oppo-air share (+-15 degrees, air = >= 10 degrees)
  HANDEDNESS        left (the side used in that PA), left x pull-air, by-side slopes of Air% and power, same-hand vs opposite-hand matchup with the pitcher
For each candidate added to the baseline: coefficient / t, by outcome, by season, and the out-of-season test (fit on one season, predict the other season's Trustmark balls in play: share of squared error removed vs the baseline, game-cluster bootstrap P(better)).
"""
import warnings

import numpy as np
import pandas as pd

import fl_park_final_visitors as fv
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
BIPK = ["single", "double", "triple", "home_run", "out", "fielders_choice", "error"]


def build():
    est, ml = fv.hitter_frame(); est = est[est["kind"].astype(str).isin(BIPK)].copy()
    d = pd.read_parquet("2025_2026_pitches.parquet", columns=["game_id", "pa_id", "is_terminal_pitch", "direction", "batter_side", "barrel"]); t = d[d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"])]
    est = est.merge(t[["game_id", "pa_id", "direction", "batter_side", "barrel"]], on=["game_id", "pa_id"], how="left").reset_index(drop=True)
    k = est["kind"].astype(str); est["o_hr"] = (k == "home_run").astype(float); est["o_hit"] = k.isin(["single", "double", "triple"]).astype(float); est["y"] = est["woba"]
    est["pull_deg"] = np.where(est["batter_side"] == "right", -est["direction"], np.where(est["batter_side"] == "left", est["direction"], np.nan))
    est["air"] = (est["angle"] >= 10).astype(float).where(est["angle"].notna()); est["pull"] = (est["pull_deg"] > 15).astype(float).where(est["pull_deg"].notna()); est["oppo"] = (est["pull_deg"] < -15).astype(float).where(est["pull_deg"].notna())
    est["pull_air"] = (est["pull"] * est["air"]); est["oppo_air"] = (est["oppo"] * est["air"])
    for nm, lo, hi in (("gb", -91, 10), ("ld", 10, 25), ("fb", 25, 40), ("pu", 40, 91), ("fbpu", 25, 91)):
        est[nm] = ((est["angle"] >= lo) & (est["angle"] < hi)).astype(float).where(est["angle"].notna())
    est["hard"] = (est["exit_speed"] >= 95).astype(float).where(est["exit_speed"].notna()); est["brl"] = est["barrel"].astype(float)
    phand = pd.read_parquet("output/pitcher_park_profile.parquet").drop_duplicates("pitcher_id"); phand["pitcher_id"] = phand["pitcher_id"].astype(str); est["p_lefty"] = est["pitcher"].astype(str).map(phand.set_index("pitcher_id")["lefty"])
    est["same_hand"] = np.where(est["p_lefty"].notna(), (est["left"] == est["p_lefty"]).astype(float), np.nan); return est.dropna(subset=["y", "z_air", "z_power"]).reset_index(drop=True)


def add_feature(est, col, n0, name):
    non = est[(est["tm"] == 0) & est[col].notna()]; g = non.groupby("name_clean")[col].agg(["mean", "size"]); lg = np.average(g["mean"], weights=g["size"]); d = (g["mean"] - lg) * g["size"] / (g["size"] + n0); v = est["name_clean"].map(d); est[name] = (est["name_clean"].map(d) - v.mean()) / v.std()


def main():
    est = build(); print(f"{len(est):,} balls in play (HR included); visiting-hitter balls at Trustmark {int(((est['tm'] == 1) & (est['b_home'] == 0)).sum()):,}; switch-side rows: left share {est['left'].mean():.3f}")
    non = est[est["tm"] == 0]; g = non.groupby("name_clean")["o_hr"].agg(["mean", "size"]); lg = np.average(g["mean"], weights=g["size"]); d = (g["mean"] - lg) * g["size"] / (g["size"] + 100); v = est["name_clean"].map(d); est["z_hrbip"] = (est["name_clean"].map(d) - v.mean()) / v.std()
    for col, n0, nm in (("gb", 89, "z_gb"), ("ld", 89, "z_ld"), ("fb", 89, "z_fb"), ("pu", 89, "z_pu"), ("fbpu", 89, "z_fbpu"), ("hard", 60, "z_hard"), ("brl", 120, "z_brl"), ("pull", 60, "z_pull"), ("oppo", 60, "z_oppo"), ("pull_air", 89, "z_pullair"), ("oppo_air", 89, "z_oppoair")):
        add_feature(est, col, n0, nm)
    non = est[(est["tm"] == 0) & est["exit_speed"].notna()]; g = non.groupby("name_clean")["exit_speed"].agg(["mean", "size", lambda s: np.percentile(s, 90) if len(s) >= 15 else np.nan]); g.columns = ["mean", "size", "p90"]
    lgm = np.average(g["mean"], weights=g["size"]); est["z_ev"] = est["name_clean"].map((g["mean"] - lgm) * g["size"] / (g["size"] + 60)); lg9 = g["p90"].mean(); est["z_ev90"] = est["name_clean"].map((g["p90"] - lg9) * g["size"] / (g["size"] + 14.26))
    for c in ("z_ev", "z_ev90"):
        est[c] = (est[c] - est[c].mean()) / est[c].std()
    est["left_c"] = est["left"] - est["left"].mean(); est["z_left"] = est["left_c"] / est["left"].std()
    for base in ("z_air", "z_hrbip", "z_pullair"):
        est[base + "_L"] = est[base] * est["left"]; est[base + "_R"] = est[base] * (1 - est["left"])
    est["pullair_x_left"] = est["z_pullair"] * est["left_c"]; est["same_hand_c"] = est["same_hand"] - est["same_hand"].mean()
    cand = {"ground-ball share": "z_gb", "line-drive share": "z_ld", "fly-ball share (25-40)": "z_fb", "pop-up share": "z_pu", "fly + pop-up share": "z_fbpu", "mean exit speed": "z_ev", "EV90": "z_ev90", "hard-hit share (>= 95)": "z_hard", "barrel share": "z_brl", "pull share": "z_pull",
            "opposite-field share": "z_oppo", "pull-air share": "z_pullair", "oppo-air share": "z_oppoair", "LEFT-HANDED (side used)": "left_c", "same-hand matchup vs pitcher": "same_hand_c", "pull-air x left": "pullair_x_left"}
    est = est.dropna(subset=["z_hrbip", "z_ev", "z_ev90", "z_gb", "z_pullair", "z_hard", "z_pull"]).copy(); est = est[est["same_hand"].notna() | True].reset_index(drop=True); est["same_hand_c"] = est["same_hand_c"].fillna(0.0)
    groups = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pitcher"])[0], pd.factorize(np.where(est["b_home"] == 1, est["name_clean"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    vis = (est["b_home"] == 0).to_numpy(float); tmv = est["tm"].to_numpy(float) * vis; on = tmv == 1; cl = pd.factorize(est["game_id"])[0]; yrs = est["year"].to_numpy()
    ys = {"wOBAcon": partial_out(est["y"].to_numpy(float)[:, None], groups, iters=30)[:, 0], "HR": partial_out(est["o_hr"].to_numpy(float)[:, None], groups, iters=30)[:, 0], "hit": partial_out(est["o_hit"].to_numpy(float)[:, None], groups, iters=30)[:, 0]}
    allcols = ["z_air", "z_hrbip"] + sorted(set(cand.values()) | {"z_air_L", "z_air_R", "z_hrbip_L", "z_hrbip_R", "z_pullair_L", "z_pullair_R"}); Xr = {c: partial_out((tmv * est[c].to_numpy(float))[:, None], groups, iters=30)[:, 0] for c in allcols}
    Xy = {c: {y: partial_out((tmv * (yrs == y) * est[c].to_numpy(float))[:, None], groups, iters=30)[:, 0] for y in (2025, 2026)} for c in ["z_air", "z_hrbip"] + list(cand.values())}

    def reg(cols, y="wOBAcon", rows=None):
        X = np.column_stack([Xr[c] for c in cols]); return cluster_ols(X, ys[y], cl)
    b, se = reg(["z_air", "z_hrbip"]); print(f"\nBASELINE (visiting hitters, wOBA on contact per +1 SD): Air% {b[0]:+.4f} (t {b[0] / se[0]:+.1f}); power (HR per BIP) {b[1]:+.4f} (t {b[1] / se[1]:+.1f})")
    rng = np.random.default_rng(9); rows = []; base_cols = ["z_air", "z_hrbip"]
    for lab, c in cand.items():
        row = {"candidate (added to Air% + power)": lab}
        for y_, ylab in (("wOBAcon", "wOBA on contact"), ("HR", "HR/BIP"), ("hit", "non-HR hit/BIP")):
            b, se = reg(base_cols + [c], y_); row[ylab] = f"{b[2]:+.4f} (t {b[2] / se[2]:+.1f})"
        b, se = reg(base_cols + [c]); row["t"] = b[2] / se[2]; b1, se1 = reg([c]); row["alone: coef (t)"] = f"{b1[0]:+.4f} ({b1[0] / se1[0]:+.1f})"
        X_by = np.column_stack([Xy[x][y_] for x in base_cols + [c] for y_ in (2025, 2026)]); bb, ss = cluster_ols(X_by, ys["wOBAcon"], cl); row["2025"] = f"{bb[4]:+.4f}"; row["2026"] = f"{bb[5]:+.4f}"; row["year diff t"] = (bb[4] - bb[5]) / np.hypot(ss[4], ss[5])
        X0 = np.column_stack([Xr[x] for x in base_cols]); X1 = np.column_stack([Xr[x] for x in base_cols + [c]]); yv = ys["wOBAcon"]
        for a, t_ in ((2025, 2026), (2026, 2025)):
            tr, te = yrs == a, yrs == t_; b0 = np.linalg.lstsq(X0[tr], yv[tr], rcond=None)[0]; b1_ = np.linalg.lstsq(X1[tr], yv[tr], rcond=None)[0]; m = te & on; e0 = yv[m] - X0[m] @ b0; e1 = yv[m] - X1[m] @ b1_
            gc = pd.factorize(est["game_id"][m])[0]; by = pd.Series(np.arange(m.sum())).groupby(gc).apply(np.array); ug = by.index.to_numpy(); better = 0
            for _ in range(300):
                ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); better += (e1[ii] ** 2).sum() < (e0[ii] ** 2).sum()
            row[f"{a}->{t_} SSE gain"] = 1 - (e1 ** 2).sum() / (e0 ** 2).sum(); row[f"P {a}->{t_}"] = better / 300
        rows.append(row)
    t = pd.DataFrame(rows); print("\nCANDIDATES ADDED TO THE BASELINE (per +1 SD per ball in play at Trustmark)\n" + t.sort_values("t", key=lambda s: -s.abs()).round(4).to_string(index=False))
    # handedness block
    print("\nHANDEDNESS BLOCK")
    b, se = reg(["z_air_L", "z_air_R", "z_hrbip_L", "z_hrbip_R"]); names = ["Air% (LHH)", "Air% (RHH)", "power (LHH)", "power (RHH)"]
    for i, n in enumerate(names):
        print(f"  {n}: {b[i]:+.4f} (t {b[i] / se[i]:+.1f})")
    V = np.diag(se ** 2); print(f"  difference LHH - RHH: Air% {b[0] - b[1]:+.4f} (t {(b[0] - b[1]) / np.hypot(se[0], se[1]):+.1f}); power {b[2] - b[3]:+.4f} (t {(b[2] - b[3]) / np.hypot(se[2], se[3]):+.1f})   [independent-SE approximation]")
    b, se = reg(["z_air", "z_hrbip", "left_c", "z_pullair_L", "z_pullair_R"]); print(f"  left main effect (with Air% + power): {b[2]:+.4f} (t {b[2] / se[2]:+.1f}); pull-air share: LHH {b[3]:+.4f} (t {b[3] / se[3]:+.1f}), RHH {b[4]:+.4f} (t {b[4] / se[4]:+.1f})")
    for lab, hand in (("LHH rows", 1), ("RHH rows", 0)):
        m = est["left"].to_numpy() == hand; print(f"  {lab}: {int((m & on).sum())} visiting balls at Trustmark; mean wOBA on contact there {est.loc[m & on, 'y'].mean():.3f} vs league {est.loc[m, 'y'].mean():.3f}")


if __name__ == "__main__":
    main()
