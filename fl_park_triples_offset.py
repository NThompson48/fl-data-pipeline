"""
fl_park_triples_offset.py -- does Trustmark's triples boost offset its HR suppression, and does the Air-focused profile cover both?  (user 2026-10-09)
PART A  PARK-BLIND OUTCOME BENCHMARK per ball in play (80.7k tracked balls, HR included): cross-fitted (5-fold by game) multiclass gradient boosting of {out, single, double, triple, HR} on exit speed, launch angle, field direction, batter side.
        By park: actual / expected for triples and HR.  At Trustmark: actual - expected per outcome converted to wOBA with the FL linear weights (single .945, double 1.251, triple 1.534, HR 1.863, outs 0): how much of the HR loss is offset by
        triples, doubles and singles?  Game-cluster bootstrap for the net.  Also where the triples come from (field sector).
PART B  DOES THE AIR PROFILE COVER BOTH?  The production park model (visitors-only fixed effects, game-cluster SEs): for VISITING hitters at Trustmark, interaction of Trustmark with the Air% profile and the power (HR/PA) profile on each outcome PER PA:
        HR, triple, double, single, K, BB, wOBA; the Air% term is then converted to wOBA by outcome (coef x weight) to show how its effect splits between home runs and the other hits.  A hitter SPEED proxy (stolen bases per PA, shrunk n0 150, standardised) is added to see whether triples need their own profile.
        Pitcher side: the production pitcher profile (HR-prone, K%, Air%, left) on the same outcomes for visiting pitchers.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

import fl_park_final_visitors as fv
import fl_park_pitcher_final as pf
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
TM = "TrustmarkPark"; W = {"out": 0.0, "single": 0.945, "double": 1.251, "triple": 1.534, "hr": 1.863}; CLS = ["out", "single", "double", "triple", "hr"]


def part_a():
    d = pd.read_parquet("2025_2026_pitches.parquet")
    b = d[d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]) & d["exit_speed"].notna() & d["angle"].notna() & d["direction"].notna() & d["play_result"].isin(["out", "single", "double", "triple", "error", "fielders_choice", "sacrifice", "home_run"]) & d["batter_side"].isin(["left", "right"])].copy().reset_index(drop=True)
    b["tm"] = (b["stadium"] == TM).astype(int); b["left"] = (b["batter_side"] == "left").astype(int); pr = b["play_result"]; b["cls"] = np.select([pr == "home_run", pr == "triple", pr == "double", pr == "single"], [4, 3, 2, 1], 0)
    X = pd.DataFrame({"ev": b["exit_speed"], "angle": b["angle"], "dir": b["direction"], "side": b["left"]}); P = np.zeros((len(b), 5)); y = b["cls"].to_numpy()
    for tr, te in GroupKFold(5).split(X, y, groups=b["game_id"]):
        P[te] = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=250, min_samples_leaf=40).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
    A = np.eye(5)[y]; rows = []
    for st, g in b.groupby("stadium"):
        m = (b["stadium"] == st).to_numpy(); rows.append({"park": st, "balls": int(m.sum()), "triples": int(A[m, 3].sum()), "triples expected": P[m, 3].sum(), "triple factor": A[m, 3].sum() / P[m, 3].sum(), "HR": int(A[m, 4].sum()), "HR expected": P[m, 4].sum(), "HR factor": A[m, 4].sum() / P[m, 4].sum(), "doubles factor": A[m, 2].sum() / P[m, 2].sum()})
    t = pd.DataFrame(rows).sort_values("triple factor", ascending=False); print("A1. TRIPLES AND HR BY PARK (actual / park-blind expected per ball in play)\n" + t.round(2).to_string(index=False))
    tm = (b["tm"] == 1).to_numpy(); wv = np.array([W[c] for c in CLS]); games = b.loc[tm, "game_id"].nunique(); rows = []
    codes, uniq = pd.factorize(b["game_id"]); rng = np.random.default_rng(3); Wm = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes] for _ in range(400)]
    for k, c in enumerate(CLS):
        diff = A[tm, k].sum() - P[tm, k].sum(); bs = [((w * tm) @ (A[:, k] - P[:, k])) for w in Wm]; rows.append({"outcome": c, "actual": A[tm, k].sum(), "expected": P[tm, k].sum(), "actual - expected": diff, "SE": np.std(bs), "wOBA weight": W[c], "runs-ish: wOBA points (sum over balls)": diff * W[c], "per tracked game": diff / games})
    t = pd.DataFrame(rows); net = t["runs-ish: wOBA points (sum over balls)"].sum(); bsn = [sum(((w * tm) @ (A[:, k] - P[:, k])) * W[c] for k, c in enumerate(CLS)) for w in Wm]
    print("\nA2. TRUSTMARK vs EXPECTED, per outcome, converted to wOBA (sum over all tracked balls in play at Trustmark)\n" + t.round(2).to_string(index=False))
    hr_loss = t.loc[t["outcome"] == "hr", "runs-ish: wOBA points (sum over balls)"].iloc[0]; others = net - hr_loss
    print(f"   net {net:+.1f} wOBA-points over {int(tm.sum()):,} balls = {net / tm.sum():+.4f} wOBA per ball in play (bootstrap SE {np.std(bsn) / tm.sum():.4f}); HR part {hr_loss:+.1f}; everything else {others:+.1f} -> {100 * others / -hr_loss:.0f}% of the HR loss is offset by doubles/triples/singles/outs combined; triples alone {t.loc[t['outcome'] == 'triple', 'runs-ish: wOBA points (sum over balls)'].iloc[0]:+.1f} ({100 * t.loc[t['outcome'] == 'triple', 'runs-ish: wOBA points (sum over balls)'].iloc[0] / -hr_loss:.0f}%)")
    sector = pd.cut(b["direction"], [-90, -30, -10, 10, 30, 90.001], labels=["LF line", "LF gap", "CF", "RF gap", "RF line"], right=False); rows = []
    for s in ["LF line", "LF gap", "CF", "RF gap", "RF line"]:
        m = tm & (sector == s).to_numpy(); o = (~tm) & (sector == s).to_numpy(); rows.append({"sector": s, "TM balls": int(m.sum()), "TM triples": int(A[m, 3].sum()), "expected": P[m, 3].sum(), "factor": A[m, 3].sum() / P[m, 3].sum(), "TM HR factor": A[m, 4].sum() / P[m, 4].sum(), "triple/ball TM": A[m, 3].mean(), "triple/ball other": A[o, 3].mean()})
    print("\nA3. WHERE THE TRIPLES COME FROM (Trustmark, by field sector)\n" + pd.DataFrame(rows).round(3).to_string(index=False))


def part_b():
    # speed proxy: stolen bases per PA from the 2022-24 StatsCrew rows (SB is not in the 2025-26 Trackman rows)
    est, ml = fv.hitter_frame(); kind = est["kind"].astype(str)
    est["o_hr"] = (kind == "home_run").astype(float); est["o_3b"] = (kind == "triple").astype(float); est["o_2b"] = (kind == "double").astype(float); est["o_1b"] = (kind == "single").astype(float); est["o_k"] = (kind == "strikeout").astype(float); est["o_bb"] = (kind == "walk").astype(float)
    pan = pd.read_parquet("output/fl_hitter_panel.parquet"); pan = pan[pan["sb"].notna()].groupby("name_clean").agg(sb=("sb", "sum"), pa=("pa", "sum")); lg = pan["sb"].sum() / pan["pa"].sum(); sp = (pan["sb"] / pan["pa"] - lg) * pan["pa"] / (pan["pa"] + 150); est["spd"] = est["name_clean"].map(sp)
    import os; NOSPEED = os.environ.get("NOSPEED") == "1"
    est = est.dropna(subset=(["woba", "z_air", "z_power"] if NOSPEED else ["spd", "woba", "z_air", "z_power"])).reset_index(drop=True); est["z_spd"] = 0.0 if NOSPEED else (est["spd"] - est["spd"].mean()) / est["spd"].std()
    g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pitcher"])[0], pd.factorize(np.where(est["b_home"] == 1, est["name_clean"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    vis = (est["b_home"] == 0).to_numpy(float); tmv = est["tm"].to_numpy(float) * vis; cl = pd.factorize(est["game_id"])[0]; terms = ["z_air", "z_power"] if NOSPEED else ["z_air", "z_power", "z_spd"]; Xr = partial_out(np.column_stack([tmv * est[t].to_numpy(float) for t in terms]), g, iters=30)
    outs = [("home run", "o_hr", 1.863), ("triple", "o_3b", 1.534), ("double", "o_2b", 1.251), ("single", "o_1b", 0.945), ("walk", "o_bb", 0.82), ("strikeout", "o_k", 0.0), ("wOBA", "woba", None)]; rows = []; contrib = []
    for lab, col, w in outs:
        yr = partial_out(est[col].to_numpy(float)[:, None], g, iters=30)[:, 0]; b, se = cluster_ols(Xr, yr, cl); row = {"outcome (per PA)": lab, "league mean": est[col].mean()}
        for i, t in enumerate(terms):
            row["Air%" if t == "z_air" else ("power" if t == "z_power" else "speed")] = f"{b[i]:+.4f} (t {b[i] / se[i]:+.1f})"
        rows.append(row)
        if w is not None:
            contrib.append({"outcome": lab, "Air% coef": b[0], "x weight": b[0] * w, "power coef": b[1], "power x weight": b[1] * w, "speed coef": b[2] if len(b) > 2 else 0.0, "speed x weight": b[2] * w if len(b) > 2 else 0.0})
    print(f"\nB1. HITTERS (visiting, production FE design; per +1 SD of the profile, per PA), {len(est):,} PA\n" + pd.DataFrame(rows).to_string(index=False))
    c = pd.DataFrame(contrib); tot = c[["x weight", "power x weight", "speed x weight"]].sum(); print("\nB2. WHERE THE wOBA EFFECT OF EACH PROFILE COMES FROM (coefficient x wOBA weight, wOBA points per +1 SD)\n" + c.round(4).to_string(index=False) + f"\n   sum of outcome parts: Air% {tot['x weight']:+.4f}, power {tot['power x weight']:+.4f}, speed {tot['speed x weight']:+.4f}")
    # pitcher side
    ep, prof, stdz = pf.build(); ep = ep.dropna(subset=["woba", "z_hr", "z_k", "z_air"]).reset_index(drop=True); ep["b_home"] = (ep["batter_team"] == ep["home_team"]).astype(int); kind = ep["kind"].astype(str)
    for nm, k in (("o_hr", "home_run"), ("o_3b", "triple"), ("o_2b", "double"), ("o_1b", "single"), ("o_k", "strikeout"), ("o_bb", "walk")):
        ep[nm] = (kind == k).astype(float)
    home_batter = ep["b_home"] == 1; gp = [pd.factorize(ep["year"])[0], pd.factorize(ep["name_clean"])[0], pd.factorize(ep["pid"])[0], pd.factorize(np.where(~home_batter, ep["pid"], "AWAY"))[0], pd.factorize(ep["park"])[0]]; tp = ep["tm"].to_numpy(float) * home_batter.to_numpy(float); clp = pd.factorize(ep["game_id"])[0]
    tt = ["z_hr", "z_k", "z_air"]; Xp = partial_out(np.column_stack([tp * ep[t].to_numpy(float) for t in tt]), gp, iters=30); rows = []; contrib = []
    for lab, col, w in outs:
        yr = partial_out(ep[col].to_numpy(float)[:, None], gp, iters=30)[:, 0]; b, se = cluster_ols(Xp, yr, clp); row = {"outcome (per PA)": lab, "league mean": ep[col].mean()}
        for i, t in enumerate(tt):
            row[{"z_hr": "HR-prone", "z_k": "K%", "z_air": "Air%"}[t]] = f"{b[i]:+.4f} (t {b[i] / se[i]:+.1f})"
        rows.append(row)
        if w is not None:
            contrib.append({"outcome": lab, "Air% coef": b[2], "x weight": b[2] * w})
    c = pd.DataFrame(contrib); print(f"\nB3. PITCHERS (visiting; production profile; outcomes allowed per PA per +1 SD), {len(ep):,} PA\n" + pd.DataFrame(rows).to_string(index=False) + "\n   Air% contribution by outcome (coef x weight):\n" + c.round(4).to_string(index=False) + f"\n   sum {c['x weight'].sum():+.4f}")


if __name__ == "__main__":
    import sys
    if "--b-only" not in sys.argv:
        part_a()
    part_b()
