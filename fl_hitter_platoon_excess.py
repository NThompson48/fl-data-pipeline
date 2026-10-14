"""
fl_hitter_platoon_excess.py -- the Trustmark platoon EXCESS: same-hand matchup x Trustmark, WITH the ordinary same-hand effect (all parks) in the model.  (corrects fl_hitter_profile_screen_bip.py, which had no main platoon term)
Design: wOBA on contact (and HR, non-HR hit) per ball in play, hitter + pitcher + year + [hitter x home] + park FE; regressors: same_hand (all rows) and Trustmark(visiting) x same_hand; baseline Air% + power terms kept.
By season, out-of-season SSE gain over the baseline (fit one season, predict the other's Trustmark visiting balls), game-cluster bootstrap.
"""
import warnings
import numpy as np
import pandas as pd
import fl_hitter_profile_screen_bip as sb
from fl_park_profile_fe import cluster_ols, partial_out
warnings.filterwarnings("ignore")

est = sb.build(); non = est[est["tm"] == 0]; g = non.groupby("name_clean")["o_hr"].agg(["mean", "size"]); lg = np.average(g["mean"], weights=g["size"]); d = (g["mean"] - lg) * g["size"] / (g["size"] + 100); v = est["name_clean"].map(d); est["z_hrbip"] = (est["name_clean"].map(d) - v.mean()) / v.std()
est = est.dropna(subset=["z_hrbip", "same_hand"]).reset_index(drop=True); est["sh_c"] = est["same_hand"] - est["same_hand"].mean()
groups = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pitcher"])[0], pd.factorize(np.where(est["b_home"] == 1, est["name_clean"], "AWAY"))[0], pd.factorize(est["park"])[0]]
tmv = est["tm"].to_numpy(float) * (est["b_home"] == 0).to_numpy(float); on = tmv == 1; cl = pd.factorize(est["game_id"])[0]; yrs = est["year"].to_numpy()
Y = {"wOBAcon": est["y"], "HR": est["o_hr"], "non-HR hit": est["o_hit"]}; Yr = {k: partial_out(v.to_numpy(float)[:, None], groups, iters=30)[:, 0] for k, v in Y.items()}
col = lambda x: partial_out(np.asarray(x, float)[:, None], groups, iters=30)[:, 0]
air, pw, main_sh, tm_sh = col(tmv * est["z_air"]), col(tmv * est["z_hrbip"]), col(est["sh_c"]), col(tmv * est["sh_c"])
print(f"{len(est):,} balls in play; visiting balls at Trustmark {int(on.sum()):,}; same-hand share {est['same_hand'].mean():.3f}")
rows = []
for k in Yr:
    b, se = cluster_ols(np.column_stack([air, pw, main_sh, tm_sh]), Yr[k], cl); rows.append({"outcome": k, "ordinary same-hand effect (all parks)": f"{b[2]:+.4f} (t {b[2] / se[2]:+.1f})", "Trustmark EXCESS": f"{b[3]:+.4f} (t {b[3] / se[3]:+.1f})", "Air%": f"{b[0]:+.4f}", "power": f"{b[1]:+.4f}"})
print(pd.DataFrame(rows).to_string(index=False))
cols = [air, pw, main_sh] + [col(tmv * (yrs == y) * est["sh_c"]) for y in (2025, 2026)]; b, se = cluster_ols(np.column_stack(cols), Yr["wOBAcon"], cl); print(f"by season (wOBAcon): Trustmark excess 2025 {b[3]:+.4f} ({se[3]:.4f}), 2026 {b[4]:+.4f} ({se[4]:.4f}); difference t {(b[3] - b[4]) / np.hypot(se[3], se[4]):+.2f}")
rng = np.random.default_rng(4); X0 = np.column_stack([air, pw, main_sh]); X1 = np.column_stack([air, pw, main_sh, tm_sh]); yv = Yr["wOBAcon"]
for a, t in ((2025, 2026), (2026, 2025)):
    tr, te = yrs == a, yrs == t; b0 = np.linalg.lstsq(X0[tr], yv[tr], rcond=None)[0]; b1 = np.linalg.lstsq(X1[tr], yv[tr], rcond=None)[0]; m = te & on; e0 = yv[m] - X0[m] @ b0; e1 = yv[m] - X1[m] @ b1
    gc = pd.factorize(est["game_id"][m])[0]; by = pd.Series(np.arange(m.sum())).groupby(gc).apply(np.array); ug = by.index.to_numpy(); better = 0
    for _ in range(500):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); better += (e1[ii] ** 2).sum() < (e0[ii] ** 2).sum()
    print(f"fit {a} -> test {t}: SSE gain {1 - (e1 ** 2).sum() / (e0 ** 2).sum():+.4%}, P(better) {better / 500:.2f}")

# --- does the LEFT-HANDED main effect survive once the same-hand excess is in?  (right-handers face same-hand pitchers ~71% of the time, left-handers ~29%)
est["left_c2"] = est["left"] - est["left"].mean(); tm_left = col(tmv * est["left_c2"])
print(f"\nsame-hand share of balls: RHH {est.loc[est['left'] == 0, 'same_hand'].mean():.3f}, LHH {est.loc[est['left'] == 1, 'same_hand'].mean():.3f}")
for lab, cols_ in (("left alone (+Air%, power)", [air, pw, tm_left]), ("same-hand excess alone (+Air%, power, ordinary platoon)", [air, pw, main_sh, tm_sh]), ("BOTH", [air, pw, main_sh, tm_sh, tm_left])):
    for k in ("wOBAcon", "HR", "non-HR hit"):
        b, se = cluster_ols(np.column_stack(cols_), Yr[k], cl); n = len(cols_)
        if k == "wOBAcon":
            print(f"  {lab:58s} {k}: " + (f"left {b[-1]:+.4f} (t {b[-1] / se[-1]:+.1f})" if "left" in lab or lab == "BOTH" else "") + (f"  same-hand excess {b[3]:+.4f} (t {b[3] / se[3]:+.1f})" if "same-hand" in lab or lab == "BOTH" else ""))
