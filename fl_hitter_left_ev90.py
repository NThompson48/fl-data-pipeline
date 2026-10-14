"""
fl_hitter_left_ev90.py -- does the Trustmark handedness effect change when controlling for POWER measured as EV90?  (user 2026-10-10)
Same design as fl_hitter_profile_screen_bip.py (balls in play, visiting hitters at Trustmark x profile, hitter + pitcher + year + [hitter x home] + park FE, game-cluster SEs).
Left effect (side used in the PA, centred) under control sets: none | EV90 | Air% | EV90 + Air% | HR per ball in play + Air% (the earlier baseline) | EV90 + HR/BIP + Air%; by outcome and season; EV90 and its relation to handedness (hitter level); left x EV90 interaction;
EV90 slopes by side.  EV90 = 90th percentile exit speed on non-Trustmark balls (>= 15 balls), shrunk toward the league by n/(n + 14.26), z-scored.
"""
import warnings
import numpy as np
import pandas as pd
import fl_hitter_profile_screen_bip as sb
from fl_park_profile_fe import cluster_ols, partial_out
warnings.filterwarnings("ignore")

est = sb.build(); non = est[est["tm"] == 0]; g = non.groupby("name_clean")["o_hr"].agg(["mean", "size"]); lg = np.average(g["mean"], weights=g["size"]); d = (g["mean"] - lg) * g["size"] / (g["size"] + 100); v = est["name_clean"].map(d); est["z_hrbip"] = (est["name_clean"].map(d) - v.mean()) / v.std()
ne = est[(est["tm"] == 0) & est["exit_speed"].notna()]; ge = ne.groupby("name_clean")["exit_speed"].agg(["size", lambda s: np.percentile(s, 90) if len(s) >= 15 else np.nan]); ge.columns = ["size", "p90"]; lg9 = ge["p90"].mean(); est["ev90"] = est["name_clean"].map((ge["p90"] - lg9) * ge["size"] / (ge["size"] + 14.26)); est["z_ev90"] = (est["ev90"] - est["ev90"].mean()) / est["ev90"].std()
est = est.dropna(subset=["z_hrbip", "z_ev90"]).reset_index(drop=True); est["left_c"] = est["left"] - est["left"].mean()
h = est.drop_duplicates("name_clean"); print(f"{len(est):,} balls in play, {len(h)} hitters; corr(left share, EV90) at hitter level {est.groupby('name_clean')[['left', 'z_ev90']].mean().corr().iloc[0, 1]:+.2f}; corr(EV90, Air%) {h[['z_ev90', 'z_air']].corr().iloc[0, 1]:+.2f}; corr(EV90, HR per BIP) {h[['z_ev90', 'z_hrbip']].corr().iloc[0, 1]:+.2f}")
print(f"mean EV90 z for left-sided rows {est.loc[est['left'] == 1, 'z_ev90'].mean():+.2f}, right-sided rows {est.loc[est['left'] == 0, 'z_ev90'].mean():+.2f}")
groups = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pitcher"])[0], pd.factorize(np.where(est["b_home"] == 1, est["name_clean"], "AWAY"))[0], pd.factorize(est["park"])[0]]
tmv = est["tm"].to_numpy(float) * (est["b_home"] == 0).to_numpy(float); cl = pd.factorize(est["game_id"])[0]; yrs = est["year"].to_numpy()
Y = {"wOBA on contact": est["y"], "HR": est["o_hr"], "non-HR hit": est["o_hit"]}; Yr = {k: partial_out(v.to_numpy(float)[:, None], groups, iters=30)[:, 0] for k, v in Y.items()}
col = lambda x: partial_out(np.asarray(x, float)[:, None], groups, iters=30)[:, 0]
X = {"left": col(tmv * est["left_c"]), "ev90": col(tmv * est["z_ev90"]), "air": col(tmv * est["z_air"]), "hr": col(tmv * est["z_hrbip"]), "left_x_ev90": col(tmv * est["left_c"] * est["z_ev90"]), "ev90_L": col(tmv * est["z_ev90"] * est["left"]), "ev90_R": col(tmv * est["z_ev90"] * (1 - est["left"]))}
sets = [("none", []), ("EV90", ["ev90"]), ("Air%", ["air"]), ("EV90 + Air%", ["ev90", "air"]), ("HR/BIP + Air% (earlier baseline)", ["hr", "air"]), ("EV90 + HR/BIP + Air%", ["ev90", "hr", "air"])]; rows = []
for lab, cs in sets:
    row = {"controls": lab}
    for k in Yr:
        b, se = cluster_ols(np.column_stack([X["left"]] + [X[c] for c in cs]), Yr[k], cl); row[f"left: {k}"] = f"{b[0]:+.4f} (t {b[0] / se[0]:+.1f})"
    if "ev90" in cs:
        b, se = cluster_ols(np.column_stack([X["left"]] + [X[c] for c in cs]), Yr["wOBA on contact"], cl); row["EV90 coef (wOBAcon)"] = f"{b[1 + cs.index('ev90')]:+.4f} (t {b[1 + cs.index('ev90')] / se[1 + cs.index('ev90')]:+.1f})"
    rows.append(row)
print("\nLEFT-HANDED EFFECT (visiting hitters at Trustmark, per ball in play) UNDER DIFFERENT CONTROLS\n" + pd.DataFrame(rows).fillna("").to_string(index=False))
cols = [col(tmv * (yrs == y) * est["left_c"]) for y in (2025, 2026)] + [X["ev90"], X["air"]]; b, se = cluster_ols(np.column_stack(cols), Yr["wOBA on contact"], cl); print(f"\nby season, EV90 + Air% controlled: left 2025 {b[0]:+.4f} ({se[0]:.4f}), 2026 {b[1]:+.4f} ({se[1]:.4f}); difference t {(b[0] - b[1]) / np.hypot(se[0], se[1]):+.2f}")
b, se = cluster_ols(np.column_stack([X["left"], X["ev90_L"], X["ev90_R"], X["air"]]), Yr["wOBA on contact"], cl); print(f"EV90 slope by side (with left, Air%): LHH {b[1]:+.4f} (t {b[1] / se[1]:+.1f}), RHH {b[2]:+.4f} (t {b[2] / se[2]:+.1f}); left {b[0]:+.4f} (t {b[0] / se[0]:+.1f})")
b, se = cluster_ols(np.column_stack([X["left"], X["ev90"], X["left_x_ev90"], X["air"]]), Yr["wOBA on contact"], cl); print(f"left x EV90 interaction (with EV90, Air%): {b[2]:+.4f} (t {b[2] / se[2]:+.1f}); left {b[0]:+.4f} (t {b[0] / se[0]:+.1f}); EV90 {b[1]:+.4f} (t {b[1] / se[1]:+.1f})")
