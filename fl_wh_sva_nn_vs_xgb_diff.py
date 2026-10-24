"""
fl_wh_sva_nn_vs_xgb_diff.py -- where do the neural-network and XGBoost stuff estimators disagree most, and who is right when they do?  (user 2026-10-23: "the biggest differences in predictions and pitchers between the network and XGBoost")
Both are fp.compute_pooled_wh_sva (same 9 inputs, same pitcher-held-out 5-fold cross-fit): 'xgb' = production XGBoost, 'torch3' = the 3-seed PyTorch network.  Sections:
 A. PER PITCH on the production pool (2025 + 2026 full season): correlation, mean difference and spread by pitch type; the pitches with the biggest disagreement and what is unusual about them.
 B. PER PITCHER on the same pool (>= 200 swings): z-scored stuff levels, rank differences; the pitchers the network likes (higher than XGBoost) and dislikes, with pitch mix and shape; the implied K% difference
    (K% coefficient x level, coefficients from the training-transition fits: XGBoost .539, network .490).
 C. WHO IS RIGHT, on the fold-2 windows: stuff from 2025 + 1H2026 (both estimators), outcome = the change in whiff rate per swing from that window to 2H2026 ('surprise' = 2H whiff% - prior whiff%).  Pitchers where the estimators
    disagree by >= 1 SD (z difference) are split by direction; if the network is right, the 'network likes' group should rise and the 'network dislikes' group fall, relative to what XGBoost expected.  Pitcher bootstrap CIs.
Output: output/wh_sva_nn_vs_xgb_diff.json, output/wh_sva_nn_vs_xgb_pitchers.png
"""
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf

warnings.filterwarnings("ignore")
KEYS = ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]; COEF = {"xgb": 0.53916, "nn": 0.48988}; rng = np.random.default_rng(3)


def both(window):
    x = fp.compute_pooled_wh_sva(window, estimator="xgb").rename(columns={"sva": "xgb"}); n = fp.compute_pooled_wh_sva(window, estimator="torch3").rename(columns={"sva": "nn"})
    m = x.merge(n[KEYS + ["nn"]], on=KEYS); a = window.drop_duplicates(KEYS)[KEYS + ["pitcher", "pitcher_team", "pitch_type", "effective_velo", "ivb", "horz_break_as", "spin_rate", "extension", "release_pos_z", "plate_x_as", "plate_z", "is_whiff"]]
    return m.merge(a, on=KEYS, how="left")


def pitcher_table(m, min_sw=200):
    g = m.groupby("pitcher_id").agg(pitcher=("pitcher", "first"), team=("pitcher_team", "first"), sw=("xgb", "size"), xgb=("xgb", "mean"), nn=("nn", "mean"), whiff=("is_whiff", "mean"), velo=("effective_velo", "mean"), spin=("spin_rate", "mean"), ivb=("ivb", "mean"),
                                       ext=("extension", "mean"));
    g = g[g["sw"] >= min_sw].copy(); g["z_xgb"] = (g["xgb"] - g["xgb"].mean()) / g["xgb"].std(); g["z_nn"] = (g["nn"] - g["nn"].mean()) / g["nn"].std(); g["dz"] = g["z_nn"] - g["z_xgb"]
    g["rank_xgb"] = g["xgb"].rank(ascending=False); g["rank_nn"] = g["nn"].rank(ascending=False); g["k_pts"] = 100 * (COEF["nn"] * (g["nn"] - g["nn"].mean()) - COEF["xgb"] * (g["xgb"] - g["xgb"].mean()))
    mix = m.groupby(["pitcher_id", "pitch_type"]).size().unstack(fill_value=0); mix = mix.div(mix.sum(axis=1), axis=0)
    g["mix"] = [", ".join(f"{t} {100 * s:.0f}%" for t, s in mix.loc[i].sort_values(ascending=False).head(3).items()) if i in mix.index else "" for i in g.index]; return g


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025), fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    M = both(pool); out = {}
    print(f"A. PER PITCH: {len(M):,} swings; correlation of the two per-pitch values {np.corrcoef(M.xgb, M.nn)[0, 1]:.3f}; SD xgb {M.xgb.std():.4f} nn {M.nn.std():.4f}; mean |difference| {np.abs(M.nn - M.xgb).mean():.4f}")
    T = M.assign(diff=M.nn - M.xgb).groupby("pitch_type").agg(swings=("xgb", "size"), mean_xgb=("xgb", "mean"), mean_nn=("nn", "mean"), mean_diff=("diff", "mean"), sd_xgb=("xgb", "std"), sd_nn=("nn", "std"), mean_abs_diff=("diff", lambda s: s.abs().mean())); T["corr"] = M.groupby("pitch_type").apply(lambda q: np.corrcoef(q.xgb, q.nn)[0, 1])
    T = T[T.swings >= 1000].sort_values("swings", ascending=False); pd.set_option("display.width", 250); print(T.round(4).to_string()); out["by_pitch_type"] = T.reset_index().to_dict("records")
    M["adiff"] = (M.nn - M.xgb).abs(); top = M.sort_values("adiff", ascending=False).head(12)
    print("\nthe 12 pitches the two disagree on most (sva = whiff probability above what location alone predicts):\n" + top[["pitcher", "pitch_type", "effective_velo", "ivb", "horz_break_as", "spin_rate", "extension", "release_pos_z", "plate_x_as", "plate_z", "is_whiff", "xgb", "nn"]].round(2).to_string(index=False))
    q = M.assign(extreme=(M.effective_velo < M.effective_velo.quantile(.01)) | (M.effective_velo > M.effective_velo.quantile(.99)) | (M.spin_rate < M.spin_rate.quantile(.01)) | (M.spin_rate > M.spin_rate.quantile(.99)) | (M.ivb.abs() > M.ivb.abs().quantile(.99)))
    print(f"\nmean |difference| on ordinary pitches {q[~q.extreme].adiff.mean():.4f} vs on extreme-input pitches (top / bottom 1% of velo, spin, |ivb|) {q[q.extreme].adiff.mean():.4f} (n {int(q.extreme.sum()):,})")
    P = pitcher_table(M); print(f"\nB. PER PITCHER ({len(P)} pitchers with >= 200 swings): correlation of the two levels {np.corrcoef(P.xgb, P.nn)[0, 1]:.3f}; rank correlation {P[['rank_xgb', 'rank_nn']].corr(method='spearman').iloc[0, 1]:.3f}; "
          f"|z difference| >= 1 for {int((P.dz.abs() >= 1).sum())} pitchers, >= 1.5 for {int((P.dz.abs() >= 1.5).sum())}; implied K% difference SD {P.k_pts.std():.2f} points, max |difference| {P.k_pts.abs().max():.2f}")
    cols = ["pitcher", "team", "sw", "z_xgb", "z_nn", "dz", "rank_xgb", "rank_nn", "k_pts", "whiff", "velo", "spin", "ivb", "mix"]
    like = P.sort_values("dz", ascending=False).head(12)[cols]; dislike = P.sort_values("dz").head(12)[cols]
    print("\nPITCHERS THE NETWORK LIKES MORE THAN XGBoost (z = stuff level in SDs across pitchers; k_pts = implied K% difference, network minus XGBoost, in points):\n" + like.round(2).to_string(index=False))
    print("\nPITCHERS THE NETWORK LIKES LESS THAN XGBoost:\n" + dislike.round(2).to_string(index=False)); out["likes"] = like.to_dict("records"); out["dislikes"] = dislike.to_dict("records")
    f2 = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025), fp.filter_period(raw, bt.START_2026, bt.CUTOFF_2026)], ignore_index=True); M2 = both(f2); P2 = pitcher_table(M2)
    h2 = fp.filter_period(raw, bt._DAY_AFTER_2026_CUT, bt.END_2026); h2 = h2[h2["is_swing"] & h2["is_whiff"].notna()].groupby("pitcher_id").agg(n2=("is_whiff", "size"), y2=("is_whiff", "mean")); C = P2.join(h2, how="inner"); C = C[C.n2 >= 60].copy(); C["surprise"] = C.y2 - C.whiff
    def ci(v):
        b = [rng.choice(v, len(v)).mean() for _ in range(2000)]; return float(np.mean(v)), float(np.percentile(b, 5)), float(np.percentile(b, 95))
    print(f"\nC. WHO IS RIGHT (fold-2 windows: stuff from 2025 + 1H2026, outcome = 2H2026 whiff/swing minus the prior whiff/swing; {len(C)} pitchers):")
    rows = []
    for lab, m in (("network likes more (z diff >= +1)", C.dz >= 1), ("network likes less (z diff <= -1)", C.dz <= -1), ("agree (|z diff| < 1)", C.dz.abs() < 1)):
        v = C.loc[m, "surprise"].to_numpy(); mu, lo, hi = ci(v); rows.append({"group": lab, "n": int(m.sum()), "mean z diff": float(C.loc[m, "dz"].mean()), "mean surprise (whiff points)": 100 * mu, "90% low": 100 * lo, "90% high": 100 * hi})
    a = C.loc[C.dz >= 1, "surprise"].to_numpy(); b = C.loc[C.dz <= -1, "surprise"].to_numpy(); d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)]
    print(pd.DataFrame(rows).round(2).to_string(index=False)); print(f"difference (likes more - likes less): {100 * (a.mean() - b.mean()):+.2f} whiff points, 90% [{100 * np.percentile(d, 5):+.2f}, {100 * np.percentile(d, 95):+.2f}]; P(network right direction) {np.mean(np.array(d) > 0):.2f}")
    r = {"xgb": np.corrcoef(C.xgb, C.surprise)[0, 1], "nn": np.corrcoef(C.nn, C.surprise)[0, 1]}; print(f"correlation of each stuff level with the surprise: XGBoost {r['xgb']:+.3f}, network {r['nn']:+.3f}")
    # partial: surprise ~ z_nn + z_xgb
    X = np.column_stack([np.ones(len(C)), C.z_xgb, C.z_nn]); beta = np.linalg.lstsq(X, C.surprise.to_numpy(), rcond=None)[0]; print(f"joint regression of the surprise on both z-scores (whiff points per SD): XGBoost {100 * beta[1]:+.2f}, network {100 * beta[2]:+.2f}")
    out["who_is_right"] = rows; out["who_is_right_diff"] = {"diff_points": 100 * (a.mean() - b.mean()), "p_network_direction": float(np.mean(np.array(d) > 0))}; out["corr_with_surprise"] = r
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"; fig, ax = plt.subplots(figsize=(8.2, 7.2), facecolor=surf); ax.set_facecolor(surf)
    ax.scatter(P.z_xgb, P.z_nn, s=14, color="#9a998f", alpha=.6, linewidths=0, label="all pitchers"); lim = [min(P.z_xgb.min(), P.z_nn.min()) - .3, max(P.z_xgb.max(), P.z_nn.max()) + .3]; ax.plot(lim, lim, color=ink2, lw=1, ls=":")
    up, dn = P.sort_values("dz", ascending=False).head(8), P.sort_values("dz").head(8)
    ax.scatter(up.z_xgb, up.z_nn, s=46, color="#2a78d6", edgecolor=surf, linewidths=1.5, zorder=4, label="network likes more"); ax.scatter(dn.z_xgb, dn.z_nn, s=46, color="#eb6834", edgecolor=surf, linewidths=1.5, zorder=4, label="network likes less")
    for t, c in ((up, "#2a78d6"), (dn, "#eb6834")):
        for _, r_ in t.head(6).iterrows():
            ax.annotate(r_.pitcher.split(", ")[0], (r_.z_xgb, r_.z_nn), (5, 4), textcoords="offset points", fontsize=8, color=ink)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("XGBoost stuff level (SDs above the average pitcher)", color=ink2); ax.set_ylabel("Network stuff level (SDs above the average pitcher)", color=ink2); ax.grid(color=grid, lw=.8); ax.set_axisbelow(True)
    [ax.spines[k].set_visible(False) for k in ("top", "right")]; ax.tick_params(colors=ink2); ax.legend(frameon=False, loc="upper left", labelcolor=ink, fontsize=9); ax.set_title(f"Stuff level by pitcher: network vs XGBoost ({len(P)} pitchers, 200+ swings, r = {np.corrcoef(P.xgb, P.nn)[0, 1]:.2f})", color=ink, loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig("output/wh_sva_nn_vs_xgb_pitchers.png", dpi=150, facecolor=surf); plt.close(fig)
    json.dump(out, open("output/wh_sva_nn_vs_xgb_diff.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
