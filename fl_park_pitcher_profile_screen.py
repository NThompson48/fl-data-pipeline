"""
fl_park_pitcher_profile_screen.py -- CONTROLLING FOR K% AND BB%, does any pitcher profile do better (or worse) at Trustmark?  (user 2026-10-09)
Design = the visitors-only pitcher model of fl_park_final_visitors.py: wOBA allowed per PA, year + hitter + pitcher + [pitcher x home] + park fixed effects, game-cluster SEs; the interaction applies to VISITING pitchers at Trustmark
(home pitchers' own results are absorbed by their home effect and carry roster fit, not park).
Each candidate profile is measured on the pitcher's NON-Trustmark data, shrunk toward the league mean (n0 below), and standardised (z) on the PA-weighted estimation pool:
  controls      K% allowed (n0 113 PA), BB% allowed (n0 120 PA)                       [both always in the model]
  batted balls  ground-ball share, line-drive share, fly-ball share, pop-up share of balls in play (n0 89 balls); mean exit speed allowed (n0 60 balls)
  HR            HR/PA allowed (n0 137 PA)                                                [overlaps K and BB]
  stuff         fastball velocity (n0 100 FB), fastball induced vertical break, release height, extension, spin rate (n0 100 pitches)
  repertoire    fastball share (four-seam + sinker + cutter), breaking share (slider, sweeper, curve), offspeed share (change, splitter), n0 100 pitches
  hand          left-handed
For each candidate: coefficient on Trustmark x z (wOBA per +1 SD, positive = WORSE for the pitcher; x 39.9 = runs/9 at full exposure), the same WITHOUT the K/BB controls, and the correlation of the candidate with K% and BB%.
Then a joint model (controls + all candidates) with a game-cluster bootstrap, and a count of how many |t| > 2 results to expect by chance among the candidates (multiplicity).
"""
import warnings

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_park_pitcher_final as pf
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
TO_R9 = pf.TO_R9


def shrunk_z(x, n, n0, est, pid_col="pid"):
    """x: Series by pid of a raw mean, n: Series by pid of its sample size; shrink toward the n-weighted league mean, z-score on the estimation PA pool."""
    ok = x.notna() & n.notna() & (n > 0); x, n = x[ok], n[ok]; lg = np.average(x, weights=n); d = (x - lg) * n / (n + n0); v = est[pid_col].map(d); mu, sd = v.mean(), v.std()
    return ((d - mu) / sd)


def build_candidates(est):
    cands = {}; pa_non = est[est["tm"] == 0]
    kind = pa_non["kind"].astype(str); g = pa_non.assign(k=(kind == "strikeout").astype(float), bb=kind.isin(["walk"]).astype(float)).groupby("pid").agg(n=("woba", "size"), k=("k", "mean"), bb=("bb", "mean"), hr=("is_hr", "mean"))
    cands["K% allowed"] = shrunk_z(g["k"], g["n"], 113, est); cands["BB% allowed"] = shrunk_z(g["bb"], g["n"], 120, est); cands["HR/PA allowed"] = shrunk_z(g["hr"], g["n"], 137, est)
    bb = pf.pitcher_bbe(); bb["pid"] = bb["pitcher_id"].astype(str); bb["park"] = bb["game_id"].str.split("-").str[1]; bb = bb[bb["park"] != "TrustmarkPark"]
    for lab, lo, hi in (("ground-ball share", -91, 10), ("line-drive share", 10, 25), ("fly-ball share", 25, 40), ("pop-up share", 40, 91)):
        s = ((bb["angle"] >= lo) & (bb["angle"] < hi)).astype(float).groupby(bb["pid"]); cands[lab] = shrunk_z(s.mean(), s.size(), 89, est)
    pit = pd.read_parquet("2025_2026_pitches.parquet", columns=["pitcher_id", "stadium", "exit_speed", "is_terminal_pitch"]); pit = pit[(pit["stadium"] != "TrustmarkPark") & pit["exit_speed"].notna()]; pit["pid"] = pit["pitcher_id"].astype(str)
    s = pit.groupby("pid")["exit_speed"]; cands["exit speed allowed"] = shrunk_z(s.mean(), s.size(), 60, est)
    raw = fp.load_fl_pitches_cached(); raw = raw[raw["pitcher_id"].notna()].copy(); raw["pid"] = raw["pitcher_id"].astype(str); raw["park"] = raw["game_id"].str.split("-").str[1]; raw = raw[(raw["park"] != "TrustmarkPark") & raw["pitch_type"].isin(["four_seam", "sinker", "cutter", "slider", "sweeper", "curveball", "changeup", "splitter"])]
    fb = raw[raw["pitch_type"].isin(["four_seam", "sinker", "cutter"])]
    for lab, col in (("fastball velocity", "effective_velo"), ("fastball induced vert. break", "ivb")):
        s = fb.groupby("pid")[col]; cands[lab] = shrunk_z(s.mean(), s.size(), 100, est)
    for lab, col in (("release height", "release_pos_z"), ("extension", "extension"), ("spin rate", "spin_rate")):
        s = raw.groupby("pid")[col]; cands[lab] = shrunk_z(s.mean(), s.size(), 100, est)
    fam = np.select([raw["pitch_type"].isin(["four_seam", "sinker", "cutter"]), raw["pitch_type"].isin(["slider", "sweeper", "curveball"])], ["fb", "brk"], "off")
    for lab, key in (("fastball share", "fb"), ("breaking-ball share", "brk"), ("offspeed share", "off")):
        s = (fam == key).astype(float); sg = pd.Series(s).groupby(raw["pid"].to_numpy()); cands[lab] = shrunk_z(sg.mean(), sg.size(), 100, est)
    lm = est.drop_duplicates("pid").set_index("pid")["lefty"]; cands["left-handed"] = (lm - lm.mean()) / lm.std()
    return pd.DataFrame(cands)


def main():
    est, prof, stdz = pf.build(); est["b_home"] = (est["batter_team"] == est["home_team"]).astype(int); C = build_candidates(est); names = list(C.columns)
    for c in names:
        est[c] = est["pid"].map(C[c])
    est = est.dropna(subset=["K% allowed", "BB% allowed"]).reset_index(drop=True)
    for c in names:
        est[c] = est[c].fillna(0.0)                                                      # missing stuff data -> league average (z = 0)
    print(f"estimation sample {len(est):,} PA, {est['pid'].nunique()} pitchers; visiting-pitcher PA at Trustmark {int(((est['tm'] == 1) & (est['b_home'] == 1)).sum()):,}")
    print("correlation of candidates with K% allowed / BB% allowed (pitcher level):\n" + C.corr()[["K% allowed", "BB% allowed"]].round(2).T.to_string())
    home_batter = (est["b_home"] == 1); g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pid"])[0], pd.factorize(np.where(~home_batter, est["pid"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    vis = home_batter.to_numpy(float); tmv = est["tm"].to_numpy(float) * vis; y = est["woba"].to_numpy(float); yr = partial_out(y[:, None], g, iters=30)[:, 0]; cl = pd.factorize(est["game_id"])[0]
    X = np.column_stack([tmv * est[c] for c in names]); Xr = partial_out(X, g, iters=30); ix = {c: i for i, c in enumerate(names)}
    ctrl = [ix["K% allowed"], ix["BB% allowed"]]; b0, se0 = cluster_ols(Xr[:, ctrl], yr, cl)
    print(f"\nCONTROLS ONLY: Trustmark x K% allowed {b0[0]:+.4f} ({se0[0]:.4f}, t {b0[0] / se0[0]:+.1f}); x BB% allowed {b0[1]:+.4f} ({se0[1]:.4f}, t {b0[1] / se0[1]:+.1f})   [wOBA per +1 SD, positive = worse for the pitcher]")
    rows = []
    for c in names:
        if c in ("K% allowed", "BB% allowed"):
            continue
        cols = ctrl + [ix[c]]; b, se = cluster_ols(Xr[:, cols], yr, cl); b1, se1 = cluster_ols(Xr[:, [ix[c]]], yr, cl)
        rows.append({"profile (per +1 SD)": c, "coef | K,BB controlled": b[2], "SE": se[2], "t": b[2] / se[2], "runs/9": b[2] * TO_R9, "coef | no controls": b1[0], "t no ctrl": b1[0] / se1[0], "corr with K%": C[c].corr(C["K% allowed"]), "corr with BB%": C[c].corr(C["BB% allowed"])})
    t = pd.DataFrame(rows).sort_values("t", key=lambda s: -s.abs()); print("\nONE AT A TIME (K% and BB% allowed always in the model)\n" + t.round(4).to_string(index=False))
    m = len(t); print(f"\n{m} candidates screened; expected number with |t| > 2 by chance alone: {0.0455 * m:.1f}; observed {(t['t'].abs() > 2).sum()}")
    # joint model with all candidates + game-cluster bootstrap
    allc = list(range(len(names))); bj, sej = cluster_ols(Xr[:, allc], yr, cl); rng = np.random.default_rng(8); ug = np.unique(cl); by = pd.Series(np.arange(len(cl))).groupby(cl).apply(np.array); bs = []
    for _ in range(300):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); bs.append(np.linalg.lstsq(Xr[ii], yr[ii], rcond=None)[0])
    bs = np.array(bs); jt = pd.DataFrame({"profile (per +1 SD)": names, "joint coef": bj, "cluster SE": sej, "bootstrap SE": bs.std(axis=0), "t (bootstrap)": bj / bs.std(axis=0), "sign stability": [(np.sign(bs[:, i]) == np.sign(bj[i])).mean() for i in range(len(names))]})
    print("\nJOINT MODEL (all profiles together; with this many correlated terms expect noisier estimates)\n" + jt.sort_values("t (bootstrap)", key=lambda s: -s.abs()).round(4).to_string(index=False))
    t.to_csv("output/pitcher_profile_screen.csv", index=False); jt.to_csv("output/pitcher_profile_screen_joint.csv", index=False)


if __name__ == "__main__":
    main()
