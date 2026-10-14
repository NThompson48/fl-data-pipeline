"""
fl_pitcher_fbshare_sticky.py -- (user 2026-10-09) "is flyball share alone better than flyball+line drive share (Air%)?  Is flyball share sticky at all, or predictable from pitcher traits?"
PART 1  WHICH DEFINITION works best at Trustmark (visiting pitchers, K% and BB% controlled, same FE design as fl_park_pitcher_gbfb_deep.py):
        FB (25-40 deg) | FB+LD (10-40) | FB incl. pop-ups (>= 25) | Air% (>= 10 = 1 - GB) | GB.   In-sample coefficient / t, and the stronger test: fit on ONE season, predict the OTHER season's visiting-pitcher PA at Trustmark
        (both directions), improvement in squared error over the K%+BB%-only model, game-cluster bootstrap P(better).
PART 2  STICKINESS of batted-ball shares and K% / BB%: split-half reliability (random halves of a pitcher-season's balls / PA, Spearman-Brown), implied n0 (= mean n x (1 - r)/r), and 2025 -> 2026 correlation among pitchers with >= 60 balls in both years.
PART 3  PREDICTABILITY of FB share from pitcher TRAITS: fastball velocity, induced vertical break, vertical approach angle, release height, extension, spin, mean pitch height, pitch mix, handedness.
        (a) same season, cross-validated by pitcher (5-fold ridge) pooled over both seasons;  (b) forward 2025 -> 2026: league mean | last year's FB share (shrunk, n0 chosen on the data) | last year's traits | both.
        Weighted RMSE and MAE with inverse-variance weights (n = balls in the target season, law fitted on the league-mean model, frozen), pitcher-cluster bootstrap vs the league-mean model.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

import fl_eval_report as er
import fl_eval_weights as ew
import fl_pitcher_features as fp
import fl_park_pitcher_final as pf
import fl_park_pitcher_profile_screen as sc
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
DEFS = {"FB (25-40)": (25, 40), "FB+LD (10-40)": (10, 40), "FB incl. pop-ups (>=25)": (25, 91), "Air% (>=10)": (10, 91), "GB (<10)": (-91, 10), "LD (10-25)": (10, 25), "pop-up (>40)": (40, 91)}


def share_z(bb, lo, hi, est, n0=89):
    s = ((bb["angle"] >= lo) & (bb["angle"] < hi)).astype(float).groupby(bb["pid"]); m, n = s.mean(), s.size(); lg = np.average(m, weights=n); d = (m - lg) * n / (n + n0); v = est["pid"].map(d); return (d - v.mean()) / v.std()


def part1():
    est, prof, stdz = pf.build(); est["b_home"] = (est["batter_team"] == est["home_team"]).astype(int); C = sc.build_candidates(est)
    bb = pf.pitcher_bbe(); bb["pid"] = bb["pitcher_id"].astype(str); bb["park"] = bb["game_id"].str.split("-").str[1]; bb = bb[bb["park"] != "TrustmarkPark"]
    Z = pd.DataFrame({k: share_z(bb, lo, hi, est) for k, (lo, hi) in DEFS.items()}); Z["K"] = C["K% allowed"]; Z["BB"] = C["BB% allowed"]
    for c in Z.columns:
        est["z_" + c] = est["pid"].map(Z[c])
    est = est.dropna(subset=["z_K", "z_BB", "z_GB (<10)"]).reset_index(drop=True)
    for c in Z.columns:
        est["z_" + c] = est["z_" + c].fillna(0.0)
    home_batter = (est["b_home"] == 1); g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pid"])[0], pd.factorize(np.where(~home_batter, est["pid"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    tmv = est["tm"].to_numpy(float) * home_batter.to_numpy(float); y = est["woba"].to_numpy(float); yr = partial_out(y[:, None], g, iters=30)[:, 0]; cl = pd.factorize(est["game_id"])[0]; yrs = est["year"].to_numpy()
    terms = ["K", "BB"] + list(DEFS); Xr = {t: partial_out((tmv * est["z_" + t].to_numpy())[:, None], g, iters=30)[:, 0] for t in terms}
    print(f"PART 1.  visiting-pitcher PA at Trustmark: {int(tmv.sum()):,} (2025 {int((tmv * (yrs == 2025)).sum()):,}, 2026 {int((tmv * (yrs == 2026)).sum()):,})")
    rows = []; on = tmv == 1; rng = np.random.default_rng(12); gm = {}
    for t in DEFS:
        X = np.column_stack([Xr["K"], Xr["BB"], Xr[t]]); b, se = cluster_ols(X, yr, cl); row = {"definition": t, "coef per +1 SD": b[2], "SE": se[2], "t": b[2] / se[2], "runs/9": b[2] * pf.TO_R9}
        for a, c_ in ((2025, 2026), (2026, 2025)):
            tr, te = (yrs == a), (yrs == c_); Xb = np.column_stack([Xr["K"], Xr["BB"]]); b0 = np.linalg.lstsq(Xb[tr], yr[tr], rcond=None)[0]; b1 = np.linalg.lstsq(X[tr], yr[tr], rcond=None)[0]
            m = te & on; e0 = yr[m] - Xb[m] @ b0; e1 = yr[m] - X[m] @ b1; row[f"fit {a} -> test {c_}: SSE gain"] = 1 - (e1 ** 2).sum() / (e0 ** 2).sum()
            gcodes = pd.factorize(est["game_id"][m])[0]; by = pd.Series(np.arange(m.sum())).groupby(gcodes).apply(np.array); ug = by.index.to_numpy(); better = 0
            for _ in range(500):
                ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); better += (e1[ii] ** 2).sum() < (e0[ii] ** 2).sum()
            row[f"P(better) {a}->{c_}"] = better / 500
        rows.append(row)
    print("coefficient: wOBA allowed at Trustmark per +1 SD with K% and BB% controlled; SSE gain = share of squared error removed vs the K%+BB%-only model when the coefficient is fitted on one season and applied to the other season's Trustmark PA\n" + pd.DataFrame(rows).round(4).to_string(index=False))


def pitcher_balls():
    bb = pf.pitcher_bbe(); bb["pid"] = bb["pitcher_id"].astype(str); return bb[["pid", "year", "angle"]].dropna()


def part2():
    bb = pitcher_balls(); est, _, _ = pf.build(); rng = np.random.default_rng(3); rows = []
    indic = {k: ((lo, hi)) for k, (lo, hi) in DEFS.items()}
    # split-half reliability
    for k, (lo, hi) in indic.items():
        b = bb.copy(); b["x"] = ((b["angle"] >= lo) & (b["angle"] < hi)).astype(float); b["half"] = rng.integers(0, 2, len(b)); g = b.groupby(["pid", "year", "half"])["x"].agg(["mean", "size"]).unstack("half").dropna(); g = g[(g[("size", 0)] >= 20) & (g[("size", 1)] >= 20)]
        r = np.corrcoef(g[("mean", 0)], g[("mean", 1)])[0, 1]; rf = 2 * r / (1 + r); nbar = (g[("size", 0)] + g[("size", 1)]).mean(); rows.append({"measure": k, "pitcher-seasons": len(g), "mean balls": nbar, "split-half r (full-length)": rf, "implied n0 (balls)": nbar * (1 - rf) / rf})
    est["is_k"] = est["is_k"]; est["bbf"] = (est["kind"].astype(str) == "walk").astype(float)
    for lab, col in (("K% allowed", "is_k"), ("BB% allowed", "bbf")):
        b = est[["pid", "year", col]].copy(); b["half"] = rng.integers(0, 2, len(b)); g = b.groupby(["pid", "year", "half"])[col].agg(["mean", "size"]).unstack("half").dropna(); g = g[(g[("size", 0)] >= 30) & (g[("size", 1)] >= 30)]
        r = np.corrcoef(g[("mean", 0)], g[("mean", 1)])[0, 1]; rf = 2 * r / (1 + r); nbar = (g[("size", 0)] + g[("size", 1)]).mean(); rows.append({"measure": lab + " (per PA)", "pitcher-seasons": len(g), "mean balls": nbar, "split-half r (full-length)": rf, "implied n0 (balls)": nbar * (1 - rf) / rf})
    print("\nPART 2.  STICKINESS\n(a) split-half reliability (random halves of a pitcher-season's balls; PA for K/BB) and the implied n0 (shrinkage constant)\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    rows = []
    for k, (lo, hi) in indic.items():
        b = bb.copy(); b["x"] = ((b["angle"] >= lo) & (b["angle"] < hi)).astype(float); g = b.groupby(["pid", "year"])["x"].agg(["mean", "size"]).unstack("year").dropna(); g = g[(g[("size", 2025)] >= 60) & (g[("size", 2026)] >= 60)]
        r = np.corrcoef(g[("mean", 2025)], g[("mean", 2026)])[0, 1]; rows.append({"measure": k, "pitchers": len(g), "2025 -> 2026 r": r, "mean balls 25/26": f"{g[('size', 2025)].mean():.0f}/{g[('size', 2026)].mean():.0f}"})
    for lab, col in (("K% allowed", "is_k"), ("BB% allowed", "bbf")):
        g = est.groupby(["pid", "year"])[col].agg(["mean", "size"]).unstack("year").dropna(); g = g[(g[("size", 2025)] >= 120) & (g[("size", 2026)] >= 120)]; rows.append({"measure": lab, "pitchers": len(g), "2025 -> 2026 r": np.corrcoef(g[("mean", 2025)], g[("mean", 2026)])[0, 1], "mean balls 25/26": f"{g[('size', 2025)].mean():.0f}/{g[('size', 2026)].mean():.0f} PA"})
    print("(b) season-to-season correlation of the raw shares (pitchers with >= 60 balls in both seasons; K/BB >= 120 PA)\n" + pd.DataFrame(rows).round(3).to_string(index=False))


def trait_table():
    raw = fp.load_fl_pitches_cached(); raw = raw[raw["pitcher_id"].notna()].copy(); raw["pid"] = raw["pitcher_id"].astype(str); raw = raw[raw["pitch_type"].isin(["four_seam", "sinker", "cutter", "slider", "sweeper", "curveball", "changeup", "splitter"])]
    fb = raw[raw["pitch_type"].isin(["four_seam", "sinker", "cutter"])]; f4 = raw[raw["pitch_type"] == "four_seam"]; k = ["pid", "year"]
    T = pd.DataFrame({"fb_velo": fb.groupby(k)["effective_velo"].mean(), "fb_ivb": f4.groupby(k)["ivb"].mean(), "fb_hbreak": fb.groupby(k)["horz_break"].apply(lambda s: s.abs().mean()), "release_ht": raw.groupby(k)["release_pos_z"].mean(), "extension": raw.groupby(k)["extension"].mean(),
                      "fb_spin": fb.groupby(k)["spin_rate"].mean(), "pitch_ht": raw.groupby(k)["plate_z"].mean(), "fb_ht": fb.groupby(k)["plate_z"].mean(), "upper_zone_share": raw.groupby(k)["plate_z"].apply(lambda s: (s > 3.0).mean()), "n_pitches": raw.groupby(k).size()})
    mix = pd.crosstab([raw["pid"], raw["year"]], raw["pitch_type"], normalize="index"); mix.columns = ["mix_" + c for c in mix.columns]; T = T.join(mix)
    p2 = pd.read_parquet("2025_2026_pitches.parquet", columns=["pitcher_id", "year", "pitch_type", "vert_appr_angle", "pitcher_throws"]); p2["pid"] = p2["pitcher_id"].astype(str); p2 = p2[p2["pitch_type"].isin(["four_seam", "sinker", "cutter"])]
    T = T.join(p2.groupby(k)["vert_appr_angle"].mean().rename("fb_vaa")); th = p2.groupby(k)["pitcher_throws"].agg(lambda s: (s.astype(str).str.lower().str[0] == "l").mean()).rename("lefty"); return T.join(th)


def part3():
    bb = pitcher_balls(); bb["fb"] = ((bb["angle"] >= 25) & (bb["angle"] < 40)).astype(float); S = bb.groupby(["pid", "year"])["fb"].agg(["mean", "size"]).rename(columns={"mean": "fb", "size": "n"}); T = trait_table(); D = S.join(T, how="inner")
    D = D[(D["n"] >= 30) & (D["n_pitches"] >= 150)].copy(); feats = [c for c in T.columns if c != "n_pitches"]; D = D.reset_index(); print(f"\nPART 3.  PREDICTABILITY OF FLY-BALL SHARE FROM TRAITS: {len(D)} pitcher-seasons ({D['pid'].nunique()} pitchers), traits: {', '.join(feats)}")
    # (a) cross-sectional CV by pitcher, pooled seasons
    X = D[feats].copy(); X = X.fillna(X.median()); X = (X - X.mean()) / X.std(); y = D["fb"].to_numpy(); n = D["n"].to_numpy(float); p = np.zeros(len(D)); coefs = []
    for tr, te in GroupKFold(5).split(X, y, groups=D["pid"]):
        m = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(X.iloc[tr], y[tr], sample_weight=n[tr]); p[te] = m.predict(X.iloc[te]); coefs.append(m.coef_)
    base = np.array([np.average(y[D["pid"] != q], weights=n[D["pid"] != q]) for q in [None] * 0]) if False else np.full(len(y), np.average(y, weights=n)); a, b = ew.fit_variance_law(y - base, n); w = ew.inv_var_weights(n, a, b)
    rep = er.compare(y, base, p, w, D["pid"].to_numpy(), n=800); r2 = 1 - np.average((y - p) ** 2, weights=w) / np.average((y - base) ** 2, weights=w)
    print(f"(a) same-season traits -> FB share, 5-fold by pitcher, weighted (variance law a {a:.5f}, b {b:.3f} => n0_eval {b / a:.0f} balls): league mean RMSE {rep['RMSE old']:.4f} MAE {rep['MAE old']:.4f} | traits RMSE {rep['RMSE new']:.4f} MAE {rep['MAE new']:.4f} "
          f"(dRMSE {rep['dRMSE']:+.4f} {rep['RMSE 90%']}, P {rep['P(RMSE better)']:.2f}; dMAE {rep['dMAE']:+.4f}, P {rep['P(MAE better)']:.2f}); weighted R^2 {r2:.2f}")
    cf = pd.Series(np.mean(coefs, axis=0), index=feats).sort_values(key=lambda s: -s.abs()); print("   standardised ridge coefficients (FB share points per +1 SD of the trait; top 8): " + "; ".join(f"{k} {v * 100:+.2f}" for k, v in cf.head(8).items()))
    # (b) forward 2025 -> 2026
    A = D[D["year"] == 2025].set_index("pid"); B = D[D["year"] == 2026].set_index("pid"); ids = A.index.intersection(B.index); A, B = A.loc[ids], B.loc[ids]; yB = B["fb"].to_numpy(); nB = B["n"].to_numpy(float); lg = np.average(A["fb"], weights=A["n"])
    XA = A[feats].fillna(A[feats].median()); XA = (XA - XA.mean()) / XA.std(); preds = {"league mean": np.full(len(ids), np.average(yB, weights=nB) if False else np.average(D[D['year'] == 2025]['fb'], weights=D[D['year'] == 2025]['n']))}
    best = None
    for n0 in (50, 100, 150, 200, 300, 500, 1000):
        sh = lg + (A["fb"].to_numpy() - lg) * A["n"].to_numpy() / (A["n"].to_numpy() + n0); e = np.average((yB - sh) ** 2, weights=nB)
        if best is None or e < best[0]:
            best = (e, n0, sh)
    preds[f"2025 FB share, shrunk n0 {best[1]} (best on this sample)"] = best[2]
    # traits-only and traits+prior: leave-one-pitcher-out ridge (5-fold by pitcher)
    def cvridge(Xm):
        p_ = np.zeros(len(ids))
        for tr, te in GroupKFold(5).split(Xm, yB, groups=ids):
            m = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(Xm.iloc[tr], yB[tr], sample_weight=nB[tr]); p_[te] = m.predict(Xm.iloc[te])
        return p_
    preds["2025 traits (ridge, 5-fold by pitcher)"] = cvridge(XA); XA2 = XA.copy(); XA2["prior_fb_shrunk"] = (best[2] - lg) / (best[2] - lg).std(); preds["2025 FB share + 2025 traits"] = cvridge(XA2)
    base = preds["league mean"]; a, b = ew.fit_variance_law(yB - base, nB); w = ew.inv_var_weights(nB, a, b); rows = []
    for k, v in preds.items():
        rm, ma = er.wmetrics(yB, v, w); row = {"model": k, "n": len(ids), "RMSE": rm, "MAE": ma}
        if k != "league mean":
            rep = er.compare(yB, base, v, w, ids.to_numpy(), n=800); row.update({"dRMSE vs mean": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
        rows.append(row)
    print(f"\n(b) FORWARD 2025 -> 2026 FB share, {len(ids)} pitchers in both seasons (weights: variance law a {a:.5f}, b {b:.3f}; n = 2026 balls)\n" + er.table(rows, 4))
    print(f"   corr(2025 FB share, 2026 FB share) {np.corrcoef(A['fb'], B['fb'])[0, 1]:.2f}; corr(2025 traits-ridge prediction, 2026 FB share) {np.corrcoef(preds['2025 traits (ridge, 5-fold by pitcher)'], yB)[0, 1]:.2f}")


if __name__ == "__main__":
    part1(); part2(); part3()
