"""
fl_pitcher_fbpu_traits.py -- can the fly ball + pop-up share (>= 25 deg) be predicted from pitcher TRAITS?  (user 2026-10-09; same design as fl_pitcher_fbshare_sticky.py part 3, other targets for reference)
Targets (share of a pitcher-season's balls in play): FB+PU (>= 25), pop-up (> 40), FB (25-40), Air% (>= 10), GB (< 10).  Traits: fastball vertical approach angle, velocity, induced vertical break, horizontal break, release height, extension, spin, mean pitch height,
fastball height, upper-zone share (> 3.0 ft), pitch mix (8 types), handedness.
(a) same-season traits -> share, pitcher-grouped 5-fold ridge on pooled pitcher-seasons (n >= 30 balls); weighted RMSE / MAE vs the league mean (variance law fitted on the league-mean residuals, frozen), pitcher-cluster bootstrap, weighted R^2, top standardised drivers.
(b) forward 2025 -> 2026, pitchers in both seasons: league mean | last year's share (shrunk at its split-half n0) | last year's traits (ridge, pitcher-grouped folds) | both.
(c) the same, but how much of the trait signal is just 'pitch height / approach angle' (two-trait model) vs the full set.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

import fl_eval_report as er
import fl_eval_weights as ew
import fl_pitcher_fbshare_sticky as st

warnings.filterwarnings("ignore")
TARGETS = {"FB+PU (>=25)": (25, 91, 92), "pop-up (>40)": (40, 91, 132), "FB (25-40)": (25, 40, 559), "Air% (>=10)": (10, 91, 103), "GB (<10)": (-91, 10, 115)}


def cvridge(X, y, n, groups):
    p = np.zeros(len(y)); coefs = []
    for tr, te in GroupKFold(5).split(X, y, groups=groups):
        m = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(X.iloc[tr], y[tr], sample_weight=n[tr]); p[te] = m.predict(X.iloc[te]); coefs.append(m.coef_)
    return p, np.mean(coefs, axis=0)


def main():
    bb = st.pitcher_balls(); T = st.trait_table(); feats = [c for c in T.columns if c != "n_pitches"]; small = ["fb_vaa", "pitch_ht"]; summary = []
    for lab, (lo, hi, n0) in TARGETS.items():
        bb["x"] = ((bb["angle"] >= lo) & (bb["angle"] < hi)).astype(float); S = bb.groupby(["pid", "year"])["x"].agg(["mean", "size"]).rename(columns={"mean": "y", "size": "n"}); D = S.join(T, how="inner")
        D = D[(D["n"] >= 30) & (D["n_pitches"] >= 150)].reset_index(); X = D[feats].copy(); X = X.fillna(X.median()); X = (X - X.mean()) / X.std(); y = D["y"].to_numpy(); n = D["n"].to_numpy(float); grp = D["pid"].to_numpy()
        base = np.full(len(y), np.average(y, weights=n)); a, b = ew.fit_variance_law(y - base, n); w = ew.inv_var_weights(n, a, b)
        p, cf = cvridge(X, y, n, grp); p2, _ = cvridge(X[small], y, n, grp); rep = er.compare(y, base, p, w, grp, n=800); rep2 = er.compare(y, base, p2, w, grp, n=800)
        r2 = 1 - np.average((y - p) ** 2, weights=w) / np.average((y - base) ** 2, weights=w); r2s = 1 - np.average((y - p2) ** 2, weights=w) / np.average((y - base) ** 2, weights=w)
        top = pd.Series(cf, index=feats).sort_values(key=lambda s: -s.abs()).head(6); sd_y = np.sqrt(np.average((y - base) ** 2, weights=w))
        print(f"\n=== {lab}: (a) SAME-SEASON traits, {len(D)} pitcher-seasons ({D['pid'].nunique()} pitchers); variance law a {a:.5f}, b {b:.3f} (n0_eval {b / a:.0f} balls)\n"
              f"   league mean  RMSE {rep['RMSE old']:.4f}  MAE {rep['MAE old']:.4f}\n   all traits   RMSE {rep['RMSE new']:.4f}  MAE {rep['MAE new']:.4f}  dRMSE {rep['dRMSE']:+.4f} {rep['RMSE 90%']} P {rep['P(RMSE better)']:.2f}; dMAE {rep['dMAE']:+.4f} P {rep['P(MAE better)']:.2f}; weighted R^2 {r2:.2f}\n"
              f"   approach angle + pitch height only: RMSE {rep2['RMSE new']:.4f} MAE {rep2['MAE new']:.4f}  (R^2 {r2s:.2f})\n   top drivers (share points per +1 SD of trait): " + "; ".join(f"{k} {v * 100:+.2f}" for k, v in top.items()))
        # (b) forward
        A = D[D["year"] == 2025].set_index("pid"); B = D[D["year"] == 2026].set_index("pid"); ids = A.index.intersection(B.index); A, B = A.loc[ids], B.loc[ids]; yB = B["y"].to_numpy(); nB = B["n"].to_numpy(float); lg = np.average(A["y"], weights=A["n"])
        shrunk = lg + (A["y"].to_numpy() - lg) * A["n"].to_numpy() / (A["n"].to_numpy() + n0); XA = A[feats].fillna(A[feats].median()); XA = (XA - XA.mean()) / XA.std(); XA2 = XA.copy(); XA2["prior"] = (shrunk - shrunk.mean()) / shrunk.std()
        pt, _ = cvridge(XA, yB, nB, ids.to_numpy()); pb, _ = cvridge(XA2, yB, nB, ids.to_numpy()); preds = {"league mean": np.full(len(ids), lg), f"2025 share shrunk (n0 {n0})": shrunk, "2025 traits (ridge)": pt, "2025 share + traits": pb}
        base_b = preds["league mean"]; a2, b2 = ew.fit_variance_law(yB - base_b, nB); w2 = ew.inv_var_weights(nB, a2, b2); rows = []
        for k, v in preds.items():
            rm, ma = er.wmetrics(yB, v, w2); row = {"model": k, "RMSE": rm, "MAE": ma}
            if k != "league mean":
                r_ = er.compare(yB, base_b, v, w2, ids.to_numpy(), n=800); row.update({"dRMSE": r_["dRMSE"], "RMSE 90%": r_["RMSE 90%"], "P(RMSE better)": r_["P(RMSE better)"], "dMAE": r_["dMAE"], "P(MAE better)": r_["P(MAE better)"]})
            rows.append(row)
        print(f"(b) FORWARD 2025 -> 2026, {len(ids)} pitchers in both seasons (corr 2025 vs 2026 share {np.corrcoef(A['y'], B['y'])[0, 1]:.2f}; traits-ridge prediction vs 2026 {np.corrcoef(pt, yB)[0, 1]:.2f})\n" + er.table(rows, 4))
        summary.append({"target": lab, "same-season R^2 (all traits)": r2, "R^2 (2 traits)": r2s, "dRMSE same-season": rep["dRMSE"], "P": rep["P(RMSE better)"], "fwd corr 25-26": np.corrcoef(A["y"], B["y"])[0, 1], "fwd corr traits": np.corrcoef(pt, yB)[0, 1]})
    print("\nSUMMARY\n" + pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
