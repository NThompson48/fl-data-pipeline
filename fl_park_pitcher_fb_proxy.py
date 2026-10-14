"""
fl_park_pitcher_fb_proxy.py -- follow-up to fl_pitcher_fbshare_sticky.py: the 25-40 degree fly-ball share has split-half reliability .15 (implied n0 ~ 770 balls), yet the Trustmark screen used n0 = 89.
(1) Re-run the Trustmark x FB-share term with the share shrunk at its OWN reliability (n0 770), and GB / Air% / pop-up at theirs (n0 ~ 100-135): does the result change?
(2) Trustmark x TRAIT-PREDICTED fly-ball share (ridge on fastball vertical approach angle, velocity, release height, pitch height, mix ..., 5-fold CROSS-FITTED by pitcher so a pitcher's own balls never predict his own share): a stable pitch-shape proxy of fly-ball tendency.
(3) Trustmark x Air% / GB at proper shrinkage; year split for FB (proper n0) and the trait proxy.
All with K% and BB% allowed controlled, visiting pitchers only, FE design as before.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

import fl_pitcher_fbshare_sticky as st
import fl_park_pitcher_final as pf
import fl_park_pitcher_profile_screen as sc
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")


def trait_proxy():
    bb = st.pitcher_balls(); bb["fb"] = ((bb["angle"] >= 25) & (bb["angle"] < 40)).astype(float); S = bb.groupby(["pid", "year"])["fb"].agg(["mean", "size"]).rename(columns={"mean": "fb", "size": "n"}); T = st.trait_table(); feats = [c for c in T.columns if c != "n_pitches"]
    D = S.join(T, how="right"); D = D[D["n_pitches"] >= 150].copy().reset_index(); X = D[feats].copy(); X = X.fillna(X.median()); X = (X - X.mean()) / X.std(); ok = (D["n"] >= 30).to_numpy(); pred = np.full(len(D), np.nan)
    pids = D["pid"].to_numpy(); folds = pd.factorize(pd.Series(pids).map(lambda p: hash(p) % 5))[0]
    for f in range(5):
        tr = ok & (folds != f); te = folds == f; m = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(X[tr], D.loc[tr, "fb"], sample_weight=D.loc[tr, "n"]); pred[te] = m.predict(X[te])
    D["fb_pred"] = pred; return D.groupby("pid").apply(lambda g: np.average(g["fb_pred"].dropna(), weights=g.loc[g["fb_pred"].notna(), "n_pitches"]) if g["fb_pred"].notna().any() else np.nan)


def main():
    est, prof, stdz = pf.build(); est["b_home"] = (est["batter_team"] == est["home_team"]).astype(int); C = sc.build_candidates(est)
    bb = pf.pitcher_bbe(); bb["pid"] = bb["pitcher_id"].astype(str); bb["park"] = bb["game_id"].str.split("-").str[1]; bb = bb[bb["park"] != "TrustmarkPark"]
    Z = pd.DataFrame({"FB n0 89 (screen)": st.share_z(bb, 25, 40, est, 89), "FB n0 770 (own reliability)": st.share_z(bb, 25, 40, est, 770), "GB n0 115": st.share_z(bb, -91, 10, est, 115), "Air% n0 100": st.share_z(bb, 10, 91, est, 100), "pop-up n0 135": st.share_z(bb, 40, 91, est, 135)})
    px = trait_proxy(); Z["FB predicted from traits"] = (px - px.reindex(est["pid"]).mean()) / px.reindex(est["pid"]).std(); Z["K"] = C["K% allowed"]; Z["BB"] = C["BB% allowed"]
    for c in Z.columns:
        est["z_" + c] = est["pid"].map(Z[c])
    est = est.dropna(subset=["z_K", "z_BB"]).reset_index(drop=True)
    for c in Z.columns:
        est["z_" + c] = est["z_" + c].fillna(0.0)
    home_batter = (est["b_home"] == 1); g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pid"])[0], pd.factorize(np.where(~home_batter, est["pid"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    tmv = est["tm"].to_numpy(float) * home_batter.to_numpy(float); yr = partial_out(est["woba"].to_numpy(float)[:, None], g, iters=30)[:, 0]; cl = pd.factorize(est["game_id"])[0]; y25, y26 = (est["year"] == 2025).to_numpy(float), (est["year"] == 2026).to_numpy(float)
    Xr = {c: partial_out((tmv * est["z_" + c].to_numpy())[:, None], g, iters=30)[:, 0] for c in Z.columns}; rows = []
    print(f"corr(FB trait proxy, FB share n0 770) among pitchers: {Z['FB predicted from traits'].corr(Z['FB n0 770 (own reliability)']):+.2f}; with GB: {Z['FB predicted from traits'].corr(Z['GB n0 115']):+.2f}; with K%: {Z['FB predicted from traits'].corr(Z['K']):+.2f}")
    for c in [c for c in Z.columns if c not in ("K", "BB")]:
        b, se = cluster_ols(np.column_stack([Xr["K"], Xr["BB"], Xr[c]]), yr, cl); ys = {}
        for yy, m in (("2025", y25), ("2026", y26)):
            cols = [partial_out((tmv * m * est["z_" + t].to_numpy())[:, None], g, iters=30)[:, 0] for t in ("K", "BB", c)]; by, sy = cluster_ols(np.column_stack(cols), yr, cl); ys[yy] = f"{by[2]:+.4f} ({sy[2]:.4f})"
        rows.append({"profile (K%, BB% controlled)": c, "coef per +1 SD": b[2], "SE": se[2], "t": b[2] / se[2], "runs/9": b[2] * pf.TO_R9, "2025": ys["2025"], "2026": ys["2026"]})
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
