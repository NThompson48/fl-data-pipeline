"""
fl_park_pitcher_profile_replicate.py -- does the 'contact-prone / fly-ball pitchers do better at Trustmark' pattern replicate across seasons?  (follows fl_park_pitcher_profile_screen.py)
(1) Year split: Trustmark x profile coefficient estimated separately for 2025 and 2026 visiting-pitcher PA (same FE, K% and BB% allowed controlled), for the five screened candidates.
(2) One composite (fixed BEFORE looking at the year split, from the screen's signs): contact-prone = mean of z(fly-ball share), z(HR/PA allowed), z(exit speed allowed), -z(ground-ball share), z(-fastball IVB is NOT included: stuff, kept separate);
    re-standardised; full-sample coefficient with a game-cluster bootstrap, by year, and with the old production profile terms (HR-prone, K%, Air%, left) in the model to see whether it adds anything.
"""
import warnings

import numpy as np
import pandas as pd

import fl_park_pitcher_final as pf
import fl_park_pitcher_profile_screen as sc
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")


def main():
    est, prof, stdz = pf.build(); est["b_home"] = (est["batter_team"] == est["home_team"]).astype(int); C = sc.build_candidates(est); names = list(C.columns)
    for c in names:
        est[c] = est["pid"].map(C[c])
    est = est.dropna(subset=["K% allowed", "BB% allowed"]).reset_index(drop=True)
    for c in names:
        est[c] = est[c].fillna(0.0)
    comp = (est["fly-ball share"] + est["HR/PA allowed"] + est["exit speed allowed"] - est["ground-ball share"]) / 4; est["contact-prone composite"] = (comp - comp.mean()) / comp.std()
    home_batter = (est["b_home"] == 1); g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pid"])[0], pd.factorize(np.where(~home_batter, est["pid"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    vis = home_batter.to_numpy(float); tmv = est["tm"].to_numpy(float) * vis; y = est["woba"].to_numpy(float); yr = partial_out(y[:, None], g, iters=30)[:, 0]; cl = pd.factorize(est["game_id"])[0]; y25 = (est["year"] == 2025).to_numpy(float); y26 = (est["year"] == 2026).to_numpy(float)
    print(f"visiting-pitcher PA at Trustmark: 2025 {int((tmv * y25).sum()):,}, 2026 {int((tmv * y26).sum()):,}")
    cands = ["fastball induced vert. break", "fly-ball share", "HR/PA allowed", "exit speed allowed", "ground-ball share", "contact-prone composite"]
    cols, labs = [], []
    for c in ["K% allowed", "BB% allowed"] + cands:
        for yy, m in (("2025", y25), ("2026", y26)):
            cols.append(tmv * m * est[c].to_numpy()); labs.append((c, yy))
    Xr = partial_out(np.column_stack(cols), g, iters=30); ix = {l: i for i, l in enumerate(labs)}; rows = []
    for c in cands:
        sel = [ix[("K% allowed", "2025")], ix[("K% allowed", "2026")], ix[("BB% allowed", "2025")], ix[("BB% allowed", "2026")], ix[(c, "2025")], ix[(c, "2026")]]
        b, se = cluster_ols(Xr[:, sel], yr, cl); rows.append({"profile": c, "2025 coef": b[4], "2025 SE": se[4], "2026 coef": b[5], "2026 SE": se[5], "same sign": np.sign(b[4]) == np.sign(b[5]), "pooled-by-year t of difference": (b[4] - b[5]) / np.hypot(se[4], se[5])})
    print("\n(1) BY YEAR (K% and BB% allowed controlled; each year's own K/BB terms)\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    # (2) composite pooled with a bootstrap, alone and with the production terms
    def pooled(extra):
        cs = ["K% allowed", "BB% allowed", "contact-prone composite"] + extra; X = np.column_stack([tmv * est[c].to_numpy() for c in cs]); return cs, partial_out(X, g, iters=30)
    for extra, lab in (([], "composite + K% + BB%"), (["HR/PA allowed", "left-handed"], "composite + K% + BB% + HR/PA + left")):
        cs, X = pooled(extra); b, se = cluster_ols(X, yr, cl); rng = np.random.default_rng(6); ug = np.unique(cl); by = pd.Series(np.arange(len(cl))).groupby(cl).apply(np.array); bs = []
        for _ in range(300):
            ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); bs.append(np.linalg.lstsq(X[ii], yr[ii], rcond=None)[0])
        bs = np.array(bs); print(f"\n(2) {lab}\n" + pd.DataFrame({"term x Trustmark": cs, "coef per +1 SD": b, "cluster SE": se, "bootstrap SE": bs.std(axis=0), "t (bootstrap)": b / bs.std(axis=0), "runs/9": b * pf.TO_R9}).round(4).to_string(index=False))
    sd_c = est.drop_duplicates("pid")["contact-prone composite"].std(); print(f"\ncomposite SD across pitchers (PA-weighted pool standardised to 1; pitcher-level SD {sd_c:.2f})")


if __name__ == "__main__":
    main()
