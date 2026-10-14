"""fl_park_pitcher_fbpu_byyear.py -- Air% and FB+PU at Trustmark with K% and BB% controlled: pooled and by season (single term each, shrunk at its own n0), plus the year-difference t."""
import warnings
import numpy as np
import pandas as pd
import fl_park_pitcher_final as pf
import fl_park_pitcher_profile_screen as sc
import fl_pitcher_fbshare_sticky as st
from fl_park_profile_fe import cluster_ols, partial_out
warnings.filterwarnings("ignore")

est, prof, stdz = pf.build(); est["b_home"] = (est["batter_team"] == est["home_team"]).astype(int); C = sc.build_candidates(est)
bb = pf.pitcher_bbe(); bb["pid"] = bb["pitcher_id"].astype(str); bb["park"] = bb["game_id"].str.split("-").str[1]; bb = bb[bb["park"] != "TrustmarkPark"]
Z = {"Air% (>=10)": st.share_z(bb, 10, 91, est, 103), "FB+PU (>=25)": st.share_z(bb, 25, 91, est, 92), "K": C["K% allowed"], "BB": C["BB% allowed"]}
for k, v in Z.items():
    est["z_" + k] = est["pid"].map(v)
est = est.dropna(subset=["z_K", "z_BB"]).reset_index(drop=True)
for k in Z:
    est["z_" + k] = est["z_" + k].fillna(0.0)
hb = est["b_home"] == 1; g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pid"])[0], pd.factorize(np.where(~hb, est["pid"], "AWAY"))[0], pd.factorize(est["park"])[0]]
tmv = est["tm"].to_numpy(float) * hb.to_numpy(float); yr = partial_out(est["woba"].to_numpy(float)[:, None], g, iters=30)[:, 0]; cl = pd.factorize(est["game_id"])[0]
y25, y26 = (est["year"] == 2025).to_numpy(float), (est["year"] == 2026).to_numpy(float); rows = []
for c in ("Air% (>=10)", "FB+PU (>=25)"):
    def col(t, m=None):
        v = tmv * est["z_" + t].to_numpy(float); v = v if m is None else v * m
        return partial_out(v[:, None], g, iters=30)[:, 0]
    b, se = cluster_ols(np.column_stack([col("K"), col("BB"), col(c)]), yr, cl)
    cols = [col(t, m) for t in ("K", "BB", c) for m in (y25, y26)]; by, sy = cluster_ols(np.column_stack(cols), yr, cl)
    rows.append({"profile": c, "pooled": f"{b[2]:+.4f} ({se[2]:.4f})", "t": b[2] / se[2], "runs/9": b[2] * pf.TO_R9, "2025": f"{by[4]:+.4f} ({sy[4]:.4f})", "2026": f"{by[5]:+.4f} ({sy[5]:.4f})", "t of difference": (by[4] - by[5]) / np.hypot(sy[4], sy[5])})
print(pd.DataFrame(rows).round(4).to_string(index=False))
