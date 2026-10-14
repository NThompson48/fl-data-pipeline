"""
fl_park_matched_pair_ms.py -- Mud Monsters home vs road as a matched pair (same players, hitters and pitchers separately), beyond the league-wide home advantage, with cluster-robust SEs.
woba = year + hitter + pitcher effects + b x [batter at home] + h x [Mississippi hitter at home] + p x [Mississippi pitcher at home]; h = how much better Mississippi's hitters are at Trustmark than the average team's hitters are at
home (relative to the road), p = the same for wOBA allowed by Mississippi's pitchers (positive = worse for them).  Compare with the visitor-only park estimates (visiting hitters -.0126, visiting pitchers +.0127, average 0).
Matched-pair PA counts and the same extra effect for the other 17 teams (point estimates) are printed for context.
"""
import warnings
import numpy as np
import pandas as pd
from fl_park_profile_fe import cluster_ols, partial_out
from fl_park_visitors_fe import load

warnings.filterwarnings("ignore")
pa = load(); MS = "Mississippi Mud Monsters"
g = [pd.factorize(pa["year"])[0], pd.factorize(pa["name_clean"])[0], pd.factorize(pa["pid"])[0]]; y = pa["woba"].to_numpy(float)
mh = ((pa["b_home"] == 1) & (pa["batter_team"] == MS)).to_numpy(float); mp = ((pa["b_home"] == 0) & (pa["pitcher_team"] == MS)).to_numpy(float)
X = np.column_stack([pa["b_home"].to_numpy(float), mh, mp]); Xr = partial_out(X, g, iters=40); yr = partial_out(y[:, None], g, iters=40)[:, 0]; b, se = cluster_ols(Xr, yr, pd.factorize(pa["game_id"])[0])
print(f"generic home-minus-road gap (hitters minus pitchers sides) {b[0]:+.4f} (SE {se[0]:.4f})")
print(f"MISSISSIPPI HITTERS at home vs road, extra vs the average team: {b[1]:+.4f} wOBA (SE {se[1]:.4f}, t {b[1] / se[1]:.2f}); PA at home {int(mh.sum()):,}")
print(f"MISSISSIPPI PITCHERS at home vs road, extra wOBA ALLOWED vs the average team: {b[2]:+.4f} wOBA (SE {se[2]:.4f}, t {b[2] / se[2]:.2f}); PA at home {int(mp.sum()):,}")
comb = (b[1] - b[2]) / 2; v = (se[1] ** 2 + se[2] ** 2) / 4; print(f"average gain (hitters up, pitchers down): {comb:+.4f} wOBA (SE about {np.sqrt(v):.4f}) = a team home-field fit of about {comb * 39.9:+.2f} runs/9-equivalent; the park-only estimate (visitor-only average) is 0.000 +- 0.009")
