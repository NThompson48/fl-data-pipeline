"""
fl_park_air_power_confound.py -- are Air% and Power confounding each other in the Trustmark profile terms?  (visiting hitters only, wOBA, same fixed effects as fl_park_final_visitors.py; cluster-robust SEs by game)
(1) Correlation of the two profiles (hitters, PA-weighted, and among the visitor PAs at Trustmark) and the variance inflation of each coefficient when both are in the model.
(2) Trustmark x profile coefficients: Air% alone | Power alone | both (production) | Air% made ORTHOGONAL to power (Air% residual after regressing on power) with power | power orthogonal to Air% with Air%.
(3) 2 x 2 cells (median splits of Air% and Power): Trustmark-specific wOBA effect of each cell relative to the Trustmark average, so the four combinations (high air / low power ... low air / high power) can be compared directly,
    with contrasts: air effect holding power low, air effect holding power high, power effect holding air low, power effect holding air high.
"""
import warnings

import numpy as np
import pandas as pd

from fl_park_final_visitors import hitter_frame
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")


def main():
    est, ml = hitter_frame(); g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pitcher"])[0], pd.factorize(np.where(est["b_home"] == 1, est["name_clean"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    cl = pd.factorize(est["game_id"])[0]; vis_tm = est["tm"].to_numpy(float) * (est["b_home"] == 0).to_numpy(float); y = est["woba"].to_numpy(float); yr = partial_out(y[:, None], g, iters=30)[:, 0]
    za, zp = est["z_air"].to_numpy(float), est["z_power"].to_numpy(float); print(f"(1) correlation of the profiles: all PA {np.corrcoef(za, zp)[0, 1]:+.2f}; visitor PA at Trustmark {np.corrcoef(za[vis_tm == 1], zp[vis_tm == 1])[0, 1]:+.2f}; hitters (one row each) "
          f"{est.drop_duplicates('name_clean')[['z_air', 'z_power']].corr().iloc[0, 1]:+.2f}")

    def reg(cols, names):
        X = np.column_stack([vis_tm * c for c in cols]); Xr = partial_out(X, g, iters=30); b, se = cluster_ols(Xr, yr, cl); return {n: f"{b[i]:+.4f} ({se[i]:.4f}) t {b[i] / se[i]:+.1f}" for i, n in enumerate(names)}, Xr
    rows = []
    r, _ = reg([za], ["Air%"]); rows.append({"model": "Air% alone", **r}); r, _ = reg([zp], ["Power"]); rows.append({"model": "Power alone", **r}); r, Xr2 = reg([za, zp], ["Air%", "Power"]); rows.append({"model": "both (production)", **r})
    vif = 1 / (1 - np.corrcoef(Xr2[:, 0], Xr2[:, 1])[0, 1] ** 2); beta = np.polyfit(zp, za, 1); za_perp = za - np.polyval(beta, zp); zp_perp = zp - np.polyval(np.polyfit(za, zp, 1), za)
    r, _ = reg([za_perp, zp], ["Air% (orthogonal to power)", "Power"]); rows.append({"model": "Air% orthogonalised to power", **r}); r, _ = reg([za, zp_perp], ["Air%", "Power (orthogonal to Air%)"]); rows.append({"model": "power orthogonalised to Air%", **r})
    print("(2) Trustmark x profile on wOBA, visiting hitters (coefficient (cluster-robust SE) per +1 SD)\n" + pd.DataFrame(rows).fillna("").to_string(index=False)); print(f"   variance inflation from including both (in the interaction design): {vif:.2f}x on each variance (SE x {np.sqrt(vif):.2f})")
    # (3) 2 x 2 cells
    ha = (za > np.median(est.drop_duplicates("name_clean")["z_air"])).astype(float); hp = (zp > np.median(est.drop_duplicates("name_clean")["z_power"])).astype(float)
    cells = {"low air / low power": (1 - ha) * (1 - hp), "low air / HIGH power": (1 - ha) * hp, "HIGH air / low power": ha * (1 - hp), "HIGH air / HIGH power": ha * hp}
    X = np.column_stack([vis_tm * c for c in cells.values()]); Xr = partial_out(X, g, iters=30); b, se = cluster_ols(Xr, yr, cl); n_vis = {k: int((vis_tm * c).sum()) for k, c in cells.items()}
    print("\n(3) 2 x 2 CELLS (median splits), Trustmark-specific wOBA effect of each cell (visiting-hitter PA; the cells' average is the park effect)\n" + pd.DataFrame({"cell": list(cells), "visitor PA at Trustmark": [n_vis[k] for k in cells], "effect (SE)": [f"{b[i]:+.4f} ({se[i]:.4f})" for i in range(4)]}).to_string(index=False))
    V = np.linalg.inv(Xr.T @ Xr)
    def contrast(w, lab):
        w = np.array(w, float); est_ = w @ b; meat = pd.DataFrame(Xr * (yr - Xr @ b)[:, None]).groupby(cl).sum().to_numpy(); Vc = V @ (meat.T @ meat) @ V; s = np.sqrt(w @ Vc @ w); print(f"   {lab}: {est_:+.4f} (SE {s:.4f}, t {est_ / s:+.1f})")
    print("   contrasts:"); contrast([-1, 0, 1, 0], "air effect holding power LOW (high-air/low-power minus low-air/low-power)"); contrast([0, -1, 0, 1], "air effect holding power HIGH (high-air/high-power minus low-air/high-power)")
    contrast([-1, 1, 0, 0], "power effect holding air LOW (low-air/high-power minus low-air/low-power)"); contrast([0, 0, -1, 1], "power effect holding air HIGH (high-air/high-power minus high-air/low-power)")


if __name__ == "__main__":
    main()
