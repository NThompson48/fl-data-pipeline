"""
fl_hitter_quality_by_group.py -- is the 'air-ball hitters make better contact at Trustmark' finding a PARK effect or competition (Mississippi's staff)?  (user 2026-10-10: Mississippi hitters face many different defences, which randomises that)
Contact quality = xwOBA of the ball.  xw_adj = xwOBA - the hitter's own AWAY mean xwOBA (shrunk toward the league mean, n0 100).  For each group of Trustmark balls regress xw_adj on the hitter's Air% profile (z, shrunk n0 89, from his away balls) and on power (HR per ball in play, n0 100):
 VISITING hitters at Trustmark (all face the Mississippi staff)  vs  MISSISSIPPI hitters at home (face the 17 visiting staffs); same for the away baseline (visitors elsewhere).  Game-cluster bootstrap SEs.
Also the same regression on the residual (wOBA - xwOBA - own away mean), i.e. what the park does to a ball of the given quality.
"""
import warnings
import numpy as np
import pandas as pd
import fl_hitter_park_ms_check as ms
warnings.filterwarnings("ignore")
b = ms.prep(); away = b[(b["tm"] == 0) & (b["vis"] == 1)]; lgx = away["xwoba"].mean(); g = b[b["tm"] == 0].groupby("name_clean").agg(n=("la", "size"), xw=("xwoba", "mean"), air=("la", lambda s: (s >= 10).mean()), hr=("woba", lambda s: (s > 1.8).mean()))
base = (g["xw"] * g["n"] + lgx * 100) / (g["n"] + 100); lg_air = np.average(g["air"], weights=g["n"]); lg_hr = np.average(g["hr"], weights=g["n"])
z_air = (g["air"] - lg_air) * g["n"] / (g["n"] + 89); z_hr = (g["hr"] - lg_hr) * g["n"] / (g["n"] + 100); z_air = (z_air - z_air.mean()) / z_air.std(); z_hr = (z_hr - z_hr.mean()) / z_hr.std()
b["xw_adj"] = b["xwoba"] - b["name_clean"].map(base); b["z_air"] = b["name_clean"].map(z_air); b["z_hr"] = b["name_clean"].map(z_hr); b = b.dropna(subset=["xw_adj", "z_air", "z_hr"]).reset_index(drop=True)
codes, uniq = pd.factorize(b["game_id"]); rng = np.random.default_rng(3); W = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes] for _ in range(300)]


def wls(y, X, w):
    A = X.T @ (X * w[:, None]); return np.linalg.solve(A, X.T @ (w * y))


groups = {"away baseline (visitors elsewhere)": ((b["tm"] == 0) & (b["vis"] == 1)).to_numpy(), "VISITING hitters at Trustmark (face MS staff)": ((b["tm"] == 1) & (b["vis"] == 1)).to_numpy(), "MISSISSIPPI hitters at home (face 17 staffs)": (b["ms_home"] == 1).to_numpy()}
for tgt, title in (("xw_adj", "CONTACT QUALITY (xwOBA minus own away mean)"), ("res_adj", "PARK EFFECT ON SAME-QUALITY CONTACT (wOBA - xwOBA - own away mean)")):
    rows = []; y = b[tgt].to_numpy(float); X = np.column_stack([np.ones(len(b)), b["z_air"], b["z_hr"]])
    for lab, m in groups.items():
        est = wls(y[m], X[m], np.ones(m.sum())); bs = np.array([wls(y[m], X[m], W[i][m].astype(float)) for i in range(len(W)) if W[i][m].sum() > 0]); se = bs.std(axis=0)
        rows.append({"group": lab, "balls": int(m.sum()), "mean": y[m].mean(), "Air% per SD": f"{est[1]:+.4f} ({se[1]:.4f}, t {est[1] / se[1]:+.1f})", "power per SD": f"{est[2]:+.4f} ({se[2]:.4f}, t {est[2] / se[2]:+.1f})"})
    print(f"\n{title}\n" + pd.DataFrame(rows).round(4).to_string(index=False))
