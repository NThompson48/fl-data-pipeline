"""
fl_hitter_park_hr_surface.py -- HOME-RUN version of the Trustmark batted-ball surface, so HR/PA history is neutralised on the same basis as wOBA (user 2026-10-13: park effect applied
post-projection from the surface with handedness; history neutralised with the surface for wOBA AND HR/PA, replacing the old 0.446 HR factor; K% and BB% effects from the component table).
Same design as fl_hitter_park_surface_final.py: effect(x) = f_TM(x) - f_away(x), x = exit speed, launch angle, batter side; Trustmark balls = visiting hitters + Mississippi hitters at home,
away balls = hitters away from their own park; residual = is_HR - xHR (xHR = P(HR | x) from away balls, 5-fold cross-fitted), hitter-adjusted by his own away mean (shrunk n0 150);
hitter shift per ball in play = mean effect over his non-Trustmark balls, shrunk toward the league mean by n / (n + 60).
Also stores the inputs of the K% / BB% wOBA consequence: league wOBA per ball in play away from Trustmark and the FL walk weight.
Outputs: output/hitter_park_hr_surface_effect.parquet (name_clean, n_balls, shift_hr_bip), output/hitter_park_hr_surface_coefs.json.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

import fl_hitter_park_ms_check as ms
import fl_hitter_park_surface as ps

warnings.filterwarnings("ignore")
FEAT = "evla_side"


def main():
    b = ms.prep(); fm = ps.FEATS[FEAT]; b["is_hr"] = (b["play_result"] == "home_run").astype(float)
    away = (b["tm"] == 0).to_numpy(); xhr = np.zeros(len(b)); kw = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    ia = np.nonzero(away)[0]
    for tr, te in KFold(5, shuffle=True, random_state=0).split(ia):          # away balls: cross-fitted; Trustmark balls: model fit on all away balls
        c = HistGradientBoostingClassifier(**kw).fit(b.iloc[ia[tr]][fm], b.iloc[ia[tr]]["is_hr"]); xhr[ia[te]] = c.predict_proba(b.iloc[ia[te]][fm])[:, 1]
    c = HistGradientBoostingClassifier(**kw).fit(b.iloc[ia][fm], b.iloc[ia]["is_hr"]); xhr[~away] = c.predict_proba(b.loc[~away, fm])[:, 1]
    b["res_hr"] = b["is_hr"] - xhr; b["res_adj"] = b["res_hr"] - b["name_clean"].map(ps.hitter_baseline(b, "res_hr")).fillna(0.0)
    eff = ms.fit_effect_pooled(b, FEAT); sh, mu = ps.hitter_shift(b, eff, FEAT)
    n = b[b["tm"] == 0].groupby("name_clean").size()
    pd.DataFrame({"name_clean": sh.index, "n_balls": n.reindex(sh.index).to_numpy(), "shift_hr_bip": sh.to_numpy()}).to_parquet("output/hitter_park_hr_surface_effect.parquet", index=False)
    from fl_hitter_panel import woba_weights
    wcon = float(b.loc[away, "woba"].mean()); wbb = float(woba_weights()["BB"])
    tm = b[b["tm"] == 1]; raw_ratio = tm["is_hr"].mean() / xhr[~away].mean()
    json.dump({"model": "residual surface (is_HR - xHR), features " + FEAT, "league_mean_hr_effect_per_bip": float(mu), "bip_per_pa": 0.671, "n0_shrink_balls": 60,
               "wobacon_away": wcon, "woba_weight_bb": wbb, "trustmark_hr_over_expected_raw": float(raw_ratio), "training": "visiting hitters + Mississippi hitters at home, 2025-26"},
              open("output/hitter_park_hr_surface_coefs.json", "w"), indent=1)
    lg_hr = b.loc[away, "is_hr"].mean()
    print(f"{len(b):,} balls; HR per ball in play away {lg_hr:.4f}; Trustmark actual / expected HR {raw_ratio:.3f}")
    print(f"surface: league-mean HR effect per ball in play {mu:+.4f} (= {mu / lg_hr:+.1%} of the away HR rate; per PA at full park {0.671 * mu:+.4f}); hitter shift SD {sh.std():.4f}, range [{sh.min():+.4f}, {sh.max():+.4f}] over {len(sh)} hitters")
    print(f"K%/BB% wOBA inputs: wOBA per ball in play away {wcon:.4f}; walk weight {wbb:.3f}")


if __name__ == "__main__":
    main()
