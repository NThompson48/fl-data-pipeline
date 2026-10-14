"""
fl_park_kbb_neutral.py -- TrustmarkPark effect on K% and BB% for neutralising hitter history (user 2026-10-12, item 1a: the wOBA components must be
park-neutral on the same basis as the direct half; HR/PA is already neutralised with the HR factor, K% and BB% were not).

Design = fl_park_visitors_fe.park_effects, hitter side: outcome = year + hitter + pitcher + [hitter x home] + park, so the park effect is identified
from VISITING hitters (Mississippi hitters' extra home K% drop -- adaptation vs selection, unresolved -- is not attributed to the park).  Game-cluster
bootstrap (B = 100) for the SE; production shrink max(0, 1 - 1/t^2) (positive-part James-Stein, as for every other park term).
Alternative reported, not used: the all-PA fixed-effects estimate (output/park_outcome_fe_2025_26.csv), which mixes in Mississippi's home fit.
Writes output/trustmark_kbb_effects.json.
"""
import json
import warnings

import numpy as np
import pandas as pd

from fl_park_visitors_fe import load, park_effects

warnings.filterwarnings("ignore")
OUT = "output/trustmark_kbb_effects.json"


def main():
    pa = load(); res = {}
    rng = np.random.default_rng(12); games = pa["game_id"].to_numpy(); ug = pd.unique(games)
    by_game = pd.Series(np.arange(len(pa))).groupby(games).apply(np.array)
    est = {c: park_effects(pa, "hitter", c)["TrustmarkPark"] for c in ("is_k", "is_bb")}
    boots = {c: [] for c in est}
    for _ in range(100):
        ii = np.concatenate([by_game[g] for g in rng.choice(ug, len(ug))]); sub = pa.iloc[ii].reset_index(drop=True)
        for c in boots:
            boots[c].append(park_effects(sub, "hitter", c, iters=20).get("TrustmarkPark", np.nan))
    allpa = pd.read_csv("output/park_outcome_fe_2025_26.csv", index_col=0)["effect: TrustmarkPark"]
    for c, lab in (("is_k", "k"), ("is_bb", "bbr")):
        se = float(np.nanstd(boots[c])); t = est[c] / se; shrink = max(0.0, 1 - 1 / t ** 2)
        res[lab] = {"raw": float(est[c]), "se": se, "t": float(t), "shrink": shrink, "final": float(est[c] * shrink),
                    "all_pa_estimate": float(allpa["K%" if lab == "k" else "BB%"]), "league": float(pa[c].mean())}
        print(f"{lab:4s} raw {est[c]:+.4f}  SE {se:.4f}  t {t:+.2f}  shrink {shrink:.2f}  final {est[c] * shrink:+.4f}   (all-PA FE {res[lab]['all_pa_estimate']:+.4f}; league {res[lab]['league']:.4f})")
    res["design"] = "visiting hitters, year + hitter + pitcher + hitter x home + park FE, 2025-26; game-cluster bootstrap B=100; shrink max(0, 1-1/t^2)"
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
