"""
fl_park_ms_kbb.py -- where does the Mud Monsters' home edge come from if not from balls in play?  (follows fl_park_ms_approach.py)
PA-level difference-in-differences, 2025-26: Mississippi hitters (home - road) minus other teams' hitters (at Trustmark - the same hitters elsewhere); same for Mississippi pitchers.  Outcomes per PA: K, BB, HR, ball in play, wOBA.  Game-cluster bootstrap.
"""
import warnings

import numpy as np
import pandas as pd

from fl_park_visitors_fe import load

warnings.filterwarnings("ignore")
MS = "Mississippi Mud Monsters"


def main():
    pa = load(); pa["bip_flag"] = pa["kind"].astype(str).isin(["single", "double", "triple", "out", "fielders_choice", "error", "sacrifice"]).astype(float); pa["tm"] = pa["tm"].astype(bool)
    codes, uniq = pd.factorize(pa["game_id"]); rng = np.random.default_rng(4); W = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes] for _ in range(300)]
    outs = {"K%": "is_k", "BB%": "is_bb", "HR/PA": "is_hr", "ball in play/PA": "bip_flag", "wOBA": "woba"}
    for side in ("hitters", "pitchers"):
        if side == "hitters":
            team = pa["batter_team"]; own_home = (pa["batter_team"] == pa["home_team"]); who = pa["name_clean"]
        else:
            team = pa["pitcher_team"]; own_home = (pa["pitcher_team"] == pa["home_team"]); who = pa["pid"]
        ms = (team == MS).to_numpy(); oh = own_home.to_numpy(); tm = pa["tm"].to_numpy(); ms_at = ms & tm & oh; ms_away = ms & ~oh; oth = ~ms; base_at = oth & tm & ~oh; base_else = oth & ~tm & ~oh & who.isin(who[base_at]).to_numpy()
        rows = []
        for lab, col in outs.items():
            v = pa[col].to_numpy(float)
            def f(w):
                m = lambda mask: (w * mask * np.nan_to_num(v)).sum() / (w * mask * ~np.isnan(v)).sum()
                return (m(ms_at) - m(ms_away)) - (m(base_at) - m(base_else))
            one = np.ones(len(pa)); est = f(one); se = float(np.std([f(w) for w in W]))
            rows.append({"outcome": lab, "MS home": np.nanmean(v[ms_at]), "MS road": np.nanmean(v[ms_away]), "MS home-road": np.nanmean(v[ms_at]) - np.nanmean(v[ms_away]), "others TM - elsewhere": np.nanmean(v[base_at]) - np.nanmean(v[base_else]), "DiD (MS edge beyond the park)": est, "SE": se, "t": est / se})
        print(f"\n=== {side.upper()} (PA: MS home {int(ms_at.sum()):,}, road {int(ms_away.sum()):,}; others at TM {int(base_at.sum()):,})\n" + pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
