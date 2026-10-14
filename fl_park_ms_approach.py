"""
fl_park_ms_approach.py -- what do the Mud Monsters do differently at home, and where does their home edge come from?  (user 2026-10-09: "how to best play at our park")
Same ball-level residual as fl_park_play_the_park.py (wOBA value minus a park-blind, cross-fitted expectation from exit speed, launch angle, field direction, batter side).
HITTERS: Mississippi hitters at home vs on the road, and (as the no-adaptation baseline) every OTHER team's hitters at Trustmark vs the same hitters elsewhere.
PITCHERS: Mississippi pitchers at home vs on the road, vs other teams' pitchers at Trustmark vs elsewhere.
Reported: (1) batted-ball MIX (type x spray shares, exit speed, launch angle, pull degrees) -- did they change what they put in play?  (2) the RESIDUAL wOBA per ball by type -- did the same type of ball work better?  Differences-in-differences with game-cluster bootstrap SEs.
Note: the balls-in-play residual is only part of the home edge (strikeouts / walks are outside it).
"""
import warnings

import numpy as np
import pandas as pd

import fl_park_play_the_park as pp

warnings.filterwarnings("ignore")
MS = "Mississippi Mud Monsters"


def did(b, bt, v, grp_mask, tm):
    """(group at Trustmark - group elsewhere) for the Mississippi group minus the same for the baseline group, per mean of v."""
    out = {}
    for lab, m in grp_mask.items():
        out[lab] = m
    return out


def boot_stat(bt, fn):
    vals = [fn(w) for w in bt.W]; return float(np.std(vals))


def wmean(v, w, m):
    ww = w * m; return (ww * v).sum() / ww.sum() if ww.sum() > 0 else np.nan


def side_report(b, bt, side):
    tm = (b["tm"] == 1).to_numpy()
    if side == "hitters":
        ms = (b["batter_team"] == MS).to_numpy(); own_home = (b["batter_team"] == b["home_team"]).to_numpy(); ms_at = ms & tm & own_home; ms_away = ms & ~own_home; oth = ~ms
        base_at = oth & tm & ~own_home; base_else = oth & ~tm & ~own_home; base_else = base_else & b["batter"].isin(b.loc[base_at, "batter"]).to_numpy()
    else:
        ms = (b["pitcher_team"] == MS).to_numpy(); own_home = (b["pitcher_team"] == b["home_team"]).to_numpy(); ms_at = ms & tm & own_home; ms_away = ms & ~own_home; oth = ~ms
        base_at = oth & tm & ~own_home; base_else = oth & ~tm & ~own_home; base_else = base_else & b["pitcher_id"].isin(b.loc[base_at, "pitcher_id"]).to_numpy()
    print(f"\n=== {side.upper()}: Mississippi home balls {int(ms_at.sum()):,}, road {int(ms_away.sum()):,}; other teams at Trustmark {int(base_at.sum()):,}, the same people elsewhere {int(base_else.sum()):,}")
    res = b["res_woba"].to_numpy(); rows = []
    cells = [("all balls", np.ones(len(b), bool))] + [(t[2], (b["type"] == t[2]).to_numpy()) for t in pp.TYPES]
    for lab, c in cells:
        def stat(w):
            return (wmean(res, w, ms_at & c) - wmean(res, w, ms_away & c)) - (wmean(res, w, base_at & c) - wmean(res, w, base_else & c))
        one = np.ones(len(b)); est = stat(one); se = boot_stat(bt, stat)
        rows.append({"balls": lab, "MS home": res[ms_at & c].mean(), "MS road": res[ms_away & c].mean(), "MS home-road": res[ms_at & c].mean() - res[ms_away & c].mean(), "others TM - elsewhere": res[base_at & c].mean() - res[base_else & c].mean(), "DiD (MS edge beyond the park)": est, "SE": se, "t": est / se})
    print("(2) RESIDUAL wOBA PER BALL BY TYPE (park-blind expectation)\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    rows = []
    for lab, col, fn in [("share ground", None, lambda d: (d["type"] == "ground")), ("share line drive", None, lambda d: (d["type"] == "line drive")), ("share fly ball", None, lambda d: (d["type"] == "fly ball")), ("share pop-up", None, lambda d: (d["type"] == "pop-up")),
                         ("share pulled", None, lambda d: (d["spray"] == "pull")), ("share oppo", None, lambda d: (d["spray"] == "oppo")), ("exit speed (mph)", "exit_speed", None), ("launch angle (deg)", "angle", None)]:
        v = (fn(b).to_numpy(float) if fn is not None else b[col].to_numpy(float))
        def stat(w, v=v):
            return (wmean(v, w, ms_at) - wmean(v, w, ms_away)) - (wmean(v, w, base_at) - wmean(v, w, base_else))
        one = np.ones(len(b)); est = stat(one); se = boot_stat(bt, stat)
        rows.append({"mix of balls in play": lab, "MS home": v[ms_at].mean(), "MS road": v[ms_away].mean(), "MS home-road": v[ms_at].mean() - v[ms_away].mean(), "others TM - elsewhere": v[base_at].mean() - v[base_else].mean(), "DiD": est, "SE": se, "t": est / se})
    print("(1) WHAT THEY PUT IN PLAY\n" + pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    b = pp.load(); b["exp_woba"] = pp.cross_fit(b, "woba"); b["res_woba"] = b["woba"] - b["exp_woba"]; bt = pp.Boot(b["game_id"].to_numpy(), B=300)
    side_report(b, bt, "hitters"); side_report(b, bt, "pitchers")


if __name__ == "__main__":
    main()
