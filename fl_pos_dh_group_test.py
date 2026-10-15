"""
fl_pos_dh_group_test.py -- model-change report: position adjustment from the 2026 table with DH as its own shrinkage group (POS_DH_GROUP = True) vs the hitter's
listed fielding position (False).  User 2026-10-14: "Use 2026 only for adjustments" / "DHs?".

WAR backtest -> 2026 (fl_hitter_pre2026.build_shifted, production path incl. the baserunning proxy), hitters with >= 20 PA in 2026; point WAR = projected PA x rate;
target = realised 2026 WAR (batting + published position runs + replacement + bsr_runs); realised-PA weights, hitter bootstrap (fl_eval_report.compare).
CAVEAT: the position table is 2026, the target season, so the backtest's position input is the realised 2026 role (accepted by the user); this favours LESS shrinkage, so
read the comparison as how the two shrinkage targets differ, not as an out-of-sample test of role.  Also printed: the backtest level with 2026 adjustments, to compare
with the earlier harness that used the broken 2025 adjustments (WAR RMSE 1.3476 / MAE 1.0983 with the baserunning proxy).
"""
import warnings

import numpy as np
import pandas as pd

import fl_bsr_proxy as bp
import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev
import fl_hitter_pre2026 as pre
import fl_hitter_projection as hp
from fl_eval_report import compare, table

warnings.filterwarnings("ignore")


def war_points(flag):
    hp.POS_DH_GROUP = flag; d, c, _ = pre.build_shifted(); d = d.set_index("name_clean")
    return d["proj_pa"] * d["rar_rate"] / c["rpw"], d, c


def main():
    old, d0, c = war_points(False); new, d1, _ = war_points(True); hp.POS_DH_GROUP = True
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; lg26 = float(bt.league(q, "woba")[2026])
    act = p[p["year"] == 2026].drop_duplicates("name_clean").set_index("name_clean")
    pt = bp.positions(2026)
    j = pd.DataFrame({"old": old, "new": new}).join(act[["pa", "woba_n"]].rename(columns={"pa": "pa26", "woba_n": "woba26"}), how="inner").join(pt[["pos_adj", "bsr", "pos"]])
    j = j[(j["pa26"] >= 20) & j["old"].notna()]
    y = (j["pa26"] * ((j["woba26"] - lg26) / hp.SC + c["repl"]) + j["pos_adj"].fillna(0) + j["bsr"].fillna(0)) / c["rpw"]
    rows = [{"comparison": "listed position group -> DH group", **compare(y.to_numpy(), j["old"].to_numpy(), j["new"].to_numpy(), j["pa26"].to_numpy(), j.index.to_numpy(), n=2000)}]
    print(f"{len(j)} hitters (DH by the 2026 rule: {(j['pos'] == 'DH').sum()})\n" + table(rows, 4))
    dh = j["pos"] == "DH"; print("DHs mean error (proj - actual) old -> new:", round(float((j.loc[dh, 'old'] - y[dh]).mean()), 3), "->", round(float((j.loc[dh, 'new'] - y[dh]).mean()), 3),
                                 "| mean pos_rate per 600 old -> new:", round(600 * float(d0.loc[j.index[dh], 'pos_rate'].mean()), 2), "->", round(600 * float(d1.loc[j.index[dh], 'pos_rate'].mean()), 2))


if __name__ == "__main__":
    main()
