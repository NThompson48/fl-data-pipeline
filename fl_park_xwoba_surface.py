"""
fl_park_xwoba_surface.py -- overall batted-ball performance at Trustmark vs xwOBA, across exit-speed x launch-angle cells  (user 2026-10-10)
Every tracked ball in play (HR included, 80.4k with a wOBA value).  xwOBA = the tracking data's expected wOBA from exit speed / launch angle (calibrated: league mean .386 vs actual .389).  residual = wOBA - xwOBA = what the park (and fielders) do to a ball of that quality.
1  SURFACE: for each EV x LA cell, balls at Trustmark, mean xwOBA, mean wOBA and residual at Trustmark vs the other parks, Trustmark-minus-other residual with a game-cluster bootstrap SE (B = 300).  Marginals by launch-angle band and by EV band.
2  WITHIN-HITTER DECOMPOSITION for visiting hitters (hitter + pitcher + year + [hitter x home] fixed effects, game-cluster SEs): the Trustmark effect on wOBA on contact = effect on xwOBA (contact QUALITY the hitter produces there) + effect on the residual (what the PARK does to that quality),
   overall, by batter side, by launch-angle band, by EV band.
"""
import warnings

import numpy as np
import pandas as pd

import fl_park_play_the_park as pp
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
TM = "TrustmarkPark"
EVB = [(0, 80, "< 80"), (80, 90, "80-90"), (90, 95, "90-95"), (95, 100, "95-100"), (100, 105, "100-105"), (105, 130, "105+")]
LAB = [(-91, 0, "< 0"), (0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 40, "30-40"), (40, 91, "> 40")]


def load():
    d = pd.read_parquet("2025_2026_pitches.parquet", columns=["game_id", "year", "stadium", "home_team", "batter_team", "batter_id", "pitcher_id", "is_terminal_pitch", "exit_speed", "angle", "xwoba", "woba", "play_result", "batter_side"])
    b = d[d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]) & d["exit_speed"].notna() & d["angle"].notna() & d["woba"].notna() & d["xwoba"].notna() & d["play_result"].isin(["out", "single", "double", "triple", "error", "fielders_choice", "sacrifice", "home_run"]) & d["batter_side"].isin(["left", "right"])].copy().reset_index(drop=True)
    b["tm"] = (b["stadium"] == TM).astype(int); b["res"] = b["woba"] - b["xwoba"]; b["vis"] = (b["batter_team"] != b["home_team"]).astype(int); b["left"] = (b["batter_side"] == "left").astype(int); return b


def main():
    b = load(); tm = (b["tm"] == 1).to_numpy(); print(f"{len(b):,} balls in play; at Trustmark {int(tm.sum()):,}; mean wOBA {b['woba'].mean():.4f}, xwOBA {b['xwoba'].mean():.4f}")
    print(f"Trustmark: wOBA {b.loc[tm, 'woba'].mean():.4f}, xwOBA {b.loc[tm, 'xwoba'].mean():.4f}, residual {b.loc[tm, 'res'].mean():+.4f} | other parks: wOBA {b.loc[~tm, 'woba'].mean():.4f}, xwOBA {b.loc[~tm, 'xwoba'].mean():.4f}, residual {b.loc[~tm, 'res'].mean():+.4f}")
    bt = pp.Boot(b["game_id"].to_numpy(), B=300); res = b["res"].to_numpy(); rows = []
    for lo, hi, ll in LAB:
        row = {"launch angle": ll}
        for elo, ehi, el in EVB:
            m = ((b["angle"] >= lo) & (b["angle"] < hi) & (b["exit_speed"] >= elo) & (b["exit_speed"] < ehi)).to_numpy(); n_t = int((m & tm).sum())
            if n_t >= 15:
                e, s = bt.diff(res, tm, m); row[el] = f"{e:+.3f} ({s:.3f}) n{n_t}"
            else:
                row[el] = f"n{n_t}"
        rows.append(row)
    print("\n1a. RESIDUAL (wOBA - xwOBA), TRUSTMARK MINUS OTHER PARKS, by launch angle (rows) x exit speed (columns, mph); (SE) and Trustmark ball count\n" + pd.DataFrame(rows).to_string(index=False))
    rows = []
    for lo, hi, ll in LAB:
        m = ((b["angle"] >= lo) & (b["angle"] < hi)).to_numpy(); e, s = bt.diff(res, tm, m); ex, sx = bt.diff(b["xwoba"].to_numpy(), tm, m); rows.append({"launch angle": ll, "TM balls": int((m & tm).sum()), "TM xwOBA": b.loc[m & tm, "xwoba"].mean(), "TM wOBA": b.loc[m & tm, "woba"].mean(), "residual TM - other": e, "SE": s, "t": e / s, "xwOBA TM - other (contact quality)": ex})
    print("\n1b. BY LAUNCH-ANGLE BAND\n" + pd.DataFrame(rows).round(3).to_string(index=False)); rows = []
    for elo, ehi, el in EVB:
        m = ((b["exit_speed"] >= elo) & (b["exit_speed"] < ehi)).to_numpy(); e, s = bt.diff(res, tm, m); ex, sx = bt.diff(b["xwoba"].to_numpy(), tm, m); rows.append({"exit speed": el, "TM balls": int((m & tm).sum()), "TM xwOBA": b.loc[m & tm, "xwoba"].mean(), "TM wOBA": b.loc[m & tm, "woba"].mean(), "residual TM - other": e, "SE": s, "t": e / s, "xwOBA TM - other": ex})
    print("\n1c. BY EXIT-SPEED BAND\n" + pd.DataFrame(rows).round(3).to_string(index=False)); rows = []
    for side in (0, 1):
        m = (b["left"] == side).to_numpy(); e, s = bt.diff(res, tm, m); ex, sx = bt.diff(b["xwoba"].to_numpy(), tm, m); rows.append({"batter": "LHH" if side else "RHH", "TM balls": int((m & tm).sum()), "TM xwOBA": b.loc[m & tm, "xwoba"].mean(), "TM wOBA": b.loc[m & tm, "woba"].mean(), "residual TM - other": e, "SE": s, "t": e / s, "xwOBA TM - other": ex})
    print("\n1d. BY BATTER SIDE (all hitters)\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    # 2 within-hitter, visiting hitters
    V = b[b["vis"] == 1].reset_index(drop=True); bid = V["batter_id"].astype(str); pid = V["pitcher_id"].astype(object).where(V["pitcher_id"].notna(), "unk").astype(str)
    g = [pd.factorize(V["year"].astype(str))[0], pd.factorize(bid)[0], pd.factorize(pid)[0]]; cl = pd.factorize(V["game_id"])[0]; tmv = V["tm"].to_numpy(float)

    def eff(mask, label):
        sub = np.where(mask)[0]; Vs = V.iloc[sub]; gs = [pd.factorize(x[sub])[0] for x in g]; t = tmv[sub][:, None]; out = {"group": label, "TM balls": int(tmv[sub].sum())}; tr = partial_out(t, gs, iters=30)[:, 0]
        for k, c in (("wOBAcon", "woba"), ("xwOBA (quality)", "xwoba"), ("residual (park)", "res")):
            yr = partial_out(Vs[c].to_numpy(float)[:, None], gs, iters=30)[:, 0]; bb, se = cluster_ols(tr[:, None], yr, cl[sub]); out[k] = f"{bb[0]:+.4f} (t {bb[0] / se[0]:+.1f})"
        return out
    rows = [eff(np.ones(len(V), bool), "all visiting hitters"), eff((V["left"] == 1).to_numpy(), "LHH"), eff((V["left"] == 0).to_numpy(), "RHH")]
    for lo, hi, ll in LAB:
        rows.append(eff(((V["angle"] >= lo) & (V["angle"] < hi)).to_numpy(), f"launch angle {ll}"))
    for elo, ehi, el in EVB:
        rows.append(eff(((V["exit_speed"] >= elo) & (V["exit_speed"] < ehi)).to_numpy(), f"exit speed {el}"))
    print("\n2. WITHIN-HITTER (visiting hitters, hitter + pitcher + year FE): TRUSTMARK EFFECT per ball in play = contact quality (xwOBA) + park effect on that quality (residual)\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
