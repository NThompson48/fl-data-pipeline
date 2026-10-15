"""
fl_baserunning_position_height.py -- do baserunning runs differ by POSITION, and does player HEIGHT matter? (user 2026-10-14, after fixing hitter WAR = batting + position adjustment
+ replacement: "Can we check if baserunning runs typically differ by position also?" / "Or if player height matters for baserunning runs?")

Data (user 2026-10-14: "use bsr_runs from the war_players table for this analysis. that is all I care about."):
  baserunning runs  war_players.bsr_runs, reg_season (populated for 2026 only; 2025 is null) with that table's pa and primary_pos; a player on two teams is
                    summed, his position = the row with more PA
  height / weight   pbp_players (height "6'1\"", weight, dob), falling back to rosters.height_in; age on 2026-07-01
Methods:
  position: PA-weighted runs per 600 PA vs league with a 90% player bootstrap; 'no position differences' = permutation of position labels
            (statistic: PA-weighted between-position variance of the rate)
  height:   weighted least squares of runs per 600 PA on height (weights = PA, the inverse-variance weighting for a per-PA rate); slope per inch with a 90% player
            bootstrap; permutation p (shuffle heights) for the height-only model; then height adjusted for weight + age, for position, and for both
"""
import re

import numpy as np
import pandas as pd
from sqlalchemy import text

import fl_hitter_bridge_test as hb
from fl_supabase import read_sql_df

POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
rng = np.random.default_rng(0)


def inches(s):
    m = re.match(r"\s*(\d+)'\s*(\d+)", str(s)); return float(m.group(1)) * 12 + float(m.group(2)) if m else np.nan


def bio():
    p = read_sql_df(text("select name, height, weight, dob from pbp_players")); p["name_clean"] = p["name"].map(hb.norm)
    p["ht"] = p["height"].map(inches); p["wt"] = pd.to_numeric(p["weight"], errors="coerce")
    p["age"] = (pd.Timestamp("2026-07-01") - pd.to_datetime(p["dob"], errors="coerce")).dt.days / 365.25
    p = p.groupby("name_clean")[["ht", "wt", "age"]].first()
    r = read_sql_df(text("select name, height_in, weight from rosters")); r["name_clean"] = r["name"].map(lambda s: hb.norm(" ".join(reversed(str(s).split(", ")))))
    r = r.groupby("name_clean")[["height_in", "weight"]].first().rename(columns={"height_in": "ht", "weight": "wt"})
    return p.combine_first(r)


def by_position(d, col, label, B=2000, P=5000):
    lg = d[col].sum() / d["pa"].sum(); rows = {}
    for pos, q in d.groupby("pos"):
        x, n = q[col].to_numpy(), q["pa"].to_numpy(); bs = [x[i].sum() / n[i].sum() for i in (rng.integers(0, len(q), len(q)) for _ in range(B))]
        rows[pos] = dict(players=len(q), pa=int(n.sum()), vs_lg=600 * (x.sum() / n.sum() - lg), lo=600 * (np.percentile(bs, 5) - lg), hi=600 * (np.percentile(bs, 95) - lg))
    T = pd.DataFrame(rows).T.reindex([p for p in POS if p in rows])

    def stat(pos):
        g = pd.DataFrame({"p": pos, "x": d[col].to_numpy(), "n": d["pa"].to_numpy()}).groupby("p").sum(); rr = g["x"] / g["n"]; return float((g["n"] * (rr - lg) ** 2).sum() / g["n"].sum())
    obs = stat(d["pos"].to_numpy()); pv = (1 + sum(stat(rng.permutation(d["pos"].to_numpy())) >= obs for _ in range(P))) / (P + 1)
    print(f"\n== {label}: league {600 * lg:.2f} per 600 PA; permutation p (no position differences) = {pv:.4f}; {len(d)} players\n" + T.round(2).to_string())


def wls(X, y, w):
    W = np.sqrt(w); return np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]


def design(d, extra):
    cols = [np.ones(len(d)), d["ht"].to_numpy() - 72]
    if "wa" in extra:
        cols += [d["wt"].to_numpy() - d["wt"].mean(), d["age"].to_numpy() - d["age"].mean()]
    if "pos" in extra:
        cols += [(d["pos"] == p).astype(float).to_numpy() for p in POS[1:]]
    return np.column_stack(cols)


def by_height(d, col, label, unit, B=2000, P=5000):
    y = 600 * d[col].to_numpy() / d["pa"].to_numpy(); w = d["pa"].to_numpy().astype(float)
    print(f"\n== {label}: {len(d)} players, height {d['ht'].min():.0f}-{d['ht'].max():.0f} in (PA-weighted mean {np.average(d['ht'], weights=w):.1f})")
    bins = pd.cut(d["ht"], [0, 70, 72, 74, 99], labels=["<=5'10", "5'11-6'0", "6'1-6'2", "6'3+"])
    t = pd.DataFrame({"bin": bins, "x": d[col], "n": d["pa"]}).groupby("bin", observed=True).agg(players=("x", "size"), x=("x", "sum"), n=("n", "sum"))
    t["per600_vs_lg"] = 600 * (t["x"] / t["n"] - d[col].sum() / d["pa"].sum()); print(t[["players", "per600_vs_lg"]].round(2).to_string())
    for extra, nm in ((set(), "height only"), ({"wa"}, "+ weight, age"), ({"pos"}, "+ position"), ({"wa", "pos"}, "+ weight, age, position")):
        dd = d.dropna(subset=["wt", "age"]) if "wa" in extra else d; dd = dd[dd["pos"].notna()] if "pos" in extra else dd
        yy, ww = 600 * dd[col].to_numpy() / dd["pa"].to_numpy(), dd["pa"].to_numpy().astype(float); X = design(dd, extra)
        bb = wls(X, yy, ww); b = bb[1]; boots = [wls(X[i], yy[i], ww[i]) for i in (rng.integers(0, len(dd), len(dd)) for _ in range(B))]; bs = [q[1] for q in boots]
        line = f"  {nm:<26} height {b:+.3f} {unit} per 600 PA per inch  (90% {np.percentile(bs, 5):+.3f} to {np.percentile(bs, 95):+.3f}; {len(dd)} players)"
        if "wa" in extra:
            wq = [10 * q[2] for q in boots]; line += f"  | weight {10 * bb[2]:+.3f} per 10 lb (90% {np.percentile(wq, 5):+.3f} to {np.percentile(wq, 95):+.3f})"
        if not extra:
            pv = (1 + sum(abs(wls(design(dd.assign(ht=rng.permutation(dd["ht"].to_numpy())), extra), yy, ww)[1]) >= abs(b) for _ in range(P))) / (P + 1); line += f"  permutation p {pv:.4f}"
        print(line)


def main():
    w = read_sql_df(text("select player, pa, bsr_runs, primary_pos from war_players where game_type = 'reg_season' and bsr_runs is not null and pa > 0"))
    w["name_clean"] = w["player"].map(lambda x: hb.norm(" ".join(reversed(str(x).split(", "))) if ", " in str(x) else x))
    pos = w.sort_values("pa").groupby("name_clean")["primary_pos"].last()
    b = w.groupby("name_clean").agg(pa=("pa", "sum"), bsr=("bsr_runs", "sum")).join(pos.rename("pos")).join(bio())
    print(f"war_players 2026 reg_season: {len(b)} hitters, {b['pa'].sum():.0f} PA, bsr_runs sum {b['bsr'].sum():+.2f}; positions {b['pos'].value_counts(dropna=False).to_dict()}; "
          f"with height {b['ht'].notna().sum()} ({b.loc[b['ht'].notna(), 'pa'].sum() / b['pa'].sum():.1%} of PA)")
    by_position(b[b["pos"].isin(POS)], "bsr", "war_players bsr_runs by position, per 600 PA vs league (90% player bootstrap)")
    by_height(b[b["ht"].notna()], "bsr", "war_players bsr_runs vs HEIGHT", "runs")


if __name__ == "__main__":
    main()
