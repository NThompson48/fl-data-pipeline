"""
fl_baserunning_position_height.py -- do baserunning runs differ by POSITION, and does player HEIGHT matter? (user 2026-10-14, after fixing hitter WAR = batting + position adjustment
+ replacement: "Can we check if baserunning runs typically differ by position also?" / "Or if player height matters for baserunning runs?")

Data:
  baserunning runs  2026 published bsr_runs (the 2025 column is all zeros = placeholder; 2022-24 have none) -> one season, 2026 only
  steal attempts    SB + CS, 2022-24 only (the panel has no SB / CS for 2025-26): a second, independent speed measure
  position          primary_pos (2025-26 only); for the 2022-24 steal check each player takes his latest 2025-26 primary position
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
    h = pd.read_parquet("output/fl_hitter_panel.parquet"); bi = bio()
    pos = h[h["year"].isin([2025, 2026]) & h["primary_pos"].isin(POS)].sort_values("year").groupby("name_clean")["primary_pos"].last()
    b = h[(h["year"] == 2026) & h["bsr_runs"].notna() & (h["pa"] > 0)].groupby("name_clean").agg(pa=("pa", "sum"), bsr=("bsr_runs", "sum"), pos=("primary_pos", "first")).join(bi)
    s = h[h["year"].between(2022, 2024) & h["sb"].notna() & (h["pa"] > 0)].assign(att=lambda q: q["sb"] + q["cs"]).groupby("name_clean").agg(pa=("pa", "sum"), att=("att", "sum")).join(bi)
    s["pos"] = pos.reindex(s.index)
    print(f"2026 baserunning hitters {len(b)}, with height {b['ht'].notna().sum()} ({b.loc[b['ht'].notna(), 'pa'].sum() / b['pa'].sum():.1%} of PA); "
          f"2022-24 steal hitters {len(s)}, with a 2025-26 position {s['pos'].notna().sum()}, with height {s['ht'].notna().sum()}")
    by_position(b[b["pos"].isin(POS)], "bsr", "2026 BASERUNNING RUNS by position, per 600 PA vs league (90% player bootstrap)")
    by_position(s[s["pos"].isin(POS)], "att", "2022-24 STEAL ATTEMPTS (SB + CS) by later position, per 600 PA vs league")
    by_height(b[b["ht"].notna()], "bsr", "2026 BASERUNNING RUNS vs HEIGHT", "runs")
    by_height(s[s["ht"].notna()], "att", "2022-24 STEAL ATTEMPTS vs HEIGHT", "attempts")


if __name__ == "__main__":
    main()
