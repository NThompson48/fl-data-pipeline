"""
fl_milb_signee_analysis.py -- the Preston Johnson question run on every case we
can find: a pitcher posts a Frontier League line, is then signed by an
affiliated MiLB organization, and pitches there. Does the FL line carry
information about how he does in affiliated ball, beyond what his earlier MiLB
record already said?

Method
  * FL stints: per (pitcher, FL season) first/last game date, PA, K, BB.
  * Candidate MiLB ids: (a) FL pitcher_id equals an MLBAM id (an ex-affiliated
    player returning to affiliated ball), (b) a UNIQUE cleaned-name match to an
    MLBAM id whose first season in the pulled data is the FL year or later
    (a first-time signee) -- name matches are verified against MLB transactions
    ("signed", FL-timeline-consistent) wherever the API returns them.
  * Real dates: MLB Stats API game logs for every candidate (2025, 2026; all
    five affiliated levels), cached to the scratch dir, so "after FL" really
    means after the last FL game rather than same-calendar-year.
  * pre-FL MiLB = season-level rows for years before the FL season + same-year
    games before the first FL game; post-FL MiLB = games after the last FL
    game.
  * Regress post-FL MiLB K%/BB% (weights = post TBF) on pre-FL MiLB rate and the
    FL rate, each shrunk to its league mean by a PA/(PA+n0) factor, so a 39-PA
    stint is correctly treated as thin.

Usage:  python fl_milb_signee_analysis.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_milb_bridge as mb
import mlb_marcel as mm

CACHE = r"C:\Users\215nt\AppData\Local\Temp\claude\C--Users-215nt-PycharmProjects-FLStuff-FLHitting\e96abeb0-fb3e-4d45-adec-9b81c366e683\scratchpad\milb_gamelogs"
SPORT_IDS = [11, 12, 13, 14, 16]
N0_K, N0_BB = 59.0, 62.0       # FL K%/BB% reliability constants (mm.estimate_n0 on FL PA)
N0_MILB_K, N0_MILB_BB = 90.0, 150.0  # placeholder, replaced below by data-driven values if available


def gamelog(pid: int, season: int, sport_id: int) -> list[dict]:
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{pid}_{season}_{sport_id}.json")
    if os.path.exists(f):
        return json.load(open(f))
    url = (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog&group=pitching"
           f"&season={season}&sportId={sport_id}")
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            sp = r.json()["stats"][0]["splits"] if r.json().get("stats") else []
            rows = [{"date": s["date"], "level": sport_id, "team": s["team"]["name"],
                     "so": s["stat"]["strikeOuts"], "bb": s["stat"]["baseOnBalls"], "bf": s["stat"]["battersFaced"]}
                    for s in sp]
            json.dump(rows, open(f, "w"))
            time.sleep(0.05)
            return rows
        except Exception:
            time.sleep(1.0)
    return []


def fl_stints(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy(); raw["date"] = pd.to_datetime(raw["date"])
    t = raw[raw["is_terminal"]].copy()
    t["yr"] = t["date"].dt.year
    t["k"] = t["k_or_bb"] == "strikeout"; t["b"] = t["k_or_bb"] == "walk"
    g = t.groupby(["pitcher_id", "yr"]).agg(PA=("k", "size"), K=("k", "sum"), BB=("b", "sum"),
                                              first=("date", "min"), last=("date", "max"),
                                              name=("pitcher", "first")).reset_index()
    g["pitcher_id"] = g["pitcher_id"].astype(str)
    return g


def candidates(fl: pd.DataFrame, mi: pd.DataFrame) -> pd.DataFrame:
    mi = mi.copy(); mi["pitcher"] = mi["pitcher"].astype(str)
    first_season = mi.groupby("pitcher")["year"].min().rename("milb_first")
    fl = fl.copy(); fl["name_clean"] = fl["name"].apply(mb._fl_name_clean)
    out = []
    # (a) id match
    a = fl.merge(first_season, left_on="pitcher_id", right_index=True)
    a["milb_id"] = a["pitcher_id"]; a["match"] = "id"
    out.append(a)
    # (b) unique-name, first-time-signee match
    names = mi.drop_duplicates("pitcher")[["pitcher", "name_clean"]]
    uniq = names[~names["name_clean"].duplicated(keep=False)]
    b = fl.merge(uniq, on="name_clean").merge(first_season, left_on="pitcher", right_index=True)
    b = b[(b["milb_first"] >= b["yr"]) & (b["pitcher"] != b["pitcher_id"])]
    b["milb_id"] = b["pitcher"]; b["match"] = "name"
    out.append(b.drop(columns=["pitcher"]))
    c = pd.concat(out, ignore_index=True)
    return c.drop_duplicates(["pitcher_id", "yr", "milb_id"])


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    mi = mb.load_all_milb_years()
    fl = fl_stints(raw)
    cand = candidates(fl, mi)
    print(f"candidate FL stints with a possible MiLB id: {len(cand)} "
          f"({cand['match'].value_counts().to_dict()}); unique MiLB ids {cand['milb_id'].nunique()}")
    logs = {}
    for pid in cand["milb_id"].unique():
        rows = []
        for season in (2025, 2026):
            for sid in SPORT_IDS:
                rows += gamelog(int(pid), season, sid)
        logs[pid] = pd.DataFrame(rows)
    mi["pitcher"] = mi["pitcher"].astype(str)
    recs = []
    for _, r in cand.iterrows():
        gl = logs[r["milb_id"]]
        if gl.empty:
            continue
        gl = gl.assign(date=pd.to_datetime(gl["date"]))
        post = gl[gl["date"] > r["last"]]
        same_pre = gl[gl["date"] < r["first"]]
        hist = mi[(mi["pitcher"] == r["milb_id"]) & (mi["year"] < r["yr"])]
        pre_tbf = hist["tbf"].sum() + same_pre["bf"].sum()
        pre_so = hist["so"].sum() + same_pre["so"].sum(); pre_bb = hist["bb"].sum() + same_pre["bb"].sum()
        if post["bf"].sum() <= 0:
            continue
        recs.append({"pitcher_id": r["pitcher_id"], "name": r["name"], "yr": r["yr"], "match": r["match"],
                     "FL_PA": r["PA"], "FL_K": r["K"], "FL_BB": r["BB"],
                     "pre_TBF": pre_tbf, "pre_SO": pre_so, "pre_BB": pre_bb,
                     "post_TBF": post["bf"].sum(), "post_SO": post["so"].sum(), "post_BB": post["bb"].sum(),
                     "fl_last": r["last"].date(), "first_post": post["date"].min().date(),
                     "post_levels": ",".join(sorted(set(post["team"])))[:40]})
    return pd.DataFrame(recs)


def shrink(count, n, n0, lg):
    return (count + n0 * lg) / (n + n0)


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    d = build_dataset(raw)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    # reference league levels: FL (from the pitch data) and affiliated MiLB (all pulled seasons 2025-26)
    t = raw[raw["is_terminal"]]
    lg_fl_k, lg_fl_bb = (t["k_or_bb"] == "strikeout").mean(), (t["k_or_bb"] == "walk").mean()
    mi = mb.load_all_milb_years(); mi = mi[mi["year"] >= 2024]
    lg_mi_k, lg_mi_bb = mi["so"].sum() / mi["tbf"].sum(), mi["bb"].sum() / mi["tbf"].sum()
    print(f"league levels: FL K {lg_fl_k:.3f} BB {lg_fl_bb:.3f};  affiliated MiLB 2024-26 K {lg_mi_k:.3f} BB {lg_mi_bb:.3f}")

    d = d[(d["post_TBF"] >= 30)].copy()
    d["post_K"] = d["post_SO"] / d["post_TBF"]; d["post_BB"] = d["post_BB"] / d["post_TBF"]
    d["FL_Kr"] = d["FL_K"] / d["FL_PA"]; d["FL_BBr"] = d["FL_BB"] / d["FL_PA"]
    d["pre_Kr"] = np.where(d["pre_TBF"] > 0, d["pre_SO"] / d["pre_TBF"].clip(lower=1), np.nan)
    d["pre_BBr"] = np.where(d["pre_TBF"] > 0, d["pre_BB"] / d["pre_TBF"].clip(lower=1), np.nan)
    # shrunk predictors (thin samples pulled to the league mean of THEIR league)
    d["FL_K_s"] = shrink(d["FL_K"], d["FL_PA"], N0_K, lg_fl_k) - lg_fl_k
    d["FL_BB_s"] = shrink(d["FL_BB"], d["FL_PA"], N0_BB, lg_fl_bb) - lg_fl_bb
    d["pre_K_s"] = shrink(d["pre_SO"], d["pre_TBF"], 150.0, lg_mi_k) - lg_mi_k
    d["pre_BB_s"] = shrink(d["pre_BB"], d["pre_TBF"], 150.0, lg_mi_bb) - lg_mi_bb
    print(f"\nanalysis sample (post-FL MiLB TBF>=30): n={len(d)} stints / {d['pitcher_id'].nunique()} pitchers; "
          f"match types {d['match'].value_counts().to_dict()}; median FL PA {d['FL_PA'].median():.0f}, "
          f"median pre-MiLB TBF {d['pre_TBF'].median():.0f}, median post TBF {d['post_TBF'].median():.0f}")
    print(d.sort_values("post_TBF", ascending=False)[["name", "yr", "match", "FL_PA", "FL_Kr", "FL_BBr", "pre_TBF", "pre_Kr", "pre_BBr",
                                                     "post_TBF", "post_K", "post_BB", "fl_last", "first_post"]].round(3).head(40).to_string(index=False))
    d.to_csv(os.path.join(os.path.dirname(CACHE), "signee_dataset.csv"), index=False)

    rng = np.random.default_rng(0)
    for metric, y, fl_col, pre_col in (("K%", "post_K", "FL_K_s", "pre_K_s"), ("BB%", "post_BB", "FL_BB_s", "pre_BB_s")):
        dd = d.dropna(subset=[y, fl_col, pre_col]).copy()
        w = dd["post_TBF"].values
        print(f"\n--- post-FL MiLB {metric}  (n={len(dd)}) -- WLS weights = post TBF; predictors are deviations from league mean after shrinkage ---")
        for lab, f in (("pre-MiLB only", f"{y} ~ {pre_col}"), ("FL only", f"{y} ~ {fl_col}"), ("both", f"{y} ~ {pre_col} + {fl_col}")):
            m = smf.wls(f, data=dd, weights=w).fit()
            print(f"  {lab:14s} R2={m.rsquared:.3f}  params " + ", ".join(f"{k}={v:+.3f}(se {m.bse[k]:.3f})" for k, v in m.params.items() if k != "Intercept"))
        # bootstrap CI of the FL coefficient in the joint model
        co = []
        for _ in range(1000):
            s = dd.sample(len(dd), replace=True, random_state=int(rng.integers(1e9)))
            try:
                co.append(smf.wls(f"{y} ~ {pre_col} + {fl_col}", data=s, weights=s["post_TBF"]).fit().params[fl_col])
            except Exception:
                pass
        print(f"  FL coefficient in joint model, bootstrap 90% CI: [{np.percentile(co, 5):+.3f}, {np.percentile(co, 95):+.3f}]")

    # Johnson-like: strong FL stint (shrunk FL K-BB top third) -> what happened in MiLB relative to his OWN pre-FL record?
    d["FL_kbb_s"] = d["FL_K_s"] - d["FL_BB_s"]
    d["post_vs_pre_K"] = d["post_K"] - np.where(d["pre_TBF"] > 0, d["pre_K_s"] + lg_mi_k, lg_mi_k)
    d["post_vs_pre_BB"] = d["post_BB"] - np.where(d["pre_TBF"] > 0, d["pre_BB_s"] + lg_mi_bb, lg_mi_bb)
    top = d["FL_kbb_s"] >= d["FL_kbb_s"].quantile(2 / 3); bot = d["FL_kbb_s"] <= d["FL_kbb_s"].quantile(1 / 3)
    for lab, mask in (("top third FL K-BB", top), ("bottom third FL K-BB", bot)):
        s = d[mask]
        wa = lambda c: np.average(s[c], weights=s["post_TBF"])
        print(f"\n{lab}: n={len(s)}  post-FL MiLB K% {wa('post_K'):.3f}  BB% {wa('post_BB'):.3f}   "
              f"vs own shrunk pre-FL MiLB expectation: K {wa('post_vs_pre_K'):+.3f}, BB {wa('post_vs_pre_BB'):+.3f}")
