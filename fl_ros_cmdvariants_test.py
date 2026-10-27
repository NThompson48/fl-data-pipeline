"""
fl_ros_cmdvariants_test.py -- do the better-fitting all-pitch Command Score variants hold up in the ROS (in-season) BB% power prior?  (user 2026-10-27: "if the new all-pitch command measure is better ... with roughly the same performance in the model, I would add it")
Same design as fl_pitcher_ros_stuff.py / fl_ros_fbcmd_test.py: 5 cutoffs (15-75% of the season), pool = last season + this season before the cutoff, each measure shrunk by its own random-half reliability (n0 in pitches), B fit on one season's ROS rows and frozen, scored on the other (FORWARD 2025 -> 2026, REVERSE 2026 -> 2025),
pitcher-cluster two-stage bootstrap, PAIRED difference vs the production all-pitch score on the same draws.
Variants (frozen MLB-OpenCommand fits of fl_command_variants.py): v1 cond3, v2 cond12 shrunk, v4 target-cluster, v7 cond12 shrunk + cluster.
Output: output/ros_cmdvariants_test.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_command_variants as cvr
import fl_pitcher_features as fp
import fl_pitcher_ros as pr
import fl_pitcher_ros_stuff as rs
import fl_ros_fbcmd_test as fbt

warnings.filterwarnings("ignore")
rng = np.random.default_rng(406)
LAB = {"v1": "v1 cond3", "v2": "v2 cond12 shrunk", "v4": "v4 target-cluster", "v7": "v7 cond12 shrunk + cluster"}; REPS = 4


def prep(d):
    d = d[d["batter_side"].isin(["left", "right"]) & d["pitcher_throws"].isin(["left", "right"])].dropna(subset=["plate_x_as", "plate_z", "pitch_type", "pitcher_id", "balls", "strikes"])
    return d.assign(pid=d["pitcher_id"], x=d["plate_x_as"], z=d["plate_z"], same=(d["batter_side"] == d["pitcher_throws"]).astype(int))[["pid", "pitch_type", "x", "z", "same", "balls", "strikes"]]


def vscores(d, M):
    F = cvr.build(prep(d)); S = cvr.score_fl(F, M["coefs"], M["lo"], M["hi"]); S.index = F["pid"].to_numpy(); S = S.drop(columns="pid"); S = S.rename(columns={v: k for k, v in LAB.items()}); S["n_pitch"] = F["n"].to_numpy(); return S[list(LAB) + ["n_pitch"]]


def reliability(rc, M):
    res = {k: [] for k in LAB}; sz = []
    for yr in (2025, 2026):
        d = fp.filter_period(rc, *rs.SEASON[yr]); cnt = d.groupby("pitcher_id").size(); d = d[d["pitcher_id"].isin(cnt[cnt >= 400].index)]
        key = d["game_id"].astype(str) + "|" + d["pa_id"].astype(str); codes, uniq = pd.factorize(key)
        for _ in range(REPS):
            hh = rng.integers(0, 2, len(uniq))[codes]; a = vscores(d[hh == 0], M); b = vscores(d[hh == 1], M); j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            for k in LAB:
                res[k].append(j[f"{k}_a"].corr(j[f"{k}_b"]))
            sz.append(float(j["n_pitch_a"].mean()))
        print(f"   reliability {yr} done", flush=True)
    return {k: (max(float(np.mean(sz)) * (1 - float(np.mean(v))) / float(np.mean(v)), 1.0), float(np.mean(v))) for k, v in res.items()}


def attach(R, rc, T, year, names, n0s, M):
    t0, t1 = T.season_bounds(year); R = R.copy(); R["name"] = R["pitcher"].str.rsplit("|", n=1).str[0]; parts = []
    for c, g in R.groupby("cut"):
        asof = t0 + (t1 - t0) * c; pool = fp.filter_period(rc, rs.SEASON[year][0], (pd.Timestamp(asof) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        if year - 1 >= 2025:
            pool = pd.concat([fp.filter_period(rc, *rs.SEASON[year - 1]), pool], ignore_index=True)
        S = vscores(pool, M); out = pd.DataFrame(index=S.index)
        for k in LAB:
            ok = S[k].notna(); lg = np.average(S.loc[ok, k], weights=S.loc[ok, "n_pitch"]); out[k] = ((S[k] - lg) * S["n_pitch"] / (S["n_pitch"] + n0s[k][0])).where(ok)
        out["name_clean"] = out.index.map(names); f = out.dropna(subset=["name_clean"]).groupby("name_clean")[list(LAB)].mean().reindex(g["name"]).fillna(0.0); g = g.copy()
        for k in LAB:
            g[k] = f[k].to_numpy()
        parts.append(g); print(f"   features {year} cutoff {c:.2f} done", flush=True)
    return pd.concat(parts, ignore_index=True)


def main():
    pd.set_option("display.width", 250); D = cvr.mlb_frame(); A, coefs, lo, hi = cvr.mlb_fit(D); M = {"coefs": {LAB[k]: coefs[LAB[k]] for k in LAB}, "lo": lo, "hi": hi}
    M["coefs"] = {LAB[k]: coefs[LAB[k]] for k in LAB}; raw = fp.load_fl_pitches_cached(); rc = fp.add_count(raw); names = rs.idmap(raw); T = pr.Tables()
    # score_fl names columns by the coefficient labels; vscores renames them to v1.. via LAB
    n0s = reliability(rc, M); print("reliability (random-half r, n0 pitches): " + str({k: (round(v[1], 2), round(v[0], 0)) for k, v in n0s.items()}), " production all-pitch: r .82, n0 95", flush=True)
    consts = {2025: pr.fit_constants(2024), 2026: pr.fit_constants(2025)}; R = {y: attach(pd.read_parquet(f"output/pitcher_ros_stuff_rows_{y}.parquet"), rc, T, y, names, n0s, M) for y in (2025, 2026)}; res = []
    for tag, tr, te in (("FORWARD (fit 2025 ROS rows, score 2026)", 2025, 2026), ("REVERSE (fit 2026, score 2025)", 2026, 2025)):
        n0tr, n0te = consts[tr]["bbr"]["n0"], consts[te]["bbr"]["n0"]; print(f"\n{'=' * 90}\n{tag}")
        for k in LAB:
            o = fbt.paired(R[tr], R[te], ["command_score"], [k], n0tr, n0te); o = {"design": tag, "variant": LAB[k], **o}; res.append(o)
            print(f"  {LAB[k]:28s} vs production: production {o['point a']:+.5f}, variant {o['point b']:+.5f}, diff {o['point b-a']:+.5f} {o['diff 90%']}, P(variant better) {o['P(b better than a)']:.2f}", flush=True)
    json.dump({"n0": {k: list(v) for k, v in n0s.items()}, "ros": res}, open("output/ros_cmdvariants_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
