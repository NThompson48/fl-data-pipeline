"""
fl_pitcher_ros_stuff.py -- the production pitcher skill measures as the ROS "power prior":  wh_sva_level -> K%,  command_score / horiz_var + vert_var -> BB%   (user suggestion).

wh_sva_level  = mean over a pitcher's swings against him of (whiff - expected whiff given the pitch's characteristics), the production stuff measure (fp.compute_pooled_wh_sva, XGBoost fit on the pool).
command_score = OpenCommand-calibrated inferred-miss inches from horiz_var / vert_var (location variance), fp.compute_command_score.
At every as-of date the pool = last season's pitches (full) + this season's pitches BEFORE the date, so nothing from the ROS window enters the features (the whiff model is refit on the pool each time).
Each measure is shrunk toward the league by its own reliability (n0 swings / pitches, random-half splits of full seasons) and enters as
    proj = ROS update + B x (shrunk deviation) x n0stat / (PA_eff + n0stat),
the same form as every other term.  Two-stage test: B fit on the 2025 chronological ROS rows only (five cutoffs, intercept), frozen, scored on 2026; the reverse is a check; pitcher-cluster two-stage bootstrap;
every model gets the same constant.

Usage:  python fl_pitcher_ros_stuff.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_innings_projection as ip
import fl_pitcher_features as fp
import fl_pitcher_ros as pr
import fl_pitcher_ros_backtest as pb
import fl_pitcher_ros_terms as tm

warnings.filterwarnings("ignore")
rng = np.random.default_rng(404)
KSET = {"wh_sva_level": ["wh_sva"]}
BSET = {"command_score": ["command_score"], "horiz_var + vert_var": ["horiz_var", "vert_var"], "command_score + horiz_var + vert_var": ["command_score", "horiz_var", "vert_var"]}
SEASON = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}


def idmap(raw):
    m = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id").copy(); m["name_clean"] = m["pitcher"].apply(ip.lastfirst_to_clean)
    return m.set_index("pitcher_id")["name_clean"]


def reliability(raw):
    """n0 (swings for wh_sva_level, pitches for command_score) from random-half splits of full pitcher-seasons."""
    out = {}
    full = pd.concat([fp.filter_period(raw, *SEASON[2025]), fp.filter_period(raw, *SEASON[2026])], ignore_index=True)
    sva = fp.compute_pooled_wh_sva(full); sva["year"] = pd.to_datetime(sva["date"]).dt.year
    g = sva.groupby(["pitcher_id", "year"]).size(); keep = g[g >= 150].index
    s = sva.set_index(["pitcher_id", "year"]).loc[keep].reset_index()
    r, hs = [], []
    for _ in range(20):
        h = s.assign(h=rng.integers(0, 2, len(s))).groupby(["pitcher_id", "year", "h"])["sva"].agg(["mean", "size"]).reset_index().pivot_table(index=["pitcher_id", "year"], columns="h", values=["mean", "size"]).dropna()
        r.append(h[("mean", 0)].corr(h[("mean", 1)])); hs.append(h["size"].to_numpy().mean())
    rr = float(np.mean(r)); out["wh_sva"] = (max(float(np.mean(hs)) * (1 - rr) / rr, 1.0), rr)
    # command score: halves by plate appearance within pitcher-seasons
    res = {"command_score": [], "horiz_var": [], "vert_var": []}; sizes = []
    for yr in (2025, 2026):
        d = fp.filter_period(raw, *SEASON[yr]); cnt = d.groupby("pitcher_id").size(); d = d[d["pitcher_id"].isin(cnt[cnt >= 400].index)]
        key = d["game_id"].astype(str) + "|" + d["pa_id"].astype(str); codes, uniq = pd.factorize(key)
        for _ in range(8):
            hh = rng.integers(0, 2, len(uniq))[codes]
            a = fp.compute_command_score(d[hh == 0]).set_index("pitcher_id"); b = fp.compute_command_score(d[hh == 1]).set_index("pitcher_id"); j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            for c in res:
                res[c].append(j[f"{c}_a"].corr(j[f"{c}_b"]))
            sizes.append(float((cnt[cnt >= 400] / 2).mean()))
    for c in res:
        rr = float(np.mean(res[c])); out[c] = (max(float(np.mean(sizes)) * (1 - rr) / rr, 1.0), rr)
    return out


def stuff_features(raw, year, asof, names, n0s):
    """Shrunk deviations of wh_sva / command_score / horiz_var / vert_var for every pitcher, from last season's pitches + this season's pitches before `asof`."""
    pool = fp.filter_period(raw, SEASON[year][0], (pd.Timestamp(asof) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    if year - 1 >= 2025:
        pool = pd.concat([fp.filter_period(raw, *SEASON[year - 1]), pool], ignore_index=True)
    sva = fp.compute_pooled_wh_sva(pool)
    lvl = sva.groupby("pitcher_id")["sva"].agg(wh_sva="mean", n_sw="size"); lg_sva = sva["sva"].mean()
    lv = bt.loc_var_levels(pool).set_index("pitcher_id"); npit = pool[pool["plate_x"].notna()].groupby("pitcher_id").size().rename("n_pitch")
    lv = lv.join(npit)
    out = pd.DataFrame(index=lvl.index.union(lv.index))
    out["wh_sva"] = (lvl["wh_sva"] - lg_sva) * lvl["n_sw"] / (lvl["n_sw"] + n0s["wh_sva"][0])
    for c in ("command_score", "horiz_var", "vert_var"):
        lgc = np.average(lv[c].dropna(), weights=lv.loc[lv[c].notna(), "n_pitch"])
        out[c] = (lv[c] - lgc) * lv["n_pitch"] / (lv["n_pitch"] + n0s[c][0])
    out["name_clean"] = out.index.map(names)
    return out.dropna(subset=["name_clean"]).groupby("name_clean")[["wh_sva", "command_score", "horiz_var", "vert_var"]].mean()


def rows(T, raw, names, consts, year, n0s):
    t0, t1 = T.season_bounds(year); out = []
    for c in pb.CUTS:
        asof = t0 + (t1 - t0) * c
        f, lg = pr.features(T, year, asof); base = pr.predict(f, lg, consts)
        sf = stuff_features(raw, year, asof, names, n0s).reindex(f.index).fillna(0.0)
        d = pd.DataFrame({"y_k": f["k_ros"].to_numpy(), "y_bb": f["bb_ros"].to_numpy(), "n_ros": f["n_ros"].to_numpy(), "cut": c, "year": year, "pitcher": (f.index + "|" + str(year)).to_numpy(),
                          "base_k": base["k"].to_numpy(), "base_bb": base["bbr"].to_numpy(), "pa_eff_k": base["pa_eff_k"].to_numpy(), "pa_eff_bb": base["pa_eff_bbr"].to_numpy()})
        for col in ("wh_sva", "command_score", "horiz_var", "vert_var"):
            d[col] = sf[col].to_numpy()
        out.append(d); print(f"   built {year} cutoff {c:.2f}: {len(d)} pitchers", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    raw = fp.load_fl_pitches_cached(); names = idmap(raw); T = pr.Tables()
    n0s = reliability(raw)
    print("reliability (random-half r) and n0 (swings for wh_sva, pitches for command measures):", {k: (round(v[1], 2), round(v[0], 0)) for k, v in n0s.items()})
    consts = {2025: pr.fit_constants(2024), 2026: pr.fit_constants(2025)}
    R = {y: rows(T, raw, names, consts[y], y, n0s) for y in (2025, 2026)}
    R[2025].to_parquet("output/pitcher_ros_stuff_rows_2025.parquet", index=False); R[2026].to_parquet("output/pitcher_ros_stuff_rows_2026.parquet", index=False)
    for tag, tr, te in (("FORWARD: coefficient fit on 2025 ROS rows -> scored on 2026", 2025, 2026), ("reverse check: fit 2026 -> score 2025", 2026, 2025)):
        print(f"\n{'=' * 100}\n{tag}")
        for stat, lab, sets in (("k", "K%", KSET), ("bb", "BB%", BSET)):
            skey = "k" if stat == "k" else "bbr"; rows_ = []
            for name, vs in sets.items():
                full, reps = tm.stage1(R[tr], stat, vs, consts[tr][skey]["n0"])
                point, ds = tm.stage2(R[te], stat, vs, consts[te][skey]["n0"], full, reps)
                rows_.append({"term": name, "coef (stage 1)": ", ".join(f"{c:+.3f}" for c in full[1:]), "point": point, "boot mean": ds.mean(), "90% interval": f"[{np.percentile(ds,5):+.5f}, {np.percentile(ds,95):+.5f}]", "P(helps)": float((ds < 0).mean())})
            y = R[te][f"y_{stat}"].to_numpy(); b = R[te][f"base_{stat}"].to_numpy(); w = ew.weights_from_base(y, b, R[te]["n_ros"].to_numpy())[0]
            print(f"\n  {lab} (ROS update RMSE on the scored season {pb.wrmse(y, b, w):.4f}); change in RMSE vs the ROS update:"); print(pd.DataFrame(rows_).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
