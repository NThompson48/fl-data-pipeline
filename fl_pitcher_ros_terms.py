"""
fl_pitcher_ros_terms.py -- which pitch-level measures improve the pitcher ROS K% and BB% update (fl_pitcher_ros.py), tested CHRONOLOGICALLY.

Candidates (computed from every pitch thrown, this season to date pooled with last season's where it exists):
  K%:   SwStr% (whiffs per pitch), Whiff% (whiffs per swing), CSW% (called strikes + whiffs per pitch), Chase% (swings per pitch outside the zone), Zone%
  BB%:  Zone% (pitches in the zone), FPS% (first-pitch strike rate), Chase%, CSW%, SwStr%
Form (as for hitters): proj = ROS-update + B x (shrunk measure deviation from the league) x n0 / (PA_eff + n0), PA_eff = nu n_ytd + d n_1 + ... (the ROS update's own effective sample), n0 the stat's
regression constant.  Measures are shrunk by their own reliability (n0 pitches / swings from random-half splits).
Two-stage design (no leakage): Stage 1 fits B on the 2025 chronological ROS rows only (all five cutoffs pooled, with an intercept); Stage 2 applies the frozen B to 2026 ROS rows with the
2026 constants; the reverse (fit 2026, score 2025) is a check.  Bootstrap resamples pitchers in BOTH stages; every model gets the same constant (fair reference).  Change in RMSE vs the
ROS update, negative = the term helps; share of replicates in which it helps.

Usage:  python fl_pitcher_ros_terms.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_pitcher_ros as pr
import fl_pitcher_ros_backtest as pb

warnings.filterwarnings("ignore")
rng = np.random.default_rng(88)
SWING = {"strike_swinging", "foul_not_fieldable", "foul_fieldable", "foul_ball", "in_play"}
STRIKE = SWING | {"strike_called"}
MEAS = {"SwStr%": ("whiffs", "pitches"), "Whiff%": ("whiffs", "swings"), "CSW%": ("csw", "pitches"), "Chase%": ("o_swings", "o_pitches"), "Zone%": ("z_pitches", "pitches"), "FPS%": ("fps", "first")}
KSET = {"SwStr%": ["SwStr%"], "Whiff%": ["Whiff%"], "CSW%": ["CSW%"], "Chase%": ["Chase%"], "Zone%": ["Zone%"], "SwStr% + Chase%": ["SwStr%", "Chase%"]}
BSET = {"Zone%": ["Zone%"], "FPS%": ["FPS%"], "Chase%": ["Chase%"], "CSW%": ["CSW%"], "SwStr%": ["SwStr%"], "Zone% + FPS%": ["Zone%", "FPS%"], "Zone% + Chase%": ["Zone%", "Chase%"]}
R_FIT, R_EVAL = 200, 3000


def prep_pitches(T):
    pi = T.pi[T.pi["outside_zone"].notna()].copy()
    pi["pitches"] = 1; pi["swings"] = pi["pitch_call"].isin(SWING).astype(int); pi["whiffs"] = pi["pitch_call"].eq("strike_swinging").astype(int)
    pi["csw"] = (pi["pitch_call"].eq("strike_called") | pi["pitch_call"].eq("strike_swinging")).astype(int)
    out = pi["outside_zone"] == 1
    pi["o_pitches"] = out.astype(int); pi["z_pitches"] = (~out).astype(int); pi["o_swings"] = (out & (pi["swings"] == 1)).astype(int)
    first = (pi["balls"] == 0) & (pi["strikes"] == 0)
    pi["first"] = first.astype(int); pi["fps"] = (first & pi["pitch_call"].isin(STRIKE)).astype(int)
    return pi


def counts(pi):
    cols = ["pitches", "swings", "whiffs", "csw", "o_pitches", "z_pitches", "o_swings", "first", "fps"]
    return pi.groupby("name_clean")[cols].sum()


def reliability_n0(pi, T):
    """n0 (own denominator) from random-half split reliability: pitcher-seasons with >= 150 BF, halves by plate appearance."""
    big = T.pa.groupby(["name_clean", "year"]).size(); keep = set(big[big >= 150].index)
    p = pi[[(n, y) in keep for n, y in zip(pi["name_clean"], pi["year"])]].copy()
    key = p["game_id"].astype(str) + "|" + p["pa_id"].astype(str)
    codes, uniq = pd.factorize(key)
    res = {m: [] for m in MEAS}; dens = {m: [] for m in MEAS}
    for rep in range(8):
        h = rng.integers(0, 2, len(uniq))[codes]
        g = p.assign(h=h).groupby(["name_clean", "year", "h"])[["pitches", "swings", "whiffs", "csw", "o_pitches", "z_pitches", "o_swings", "first", "fps"]].sum().reset_index()
        for m, (nu, de) in MEAS.items():
            g[m] = g[nu] / g[de].replace(0, np.nan)
        w = g.pivot_table(index=["name_clean", "year"], columns="h", values=list(MEAS) + ["pitches", "swings", "o_pitches", "first"]).dropna()
        for m, (nu, de) in MEAS.items():
            res[m].append(w[(m, 0)].corr(w[(m, 1)])); dens[m].append(float(w[(de, 0)].mean()))
    return {m: max(float(np.mean(dens[m])) * (1 - np.mean(res[m])) / np.mean(res[m]), 1.0) for m in MEAS}, {m: float(np.mean(res[m])) for m in MEAS}


def rows_for(T, pi, consts, year, n0m):
    t0, t1 = T.season_bounds(year); out = []
    pcur = pi[pi["year"] == year]; pprev = counts(pi[pi["year"] == year - 1]) if year - 1 >= 2025 else None
    for c in pb.CUTS:
        asof = t0 + (t1 - t0) * c
        f, lg = pr.features(T, year, asof)
        base = pr.predict(f, lg, consts)
        cur = counts(pcur[pcur["date"] < asof]).reindex(f.index).fillna(0.0)
        prv = (pprev.reindex(f.index).fillna(0.0) if pprev is not None else cur * 0.0)
        tot = cur + prv
        lgc = counts(pcur[pcur["date"] < asof]).sum()
        d = pd.DataFrame({"y_k": f["k_ros"].to_numpy(), "y_bb": f["bb_ros"].to_numpy(), "n_ros": f["n_ros"].to_numpy(), "cut": c, "year": year, "pitcher": (f.index + "|" + str(year)).to_numpy(),
                          "base_k": base["k"].to_numpy(), "base_bb": base["bbr"].to_numpy(), "pa_eff_k": base["pa_eff_k"].to_numpy(), "pa_eff_bb": base["pa_eff_bbr"].to_numpy()})
        for m, (nu, de) in MEAS.items():
            den = tot[de].to_numpy(float); x = np.where(den > 0, tot[nu].to_numpy(float) / np.where(den > 0, den, 1), np.nan); lgv = lgc[nu] / lgc[de]
            d[m] = np.where(np.isnan(x), 0.0, (x - lgv) * den / (den + n0m[m]))
        out.append(d)
    return pd.concat(out, ignore_index=True)


def stage1(R, stat, vs, n0stat):
    y, base = R[f"y_{stat}"].to_numpy(), R[f"base_{stat}"].to_numpy(); w, _, _ = ew.weights_from_base(y, base, R["n_ros"].to_numpy())
    fade = n0stat / (R[f"pa_eff_{stat}"].to_numpy() + n0stat)
    X = np.column_stack([np.ones(len(y))] + [R[v].to_numpy() * fade for v in vs]); Y = y - base
    fit = lambda idx: np.linalg.lstsq(X[idx] * np.sqrt(w[idx])[:, None], Y[idx] * np.sqrt(w[idx]), rcond=None)[0]
    full = fit(np.arange(len(Y))); g = R["pitcher"].to_numpy(); ug = pd.Series(g).unique(); by = pd.Series(np.arange(len(g))).groupby(g).apply(np.array)
    reps = np.array([fit(np.concatenate([by[u] for u in rng.choice(ug, len(ug))])) for _ in range(R_FIT)])
    return full, reps


def stage2(R, stat, vs, n0stat, full, reps):
    y, base = R[f"y_{stat}"].to_numpy(), R[f"base_{stat}"].to_numpy(); w, _, _ = ew.weights_from_base(y, base, R["n_ros"].to_numpy())
    fade = n0stat / (R[f"pa_eff_{stat}"].to_numpy() + n0stat); cols = [R[v].to_numpy() * fade for v in vs]
    term = lambda coef: sum(coef[1 + i] * cols[i] for i in range(len(cols)))
    g = R["pitcher"].to_numpy(); ug = pd.Series(g).unique(); by = pd.Series(np.arange(len(g))).groupby(g).apply(np.array)
    rm = lambda a, p, ww: float(np.sqrt(np.average((a - p) ** 2, weights=ww)))

    def delta(idx, coef):
        m1 = base[idx] + term(coef)[idx]; c0 = np.average(y[idx] - base[idx], weights=w[idx]); c1 = np.average(y[idx] - m1, weights=w[idx])
        return rm(y[idx], m1 + c1, w[idx]) - rm(y[idx], base[idx] + c0, w[idx])
    point = delta(np.arange(len(y)), full); ds = np.empty(R_EVAL)
    for i in range(R_EVAL):
        idx = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); ds[i] = delta(idx, reps[rng.integers(0, len(reps))])
    return point, ds


def main():
    T = pr.Tables(); pi = prep_pitches(T)
    n0m, rel = reliability_n0(pi, T)
    print("measure reliability (random-half r) and n0 (own denominator):", {m: (round(rel[m], 2), round(n0m[m], 1)) for m in MEAS})
    consts = {2025: pr.fit_constants(2024), 2026: pr.fit_constants(2025)}
    R = {y: rows_for(T, pi, consts[y], y, n0m) for y in (2025, 2026)}
    print({y: len(r) for y, r in R.items()}, "rows (pitcher-cutoffs)")
    for tag, tr, te in (("FORWARD: coefficient fit on 2025 ROS rows -> scored on 2026", 2025, 2026), ("reverse check: fit 2026 -> score 2025", 2026, 2025)):
        print(f"\n{'=' * 100}\n{tag}")
        for stat, lab, sets, n0stat in (("k", "K%", KSET, consts[te]["k"]["n0"]), ("bb", "BB%", BSET, consts[te]["bbr"]["n0"])):
            rows = []
            for name, vs in sets.items():
                full, reps = stage1(R[tr], stat, vs, consts[tr]["k" if stat == "k" else "bbr"]["n0"])
                point, ds = stage2(R[te], stat, vs, n0stat, full, reps)
                rows.append({"term": name, "coef (stage 1)": ", ".join(f"{c:+.3f}" for c in full[1:]), "point": point, "boot mean": ds.mean(), "90% interval": f"[{np.percentile(ds,5):+.5f}, {np.percentile(ds,95):+.5f}]", "P(helps)": float((ds < 0).mean())})
            base_rm = pb.wrmse(R[te][f"y_{stat}"].to_numpy(), R[te][f"base_{stat}"].to_numpy(), ew.weights_from_base(R[te][f"y_{stat}"].to_numpy(), R[te][f"base_{stat}"].to_numpy(), R[te]["n_ros"].to_numpy())[0])
            print(f"\n  {lab} (ROS update RMSE on the scored season {base_rm:.4f}); change in RMSE vs the ROS update:")
            print(pd.DataFrame(rows).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
