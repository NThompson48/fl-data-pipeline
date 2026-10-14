"""
fl_hitter_k_study.py -- can plate-discipline measures improve the hitter K% projection?

Current model: K% regressed to the league (n0 79 PA, last two seasons, prior-season weight .75); year-to-year correlation .66.
Candidate inputs, computed from every pitch a hitter saw (2025-26, ~530k pitches):
  SwStr%  whiffs per pitch            Whiff%   whiffs per swing                  Chase%   swings per pitch outside the zone
  ZSwing% swings per pitch in zone    ZContact% contact per swing in zone      OContact% contact per swing outside the zone
  Swing%  swings per pitch            Take-K%  called strikes per pitch         FPSwing% swings per first pitch
A  Reliability (random halves of a hitter-season's PA, Spearman-Brown), 2025 -> 2026 correlation, same-season correlation with K%.
B  Within-season PA split (hitter-seasons with >= 150 PA; known sample m = 30 / 60 / 120 / 200 PA -> K% over the rest): add-ons on top of the regressed K%, ALL models
   including the reference carry the same cross-validated intercept; hitter-grouped CV; standard inverse-variance weights; hitter-cluster bootstrap.
C  Year to year (2025 -> 2026 with the production two-season K% base, and the reverse), same fair reference.

Usage:  python fl_hitter_k_study.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev
import fl_hitter_pa_split_test as ps
import fl_innings_projection as ip

warnings.filterwarnings("ignore")
rng = np.random.default_rng(31)
PITCHES = "output/hitter_pitches_2025_2026.parquet"
SWING = {"strike_swinging", "foul_not_fieldable", "foul_fieldable", "foul_ball", "in_play"}
# variable -> (numerator column, denominator column)
VARS = {"SwStr%": ("whiff", "pitches"), "Whiff%": ("whiff", "swings"), "Chase%": ("o_swing", "o_pitches"), "ZSwing%": ("z_swing", "z_pitches"),
        "ZContact%": ("z_contact", "z_swing"), "OContact%": ("o_contact", "o_swing"), "Swing%": ("swings", "pitches"), "Take-K%": ("cstrike", "pitches"), "FPSwing%": ("fp_swing", "fp_pitches")}
N0_K = 79.3
N0W = 79.3


def pa_table():
    d = pd.read_parquet(PITCHES)
    d = d[d["pa_id"].notna() & d["plate_x"].notna() & d["outside_zone"].notna()].copy()
    d["name_clean"] = d["batter"].apply(ip.lastfirst_to_clean)
    sw = d["pitch_call"].isin(SWING)
    wh = d["pitch_call"].eq("strike_swinging")
    out = d["outside_zone"] == 1; inz = ~out
    d["swings"] = sw.astype(int); d["whiff"] = wh.astype(int); d["pitches"] = 1
    d["o_pitches"] = out.astype(int); d["z_pitches"] = inz.astype(int)
    d["o_swing"] = (sw & out).astype(int); d["z_swing"] = (sw & inz).astype(int)
    d["o_contact"] = (sw & ~wh & out).astype(int); d["z_contact"] = (sw & ~wh & inz).astype(int)
    d["cstrike"] = d["pitch_call"].eq("strike_called").astype(int)
    first = (d["balls"] == 0) & (d["strikes"] == 0)
    d["fp_pitches"] = first.astype(int); d["fp_swing"] = (first & sw).astype(int)
    d["k"] = d["k_or_bb"].eq("strikeout").astype(int)
    d["bb"] = d["k_or_bb"].eq("walk").astype(int)
    cols = ["swings", "whiff", "pitches", "o_pitches", "z_pitches", "o_swing", "z_swing", "o_contact", "z_contact", "cstrike", "fp_pitches", "fp_swing", "k", "bb"]
    a = d.groupby(["name_clean", "year", "game_id", "pa_id"])[cols].sum().reset_index()
    a["k"] = (a["k"] > 0).astype(int); a["bb"] = (a["bb"] > 0).astype(int); a["pa"] = 1
    return a


def season_stats(a):
    cols = ["swings", "whiff", "pitches", "o_pitches", "z_pitches", "o_swing", "z_swing", "o_contact", "z_contact", "cstrike", "fp_pitches", "fp_swing", "k", "bb", "pa"]
    s = a.groupby(["name_clean", "year"])[cols].sum().reset_index()
    for v, (nu, de) in VARS.items():
        s[v] = s[nu] / s[de].replace(0, np.nan)
    s["kpct"] = s["k"] / s["pa"]; s["bbpct"] = s["bb"] / s["pa"]
    return s


def partA(a, s):
    print("A. RELIABILITY AND WHAT EACH TRACKS (hitter-seasons with >= 100 PA)")
    big = a.groupby(["name_clean", "year"]).filter(lambda g: len(g) >= 100)
    res = {v: [] for v in VARS}; res["K%"] = []; dens = {v: [] for v in VARS}
    cols = ["swings", "whiff", "pitches", "o_pitches", "z_pitches", "o_swing", "z_swing", "o_contact", "z_contact", "cstrike", "fp_pitches", "fp_swing", "k", "pa"]
    for rep in range(20):
        b = big.assign(h=rng.integers(0, 2, len(big)))
        h = b.groupby(["name_clean", "year", "h"])[cols].sum().reset_index()
        for v, (nu, de) in VARS.items():
            h[v] = h[nu] / h[de].replace(0, np.nan)
        h["K%"] = h["k"] / h["pa"]
        w = h.pivot_table(index=["name_clean", "year"], columns="h", values=list(VARS) + ["K%"] + ["pitches", "swings", "o_pitches", "z_pitches", "o_swing", "z_swing", "fp_pitches", "pa"]).dropna()
        for v in list(VARS) + ["K%"]:
            res[v].append(w[(v, 0)].corr(w[(v, 1)]))
        for v, (nu, de) in VARS.items():
            dens[v].append(float(w[(de, 0)].mean()))
    a25 = s[(s["year"] == 2025) & (s["pa"] >= 100)].set_index("name_clean"); a26 = s[(s["year"] == 2026) & (s["pa"] >= 100)].set_index("name_clean")
    j = a25.join(a26, lsuffix="_25", rsuffix="_26", how="inner")
    big_s = s[s["pa"] >= 100]
    rows = []; n0 = {}
    for v in list(VARS) + ["K%"]:
        r = float(np.mean(res[v]))
        if v in VARS:
            n0[v] = max(np.mean(dens[v]) * (1 - r) / r, 1.0)
        rows.append({"measure": v, "split-half r": r, "full-season reliability": 2 * r / (1 + r), "n0 (own denominator)": n0.get(v, N0_K), "y/y 2025->26": (j[f"{v}_25"].corr(j[f"{v}_26"]) if v in VARS else j["kpct_25"].corr(j["kpct_26"])),
                     "corr w/ K% (same season)": big_s[v].corr(big_s["kpct"]) if v in VARS else 1.0})
    print(pd.DataFrame(rows).round(3).to_string(index=False) + f"\n   (n = {len(j)} hitters in both years)")
    return n0


def draws(a, s, n0, sizes=(30, 60, 120, 200), ndraw=25):
    lgv = {v: (s["pitches"].sum() and (s[VARS[v][0]].sum() / s[VARS[v][1]].sum())) for v in VARS}
    lg_k = a.groupby("year")["k"].mean()
    rows = []
    for (nm, y), g in a.groupby(["name_clean", "year"]):
        n = len(g)
        if n < 150:
            continue
        arr = {c: g[c].to_numpy() for c in ["swings", "whiff", "pitches", "o_pitches", "z_pitches", "o_swing", "z_swing", "o_contact", "z_contact", "cstrike", "fp_pitches", "fp_swing"]}; kk = g["k"].to_numpy()
        for m in sizes:
            if n - m < 30:
                continue
            for _ in range(ndraw):
                idx = rng.permutation(n); ia, io = idx[:m], idx[m:]
                row = {"name_clean": nm, "year": y, "m": m, "n_out": n - m, "k_in": kk[ia].mean(), "y": kk[io].mean(), "lg_k": lg_k[y]}
                for v, (nu, de) in VARS.items():
                    num, den = arr[nu][ia].sum(), arr[de][ia].sum()
                    x = num / den if den > 0 else np.nan
                    row[v] = np.nan if np.isnan(x) else (x - lgv[v]) * den / (den + n0[v])
                rows.append(row)
    return pd.DataFrame(rows)


def partB(D):
    print("\nB. WITHIN-SEASON PA SPLIT: known sample m PA -> K% over the rest (fair reference: every model has the same CV intercept)")
    sizes = sorted(D["m"].unique()); out = []
    for m in sizes:
        S = D[D["m"] == m].reset_index(drop=True)
        y = S["y"].to_numpy(); base = (S["lg_k"] + m * (S["k_in"] - S["lg_k"]) / (m + N0_K)).to_numpy()
        wt, _, _ = ew.weights_from_base(y, base, S["n_out"].to_numpy())
        groups = (S["name_clean"] + "|" + S["year"].astype(str)).to_numpy()
        fade = N0W / (m + N0W)
        Z = {v: np.where(S[v].isna(), 0.0, S[v].to_numpy()) for v in VARS}
        sd = {v: Z[v].std() for v in VARS}
        models = {"intercept only (reference)": []}
        for v in VARS:
            models[v] = [v]
        models["SwStr%+Chase%"] = ["SwStr%", "Chase%"]; models["Whiff%+Chase%+ZContact%"] = ["Whiff%", "Chase%", "ZContact%"]; models["all nine"] = list(VARS)
        pr = {}
        for k, vs in models.items():
            X = np.column_stack([Z[v] / sd[v] * fade for v in vs]) if vs else np.zeros((len(y), 0))
            pr[k] = ps.cv_group(y, base, X, wt, groups)
        ug = pd.Series(groups).unique(); by = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array); bs = {k: [] for k in pr}
        for _ in range(300):
            ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); r0 = ps.rm(y[ii], pr["intercept only (reference)"][ii], wt[ii])
            for k in pr:
                bs[k].append(ps.rm(y[ii], pr[k][ii], wt[ii]) - r0)
        for k in pr:
            out.append({"model": k, "m": m, "change": np.mean(bs[k]), "lo": np.percentile(bs[k], 5), "hi": np.percentile(bs[k], 95), "ref RMSE": ps.rm(y, pr["intercept only (reference)"], wt)})
    R = pd.DataFrame(out)
    for m in sizes:
        r = R[R["m"] == m]
        print(f"\n  known sample {m} PA (reference RMSE {r['ref RMSE'].iloc[0]:.4f}); change in K% RMSE vs the intercept-only reference (negative = better), 90% interval")
        print(r[r["model"] != "intercept only (reference)"].assign(interval=lambda x: [f"[{a:+.5f}, {b:+.5f}]" for a, b in zip(x["lo"], x["hi"])])[["model", "change", "interval"]].round(5).to_string(index=False))
    return R


def main():
    a = pa_table(); s = season_stats(a)
    print(f"plate appearances with pitch-level data: {len(a):,} ({a['k'].mean():.1%} strikeouts); hitter-seasons with >= 150 PA: {int((s['pa'] >= 150).sum())}")
    n0 = partA(a, s)
    D = draws(a, s, n0)
    partB(D)


if __name__ == "__main__":
    main()
