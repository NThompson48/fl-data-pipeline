"""
fl_hitter_position_test.py -- validate the hitter POSITION adjustment (pos_rate = (PA_t x own pos runs/PA + n0 x position-group rate) / (PA_t + n0), n0 = 100, group = his latest primary position).
Only 2025 and 2026 carry position runs (WAR table), so there is one transition each way: forward 2025 -> 2026 (position in 2025 known for 38% of hitters) and reverse 2026 -> 2025 (position in 2026 known for 93%).
Target: next season's position runs per PA (hitters with >= 100 PA in both seasons), in runs per 600 PA; standard inverse-variance weights fitted on the target law; hitter-cluster bootstrap for differences.
Models: zero | group mean of the predictor-season position | own rate at face value | production blend n0 = 100 | blend with n0 chosen on the OTHER direction | ORACLE group mean of the TARGET-season position (what knowing the position buys).
Also: how often the primary position changes, and what the realised SD of the residual is vs the 6.0 runs/600 PA the simulation uses.
"""
import warnings
import numpy as np
import pandas as pd
import fl_eval_weights as ew
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore"); rng = np.random.default_rng(5)


def wr(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def groups(p, yr):
    t = p[(p["year"] == yr) & p["primary_pos"].notna()]; return t.groupby("primary_pos")["pos_adj_runs"].sum() / t.groupby("primary_pos")["pa"].sum()


def pair(p, a, b, min_pa=100):
    A = p[p["year"] == a].drop_duplicates("name_clean").set_index("name_clean"); B = p[p["year"] == b].drop_duplicates("name_clean").set_index("name_clean")
    d = A[["pa", "pos_adj_runs", "primary_pos"]].join(B[["pa", "pos_adj_runs", "primary_pos"]], lsuffix="_a", rsuffix="_b", how="inner")
    d = d[(d["pa_a"] >= min_pa) & (d["pa_b"] >= min_pa) & d["pos_adj_runs_a"].notna() & d["pos_adj_runs_b"].notna()].copy(); return d


def blend(d, g, n0):
    grp = d["primary_pos_a"].map(g); own = d["pos_adj_runs_a"] / d["pa_a"]
    return np.where(grp.notna(), (d["pa_a"] * own + n0 * grp) / (d["pa_a"] + n0), np.nan)


def run(p, a, b, n0_other):
    d = pair(p, a, b); g_a = groups(p, a); g_b = groups(p, b)
    d["y"] = d["pos_adj_runs_b"] / d["pa_b"] * 600; d["own"] = d["pos_adj_runs_a"] / d["pa_a"] * 600
    d["grp"] = d["primary_pos_a"].map(g_a) * 600; d["oracle"] = d["primary_pos_b"].map(g_b) * 600
    d = d[d["primary_pos_a"].notna() & d["primary_pos_b"].notna()].copy()
    d["prod"] = blend(d, g_a, 100.0) * 600; d["alt"] = blend(d, g_a, n0_other) * 600
    d["zero"] = 0.0
    # choose n0 in-direction on a grid for reference (in-sample)
    w = ew.weights_from_base(d["y"].to_numpy(), d["prod"].to_numpy(), d["pa_b"].to_numpy())[0]; y = d["y"].to_numpy(); ids = d.index.to_numpy()
    by = pd.Series(np.arange(len(d))).groupby(ids).apply(np.array); ug = by.index.to_numpy(); rows = []
    print(f"\n{'=' * 100}\n{a} -> {b}: n = {len(d)} hitters (>= 100 PA in both, position known in both); position changed: {(d['primary_pos_a'] != d['primary_pos_b']).mean():.0%}\n{'=' * 100}")
    for lab, c in (("zero", "zero"), ("group mean of the predictor-season position", "grp"), ("own rate at face value", "own"), ("production blend (n0 = 100)", "prod"), (f"blend with n0 = {n0_other:.0f} (chosen in the other direction)", "alt"), ("ORACLE: group mean of the target-season position", "oracle")):
        r = {"model": lab, "RMSE runs/600": wr(y, d[c].to_numpy(), w)}
        if c != "prod":
            dd = []
            for _ in range(500):
                ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wr(y[ii], d[c].to_numpy()[ii], w[ii]) - wr(y[ii], d["prod"].to_numpy()[ii], w[ii]))
            dd = np.array(dd); r.update({"vs production": np.mean(dd), "90% interval": f"[{np.percentile(dd, 5):+.2f}, {np.percentile(dd, 95):+.2f}]", "P(better)": float((dd < 0).mean())})
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    grid = {n0: wr(y, blend(d, g_a, n0) * 600, w) for n0 in (0, 25, 50, 100, 200, 400, 1000, 5000)}; best = min(grid, key=grid.get)
    print("  n0 grid (RMSE runs/600):", {k: round(v, 3) for k, v in grid.items()}, "-> best in this direction:", best)
    res = d["y"] - d["prod"]; print(f"  residual of the production blend: mean {res.mean():+.2f}, SD {res.std():.2f} runs per 600 PA (the simulation uses 6.0); PA-weighted RMS {np.sqrt(np.average(res ** 2, weights=d['pa_b'])):.2f}")
    return best, d


if __name__ == "__main__":
    p = dev.load()
    print("hitters with primary_pos:", p[p.year.isin([2025, 2026])].groupby("year")["primary_pos"].apply(lambda s: f"{s.notna().mean():.0%}").to_dict())
    b1, d1 = run(p, 2026, 2025, 100.0)      # reverse first: position known for 93%
    b2, d2 = run(p, 2025, 2026, b1)
    run(p, 2026, 2025, b2)
