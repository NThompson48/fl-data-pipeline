"""fl_fielding_range_split_2026.py -- EXPLORATORY (user 2026-10-14 "How stable would just a Range component be for infielders?"): splits the PART range+throw credit into RANGE (reached the ball vs expected from ball + starting spot) and THROW (out given reached).  Needs the joined frame from fl_fielding_part_2026.load() saved as <dir>/m.parquet.  Run: python fl_fielding_range_split_2026.py <dir>."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\215nt\PycharmProjects\FLStuff\FLHitting")
import fl_fielding_part_2026 as fp
S = sys.argv[1]; m = pd.read_parquet(S + "/m.parquet"); IF = fp.IF
rows = []
for s in ("GSL", "AIR"):
    ms = m[m["set"] == s].reset_index(drop=True); gid = ms["game_id"].to_numpy()
    for pos in IF:
        X, y_out = fp.fielder_frame(ms, pos)
        touch = (ms["fielded_by"] == pos).astype(int).to_numpy()
        p_out = fp.crossfit(X, y_out, gid)                 # current: range+throw
        p_t = fp.crossfit(X, touch, gid)                   # range: reached the ball
        # throw: out given touched (expected from ball + start), fitted on touched balls only
        ti = np.nonzero(touch)[0]; p_c = np.full(len(ms), np.nan)
        p_c[ti] = fp.crossfit(X.iloc[ti].reset_index(drop=True), ms["out"].to_numpy()[ti], gid[ti])
        rows.append(pd.DataFrame({"set": s, "pos": pos, "fielder": ms[f"name_{pos.lower()}"].astype(str), "game_id": gid,
                                  "rt_d": y_out, "rt_p": p_out, "r_d": touch, "r_p": p_t, "c_d": np.where(touch == 1, ms["out"], np.nan), "c_p": p_c}))
P = pd.concat(rows, ignore_index=True); P = P[~P["fielder"].isin(["nan", "None"])]
# centre within pos x set
for k in ("rt", "r"):
    P[k + "_res"] = P[k + "_d"] - P[k + "_p"]; P[k + "_opp"] = P[k + "_p"]
    P[k + "_res"] -= P[k + "_opp"] * P.groupby(["pos", "set"])[k + "_res"].transform("sum") / P.groupby(["pos", "set"])[k + "_opp"].transform("sum")
P["c_res"] = P["c_d"] - P["c_p"]; P["c_opp"] = P["c_d"].notna().astype(float); P["c_res"] = P["c_res"].fillna(0)
P["c_res"] -= P["c_opp"] * P.groupby(["pos", "set"])["c_res"].transform("sum") / P.groupby(["pos", "set"])["c_opp"].transform("sum")
gm = P[["fielder", "pos", "game_id"]].drop_duplicates().sort_values("game_id"); gm["half"] = gm.groupby(["fielder", "pos"]).cumcount() % 2
P = P.merge(gm, on=["fielder", "pos", "game_id"])
rng = np.random.default_rng(5)
def rel(Q, k, label):
    H = Q.groupby(["fielder", "pos", "half"]).agg(o=(k + "_opp", "sum"), r=(k + "_res", "sum")).unstack("half").dropna()
    H = H[(H[("o", 0)] >= 15) & (H[("o", 1)] >= 15)]
    tot = Q.groupby(["fielder", "pos"])[k + "_opp"].sum(); ft_all = tot.groupby(level="pos").quantile(0.9)
    out = {}
    for pos, hh in list(H.groupby(level="pos")) + [("all IF", H)]:
        a, b = (hh[("r", 0)] / hh[("o", 0)]).to_numpy(), (hh[("r", 1)] / hh[("o", 1)]).to_numpy(); nh = float(((hh[("o", 0)] + hh[("o", 1)]) / 2).mean())
        r = float(np.corrcoef(a, b)[0, 1]); rs = [np.corrcoef(a[i], b[i])[0, 1] for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
        n0 = nh * (1 - r) / r if r > 0 else np.inf; ft = float(ft_all.get(pos, ft_all.mean()))
        out[pos] = dict(n=len(hh), r=round(r, 3), r_lo=round(np.nanpercentile(rs, 5), 3), r_hi=round(np.nanpercentile(rs, 95), 3), n0=round(n0) if np.isfinite(n0) else None, ft_opps=round(ft), rel_ft=round(ft / (ft + n0), 3) if np.isfinite(n0) else 0.0)
    print(f"\n== {label}\n" + pd.DataFrame(out).T.to_string())
rel(P, "rt", "CURRENT: range+throw, grounders + infield air (reproduces the logged metric)")
G = P[P["set"] == "GSL"]
rel(G, "rt", "range+throw, GROUNDERS ONLY (UZR drops infield air)")
rel(G, "r", "RANGE ONLY: reached the grounder vs expected (ball + starting spot)")
rel(G, "c", "THROW ONLY: out given reached, vs expected")
