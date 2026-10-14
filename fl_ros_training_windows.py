"""
fl_ros_training_windows.py -- how sensitive are the rest-of-season (ROS) models to the TRAINING windows behind their constants and coefficients?

A  Preseason regression constants (n0 per stat; prior-season weights fixed) fit on: all earlier transitions / the last two / the last one, vs the production constants (which, for scoring 2026, include the
   2025 -> 2026 transition, i.e. the scored season: a leak this section quantifies).  Hitters wOBA / K% / BB%; ROS update and ROS update + terms; 8 cutoffs (10%-80%), both seasons.
B  The same windows for the pitcher constants (K%, BB% n0 and prior-season weight).
C  Training window of the PITCH-LEVEL TERM coefficients: hitters (scale on the production coefficients, grid 0 .. 2) and pitchers (wh_sva_level -> K%, command_score -> BB%):
   by season (2025 / 2026 / pooled) and by position in the season (early cutoffs <= 40% / late cutoffs >= 50%), each scored on the held-out part.
D  History depth inside the ROS update: how many earlier seasons enter (0 = the season to date alone ... 3 hitters / 4 pitchers).
Standard inverse-variance weights, frozen per table; player-cluster bootstrap for differences.

Usage:  python fl_ros_training_windows.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_baseline_backtest as bt
import fl_hitter_ros as hr
import fl_pitcher_ros as pr
import fl_pitcher_ros_terms as tm
import fl_ros_production as rp

warnings.filterwarnings("ignore")
rng = np.random.default_rng(909)
C8 = [round(x, 2) for x in np.arange(0.10, 0.801, 0.10)]
HS = {"woba": ("woba_ros", "blend", "woba_final"), "k": ("k_ros", "k", "k_final"), "bbr": ("bb_ros", "bbr", "bbr_final")}


def wr(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def boot_diff(y, a, b, w, ids, n=400):
    ug = pd.Series(ids).unique(); by = pd.Series(np.arange(len(y))).groupby(ids).apply(np.array); out = np.empty(n)
    for i in range(n):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); out[i] = wr(y[ii], a[ii], w[ii]) - wr(y[ii], b[ii], w[ii])
    return out


def ivl(x):
    return f"[{np.percentile(x, 5):+.5f}, {np.percentile(x, 95):+.5f}]"


# ---------------------------------------------------------------------------------------------------------------------------- hitters
def fit_hitter_spec(T, max_target, min_target):
    """Production prior-season weights, n0 re-fit on transitions with target year in [min_target, max_target] (standard weights)."""
    spec = {}
    for s, (delta, n0) in hr.SPEC.items():
        d = bt.transitions(T.q, s); d = d[((d["t"] + 1) <= max_target) & ((d["t"] + 1) >= min_target)]
        spec[s] = (delta, float(bt.fit_n0(d, delta)))
    return spec


def hitter_frames(T):
    fr = {}
    for y in (2025, 2026):
        t0, t1 = T.season_bounds(y)
        for c in C8:
            f, lg = hr.features(T, y, t0 + (t1 - t0) * c, min_ros_pa=30); fr[(y, c)] = (f, lg)
    return fr


def hitter_variant(fr, spec_by_year, mh=3, tscale=1.0):
    out = {s: [] for s in HS}
    for (y, c), (f, lg) in fr.items():
        spec = spec_by_year[y]
        pre = hr.predict(f, lg, trackman=False, use_ytd=False, spec=spec, max_hist=mh); upd = hr.predict(f, lg, trackman=False, spec=spec, max_hist=mh)
        full = hr.predict(f, lg, trackman=True, params={"trackman_scale": tscale}, spec=spec, max_hist=mh)
        for s, (ycol, ucol, fcol) in HS.items():
            out[s].append(pd.DataFrame({"y": f[ycol].to_numpy(), "n_ros": f["n_ros"].to_numpy(), "year": y, "cut": c, "id": (f.index + "|" + str(y)).to_numpy(), "pre": pre[ucol].to_numpy(),
                                        "upd": upd[ucol].to_numpy(), "full": full[fcol].to_numpy()}))
    return {s: pd.concat(v, ignore_index=True) for s, v in out.items()}


def compare(variants, ref, label):
    print(f"\n  {label}")
    for s in HS:
        d0 = variants[ref][s]; w = ew.weights_from_base(d0["y"].to_numpy(), d0["upd"].to_numpy(), d0["n_ros"].to_numpy())[0]; y = d0["y"].to_numpy(); ids = d0["id"].to_numpy()
        rows = []
        for name, V in variants.items():
            d = V[s]; r = {"variant": name, "update RMSE": wr(y, d["upd"].to_numpy(), w), "update+terms RMSE": wr(y, d["full"].to_numpy(), w)}
            for yy in (2025, 2026):
                m = (d["year"] == yy).to_numpy(); r[f"{yy} upd+terms"] = wr(y[m], d["full"].to_numpy()[m], w[m])
            if name != ref:
                bd = boot_diff(y, d["full"].to_numpy(), d0["full"].to_numpy(), w, ids); r["vs ref (upd+terms)"] = np.mean(bd); r["90% interval"] = ivl(bd)
            rows.append(r)
        print(f"   {s}:"); print(pd.DataFrame(rows).round(5).to_string(index=False))


def main():
    T = hr.Tables(); fr = hitter_frames(T)
    # ------------------------------------------------------------------ A. hitter constants windows
    print("=" * 110 + "\nA. HITTERS: training window of the preseason regression constants (n0), prior-season weights fixed\n" + "=" * 110)
    prod = {2025: hr.SPEC, 2026: hr.SPEC}
    wins = {"all earlier transitions": lambda y: fit_hitter_spec(T, y - 1, 2023), "last two": lambda y: fit_hitter_spec(T, y - 1, max(2023, y - 2)), "last one": lambda y: fit_hitter_spec(T, y - 1, y - 1)}
    V = {}
    specs = {k: {y: fn(y) for y in (2025, 2026)} for k, fn in wins.items()}
    for k, sp in specs.items():
        print(f"   n0 ({k}): 2025 " + ", ".join(f"{s} {v[1]:.0f}" for s, v in sp[2025].items()) + " | 2026 " + ", ".join(f"{s} {v[1]:.0f}" for s, v in sp[2026].items()))
        V[k] = hitter_variant(fr, sp)
    print("   production constants: " + ", ".join(f"{s} {v[1]:.0f}" for s, v in hr.SPEC.items()) + "  (fit on all four transitions, 2022->23 ... 2025->26)")
    V["production (includes the scored season: leak)"] = hitter_variant(fr, prod)
    compare(V, "all earlier transitions", "ROS error by constants window (reference = all earlier transitions; differences are update + terms)")

    # ------------------------------------------------------------------ B. pitcher constants windows
    print("\n" + "=" * 110 + "\nB. PITCHERS: training window of the K% / BB% regression constants\n" + "=" * 110)
    Tp = pr.Tables(); frp = {}
    for y in (2025, 2026):
        t0, t1 = Tp.season_bounds(y)
        for c in C8:
            f, lg = pr.features(Tp, y, t0 + (t1 - t0) * c); frp[(y, c)] = (f, lg)
    pw = {"all earlier transitions": lambda y: pr.fit_constants(y - 1, 2023), "last two": lambda y: pr.fit_constants(y - 1, max(2023, y - 2)), "last one": lambda y: pr.fit_constants(y - 1, y - 1),
          "all four (leak for the scored season)": lambda y: pr.fit_constants(2026, 2023)}
    PV = {}
    for name, fn in pw.items():
        cs = {y: fn(y) for y in (2025, 2026)}
        print(f"   {name}: 2025 K% d {cs[2025]['k']['delta']} n0 {cs[2025]['k']['n0']:.0f} / BB% d {cs[2025]['bbr']['delta']} n0 {cs[2025]['bbr']['n0']:.0f} | 2026 K% d {cs[2026]['k']['delta']} n0 {cs[2026]['k']['n0']:.0f} / BB% d {cs[2026]['bbr']['delta']} n0 {cs[2026]['bbr']['n0']:.0f}")
        out = {"k": [], "bb": []}
        for (y, c), (f, lg) in frp.items():
            upd = pr.predict(f, lg, cs[y]); pre = pr.predict(f, lg, cs[y], use_ytd=False)
            for s, (ycol, key) in (("k", ("k_ros", "k")), ("bb", ("bb_ros", "bbr"))):
                out[s].append(pd.DataFrame({"y": f[ycol].to_numpy(), "n_ros": f["n_ros"].to_numpy(), "year": y, "cut": c, "id": (f.index + "|" + str(y)).to_numpy(), "upd": upd[key].to_numpy(), "pre": pre[key].to_numpy()}))
        PV[name] = {s: pd.concat(v, ignore_index=True) for s, v in out.items()}
    for s, lab in (("k", "K%"), ("bb", "BB%")):
        d0 = PV["all earlier transitions"][s]; w = ew.weights_from_base(d0["y"].to_numpy(), d0["upd"].to_numpy(), d0["n_ros"].to_numpy())[0]; y = d0["y"].to_numpy(); rows = []
        for name, Vn in PV.items():
            d = Vn[s]; r = {"variant": name, "preseason RMSE": wr(y, d["pre"].to_numpy(), w), "ROS update RMSE": wr(y, d["upd"].to_numpy(), w), "2025": wr(y[(d['year'] == 2025).to_numpy()], d["upd"].to_numpy()[(d['year'] == 2025).to_numpy()], w[(d['year'] == 2025).to_numpy()]),
                         "2026": wr(y[(d['year'] == 2026).to_numpy()], d["upd"].to_numpy()[(d['year'] == 2026).to_numpy()], w[(d['year'] == 2026).to_numpy()])}
            if name != "all earlier transitions":
                bd = boot_diff(y, d["upd"].to_numpy(), d0["upd"].to_numpy(), w, d0["id"].to_numpy()); r["vs all earlier"] = np.mean(bd); r["90% interval"] = ivl(bd)
            rows.append(r)
        print(f"  {lab}:"); print(pd.DataFrame(rows).round(5).to_string(index=False))

    # ------------------------------------------------------------------ C. term-coefficient training windows
    print("\n" + "=" * 110 + "\nC. TRAINING WINDOW OF THE PITCH-LEVEL TERM COEFFICIENTS\n" + "=" * 110)
    base_sp = specs["all earlier transitions"]
    grid = (0.0, 0.5, 1.0, 1.5, 2.0)
    G = {ts: hitter_variant(fr, base_sp, tscale=ts) for ts in grid}
    print("  HITTERS: scale on the production coefficients, chosen on a training slice, scored on the held-out slice (scale 1 = production)")
    slices = {"2025": lambda d: d["year"] == 2025, "2026": lambda d: d["year"] == 2026, "early cutoffs (<=40%)": lambda d: d["cut"] <= 0.4, "late cutoffs (>=50%)": lambda d: d["cut"] >= 0.5}
    plan = [("2025", "2026"), ("2026", "2025"), ("early cutoffs (<=40%)", "late cutoffs (>=50%)"), ("late cutoffs (>=50%)", "early cutoffs (<=40%)")]
    for s in HS:
        d1 = G[1.0][s]; w = ew.weights_from_base(d1["y"].to_numpy(), d1["upd"].to_numpy(), d1["n_ros"].to_numpy())[0]; y = d1["y"].to_numpy(); rows = []
        for tr, te in plan:
            mtr = slices[tr](d1).to_numpy(); mte = slices[te](d1).to_numpy()
            errs = {ts: wr(y[mtr], G[ts][s]["full"].to_numpy()[mtr], w[mtr]) for ts in grid}; best = min(errs, key=errs.get)
            e_best = wr(y[mte], G[best][s]["full"].to_numpy()[mte], w[mte]); e_prod = wr(y[mte], G[1.0][s]["full"].to_numpy()[mte], w[mte]); e_zero = wr(y[mte], G[0.0][s]["full"].to_numpy()[mte], w[mte])
            rows.append({"train -> test": f"{tr} -> {te}", "best scale on train": best, "test RMSE (best)": e_best, "(production, scale 1)": e_prod, "(no terms, scale 0)": e_zero})
        print(f"   {s}:"); print(pd.DataFrame(rows).round(5).to_string(index=False))
    print("  PITCHERS: coefficient (wh_sva_level -> K%, command_score -> BB%) fit on a training slice, scored on the held-out slice (change in RMSE vs the ROS update; negative = helps; P(helps))")
    cs26 = pr.fit_constants(2025, 2023); cs25 = pr.fit_constants(2024, 2023)
    R = {y: pd.read_parquet(f"output/pitcher_ros_stuff_rows_{y}.parquet") for y in (2025, 2026)}; cons = {2025: cs25, 2026: cs26}
    ALL = pd.concat([R[2025].assign(cy=2025), R[2026].assign(cy=2026)], ignore_index=True)
    pslices = {"2025": lambda d: d["year"] == 2025, "2026": lambda d: d["year"] == 2026, "early (cuts <= .30)": lambda d: d["cut"] <= 0.30, "late (cuts >= .60)": lambda d: d["cut"] >= 0.60}
    pplan = [("2025", "2026"), ("2026", "2025"), ("early (cuts <= .30)", "late (cuts >= .60)"), ("late (cuts >= .60)", "early (cuts <= .30)")]
    for stat, lab, vs, key in (("k", "K% wh_sva_level", ["wh_sva"], "k"), ("bb", "BB% command_score", ["command_score"], "bbr")):
        rows = []
        for tr, te in pplan:
            dtr = ALL[pslices[tr](ALL)].copy(); dte = ALL[pslices[te](ALL)].copy()
            n0tr = np.mean([cons[y][key]["n0"] for y in dtr["year"].unique()]); n0te = np.mean([cons[y][key]["n0"] for y in dte["year"].unique()])
            full, reps = tm.stage1(dtr, stat, vs, n0tr); point, ds = tm.stage2(dte, stat, vs, n0te, full, reps)
            rows.append({"train -> test": f"{tr} -> {te}", "coef": f"{full[1]:+.4f}", "point": point, "90% interval": ivl(ds), "P(helps)": float((ds < 0).mean())})
        print(f"   {lab}:"); print(pd.DataFrame(rows).round(5).to_string(index=False))

    # ------------------------------------------------------------------ D. history depth in the ROS update
    print("\n" + "=" * 110 + "\nD. HISTORY DEPTH inside the ROS update: earlier seasons entering the regression (0 = season to date only)\n" + "=" * 110)
    Dh = {f"{m} earlier season(s)": hitter_variant(fr, base_sp, mh=m) for m in (0, 1, 2, 3)}
    compare(Dh, "3 earlier season(s)", "HITTERS (reference = 3; with 0 the ROS update ignores all history)")
    Dp = {}
    for m in (0, 1, 2, 3, 4):
        out = {"k": [], "bb": []}
        for (y, c), (f, lg) in frp.items():
            upd = pr.predict(f, lg, cons[y], max_hist=m)
            for s, (ycol, key) in (("k", ("k_ros", "k")), ("bb", ("bb_ros", "bbr"))):
                out[s].append(pd.DataFrame({"y": f[ycol].to_numpy(), "n_ros": f["n_ros"].to_numpy(), "year": y, "cut": c, "id": (f.index + "|" + str(y)).to_numpy(), "upd": upd[key].to_numpy()}))
        Dp[m] = {s: pd.concat(v, ignore_index=True) for s, v in out.items()}
    for s, lab in (("k", "K%"), ("bb", "BB%")):
        d0 = Dp[4][s]; w = ew.weights_from_base(d0["y"].to_numpy(), d0["upd"].to_numpy(), d0["n_ros"].to_numpy())[0]; y = d0["y"].to_numpy(); rows = []
        for m in (0, 1, 2, 3, 4):
            d = Dp[m][s]; r = {"earlier seasons": m, "ROS update RMSE": wr(y, d["upd"].to_numpy(), w)}
            if m != 4:
                bd = boot_diff(y, d["upd"].to_numpy(), d0["upd"].to_numpy(), w, d0["id"].to_numpy()); r["vs 4"] = np.mean(bd); r["90% interval"] = ivl(bd)
            rows.append(r)
        print(f"  PITCHERS {lab}:"); print(pd.DataFrame(rows).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
