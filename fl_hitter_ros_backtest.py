"""
fl_hitter_ros_backtest.py -- walk-forward backtest of the hitter rest-of-season (ROS) model, 2025 and 2026, chronological cutoffs.

For each season and cutoff (15 / 30 / 45 / 60 / 75% of the season by date) every hitter with >= 30 PA after the cutoff is forecast from information BEFORE the cutoff only
(history from earlier seasons + this season's PA and pitches to date) and scored on his wOBA / K% / BB% over the PA after the cutoff, with the standard inverse-variance weights
(law fit on the Marcel-update residuals, then frozen).  Models:
  0 league average now          1 preseason only (history, no in-season data)          2 YTD at face value          3 YTD naive shrinkage (one n0, ignores history)
  4 ROS update (production constants, in-season PA = newest season)                    5 = 4 + pitch-level terms (xwOBA nudge + EV90 for wOBA, Whiff% for K%, Swing% for BB%)
  6 = 5 with the in-season weight nu and the Trackman scale refit on the OTHER season (forward: fit 2025 -> score 2026; the reverse is reported as a check, not as evidence)
Usage:  python fl_hitter_ros_backtest.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_ros as ros

warnings.filterwarnings("ignore")
rng = np.random.default_rng(12)
CUTS = (0.15, 0.30, 0.45, 0.60, 0.75)
STATS = {"woba": ("woba_ros", "woba_final", "woba"), "k": ("k_ros", "k_final", "k"), "bbr": ("bb_ros", "bbr_final", "bbr")}


def wrmse(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def build(T):
    rows = []
    for year in (2025, 2026):
        t0, t1 = T.season_bounds(year)
        for c in CUTS:
            asof = t0 + (t1 - t0) * c
            f, lg = ros.features(T, year, asof)
            f["year"], f["cut"], f["asof"] = year, c, asof
            rows.append((f, lg))
    return rows


def run_models(f, lg, **kw):
    """All model predictions for one (season, cutoff) frame -> dict stat -> dict model -> array."""
    base = ros.predict(f, lg, trackman=False, **{k: v for k, v in kw.items() if k in ("nu", "n0_scale")})
    full = ros.predict(f, lg, trackman=True, **kw)
    pre = ros.predict(f, lg, trackman=False, use_ytd=False)
    out = {}
    ytd = {"woba": (f["woba_sum"], f["n"]), "k": (f["k"], f["n"]), "bbr": (f["bb"], f["n"])}
    for s, (ycol, fcol, key) in STATS.items():
        num, den = ytd[s]; x = np.where(den > 0, num / den.replace(0, np.nan), lg[key])
        n0 = ros.SPEC[key][1]
        out[s] = {"0 league average now": np.full(len(f), lg[key]), "1 preseason only": pre["blend"].to_numpy() if s == "woba" else pre[key].to_numpy(),
                  "2 YTD face value": x, "3 YTD naive shrinkage": lg[key] + den.to_numpy() * (x - lg[key]) / (den.to_numpy() + n0),
                  "4 ROS update (no Trackman)": base["blend"].to_numpy() if s == "woba" else base[key].to_numpy(), "5 + pitch-level terms": full[fcol].to_numpy()}
    return out


def collect(frames, **kw):
    recs = {s: [] for s in STATS}
    for f, lg in frames:
        m = run_models(f, lg, **kw)
        for s, (ycol, _, _) in STATS.items():
            d = pd.DataFrame({"y": f[ycol].to_numpy(), "n_ros": f["n_ros"].to_numpy(), "year": f["year"].to_numpy(), "cut": f["cut"].to_numpy(), "hitter": (f.index + "|" + f["year"].astype(str)).to_numpy(), "ytd_pa": f["n"].to_numpy()})
            for k, v in m[s].items():
                d[k] = v
            recs[s].append(d)
    return {s: pd.concat(v, ignore_index=True) for s, v in recs.items()}


def grid_refit(frames_train, stat, nus=(1.0, 1.5, 2.0, 3.0), scales=(0.0, 0.5, 1.0, 1.5)):
    best = None
    for nu in nus:
        for sc in scales:
            D = collect(frames_train, nu=nu, params={"trackman_scale": sc})[stat]
            w, _, _ = ew.weights_from_base(D["y"].to_numpy(), D["4 ROS update (no Trackman)"].to_numpy(), D["n_ros"].to_numpy())
            e = wrmse(D["y"].to_numpy(), D["5 + pitch-level terms"].to_numpy(), w)
            if best is None or e < best[0]:
                best = (e, nu, sc)
    return best[1], best[2]


def main():
    T = ros.Tables()
    frames = build(T)
    print("Rows per (season, cutoff):", {(int(f['year'].iloc[0]), c): len(f) for (f, _), c in zip(frames, CUTS * 2)})
    D = collect(frames)
    models = list(next(iter(D.values())).columns)
    models = [m for m in models if m[0].isdigit()]
    for s in STATS:
        d = D[s]
        w, a, b = ew.weights_from_base(d["y"].to_numpy(), d["4 ROS update (no Trackman)"].to_numpy(), d["n_ros"].to_numpy()); d["w"] = w
        print(f"\n{'=' * 100}\n{s.upper()} (ROS target: {STATS[s][0]}; n = {len(d)} hitter-cutoffs; variance law a={a:.5f}, b={b:.4f})")
        rows = []
        for m in models:
            r = {"model": m, "RMSE all": wrmse(d["y"].to_numpy(), d[m].to_numpy(), w), "mean err": float(np.average(d["y"] - d[m], weights=w))}
            for c in CUTS:
                mk = (d["cut"] == c).to_numpy(); r[f"{int(c*100)}%"] = wrmse(d["y"].to_numpy()[mk], d[m].to_numpy()[mk], w[mk])
            rows.append(r)
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        ug = pd.Series(d["hitter"]).unique(); by = pd.Series(np.arange(len(d))).groupby(d["hitter"].to_numpy()).apply(np.array)
        print("   paired bootstrap of RMSE differences (90% interval, hitter clusters):")
        for A, B in (("4 ROS update (no Trackman)", "1 preseason only"), ("4 ROS update (no Trackman)", "3 YTD naive shrinkage"), ("5 + pitch-level terms", "4 ROS update (no Trackman)")):
            dd = []
            for _ in range(500):
                ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wrmse(d["y"].to_numpy()[ii], d[A].to_numpy()[ii], w[ii]) - wrmse(d["y"].to_numpy()[ii], d[B].to_numpy()[ii], w[ii]))
            print(f"     {A:30s} minus {B:28s}: {np.mean(dd):+.5f}  [{np.percentile(dd,5):+.5f}, {np.percentile(dd,95):+.5f}]")
        # by YTD sample size
        d["bk"] = pd.cut(d["ytd_pa"], [-1, 29, 99, 10 ** 5], labels=["< 30 PA YTD", "30-99", "100+"])
        t = d.groupby("bk", observed=True).apply(lambda g: pd.Series({"n": len(g), **{m.split(" ", 1)[0] + " " + m.split(" ", 1)[1][:16]: wrmse(g["y"].to_numpy(), g[m].to_numpy(), g["w"].to_numpy()) for m in ("1 preseason only", "3 YTD naive shrinkage", "4 ROS update (no Trackman)", "5 + pitch-level terms")}}))
        print("   by the hitter's PA so far this season (RMSE):"); print(t.round(4).to_string())

    # ---- forward refit: fit nu and the Trackman scale on 2025 cutoffs, score on 2026 --------------------------------------------------------
    print(f"\n{'=' * 100}\nFORWARD REFIT: in-season weight nu and the Trackman-term scale fit on 2025 cutoffs, scored on 2026 (the reverse shown as a check only)")
    f25 = [fr for fr in frames if int(fr[0]["year"].iloc[0]) == 2025]; f26 = [fr for fr in frames if int(fr[0]["year"].iloc[0]) == 2026]
    for tag, tr, te in (("2025 -> 2026", f25, f26), ("2026 -> 2025 (check)", f26, f25)):
        print(f"\n  {tag}")
        for s in STATS:
            nu, sc = grid_refit(tr, s)
            Dt = collect(te, nu=nu, params={"trackman_scale": sc})[s]; D0 = collect(te)[s]
            w, _, _ = ew.weights_from_base(D0["y"].to_numpy(), D0["4 ROS update (no Trackman)"].to_numpy(), D0["n_ros"].to_numpy())
            y = D0["y"].to_numpy()
            print(f"    {s:5s}: fitted nu={nu}, Trackman scale={sc} | RMSE on the held-out season: ROS update {wrmse(y, D0['4 ROS update (no Trackman)'].to_numpy(), w):.4f}; + terms (production constants) {wrmse(y, D0['5 + pitch-level terms'].to_numpy(), w):.4f}; "
                  f"+ terms (refit) {wrmse(y, Dt['5 + pitch-level terms'].to_numpy(), w):.4f}")


if __name__ == "__main__":
    main()
