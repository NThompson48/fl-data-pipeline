"""
fl_ros_inseason_refit.py -- leak-free refit of the in-season strength of the ROS update, tuned on ROS targets (not season-to-season transitions).

Why: by-day study (fl_ros_by_day.py) found (i) pitchers K% / BB% want a weaker preseason prior in-season (n0 x ~.5), (ii) hitter BB% wants a lower in-season weight nu.
The preseason constants (n0, delta) were fit on full-season -> next-season transitions; the ROS update applies them to partial seasons.  One free parameter per stat, fit on ONE season's
as-of days (7, 14, ..., 105) with standard inverse-variance weights (frozen from the production predictions), scored on the OTHER season (forward 2025 -> 2026; reverse 2026 -> 2025 as a check).
Two-stage bootstrap: stage 1 resamples players in the fitting season and re-picks the parameter; stage 2 resamples players in the scored season.

A  Pitchers K%, BB%: n0_scale in {.25 ... 1.25}; update only, and the full model (update + wh_sva / command_score term, B refit on the fitting season's stuff rows) with the term's fade using the
   scaled n0 (matches the new update) vs the unscaled n0 (the old fade).
B  Hitter BB%: nu in {.1 ... 1.5} (final BB% includes the Swing% term); then the term scale re-checked given the new nu.
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_ros as hr
import fl_pitcher_ros as pr
import fl_ros_by_day as bd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2027)
SG = (0.25, 0.35, 0.5, 0.65, 0.75, 1.0, 1.25)
NG = (0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5)
R_FIT, R_EVAL = 200, 400


def wr(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def groups(ids):
    by = pd.Series(np.arange(len(ids))).groupby(ids).apply(np.array); return by, by.index.to_numpy()


def draw(by, ug):
    return np.concatenate([by[u] for u in rng.choice(ug, len(ug))])


# ---------------------------------------------------------------------------------------------------------------------------- pitchers
def pitcher_days(T, consts, grid):
    out = []
    for yr in (2025, 2026):
        t0, _ = T.season_bounds(yr)
        for d in bd.DAYS:
            f, lg = pr.features(T, yr, t0 + pd.Timedelta(days=d))
            if len(f) < 40:
                continue
            P = {s: pr.predict(f, lg, consts[yr], n0_scale=s) for s in grid}
            for stat, ycol, key in (("k", "k_ros", "k"), ("bb", "bb_ros", "bbr")):
                df = pd.DataFrame({"y": f[ycol].to_numpy(), "n": f["n_ros"].to_numpy(), "id": (f.index + "|" + str(yr)).to_numpy(), "year": yr, "stat": stat})
                for s in grid:
                    df[s] = P[s][key].to_numpy()
                out.append(df)
    return pd.concat(out, ignore_index=True)


def pick(d, grid, idx=None):
    idx = np.arange(len(d)) if idx is None else idx
    y = d["y"].to_numpy()[idx]; w = d["w"].to_numpy()[idx]
    return min(grid, key=lambda s: wr(y, d[s].to_numpy()[idx], w))


def stuff_frames(T, consts, grid, R):
    """For each year: the stuff rows with the base recomputed at every n0_scale in the grid (rows are in cutoff order, aligned on the pitcher key)."""
    out = {}
    for yr in (2025, 2026):
        r = R[yr].copy(); t0, t1 = T.season_bounds(yr); parts = []
        for c in sorted(r["cut"].unique()):
            sub = r[r["cut"] == c].copy(); f, lg = pr.features(T, yr, t0 + (t1 - t0) * c); keys = f.index + "|" + str(yr)
            assert (keys.to_numpy() == sub["pitcher"].to_numpy()).all()
            for s in grid:
                p = pr.predict(f, lg, consts[yr], n0_scale=s)
                sub[f"bk_{s}"] = p["k"].to_numpy(); sub[f"bb_{s}"] = p["bbr"].to_numpy()
            parts.append(sub)
        out[yr] = pd.concat(parts, ignore_index=True)
    return out


def full_pred(R, stat, s, B, n0, fade_scaled):
    """update (n0 x s) + B x deviation x fade; fade built from the scaled or the unscaled n0."""
    base = R[f"b{'k' if stat == 'k' else 'b'}_{s}"].to_numpy(); n0f = n0 * s if fade_scaled else n0
    dev = R["wh_sva" if stat == "k" else "command_score"].to_numpy(); fade = n0f / (R[f"pa_eff_{stat}"].to_numpy() + n0f)
    return base, dev * fade


def fit_B(R, stat, s, n0, fade_scaled, idx, w):
    y = R[f"y_{stat}"].to_numpy()[idx]; base, term = full_pred(R, stat, s, None, n0, fade_scaled); X = np.column_stack([np.ones(len(idx)), term[idx]]); sw = np.sqrt(w[idx])
    return np.linalg.lstsq(X * sw[:, None], (y - base[idx]) * sw, rcond=None)[0][1]


def eval_rmse(R, stat, s, B, n0, fade_scaled, idx, w, with_term=True):
    y = R[f"y_{stat}"].to_numpy()[idx]; base, term = full_pred(R, stat, s, None, n0, fade_scaled); p = base[idx] + (B * term[idx] if with_term else 0.0)
    c = np.average(y - p, weights=w[idx])                 # CV intercept in every model
    return wr(y, p + c, w[idx])


def pitcher_study(T, consts, R):
    day = pitcher_days(T, consts, SG); SF = stuff_frames(T, consts, SG, R)
    for tag, tr, te in (("FORWARD (fit on 2025 -> score 2026)", 2025, 2026), ("reverse check (fit on 2026 -> score 2025)", 2026, 2025)):
        print(f"\n{'=' * 110}\nPITCHERS {tag}\n{'=' * 110}")
        for stat, lab in (("k", "K%"), ("bb", "BB%")):
            d = day[day["stat"] == stat].copy(); d["w"] = ew.weights_from_base(d["y"].to_numpy(), d[1.0].to_numpy(), d["n"].to_numpy())[0]
            dtr, dte = d[d["year"] == tr].reset_index(drop=True), d[d["year"] == te].reset_index(drop=True)
            tab = pd.DataFrame({"train RMSE": {s: wr(dtr["y"], dtr[s], dtr["w"]) for s in SG}, "test RMSE": {s: wr(dte["y"], dte[s], dte["w"]) for s in SG}}); tab.index.name = "n0_scale"
            s_star = pick(dtr, SG); by_tr, ug_tr = groups(dtr["id"].to_numpy()); reps_s = np.array([pick(dtr, SG, draw(by_tr, ug_tr)) for _ in range(R_FIT)])
            print(f"\n {lab}: update-only error by n0_scale (frozen weights)  ->  chosen on the fitting season: n0_scale = {s_star}; stage-1 bootstrap choices: " +
                  ", ".join(f"{v}: {np.mean(reps_s == v):.0%}" for v in SG if (reps_s == v).any()))
            print(tab.round(5).to_string())
            key = "k" if stat == "k" else "bbr"; n0tr, n0te = consts[tr][key]["n0"], consts[te][key]["n0"]
            Rtr, Rte = SF[tr], SF[te]
            wtr = ew.weights_from_base(Rtr[f"y_{stat}"].to_numpy(), Rtr[f"b{'k' if stat == 'k' else 'b'}_1.0"].to_numpy(), Rtr["n_ros"].to_numpy())[0]
            wte = ew.weights_from_base(Rte[f"y_{stat}"].to_numpy(), Rte[f"b{'k' if stat == 'k' else 'b'}_1.0"].to_numpy(), Rte["n_ros"].to_numpy())[0]
            by_s, ug_s = groups(Rtr["pitcher"].to_numpy()); by_e, ug_e = groups(Rte["pitcher"].to_numpy()); allte = np.arange(len(Rte))
            variants = {"production (n0 x 1, unscaled fade)": (lambda s: 1.0, False, True),
                        "update only, refit n0": (lambda s: s, False, False),
                        "refit n0 + term, UNSCALED fade": (lambda s: s, False, True),
                        "refit n0 + term, SCALED fade": (lambda s: s, True, True)}
            B_pt = {k: fit_B(Rtr, stat, v[0](s_star), n0tr, v[1], np.arange(len(Rtr)), wtr) for k, v in variants.items()}
            pt = {k: eval_rmse(Rte, stat, v[0](s_star), B_pt[k], n0te, v[1], allte, wte, v[2]) for k, v in variants.items()}
            ref = pt["production (n0 x 1, unscaled fade)"]; rows = []; draws = {k: [] for k in variants}
            # two-stage bootstrap of the change vs production
            for _ in range(R_EVAL):
                s_r = reps_s[rng.integers(0, len(reps_s))]; itr = draw(by_s, ug_s); ie = draw(by_e, ug_e); r0 = None
                for k, v in variants.items():
                    B_r = fit_B(Rtr, stat, v[0](s_r), n0tr, v[1], itr, wtr)
                    draws[k].append(eval_rmse(Rte, stat, v[0](s_r), B_r, n0te, v[1], ie, wte, v[2]))
                    if k.startswith("production"):
                        r0 = draws[k][-1]
                for k in variants:
                    draws[k][-1] -= r0
            for k in variants:
                dd = np.array(draws[k]); rows.append({"model": k, "B (fit season)": B_pt[k], "scored RMSE": pt[k], "change vs production": pt[k] - ref, "90% interval": f"[{np.percentile(dd, 5):+.5f}, {np.percentile(dd, 95):+.5f}]", "P(better)": float((dd < 0).mean())})
            print(f"\n {lab}: scored season, stuff rows (5 cutoffs, {len(Rte)} pitcher-cutoffs):"); print(pd.DataFrame(rows).round(5).to_string(index=False))


# ---------------------------------------------------------------------------------------------------------------------------- hitter BB%
def hitter_days(T, grid_nu, tgrid):
    out = []
    for yr in (2025, 2026):
        t0, _ = T.season_bounds(yr)
        for d in bd.DAYS:
            f, lg = hr.features(T, yr, t0 + pd.Timedelta(days=d), min_ros_pa=30)
            if len(f) < 40:
                continue
            df = pd.DataFrame({"y": f["bb_ros"].to_numpy(), "n": f["n_ros"].to_numpy(), "id": (f.index + "|" + str(yr)).to_numpy(), "year": yr})
            for nu in grid_nu:
                df[f"nu={nu}"] = hr.predict(f, lg, nu=nu)["bbr_final"].to_numpy()
            for nu in (0.5, 0.3):
                for ts in tgrid:
                    df[f"nuts={nu}|{ts}"] = hr.predict(f, lg, nu=nu, params={"trackman_scale": ts})["bbr_final"].to_numpy()
            out.append(df)
    return pd.concat(out, ignore_index=True)


def hitter_study(T):
    tg = (0.0, 0.5, 1.0, 1.5, 2.0); d = hitter_days(T, NG, tg)
    d["w"] = ew.weights_from_base(d["y"].to_numpy(), d["nu=1.0"].to_numpy(), d["n"].to_numpy())[0]
    for tag, tr, te in (("FORWARD (fit on 2025 -> score 2026)", 2025, 2026), ("reverse check (fit on 2026 -> score 2025)", 2026, 2025)):
        a, b = d[d["year"] == tr].reset_index(drop=True), d[d["year"] == te].reset_index(drop=True)
        print(f"\n{'=' * 110}\nHITTER BB% {tag}\n{'=' * 110}")
        tab = pd.DataFrame({"train RMSE": {nu: wr(a["y"], a[f"nu={nu}"], a["w"]) for nu in NG}, "test RMSE": {nu: wr(b["y"], b[f"nu={nu}"], b["w"]) for nu in NG}}); tab.index.name = "nu"
        pk = lambda dd, idx=None: min(NG, key=lambda nu: wr(dd["y"].to_numpy()[idx if idx is not None else slice(None)], dd[f"nu={nu}"].to_numpy()[idx if idx is not None else slice(None)], dd["w"].to_numpy()[idx if idx is not None else slice(None)]))
        nu_star = pk(a); by_a, ug_a = groups(a["id"].to_numpy()); by_b, ug_b = groups(b["id"].to_numpy()); reps = np.array([pk(a, draw(by_a, ug_a)) for _ in range(R_FIT)])
        print(tab.round(5).to_string()); print(f"  chosen on the fitting season: nu = {nu_star}; stage-1 bootstrap: " + ", ".join(f"{v}: {np.mean(reps == v):.0%}" for v in NG if (reps == v).any()))
        dd = []
        for _ in range(R_EVAL):
            nr = reps[rng.integers(0, len(reps))]; ie = draw(by_b, ug_b); y = b["y"].to_numpy()[ie]; w = b["w"].to_numpy()[ie]
            dd.append(wr(y, b[f"nu={nr}"].to_numpy()[ie], w) - wr(y, b["nu=1.0"].to_numpy()[ie], w))
        dd = np.array(dd); print(f"  scored season: nu = {nu_star} vs production nu 1: {wr(b['y'], b[f'nu={nu_star}'], b['w']) - wr(b['y'], b['nu=1.0'], b['w']):+.5f}  two-stage 90% [{np.percentile(dd, 5):+.5f}, {np.percentile(dd, 95):+.5f}]  P(better) {np.mean(dd < 0):.2f}")
        for nu in (0.5, 0.3):
            print(f"  Swing% term scale given nu {nu} (train / test RMSE): " + "; ".join(f"x{ts}: {wr(a['y'], a[f'nuts={nu}|{ts}'], a['w']):.5f} / {wr(b['y'], b[f'nuts={nu}|{ts}'], b['w']):.5f}" for ts in tg))


def main():
    T = pr.Tables(); consts = {2025: pr.fit_constants(2024), 2026: pr.fit_constants(2025)}
    R = {y: pd.read_parquet(f"output/pitcher_ros_stuff_rows_{y}.parquet") for y in (2025, 2026)}
    pitcher_study(T, consts, R)
    hitter_study(hr.Tables())


if __name__ == "__main__":
    main()
