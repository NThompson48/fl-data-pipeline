"""
fl_ros_fbcmd_test.py -- follow-up checks on the FASTBALL-BLEND command term  (user 2026-10-27: "run checks")
Candidate (from fl_fastball_command_test.py): BB% command term = average of the all-pitch Command Score and the frozen-formula score computed on the pitcher's four-seams and sinkers (>= 50 fastballs in the window, else the all-pitch score alone).
CHECK 1 -- ROS (in-season) BB% power prior, the exact design of fl_pitcher_ros_stuff.py: at each of 5 cutoffs (15-75% of the season) the pool = last season (full) + this season before the cutoff; the measure is shrunk by its own reliability (random-half n0, recomputed here for the blend and the fastball score) and enters as B x shrunk deviation x n0stat / (PA_eff + n0stat);
   B fit on one season's ROS rows, frozen, scored on the other (FORWARD 2025 -> 2026; REVERSE 2026 -> 2025); pitcher-cluster two-stage bootstrap.  Compared: all-pitch command_score (production), blend, command_score + fastball score as two terms.  Reported: the change in RMSE vs the ROS update (as the earlier test), AND the PAIRED difference blend minus production (same bootstrap draws).
CHECK 2 -- STICKINESS of the blend itself (2025 vs 2026 regular season) at >= 150 / 300 / 500 / 800 pitches (all-pitch) next to the production score.
Output: output/ros_fbcmd_test.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_pitcher_ros as pr
import fl_pitcher_ros_stuff as rs

warnings.filterwarnings("ignore")
rng = np.random.default_rng(405)
MIN_FB = 50; FB = list(fp.FASTBALL_TYPES); R_FIT = 400; R_EVAL = 1500
CH, CV_, I0 = fp.COMMAND_SCORE_HORIZ_COEF, fp.COMMAND_SCORE_VERT_COEF, fp.COMMAND_SCORE_INTERCEPT


def scores(d, min_fb=MIN_FB):
    """per pitcher: command_score (all pitches), fb_cmd (frozen formula on four-seams + sinkers, NaN if < min_fb), cmd_blend, n_pitch, n_fb"""
    a = fp.compute_command_score(d).set_index("pitcher_id")[["command_score", "n_pitches"]].rename(columns={"n_pitches": "n_pitch"})
    f = fp.compute_loc_var(d[d["pitch_type"].isin(FB)], min_pitches=min_fb).set_index("pitcher_id")
    f["fb_cmd"] = I0 + CH * f["horiz_var"].clip(*fp.COMMAND_SCORE_HORIZ_RANGE) + CV_ * f["vert_var"].clip(*fp.COMMAND_SCORE_VERT_RANGE); out = a.join(f[["fb_cmd", "n_pitches"]].rename(columns={"n_pitches": "n_fb"}), how="left")
    out["cmd_blend"] = np.where(out["fb_cmd"].notna(), (out["command_score"] + out["fb_cmd"]) / 2, out["command_score"]); out["n_fb"] = out["n_fb"].fillna(0); return out


def reliability(raw):
    res = {"command_score": [], "fb_cmd": [], "cmd_blend": []}; sz = {"n_pitch": [], "n_fb": []}
    for yr in (2025, 2026):
        d = fp.filter_period(raw, *rs.SEASON[yr]); cnt = d.groupby("pitcher_id").size(); d = d[d["pitcher_id"].isin(cnt[cnt >= 400].index)]
        key = d["game_id"].astype(str) + "|" + d["pa_id"].astype(str); codes, uniq = pd.factorize(key)
        for _ in range(8):
            hh = rng.integers(0, 2, len(uniq))[codes]; a = scores(d[hh == 0], 25); b = scores(d[hh == 1], 25); j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            for c in res:
                jj = j[[f"{c}_a", f"{c}_b"]].dropna(); res[c].append(jj.iloc[:, 0].corr(jj.iloc[:, 1]))
            sz["n_pitch"].append(float(j["n_pitch_a"].mean())); sz["n_fb"].append(float(j.loc[j["n_fb_a"] > 0, "n_fb_a"].mean()))
    out = {}
    for c, nk in (("command_score", "n_pitch"), ("fb_cmd", "n_fb"), ("cmd_blend", "n_pitch")):
        r = float(np.mean(res[c])); out[c] = (max(float(np.mean(sz[nk])) * (1 - r) / r, 1.0), r)
    return out


def features(raw, T, year, asof, names, n0s):
    pool = fp.filter_period(raw, rs.SEASON[year][0], (pd.Timestamp(asof) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    if year - 1 >= 2025:
        pool = pd.concat([fp.filter_period(raw, *rs.SEASON[year - 1]), pool], ignore_index=True)
    s = scores(pool); out = pd.DataFrame(index=s.index)
    for c, nk in (("command_score", "n_pitch"), ("cmd_blend", "n_pitch"), ("fb_cmd", "n_fb")):
        ok = s[c].notna() & (s[nk] > 0); lg = np.average(s.loc[ok, c], weights=s.loc[ok, nk]); out[c] = ((s[c] - lg) * s[nk] / (s[nk] + n0s[c][0])).where(ok)
    out["name_clean"] = out.index.map(names); return out.dropna(subset=["name_clean"]).groupby("name_clean")[["command_score", "cmd_blend", "fb_cmd"]].mean()


def attach(R, raw, T, year, names, n0s):
    t0, t1 = T.season_bounds(year); R = R.copy(); R["name"] = R["pitcher"].str.rsplit("|", n=1).str[0]; parts = []
    for c, g in R.groupby("cut"):
        asof = t0 + (t1 - t0) * c; f = features(raw, T, year, asof, names, n0s).reindex(g["name"]).fillna(0.0); g = g.copy()
        for col in f.columns:
            g[col] = f[col].to_numpy()
        parts.append(g); print(f"   features {year} cutoff {c:.2f} done", flush=True)
    return pd.concat(parts, ignore_index=True)


def paired(Rtr, Rte, spec_a, spec_b, n0_tr, n0_te):
    """paired two-stage bootstrap: dRMSE(model with terms spec_b) - dRMSE(model with terms spec_a), both vs the ROS update, same pitcher draws and same stage-1 replicate."""
    def parts(R, n0):
        y, base = R["y_bb"].to_numpy(), R["base_bb"].to_numpy(); w, _, _ = ew.weights_from_base(y, base, R["n_ros"].to_numpy()); fade = n0 / (R["pa_eff_bb"].to_numpy() + n0); return y, base, w, fade
    y, base, w, fade = parts(Rtr, n0_tr); Y = y - base; ug = Rtr["pitcher"].unique(); by = pd.Series(np.arange(len(Rtr))).groupby(Rtr["pitcher"].to_numpy()).apply(np.array); fits = {}
    for nm, vs in (("a", spec_a), ("b", spec_b)):
        X = np.column_stack([np.ones(len(y))] + [Rtr[v].to_numpy() * fade for v in vs]); f = lambda idx, X=X: np.linalg.lstsq(X[idx] * np.sqrt(w[idx])[:, None], Y[idx] * np.sqrt(w[idx]), rcond=None)[0]; fits[nm] = (f, vs)
    draws = [np.concatenate([by[u] for u in rng.choice(ug, len(ug))]) for _ in range(R_FIT)]; coefs = {nm: [fits[nm][0](d) for d in draws] for nm in fits}; full = {nm: fits[nm][0](np.arange(len(y))) for nm in fits}
    y, base, w, fade = parts(Rte, n0_te); g = Rte["pitcher"].to_numpy(); ug = pd.Series(g).unique(); by = pd.Series(np.arange(len(g))).groupby(g).apply(np.array); rm = lambda a, p, ww: float(np.sqrt(np.average((a - p) ** 2, weights=ww)))
    term = {nm: (lambda coef, vs=fits[nm][1]: sum(coef[1 + i] * Rte[v].to_numpy() * fade for i, v in enumerate(vs))) for nm in fits}
    def delta(idx, nm, coef):
        m1 = base[idx] + term[nm](coef)[idx]; c0 = np.average(y[idx] - base[idx], weights=w[idx]); c1 = np.average(y[idx] - m1, weights=w[idx]); return rm(y[idx], m1 + c1, w[idx]) - rm(y[idx], base[idx] + c0, w[idx])
    allx = np.arange(len(y)); pt = {nm: delta(allx, nm, full[nm]) for nm in fits}; ds = {"a": [], "b": []}
    for _ in range(R_EVAL):
        idx = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); k = rng.integers(0, R_FIT)
        for nm in fits:
            ds[nm].append(delta(idx, nm, coefs[nm][k]))
    a, b = np.array(ds["a"]), np.array(ds["b"]); diff = b - a
    return {"point a": pt["a"], "point b": pt["b"], "point b-a": pt["b"] - pt["a"], "P(a helps)": float((a < 0).mean()), "P(b helps)": float((b < 0).mean()), "diff 90%": f"[{np.percentile(diff, 5):+.5f}, {np.percentile(diff, 95):+.5f}]", "P(b better than a)": float((diff < 0).mean())}


def stickiness(raw):
    S = {yr: scores(fp.filter_period(raw, *rs.SEASON[yr])) for yr in (2025, 2026)}; rows = []
    for c in ("command_score", "fb_cmd", "cmd_blend"):
        r = {"measure": c}
        for thr in (150, 300, 500, 800):
            j = pd.concat([S[2025][c].rename("a"), S[2026][c].rename("b"), S[2025]["n_pitch"].rename("n25"), S[2026]["n_pitch"].rename("n26")], axis=1).dropna(); j = j[(j["n25"] >= thr) & (j["n26"] >= thr)]
            rr = float(np.corrcoef(j["a"], j["b"])[0, 1]); z = np.arctanh(rr); se = 1 / np.sqrt(len(j) - 3); r[f"r>={thr}"] = rr; r[f"CI{thr}"] = f"[{np.tanh(z - 1.96 * se):.2f},{np.tanh(z + 1.96 * se):.2f}]"; r[f"n{thr}"] = len(j)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250); raw = fp.load_fl_pitches_cached(); names = rs.idmap(raw); T = pr.Tables()
    St = stickiness(raw); print("CHECK 2: STICKINESS (2025 vs 2026)\n" + St.round(3).to_string(index=False), flush=True)
    n0s = reliability(raw); print("\nreliability (random-half r, n0): " + str({k: (round(v[1], 2), round(v[0], 0)) for k, v in n0s.items()}), flush=True)
    consts = {2025: pr.fit_constants(2024), 2026: pr.fit_constants(2025)}; R = {}
    for y in (2025, 2026):
        R[y] = attach(pd.read_parquet(f"output/pitcher_ros_stuff_rows_{y}.parquet"), raw, T, y, names, n0s)
    res = []
    for tag, tr, te in (("FORWARD (fit 2025 ROS rows, score 2026)", 2025, 2026), ("REVERSE (fit 2026, score 2025)", 2026, 2025)):
        n0tr, n0te = consts[tr]["bbr"]["n0"], consts[te]["bbr"]["n0"]; print(f"\n{'=' * 90}\n{tag}")
        for lab, a, b in (("blend vs production", ["command_score"], ["cmd_blend"]), ("command_score + fastball score vs production", ["command_score"], ["command_score", "fb_cmd"])):
            o = paired(R[tr], R[te], a, b, n0tr, n0te); o = {"design": tag, "comparison": lab, **o}; res.append(o); print(f"  {lab}: " + "; ".join(f"{k} {v:+.5f}" if isinstance(v, float) and 'P(' not in k else f"{k} {v}" for k, v in o.items() if k not in ('design', 'comparison')), flush=True)
    json.dump({"stickiness": St.to_dict("records"), "n0": {k: list(v) for k, v in n0s.items()}, "ros": res}, open("output/ros_fbcmd_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
