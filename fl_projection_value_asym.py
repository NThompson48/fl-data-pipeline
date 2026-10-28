"""
fl_projection_value_asym.py -- DOES "WE WOULD RATHER BE SURPRISED UPWARD THAN DOWNWARD" CHANGE THE VALUE OF THE PROJECTIONS?  (user 2026-10-28)
Same veteran backtest as fl_projection_value_runs.py (2025 -> 2026; 126 hitters, 87 pitchers; rates at a fixed workload, runs = wOBA / 1.0175 x 400 PA, RA9 x 100 IP / 9).  Two ways asymmetry can matter:
 A  HOW THE PROJECTIONS ARE SCORED.  Symmetric RMSE / MAE treat over- and under-projection alike.  Here an over-projection (the player turns out WORSE than projected) costs W times an under-projection (W = 2, 3):
    weighted asymmetric squared error, same frozen standard weights.  Also: the best CONSERVATIVE SHIFT (project every player s lower; the level-free scoring of the main test removes any such shift) and the loss it reaches.
 B  HOW PLAYERS ARE PICKED.  If a player who disappoints can be benched and replaced by an average player (an OPTION on the downside), uncertainty is valuable.  Picks by MEAN (the projection) vs by UPSIDE (projection + z x SD, SD from the variance law of the model's own
    out-of-sample errors as a function of the player's last-year sample: var = a + b / n_last) are scored (i) at their realised rate (linear value) and (ii) with a downside option: a pick whose realised rate lands below the pool's 25th percentile is replaced by the pool median
    (hindsight benching: an UPPER BOUND on what in-season benching could capture, since real benching sees only part of the season).  Hitters 9 of 40, pitchers 5 of 30 candidates; runs per season vs random picks; players bootstrapped.
Output: output/projection_value_asym_log.txt, output/projection_value_asym.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_pitcher_features as fp
import fl_projection_value_runs as pr

warnings.filterwarnings("ignore")
SC = pr.SC


def frames():
    H = pr.hitters(); h = pd.read_parquet("output/hitter_war_calibration_2026.parquet"); h = h[(h["pa_t"] >= 20) & (h["pa26"] >= 100)].dropna(subset=["proj_woba", "woba_n", "woba26"]); H["hist"] = h["pa_t"].to_numpy()
    import fl_pwar_calibration as cal
    cal.RETURN_ALL = True; b, war = cal.main(); b = b[b["ip_25"].notna() & (b["ip_25"] >= 5) & (b["ip"] >= 30)].copy(); w25 = war[war["year"] == 2025].set_index("player"); b["ra9_25"] = b["player"].map(w25["ra9"]); b = b.dropna(subset=["ra9_25", "pred", "target"])
    P, n0 = pr.pitchers(); P["hist"] = b["ip_25"].to_numpy(); return H, P


def sd_law(D):
    """var(error of MODEL) = a + b / history, by weighted least squares of squared (level-free) errors on 1 / history"""
    e = D["y"].to_numpy() - D["MODEL"].to_numpy(); e = e - e.mean(); X = np.column_stack([np.ones(len(e)), 1 / D["hist"].to_numpy(float)]); beta = np.linalg.lstsq(X, e ** 2, rcond=None)[0]; beta[0] = max(beta[0], 1e-6); beta[1] = max(beta[1], 0.0); return np.sqrt(beta[0] + beta[1] / D["hist"].to_numpy(float)), beta


def asym_loss(D, scale, W):
    y = D["y"].to_numpy(); n = D["n"].to_numpy(float); w = ew.weights_from_base(y, D["MODEL"].to_numpy(), n)[0]; out = {}
    for m, q in (("league average", np.full(len(y), np.average(y, weights=w))), ("last year at face value", D["FACE"].to_numpy()), ("last year SHRUNK", D["SHRUNK"].to_numpy()), ("MODEL", D["MODEL"].to_numpy())):
        q = q + np.average(y - q, weights=w); best = None
        for s in np.linspace(0, 0.6 * np.std(y), 61):
            e = (q - s) - y; loss = np.sqrt(np.average(np.where(e > 0, W, 1.0) * e ** 2, weights=w)) * scale             # over-projection (e > 0) costs W times
            if s == 0:
                l0 = loss
            if best is None or loss < best[0]:
                best = (loss, s)
        out[m] = {"loss at no shift": float(l0), "best loss": float(best[0]), "best shift (runs)": float(best[1] * scale)}
    return out


def pick(D, kind, K, M, rng, reps, z, floor_q=None):
    """realised runs of K picks chosen by MODEL mean (z = 0), MODEL upside (z > 0), FACE, SHRUNK; optional hindsight floor"""
    sgn = 1 if kind == "h" else -1; scale = pr.runs_scale(kind); v = sgn * D["y"].to_numpy() * scale; sd = D["sd"].to_numpy() * scale
    S = {"MODEL mean": sgn * D["MODEL"].to_numpy() * scale, "FACE": sgn * D["FACE"].to_numpy() * scale, "SHRUNK": sgn * D["SHRUNK"].to_numpy() * scale}; S["MODEL upside z=%.1f" % z] = S["MODEL mean"] + z * sd; n = len(D); out = np.zeros((reps, len(S)))
    for r in range(reps):
        i = rng.choice(n, min(M, n), replace=False); vv = v[i]; base = vv.mean()
        if floor_q is not None:
            lo = np.percentile(vv, 100 * floor_q); med = np.median(vv); vv_eff = np.where(vv < lo, med, vv); base = vv_eff.mean()
        else:
            vv_eff = vv
        for j, (m, s) in enumerate(S.items()):
            idx = np.argsort(-s[i])[:K]; out[r, j] = vv_eff[idx].sum() - K * base
    return out.mean(axis=0), list(S)


def main():
    fp.WH_SVA_ESTIMATOR = "swstr_ct"; H, P = frames(); res = {}
    for D, kind, label in ((H, "h", "HITTERS (wOBA, runs per 400 PA)"), (P, "p", "PITCHERS (RA9, runs per 100 IP)")):
        D["sd"], beta = sd_law(D); print(f"\n{label}: n = {len(D)}; SD of the model's error (runs per player): median {np.median(D['sd']) * pr.runs_scale(kind):.1f}, 10th-90th pct {np.percentile(D['sd'], 10) * pr.runs_scale(kind):.1f}-{np.percentile(D['sd'], 90) * pr.runs_scale(kind):.1f}")
        sc = pr.runs_scale(kind); rows = []
        for W in (1, 2, 3):
            for m, o in asym_loss(D, sc, W).items():
                rows.append({"over-projection costs": f"{W}x", "projection": m, "RMSE-type loss (runs)": o["loss at no shift"], "best conservative shift (runs)": o["best shift (runs)"], "loss with best shift": o["best loss"]})
        T = pd.DataFrame(rows); print("A. ASYMMETRIC SCORING\n" + T.round(2).to_string(index=False)); res[label + " A"] = T.to_dict("records")
    rng = np.random.default_rng(2); rowsB = []
    for D, kind, K, M, label in ((H, "h", 9, 40, "hitters 9 of 40"), (P, "p", 5, 30, "pitchers 5 of 30")):
        for z in (0.5, 1.0):
            for fq, flab in ((None, "linear value (as realised)"), (0.25, "with a downside option (bench the bottom 25% of the pool)")):
                est, names = pick(D, kind, K, M, rng, 4000, z, fq); bs = np.array([pick(D.iloc[rng.integers(0, len(D), len(D))].reset_index(drop=True), kind, K, M, rng, 200, z, fq)[0] for _ in range(150)])
                row = {"scenario": label, "value": flab, "z": z, **{nm: est[j] for j, nm in enumerate(names)}}; d = bs[:, 3] - bs[:, 0]; row["upside - mean"] = est[3] - est[0]; row["90%"] = f"[{np.percentile(d, 5):+.1f}, {np.percentile(d, 95):+.1f}]"; rowsB.append(row)
    TB = pd.DataFrame(rowsB); print("\nB. PICKING BY MEAN vs BY UPSIDE (runs per season vs random picks)\n" + TB.round(1).to_string(index=False)); res["B"] = TB.to_dict("records"); json.dump(res, open("output/projection_value_asym.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
