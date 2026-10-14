"""
fl_error_analysis_nested.py -- NESTED validation of the error-analysis findings (user 2026-10-13: "Are you confident these variables hold up to cross validation / bootstrapping?").

Why: in fl_error_analysis_select.py the shortlist and the forward selection used BOTH in-season samples (ROS25, ROS26), and the reported "held-out" tests were the same two
directions -> optimistic.  Also the pitcher PRE fold (2H2026 targets) shares its outcome period with ROS26, so PRE was not an independent check for pitchers (hitters: half overlap).

Design (everything that chooses or fits uses the TRAINING sample only):
  pipeline(A): (1) screen every candidate trait inside A (WLS residual ~ z(trait) + centred prediction + log history, HC1 robust SE), BH q <= .10 within A, plus the trait
               stickiness rules (split-half >= .40, disattenuated y/y >= .50 -- trait-only properties, no outcomes); (2) forward selection inside A by 10-fold-by-player CV
               (x 3 repeats): add the shortlisted trait that most lowers the CV weighted SSE, stop when nothing lowers it, max 4; (3) fit b on all of A.
  Test on samples whose OUTCOMES were never touched: A = ROS25 -> ROS26 and PRE (2026 outcomes); A = ROS26 -> ROS25.
  Point result: the pipeline on A, scored on the test sample, player bootstrap over test players (1,000).
  PIPELINE BOOTSTRAP (selection uncertainty included): resample A's players 200 times, rerun the whole pipeline, score each on the fixed test sample -> distribution of the
  held-out change (5th-95th percentile, P(better)) and how often each trait is selected.
  Forms: constant term (primary, the better form in the earlier tests) and faded (n0 / (history + n0)).
Usage:  python fl_error_analysis_nested.py [h|p]
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import fl_error_analysis_common as ea
import fl_eval_report as er
from fl_error_analysis_select import load

warnings.filterwarnings("ignore")
B_PIPE = 200


class D:
    """numpy view of one (sample, target): residual, weights, controls, history, rookie flag, and the shrunk traits (population-mean filled) as a matrix."""
    def __init__(self, s, tg, st, cands, n0t, rows=None):
        F = s["targets"][tg]; self.ids = F.index.to_numpy(); self.e = F["e"].to_numpy(); self.w = F["w"].to_numpy(); self.pc = F["pc"].to_numpy(); self.lh = F["lh"].to_numpy()
        self.hist = F["hist"].to_numpy(); self.rook = F["rookie"].to_numpy().astype(bool); self.n0 = n0t[tg]; self.names = []; cols = []
        for k in cands:
            if k not in s["traits"]:
                continue
            v, n = s["traits"][k]; x = ea.shrink(v, n, st.loc[k, "n0"]).reindex(F.index).to_numpy().astype(float).copy()
            if np.isfinite(x).mean() < 0.3:
                continue
            ok = np.isfinite(x); x[~ok] = np.average(x[ok], weights=self.w[ok]); self.names.append(k); cols.append(x)
        self.T = np.column_stack(cols)
        if rows is not None:
            self.take(rows)

    def take(self, rows):
        for a in ("ids", "e", "w", "pc", "lh", "hist", "rook"):
            setattr(self, a, getattr(self, a)[rows])
        self.T = self.T[rows]; return self

    def terms(self, L, fade):
        if not L:
            return np.zeros((len(self.e), 0))
        j = [self.names.index(k) for k in L]; x = self.T[:, j]; mu = np.average(x, axis=0, weights=self.w)
        f = (self.n0 / (self.hist + self.n0))[:, None] if fade else 1.0; return (x - mu) * f


def wls(e, w, M):
    return np.linalg.lstsq(M * np.sqrt(w)[:, None], e * np.sqrt(w), rcond=None)[0]


def screen_fast(d):
    w, y = d.w, d.e; ts = []
    for j, k in enumerate(d.names):
        x = d.T[:, j]; sd = np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w))
        if sd == 0:
            ts.append(0.0); continue
        z = (x - np.average(x, weights=w)) / sd; M = np.column_stack([np.ones(len(y)), z, d.pc, d.lh]); sw = np.sqrt(w); Mw = M * sw[:, None]; b = np.linalg.lstsq(Mw, y * sw, rcond=None)[0]
        r = (y - M @ b) * sw; XtX = np.linalg.pinv(Mw.T @ Mw); meat = (Mw * r[:, None]).T @ (Mw * r[:, None]); V = XtX @ meat @ XtX * len(y) / (len(y) - 4); ts.append(b[1] / np.sqrt(V[1, 1]))
    t = np.array(ts); q = ea.bh(2 * norm.sf(np.abs(t))); return t, q


def cv_sse(d, M, groups, folds=10, reps=3, seed=11):
    rng = np.random.default_rng(seed); ug = np.unique(groups); tot = 0.0
    for _ in range(reps):
        fmap = dict(zip(ug, rng.integers(0, folds, len(ug)))); fid = np.array([fmap[g] for g in groups]); pred = np.zeros(len(d.e))
        if M.shape[1]:
            for k in range(folds):
                tr, te = fid != k, fid == k
                if te.any() and tr.sum() >= 10:
                    pred[te] = M[te] @ wls(d.e[tr], d.w[tr], M[tr])
        r = d.e - pred; tot += np.sum(d.w * (r - np.average(r, weights=d.w)) ** 2)
    return tot / reps


def pipeline(d, st, fade, groups):
    t, q = screen_fast(d)
    ok = [k for j, k in enumerate(d.names) if q[j] <= .10 and st.loc[k, "split-half r"] >= .40 and pd.notna(st.loc[k, "y/y disattenuated"]) and st.loc[k, "y/y disattenuated"] >= .50]
    cur, cur_s = [], cv_sse(d, np.zeros((len(d.e), 0)), groups)
    while ok and len(cur) < 4:
        best = None
        for c in ok:
            if c not in cur:
                sc = cv_sse(d, d.terms(cur + [c], fade), groups)
                if sc < cur_s and (best is None or sc < best[1]):
                    best = (c, sc)
        if best is None:
            break
        cur.append(best[0]); cur_s = best[1]
    b = wls(d.e, d.w, d.terms(cur, fade)) if cur else np.zeros(0)
    return cur, b, [k for j, k in enumerate(d.names) if q[j] <= .10]


def held_out(dT, L, b, fade):
    """dT: test D.  Returns {group: (dRMSE, dMAE, c0, c1, mask)}; the training b is applied to the test sample's own trait values (centred on the test sample)."""
    e0 = dT.e; e1 = e0 - (dT.terms(L, fade) @ b if L else 0.0); out = {}
    for g, m in (("all", np.ones(len(e0), bool)), ("low-history", dT.hist < 150), ("rookies", dT.rook)):
        if m.sum() < 15:
            continue
        w = dT.w[m]; c0 = e0[m] - np.average(e0[m], weights=w); c1 = e1[m] - np.average(e1[m], weights=w)
        out[g] = (np.sqrt(np.average(c1 ** 2, weights=w)) - np.sqrt(np.average(c0 ** 2, weights=w)), np.average(np.abs(c1), weights=w) - np.average(np.abs(c0), weights=w), c0, c1, m)
    return out


def main(kind):
    st, S, n0t, _ = load(kind)
    targets = ["woba", "bb", "k"] if kind == "h" else ["k", "bb"]
    rows, boot_rows = [], []
    for tg in targets:
        cands = list(st.index)
        for fade in (False, True):
            form = "faded" if fade else "constant"
            for A, tests in (("ROS25", ["ROS26", "PRE"]), ("ROS26", ["ROS25"])):
                dA = D(S[A], tg, st, cands, n0t); dTs = {T: D(S[T], tg, st, dA.names, n0t) for T in tests}
                L, b, passed = pipeline(dA, st, fade, dA.ids)
                print(f"\n[{tg}, {form}] trained on {A}: screen passes (q <= .10 within {A}) {passed}; selected {L if L else 'nothing'}" + (f"; b = {np.round(b, 5).tolist()}" if L else ""), flush=True)
                for T in tests:
                    dT = dTs[T]
                    if not L:
                        rows.append({"target": tg, "form": form, "train": A, "test": T, "selected": "-", "group": "all", "dRMSE": 0.0, "dMAE": 0.0}); continue
                    for g, (dr, dm, c0, c1, m) in held_out(dT, L, b, fade).items():
                        r = er.compare(np.zeros(m.sum()), -c0, -c1, dT.w[m], dT.ids[m], n=1000)
                        rows.append({"target": tg, "form": form, "train": A, "test": T, "selected": " + ".join(L), "group": g, "n": int(m.sum()), "RMSE old": r["RMSE old"], "RMSE new": r["RMSE new"], "dRMSE": r["dRMSE"],
                                     "RMSE 90%": r["RMSE 90%"], "P(RMSE better)": r["P(RMSE better)"], "MAE old": r["MAE old"], "MAE new": r["MAE new"], "dMAE": r["dMAE"], "P(MAE better)": r["P(MAE better)"]})
                # ---- pipeline bootstrap: selection + fit redone on resamples of A
                rng = np.random.default_rng(29); n = len(dA.e); cnt = {}; res = {T: {} for T in tests}; empty = 0
                for _ in range(B_PIPE):
                    pick = rng.integers(0, n, n); db = D.__new__(D); db.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in dA.__dict__.items()}; db.take(pick)
                    Lb, bb, _ = pipeline(db, st, fade, np.arange(n))
                    empty += not Lb
                    for k in Lb:
                        cnt[k] = cnt.get(k, 0) + 1
                    for T in tests:
                        ho = held_out(dTs[T], Lb, bb, fade)
                        for g, v in ho.items():
                            res[T].setdefault(g, []).append((v[0], v[1]))
                for T in tests:
                    for g, v in res[T].items():
                        v = np.array(v); boot_rows.append({"target": tg, "form": form, "train": A, "test": T, "group": g, "mean dRMSE": v[:, 0].mean(), "dRMSE 5%": np.percentile(v[:, 0], 5),
                                                           "dRMSE 95%": np.percentile(v[:, 0], 95), "P(RMSE better)": float((v[:, 0] < 0).mean()), "P(RMSE worse)": float((v[:, 0] > 0).mean()),
                                                           "mean dMAE": v[:, 1].mean(), "P(MAE better)": float((v[:, 1] < 0).mean()), "nothing selected": empty / B_PIPE})
                print(f"   selection frequency over {B_PIPE} pipeline bootstraps: " + ", ".join(f"{k} {v / B_PIPE:.2f}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:8]) + f"; nothing selected {empty / B_PIPE:.2f}", flush=True)
    pd.set_option("display.width", 260)
    R = pd.DataFrame(rows); print("\nNESTED HOLD-OUT (pipeline fit on the training sample only; player bootstrap over the test sample)\n" + R.round(5).to_string(index=False))
    Bt = pd.DataFrame(boot_rows); print(f"\nPIPELINE BOOTSTRAP (selection + fit redone on {B_PIPE} resamples of the training sample; scored on the fixed test sample; P(RMSE better) + P(worse) < 1 = runs that selected nothing)\n" + Bt.round(5).to_string(index=False))
    R.to_csv(f"output/error_analysis_nested_{kind}.csv", index=False); Bt.to_csv(f"output/error_analysis_nested_boot_{kind}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h")
