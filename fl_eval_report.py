"""
fl_eval_report.py -- the standard before/after report for any model change (user rule, 2026-10-08): weighted RMSE AND weighted MAE, old vs new, same frozen weights, with a cluster-bootstrap interval and P(better) for both.
Usage:  rep = compare(y, old, new, w, ids);  rows.append({"stat": ..., **rep});  print(table(rows))
"""
import numpy as np
import pandas as pd

_rng = np.random.default_rng(2026)


def wmetrics(y, p, w):
    e = np.asarray(y, float) - np.asarray(p, float); w = np.asarray(w, float)
    return float(np.sqrt(np.average(e ** 2, weights=w))), float(np.average(np.abs(e), weights=w))


def compare(y, old, new, w, ids, n=600, level=90):
    y, old, new, w, ids = (np.asarray(a) for a in (y, old, new, w, ids))
    r0, m0 = wmetrics(y, old, w); r1, m1 = wmetrics(y, new, w)
    by = pd.Series(np.arange(len(y))).groupby(ids).apply(np.array); ug = by.index.to_numpy(); dr, dm = [], []
    for _ in range(n):
        ii = np.concatenate([by[u] for u in _rng.choice(ug, len(ug))]); a = wmetrics(y[ii], old[ii], w[ii]); b = wmetrics(y[ii], new[ii], w[ii]); dr.append(b[0] - a[0]); dm.append(b[1] - a[1])
    dr, dm = np.array(dr), np.array(dm); lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    return {"n": len(y), "RMSE old": r0, "RMSE new": r1, "dRMSE": r1 - r0, f"RMSE {level}%": f"[{np.percentile(dr, lo):+.5f}, {np.percentile(dr, hi):+.5f}]", "P(RMSE better)": float((dr < 0).mean()),
            "MAE old": m0, "MAE new": m1, "dMAE": m1 - m0, f"MAE {level}%": f"[{np.percentile(dm, lo):+.5f}, {np.percentile(dm, hi):+.5f}]", "P(MAE better)": float((dm < 0).mean())}


def table(rows, digits=5):
    return pd.DataFrame(rows).round(digits).to_string(index=False)
