"""
fl_hitter_bridge_wobamix.py -- for a hitter with MiLB history only, which wOBA is better: DIRECT bridge, COMPONENT-built (K -.177, BB +.516, HR +1.414 of the bridged component deviations, the production BETA), or the 50/50 blend?
Uses the out-of-fold predictions of fl_hitter_bridge_test (M2 pooled slope; leave-one-debut-year-out, nested tuning) and the frozen wOBA weights.  Also a stacked version with a free offset (BABIP is not in the components).
"""
import numpy as np
import fl_eval_report as er
import fl_hitter_bridge_test as bt

BETA = {"k": -0.177, "bbr": 0.516, "hrr": 1.414}
R = bt.main.__call__() if False else None


def run():
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        res = bt.main()
    r = res["woba"]; y, w, yr, ids = r["y"], r["w"], r["yr"], r["dd"]["nm"].to_numpy()
    # all four stats use the same debut set (n = 200)? align by name
    comp = np.zeros(len(y))
    for s, b in BETA.items():
        q = res[s]; idx = {nm: i for i, nm in enumerate(q["dd"]["nm"])}; comp += b * np.array([q["P"]["M2 pooled slope"][idx[nm]] if nm in idx else 0.0 for nm in ids])
    direct = r["P"]["M2 pooled slope"]; base = r["P"]["M1 debut-class mean"]; blend = 0.5 * direct + 0.5 * comp
    stack = np.zeros(len(y))
    for t in sorted(set(yr)):
        te, tr = yr == t, yr != t; A = np.column_stack([np.ones(tr.sum()), direct[tr], comp[tr]]); b = np.linalg.solve(A.T @ (A * w[tr][:, None]), A.T @ (w[tr] * y[tr])); stack[te] = np.column_stack([np.ones(te.sum()), direct[te], comp[te]]) @ b
    rows = []
    for k, v in (("M1 debut-class mean", base), ("direct bridge", direct), ("component-built", comp), ("50/50 blend", blend), ("stacked (free offset, in-fold)", stack)):
        rm, ma = er.wmetrics(y, v, w); row = {"model": k, "RMSE": rm, "MAE": ma}
        if k != "M1 debut-class mean":
            rep = er.compare(y, base, v, w, ids, n=800); row.update({"dRMSE vs M1": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE vs M1": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
        rows.append(row)
    print(f"n = {len(y)}\n" + er.table(rows, 5))
    rep = er.compare(y, direct, blend, w, ids, n=800); print(f"blend vs direct: dRMSE {rep['dRMSE']:+.5f} {rep['RMSE 90%']} P {rep['P(RMSE better)']:.2f}; dMAE {rep['dMAE']:+.5f} P {rep['P(MAE better)']:.2f}")
    rep = er.compare(y, comp, blend, w, ids, n=800); print(f"blend vs component: dRMSE {rep['dRMSE']:+.5f} {rep['RMSE 90%']} P {rep['P(RMSE better)']:.2f}; dMAE {rep['dMAE']:+.5f} P {rep['P(MAE better)']:.2f}")


if __name__ == "__main__":
    run()
