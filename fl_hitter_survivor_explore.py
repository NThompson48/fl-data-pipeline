"""
fl_hitter_survivor_explore.py -- how big is the survivor effect, by prior-season PA and by how "survivor" is defined?  (exploration; the fix is tested in fl_hitter_survivor_test.py)
Production regression (n0 fit on the >= 100 PA sample, delta .75 for wOBA) predicts next-year wOBA.  Mean error (actual - predicted) by prior-season PA bucket, for three targets:
  (i)  hitters with >= 100 PA next year (the sample the model is fit on), inverse-variance weights
  (ii) every hitter with >= 20 PA next year, inverse-variance weights
  (iii) every hitter with >= 1 PA next year, PA-weighted (what total runs add up to)
and the retention rate (any PA next year; >= 100 PA next year).
"""
import warnings
import numpy as np
import pandas as pd
import fl_eval_weights as ew
import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore")


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; delta = 0.75
    bt.MIN_NEXT = 100; d100 = bt.transitions(q, "woba"); n0 = bt.fit_n0(d100, delta); print(f"n0 (>=100 PA sample, n={len(d100)}) = {n0:.0f}")
    res = {}
    for lab, m in (("(i) >=100 PA, inv-var", 100), ("(ii) >=20 PA, inv-var", 20), ("(iii) >=1 PA, PA-weighted", 1)):
        bt.MIN_NEXT = m; d = bt.transitions(q, "woba"); d["pred"] = bt.predict(d, n0, delta); d["err"] = d["x_next"] - d["pred"]
        d["w_use"] = d["n_next"] if "PA-weighted" in lab else d["w"]
        d["hist"] = d["n_t"] + delta * d["n_prev"]; res[lab] = d
    bins = [0, 50, 100, 200, 400, 5000]; names = ["<50", "50-100", "100-200", "200-400", "400+"]
    rows = []
    for lab, d in res.items():
        d = d.assign(b=pd.cut(d["n_t"], bins, labels=names))
        for b, g in d.groupby("b"):
            rows.append({"target": lab, "prior PA": b, "n": len(g), "mean err": np.average(g["err"], weights=g["w_use"]), "se": np.sqrt(np.cov(g["err"], aweights=g["w_use"]) / len(g)) if len(g) > 2 else np.nan})
    t = pd.DataFrame(rows).pivot(index="prior PA", columns="target", values=["mean err", "n"]); print("\nMean error (actual - predicted), by prior-season PA\n", t.round(4).to_string())
    # retention
    pa = p.copy(); out = []
    for t_ in range(2022, 2026):
        a = pa[pa.year == t_].set_index("name_clean"); b = pa[pa.year == t_ + 1].set_index("name_clean")
        r = pd.DataFrame({"pa_t": a["pa"]}); r["next"] = b["pa"].reindex(r.index); r["t"] = t_; out.append(r.reset_index())
    r = pd.concat(out); r = r[r.pa_t > 0]; r["b"] = pd.cut(r["pa_t"], bins, labels=names)
    print("\nRetention by prior PA:\n", r.groupby("b").apply(lambda g: pd.Series({"n": len(g), "any PA next yr": g["next"].notna().mean(), ">=20 PA": (g["next"] >= 20).mean(), ">=100 PA": (g["next"] >= 100).mean(), "mean PA | retained": g["next"].mean()})).round(3).to_string())
    # league levels
    lg = {t_: (q[q.year == t_]["woba"] * q[q.year == t_]["pa"]).sum() / q[q.year == t_]["pa"].sum() for t_ in range(2022, 2027)}
    print("\nleague wOBA (all PA):", {k: round(v, 4) for k, v in lg.items()})
    for t_ in range(2023, 2027):
        y = q[(q.year == t_)]; print(t_, "wOBA of hitters with >=100 PA %.4f | >=20 PA %.4f | all %.4f" % tuple((g["woba"] * g["pa"]).sum() / g["pa"].sum() for g in (y[y.pa >= 100], y[y.pa >= 20], y)))


if __name__ == "__main__":
    main()
