"""
fl_lowhistory_swstr_test.py -- how does the production K% model (additive swstr_ct) do for LOW-HISTORY, ZERO-PA and ROOKIE pitchers?  (user 2026-10-25: "How are we doing on low/zero history/rookie pitchers?")
MODELS (each fit on the fit transition, scored on the test transition; the four-pairing set of fl_wh_sva_prior_xgb_robust_test.py): published = additive K% with the XGBoost stuff term | swstr_ct = additive K% with the per-pitch swinging-strike stuff level (production now)
| naive shrunk own rate (naive_K_rate, the model's own shrunk rate, no stuff / bridge) | league mean.   GROUPS by the pitcher's plate appearances in the feature window: ZERO PA (pitches but no plate appearance), 1-49, 50-99 (so 'thin' < 100), 100-249, 250+;
ROOKIES = no 2025 appearance, defined on the test frame whose features are 2025 + 1H2026 (pairings T1->T2 and T0->T2).   Level-free per group (each method gets its own constant removed), frozen inverse-variance weights from the published model within the group, weighted RMSE AND MAE, pitcher bootstrap (1,000).
Rookies with NO FL history at all (bridge-only) cannot appear in this harness (they have no features); they are scored by the leave-one-out bridge test of fl_lowhistory_baseline_test.part_b (K%, BB% of the first FL season) and described by count in the 2027 file.   Output: output/lowhistory_swstr_test.json
"""
import contextlib
import io
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import fl_wh_sva_prior_xgb_robust_test as rb
import mlb_marcel as mm

warnings.filterwarnings("ignore")
ADD = pp.SPEC["K"]; _c = {}


def frame(raw, est, name):
    if (est, name) not in _c:
        fp.WH_SVA_ESTIMATOR = est; feat, lab = rb.TR[name]; df, n0_k, _ = bt.assemble(raw, feat, lab, with_changepoints=False); _c[(est, name)] = (df, n0_k)
    return _c[(est, name)]


def score(rows, pairing, group, y, n, preds, ids):
    ok = np.isfinite(y) & np.all([np.isfinite(v) for v in preds.values()], axis=0); y, n, ids = y[ok], n[ok], ids[ok]; preds = {k: v[ok] for k, v in preds.items()}
    if len(y) < 15:
        return
    w = ew.weights_from_base(y, preds["published"], n)[0]; cen = lambda q: q + np.average(y - q, weights=w); r = {"pairing": pairing, "group": group, "n": len(y)}
    for k, q in preds.items():
        rm, ma = er.wmetrics(y, cen(q), w); r[f"RMSE {k}"] = rm; r[f"MAE {k}"] = ma
    for k in ("naive shrunk rate", "league mean"):
        c = er.compare(y, cen(preds[k]), cen(preds["swstr_ct"]), w, ids, n=1000); r[f"swstr_ct vs {k}: dRMSE"] = c["dRMSE"]; r[f"P RMSE {k}"] = c["P(RMSE better)"]; r[f"dMAE {k}"] = c["dMAE"]; r[f"P MAE {k}"] = c["P(MAE better)"]
    c = er.compare(y, cen(preds["published"]), cen(preds["swstr_ct"]), w, ids, n=1000); r["swstr_ct vs published: dRMSE"] = c["dRMSE"]; r["RMSE 90%"] = c["RMSE 90%"]; r["P RMSE published"] = c["P(RMSE better)"]; r["dMAE published"] = c["dMAE"]; r["P MAE published"] = c["P(MAE better)"]
    rows.append(r)


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); rows = []
    had25 = set(fp.filter_period(raw, bt.START_2025, bt.END_2025)["pitcher_id"].astype(str))
    for a, b in rb.PAIRS:
        M = {}
        for est, lab in (("xgb", "published"), ("swstr_ct", "swstr_ct")):
            df, n0_k = frame(raw, est, a); d = df.dropna(subset=["K_rate_actual"] + ADD); M[lab] = bt.fit_weighted(d, "K_rate_actual", ADD, mm.reliability_weight(d["PA_test"], n0_k).values)
        t_x, _ = frame(raw, "xgb", b); t_s, _ = frame(raw, "swstr_ct", b); t = t_x.dropna(subset=["K_rate_actual"] + ADD).merge(t_s[["pitcher_id", "wh_sva_level"]].rename(columns={"wh_sva_level": "sw"}), on="pitcher_id", how="inner").dropna(subset=["sw"]).copy()
        ts = t.drop(columns="wh_sva_level").rename(columns={"sw": "wh_sva_level"}); p = {"published": M["published"].predict(t).to_numpy(), "swstr_ct": M["swstr_ct"].predict(ts).to_numpy(), "naive shrunk rate": t["naive_K_rate"].to_numpy(), "league mean": np.full(len(t), np.average(t["K_rate_actual"], weights=t["PA_test"]))}
        y = t["K_rate_actual"].to_numpy(float); n = t["PA_test"].to_numpy(float); ids = t["pitcher_id"].astype(str).to_numpy(); pa = t["PA"].fillna(0).to_numpy(float); pair = f"{a}->{b}"
        for g, m in (("ALL", np.ones(len(t), bool)), ("zero PA (pitches, no PA)", pa == 0), ("1-49 PA", (pa >= 1) & (pa < 50)), ("50-99 PA", (pa >= 50) & (pa < 100)), ("thin: < 100 PA", pa < 100), ("100-249 PA", (pa >= 100) & (pa < 250)), ("250+ PA", pa >= 250)):
            score(rows, pair, g, y[m], n[m], {k: v[m] for k, v in p.items()}, ids[m])
        if b == "T2":
            m = ~pd.Series(ids).isin(had25).to_numpy(); score(rows, pair, "rookies (no 2025 appearance)", y[m], n[m], {k: v[m] for k, v in p.items()}, ids[m])
            m2 = m & (pa < 100); score(rows, pair, "rookies with < 100 PA", y[m2], n[m2], {k: v[m2] for k, v in p.items()}, ids[m2])
        print(f"{pair} done", flush=True)
    T = pd.DataFrame(rows); pd.set_option("display.width", 300); keep = ["pairing", "group", "n", "RMSE published", "RMSE swstr_ct", "RMSE naive shrunk rate", "RMSE league mean", "MAE published", "MAE swstr_ct", "MAE naive shrunk rate", "MAE league mean"]
    print("\nK%: LEVELS (weighted RMSE and MAE, level-free within group)\n" + T[keep].round(5).to_string(index=False))
    k2 = ["pairing", "group", "n", "swstr_ct vs published: dRMSE", "P RMSE published", "dMAE published", "P MAE published", "swstr_ct vs naive shrunk rate: dRMSE", "P RMSE naive shrunk rate", "swstr_ct vs league mean: dRMSE", "P RMSE league mean"]
    print("\nswstr_ct (negative d = swstr_ct better; P = bootstrap probability swstr_ct better)\n" + T[k2].round(5).to_string(index=False))
    # bridge-only rookies (leave-one-out bridge test) and the 2027 file's composition
    import fl_lowhistory_baseline_test as L
    with contextlib.redirect_stdout(io.StringIO()):
        L.part_b()
    B = pd.DataFrame(L.ROWS); print("\nBRIDGE-ONLY PITCHER ROOKIES (leave-one-out unified bridge; first FL season)\n" + B.round(5).to_string(index=False))
    pr = pd.read_parquet("output/fl_2027_projections.parquet"); pa = pr["PA"].fillna(0)
    comp = {"pitchers in the 2027 file": int(len(pr)), "PA = 0": int((pa == 0).sum()), "1-49 PA": int(((pa >= 1) & (pa < 50)).sum()), "50-99 PA": int(((pa >= 50) & (pa < 100)).sum()), "100+ PA": int((pa >= 100).sum()), "proj_basis counts": pr["proj_basis"].value_counts().to_dict(), "bridge matched": int(pr["has_college_match"].sum())}
    print("\n2027 FILE COMPOSITION:", comp)
    json.dump({"table": rows, "bridge_rookies": L.ROWS, "composition": comp}, open("output/lowhistory_swstr_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
