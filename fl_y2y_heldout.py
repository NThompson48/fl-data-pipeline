"""
fl_y2y_heldout.py -- year-to-year stickiness with a FROZEN model and HELD-OUT pitchers  (user 2026-10-26: "A better test of stickiness would be using held out pitchers")
DESIGN: pitchers are split into 5 folds (the same folds for every measure).  For each fold, ONE model is fitted on a SINGLE season's pitches of the pitchers NOT in the fold, then FROZEN and applied to the held-out pitchers' pitches in BOTH seasons.
So a pitcher's 2025 level and 2026 level come from the same model that never saw him (nor, for the 'fwd25' design, any 2026 data at all): no refit noise between his two seasons and no contamination from his own pitches or from the later season.
   fwd25 = train on 2025 only, score 2025 and 2026 (a forward-in-time frozen model);   rev26 = train on 2026 only, score both.
MEASURES (same definitions as fl_y2y_rankings.py; each is the mean over the pitcher's scored pitches of P_full - P_loc, or of P_stuff for the stuff-only measures): XGBoost wh_sva | swing network wh_sva | swstr_plain | swstr_ct (network and XGBoost) | strike_loc | strike_ct |
P(whiff | stuff), P(swstr | stuff), P(strike | stuff) (network and XGBoost).   Networks: 3 seeds, MLP 128-64-32 (location net 64-64), pitch type through a 4-d embedding; XGBoost: production params, pitch type one-hot.
REPORT: year-to-year r with Fisher CI at >= 150 / 300 / 500 pitches per year on the SAME pitchers for every measure, bootstrap rank and P(best) (2,000 pitcher resamples), and the comparison with the pooled cross-fit design (output/y2y_rankings.json).   Output: output/y2y_heldout.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from xgboost import XGBClassifier

import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
FULL, LOC = fp.FULL_COLS, fp.LOC_COLS; STUFF = [c for c in FULL if c not in LOC]; CNT = ["balls", "strikes"]; SEEDS = (1, 2, 3); THR = (150, 300, 500); K = 5
SEASONS = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}; CS, CB = fp._STRIKE_CALLS, fp._BALL_CALLS; rng = np.random.default_rng(99)
SPECS = {"XGBoost wh_sva": dict(m="xgb", pop="swing", tgt="whiff", full=FULL, loc=LOC), "swing network wh_sva": dict(m="net", pop="swing", tgt="whiff", full=FULL, loc=LOC),
         "swstr_plain (net)": dict(m="net", pop="all", tgt="swstr", full=FULL, loc=LOC),
         "swstr_ct (net)": dict(m="net", pop="all", tgt="swstr", full=FULL + CNT, loc=LOC + CNT, ptype=True), "swstr_ct (xgb)": dict(m="xgb", pop="all", tgt="swstr", full=FULL + CNT, loc=LOC + CNT, ptype=True),
         "strike_loc (net)": dict(m="net", pop="sb", tgt="strike", full=FULL, loc=LOC), "strike_ct (net)": dict(m="net", pop="sb", tgt="strike", full=FULL + CNT, loc=LOC + CNT),
         "whiff|stuff (net)": dict(m="net", pop="all", tgt="whiff", full=STUFF, loc=None, train="swing"), "whiff|stuff (xgb)": dict(m="xgb", pop="all", tgt="whiff", full=STUFF, loc=None, train="swing"),
         "swstr|stuff (net)": dict(m="net", pop="all", tgt="swstr", full=STUFF, loc=None), "swstr|stuff (xgb)": dict(m="xgb", pop="all", tgt="swstr", full=STUFF, loc=None),
         "strike|stuff (net)": dict(m="net", pop="sb", tgt="strike", full=STUFF, loc=None), "strike|stuff (xgb)": dict(m="xgb", pop="sb", tgt="strike", full=STUFF, loc=None)}


def _net(Xtr, ytr, Xte, g, seed, small, ctr, cte):
    torch.set_num_threads(2); cfg = {"hidden": (64, 64)} if small else {}
    if ctr is not None:
        cfg = {**cfg, "n_cat": int(max(ctr.max(), cte.max())) + 1}
    p, ep, b = ft.fit_predict(Xtr, ytr, Xte, g, seed=seed, cfg=cfg, cat_tr=ctr, cat_te=cte); return p


def _xgb(Xtr, ytr, Xte):
    return XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 2}).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def run(spec, D, fold, design):
    ytrain = 2025 if design == "fwd25" else 2026; need = spec["full"] + (["pitch_type_code"] if False else [])
    ok = D["ok_stuff"] if spec["full"] == STUFF else D["ok_full"]; ok = ok & (D["ok_cnt"] if "balls" in spec["full"] else True)
    mscore = ok & {"all": D["m_all"], "swing": D["m_swing"], "sb": D["m_sb"]}[spec["pop"]]; mtrain = ok & {"all": D["m_all"], "swing": D["m_swing"], "sb": D["m_sb"]}[spec.get("train", spec["pop"])]
    y = {"whiff": D["y_whiff"], "swstr": D["y_swstr"], "strike": D["y_strike"]}[spec["tgt"]]; S = np.where(mscore)[0]; sva = np.full(len(D["pid"]), np.nan); cat = D["ptype"] if spec.get("ptype") else None
    mats = {"full": D["X"][spec["full"]].to_numpy(float)}; mats["loc"] = D["X"][spec["loc"]].to_numpy(float) if spec["loc"] else None; jobs = []
    if cat is not None and spec["m"] == "xgb":                                  # XGBoost gets pitch type as a one-hot block (the network uses an embedding)
        oh = np.eye(int(cat.max()) + 1)[cat]; mats["full"] = np.column_stack([mats["full"], oh]); mats["loc"] = np.column_stack([mats["loc"], oh])
    for k in range(K):
        tr = np.where(mtrain & (D["year"] == ytrain) & (fold != k))[0]; te = np.where(mscore & (fold == k))[0]
        for tag in (("full", "loc") if spec["loc"] else ("full",)):
            X = mats[tag]
            if spec["m"] == "net":
                jobs += [(k, tag, s, delayed(_net)(X[tr], y[tr], X[te], D["pid"][tr], s, tag == "loc", cat[tr] if cat is not None else None, cat[te] if cat is not None else None)) for s in SEEDS]
            else:
                jobs.append((k, tag, 0, delayed(_xgb)(X[tr], y[tr], X[te])))
    res = Parallel(n_jobs=8, backend="loky")([j[3] for j in jobs]); P = {}
    for (k, tag, s, _), p in zip(jobs, res):
        P.setdefault((k, tag), []).append(p)
    for k in range(K):
        te = np.where(mscore & (fold == k))[0]; pf = np.mean(P[(k, "full")], axis=0); sva[te] = pf - (np.mean(P[(k, "loc")], axis=0) if spec["loc"] else 0.0)
    sva[mscore] -= np.nanmean(sva[mscore]); return pd.DataFrame({"pitcher_id": D["pid"][mscore], "year": D["year"][mscore], "sva": sva[mscore]})


def fisher(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); return f"[{np.tanh(z - 1.96 * se):.2f}, {np.tanh(z + 1.96 * se):.2f}]"


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); S = {yr: fp.filter_period(raw, *w) for yr, w in SEASONS.items()}; n_thrown = {yr: S[yr].groupby("pitcher_id").size() for yr in S}
    pool = pd.concat([S[2025], S[2026]], ignore_index=True); pool["year"] = pd.to_datetime(pool["date"]).dt.year; d = fp.add_count(pool[pool["pitch_call"] != "undefined"]).reset_index(drop=True)
    pt = pd.Categorical(d["pitch_type"].fillna("unk").replace("", "unk")); ids = d["pitcher_id"].astype(str).to_numpy(); up = np.unique(ids); fid = dict(zip(up, rng.permutation(len(up)) % K)); fold = np.array([fid[i] for i in ids])
    isw = d["is_swing"].fillna(False).astype(bool).to_numpy(); wh = d["is_whiff"].fillna(False).astype(bool).to_numpy(); strikeball = d["pitch_call"].isin(CS | CB).to_numpy()
    D = {"pid": ids, "year": d["year"].to_numpy(), "X": d[list(dict.fromkeys(FULL + CNT))], "ptype": pt.codes.astype(int), "m_all": np.ones(len(d), bool), "m_swing": isw, "m_sb": strikeball,
         "ok_full": d[FULL].notna().all(axis=1).to_numpy(), "ok_stuff": d[STUFF].notna().all(axis=1).to_numpy(), "ok_cnt": d[CNT].notna().all(axis=1).to_numpy(), "y_whiff": (isw & wh).astype(int), "y_swstr": (isw & wh).astype(int), "y_strike": d["pitch_call"].isin(CS).astype(int).to_numpy()}
    out = {}; base = pd.concat([n_thrown[2025].rename("n25"), n_thrown[2026].rename("n26")], axis=1).dropna(); base.index = base.index.astype(str)
    for design in ("fwd25", "rev26"):
        L = {}
        for name, spec in SPECS.items():
            r = run(spec, D, fold, design); L[name] = {yr: r[r["year"] == yr].groupby("pitcher_id")["sva"].mean() for yr in (2025, 2026)}
            pd.DataFrame({yr: L[name][yr] for yr in (2025, 2026)}).to_csv(f"output/y2y_heldout_{design}_{name.replace(' ', '_').replace('|', '-').replace('(', '').replace(')', '')}.csv"); print(f"  {design} {name} done", flush=True)
        sw = {yr: S[yr][S[yr]["is_swing"]].groupby("pitcher_id")["is_whiff"].mean() for yr in S}; sp = {yr: (S[yr]["is_swing"] & S[yr]["is_whiff"]).groupby(S[yr]["pitcher_id"]).mean() for yr in S}
        L["raw Whiff% (per swing)"] = sw; L["raw SwStr% (per pitch)"] = sp
        for m in L:
            for yr in (2025, 2026):
                L[m][yr].index = L[m][yr].index.astype(str)
        rows = []
        for name in L:
            r = {"measure": name}
            for thr in THR:
                idx = base[(base["n25"] >= thr) & (base["n26"] >= thr)].index; j = pd.concat([L[name][2025].rename("a"), L[name][2026].rename("b")], axis=1).reindex(idx).dropna(); rr = float(np.corrcoef(j["a"], j["b"])[0, 1])
                r[f"r>={thr}"] = rr; r[f"CI{thr}"] = fisher(rr, len(j)); r[f"n{thr}"] = len(j)
            rows.append(r)
        T = pd.DataFrame(rows).sort_values("r>=300", ascending=False); pd.set_option("display.width", 300); print(f"\n== {design}: year-to-year r, frozen model on held-out pitchers\n" + T.round(3).to_string(index=False)); out[design] = {"table": rows}
        names = list(L); idx = base[(base["n25"] >= 300) & (base["n26"] >= 300)].index; M = {m: pd.concat([L[m][2025].rename("a"), L[m][2026].rename("b")], axis=1).reindex(idx) for m in names}
        idx = idx[pd.concat([M[m].notna().all(axis=1) for m in names], axis=1).all(axis=1).to_numpy()]; A = np.column_stack([M[m].loc[idx, "a"].to_numpy() for m in names]); B = np.column_stack([M[m].loc[idx, "b"].to_numpy() for m in names]); n = len(idx)
        def corr(ix):
            a = A[ix] - A[ix].mean(0); b = B[ix] - B[ix].mean(0); return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        R = np.array([corr(rng.integers(0, n, n)) for _ in range(2000)]); ranks = (-R).argsort(axis=1).argsort(axis=1) + 1; full = corr(np.arange(n))
        tb = pd.DataFrame({"measure": names, "r": full, "mean rank": ranks.mean(0), "5th-95th rank": [f"{int(np.percentile(ranks[:, k], 5))}-{int(np.percentile(ranks[:, k], 95))}" for k in range(len(names))], "P(best)": (ranks == 1).mean(0), "P(top 3)": (ranks <= 3).mean(0)}).sort_values("mean rank")
        print(f"\n   {design} bootstrap ranks (same {n} pitchers with >= 300 pitches in each year)\n" + tb.round(3).to_string(index=False)); out[design]["ranks300"] = tb.to_dict("records")
    try:
        pooled = {r["measure"]: r["r>=300"] for r in json.load(open("output/y2y_rankings.json"))["table"]}; cmp = pd.DataFrame({"measure": list(pooled), "pooled cross-fit r>=300": list(pooled.values())})
        for design in ("fwd25", "rev26"):
            cmp[f"{design} r>=300"] = cmp["measure"].map({r["measure"]: r["r>=300"] for r in out[design]["table"]})
        print("\nCOMPARISON OF DESIGNS (r at >= 300 pitches per year)\n" + cmp.round(3).to_string(index=False)); out["comparison"] = cmp.to_dict("records")
    except Exception as ex:
        print("comparison skipped:", ex)
    json.dump(out, open("output/y2y_heldout.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
