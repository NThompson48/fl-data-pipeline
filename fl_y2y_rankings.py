"""
fl_y2y_rankings.py -- the year-to-year (2025 -> 2026) ranking of EVERY pitcher-level measure, redone with ONE POOLED fit per measure  (user 2026-10-26: "Can we revisit the year to year rankings")
Why: fl_y2y_audit.py showed that the earlier tables fitted each season separately, which adds refit noise to every measure built on a location baseline and made them look unstable (swstr_ct .65 vs .84 pooled).  Here each model measure is fitted ONCE on the
pooled 2025+2026 pitches (cross-fitted by pitcher; levels split by season), the design production uses for the 2027 features.  Raw rates (Whiff% per swing, SwStr% per pitch, K%) need no fit.
MEASURES: XGBoost wh_sva | swing network wh_sva | swstr_plain (per pitch, 9 inputs, no type or count; network) | swstr_ct | strike_loc | strike_ct | P(whiff | stuff) net / xgb | P(swstr | stuff) net / xgb | P(strike | stuff) net / xgb | the four audit measures are read
from the audit's saved pooled levels (network and XGBoost swstr_ct, P(swstr | stuff)).
REPORT: year-to-year r with Fisher CI at >= 150 / 300 / 500 / 800 pitches in each year (nested; same pitchers for every measure within a threshold); disattenuated r (split-half S-B, alternating games) at >= 300; the separate-fit r for reference where logged;
a pitcher bootstrap (2,000 resamples at >= 300 and >= 150) of every measure's rank and P(best), and paired differences vs swstr_ct and vs XGBoost wh_sva.   Output: output/y2y_rankings.json, output/y2y_rank_level_*.csv
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
KEYS = ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]; STUFF = [c for c in fp.FULL_COLS if c not in fp.LOC_COLS]; THR = (150, 300, 500, 800); SEEDS = (1, 2, 3); rng = np.random.default_rng(77)
SEASONS = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}; CALLS_S = fp._STRIKE_CALLS; CALLS_B = fp._BALL_CALLS


def _tjob(tag, seed, tr, te, X, y, g):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg={} if tag == "full" else {"hidden": (64, 64)}); return tag, seed, te, p


def swstr_plain(pool):
    d = fp.add_count(pool[pool["pitch_call"] != "undefined"]); pop = d.dropna(subset=fp.FULL_COLS + ["pitcher_id"]).copy(); y = (pop["is_swing"] & pop["is_whiff"]).astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy()
    Xf = pop[fp.FULL_COLS].astype(float).to_numpy(); Xl = pop[fp.LOC_COLS].astype(float).to_numpy(); folds = list(GroupKFold(5).split(Xf, y, g)); jobs = []
    for tag, X in (("full", Xf), ("loc", Xl)):
        jobs += [delayed(_tjob)(tag, s, tr, te, X, y, g) for s in SEEDS for tr, te in folds]
    O = {}
    for tag, s, te, p in Parallel(n_jobs=8, backend="loky")(jobs):
        O.setdefault((tag, s), np.zeros(len(y)))[te] = p
    return pop[KEYS].assign(sva=np.mean([O[("full", s)] for s in SEEDS], axis=0) - np.mean([O[("loc", s)] for s in SEEDS], axis=0))


def xcv(X, y, g, mask=None):
    p = np.zeros(len(y)); mask = np.ones(len(y), bool) if mask is None else mask
    for tr, te in GroupKFold(5).split(X, y, g):
        trs = tr[mask[tr]]; p[te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(X[trs], y[trs]).predict_proba(X[te])[:, 1]
    return p


def xgb_whiff_stuff(pool):
    pop = pool[pool["pitch_call"] != "undefined"].dropna(subset=STUFF + KEYS).copy(); sw = pop["is_swing"].to_numpy().astype(bool); y = pop["is_whiff"].fillna(False).astype(int).to_numpy(); p = xcv(pop[STUFF].astype(float).to_numpy(), y, pop["pitcher_id"].to_numpy(), sw)
    return pop[KEYS].assign(sva=p - p.mean())


def xgb_strike_stuff(pool):
    pop = pool[pool["pitch_call"] != "undefined"].dropna(subset=STUFF + KEYS).copy(); pop = pop[pop["pitch_call"].isin(CALLS_S | CALLS_B)]; y = pop["pitch_call"].isin(CALLS_S).astype(int).to_numpy(); p = xcv(pop[STUFF].astype(float).to_numpy(), y, pop["pitcher_id"].to_numpy())
    return pop[KEYS].assign(sva=p - p.mean())


MEAS = {"XGBoost wh_sva": lambda p: fp.compute_pooled_wh_sva(p, estimator="xgb"), "swing network wh_sva": lambda p: fp.compute_pooled_wh_sva(p, estimator="torch3"), "swstr_plain (net)": swstr_plain,
        "strike_loc (net)": lambda p: fp.compute_pooled_wh_sva(p, estimator="strike_loc"), "strike_ct (net)": lambda p: fp.compute_pooled_wh_sva(p, estimator="strike_ct"),
        "whiff|stuff (net)": lambda p: fp.compute_pooled_wh_sva(p, estimator="whiff_stuff"), "whiff|stuff (xgb)": xgb_whiff_stuff, "strike|stuff (net)": lambda p: fp.compute_pooled_wh_sva(p, estimator="strike_stuff"), "strike|stuff (xgb)": xgb_strike_stuff}
AUDIT = {"swstr_ct (net)": "net_swstr_ct", "swstr|stuff (net)": "net_swstr-stuff", "swstr_ct (xgb)": "xgb_swstr_ct", "swstr|stuff (xgb)": "xgb_swstr-stuff"}


def fisher(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); return f"[{np.tanh(z - 1.96 * se):.2f}, {np.tanh(z + 1.96 * se):.2f}]"


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); S = {yr: fp.filter_period(raw, *w) for yr, w in SEASONS.items()}; pool = pd.concat([S[2025], S[2026]], ignore_index=True); n_thrown = {yr: S[yr].groupby("pitcher_id").size() for yr in S}
    L, SB = {}, {}
    g = {yr: S[yr].drop_duplicates(["game_id", "pitcher_id"])[["game_id", "pitcher_id", "date"]].sort_values("date") for yr in S}
    for yr in S:
        g[yr]["half"] = g[yr].groupby("pitcher_id").cumcount() % 2
    for name, f in MEAS.items():
        rows = f(pool); rows["year"] = pd.to_datetime(rows["date"]).dt.year; L[name] = {}; SB[name] = {}
        for yr in S:
            r = rows[rows["year"] == yr]; L[name][yr] = r.groupby("pitcher_id")["sva"].mean(); L[name][yr].to_csv(f"output/y2y_rank_level_{name.replace(' ', '_').replace('|', '-').replace('(', '').replace(')', '')}_{yr}.csv")
            SB[name][yr] = r.merge(g[yr][["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"]).groupby(["pitcher_id", "half"])["sva"].mean().unstack().dropna()
        print(f"  {name} done", flush=True)
    for name, tag in AUDIT.items():
        L[name] = {yr: pd.read_csv(f"output/y2y_audit_level_{tag}_{yr}.csv", index_col=0, dtype={0: str}).iloc[:, 0] for yr in S}; SB[name] = None
    sw = {yr: S[yr][S[yr]["is_swing"]].groupby("pitcher_id")["is_whiff"].mean() for yr in S}; sp = {yr: (S[yr]["is_swing"] & S[yr]["is_whiff"]).groupby(S[yr]["pitcher_id"]).mean() for yr in S}
    kk = {}
    for yr in S:
        t = S[yr][S[yr]["is_terminal"]]; k = (t["k_or_bb"] == "strikeout").groupby(t["pitcher_id"]).mean(); kk[yr] = k[t.groupby("pitcher_id").size() >= 40]
    L["raw Whiff% (per swing)"] = sw; L["raw SwStr% (per pitch)"] = sp; L["raw K% (PA >= 40)"] = kk
    for m in L:
        for yr in S:
            L[m][yr].index = L[m][yr].index.astype(str)
    base = pd.concat([n_thrown[2025].rename("n25"), n_thrown[2026].rename("n26")], axis=1).dropna(); base.index = base.index.astype(str); rows = []
    for name in L:
        r = {"measure": name}
        for thr in THR:
            ids = base[(base["n25"] >= thr) & (base["n26"] >= thr)].index; j = pd.concat([L[name][2025].rename("a"), L[name][2026].rename("b")], axis=1).reindex(ids).dropna(); rr = float(np.corrcoef(j["a"], j["b"])[0, 1]); r[f"r>={thr}"] = rr; r[f"CI{thr}"] = fisher(rr, len(j)); r[f"n{thr}"] = len(j)
        rows.append(r)
    T = pd.DataFrame(rows).sort_values("r>=300", ascending=False); pd.set_option("display.width", 300); print("\nYEAR-TO-YEAR r, ONE POOLED FIT PER MEASURE, ranked by r at >= 300 pitches per year\n" + T.round(3).to_string(index=False))
    # bootstrap ranks (same pitchers, all measures at once)
    out = {"table": rows}; names = list(L)
    for thr in (150, 300):
        ids = base[(base["n25"] >= thr) & (base["n26"] >= thr)].index; M = {m: pd.concat([L[m][2025].rename("a"), L[m][2026].rename("b")], axis=1).reindex(ids) for m in names}; ok = pd.concat([M[m].notna().all(axis=1) for m in names], axis=1).all(axis=1); ids = ids[ok.to_numpy()]
        A = np.column_stack([M[m].loc[ids, "a"].to_numpy() for m in names]); B = np.column_stack([M[m].loc[ids, "b"].to_numpy() for m in names]); n = len(ids); R = np.zeros((2000, len(names)))
        def corr(ix):
            a = A[ix] - A[ix].mean(0); b = B[ix] - B[ix].mean(0); return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        for i in range(2000):
            R[i] = corr(rng.integers(0, n, n))
        full = corr(np.arange(n)); ranks = (-R).argsort(axis=1).argsort(axis=1) + 1; sw_i, xg_i = names.index("swstr_ct (net)"), names.index("XGBoost wh_sva")
        tb = pd.DataFrame({"measure": names, "r": full, "mean rank": ranks.mean(0), "5th-95th pct rank": [f"{int(np.percentile(ranks[:, k], 5))}-{int(np.percentile(ranks[:, k], 95))}" for k in range(len(names))], "P(best)": (ranks == 1).mean(0),
                           "P(top 3)": (ranks <= 3).mean(0), "diff vs swstr_ct (net)": full - full[sw_i], "P(> swstr_ct net)": (R > R[:, [sw_i]]).mean(0), "diff vs XGBoost wh_sva": full - full[xg_i], "P(> XGBoost wh_sva)": (R > R[:, [xg_i]]).mean(0)}).sort_values("mean rank")
        print(f"\nBOOTSTRAP RANKS (same {n} pitchers with >= {thr} pitches in each year; 2,000 resamples; rank 1 = highest r)\n" + tb.round(3).to_string(index=False)); out[f"rank_{thr}"] = tb.to_dict("records")
    # disattenuated r at >= 300 for the measures refit here
    dis = []
    for name in MEAS:
        ids = base[(base["n25"] >= 300) & (base["n26"] >= 300)].index; sb = []
        for yr in S:
            h = SB[name][yr].copy(); h.index = h.index.astype(str); h = h.reindex(ids).dropna(); rh = float(np.corrcoef(h[0], h[1])[0, 1]); sb.append(2 * rh / (1 + rh))
        rr = float(T.loc[T["measure"] == name, "r>=300"].iloc[0]); dis.append({"measure": name, "r>=300": rr, "SB 2025": sb[0], "SB 2026": sb[1], "disattenuated": min(rr / np.sqrt(sb[0] * sb[1]), 1.0)})
    print("\nDISATTENUATED (>= 300 pitches)\n" + pd.DataFrame(dis).round(3).to_string(index=False)); out["disattenuated"] = dis
    json.dump(out, open("output/y2y_rankings.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
