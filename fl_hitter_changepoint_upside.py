"""
fl_hitter_changepoint_upside.py -- (1) CHANGEPOINTS in hitters' underlying skill series and (2) a model fitted on BIG UPSIDE MISSES  (user 2026-10-28: "Changepoints would be interesting to look at. Fitting a model on big upside misses would also be good to do.")
The project's earlier changepoint work (DECISION_LOG, half-season backtest) was PITCHER-only (per-start Stuff / location series) and did not survive a holdout; this is the hitter version, on the project's hitter error-analysis samples (fl_error_analysis_hitters / fl_hitter_battrack_test: PRE = 2025 -> 2026 preseason
path on 143 hitters, ROS25 / ROS26 = production rest-of-season model as of mid-season -> rest of season, 240 / 244 hitters; targets = level-free residuals, frozen standard weights).
CHANGEPOINT FEATURES (per hitter, from his chronological pitch / batted-ball series in the feature window only; five series: exit speed and launch angle on balls in play, whiff per swing, swing per pitch, measured bat speed on swings):
  cp_t        signed t-statistic of the best single split (the split with the largest |t|; both sides >= MIN observations)        cp_shift    mean after - mean before that split (population-SD units)
  cp_recent   mean after the split - the window mean (the "recent level" relative to the window)                                   rec25       mean of the last 25% of observations - the window mean (FIXED split, no search: the control for the search)
PART 1 SCREEN: residual ~ z(feature) + centred prediction + log history (player bootstrap), pooled across ROS25 / ROS26, BH q over ALL traits (the 38 earlier + these 20), targets wOBA, HR/PA, K%, BB%.
PART 2 UPSIDE MODEL: breakout = a residual in the top 20% of its sample (downside = bottom 20%).  Ridge logistic regression on all traits (shrunk by their reliability) + log history + the centred prediction, fit on one sample and scored on the other (ROS25 -> ROS26, ROS26 -> ROS25, ROS25 -> PRE) and by 10-fold CV x 5 inside PRE;
  with and without the changepoint features, against a baseline that uses only log history and the prediction (the thin-sample effect).  AUC for the upside and the downside tail, the hit rate and the mean residual of the 15 hitters with the highest predicted upside (base rate .20).
Usage: python fl_hitter_changepoint_upside.py   (log: output/changepoint_upside_log.txt)   Output: output/changepoint_upside.json
"""
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import fl_error_analysis_common as ea
import fl_error_analysis_hitters as eh
import fl_hitter_battrack_test as bk

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
SERIES = {"ev": ("bip", "exit_speed", 20), "la": ("bip", "angle", 20), "whiff": ("swing", "whiff", 40), "swing": ("pitch", "swing", 80), "bat": ("swing_bat", "batspeed", 40)}
TARGETS = ("woba", "hr", "k", "bb")


def series_features(x):
    """x: chronological values; returns (best signed t, shift in SD units, post-split mean - window mean) for the best single split"""
    n = len(x); cs = np.concatenate([[0.0], np.cumsum(x)]); sd = x.std()
    return n, cs, sd


def cp_one(x, minn):
    n = len(x)
    if n < 2 * minn + 5:
        return None
    sd = x.std()
    if sd == 0:
        return None
    cs = np.concatenate([[0.0], np.cumsum(x)]); ks = np.arange(minn, n - minn + 1); m1 = cs[ks] / ks; m2 = (cs[n] - cs[ks]) / (n - ks); tt = (m2 - m1) / (sd * np.sqrt(1 / ks + 1 / (n - ks))); j = int(np.argmax(np.abs(tt))); k = ks[j]; mean_all = cs[n] / n
    r25 = x[int(0.75 * n):].mean() - mean_all
    return {"cp_t": float(tt[j]), "cp_shift": float((m2[j] - m1[j]) / sd), "cp_recent": float((m2[j] - mean_all) / sd), "rec25": float(r25 / sd), "n": n}


def cp_traits(t):
    """changepoint features from a window of tracking rows (t['key'] = hitter); returns {trait: (Series by key, n Series)}"""
    t = t.sort_values(["key", "date", "game_id", "pa_id", "pa_pitch_count"], kind="mergesort"); out = {f"{s}_{f}": {} for s in SERIES for f in ("cp_t", "cp_shift", "cp_recent", "rec25")}; ns = {}
    for s, (kind, col, minn) in SERIES.items():
        if kind == "bip":
            d = t[t["is_bip"] & t[col].notna()]
        elif kind == "swing":
            d = t[t["swing"] == 1]
        elif kind == "swing_bat":
            d = t[(t["swing"] == 1) & t[col].notna()]
        else:
            d = t
        for key, g in d.groupby("key"):
            r = cp_one(g[col].to_numpy(float), minn)
            if r is None:
                continue
            for f in ("cp_t", "cp_shift", "cp_recent", "rec25"):
                out[f"{s}_{f}"][key] = r[f]
            ns[f"{s}_{f}" if False else s] = ns.get(s, {}); ns[s][key] = r["n"]
    res = {}
    for name, v in out.items():
        s = name.split("_")[0]; v = pd.Series(v, dtype=float); res[name] = (v, pd.Series(ns[s], dtype=float).reindex(v.index))
    return res


def build_all(t):
    return {**ea.hitter_traits(t), **bk.battrack_traits(t), **cp_traits(t)}


def design(F, traits, n0s, names):
    """standardised design matrix: shrunk traits (zero-mean, unit SD within the sample) + log history + centred prediction; missing -> 0"""
    cols = []; used = []
    for k in names:
        v, n = traits[k]; xs = ea.shrink(v, n, n0s.get(k, 0.0)).reindex(F.index)
        if xs.notna().mean() < 0.3 or xs.std() == 0:
            continue
        mu = np.average(xs.dropna(), weights=F.loc[xs.notna(), "w"]); z = (xs.fillna(mu) - mu) / xs.std(); cols.append(z.to_numpy()); used.append(k)
    X = np.column_stack(cols) if cols else np.zeros((len(F), 0)); base = np.column_stack([(F["lh"] - F["lh"].mean()) / F["lh"].std(), (F["pc"] - F["pc"].mean()) / F["pc"].std()]); return np.column_stack([X, base]), used


def tails(F, q=0.2):
    e = F["e"].to_numpy(); return (e >= np.quantile(e, 1 - q)).astype(int), (e <= np.quantile(e, q)).astype(int)


def fit_score(Xa, ya, Xb, C):
    m = LogisticRegression(C=C, max_iter=2000).fit(Xa, ya); return m.predict_proba(Xb)[:, 1], m


def metrics(Ft, p_up, p_dn, yup, ydn, topn=15):
    top = np.argsort(-p_up)[:topn]; e = Ft["e"].to_numpy(); w = Ft["w"].to_numpy(); sc = 400 / 1.0175
    return {"AUC up": roc_auc_score(yup, p_up), "AUC down": roc_auc_score(ydn, p_dn), "top-15 hit rate (base .20)": float(yup[top].mean()), "top-15 mean surprise (runs per 400 PA)": float(e[top].mean() * sc), "all mean": float(e.mean() * sc)}


def main():
    pd.set_option("display.width", 250); t = bk.load_tracking_tolerant(); t["key"] = t["bat"]
    st, S = pickle.load(open("output/_battrack_samples.pkl", "rb")); S2 = pickle.load(open("output/_battrack_hrbabip_samples.pkl", "rb")); n0s = st["n0"].to_dict()
    for name, s_ in S.items():
        s_["targets"].update(S2[name]); s_["window"] = eh.window(t, name, s_.get("asof")); s_["traits"] = build_all(s_["window"].assign(key=s_["window"]["bat"]))
    CP = [k for k in S["ROS25"]["traits"] if k.split("_")[0] in SERIES and k.split("_", 1)[1] in ("cp_t", "cp_shift", "cp_recent", "rec25")]; OLD = [k for k in S["ROS25"]["traits"] if k not in CP]; print(f"{len(OLD)} earlier traits, {len(CP)} changepoint features", flush=True)
    for k in CP:
        n0s[k] = 0.0                      # the changepoint features are already summary statistics of the window: no extra shrinkage
    # ---- Part 1: screen
    res = {}; tables = {}
    for sname, s in S.items():
        for tg in TARGETS:
            r = ea.screen(s["targets"][tg], s["traits"], n0s); r["q"] = ea.bh(r["p"].to_numpy()); res[(sname, tg)] = r
    for tg in TARGETS:
        a, b, pr = res[("ROS25", tg)], res[("ROS26", tg)], res[("PRE", tg)]; k = a.index.intersection(b.index); wa, wb = 1 / a.loc[k, "se"] ** 2, 1 / b.loc[k, "se"] ** 2; pc = (a.loc[k, "coef per SD"] * wa + b.loc[k, "coef per SD"] * wb) / (wa + wb); ps = 1 / np.sqrt(wa + wb)
        T = pd.DataFrame({"t ROS25": a.loc[k, "t"], "t ROS26": b.loc[k, "t"], "t PRE": pr["t"].reindex(k), "pooled coef/SD": pc, "pooled t": pc / ps}); T["pooled q"] = ea.bh(2 * norm.sf(np.abs(T["pooled t"].to_numpy()))); T["same sign"] = np.sign(T["t ROS25"]) == np.sign(T["t ROS26"]); tables[tg] = T
        cpt = T.loc[[x for x in CP if x in T.index]].sort_values("pooled t", key=np.abs, ascending=False); print(f"\nSCREEN, target {tg}: changepoint features (BH q over {len(T)} traits; coef per SD in {tg} units)\n" + cpt.head(8).round(4).to_string(), flush=True)
        print("   changepoint features with pooled q <= .10 and the same sign: " + (", ".join(cpt.index[(cpt["pooled q"] <= .10) & cpt["same sign"]]) or "none"), flush=True)
    # ---- Part 2: upside model
    out = []
    for tg in ("woba", "hr"):
        for lab, names in (("baseline: history + prediction only", []), ("earlier traits", OLD), ("earlier traits + changepoint features", OLD + CP), ("changepoint features only", CP)):
            for C in (0.03,):
                for a, b in (("ROS25", "ROS26"), ("ROS26", "ROS25"), ("ROS25", "PRE")):
                    Fa, Fb = S[a]["targets"][tg], S[b]["targets"][tg]; Xa, ua = design(Fa, S[a]["traits"], n0s, names); Xb, ub = design(Fb, S[b]["traits"], n0s, names)
                    keep = [i for i, k in enumerate(ua + ["lh", "pc"]) if (ua + ["lh", "pc"])[i] in (ub + ["lh", "pc"])]; ua_, ub_ = ua + ["lh", "pc"], ub + ["lh", "pc"]; common = [k for k in ua_ if k in ub_]; Xa = Xa[:, [ua_.index(k) for k in common]]; Xb = Xb[:, [ub_.index(k) for k in common]]
                    yua, yda = tails(Fa); yub, ydb = tails(Fb); pu, mu_ = fit_score(Xa, yua, Xb, C); pdn, _ = fit_score(Xa, yda, Xb, C); out.append({"target": tg, "model": lab, "test": f"{a} -> {b}", **metrics(Fb, pu, pdn, yub, ydb)})
                F = S["PRE"]["targets"][tg]; X, u = design(F, S["PRE"]["traits"], n0s, names); yu, yd = tails(F); pu = np.zeros(len(F)); pdn = np.zeros(len(F))
                for rep in range(5):
                    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, yu):
                        pu[te] += fit_score(X[tr], yu[tr], X[te], C)[0] / 5; pdn[te] += fit_score(X[tr], yd[tr], X[te], C)[0] / 5
                out.append({"target": tg, "model": lab, "test": "CV within PRE", **metrics(F, pu, pdn, yu, yd)})
    O = pd.DataFrame(out); print("\nUPSIDE MODEL (breakout = residual in the top 20% of its sample)\n" + O.round(3).to_string(index=False))
    g = O.groupby(["target", "model"]).mean(numeric_only=True).round(3); print("\nMEAN OVER THE FOUR TESTS\n" + g.to_string())
    # coefficients of the full model fit on ROS25 + ROS26 (pooled), for the record
    for tg in ("woba",):
        Xs, ys = [], []
        for s in ("ROS25", "ROS26"):
            F = S[s]["targets"][tg]; X, u = design(F, S[s]["traits"], n0s, OLD + CP); Xs.append(X); ys.append(tails(F)[0])
        m = LogisticRegression(C=0.03, max_iter=2000).fit(np.vstack(Xs), np.concatenate(ys)); names = u + ["lh", "pc"]; co = pd.Series(m.coef_[0], index=names).sort_values(key=np.abs, ascending=False); print(f"\nLARGEST COEFFICIENTS of the pooled ROS25 + ROS26 upside model ({tg}; standardised features)\n" + co.head(12).round(3).to_string())
    json.dump({"upside": out, "screen": {tg: T.reset_index().to_dict("records") for tg, T in tables.items()}}, open("output/changepoint_upside.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
