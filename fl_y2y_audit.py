"""
fl_y2y_audit.py -- audit of the year-to-year (2025 -> 2026) correlation of the swstr_ct stuff level, and a head-to-head with P(swstr | stuff), network AND XGBoost  (user 2026-10-26: "I am concerned about the weak yoy correlation. Can we double check ...")
FOUR MEASURES (per-pitch swinging strike target, level = mean over a pitcher's pitches, cross-fitted by pitcher): net swstr_ct (9 inputs + pitch type + count, location baseline with the same type and count) | net swstr|stuff (7 stuff inputs only) |
xgb swstr_ct (same inputs as net swstr_ct; pitch type one-hot) | xgb swstr|stuff.
CHECKS:  (1) REPRODUCTION: the separate-fit year-to-year r recomputed from scratch against the logged values (saved level files).   (2) TWO DESIGNS: SEPARATE fits (each season's level from that season's own cross-fit, as before) vs ONE POOLED fit on 2025+2026
(a single model, levels split by season: removes any year-specific refit / baseline difference).  (3) DISATTENUATION: year-to-year r divided by sqrt(split-half reliability 2025 x 2026) (alternating games, Spearman-Brown) on the same pitchers: separates measurement noise from real change.
(4) RANGE: the between-pitcher SD of the level by volume group (a correlation falls when the spread falls).  (5) PAIRED BOOTSTRAP (2,000 pitcher resamples) of the difference in year-to-year r between measures on the SAME pitchers.
(6) CELL LEVEL: year-to-year r of pitcher x pitch-type cells (>= 100 pitches of that type in each year), by type, i.e. is stuff WITHIN a pitch type stable, apart from pitch mix.   Thresholds = pitches thrown in EACH year.   Output: output/y2y_audit.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf

warnings.filterwarnings("ignore")
KEYS = ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]; STUFF = [c for c in fp.FULL_COLS if c not in fp.LOC_COLS]; THR = (150, 300, 500, 800); rng = np.random.default_rng(2026)
SEASONS = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}
TYPES = ["four_seam", "sinker", "slider", "changeup", "curveball"]


def xgb_cv(X, y, g):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        p[te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def xgb_swstr_ct(s):
    d = fp.add_count(s[s["pitch_call"] != "undefined"]); pop = d.dropna(subset=fp.FULL_COLS + ["pitcher_id", "balls", "strikes"]).copy(); pop["pitch_type"] = pop["pitch_type"].fillna("unk").replace("", "unk")
    y = (pop["is_swing"] & pop["is_whiff"]).astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); oh = pd.get_dummies(pop["pitch_type"]).astype(float).to_numpy()
    Xf = np.column_stack([pop[fp.FULL_COLS + ["balls", "strikes"]].astype(float).to_numpy(), oh]); Xl = np.column_stack([pop[fp.LOC_COLS + ["balls", "strikes"]].astype(float).to_numpy(), oh])
    return pop[KEYS + ["pitch_type"]].assign(sva=xgb_cv(Xf, y, g) - xgb_cv(Xl, y, g))


def xgb_swstr_stuff(s):
    pop = s[s["pitch_call"] != "undefined"].dropna(subset=STUFF + KEYS).copy(); y = (pop["is_swing"] & pop["is_whiff"]).astype(int).to_numpy(); p = xgb_cv(pop[STUFF].astype(float).to_numpy(), y, pop["pitcher_id"].to_numpy())
    return pop[KEYS + ["pitch_type"]].assign(sva=p - p.mean())


def net(s, est):
    sv = fp.compute_pooled_wh_sva(s, estimator=est); return sv.merge(s[KEYS + ["pitch_type"]].drop_duplicates(KEYS), on=KEYS, how="left")


MEAS = {"net swstr_ct": lambda s: net(s, "swstr_ct"), "net swstr|stuff": lambda s: net(s, "swstr_stuff"), "xgb swstr_ct": xgb_swstr_ct, "xgb swstr|stuff": xgb_swstr_stuff}


def fisher(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); return f"[{np.tanh(z - 1.96 * se):.2f}, {np.tanh(z + 1.96 * se):.2f}]"


def halves(rows, s):
    g = s.drop_duplicates(["game_id", "pitcher_id"])[["game_id", "pitcher_id", "date"]].sort_values("date"); g["half"] = g.groupby("pitcher_id").cumcount() % 2
    d = rows.merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"]); return d.groupby(["pitcher_id", "half"])["sva"].mean().unstack().dropna()


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); S = {yr: fp.filter_period(raw, *w) for yr, w in SEASONS.items()}; pool = pd.concat([S[2025], S[2026]], ignore_index=True)
    n_thrown = {yr: S[yr].groupby("pitcher_id").size() for yr in S}; R = {}
    for name, f in MEAS.items():
        R[name] = {"sep": {yr: f(S[yr]) for yr in S}}; allr = f(pool); allr["year"] = pd.to_datetime(allr["date"]).dt.year; R[name]["pool"] = {yr: allr[allr["year"] == yr] for yr in S}; print(f"  {name} done", flush=True)
    lev = lambda rows: rows.groupby("pitcher_id")["sva"].mean(); out = {}
    # (1) reproduction against the logged / saved separate-fit levels
    sv = {yr: pd.read_csv(f"output/y2y_levels_{yr}.csv", index_col=0, dtype={0: str}) for yr in S}; sv2 = {yr: pd.read_csv(f"output/strike_stuff_levels_{yr}.csv", index_col=0, dtype={0: str}) for yr in S}
    print("\n(1) REPRODUCTION: separate-fit level now vs the saved level (correlation across pitchers) and the year-to-year r (>= 300 pitches each year)")
    for name, saved in (("net swstr_ct", lambda y: sv[y]["swstr_ct"]), ("net swstr|stuff", lambda y: sv2[y]["swstr|stuff network"])):
        c = [float(np.corrcoef(*pd.concat([lev(R[name]["sep"][y]).rename("a"), saved(y).rename("b").set_axis(saved(y).index.astype(str))], axis=1).dropna().T.to_numpy())[0, 1]) for y in S]
        j = pd.concat([lev(R[name]["sep"][2025]).rename("a"), lev(R[name]["sep"][2026]).rename("b"), n_thrown[2025].rename("n25"), n_thrown[2026].rename("n26")], axis=1).dropna(); j = j[(j["n25"] >= 300) & (j["n26"] >= 300)]
        js = pd.concat([saved(2025).rename("a"), saved(2026).rename("b"), sv[2025]["n_pitches"].rename("n25"), sv[2026]["n_pitches"].rename("n26")], axis=1).dropna(); js = js[(js["n25"] >= 300) & (js["n26"] >= 300)]
        print(f"   {name:18s} level corr with saved: 2025 {c[0]:.4f}, 2026 {c[1]:.4f};  y2y r now {np.corrcoef(j['a'], j['b'])[0, 1]:.3f} (n {len(j)}) vs saved {np.corrcoef(js['a'], js['b'])[0, 1]:.3f} (n {len(js)})")
    # (2)-(4) y2y by design and threshold, disattenuation, range
    rows = []
    for design in ("sep", "pool"):
        for name in MEAS:
            L = {yr: lev(R[name][design][yr]) for yr in S}; H = {yr: halves(R[name][design][yr], S[yr]) for yr in S}
            for thr in THR:
                j = pd.concat([L[2025].rename("a"), L[2026].rename("b"), n_thrown[2025].rename("n25"), n_thrown[2026].rename("n26")], axis=1).dropna(); j = j[(j["n25"] >= thr) & (j["n26"] >= thr)]; r = float(np.corrcoef(j["a"], j["b"])[0, 1])
                sb = []
                for yr in S:
                    h = H[yr].reindex(j.index).dropna(); rh = float(np.corrcoef(h[0], h[1])[0, 1]); sb.append(2 * rh / (1 + rh))
                rows.append({"design": design, "measure": name, "thr": thr, "n": len(j), "r": r, "CI": fisher(r, len(j)), "SB 2025": sb[0], "SB 2026": sb[1], "disattenuated r": min(r / np.sqrt(sb[0] * sb[1]), 1.0), "SD level 2025": float(j["a"].std()), "SD level 2026": float(j["b"].std())})
    T = pd.DataFrame(rows); pd.set_option("display.width", 280); out["table"] = rows
    for design in ("sep", "pool"):
        print(f"\n(2-4) {'SEPARATE per-season fits' if design == 'sep' else 'ONE POOLED fit on 2025+2026'}: year-to-year r (CI), n, split-half S-B 2025 | 2026, disattenuated r, between-pitcher SD of the level (2025 | 2026)\n" + T[T["design"] == design].drop(columns="design").round(3).to_string(index=False))
    # (5) paired bootstrap of r differences
    print("\n(5) PAIRED BOOTSTRAP of the difference in year-to-year r (A minus B; same pitchers; 2,000 pitcher resamples; pooled-fit levels | separate-fit levels)")
    pairs = (("net swstr_ct", "net swstr|stuff"), ("xgb swstr_ct", "xgb swstr|stuff"), ("net swstr_ct", "xgb swstr_ct"), ("net swstr|stuff", "xgb swstr|stuff")); boot = []
    for design in ("pool", "sep"):
        L = {name: pd.concat([lev(R[name][design][2025]).rename("a"), lev(R[name][design][2026]).rename("b")], axis=1) for name in MEAS}
        for a, b in pairs:
            for thr in (150, 300, 500):
                base = pd.concat([n_thrown[2025].rename("n25"), n_thrown[2026].rename("n26")], axis=1).dropna(); ids = base[(base["n25"] >= thr) & (base["n26"] >= thr)].index
                A = L[a].reindex(ids).dropna(); B = L[b].reindex(A.index).dropna(); A = A.reindex(B.index); ra = np.corrcoef(A["a"], A["b"])[0, 1]; rb = np.corrcoef(B["a"], B["b"])[0, 1]; ds = []
                for _ in range(2000):
                    ix = rng.integers(0, len(A), len(A)); ds.append(np.corrcoef(A["a"].to_numpy()[ix], A["b"].to_numpy()[ix])[0, 1] - np.corrcoef(B["a"].to_numpy()[ix], B["b"].to_numpy()[ix])[0, 1])
                boot.append({"design": design, "A": a, "B": b, "thr": thr, "n": len(A), "r A": ra, "r B": rb, "diff": ra - rb, "90% CI": f"[{np.percentile(ds, 5):+.3f}, {np.percentile(ds, 95):+.3f}]", "P(A > B)": float(np.mean(np.array(ds) > 0))})
    out["paired"] = boot; print(pd.DataFrame(boot).round(3).to_string(index=False))
    # (6) cell level
    print("\n(6) CELL LEVEL (pitcher x pitch type, >= 100 pitches of that type in EACH year; pooled-fit levels): year-to-year r (n)")
    cell = []
    for name in MEAS:
        r = {"measure": name}
        for t in TYPES:
            c = {}
            for yr in S:
                d = R[name]["pool"][yr]; d = d[d["pitch_type"] == t]; g = d.groupby("pitcher_id")["sva"].agg(["mean", "size"]); c[yr] = g[g["size"] >= 100]["mean"]
            j = pd.concat([c[2025].rename("a"), c[2026].rename("b")], axis=1).dropna(); r[t] = f"{np.corrcoef(j['a'], j['b'])[0, 1]:.2f} ({len(j)})" if len(j) > 8 else "n/a"
        cell.append(r)
    out["cell"] = cell; print(pd.DataFrame(cell).to_string(index=False))
    json.dump(out, open("output/y2y_audit.json", "w"), indent=1, default=float)
    for name in MEAS:
        for yr in S:
            R[name]["pool"][yr][["pitcher_id", "sva"]].groupby("pitcher_id")["sva"].mean().to_csv(f"output/y2y_audit_level_{name.replace(' ', '_').replace('|', '-')}_{yr}.csv")


if __name__ == "__main__":
    main()
