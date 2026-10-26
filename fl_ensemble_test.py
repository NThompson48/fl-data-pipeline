"""
fl_ensemble_test.py -- does AVERAGING the predictions of two candidate models help the projection models?  (user 2026-10-25)
K% CANDIDATES (each an additive K% model with the stuff term replaced by that measure; same other terms, each fit on the fit transition): pub = XGBoost wh_sva (published) | net = swing network wh_sva | sw = swstr_ct | ss = P(swstr | stuff) network | st = P(strike | stuff) network.
ENSEMBLES: the equal-weight average of the two models' K% predictions, ONLY for pairs whose stuff models have different target variables (user 2026-10-25): targets are whiff per swing (pub, net), swinging strike per pitch (sw, ss), strike per pitch (st) -> 8 pairs:
pub+sw, pub+ss, net+sw, net+ss, pub+st, net+st, sw+st, ss+st.   BB% CANDIDATES: pub = production blended rate + command score | +sw | +ss | +st (the stuff level added); ensembles = sw+st and ss+st (the no-stuff production model is not paired).
PROTOCOL: the four-pairing robustness set of fl_wh_sva_prior_xgb_robust_test.py (T0->T1b, T1->T2, T0->T2, T2->T0; fit on the first, score the second, nothing refit); frozen standard weights from the PUBLISHED model on each test set; level-free; weighted RMSE and MAE; pitcher bootstrap (1,000) vs the published model and vs the best single candidate.
Averaging is of final predictions, not of the stuff levels (an average of levels would be one more feature and is not tested).   Output: output/ensemble_test.json
"""
import itertools
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
import fl_wh_sva_prior_k_test as pk
import fl_wh_sva_prior_xgb_robust_test as rb
import mlb_marcel as mm

warnings.filterwarnings("ignore")
ADD = pp.SPEC["K"]; BBS = pp.BB_SPEC_BLENDED; EST = {"pub": "xgb", "net": "torch3", "sw": "swstr_ct", "ss": "swstr_stuff", "st": "strike_stuff"}; _c = {}
BASE_K = [f for f in ADD if f != "wh_sva_level"]
TARGET = {"pub": "whiff per swing", "net": "whiff per swing", "sw": "swinging strike per pitch", "ss": "swinging strike per pitch", "st": "strike per pitch"}   # target variable of each candidate's stuff model


def lg_rate(raw, windows, what):
    t = pd.concat([fp.filter_period(raw, a, b) for a, b in windows], ignore_index=True); t = t[t["is_terminal"]]; return float((t["k_or_bb"] == what).mean())


def frame(raw, est, name):
    if (est, name) not in _c:
        fp.WH_SVA_ESTIMATOR = est; feat, lab = rb.TR[name]; df, n0_k, n0_bb = bt.assemble(raw, feat, lab, with_changepoints=False); _c[(est, name)] = (df, n0_k, n0_bb, lg_rate(raw, feat, "strikeout"), lg_rate(raw, feat, "walk"))
    return _c[(est, name)]


def combined(raw, name):
    """the pub-estimator frame with every candidate's stuff level merged in (pub, net, sw, ss, st), plus the frame constants."""
    df, n0_k, n0_bb, lg_k, lg_bb = frame(raw, "xgb", name); d = df.rename(columns={"wh_sva_level": "pub"})
    for key, est in EST.items():
        if key != "pub":
            d = d.merge(frame(raw, est, name)[0][["pitcher_id", "wh_sva_level"]].rename(columns={"wh_sva_level": key}), on="pitcher_id", how="left")
    return d, n0_k, n0_bb, lg_k, lg_bb


def k_specs():
    return {k: BASE_K + [k] for k in EST}


def bb_specs():
    return {"pub": BBS, "sw": BBS + ["sw"], "ss": BBS + ["ss"], "st": BBS + ["st"]}


def fit(raw, name):
    d, n0_k, n0_bb, lg_k, lg_bb = combined(raw, name); out = {"K": {}, "BB": {}}
    dk = d.dropna(subset=["K_rate_actual"] + BASE_K + list(EST)).copy(); wk = mm.reliability_weight(dk["PA_test"], n0_k).values
    for k, spec in k_specs().items():
        out["K"][k] = bt.fit_weighted(dk.rename(columns={"pub": "wh_sva_level"}), "K_rate_actual", [("wh_sva_level" if s == "pub" else s) for s in spec], wk)
    need = ["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score"] + list(EST); db = d.dropna(subset=need).copy(); wb = mm.reliability_weight(db["PA_test"], n0_bb).values
    db["BB_raw"] = db["naive_BB_rate"] * (db["PA"] + n0_bb) - n0_bb * lg_bb; n0b = mm.fit_n0_bridge(db["BB_raw"], db["PA"], n0_bb, lg_bb, db["fl_bb_rate_bridge"], db["BB_rate_actual"], wb)
    db["blended_BB_rate"] = mm.blended_rate(db["BB_raw"], db["PA"], n0_bb, lg_bb, n0b, db["fl_bb_rate_bridge"])
    for k, spec in bb_specs().items():
        out["BB"][k] = bt.fit_weighted(db, "BB_rate_actual", spec, wb)
    out["n0_bridge_bb"] = n0b; out["n0_bb"] = n0_bb; return out


def predictions(raw, M, test):
    d, n0_k, n0_bb, lg_k, lg_bb = combined(raw, test); res = {}
    tk = d.dropna(subset=["K_rate_actual"] + BASE_K + list(EST)).copy().rename(columns={"pub": "wh_sva_level"})
    P = {k: M["K"][k].predict(tk).to_numpy() for k in EST}; res["K"] = (tk["K_rate_actual"].to_numpy(), tk["PA_test"].to_numpy(), tk["pitcher_id"].to_numpy(), P)
    need = ["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score"] + list(EST); tb = d.dropna(subset=need).copy(); tb["BB_raw"] = tb["naive_BB_rate"] * (tb["PA"] + n0_bb) - n0_bb * lg_bb
    tb["blended_BB_rate"] = mm.blended_rate(tb["BB_raw"], tb["PA"], n0_bb, lg_bb, M["n0_bridge_bb"], tb["fl_bb_rate_bridge"]); PB = {k: M["BB"][k].predict(tb).to_numpy() for k in bb_specs()}
    res["BB"] = (tb["BB_rate_actual"].to_numpy(), tb["PA_test"].to_numpy(), tb["pitcher_id"].to_numpy(), PB); return res


def evaluate(stat, y, n_t, ids, P, best):
    keys = list(P); cands = dict(P)
    for a, b in itertools.combinations(keys, 2):
        if TARGET[a] != TARGET[b] and not (stat == "BB" and "pub" in (a, b)):      # only pairs whose stuff models have DIFFERENT target variables (BB: the no-stuff production model is not paired)
            cands[f"{a}+{b}"] = (P[a] + P[b]) / 2
    w = ew.weights_from_base(y, P["pub"], n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w); rows = []
    for lab, q in cands.items():
        rm, ma = er.wmetrics(y, cen(q), w); r = {"model": lab, "RMSE": rm, "MAE": ma}
        if lab != "pub":
            c = er.compare(y, cen(P["pub"]), cen(q), w, ids, n=1000); r.update({"dRMSE": c["dRMSE"], "P(RMSE)": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE)": c["P(MAE better)"]})
            if lab != best:
                c2 = er.compare(y, cen(cands[best]), cen(q), w, ids, n=1000); r.update({f"dRMSE vs {best}": c2["dRMSE"], f"P vs {best}": c2["P(RMSE better)"], f"dMAE vs {best}": c2["dMAE"]})
        rows.append(r)
    return rows


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); allrows = []; res = {}
    for a, b in rb.PAIRS:
        M = fit(raw, a); R = predictions(raw, M, b); res[f"{a}->{b}"] = {}
        for stat, best in (("K", "sw"), ("BB", "pub")):
            y, n_t, ids, P = R[stat]; rows = evaluate(stat, y, n_t, ids, P, best); res[f"{a}->{b}"][stat] = {"n": len(y), "rows": rows}
            for r in rows:
                allrows.append({"stat": stat, "pairing": f"{a}->{b}", "n": len(y), **r})
            pd.set_option("display.width", 280); print(f"\n== {stat}%: fit {a} -> test {b} (n = {len(y)})\n" + pd.DataFrame(rows).round(5).to_string(index=False), flush=True)
    T = pd.DataFrame(allrows); print("\n\nSUMMARY ACROSS THE FOUR PAIRINGS (mean RMSE / MAE; mean change vs pub; pairings where RMSE and MAE are both better than pub)")
    for stat in ("K", "BB"):
        t = T[T["stat"] == stat]; g = t.groupby("model").agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), dRMSE=("dRMSE", "mean"), dMAE=("dMAE", "mean"), both_better=("dRMSE", lambda s: 0)).reset_index()
        bb = t[(t["model"] != "pub")].assign(ok=lambda x: (x["dRMSE"] < 0) & (x["dMAE"] < 0)).groupby("model")["ok"].sum(); g["both_better"] = g["model"].map(bb).fillna(0).astype(int)
        print(f"\n{stat}%\n" + g.sort_values("RMSE").round(5).to_string(index=False))
    json.dump({"pairs": res}, open("output/ensemble_test.json", "w"), indent=1, default=float); T.to_csv("output/ensemble_test_rows.csv", index=False)


if __name__ == "__main__":
    main()
