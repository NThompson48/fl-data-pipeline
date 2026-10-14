"""
fl_hitter_swing_preseason_check.py -- implementation check for the preseason BB% Swing% term (B = -0.387 per unit of shrunk Swing% deviation, faded by n0_bb / (PA_eff + n0_bb)).
Forward emulation of the production path: features = 2025 (+ 2024 history for the regression), target = 2026, term built from 2025 pitches only (no 2026 pitch seen), production constants.
(1) BB%: base vs base + term, fair free intercept in both, standard inverse-variance weights, hitter-cluster bootstrap.
(2) wOBA: the production wOBA = 50/50 of direct and component projections; does letting the BB% term feed the components (as the K% Whiff term does) help, hurt or do nothing?  Same fair intercept.
"""
import warnings
import numpy as np
import pandas as pd
import fl_eval_weights as ew
import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev
import fl_hitter_projection as hp
import fl_hitter_swing as hs
import fl_hitter_k_study as ks
import fl_innings_projection as ip

warnings.filterwarnings("ignore"); rng = np.random.default_rng(8)
B = -0.387


def pitches25():
    d = pd.read_parquet(ks.PITCHES); d = d[(d["year"] == 2025) & d["pa_id"].notna() & d["plate_x"].notna() & d["outside_zone"].notna()].copy()
    d["name_clean"] = d["batter"].apply(ip.lastfirst_to_clean)
    t = pd.DataFrame({"name_clean": d["name_clean"], "sw": d["pitch_call"].isin(ks.SWING).astype(int), "pit": 1}).groupby("name_clean").sum().reset_index()
    lg = t["sw"].sum() / t["pit"].sum(); t["dev"] = (t["sw"] / t["pit"] - lg) * t["pit"] / (t["pit"] + hs.N0_SWING)
    return t.set_index("name_clean")["dev"]


def boot(y, a, b, w, ids, n=800):
    by = pd.Series(np.arange(len(y))).groupby(ids).apply(np.array); ug = by.index.to_numpy(); out = []
    r = lambda p, ii: float(np.sqrt(np.average((y[ii] - p[ii]) ** 2, weights=w[ii])))
    for _ in range(n):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); out.append(r(a, ii) - r(b, ii))
    return np.array(out)


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; fit = hp.fit_all(p); dev_sw = pitches25()
    preds = {}
    for s, (delta, _) in hp.SPEC.items():
        d = bt.transitions(q, s); d = d[d["t"] == 2025].copy(); d["pred"] = bt.predict(d, fit["n0"][s], delta); preds[s] = d.set_index("name_clean")
    bb = preds["bbr"]; sd = dev_sw.reindex(bb.index).fillna(0.0)
    pa_eff = bb["n_t"] + 0.5 * bb["n_prev"]; term = B * sd * fit["n0"]["bbr"] / (pa_eff + fit["n0"]["bbr"])
    y = bb["x_next"].to_numpy(); w = bb["w"].to_numpy(); ids = bb.index.to_numpy()
    base = bb["pred"].to_numpy(); full = (bb["pred"] + term).to_numpy()
    fair = lambda p_: p_ + np.average(y - p_, weights=w)
    r = lambda p_: float(np.sqrt(np.average((y - fair(p_)) ** 2, weights=w)))
    d_ = boot(y, fair(full), fair(base), w, ids)
    print(f"BB% 2025 -> 2026 (n={len(bb)} hitters, >= 100 PA in 2026; production constants; term from 2025 pitches only): base {r(base):.5f}  + Swing% term {r(full):.5f}  change {r(full) - r(base):+.5f} [{np.percentile(d_, 5):+.5f}, {np.percentile(d_, 95):+.5f}]  P(better) {np.mean(d_ < 0):.2f}")
    # wOBA with the term feeding the components or not
    idx = bb.index; comp = fit["comp_beta"]; lgn = {s: preds[s]["lg_next"].iloc[0] for s in preds}
    def comp_dev(bb_pred):
        c = comp["k"] * (preds["k"]["pred"].reindex(idx) - lgn["k"]) + comp["bbr"] * (bb_pred - lgn["bbr"]) + comp["hrr"] * (preds["hrr"]["pred"].reindex(idx) - lgn["hrr"]) + comp["babip"] * (preds["babip"]["pred"].reindex(idx) - lgn["babip"])
        return c.fillna(0.0)
    wo = preds["woba"].reindex(idx); yw = wo["x_next"].to_numpy(); ww = wo["w"].to_numpy(); ok = wo["x_next"].notna().to_numpy()
    direct = wo["pred"]; lg_w = wo["lg_next"].iloc[0]
    blend0 = 0.5 * direct + 0.5 * (lg_w + comp_dev(bb["pred"])); blend1 = 0.5 * direct + 0.5 * (lg_w + comp_dev(bb["pred"] + term))
    f2 = lambda p_: p_[ok] + np.average(yw[ok] - p_[ok], weights=ww[ok])
    r2 = lambda p_: float(np.sqrt(np.average((yw[ok] - f2(p_)) ** 2, weights=ww[ok])))
    d2 = boot(yw[ok], f2(blend1.to_numpy()), f2(blend0.to_numpy()), ww[ok], idx.to_numpy()[ok])
    print(f"wOBA (blend, no pitch-level terms; n={ok.sum()}): term NOT feeding components {r2(blend0.to_numpy()):.5f}; feeding {r2(blend1.to_numpy()):.5f}  change {r2(blend1.to_numpy()) - r2(blend0.to_numpy()):+.5f} [{np.percentile(d2, 5):+.5f}, {np.percentile(d2, 95):+.5f}]  P(better) {np.mean(d2 < 0):.2f}")
    print(f"  mean |BB% term| {np.abs(term).mean():.4f}, max {np.abs(term).max():.4f}; mean |wOBA shift when feeding| {np.abs(blend1 - blend0).mean():.5f}")


if __name__ == "__main__":
    main()
