"""
fl_wh_sva_pdp2d.py -- the MOST DIFFERENT two-dimensional partial dependence plots between the XGBoost and network stuff models  (user 2026-10-24: "Can you show the most different 2D PDPs?")
Same models as fl_wh_sva_pdp.py (full stuff model P(whiff | swing, 9 inputs); XGBoost production params; network = 3-seed PyTorch MLP; fitted on all 225,830 swings).  For every pair of the 9 inputs (36 pairs) and each group of swings
(all pitches, changeups, splitters; 800 random swings each, all splitters if fewer), the 2D partial dependence on an 11 x 11 grid (each input's 2nd-98th percentile in the group): the average prediction with the pair set to the grid values and every
other input left at its own value.  Cells with little data are MASKED: a cell is supported if at least 0.3% of the group's swings fall in it (cell edges halfway between grid points); all metrics use supported cells only, so sparse corners (where either model just extrapolates) cannot win the ranking.
RANKING (per group), over supported cells:  (1) 'shape difference' = RMS of the centred difference D = network - XGBoost (level shifts removed);  (2) 'interaction difference' = RMS difference between the two models' INTERACTION surfaces, where a surface's interaction part is what is left after
removing the additive row + column effects (the part that is not explained by the two inputs separately);  also max |D|, and each model's own interaction strength (RMS of its interaction part).  The figures show the top pairs by shape difference: XGBoost | network | network - XGBoost.
Output: output/wh_sva_pdp2d_{all,changeup,splitter}.png, output/wh_sva_pdp2d.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import itertools
import json
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
FE = fp.FULL_COLS; G = 11; NS = 800; SEEDS = (1, 2, 3); SUPPORT = 0.003; TOP = 4
SHORT = {"effective_velo": "Velocity (mph)", "ivb": "Induced vert. break (in)", "horz_break_as": "Arm-side horiz. break (in)", "spin_rate": "Spin rate (rpm)", "plate_x_as": "Plate location, arm-side (ft)", "plate_z": "Plate height (ft)",
         "extension": "Extension (ft)", "release_pos_z": "Release height (ft)", "rel_side_as": "Release side (ft)"}
PAIRS = list(itertools.combinations(range(len(FE)), 2)); rng = np.random.default_rng(11)


def _nn(Xtr, ytr, gtr, Xte, seed):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(Xtr, ytr, Xte, gtr, seed=seed, cfg={}); return p


def interaction(S):
    return S - (S.mean(axis=1, keepdims=True) + S.mean(axis=0, keepdims=True) - S.mean())


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); pop = raw[raw["is_swing"]].dropna(subset=fp.FULL_COLS + ["pitcher_id", "is_whiff"]).reset_index(drop=True)
    X = pop[FE].astype(float).to_numpy(); y = pop["is_whiff"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); pt = pop["pitch_type"].to_numpy(); t0 = time.time()
    idx = {"All pitches": rng.choice(len(y), NS, replace=False), "Changeups": rng.choice(np.where(pt == "changeup")[0], NS, replace=False), "Splitters": np.where(pt == "splitter")[0]}
    groups = {k: X[v] for k, v in idx.items()}; parts, meta, pos = [], {}, 0
    for gn, Xg in groups.items():
        for (i, j) in PAIRS:
            gi, gj = (np.quantile(Xg[:, c], np.linspace(0.02, 0.98, G)) for c in (i, j)); S = np.repeat(Xg[None], G * G, axis=0).copy(); a, b = np.meshgrid(gi, gj, indexing="ij"); S[:, :, i] = a.reshape(-1, 1); S[:, :, j] = b.reshape(-1, 1)
            parts.append(S.reshape(-1, len(FE)).astype(np.float32)); meta[(gn, i, j)] = (pos, gi, gj, len(Xg)); pos += len(parts[-1])
    big = np.vstack(parts); del parts; print(f"{len(big):,} grid rows ({len(PAIRS)} pairs x {len(groups)} groups)", flush=True)
    xg = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(X, y); px = np.concatenate([xg.predict_proba(big[i:i + 500000])[:, 1] for i in range(0, len(big), 500000)])
    pn = np.mean(Parallel(n_jobs=3, backend="loky")(delayed(_nn)(X, y, g, big, s) for s in SEEDS), axis=0); del big; print(f"predictions done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    R = {}; rows = []
    for (gn, i, j), (p0, gi, gj, n) in meta.items():
        sl = slice(p0, p0 + G * G * n); Sx = 100 * px[sl].reshape(G, G, n).mean(axis=2); Sn = 100 * pn[sl].reshape(G, G, n).mean(axis=2); Xg = groups[gn]
        ei = np.r_[-np.inf, (gi[:-1] + gi[1:]) / 2, np.inf]; ej = np.r_[-np.inf, (gj[:-1] + gj[1:]) / 2, np.inf]; cnt = np.histogram2d(Xg[:, i], Xg[:, j], bins=[ei, ej])[0] / len(Xg); sup = cnt >= SUPPORT
        D = Sn - Sx; Dc = D - D[sup].mean(); Ix, In = interaction(Sx), interaction(Sn); R[(gn, i, j)] = {"gi": gi, "gj": gj, "Sx": Sx, "Sn": Sn, "sup": sup}
        rows.append({"group": gn, "pair": f"{FE[i]} x {FE[j]}", "i": i, "j": j, "supported cells": int(sup.sum()), "shape difference (RMS, pts)": float(np.sqrt((Dc[sup] ** 2).mean())), "max |difference| (pts)": float(np.abs(D[sup]).max()),
                     "interaction difference (RMS, pts)": float(np.sqrt(((In - Ix)[sup] ** 2).mean())), "XGBoost interaction (RMS)": float(np.sqrt((Ix[sup] ** 2).mean())), "network interaction (RMS)": float(np.sqrt((In[sup] ** 2).mean()))})
    T = pd.DataFrame(rows); pd.set_option("display.width", 250)
    for gn in groups:
        t = T[T["group"] == gn].sort_values("shape difference (RMS, pts)", ascending=False); print(f"\n== {gn}: pairs ranked by shape difference (supported cells only)\n" + t.drop(columns=["group", "i", "j"]).head(10).round(2).to_string(index=False))
    seq = LinearSegmentedColormap.from_list("seq", ["#eef3fb", "#2a78d6", "#0e2f5a"]); div = LinearSegmentedColormap.from_list("div", ["#eb6834", "#e9e8e3", "#2a78d6"]); ink, ink2, surf = "#0b0b0b", "#52514e", "#fcfcfb"
    for gn, tag in (("All pitches", "all"), ("Changeups", "changeup"), ("Splitters", "splitter")):
        top = T[T["group"] == gn].sort_values("shape difference (RMS, pts)", ascending=False).head(TOP); fig, axes = plt.subplots(TOP, 3, figsize=(13.2, 3.55 * TOP), facecolor=surf)
        vmax = max(np.abs(R[(gn, r.i, r.j)]["Sn"] - R[(gn, r.i, r.j)]["Sx"])[R[(gn, r.i, r.j)]["sup"]].max() for r in top.itertuples()); vmax = np.ceil(vmax)
        for k, r in enumerate(top.itertuples()):
            d = R[(gn, r.i, r.j)]; lo = min(np.ma.masked_where(~d["sup"], d["Sx"]).min(), np.ma.masked_where(~d["sup"], d["Sn"]).min()); hi = max(np.ma.masked_where(~d["sup"], d["Sx"]).max(), np.ma.masked_where(~d["sup"], d["Sn"]).max())
            for c, (Z, cm, vl, vh, ttl) in enumerate(((d["Sx"], seq, lo, hi, "XGBoost"), (d["Sn"], seq, lo, hi, "Network"), (d["Sn"] - d["Sx"], div, -vmax, vmax, "Network - XGBoost"))):
                ax = axes[k, c]; ax.set_facecolor("#ecebe6"); M = np.ma.masked_where(~d["sup"], Z); q = ax.pcolormesh(d["gi"], d["gj"], M.T, cmap=cm, vmin=vl, vmax=vh, shading="gouraud" if False else "auto")
                ax.set_xlabel(SHORT[FE[r.i]], color=ink2, fontsize=9); ax.set_ylabel(SHORT[FE[r.j]], color=ink2, fontsize=9); ax.tick_params(colors=ink2, labelsize=8); [ax.spines[s].set_visible(False) for s in ax.spines]
                if k == 0:
                    ax.set_title(ttl + ("  (whiff %)" if c < 2 else "  (points)"), color=ink, fontsize=10.5, loc="left")
                cb = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=8, colors=ink2); cb.outline.set_visible(False)
        fig.suptitle(f"Most different 2D partial dependence, {gn.lower()}: whiff probability on a swing.  Grey = too little data.", color=ink, fontsize=12, x=0.01, ha="left", y=0.998); fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(f"output/wh_sva_pdp2d_{tag}.png", dpi=130, facecolor=surf); plt.close(fig)
    json.dump({"ranking": T.to_dict("records")}, open("output/wh_sva_pdp2d.json", "w"), indent=1, default=float); print("saved figures", flush=True)


if __name__ == "__main__":
    main()
