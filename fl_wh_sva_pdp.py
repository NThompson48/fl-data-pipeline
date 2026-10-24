"""
fl_wh_sva_pdp.py -- partial dependence of the two stuff models, and a calibration check by pitch type  (user 2026-10-23: "Changeups and splitters seemed to be a blind spot for the XGBoost model ... partial dependence plots for the two models")
MODELS: the FULL stuff model P(whiff | swing, 9 inputs) of production XGBoost (fp.XGB_PARAMS) and of the network (PyTorch MLP 128-64-32, average of 3 seeds), each fitted on ALL 2025-26 swings (position players excluded; 225,828 swings) for interpretation.
PARTIAL DEPENDENCE of input j at grid value v = the average prediction over a GROUP of swings with input j set to v and every other input left at its own value.  Groups: all pitches (5,000 random swings), changeups (5,000 random), splitters (all swings).
Averaging within a group shows each model's response INSIDE that pitch regime.  Grid = 30 points from the group's 2nd to 98th percentile of the input; the grey bars show where the group's data actually lie (the curves are only trustworthy where there are bars).
Caveat: the inputs are correlated (velocity with ivb .71, ivb with horizontal break .56), so a partial dependence curve sets an input to values it rarely takes with the others; the group restriction limits but does not remove that.
Because the location inputs do not move when a shape input is varied, a shape curve is the stuff effect (the location baseline is a constant); the two location panels include the location effect itself.
CALIBRATION by pitch type: out-of-fold (5-fold GroupKFold by pitcher; network 3 seeds) full-model predictions vs the observed whiff rate, log-loss and AUC by pitch type.
Output: output/wh_sva_pdp_{all,changeup,splitter}.png, output/wh_sva_pdp.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
SEEDS = (1, 2, 3); G = 30; NS = 5000; FE = fp.FULL_COLS
LABEL = {"effective_velo": "Velocity (mph)", "ivb": "Induced vertical break (in)", "horz_break_as": "Arm-side horizontal break (in)", "spin_rate": "Spin rate (rpm)", "plate_x_as": "Plate location, arm-side (ft)",
         "plate_z": "Plate height (ft)", "extension": "Extension (ft)", "release_pos_z": "Release height (ft)", "rel_side_as": "Release side, arm-side (ft)"}
rng = np.random.default_rng(7)


def _nn(Xtr, ytr, gtr, Xte, seed):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(Xtr, ytr, Xte, gtr, seed=seed, cfg={}); return p


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); pop = raw[raw["is_swing"]].dropna(subset=fp.FULL_COLS + ["pitcher_id", "is_whiff"]).reset_index(drop=True)
    X = pop[FE].astype(float).to_numpy(); y = pop["is_whiff"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); pt = pop["pitch_type"].to_numpy(); t0 = time.time()
    print(f"{len(y):,} swings; changeups {int((pt == 'changeup').sum()):,}, splitters {int((pt == 'splitter').sum()):,}", flush=True)
    # ---- calibration by pitch type (out of fold)
    folds = list(GroupKFold(5).split(X, y, g)); oof_x = np.zeros(len(y))
    for tr, te in folds:
        oof_x[te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    res = Parallel(n_jobs=8, backend="loky")(delayed(_nn)(X[tr], y[tr], g[tr], X[te], s) for s in SEEDS for tr, te in folds); oof_n = np.zeros(len(y)); i = 0
    for s in SEEDS:
        for tr, te in folds:
            oof_n[te] += res[i] / len(SEEDS); i += 1
    clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6); rows = []
    for t in pd.Series(pt).value_counts().index:
        m = pt == t
        if m.sum() < 1500:
            continue
        rows.append({"pitch type": t, "swings": int(m.sum()), "observed whiff %": 100 * y[m].mean(), "XGBoost mean predicted %": 100 * oof_x[m].mean(), "network mean predicted %": 100 * oof_n[m].mean(), "XGBoost logloss": log_loss(y[m], clip(oof_x[m])), "network logloss": log_loss(y[m], clip(oof_n[m])),
                     "network gain": log_loss(y[m], clip(oof_x[m])) - log_loss(y[m], clip(oof_n[m])), "XGBoost AUC": roc_auc_score(y[m], oof_x[m]), "network AUC": roc_auc_score(y[m], oof_n[m])})
    C = pd.DataFrame(rows).set_index("pitch type"); pd.set_option("display.width", 250); print(f"\nCALIBRATION BY PITCH TYPE (out of fold; {(time.time() - t0) / 60:.1f} min)\n" + C.round(4).to_string(), flush=True)
    # ---- partial dependence
    groups = {"All pitches": X[rng.choice(len(y), NS, replace=False)], "Changeups": X[rng.choice(np.where(pt == "changeup")[0], NS, replace=False)], "Splitters": X[pt == "splitter"]}
    blocks, parts, pos = {}, [], 0
    for gn, Xg in groups.items():
        for j, f in enumerate(FE):
            grid = np.quantile(Xg[:, j], np.linspace(0.02, 0.98, G)); S = np.repeat(Xg[None], G, axis=0).copy(); S[:, :, j] = grid[:, None]; parts.append(S.reshape(-1, len(FE))); blocks[(gn, f)] = (pos, grid, len(Xg)); pos += len(parts[-1])
    big = np.vstack(parts); del parts; print(f"partial-dependence grid: {len(big):,} rows", flush=True)
    xg = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(X, y); px = np.concatenate([xg.predict_proba(big[i:i + 500000])[:, 1] for i in range(0, len(big), 500000)])
    pn = np.mean(Parallel(n_jobs=3, backend="loky")(delayed(_nn)(X, y, g, big, s) for s in SEEDS), axis=0); print(f"partial-dependence predictions done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    PD = {}
    for (gn, f), (p0, grid, n) in blocks.items():
        sl = slice(p0, p0 + G * n); PD[(gn, f)] = {"grid": grid, "xgb": px[sl].reshape(G, n).mean(axis=1), "nn": pn[sl].reshape(G, n).mean(axis=1), "values": groups[gn][:, FE.index(f)]}
    summ = []
    for (gn, f), d in PD.items():
        summ.append({"group": gn, "input": f, "XGBoost range (points)": 100 * (d["xgb"].max() - d["xgb"].min()), "network range (points)": 100 * (d["nn"].max() - d["nn"].min()), "max |network - XGBoost| (points)": 100 * np.abs(d["nn"] - d["xgb"]).max(),
                     "curve correlation": float(np.corrcoef(d["xgb"], d["nn"])[0, 1]), "mean level diff (points)": 100 * (d["nn"] - d["xgb"]).mean()})
    S_ = pd.DataFrame(summ); print("\nPARTIAL DEPENDENCE SUMMARY (whiff probability, percentage points)\n" + S_.round(2).to_string(index=False))
    ink, ink2, grid_c, surf = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"; Cx, Cn = "#eb6834", "#2a78d6"
    for gn, tag in (("All pitches", "all"), ("Changeups", "changeup"), ("Splitters", "splitter")):
        fig, axes = plt.subplots(3, 3, figsize=(12.5, 10.2), facecolor=surf); allv = np.concatenate([np.r_[PD[(gn, f)]["xgb"], PD[(gn, f)]["nn"]] for f in FE]); lo, hi = 100 * allv.min() - 1.0, 100 * allv.max() + 1.0
        for ax, f in zip(axes.ravel(), FE):
            d = PD[(gn, f)]; ax.set_facecolor(surf); ax2 = ax.twinx(); h, e = np.histogram(d["values"], bins=24, range=(d["grid"][0], d["grid"][-1])); ax2.bar((e[:-1] + e[1:]) / 2, h / h.max(), width=(e[1] - e[0]) * 0.92, color="#d9d8d2", zorder=0); ax2.set_ylim(0, 6); ax2.axis("off")
            ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
            ax.plot(d["grid"], 100 * d["xgb"], color=Cx, lw=2.0, ls="--", label="XGBoost"); ax.plot(d["grid"], 100 * d["nn"], color=Cn, lw=2.0, ls="-", label="Network"); ax.set_ylim(lo, hi)
            ax.set_xlabel(LABEL[f], color=ink2, fontsize=9.5); ax.grid(axis="y", color=grid_c, lw=0.8); ax.set_axisbelow(True); [ax.spines[k].set_visible(False) for k in ("top", "right")]; [ax.spines[k].set_color(grid_c) for k in ("left", "bottom")]; ax.tick_params(colors=ink2, labelsize=9)
        for r in range(3):
            axes[r, 0].set_ylabel("Whiff probability on a swing (%)", color=ink2, fontsize=9.5)
        axes[0, 0].legend(frameon=False, fontsize=9.5, labelcolor=ink, loc="upper left")
        fig.suptitle(f"Partial dependence of whiff probability, {gn.lower()} ({len(groups[gn]):,} swings): XGBoost vs network.  Grey bars = where the data are.", color=ink, fontsize=12, x=0.01, ha="left", y=0.995); fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"output/wh_sva_pdp_{tag}.png", dpi=140, facecolor=surf); plt.close(fig)
    json.dump({"calibration": C.reset_index().to_dict("records"), "pdp_summary": summ, "curves": {f"{gn}|{f}": {"grid": d["grid"].tolist(), "xgb": d["xgb"].tolist(), "nn": d["nn"].tolist()} for (gn, f), d in PD.items()}}, open("output/wh_sva_pdp.json", "w"), indent=1, default=float)
    print("saved figures", flush=True)


if __name__ == "__main__":
    main()
