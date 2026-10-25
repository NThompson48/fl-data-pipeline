"""
fl_swstr_vs_net_deep.py -- deeper comparison of the per-pitch swinging-strike stuff model (swstr_ct) and the swing-based network  (user 2026-10-24: "deeper dive ... partial dependence plots, specific pitchers, and pitch types")
MODELS (both PyTorch MLPs, 3 seeds, fitted on ALL 2025-26 data for the PDPs): NET = P(whiff | swing, 9 inputs) on swings;  SWSTR = P(swinging strike | 9 inputs + balls + strikes + pitch-type embedding) on every pitch.
(1) PARTIAL DEPENDENCE by pitch type (four-seam, sinker, slider, changeup, curveball, sweeper): for each shape input the average prediction over a group of that pitch type's pitches with the input set to the grid value (everything else, including count, as thrown).
    The two models predict different things (per swing vs per pitch), so each curve is shown as the % change from that model's own average prediction on the group ('relative effect'); the shapes, not the levels, are compared.   (2) COUNT effect of SWSTR by pitch type, relative to 0-0.
(3) PITCH-TYPE table: for each pitch type, out-of-fold (pooled 2025-26 window, fl_pitcher_features.compute_pooled_wh_sva with each estimator) pitcher-level sva of both, their correlation across pitcher x type cells (>= 100 pitches),
    and split-half reliability (alternating games; cells with >= 60 pitches and >= 25 swings per half).
(4) NAMED PITCHERS: per pitch type, each pitcher's cell z-score (within pitch type) under both models, observed swinging-strike and whiff rates, and the difference.   Scatter figure: SWSTR z vs NET z by pitch type with the largest gaps labelled.
Output: output/swstr_vs_net_{pdp,count,scatter}.png, output/swstr_vs_net_deep.json / _cells.csv
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
TYPES = ["four_seam", "sinker", "slider", "changeup", "curveball", "sweeper"]
SHAPE = ["effective_velo", "ivb", "horz_break_as", "spin_rate", "extension", "release_pos_z"]
SEEDS = (1, 2, 3); G = 20; NS = 600
LAB = {"effective_velo": "Velocity (mph)", "ivb": "Induced vert. break (in)", "horz_break_as": "Arm-side horiz. break (in)", "spin_rate": "Spin rate (rpm)", "extension": "Extension (ft)", "release_pos_z": "Release height (ft)"}
NAMED = ["Rohde, Isaac", "Parks, Pavin", "Heredia-Bustos, Rolando", "Peters, Garrett", "Widener, Jacob", "Harper, Scott", "Morgan, Cooper", "Melendez, Omar", "Martinez, Gregory", "Primeaux, Parker"]
rng = np.random.default_rng(23)


def _fit(Xtr, ytr, Xte, g, seed, C=None, Cte=None):
    torch.set_num_threads(2)
    if C is None:
        p, ep, b = ft.fit_predict(Xtr, ytr, Xte, g, seed=seed, cfg={})
    else:
        p, ep, b = ft.fit_predict(Xtr, ytr, Xte, g, seed=seed, cfg={"n_cat": int(C.max()) + 1}, cat_tr=C, cat_te=Cte)
    return p


def main():
    t0 = time.time(); raw = ppf.exclude(fp.load_fl_pitches_cached()); win = fp.add_count(raw[raw["pitch_call"] != "undefined"]); win["target"] = (win["is_swing"] & win["is_whiff"]).astype(int)
    pit = win.dropna(subset=fp.FULL_COLS + ["pitcher_id", "balls", "strikes"]).reset_index(drop=True); pit["pitch_type"] = pit["pitch_type"].fillna("unk").replace("", "unk")
    cats = pd.Categorical(pit["pitch_type"]); C = cats.codes.astype(int)
    CNT = ["balls", "strikes"]; Xs = pit[fp.FULL_COLS + CNT].astype(float).to_numpy(); ys = pit["target"].to_numpy(); g = pit["pitcher_id"].to_numpy(); sw = pit["is_swing"].to_numpy()
    Xn = pit.loc[sw, fp.FULL_COLS].astype(float).to_numpy(); yn = pit.loc[sw, "is_whiff"].astype(int).to_numpy(); gn = g[sw]
    fi = {c: i for i, c in enumerate(fp.FULL_COLS + CNT)}; print(f"{len(pit):,} pitches, {int(sw.sum()):,} swings", flush=True)
    # ---- (1)+(2) partial dependence
    groups = {t: np.where(pit["pitch_type"].to_numpy() == t)[0] for t in TYPES}
    groups = {t: rng.choice(ix, min(NS, len(ix)), replace=False) for t, ix in groups.items()}; blocks, SB, CB, pos = {}, [], [], 0
    for t, ix in groups.items():
        Xg = Xs[ix]; Cg = C[ix]
        for f in SHAPE:
            grid = np.quantile(Xg[:, fi[f]], np.linspace(0.02, 0.98, G)); S = np.repeat(Xg[None], G, axis=0).copy(); S[:, :, fi[f]] = grid[:, None]
            SB.append(S.reshape(-1, Xs.shape[1])); CB.append(np.tile(Cg, G)); blocks[(t, f)] = (pos, grid, len(Xg)); pos += len(SB[-1])
        cs = [(b, s) for s in range(3) for b in range(4)]; S = np.repeat(Xg[None], len(cs), axis=0).copy()
        for k, (b, s) in enumerate(cs):
            S[k, :, fi["balls"]] = b; S[k, :, fi["strikes"]] = s
        SB.append(S.reshape(-1, Xs.shape[1])); CB.append(np.tile(Cg, len(cs))); blocks[(t, "count")] = (pos, cs, len(Xg)); pos += len(SB[-1])
        SB.append(Xg); CB.append(Cg); blocks[(t, "asis")] = (pos, None, len(Xg)); pos += len(Xg)
    bigS = np.vstack(SB); bigC = np.concatenate(CB); bigN = bigS[:, :len(fp.FULL_COLS)]; print(f"PDP rows {len(bigS):,}", flush=True)
    res = Parallel(n_jobs=6, backend="loky")([delayed(_fit)(Xs, ys, bigS, g, s, C, bigC) for s in SEEDS] + [delayed(_fit)(Xn, yn, bigN, gn, s) for s in SEEDS])
    ps = np.mean(res[:3], axis=0); pn = np.mean(res[3:], axis=0); print(f"PDP fits done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ink, ink2, grid_c, surf = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"; Cn, Cs = "#eb6834", "#2a78d6"; curves = {}
    fig, axes = plt.subplots(len(TYPES), len(SHAPE), figsize=(15.5, 2.35 * len(TYPES)), facecolor=surf, sharey="row")
    for r, t in enumerate(TYPES):
        p0, _, n = blocks[(t, "asis")]; ms, mn = ps[p0:p0 + n].mean(), pn[p0:p0 + n].mean()
        for c, f in enumerate(SHAPE):
            q0, grid, n = blocks[(t, f)]
            cs_ = 100 * (ps[q0:q0 + G * n].reshape(G, n).mean(axis=1) / ms - 1); cn_ = 100 * (pn[q0:q0 + G * n].reshape(G, n).mean(axis=1) / mn - 1)
            curves[f"{t}|{f}"] = {"grid": grid.tolist(), "swstr": cs_.tolist(), "net": cn_.tolist()}
            ax = axes[r, c]; ax.set_facecolor(surf); ax.plot(grid, cn_, color=Cn, ls="--", lw=1.8, label="Network (per swing)"); ax.plot(grid, cs_, color=Cs, lw=1.8, label="swstr_ct (per pitch)")
            ax.axhline(0, color=grid_c, lw=0.8); ax.grid(axis="y", color=grid_c, lw=0.6); ax.tick_params(colors=ink2, labelsize=7.5)
            [ax.spines[k].set_visible(False) for k in ("top", "right")]; [ax.spines[k].set_color(grid_c) for k in ("left", "bottom")]
            if r == 0:
                ax.set_title(LAB[f], color=ink, fontsize=9.5, loc="left")
            if c == 0:
                ax.set_ylabel(f"{t}\n% vs avg", color=ink2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=ink)
    fig.suptitle("Partial dependence by pitch type: relative effect on each model's own average prediction (count and everything else as thrown).", color=ink, fontsize=11.5, x=0.01, ha="left", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig("output/swstr_vs_net_pdp.png", dpi=125, facecolor=surf); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=surf); ax.set_facecolor(surf); cnt = {}
    for t, col in zip(TYPES, ["#0e2f5a", "#2a78d6", "#eb6834", "#2d9a6b", "#8a5cc2", "#b8860b"]):
        q0, cs, n = blocks[(t, "count")]; v = ps[q0:q0 + len(cs) * n].reshape(len(cs), n).mean(axis=1); v = 100 * (v / v[0] - 1); cnt[t] = {f"{b}-{s}": float(x) for (b, s), x in zip(cs, v)}
        order = [i for i, (b, s) in enumerate(cs) if s == 2]; ax.plot([cs[i][0] for i in order], [v[i] for i in order], marker="o", color=col, lw=1.8, label=t)
    ax.set_xlabel("Balls (at 2 strikes)", color=ink2); ax.set_ylabel("Swinging-strike rate, % vs 0-0", color=ink2); ax.grid(axis="y", color=grid_c); ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.set_title("swstr_ct: effect of the count (2 strikes) on swinging strikes, by pitch type", color=ink, fontsize=10.5, loc="left"); [ax.spines[k].set_visible(False) for k in ("top", "right")]
    fig.tight_layout(); fig.savefig("output/swstr_vs_net_count.png", dpi=125, facecolor=surf); plt.close(fig)
    # ---- (3) + (4) pitch-type table and named pitchers, from the pooled out-of-fold estimates
    sv = {est: fp.compute_pooled_wh_sva(win, estimator=est).rename(columns={"sva": f"sva_{est}"}) for est in ("torch3", "swstr_ct")}; K = ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]
    W = win.merge(sv["swstr_ct"], on=K, how="inner").merge(sv["torch3"], on=K, how="left"); W["pitch_type"] = W["pitch_type"].fillna("unk").replace("", "unk"); W["swing"] = W["is_swing"].astype(int)
    gm = W.drop_duplicates(["game_id", "pitcher_id"]).sort_values("date"); gm["half"] = gm.groupby("pitcher_id").cumcount() % 2; W = W.merge(gm[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"])
    cell = W.groupby(["pitcher_id", "pitch_type"]).agg(n=("target", "size"), swings=("swing", "sum"), swstr=("target", "mean"), s_sw=("sva_swstr_ct", "mean"), s_net=("sva_torch3", "mean"), pitcher=("pitcher", "first")).reset_index()
    wh = W[W["swing"] == 1].groupby(["pitcher_id", "pitch_type"])["is_whiff"].mean().rename("whiff").reset_index(); cell = cell.merge(wh, on=["pitcher_id", "pitch_type"], how="left")
    half = W.groupby(["pitcher_id", "pitch_type", "half"]).agg(n=("target", "size"), swings=("swing", "sum"), s_sw=("sva_swstr_ct", "mean"), s_net=("sva_torch3", "mean")).reset_index(); rows = []
    for t in TYPES:
        c = cell[(cell["pitch_type"] == t) & (cell["n"] >= 100)].dropna(subset=["s_sw", "s_net"]).copy(); h = half[(half["pitch_type"] == t) & (half["n"] >= 60) & (half["swings"] >= 25)]
        w = h.pivot(index="pitcher_id", columns="half", values=["s_sw", "s_net"]).dropna()
        r = {"pitch type": t, "pitches": int((W["pitch_type"] == t).sum()), "cells (>=100 pitches)": len(c), "corr of cell levels": float(np.corrcoef(c["s_sw"], c["s_net"])[0, 1]) if len(c) > 5 else np.nan,
             "SD swstr_ct sva": float(c["s_sw"].std()), "SD net sva": float(c["s_net"].std()), "cells split-half": len(w)}
        for k in ("s_sw", "s_net"):
            rr = float(np.corrcoef(w[(k, 0)], w[(k, 1)])[0, 1]) if len(w) > 5 else np.nan; r[f"split-half r {k}"] = rr; r[f"S-B {k}"] = 2 * rr / (1 + rr) if rr == rr else np.nan
        rows.append(r)
    T = pd.DataFrame(rows); pd.set_option("display.width", 260); print("\nPITCH TYPE: agreement and reliability of the pitcher x pitch-type levels (out of fold, pooled 2025-26)\n" + T.round(3).to_string(index=False), flush=True)
    cell = cell[cell["pitch_type"].isin(TYPES) & (cell["n"] >= 100)].dropna(subset=["s_sw", "s_net"]).copy()
    for k in ("s_sw", "s_net"):
        cell["z_" + k[2:]] = cell.groupby("pitch_type")[k].transform(lambda s: (s - s.mean()) / s.std())
    cell["gap"] = cell["z_sw"] - cell["z_net"]; cell.to_csv("output/swstr_vs_net_cells.csv", index=False)
    nm = cell[cell["pitcher"].isin(NAMED)].sort_values(["pitcher", "n"], ascending=[True, False])
    print("\nNAMED PITCHERS (cells with >= 100 pitches; z within pitch type across pitchers; gap = swstr_ct z - network z)\n" + nm[["pitcher", "pitch_type", "n", "swings", "swstr", "whiff", "z_sw", "z_net", "gap"]].round(3).to_string(index=False))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), facecolor=surf); lab_rows = []
    for ax, t in zip(axes.ravel(), TYPES):
        c = cell[cell["pitch_type"] == t]; ax.set_facecolor(surf); ax.scatter(c["z_net"], c["z_sw"], s=np.clip(c["n"] / 12, 8, 70), color=Cs, alpha=0.55, lw=0); lim = max(c[["z_net", "z_sw"]].abs().max()) + 0.3; ax.plot([-lim, lim], [-lim, lim], color=grid_c, lw=1)
        for _, r in pd.concat([c.nlargest(3, "gap"), c.nsmallest(3, "gap")]).iterrows():
            ax.annotate(r["pitcher"].split(",")[0], (r["z_net"], r["z_sw"]), fontsize=7.5, color=ink2, xytext=(3, 3), textcoords="offset points")
            lab_rows.append({"pitch type": t, "pitcher": r["pitcher"], "n": int(r["n"]), "z_net": float(r["z_net"]), "z_swstr": float(r["z_sw"])})
        ax.set_title(f"{t} (r = {c['z_net'].corr(c['z_sw']):.2f}, {len(c)} pitchers)", color=ink, fontsize=10, loc="left"); ax.set_xlabel("Network z", color=ink2, fontsize=9); ax.set_ylabel("swstr_ct z", color=ink2, fontsize=9)
        ax.tick_params(colors=ink2, labelsize=8); ax.grid(color=grid_c, lw=0.5); [ax.spines[k].set_visible(False) for k in ("top", "right")]
    fig.suptitle("Pitcher x pitch-type stuff: swstr_ct vs network (z-scores within pitch type; cells with >= 100 pitches; dot size = pitches)", color=ink, fontsize=11.5, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig("output/swstr_vs_net_scatter.png", dpi=125, facecolor=surf); plt.close(fig)
    json.dump({"type_table": rows, "pdp_curves": curves, "count_effect": cnt, "labelled": lab_rows}, open("output/swstr_vs_net_deep.json", "w"), indent=1, default=float); print(f"done ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
