"""
fl_wh_sva_curves.py -- train / validation / held-out loss over epochs for the wh_sva network (user 2026-10-21: "show me the train/val error chart over epochs").
One fold of the pilot's 5-fold GroupKFold (by pitcher) over all 2025-26 swings (position players excluded): ~180,000 training swings, ~45,000 swings from pitchers the network never sees (the 'outer test').
The SAME network (9 inputs -> 128 -> 64 -> 32 -> 1, ReLU, AdamW lr 1e-3, wd 1e-4, batch 1024, seed 1) is trained for 60 epochs under two validation designs, with no early stopping so the whole curve shows:
  A  validation = 10% of the training PITCHERS (new pitchers, what the cross-fit measures)         [fl_wh_sva_torch default]
  B  validation = 10% of the training ROWS (the same pitchers also in training; scikit-learn's design)  [log-loss here; scikit-learn itself watches accuracy]
The outer-test curve is only recorded, never used to stop or select.  Stopping epoch = first epoch at which patience 8 on that design's own validation loss runs out.
Output: output/wh_sva_nn_curves.png, output/wh_sva_nn_curves.json
"""
import json
import warnings

import matplotlib
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
EPOCHS, PATIENCE = 60, 8
TEST = "held-out pitchers (never trained on)"


def run(mode, tr, te, X, y, g):
    torch.set_num_threads(3); h = {}
    ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=1, cfg={"patience": 10 ** 9, "max_epochs": EPOCHS}, val_mode=mode, history=h, track={TEST: (X[te], y[te])})
    return mode, h


def stop_epoch(val, patience=PATIENCE):
    best, bad = np.inf, 0
    for e, v in enumerate(val, 1):
        if v < best - 1e-5:
            best, bad = v, 0
        else:
            bad += 1
            if bad >= patience:
                return e
    return len(val)


def summarise(h):
    val, test = np.array(h["val"]), np.array(h[TEST]); best_val = int(val.argmin()) + 1; stop = stop_epoch(val); best_test = int(test.argmin()) + 1
    chosen = best_val                                                   # the weights the trainer keeps = best validation epoch
    return {"best validation epoch": best_val, "stops at epoch (patience 8)": stop, "held-out loss at the kept epoch": float(test[chosen - 1]), "best held-out epoch": best_test, "best held-out loss": float(test.min()),
            "cost of the stopping rule (kept - best)": float(test[chosen - 1] - test.min()), "train loss at the kept epoch": h["train"][chosen - 1], "validation loss at the kept epoch": float(val[chosen - 1])}


def draw(res, path):
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    col = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}; ls = {"train": "-", "val": "--", "test": "-."}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True, facecolor=surface)
    lo = min(min(h["train"] + h["val"] + h[TEST]) for h in res.values()) - 0.004; hi = max(h["val"][0] for h in res.values()) + 0.004
    titles = {"group": "A. Validation = new pitchers (what the cross-fit measures)", "rows": "B. Validation = random rows (same pitchers in training)"}
    for ax, mode in zip(axes, ("group", "rows")):
        h = res[mode]; ep = np.arange(1, EPOCHS + 1); s = summarise(h); ax.set_facecolor(surface)
        for key, name, series in (("train", "training rows", h["train"]), ("val", "validation set", h["val"]), ("test", TEST, h[TEST])):
            ax.plot(ep, series, color=col[key], lw=2.0, ls=ls[key], label=name)
        ax.axvline(s["stops at epoch (patience 8)"], color=ink2, lw=1.0, ls=":"); ax.text(s["stops at epoch (patience 8)"] + 0.6, hi - 0.002, f"training stops here\n(epoch {s['stops at epoch (patience 8)']}, patience 8)", color=ink2, fontsize=8.5, va="top")
        ax.plot([s["best held-out epoch"]], [s["best held-out loss"]], "o", color=col["test"], mfc=surface, mew=2, ms=8, zorder=5)
        ax.annotate(f"best for new pitchers:\nepoch {s['best held-out epoch']}, loss {s['best held-out loss']:.4f}", (s["best held-out epoch"], s["best held-out loss"]), (s["best held-out epoch"] + 6, s["best held-out loss"] - 0.0105), color=ink, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
        ax.set_title(titles[mode], color=ink, fontsize=11, loc="left", pad=10); ax.set_xlabel("epoch (one pass over the training swings)", color=ink2, fontsize=9.5); ax.set_ylim(lo, hi); ax.set_xlim(1, EPOCHS)
        ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True); [ax.spines[k].set_visible(False) for k in ("top", "right")]; [ax.spines[k].set_color(grid) for k in ("left", "bottom")]; ax.tick_params(colors=ink2, labelsize=9)
    axes[0].set_ylabel("log-loss (lower is better)", color=ink2, fontsize=9.5); axes[0].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=ink)
    fig.suptitle("wh_sva network: loss by epoch, one cross-fit fold (about 180,000 training swings, 45,000 swings from unseen pitchers)", color=ink, fontsize=12, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=160, facecolor=surface); plt.close(fig)


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); pop = raw[raw["is_swing"]].dropna(subset=fp.FULL_COLS + ["pitcher_id", "is_whiff", "date", "game_id", "pa_id", "pitch_no"]).reset_index(drop=True)
    X = pop[fp.FULL_COLS].astype(float).to_numpy(); y = pop["is_whiff"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); tr, te = next(GroupKFold(5).split(X, y, g))
    print(f"fold 1: {len(tr):,} training swings ({len(np.unique(g[tr]))} pitchers), {len(te):,} swings from {len(np.unique(g[te]))} unseen pitchers; base-rate log-loss on the test fold "
          f"{-(y[te].mean() * np.log(y[tr].mean()) + (1 - y[te].mean()) * np.log(1 - y[tr].mean())):.4f}", flush=True)
    res = dict(Parallel(n_jobs=2, backend="loky")(delayed(run)(m, tr, te, X, y, g) for m in ("group", "rows")))
    out = {m: {"history": h, "summary": summarise(h)} for m, h in res.items()}; json.dump(out, open("output/wh_sva_nn_curves.json", "w"), indent=1, default=float)
    for m, lab in (("group", "A (validate on new pitchers)"), ("rows", "B (validate on random rows)")):
        print(f"\n{lab}"); [print(f"   {k:44s} {v:.4f}" if isinstance(v, float) else f"   {k:44s} {v}") for k, v in out[m]["summary"].items()]
    draw(res, "output/wh_sva_nn_curves.png"); print("\nsaved output/wh_sva_nn_curves.png")


if __name__ == "__main__":
    main()
