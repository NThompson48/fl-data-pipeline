"""
fl_wh_sva_torch.py -- PyTorch networks for the wh_sva stuff model: P(whiff | swing, inputs).  Replaces scikit-learn's MLPClassifier, whose early stopping has two problems for this task:
  (1) it monitors validation ACCURACY, not log-loss -- with a 23% whiff rate accuracy barely moves, and we only use the probabilities;
  (2) its validation rows are a random 10% of the training ROWS, so the same pitchers appear in training and validation.  A pitcher's release point / extension / spin act as a fingerprint, so a net can
      lower validation loss by memorising pitcher-specific whiff rates -- which tells it to train longer than is good for predicting NEW pitchers (the thing the cross-fit measures).
      Here the validation set is a random 10% of the training PITCHERS and the stopping criterion is validation log-loss.
Everything is seeded; training uses CPU threads set by the caller (torch.set_num_threads).  Config keys (all optional):
  hidden  tuple of widths, default (128, 64, 32)        act     'relu' | 'silu' | 'gelu'           dropout  p after each hidden layer
  res     residual blocks (width = hidden[0], depth = len(hidden)) with LayerNorm instead of a plain stack
  lr, wd  AdamW learning rate / weight decay             batch   default 1024                       patience / max_epochs  default 8 / 100
  sched   'plateau' = ReduceLROnPlateau on the validation log-loss (sched_patience epochs, sched_factor), so a long run keeps refining at smaller steps; default None
  noise   SD of Gaussian noise added to the standardised inputs in training only (default 0)
  clip    inputs standardised on the training fold and clipped to +/- clip SD (default 6; only .14% of swings have an input beyond 5 SD)
  n_cat   if > 0, the last column of X is an integer category (e.g. pitch type) fed through an embedding of size emb
"""
import numpy as np
import torch
import torch.nn as nn

DEFAULT = dict(hidden=(128, 64, 32), act="relu", dropout=0.0, res=False, lr=1e-3, wd=1e-4, batch=1024, patience=8, max_epochs=100, clip=6.0, noise=0.0, sched=None, sched_patience=4, sched_factor=0.5, n_cat=0, emb=4)
ACTS = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}


class Net(nn.Module):
    def __init__(self, d_in, cfg):
        super().__init__()
        self.n_cat = cfg["n_cat"]; self.emb = nn.Embedding(cfg["n_cat"], cfg["emb"]) if cfg["n_cat"] else None
        d = d_in + (cfg["emb"] if cfg["n_cat"] else 0); act = ACTS[cfg["act"]]; h = cfg["hidden"]
        if cfg["res"]:
            self.inp = nn.Linear(d, h[0]); self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(h[0]), nn.Linear(h[0], h[0]), act(), nn.Dropout(cfg["dropout"]), nn.Linear(h[0], h[0])) for _ in h])
            self.out = nn.Sequential(nn.LayerNorm(h[0]), act(), nn.Linear(h[0], 1)); self.plain = None
        else:
            layers, prev = [], d
            for w in h:
                layers += [nn.Linear(prev, w), act()] + ([nn.Dropout(cfg["dropout"])] if cfg["dropout"] else []); prev = w
            self.plain = nn.Sequential(*layers, nn.Linear(prev, 1))

    def forward(self, x, cat=None):
        if self.emb is not None:
            x = torch.cat([x, self.emb(cat)], dim=1)
        if self.plain is not None:
            return self.plain(x).squeeze(-1)
        z = self.inp(x)
        for b in self.blocks:
            z = z + b(z)
        return self.out(z).squeeze(-1)


def _mean_loss(net, X, y, c, lossf, chunk=50000):
    tot = 0.0
    for i in range(0, len(y), chunk):
        tot += float(lossf(net(X[i:i + chunk], c[i:i + chunk] if c is not None else None), y[i:i + chunk])) * len(y[i:i + chunk])
    return tot / len(y)


def fit_predict(Xtr, ytr, Xte, groups_tr, seed, cfg=None, cat_tr=None, cat_te=None, val_mode="group", history=None, track=None):
    """train on (Xtr, ytr); return P(whiff) on Xte plus (epochs run, best validation log-loss).
    val_mode 'group' (default): validation = a random 10% of the training PITCHERS.  'rows': validation = a random 10% of training ROWS (scikit-learn's design; the same pitchers sit in training and validation).
    history (a dict, optional): filled with per-epoch 'train' log-loss (all training rows, eval mode), 'val' log-loss, and one list per entry of `track` = {name: (X, y)} (extra sets scored each epoch, never used for stopping)."""
    cfg = {**DEFAULT, **(cfg or {})}; torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; prep = lambda X: torch.tensor(np.clip((X - mu) / sd, -cfg["clip"], cfg["clip"]), dtype=torch.float32)
    if val_mode == "rows":
        is_val = rng.random(len(ytr)) < 0.1
    else:
        ug = np.unique(groups_tr); val_g = set(rng.choice(ug, size=max(2, int(0.1 * len(ug))), replace=False)); is_val = np.array([g in val_g for g in groups_tr])
    Xt, Xv, Xe = prep(Xtr[~is_val]), prep(Xtr[is_val]), prep(Xte); yt, yv = torch.tensor(ytr[~is_val], dtype=torch.float32), torch.tensor(ytr[is_val], dtype=torch.float32)
    ct = cv = ce = None
    if cfg["n_cat"]:
        ct, cv, ce = (torch.tensor(c, dtype=torch.long) for c in (cat_tr[~is_val], cat_tr[is_val], cat_te))
    net = Net(Xt.shape[1], cfg); opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]); lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad, ep = np.inf, None, 0, 0
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=cfg["sched_factor"], patience=cfg["sched_patience"], min_lr=1e-6) if cfg["sched"] == "plateau" else None
    extra = {k: (prep(X), torch.tensor(yy, dtype=torch.float32)) for k, (X, yy) in (track or {}).items()}
    if history is not None:
        history.update({"train": [], "val": [], **{k: [] for k in extra}})
    for ep in range(1, cfg["max_epochs"] + 1):
        net.train(); perm = torch.randperm(len(yt), generator=gen)
        for i in range(0, len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]; xb = Xt[b]
            if cfg["noise"]:
                xb = xb + cfg["noise"] * torch.randn(xb.shape, generator=gen)          # Gaussian noise on the standardised inputs while training (a data-augmentation regulariser)
            opt.zero_grad(); lossf(net(xb, ct[b] if ct is not None else None), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(Xv, cv), yv))
            if history is not None:
                history["train"].append(_mean_loss(net, Xt, yt, ct, lossf))
                history["val"].append(vl)
                for k, (Xk, yk) in extra.items():
                    history[k].append(float(lossf(net(Xk), yk)))
        if sched is not None:
            sched.step(vl)                                                      # halve the learning rate when validation loss has not improved for sched_patience epochs
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():                                                           # scored in chunks so millions of rows (partial-dependence grids) fit in memory
        p = np.concatenate([torch.sigmoid(net(Xe[i:i + 200000], ce[i:i + 200000] if ce is not None else None)).numpy() for i in range(0, len(Xe), 200000)]).astype(float)
    return p, ep, best
