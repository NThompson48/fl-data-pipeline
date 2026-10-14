"""
fl_hitter_babip_ablation.py -- is the BABIP component helping the wOBA projection?
Ablation of the component half of the blend: components {K%, BB%, HR/PA, BABIP} vs the same without BABIP (weights refit on the training transitions only), and for context without each other component.
Scored as (a) the component-built wOBA alone and (b) the production 50/50 blend with the direct regression.  Out of sample: leave-one-transition-out (4 folds) and forward (tests 2024->25, 2025->26); n0 per stat refit on
the training transitions (three-season regression); next-year park-neutral wOBA, >= 100 PA, standard weights; hitter-cluster bootstrap; weighted RMSE AND MAE (reporting standard); old = with BABIP, new = without.
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_blend_weight_test as bw
import fl_hitter_depth3_test as d3
import fl_hitter_model_dev as dev
import fl_hitter_projection as hp

warnings.filterwarnings("ignore")
FULL = ("k", "bbr", "hrr", "babip")


def folds(frames, comp, plan):
    bw.COMP = comp; out = []
    for tr, te in plan:
        o, b = bw.fold_preds({s: frames[s] for s in ("woba",) + tuple(comp)}, tr, te); out.append(o.assign(fold=te, beta=str(np.round(b, 2).tolist())))
    return pd.concat(out)


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; frames = {s: d3.frame(q, s) for s in ("woba",) + FULL}
    for scheme, plan in (("LEAVE-ONE-TRANSITION-OUT", [(tuple(t for t in (2022, 2023, 2024, 2025) if t != T), T) for T in (2022, 2023, 2024, 2025)]),
                         ("FORWARD (tests 2024->25 and 2025->26)", [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)])):
        full = folds(frames, FULL, plan); keep = full.index; rows = []
        y = full["x_next"].to_numpy(); w = full["w"].to_numpy(); ids = full.index.get_level_values("name_clean").to_numpy()
        print(f"\n{'=' * 110}\n{scheme}: n = {len(full)}; component weights (K, BB, HR, BABIP) by fold: " + " | ".join(full.groupby('fold')['beta'].first()) + f"\n{'=' * 110}")
        for drop in ("babip", "hrr", "bbr", "k"):
            comp = tuple(c for c in FULL if c != drop); ab = folds(frames, comp, plan).reindex(keep)
            for lab, a_, b_ in (("components alone", full["comp"], ab["comp"]), ("50/50 blend", 0.5 * full["direct"] + 0.5 * full["comp"], 0.5 * ab["direct"] + 0.5 * ab["comp"])):
                rows.append({"drop": drop, "wOBA projection": lab, **er.compare(y, a_.to_numpy(), b_.to_numpy(), w, ids, n=500)})
        t = pd.DataFrame(rows)[["drop", "wOBA projection", "RMSE old", "RMSE new", "dRMSE", "RMSE 90%", "P(RMSE better)", "MAE old", "MAE new", "dMAE", "MAE 90%", "P(MAE better)"]]
        print("  old = all four components, new = the named component removed (refit weights)\n" + er.table(t.to_dict("records"), 5))
    bw.COMP = FULL


if __name__ == "__main__":
    main()
