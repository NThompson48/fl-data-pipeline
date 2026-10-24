"""
fl_wh_sva_prior_nn_robust_test.py -- the same four-pairing robustness check as fl_wh_sva_prior_xgb_robust_test.py, with the NETWORK stuff (fp.compute_pooled_wh_sva(estimator='torch3'), 3 seeds, separate worker),
so XGBoost-prior and network-prior can be compared pairing by pairing.  Same grid, same transitions, same standard weights / bootstrap.  Output: output/wh_sva_prior_nn_robust_test.json
"""
import json
import warnings

import pandas as pd

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_prior_xgb_robust_test as rb

warnings.filterwarnings("ignore")


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); fp.WH_SVA_ESTIMATOR = "torch3"; rb._cache.clear(); rows = []; res = {}
    for a, b in rb.PAIRS:
        F = rb.fit(raw, a); S = rb.score(raw, F, b); res[f"{a}->{b}"] = S
        print(f"== fit {a} -> test {b}: stuff coef {F['coef']:.3f} (t {F['t']:.2f}); prior c mult {F['cm']}, n0_s {F['n0_s']:.0f}", flush=True)
        rows.append({"pairing": f"{a}->{b}", "n test": S["n_test"], "c mult": F["cm"], "n0_s": F["n0_s"], "add RMSE": S["additive"][0], "prior RMSE": S["prior"][0], "nostuff RMSE": S["nostuff"][0], "prior dRMSE": S["prior_vs_additive"]["dRMSE"], "P(RMSE)": S["prior_vs_additive"]["P(RMSE better)"],
                     "add MAE": S["additive"][1], "prior MAE": S["prior"][1], "nostuff MAE": S["nostuff"][1], "prior dMAE": S["prior_vs_additive"]["dMAE"], "P(MAE)": S["prior_vs_additive"]["P(MAE better)"]})
    pd.set_option("display.width", 260); print("\nSUMMARY (NETWORK stuff; negative d = fading prior better than the constant-weight term)\n" + pd.DataFrame(rows).round(5).to_string(index=False))
    json.dump({"pairs": res, "summary": rows}, open("output/wh_sva_prior_nn_robust_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
