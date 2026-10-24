"""
fl_wh_sva_torch3_k_test.py -- STANDARD BEFORE / AFTER REPORT for putting the neural network into the pitcher projection (user 2026-10-23: "Let's add the neural network into the projection model").
The fl_wh_sva_variants_test.py protocol unchanged (frozen K% model REFIT with each variant's wh_sva_level on the training transition full 2025 -> actual 1H2026; fold 2 = features 2025 + 1H2026, targets 2H2026; position players
excluded; level-free; standard inverse-variance weights from the production variant's residuals, frozen; weighted RMSE AND MAE; pitcher bootstrap; plus the BB% add-on and the reliability of the pitcher level).
Variants: 'whiff' = production XGBoost (estimator='xgb')   |   'torch3' = the network exactly as production now runs it (fp.compute_pooled_wh_sva(estimator='torch3'): separate worker process, 3 seeds).
STOP RULE (pre-stated): if the network is worse than XGBoost on BOTH RMSE and MAE with P(better) <= .10 on each, production is NOT switched (flag stays 'xgb') and the user is asked; otherwise the switch stands (the user asked for it).
"""
import fl_pitcher_features as fp
import fl_wh_sva_variants_test as vt


def dispatch(df, min_rows=2000):
    return vt._orig(df, min_rows, estimator="xgb" if vt.CUR["target"] == "whiff" else "torch3")


if __name__ == "__main__":
    vt.VARIANTS = [("whiff", False), ("torch3", False)]
    vt.sva_variant = dispatch
    vt.main()
