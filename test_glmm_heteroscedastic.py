"""
test_glmm_heteroscedastic.py -- stress-tests the homogeneous-player-variance
assumption: simulate conferences with GENUINELY DIFFERENT player-talent spread
(e.g. a power conference with a tight band of good players vs. a weaker one
with much more top-to-bottom variance), fit the standard (mis-specified,
single tau_p^2 for everyone) model, and see how badly the shrinkage gets
distorted for the conferences whose true spread is furthest from what the
single pooled estimate assumes.

Usage:
    python test_glmm_heteroscedastic.py
"""
import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

N_CONF = 30
PLAYERS_PER_CONF = 40


def simulate(true_player_sds, seed=2):
    """true_player_sds: one true within-conference player SD per conference --
    the thing the standard model wrongly assumes is constant across all of them."""
    rng = np.random.default_rng(seed)
    conf_true_effect = rng.normal(0, 0.15, len(true_player_sds))
    rows, truth = [], []
    for c, sd in enumerate(true_player_sds):
        player_true = rng.normal(0, sd, PLAYERS_PER_CONF)
        n_pa = rng.integers(20, 300, PLAYERS_PER_CONF)
        logit_p = -1.2 + conf_true_effect[c] + player_true
        p = 1 / (1 + np.exp(-logit_p))
        k = rng.binomial(n_pa, p)
        for i in range(PLAYERS_PER_CONF):
            outcomes = np.zeros(n_pa[i]); outcomes[:k[i]] = 1
            for o in outcomes:
                rows.append({"conf": f"c{c}", "player": f"c{c}_p{i}", "y": o})
            truth.append({"conf": f"c{c}", "player": f"c{c}_p{i}", "true_effect": player_true[i],
                           "true_conf_sd": sd, "n_pa": n_pa[i]})
    return pd.DataFrame(rows), pd.DataFrame(truth)


if __name__ == "__main__":
    # half the conferences have TIGHT player spread (sd=0.10, e.g. a deep power
    # conference where even the "weak" players are pretty good), half have WIDE
    # spread (sd=0.40, a conference with a few stars and a long weak tail)
    true_sds = [0.10] * (N_CONF // 2) + [0.40] * (N_CONF // 2)
    edf, truth = simulate(true_sds)
    print(f"n trial-rows: {len(edf):,}")

    model = BinomialBayesMixedGLM.from_formula("y ~ 1", {"conf": "0 + C(conf)", "player": "0 + C(player)"}, edf)
    result = model.fit_vb()
    print(result.summary())

    # vcp_mean holds [conf_log_sd, player_log_sd], in vc_formulas insertion order
    # (verified directly against the labeled summary table before trusting this
    # indexing -- not assumed)
    pooled_player_sd = np.exp(result.vcp_mean[1])
    pooled_player_var = pooled_player_sd ** 2
    print(f"\npooled (mis-specified) player SD estimate: {pooled_player_sd:.4f}  "
          f"(true SDs were 0.10 for half the conferences, 0.40 for the other half)")

    for true_sd in [0.10, 0.40]:
        true_var = true_sd ** 2
        # implied shrinkage for a typical n=100 player under the WRONG pooled estimate
        # vs. under the CORRECT conference-specific true variance
        sigma_i2 = 0.25 * (1 - 0.25) / 100  # approx binomial sampling variance at p~0.25, n=100
        shrink_pooled = pooled_player_var / (pooled_player_var + sigma_i2)
        shrink_true = true_var / (true_var + sigma_i2)
        print(f"true conf player SD={true_sd}: correct shrinkage={shrink_true:.3f}, "
              f"shrinkage using the WRONG pooled estimate={shrink_pooled:.3f}  "
              f"(error={shrink_pooled-shrink_true:+.3f})")
