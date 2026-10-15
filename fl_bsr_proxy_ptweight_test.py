"""
fl_bsr_proxy_ptweight_test.py -- does weighting the baserunning proxy by POSITION PLAYING TIME beat the single primary position? (user 2026-10-14)

Current proxy (fl_bsr_proxy, production): one position per hitter (DH if >= 50% of PA at DH, else war_players.primary_pos) -> that position's average.
Candidate: a playing-time MIX.  share_DH = dh_pa / pa; the rest split by his 2026 innings at each fielding position (war_playing_time, pitcher innings dropped):
        share_pos = (1 - share_DH) x innings_pos / innings_all.  Hitters with no innings and no DH PA -> the primary-position row (or unknown).
        bsr / PA = sum_pos share_pos x effect_pos  (WLS, weights PA, no intercept since shares sum to 1).
Pre-stated rule: adopt the mix only if (a) held-out bsr RMSE improves vs the primary-position proxy with P(better) >= .8 AND (b) the WAR backtest is not worse.
(a) 10-fold CV by player x 20 repeats, frozen inverse-variance weights from the league-mean residuals (same as fl_bsr_proxy_validation.part_a).
(b) WAR backtest -> 2026, cross-fitted, same rows / target / weights as fl_bsr_proxy_validation.part_b; shares are 2026 roles (same caveat as the positions there).
"""
import warnings

import numpy as np
import pandas as pd

import fl_bsr_proxy as bp
from fl_eval_report import compare, table
from fl_eval_weights import fit_variance_law, inv_var_weights

warnings.filterwarnings("ignore")
POS = bp.POS


def shares(year=2026):
    from sqlalchemy import text
    from fl_supabase import read_sql_df
    pt = read_sql_df(text(f"select player, position, innings from war_playing_time where game_type = 'reg_season' and year = {int(year)} and position <> 'P'"))
    pt["name_clean"] = pt["player"].map(bp._clean_lastfirst)
    inn = pt.pivot_table(index="name_clean", columns="position", values="innings", aggfunc="sum").reindex(columns=POS[:-1]).fillna(0.0)
    pos = bp.positions(year); dh = pos["dh_share"].clip(0, 1).fillna(0.0)
    S = pd.DataFrame(0.0, index=pos.index, columns=POS)
    tot = inn.reindex(pos.index).fillna(0.0).sum(1); fs = inn.reindex(pos.index).fillna(0.0).div(tot.replace(0, np.nan), axis=0).fillna(0.0)
    S[POS[:-1]] = fs.mul(1 - dh, axis=0); S["DH"] = dh
    none = (tot == 0) & (dh == 0)                                     # no innings, no DH PA: fall back to the one-hot primary position
    for nm in S.index[none]:
        p = pos.loc[nm, "pos"]
        if p in POS:
            S.loc[nm] = 0.0; S.loc[nm, p] = 1.0
    S.loc[(tot == 0) & (dh > 0), "DH"] = 1.0                          # DH PA but no innings: all DH
    return S


def fit_mix(S, y, w):
    sw = np.sqrt(w); return np.linalg.lstsq(S * sw[:, None], y * sw, rcond=None)[0]


def onehot(pos):
    return np.column_stack([(pd.Series(pos).reset_index(drop=True) == p).astype(float).to_numpy() for p in POS])


def oof(X, y, w, folds=10, reps=20, seed=0):
    rng = np.random.default_rng(seed); out = np.zeros(len(y))
    for _ in range(reps):
        f = rng.permutation(np.arange(len(y)) % folds)
        for k in range(folds):
            tr, te = f != k, f == k; b = fit_mix(X[tr], y[tr], w[tr]); out[te] += X[te] @ b
    return out / reps


def main():
    d = bp.training_data(); S = shares(2026).reindex(d.index).fillna(0.0)
    y = (d["bsr"] / d["pa"]).to_numpy(); n = d["pa"].to_numpy(float)
    X1 = onehot(d["pos"]); Xm = S[POS].to_numpy()
    mixed = ((Xm > 0).sum(1) > 1); print(f"{len(d)} hitters; {mixed.sum()} ({np.average(mixed, weights=n):.0%} of PA) split time across 2+ positions incl. DH; "
                                       f"main-position share median {np.median(Xm.max(1)):.2f}")
    print("effects per 600 PA  primary:", dict(zip(POS, np.round(600 * fit_mix(X1, y, n), 2))), "\n                   mix:    ", dict(zip(POS, np.round(600 * fit_mix(Xm, y, n), 2))))
    rng = np.random.default_rng(0); f = rng.permutation(np.arange(len(d)) % 10); lg = np.zeros(len(d))
    for k in range(10):
        lg[f == k] = np.average(y[f != k], weights=n[f != k])
    a, b = fit_variance_law(y - lg, n); w = inv_var_weights(n, a, b)
    p1, pm = oof(X1, y, n), oof(Xm, y, n); ids = d.index.to_numpy()
    print("\n(a) held-out 2026 baserunning, runs per 600 PA")
    print(table([{"comparison": "league mean -> primary position", **compare(600 * y, 600 * lg, 600 * p1, w, ids, n=2000)},
                 {"comparison": "primary position -> playing-time mix", **compare(600 * y, 600 * p1, 600 * pm, w, ids, n=2000)}], 4))
    # (b) WAR backtest
    import fl_hitter_baseline_backtest as bt
    import fl_hitter_model_dev as dev
    import fl_hitter_pre2026 as pre
    import fl_hitter_projection as hp
    dproj, c, fit = pre.build_shifted()
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; lg26 = float(bt.league(q, "woba")[2026])
    act = p[p["year"] == 2026].drop_duplicates("name_clean").set_index("name_clean")
    j = dproj.set_index("name_clean").join(act[["pa", "woba_n", "pos_adj_runs"]].rename(columns={"pa": "pa26", "woba_n": "woba26", "pos_adj_runs": "pos26"}), how="inner")
    j = j[(j["pa26"] >= 20) & j["proj_woba"].notna()].copy(); j["bsr26"] = bp.positions(2026)["bsr"].reindex(j.index).fillna(0.0)
    j["bsr_pos"] = bp.positions(2026)["pos"].reindex(j.index).fillna(j["primary_pos"])
    Sj = shares(2026).reindex(j.index); miss = Sj.isna().all(1) | (Sj.fillna(0).sum(1) == 0)
    Xj1 = onehot(j["bsr_pos"]); Xjm = np.where(miss.to_numpy()[:, None], Xj1, Sj.fillna(0.0)[POS].to_numpy())
    names = sorted(set(d.index) | set(j.index)); fold = pd.Series(np.random.default_rng(1).permutation(np.arange(len(names)) % 10), index=names)
    fd, fj = fold.reindex(d.index).to_numpy(), fold.reindex(j.index).to_numpy(); pr1, prm = np.zeros(len(j)), np.zeros(len(j))
    for k in range(10):
        b1 = fit_mix(X1[fd != k], y[fd != k], n[fd != k]); bm = fit_mix(Xm[fd != k], y[fd != k], n[fd != k])
        unk1 = float(np.average(b1, weights=X1[fd != k].T @ n[fd != k] + 1e-9)); unkm = float(np.average(bm, weights=Xm[fd != k].T @ n[fd != k] + 1e-9))
        te = fj == k; r1 = Xj1[te] @ b1; rm = Xjm[te] @ bm; none = Xj1[te].sum(1) == 0
        r1[none] = unk1; rm[none & (Xjm[te].sum(1) == 0)] = unkm; pr1[te], prm[te] = r1, rm
    base = (j["proj_woba"] - c["lg"]) / hp.SC + j["pos_rate"] + c["repl"]
    y_w = (j["pa26"] * ((j["woba26"] - lg26) / hp.SC + j["pos26"].fillna(0) / j["pa26"] + c["repl"]) / c["rpw"] + j["bsr26"] / c["rpw"]).to_numpy()
    w0 = j["proj_pa"] * base / c["rpw"]; w1 = j["proj_pa"] * (base + pr1) / c["rpw"]; wm = j["proj_pa"] * (base + prm) / c["rpw"]
    print(f"\n(b) WAR backtest -> 2026, {len(j)} hitters")
    print(table([{"comparison": "no baserunning -> primary position", **compare(y_w, w0.to_numpy(), w1.to_numpy(), j["pa26"].to_numpy(), j.index.to_numpy(), n=2000)},
                 {"comparison": "primary position -> playing-time mix", **compare(y_w, w1.to_numpy(), wm.to_numpy(), j["pa26"].to_numpy(), j.index.to_numpy(), n=2000)}], 4))


if __name__ == "__main__":
    main()
