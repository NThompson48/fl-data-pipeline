"""
fl_bridge_unified_test.py -- test a UNIFIED other-league input policy against the
shipped, league-specific one.

Unified policy (identical for College / Pioneer / MiLB):
  1. aggregate a pitcher's source seasons with TBF x decay^lag weights
     (one decay per METRIC: K%, BB%), then SHRINK toward that source league's
     mean by n0 batters faced,
  2. DAMP by staleness: deviation x rho^(gap-1), gap = projection year - source
     season (the shipped mapping ignores gap in the coefficient; it only uses
     0.5^gap to pick among leagues),
  3. one POOLED mapping to FL (no league slopes),
  4. where a pitcher has several leagues, take the source with the highest
     reliability  r = sumW/(sumW+n0) x rho^(gap-1)  (no league-R2 weighting).

Part 1  Debut pairs (pooled LOO R2, PA-weighted): grid over rho and n0 at the
        best decay per metric.
Part 2  Held-out Experiment 2 / Experiment 1 harness: shipped bridge features vs
        the unified features, K% as a free regressor and BB% through the
        informative-prior blend exactly as production (n0_bridge refit on the
        train transition).  Paired bootstrap on weighted squared error.

Usage:  python fl_bridge_unified_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_bridge_recency_league_study as st
import fl_ppa_kbb_and_contact as base
import fl_bridge_value_deepdive as dd

N_BOOT = 2000
POLICY = {"n0": 50.0, "dK": 0.25, "dBB": 1.0}
RHOS = (1.0, 0.8, 0.6, 0.5, 0.4)


def source_feature(hist, col, decay, n0, lg):
    return st.feature(hist, col, decay, n0, lg)


def pair_grid(pairs, lgr):
    """Pooled LOO R2 for (rho, n0) at the policy decays; single pooled line, optionally with league intercepts."""
    for metric, yname, col, dec in (("K", "k_rate", "k", POLICY["dK"]), ("BB", "bb_rate", "bb", POLICY["dBB"])):
        rows = []
        for rho in RHOS:
            r = {"rho": rho}
            for n0 in (0, 25, 50, 100):
                Y, W, U = [], [], []
                for lg, p in pairs.items():
                    f = np.array([source_feature(h, col, dec, n0, lgr[lg][col]) for h in p["hist"]]) - lgr[lg][col]
                    U.append(f * rho ** (p["gap"].values - 1)); Y.append(p[yname].values); W.append(p["pa"].values.astype(float))
                Y, W, U = np.concatenate(Y), np.concatenate(W), np.concatenate(U)
                r[f"n0={n0}"] = round(st.loo_r2(Y, U[:, None], W), 3)
            rows.append(r)
        print(f"\n  {metric}% (decay {dec}) pooled single line, LOO R2_w; rows = staleness damping rho (1.0 = none; 0.5 = halve per extra year)")
        print(pd.DataFrame(rows).set_index("rho").to_string())


def per_league_check(pairs, lgr, rho):
    print(f"\n  per-league LOO R2_w under the unified policy (rho={rho}, n0={POLICY['n0']:.0f}) vs shipped:")
    rows = []
    for lg, p in pairs.items():
        for metric, yname, col, dec in (("K", "k_rate", "k", POLICY["dK"]), ("BB", "bb_rate", "bb", POLICY["dBB"])):
            y, w = p[yname].values, p["pa"].values.astype(float)
            f = np.array([source_feature(h, col, dec, POLICY["n0"], lgr[lg][col]) for h in p["hist"]]) - lgr[lg][col]
            u = f * rho ** (p["gap"].values - 1)
            sd, sn = st.SHIPPED[(lg, metric)]
            fs = np.array([source_feature(h, col, sd, sn, lgr[lg][col]) for h in p["hist"]])
            rows.append({"league": lg, "metric": metric, "shipped inputs (own line)": round(st.loo_r2(y, fs, w), 3),
                         "unified inputs (own line)": round(st.loo_r2(y, u[:, None], w), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


# --------------------------------------------------------------------------- #
#  unified features for a set of pitchers at a given projection point
# --------------------------------------------------------------------------- #
class Unified:
    def __init__(self, srcs, lgr, policy=POLICY, rho=0.5):
        self.by = {lg: {n: g for n, g in s.groupby("name_clean")} for lg, s in srcs.items()}
        self.lgr, self.pol, self.rho = lgr, policy, rho

    def for_pitchers(self, ids, names, current_year):
        """u_k, u_bb (deviation from the SOURCE league mean, shrunk + damped), source, and reliability, chosen
        per metric as the highest-reliability league. Source seasons are limited to <= current_year."""
        out = []
        proj_year = current_year + 1
        for pid, nm in zip(ids, names):
            nc = st.cb._fl_name_clean(nm); r = {"pitcher_id": str(pid)}
            for metric, col, dec in (("k", "k", self.pol["dK"]), ("bb", "bb", self.pol["dBB"])):
                best = (0.0, 0.0, "none")
                for lg, by in self.by.items():
                    g = by.get(nc)
                    if g is None:
                        continue
                    g = g[g["season"] <= current_year]
                    if g.empty:
                        continue
                    if g["kid"].nunique() > 1:
                        lat = g.groupby("kid").agg(ls=("season", "max"), tb=("tbf", "sum")).sort_values(["ls", "tb"], ascending=False)
                        g = g[g["kid"] == lat.index[0]]
                    lag = g["season"].max() - g["season"]
                    w = (g["tbf"] * (self.pol_decay(dec, lag))).sum()
                    x = source_feature(g[["season", "tbf", "k", "bb"]].reset_index(drop=True), col, dec, self.pol["n0"], self.lgr[lg][col])
                    gap = proj_year - int(g["season"].max())
                    rho_m = self.rho[metric] if isinstance(self.rho, dict) else self.rho
                    damp = rho_m ** max(gap - 1, 0)
                    rel = w / (w + self.pol["n0"]) * damp
                    if rel > best[1]:
                        best = ((x - self.lgr[lg][col]) * damp, rel, lg)
                r[f"u_{metric}"], r[f"rel_{metric}"], r[f"src_{metric}"] = best
            out.append(r)
        return pd.DataFrame(out)

    @staticmethod
    def pol_decay(dec, lag):
        return np.where(lag == 0, 1.0, dec ** lag) if dec > 0 else (lag == 0).astype(float)


def boot(y, w, se_a, se_b, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        d.append((np.sum(w[i] * se_a[i]) - np.sum(w[i] * se_b[i])) / np.sum(w[i] * se_a[i]))
    return 100 * np.mean(d), 100 * np.percentile(d, 5), 100 * np.percentile(d, 95)


def harness(raw, name, trf, trl, tef, tel, uni_by_rho, names_map, lg_fl):
    df_tr, df_te, n0_k, n0_bb = base.prep(raw, trf, trl, tef, tel)
    cy_tr = max(int(e[:4]) for _, e in trf); cy_te = max(int(e[:4]) for _, e in tef)
    pool = lambda wins: pd.concat([fp.filter_period(raw, s, e) for s, e in wins], ignore_index=True)
    lg_tr, lg_te = (lambda t: (t["k_or_bb"] == "walk").mean())(pool(trf)[pool(trf)["is_terminal"]]), (lambda t: (t["k_or_bb"] == "walk").mean())(pool(tef)[pool(tef)["is_terminal"]])
    res_k, res_bb = {}, {}
    # --- shipped ---
    res_k["shipped"] = (df_tr, df_te, dd.K_FULL)
    res_bb["shipped"] = (df_tr.assign(), df_te.assign(), None)
    shipped_bb_tr, shipped_bb_te = df_tr["blended_BB_rate"], df_te["blended_BB_rate"]
    outs = {}
    for rho, uni in uni_by_rho.items():
        ids_tr, ids_te = df_tr["pitcher_id"].astype(str), df_te["pitcher_id"].astype(str)
        u_tr = uni.for_pitchers(ids_tr, [names_map[i] for i in ids_tr], cy_tr).set_index("pitcher_id")
        u_te = uni.for_pitchers(ids_te, [names_map[i] for i in ids_te], cy_te).set_index("pitcher_id")
        a, b = df_tr.copy(), df_te.copy()
        a["u_k"], b["u_k"] = u_tr.loc[ids_tr, "u_k"].values, u_te.loc[ids_te, "u_k"].values
        a["u_bb"], b["u_bb"] = u_tr.loc[ids_tr, "u_bb"].values, u_te.loc[ids_te, "u_bb"].values
        # BB blended prior: bridge rate = FL league mean + u (slope absorbed by the fitted pseudo-count)
        for df, lg_ in ((a, lg_tr), (b, lg_te)):
            df["br_bb"] = lg_ + df["u_bb"]
        d_bb = a.dropna(subset=["BB_rate_actual", "command_score"])
        w = mm.reliability_weight(d_bb["PA_test"], n0_bb).values
        n0b = mm.fit_n0_bridge(d_bb["BB_raw"], d_bb["PA"], n0_bb, lg_tr, d_bb["br_bb"], d_bb["BB_rate_actual"], w)
        a["blended_BB_rate"] = mm.blended_rate(a["BB_raw"], a["PA"], n0_bb, lg_tr, n0b, a["br_bb"])
        b["blended_BB_rate"] = mm.blended_rate(b["BB_raw"], b["PA"], n0_bb, lg_te, n0b, b["br_bb"])
        outs[rho] = (a, b, n0b)
    # evaluate K
    def evaluate(target, regs_by_variant, frames, n0, label):
        cols = sorted({c for v in regs_by_variant.values() for c in v[1]})
        keep_te = None
        res = {}
        for vn, (fr, regs) in regs_by_variant.items():
            tr_, te_ = frames[vn]
            tr_ = tr_.dropna(subset=[target] + regs); te_ = te_.dropna(subset=[target] + regs)
            m = bt.fit_weighted(tr_, target, regs, mm.reliability_weight(tr_["PA_test"], n0).values)
            res[vn] = pd.Series(m.predict(te_).values, index=te_["pitcher_id"].astype(str).values), te_.set_index(te_["pitcher_id"].astype(str))[[target, "PA_test"]]
        common = set.intersection(*[set(v[0].index) for v in res.values()])
        idx = sorted(common)
        y = res["shipped"][1].loc[idx, target].values; w = mm.reliability_weight(res["shipped"][1].loc[idx, "PA_test"], n0).values
        se0 = (y - res["shipped"][0].loc[idx].values) ** 2
        rows = []
        for vn, (p, _) in res.items():
            se = (y - p.loc[idx].values) ** 2
            rows.append({"variant": vn, "R2_w": round(mm.weighted_r2(y, p.loc[idx].values, w), 4), "RMSE_w pp": round(mm.weighted_rmse(y, p.loc[idx].values, w) * 100, 3),
                         "SSE cut vs shipped % [90% CI]": "" if vn == "shipped" else "{:+.1f} [{:+.1f}, {:+.1f}]".format(*boot(y, w, se0, se))})
        print(f"\n{label}  (n={len(idx)})"); print(pd.DataFrame(rows).to_string(index=False))
    # K
    variants, frames = {"shipped": (None, dd.K_FULL)}, {"shipped": (df_tr, df_te)}
    for rho, (a, b, _) in outs.items():
        vn = f"unified rho={rho}"
        variants[vn] = (None, [c if c != "fl_k_rate_bridge" else "u_k" for c in dd.K_FULL]); frames[vn] = (a, b)
    evaluate("K_rate_actual", variants, frames, n0_k, f"{name} -- K%")
    # BB
    variants, frames = {"shipped": (None, dd.BB_FULL)}, {"shipped": (df_tr, df_te)}
    for rho, (a, b, n0b) in outs.items():
        vn = f"unified rho={rho} (n0_bridge={n0b:.0f})"
        variants[vn] = (None, dd.BB_FULL); frames[vn] = (a, b)
    evaluate("BB_rate_actual", variants, frames, n0_bb, f"{name} -- BB%")
    # source mix
    for rho, (a, b, _) in list(outs.items())[:1]:
        pass


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    srcs = st.source_tables(); lgr = {lg: st.lg_rates(s) for lg, s in srcs.items()}
    debuts = st.fl_debuts(raw)
    pairs = {lg: st.build_pairs(debuts, s, lg) for lg, s in srcs.items()}
    print("PART 1 -- pooled mapping + per-league check for the mixed policy (K damping .7, BB none)")
    for metric, yname, col, dec, rho in (("K", "k_rate", "k", POLICY["dK"], 0.7), ("BB", "bb_rate", "bb", POLICY["dBB"], 1.0)):
        Y, W, U, L = [], [], [], []
        for lg, p in pairs.items():
            f = np.array([source_feature(h, col, dec, POLICY["n0"], lgr[lg][col]) for h in p["hist"]]) - lgr[lg][col]
            u = f * rho ** (p["gap"].values - 1)
            y, w = p[yname].values, p["pa"].values.astype(float)
            sd, sn = st.SHIPPED[(lg, metric)]
            fs = np.array([source_feature(h, col, sd, sn, lgr[lg][col]) for h in p["hist"]])
            print(f"  {lg:8s} {metric}%  n={len(p):3d}  own-line LOO R2_w shipped {st.loo_r2(y, fs, w):.3f} -> unified {st.loo_r2(y, u[:, None], w):.3f}")
            Y.append(y); W.append(w); U.append(u); L.append(np.full(len(y), lg))
        Y, W, U = np.concatenate(Y), np.concatenate(W), np.concatenate(U)
        X = np.column_stack([np.ones(len(Y)), U]); beta = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * Y))
        e = Y - X @ beta; se = np.sqrt(np.diag((np.sum(W * e ** 2) / (len(Y) - 2)) * np.linalg.inv(X.T @ (X * W[:, None]))))
        print(f"  POOLED {metric}%: FL rate = {beta[0]:.4f} + {beta[1]:.3f} * u   (se slope {se[1]:.3f}); LOO R2_w {st.loo_r2(Y, U[:, None], W):.3f}; n={len(Y)}")
        # league-relative check: does a league shift remain?
        for lg in pairs:
            m = np.concatenate(L) == lg; print(f"      mean residual {lg:8s} {np.average(e[m], weights=W[m]):+.4f}")

    print("\n\nPART 2 -- held-out harness: shipped bridge features vs unified (n0=50, decay K .25 / BB 1.0, reliability pick)")
    names_map = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id").assign(pitcher_id=lambda d: d["pitcher_id"].astype(str)).set_index("pitcher_id")["pitcher"].to_dict()
    unis = {"K .7 / BB 1.0": Unified(srcs, lgr, POLICY, {"k": 0.7, "bb": 1.0}), "none (1.0)": Unified(srcs, lgr, POLICY, 1.0)}
    w25, w26 = (bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)
    harness(raw, "Exp2 (train full-2025 -> 1H26; test 2025+1H26 -> 2H26)", [w25], [w26], [w25, w26], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], unis, names_map, None)
    harness(raw, "Exp1 (train 1H25 -> 2H25; test 2H25 -> 1H26)", [(bt.START_2025, bt.CUTOFF_2025)], [(bt._DAY_AFTER_2025_CUT, bt.END_2025)],
            [(bt._DAY_AFTER_2025_CUT, bt.END_2025)], [w26], unis, names_map, None)
