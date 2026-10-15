"""
fl_bsr_proxy.py -- baserunning runs per PA from POSITION, HEIGHT and WEIGHT (user 2026-10-14: "let's use position, height, and weight as the proxies for baserunning WAR").

Target: war_players.bsr_runs (reg_season; populated for 2026 only), per PA.  Player rows on two teams are summed.
Position: "DH" when >= 50% of his PA came at DH (war_players.dh_pa; user 2026-10-14 "DHs probably have worse baserunning. It's important to include them"; 2026 bins:
        DH share <= 50% -> about 0 runs / 600 PA, 50-70% -5.3, > 70% -6.3, so 50% is where it breaks), else war_players.primary_pos (his row with more PA).  The 24 hitters
        with no primary_pos are all DH-only cameos (dh_pa = pa) and fall in DH.  dh_pa is valid for 2026 only (2025 puts 89% of all PA at DH), so DH status is 2026's.
Model:  bsr / PA = position effect (9 levels incl. DH) + b_ht x (height - 72 in) + b_wt x (weight - 200 lb), weighted least squares with weights = PA (a per-PA rate's noise
        variance is ~ 1 / PA).  No individual baserunning record enters a player's rate: it is a pure proxy (the user's choice; the alternative of shrinking each
        player's own 2026 bsr toward this proxy was offered and not taken).
Missing inputs: no position -> the PA-weighted average position effect (as the position adjustment defaults to league average); no height / weight -> the fitted
        sample's PA-weighted mean (contributes 0).
Uncertainty (for the WAR simulation): residual variance law around the proxy, Var(rate_obs - proxy) = TAU2 + NOISE / PA (fl_eval_weights.fit_variance_law on the
        fitted residuals); TAU2 = true-talent spread the proxy does not explain, NOISE = one season's baserunning noise per PA.
Heights / weights: pbp_players (height "6'1\"", weight), falling back to rosters (listed roster values, ~95% rounded to 0 / 5 lb).
"""
import functools
import re

import numpy as np
import pandas as pd

POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
DH_SHARE = 0.5
HT0, WT0 = 72.0, 200.0


def _inches(s):
    m = re.match(r"\s*(\d+)'\s*(\d+)", str(s)); return float(m.group(1)) * 12 + float(m.group(2)) if m else np.nan


def _clean_lastfirst(s):
    import fl_hitter_bridge_test as hb
    return hb.norm(" ".join(reversed(str(s).split(", "))) if ", " in str(s) else s)


def bio():
    """name_clean -> height (in), weight (lb), from pbp_players with rosters as the fallback."""
    from sqlalchemy import text
    from fl_supabase import read_sql_df
    import fl_hitter_bridge_test as hb
    p = read_sql_df(text("select name, height, weight from pbp_players")); p["name_clean"] = p["name"].map(hb.norm)
    p["ht"] = p["height"].map(_inches); p["wt"] = pd.to_numeric(p["weight"], errors="coerce"); p = p.groupby("name_clean")[["ht", "wt"]].first()
    r = read_sql_df(text("select name, height_in, weight from rosters")); r["name_clean"] = r["name"].map(_clean_lastfirst)
    r = r.groupby("name_clean")[["height_in", "weight"]].first().rename(columns={"height_in": "ht", "weight": "wt"})
    return p.combine_first(r)


@functools.lru_cache(maxsize=None)
def positions(year=2026):
    """name_clean -> position for a season: DH if >= DH_SHARE of PA at DH, else war_players.primary_pos (row with more PA); also pa, bsr, pos_adj (published position runs)."""
    from sqlalchemy import text
    from fl_supabase import read_sql_df
    w = read_sql_df(text(f"select player, pa, dh_pa, bsr_runs, pos_adj_runs, primary_pos from war_players where game_type = 'reg_season' and year = {int(year)} and pa > 0"))
    w["name_clean"] = w["player"].map(_clean_lastfirst)
    pos = w.sort_values("pa").groupby("name_clean")["primary_pos"].last()
    d = w.groupby("name_clean").agg(pa=("pa", "sum"), dh_pa=("dh_pa", "sum"), bsr=("bsr_runs", lambda x: x.sum(min_count=1)),
                                    pos_adj=("pos_adj_runs", lambda x: x.sum(min_count=1))).join(pos.rename("field_pos"))
    d["dh_share"] = d["dh_pa"] / d["pa"]; d["pos"] = np.where(d["dh_share"] >= DH_SHARE, "DH", d["field_pos"]); return d


def training_data():
    d = positions(2026).copy(); d = d[d["bsr"].notna()].join(bio())
    return d[d["pos"].isin(POS)]


def _X(pos, ht, wt, ht_fill, wt_fill):
    pos = pd.Series(pos).reset_index(drop=True)
    cols = [(pos == p).astype(float).to_numpy() for p in POS]
    cols += [pd.Series(ht).reset_index(drop=True).fillna(ht_fill).to_numpy() - HT0, pd.Series(wt).reset_index(drop=True).fillna(wt_fill).to_numpy() - WT0]
    return np.column_stack(cols)


def fit(d, features=("pos", "ht", "wt")):
    """Weighted least squares of bsr per PA; returns a dict usable by predict()."""
    w = d["pa"].to_numpy(float); y = (d["bsr"] / d["pa"]).to_numpy(float)
    ht_fill = float(np.average(d["ht"].dropna(), weights=d.loc[d["ht"].notna(), "pa"])); wt_fill = float(np.average(d["wt"].dropna(), weights=d.loc[d["wt"].notna(), "pa"]))
    X = _X(d["pos"], d["ht"], d["wt"], ht_fill, wt_fill)
    keep = [i for i, nm in enumerate(POS + ["ht", "wt"]) if (nm in POS and "pos" in features) or nm in features]
    if "pos" not in features:
        X = np.column_stack([np.ones(len(d)), X[:, len(POS):]]); keep = [0] + [1 + i for i, nm in enumerate(["ht", "wt"]) if nm in features]
    Xk = X[:, keep]; sw = np.sqrt(w); beta = np.linalg.lstsq(Xk * sw[:, None], y * sw, rcond=None)[0]
    full = np.zeros(X.shape[1]); full[keep] = beta
    m = dict(features=tuple(features), beta=full, ht_fill=ht_fill, wt_fill=wt_fill)
    if "pos" in features:
        pa_pos = d.groupby("pos")["pa"].sum().reindex(POS).fillna(0).to_numpy(); m["pos_unknown"] = float(np.average(full[:len(POS)], weights=pa_pos))
    resid = y - predict(m, d["pos"], d["ht"], d["wt"])
    from fl_eval_weights import fit_variance_law
    a, b = fit_variance_law(resid, w); m["tau2"], m["noise"] = float(a), float(b)
    return m


def predict(m, pos, ht, wt):
    pos = pd.Series(pos).reset_index(drop=True); X = _X(pos, ht, wt, m["ht_fill"], m["wt_fill"]); b = m["beta"]
    if "pos" in m["features"]:
        out = X @ b; unknown = ~pos.isin(POS).to_numpy(); out[unknown] += m["pos_unknown"]; return out
    return b[0] + X[:, len(POS):] @ b[1:]


PRODUCTION_FEATURES = ("pos",)      # user 2026-10-14: position incl. DH only -- height / weight hurt held-out accuracy once DH is separated (fl_bsr_proxy_validation.py)


@functools.lru_cache(maxsize=None)
def production_model():
    return fit(training_data(), PRODUCTION_FEATURES)


def player_positions(names, fallback):
    """Baserunning position for projected hitters: 2026 role (DH if >= 50% of PA at DH), else the projection's primary_pos; unknown -> average position."""
    return pd.Series(names).map(positions(2026)["pos"]).fillna(pd.Series(fallback).reset_index(drop=True)).to_numpy()


if __name__ == "__main__":
    d = training_data(); m = fit(d, PRODUCTION_FEATURES)
    print(f"fit on {len(d)} hitters, {d['pa'].sum():.0f} PA")
    print("position effects (runs per 600 PA):", {p: round(600 * v, 2) for p, v in zip(POS, m["beta"][:len(POS)])}, "unknown:", round(600 * m["pos_unknown"], 2))
    print(f"production features {m['features']}; residual true-talent SD {600 * np.sqrt(max(m['tau2'], 0)):.2f} "
          f"runs / 600 PA, season noise SD at 600 PA {600 * np.sqrt(m['noise'] / 600):.2f}")
