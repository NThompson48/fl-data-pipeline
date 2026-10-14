"""
fl_hitter_pre2026.py -- the PUBLISHED-style 2026 preseason hitter projection (what the production model would have said before 2026): the production code path (fl_hitter_projection.build_projections) run on a panel that contains
real seasons <= 2025 only, with every year shifted by +1 so that the real 2025 plays the role of "2026" (latest season) and the real 2026 is the "2027" target.  The constants are refit on real transitions through 2024 -> 2025
(shifted: through "2025" -> "2026"... i.e. nothing after 2025 is seen) and the pitch-level tables (EV90, Whiff%, Swing%) are restricted to 2025 pitches.  Leak-free for scoring any 2026 outcome.
Output: output/hitter_pre2026_published.parquet  (name_clean, proj_woba [final, park-neutral], proj_k, proj_bbr, proj_hrr, proj_babip, woba_direct, proj_k_reg, proj_bbr_reg, pa_t, pa_prev)
"""
import functools
import warnings

import numpy as np
import pandas as pd

import fl_hitter_ev as hev
import fl_hitter_model_dev as dev
import fl_hitter_projection as hp
import fl_hitter_swing as hsw
import fl_hitter_whiff as hwf

warnings.filterwarnings("ignore")


def build_shifted():
    """The full production projection table (with the WAR parts) for 2026 from 2025-and-earlier data: (d, c, fit)."""
    p = dev.load(); p = p[p["year"] <= 2025].copy(); p["year"] = p["year"] + 1                         # shift: real 2025 -> "2026"
    for mod, fn in ((hwf, "whiff_table"), (hev, "ev90_table"), (hsw, "swing_table")):
        if not isinstance(getattr(mod, fn), functools.partial):
            setattr(mod, fn, functools.partial(getattr(mod, fn), years=(2025,)))                        # 2025 pitches only (real years: the pitch tables are keyed by real year)
    hp.GAMES = {2023: 96, 2024: 96, 2025: 96, 2026: 96}; hp.TARGET_GAMES = 102                      # shifted years: the real 2022-25 seasons (96 games), projecting the real 2026 season (102 games)
    hp.read_league_rpa = lambda: 0.1529                                                                  # not used for wOBA
    hp.K_BRIDGE_PRIOR = False                                                                            # shifted years do not match real FL debut years (the prior is validated separately)
    fit = hp.fit_all(p); hp.dev_sd_pa = fit["pa"]["resid_sd"]
    d, c = hp.build_projections(p, fit)
    return d, c, fit


def build():
    d, c, fit = build_shifted()
    out = d[["name_clean", "gap", "last_season", "pa_t", "pa_prev", "proj_woba", "proj_woba_base", "woba_direct", "proj_k", "proj_k_reg", "proj_bbr", "proj_bbr_reg", "proj_hrr", "proj_babip", "xwoba_shrunk", "ev90_term", "whiff_term", "swing_term"]].copy()
    out["last_season"] = out["last_season"] - 1; out["real_last_season"] = out["last_season"]
    out.to_parquet("output/hitter_pre2026_published.parquet", index=False)
    print(f"{len(out)} hitters; n0 fit:", {k: round(v, 1) for k, v in fit["n0"].items()}, "| gap hitters:", int(out["gap"].sum())); print(out.describe().round(4).T[["count", "mean", "std"]].to_string())
    return out


if __name__ == "__main__":
    build()
