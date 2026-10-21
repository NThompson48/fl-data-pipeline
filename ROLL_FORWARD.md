# Rolling the projections forward to a new season

Written 2026-10-13. The 2027 preseason projections are already built (target 2027, 102 games). This checklist is for
**in-season 2027** (rest-of-season) and the **2028 preseason**, once 2027 data exist.

## 1. Season constants (one place)
`fl_season_config.py`: set `LATEST_SEASON`, `TARGET_YEAR`, and add the new season's games to `GAMES`
(2027 = 102). The hitter pipeline (`fl_hitter_projection.py`) and the hitter MiLB bridge (`fl_hitter_bridge.py`)
read these.

## 2. Hard-coded dates and years still to move (not yet wired to the config)
| Area | Where | What to change |
|---|---|---|
| Pitcher K% / BB% model | `fl_halfseason_backtest.py` (`START_2025`, `END_2025`, `START_2026`, `CUTOFF_2026`, `END_2026`) and `fl_production_projection.py` (about 70 year references) | The frozen model is fit on full 2025 -> 1H2026. For 2028: decide whether to refit on 2026 -> 1H2027, and move the windows. |
| Pitcher RA9 | `fl_ra9_trustmark_projection.py` (about 70 references: replacement level, league RA9, `ip_2026`, innings model) | League constants and the "latest season" columns. |
| Hitter panel | `fl_hitter_panel.py` (`year_label in ('2025','2026')` in four queries) | Add the new season to every query. |
| Hitter park surfaces | `fl_hitter_park_surface_final.py`, `fl_hitter_park_hr_surface.py` (2025-26 data) | Refit on 2025-27 tracked balls. |
| Hitter bridge | `fl_hitter_bridge.py` (`h26 = h[h["year"] == 2026]` league levels), MiLB pull `pull_milb_hitting.py` | Latest-season league levels; pull new MiLB seasons. |
| Rest of season | `fl_ros_build.py` (`SNAP_DATES` in 2026), `fl_ros_live.py` (live path) | New as-of dates for 2027; live path uses pooled coefficients and the published preseason projection (pitchers). |
| Backtest builds | `fl_hitter_pre2026.py` (shifted years) | Shift by one more season for the next calibration checks. |
| Output file names | `fl_2027_*.parquet`, `ros_*_2026.parquet` (dashboard reads these names) | Rename together with `projections-dashboard/src` and `run_all.py` MANIFEST. |

## 3. Data that must be refreshed
Supabase tables (lineup_stats, pitches, war_players, advanced_splits, war_pitchers), the tracking file
`2025_2026_pitches.parquet` (add 2027), MiLB / Pioneer / college source tables (Pioneer: rerun `pull_statscrew_pioneer_batting.py` with the new season in its year range; college: FanGraphs exports into input/fangraphs_college/).

## 3b. Data hygiene rules (added 2026-10-20)
- **Freshness is checked automatically.** `run_all.py` first runs `fl_data_freshness.py`, which compares the tracking parquet, the pitcher cache and the two hitter caches with the database and stops if any is stale
  (the tracking parquet had been left 3 weeks behind the season).  Refresh with `python pull_2025_2026.py`, delete `output/fl_raw_pitches_flagged.parquet`, or `fl_hitter_ev.load_bbe(refresh=True)` / `fl_hitter_whiff.load(refresh=True)` as the message says.
- **Row order is fixed on load.** The database returns rows in arbitrary order and results depended on it (pitcher ROS: mean .013 RA9; one pitcher's innings moved 41).  `pull_2025_2026.py` sorts on (game_id, pitch_uid) and
  `fl_pitcher_features.canonical_order` sorts the pitcher cache on `ORDER_KEY`.  Any new loader that feeds cross-validation or first-row picks should sort too.
- **One name per pitcher_id.** `fl_pitcher_features.harmonize_names` keeps the most frequent spelling (3 ids carried two).  Name-keyed tables (war_pitchers) need an alias when the spellings are the same person
  (`PITCHER_NAME_ALIASES` in `fl_ra9_trustmark_projection.py`; hitters: `NAME_ALIASES` in `fl_statscrew_pull.py`).
- **Regular season only.** The tracking parquet carries `game_type`; park surfaces, the fielding metric and `fl_error_analysis_common.load_tracking` filter to `reg_season`.

## 4. After the rebuild
`python run_all.py`, then the pre-registered 2027 checks (`fl_2027_preregistered_checks.py`).
