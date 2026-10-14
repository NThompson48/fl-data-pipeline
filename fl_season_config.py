"""
fl_season_config.py -- the season constants the production pipeline rolls forward each year (user 2026-10-13: "2027 is 102 games").
Change these (and follow ROLL_FORWARD.md) when the next season's data arrive.  Hitter pipeline and hitter bridge read them; the pitcher pipeline still carries its own
dated windows (fl_halfseason_backtest START_/END_/CUTOFF_ constants) -- see ROLL_FORWARD.md.
"""
LATEST_SEASON = 2026            # last completed FL season in the data
TARGET_YEAR = 2027              # season being projected
GAMES = {2022: 96, 2023: 96, 2024: 96, 2025: 96, 2026: 102, 2027: 102}     # regular-season games per team (user: 2025 = 96, 2026 = 102, 2027 = 102; 2022-24 = 96)
TARGET_GAMES = GAMES[TARGET_YEAR]
