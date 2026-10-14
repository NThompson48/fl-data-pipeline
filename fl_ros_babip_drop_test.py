"""
fl_ros_babip_drop_test.py -- the hitter ROS wOBA blend with BABIP removed from the component half (weights from the refit preseason fit: K -.177, BB .516, HR 1.414) vs the old weights (K -.157, BB .496, HR 1.378, BABIP .426).
2025 and 2026, as-of days 7-105, final wOBA (blend + xwOBA nudge + EV90), standard weights frozen from the old model, hitter-cluster bootstrap, weighted RMSE and MAE.
"""
import warnings
import numpy as np
import pandas as pd
import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_ros as hr
import fl_ros_by_day as bd

warnings.filterwarnings("ignore")
OLD = {"k": -0.157, "bbr": 0.496, "hrr": 1.378, "babip": 0.426}; NEW = {"k": -0.177, "bbr": 0.516, "hrr": 1.414}
T = hr.Tables(); rows = []; both = {"y": [], "o": [], "n": [], "nr": [], "id": []}
for yr in (2025, 2026):
    t0, _ = T.season_bounds(yr); Y, O, N, W, I = [], [], [], [], []
    for d in bd.DAYS:
        f, lg = hr.features(T, yr, t0 + pd.Timedelta(days=d), min_ros_pa=30)
        if len(f) < 40:
            continue
        hr.BETA = OLD; o = hr.predict(f, lg)["woba_final"].to_numpy(); hr.BETA = NEW; n = hr.predict(f, lg)["woba_final"].to_numpy()
        Y.append(f["woba_ros"].to_numpy()); O.append(o); N.append(n); W.append(f["n_ros"].to_numpy()); I.append(f.index.to_numpy())
    y, o, n, nr, ids = map(np.concatenate, (Y, O, N, W, I)); w = ew.weights_from_base(y, o, nr)[0]
    rows.append({"scored season": yr, **er.compare(y, o, n, w, ids)})
hr.BETA = OLD
print(er.table(rows))
