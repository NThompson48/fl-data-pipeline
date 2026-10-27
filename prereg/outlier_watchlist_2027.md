# Outlier watch list (frozen 2026-10-27)

Window: 2025 + 2026 regular season (data as of the 2026-10-26 refresh). 506,529 pitches of 8 main types scored against distributions fitted on other pitchers. Method and parameters are in `outlier_watchlist_2027.json` (`meta`) and `fl_outlier_watchlist.py`.

Thresholds on the isolation-forest mean score (pitchers with >= 150 scored pitches, n = 577): top 5% >= 1.87, top 10% >= 1.38; most-unusual-type score top 5% >= 3.04.

Why it is on record: 2026-10-27 outlier dummy test: top 5% isolation-forest dummy, K% RMSE better in 3 of 4 pairings (mean -.00005), flagged pitchers' K% residual +0.3 to +1.7 points in all four, coefficient +.003 to +.027; exploratory (one of 7 variants), 5-14 flagged pitchers per pairing.  Not in production.

Watch rule: report only: after 2027, flagged pitchers' (flag_oi_if_top5, flag_oi_if_top10) mean K% residual vs the frozen production projection (prereg/c10_frozen_projections.json), PA-weighted, level-free, with a pitcher bootstrap.

## Pitchers flagged top 5% (isolation-forest mean score)

| Pitcher | 2026 team | Pitches | Score | Most unusual type (score) | Stuff+ | Projected 2027 K% |
|---|---|---|---|---|---|---|
| Robbins, Zane | Florence Y'alls | 247 | 5.47 | four_seam (5.67) | 106 | 18.0% |
| Pierson, Kenny | Lake Erie Crushers | 1501 | 4.62 | sinker (5.52) | 87 | 12.6% |
| Widener, Jacob | Sussex County Miners | 709 | 4.53 | four_seam (5.59) | 115 | 26.8% |
| Smith, Jackson | Mississippi Mud Monsters | 1929 | 4.52 | cutter (5.26) | 99 | 17.8% |
| Cook, Avery | Windy City ThunderBolts | 436 | 4.09 | curveball (5.01) | 114 | 23.0% |
| MacMillan, Blake | Trois-Rivieres Aigles | 730 | 3.85 | cutter (5.09) | 115 | 26.1% |
| Bedder, Dylan | Brockton Rox | 640 | 3.79 | slider (4.59) | 94 | 21.2% |
| Sanchez, Brett | Joliet Slammers | 513 | 3.76 | curveball (5.66) | 134 | 21.7% |
| Traxel, Blaine | New York Boulders | 492 | 3.66 | curveball (4.63) | 98 | 18.9% |
| Hungate, Chase | New Jersey Jackals | 492 | 3.55 | sinker (3.85) | 95 | 20.5% |
| Ito, Ryusuke | Quebec Capitales | 277 | 3.24 | changeup (3.30) | 90 | 18.7% |
| Melendez, Omar | Brockton Rox | 1528 | 3.22 | slider (3.90) | 114 | 19.9% |
| Collett, Keegan | Gateway Grizzlies | 501 | 3.05 | curveball (3.15) | 132 | 26.4% |
| Khan, Sebastian | Quebec Capitales | 205 | 2.75 | changeup (2.87) | 87 | 15.4% |
| Harris, Everette | Tri-City ValleyCats | 200 | 2.70 | slider (3.63) | 86 | 15.6% |
| Testa, Joe | New Jersey Jackals | 697 | 2.68 | changeup (4.00) | 86 | 16.5% |
| McAuliffe, Nick | Evansville Otters | 652 | 2.64 | four_seam (2.89) | 129 | 27.3% |
| Curry, Bobby | New Jersey Jackals | 296 | 2.61 | curveball (4.16) | 108 | 17.3% |
| Webb, Colt | New Jersey Jackals | 634 | 2.58 | changeup (2.99) | 89 | 17.5% |
| Good, Ty | Gateway Grizzlies | 1884 | 2.57 | changeup (3.81) | 104 | 21.9% |
| Morgan, Cooper | Quebec Capitales | 930 | 2.50 | curveball (3.51) | 113 | 26.5% |
| Brown, Daniel | Windy City ThunderBolts | 167 | 2.40 | sinker (2.26) | 110 | 21.7% |
| Plumadore, Carsen | Windy City ThunderBolts | 1503 | 2.30 | curveball (2.76) | 100 | 21.9% |
| Sechrist, Zander | Washington Wild Things | 903 | 2.25 | cutter (3.57) | 98 | 14.9% |
| Primeaux, Parker | Sussex County Miners | 839 | 2.18 | changeup (2.67) | 91 | 20.7% |
| Harper, Scott | New York Boulders | 1754 | 2.08 | slider (2.41) | 112 | 24.6% |
| Almanzar, Elian | Down East Bird Dawgs | 274 | 2.03 | four_seam (2.20) | 117 | 24.2% |
| Dixon, Jake | Ottawa Titans | 210 | 1.95 | changeup (2.02) | 100 | 20.0% |
| Rohde, Isaac | New York Boulders | 2521 | 1.95 | curveball (2.57) | 100 | 21.0% |

## Additional pitchers flagged top 10% (not top 5%)

| Pitcher | 2026 team | Pitches | Score | Most unusual type (score) | Stuff+ | Projected 2027 K% |
|---|---|---|---|---|---|---|
| Boeree, James | Mississippi Mud Monsters | 620 | 1.86 | four_seam (2.08) | 96 | 18.9% |
| Reinoso, Dayan | Lake Erie Crushers | 775 | 1.85 | sinker (2.38) | 126 | 22.3% |
| Miranda, Agnel | nan | 158 | 1.83 | four_seam (2.01) | — | —% |
| Del Prado, Nick | Trois-Rivieres Aigles | 368 | 1.81 | four_seam (2.08) | 113 | 23.0% |
| Puccetti, Dominic | Ottawa Titans | 2140 | 1.80 | changeup (2.70) | 95 | 18.0% |
| Phelps, Holden | New York Boulders | 195 | 1.79 | changeup (1.90) | 103 | 17.9% |
| Pena, Bryan | Ottawa Titans | 570 | 1.78 | four_seam (2.12) | 89 | 19.3% |
| Oliver, Hambleton | Schaumburg Boomers | 435 | 1.76 | slider (2.99) | 94 | 17.1% |
| Roland, Cole | Quebec Capitales | 590 | 1.74 | sinker (4.38) | 98 | 23.4% |
| Savino, Dylan | Windy City ThunderBolts | 319 | 1.69 | four_seam (1.97) | 95 | 19.0% |
| McCartney, Seth | Mississippi Mud Monsters | 214 | 1.68 | four_seam (2.21) | 99 | 17.7% |
| Stil, Matt | Sussex County Miners | 373 | 1.65 | four_seam (1.77) | 130 | 26.0% |
| Plumadore, Carsen | nan | 293 | 1.64 | slider (1.94) | — | —% |
| Webster, Evan | Florence Y'alls | 3606 | 1.60 | sweeper (3.32) | 98 | 19.1% |
| Hopewell, Chase | Lake Erie Crushers | 1354 | 1.58 | changeup (3.06) | 96 | 18.3% |
| Luciano, Jordy | New Jersey Jackals | 315 | 1.57 | four_seam (2.01) | 117 | 24.6% |
| Galva, Claudio | Gateway Grizzlies | 1000 | 1.56 | sinker (2.75) | 87 | 14.8% |
| Burcham, Jacob | Gateway Grizzlies | 1301 | 1.55 | four_seam (1.82) | 101 | 21.9% |
| Walsh, John | Mississippi Mud Monsters | 760 | 1.55 | cutter (3.45) | 91 | 13.5% |
| Duquette, Jagger | Lake Erie Crushers | 242 | 1.51 | sinker (1.82) | 100 | 17.6% |
| Traver, Eliott | Lake Erie Crushers | 218 | 1.50 | curveball (3.51) | 103 | 17.8% |
| Miranda, Agnel | New Jersey Jackals | 496 | 1.49 | four_seam (1.67) | 91 | 18.3% |
| Treece, Zac | Gateway Grizzlies | 934 | 1.47 | sweeper (2.55) | 90 | 18.5% |
| Hellgeth, Bryce | Windy City ThunderBolts | 1465 | 1.47 | four_seam (1.65) | 98 | 18.6% |
| Fernandez, Matthew | Florence Y'alls | 615 | 1.46 | changeup (2.18) | 109 | 19.6% |
| Landry, Nathan | Trois-Rivieres Aigles | 323 | 1.44 | changeup (1.85) | 93 | 19.0% |
| Chapple, Bronson | Trois-Rivieres Aigles | 1180 | 1.43 | four_seam (1.98) | 93 | 17.5% |
| Ragins, Darrien | Lake Erie Crushers | 1362 | 1.39 | changeup (3.13) | 103 | 19.0% |
| Fuenmayor, Liu | Ottawa Titans | 1123 | 1.39 | changeup (2.47) | 95 | 17.7% |

## Outlier pitches by type (top 5% of pitcher cell means within each type; >= 30 pitches of that type)

### four_seam (27 of 522 cells)

| Pitcher | 2026 team | Pitches | Isolation-forest score | Mahalanobis score |
|---|---|---|---|---|
| Ordorica, Steven | Lake Erie Crushers | 36 | 8.10 | 5.59 |
| Robbins, Zane | Florence Y'alls | 192 | 5.67 | 2.39 |
| Widener, Jacob | Sussex County Miners | 104 | 5.59 | 2.81 |
| Cook, Avery | Windy City ThunderBolts | 145 | 4.36 | 2.39 |
| Gardner, Sam | Gateway Grizzlies | 41 | 4.35 | 1.75 |
| Brown, Dan | Tri-City ValleyCats | 33 | 4.11 | 1.82 |
| Marrero, Bryan | Brockton Rox | 62 | 3.96 | 2.56 |
| Bedder, Dylan | Brockton Rox | 316 | 3.67 | 1.96 |
| Morgan, Cooper | Quebec Capitales | 94 | 3.25 | 2.16 |
| Melendez, Omar | Brockton Rox | 379 | 3.18 | 1.79 |
| Collett, Keegan | Gateway Grizzlies | 378 | 3.01 | 1.41 |
| MacMillan, Blake | Trois-Rivieres Aigles | 171 | 2.99 | 1.35 |
| Webb, Colt | New Jersey Jackals | 406 | 2.94 | 1.79 |
| McAuliffe, Nick | Evansville Otters | 315 | 2.89 | 1.46 |
| Fowler, Michael | Mississippi Mud Monsters | 95 | 2.70 | 1.24 |
| Good, Ty | Gateway Grizzlies | 1101 | 2.69 | 1.42 |
| Calderon, Jean | Lake Erie Crushers | 62 | 2.60 | 1.51 |
| DeCarlo, Nick | Tri-City ValleyCats | 100 | 2.58 | 2.00 |
| Ragins, Darrien | Lake Erie Crushers | 318 | 2.52 | 1.64 |
| Hattenbach, Hayden | Trois-Rivieres Aigles | 89 | 2.50 | 1.21 |
| Davis, Tyler | Windy City ThunderBolts | 60 | 2.33 | 1.13 |
| Lodes, Jett | Florence Y'alls | 36 | 2.31 | 1.63 |
| Scott, Braden | Evansville Otters | 230 | 2.24 | 1.75 |
| Galva, Claudio | Gateway Grizzlies | 442 | 2.21 | 1.38 |
| McCartney, Seth | Mississippi Mud Monsters | 128 | 2.21 | 0.91 |
| Puccetti, Dominic | Ottawa Titans | 1164 | 2.20 | 1.10 |
| Almanzar, Elian | Down East Bird Dawgs | 194 | 2.20 | 1.21 |

### sinker (18 of 356 cells)

| Pitcher | 2026 team | Pitches | Isolation-forest score | Mahalanobis score |
|---|---|---|---|---|
| Pierson, Kenny | Lake Erie Crushers | 669 | 5.52 | 2.51 |
| Smith, Jackson | Mississippi Mud Monsters | 1266 | 4.52 | 2.08 |
| Roland, Cole | Quebec Capitales | 36 | 4.38 | 2.66 |
| Folkman, Max | Washington Wild Things | 34 | 3.95 | 2.87 |
| Hungate, Chase | New Jersey Jackals | 274 | 3.85 | 1.91 |
| Widener, Jacob | Sussex County Miners | 267 | 3.83 | 2.11 |
| Cook, Avery | Windy City ThunderBolts | 116 | 3.75 | 1.86 |
| Devers, Luis | Mississippi Mud Monsters | 290 | 3.04 | 2.38 |
| Galva, Claudio | Gateway Grizzlies | 33 | 2.75 | 1.41 |
| Primeaux, Parker | Sussex County Miners | 170 | 2.60 | 1.51 |
| Ito, Ryusuke | Quebec Capitales | 37 | 2.53 | 1.89 |
| Coe, Garrett | New York Boulders | 51 | 2.52 | 1.82 |
| Reinoso, Dayan | Lake Erie Crushers | 503 | 2.38 | 1.54 |
| Marrero, Bryan | Brockton Rox | 38 | 2.32 | 1.84 |
| Whaley, Alec | Gateway Grizzlies | 296 | 2.29 | 1.74 |
| Brown, Daniel | Windy City ThunderBolts | 151 | 2.26 | 1.55 |
| Thompson, Darrell | Schaumburg Boomers | 46 | 2.21 | 0.80 |
| Carver, Cal | Trois-Rivieres Aigles | 126 | 2.04 | 1.49 |

### cutter (5 of 97 cells)

| Pitcher | 2026 team | Pitches | Isolation-forest score | Mahalanobis score |
|---|---|---|---|---|
| Smith, Jackson | Mississippi Mud Monsters | 272 | 5.26 | 2.20 |
| MacMillan, Blake | Trois-Rivieres Aigles | 291 | 5.09 | 3.01 |
| Widener, Jacob | Sussex County Miners | 75 | 4.44 | 1.99 |
| Scott, Braden | Evansville Otters | 227 | 3.73 | 2.78 |
| Sechrist, Zander | Washington Wild Things | 138 | 3.57 | 2.01 |

### slider (26 of 504 cells)

| Pitcher | 2026 team | Pitches | Isolation-forest score | Mahalanobis score |
|---|---|---|---|---|
| Widener, Jacob | Sussex County Miners | 184 | 4.96 | 2.40 |
| Bedder, Dylan | Brockton Rox | 217 | 4.59 | 2.46 |
| Smith, Jackson | Mississippi Mud Monsters | 260 | 4.07 | 1.98 |
| Melendez, Omar | Brockton Rox | 71 | 3.90 | 2.05 |
| Testa, Joe | New Jersey Jackals | 191 | 3.82 | 1.97 |
| Hungate, Chase | New Jersey Jackals | 89 | 3.68 | 2.23 |
| Harris, Everette | Tri-City ValleyCats | 57 | 3.63 | 1.97 |
| Gwin, Riley | Quebec Capitales | 30 | 3.33 | 2.55 |
| Sanchez, Brett | Joliet Slammers | 157 | 3.16 | 2.30 |
| Ito, Ryusuke | Quebec Capitales | 64 | 3.07 | 2.28 |
| Oliver, Hambleton | Schaumburg Boomers | 159 | 2.99 | 1.68 |
| Paulina, Joshua | Mississippi Mud Monsters | 126 | 2.76 | 1.68 |
| MacMillan, Blake | Trois-Rivieres Aigles | 131 | 2.71 | 1.67 |
| Morgan, Cooper | Quebec Capitales | 102 | 2.61 | 1.55 |
| Plumadore, Carsen | Windy City ThunderBolts | 407 | 2.50 | 1.52 |
| Brown, Dan | Tri-City ValleyCats | 32 | 2.48 | 1.30 |
| Miner, Jace | Down East Bird Dawgs | 52 | 2.45 | 1.76 |
| Traver, Eliott | Lake Erie Crushers | 34 | 2.44 | 2.23 |
| Harper, Scott | New York Boulders | 550 | 2.41 | 1.86 |
| McAuliffe, Nick | Evansville Otters | 337 | 2.40 | 1.29 |
| Traxel, Blaine | New York Boulders | 95 | 2.33 | 1.59 |
| Walsh, John | Mississippi Mud Monsters | 136 | 2.26 | 1.14 |
| Good, Ty | Gateway Grizzlies | 427 | 2.22 | 1.10 |
| Khan, Sebastian | Quebec Capitales | 68 | 2.17 | 1.88 |
| Zarnoch, Kevin | New Jersey Jackals | 64 | 2.13 | 0.87 |
| Scott, Braden | Evansville Otters | 526 | 1.98 | 1.18 |

### sweeper (2 of 37 cells)

| Pitcher | 2026 team | Pitches | Isolation-forest score | Mahalanobis score |
|---|---|---|---|---|
| Cook, Avery | Windy City ThunderBolts | 82 | 3.85 | 2.93 |
| Pierson, Kenny | Lake Erie Crushers | 360 | 3.61 | 2.04 |

### curveball (16 of 316 cells)

| Pitcher | 2026 team | Pitches | Isolation-forest score | Mahalanobis score |
|---|---|---|---|---|
| Sanchez, Brett | Joliet Slammers | 108 | 5.66 | 3.63 |
| Robbins, Zane | Florence Y'alls | 35 | 5.56 | 2.84 |
| Cook, Avery | Windy City ThunderBolts | 47 | 5.01 | 2.78 |
| Widener, Jacob | Sussex County Miners | 44 | 4.95 | 2.53 |
| Traxel, Blaine | New York Boulders | 92 | 4.63 | 2.93 |
| Curry, Bobby | New Jersey Jackals | 47 | 4.16 | 2.77 |
| Pierson, Kenny | Lake Erie Crushers | 57 | 4.02 | 2.23 |
| Traver, Eliott | Lake Erie Crushers | 38 | 3.51 | 2.52 |
| Morgan, Cooper | Quebec Capitales | 196 | 3.51 | 1.87 |
| Novis, Tino | Ottawa Titans | 42 | 3.23 | 2.47 |
| Collett, Keegan | Gateway Grizzlies | 123 | 3.15 | 1.76 |
| Czyz, T.J. | Down East Bird Dawgs | 145 | 2.86 | 1.45 |
| Melendez, Omar | Brockton Rox | 390 | 2.83 | 1.63 |
| Thiels, Brenton | Mississippi Mud Monsters | 38 | 2.78 | 1.63 |
| Plumadore, Carsen | Windy City ThunderBolts | 32 | 2.76 | 1.91 |
| Rohde, Isaac | New York Boulders | 134 | 2.57 | 1.93 |

### changeup (20 of 390 cells)

| Pitcher | 2026 team | Pitches | Isolation-forest score | Mahalanobis score |
|---|---|---|---|---|
| MacMillan, Blake | Trois-Rivieres Aigles | 68 | 4.51 | 1.92 |
| Widener, Jacob | Sussex County Miners | 35 | 4.13 | 2.35 |
| Pierson, Kenny | Lake Erie Crushers | 414 | 4.13 | 1.99 |
| Testa, Joe | New Jersey Jackals | 80 | 4.00 | 2.47 |
| Smith, Jackson | Mississippi Mud Monsters | 131 | 3.91 | 2.17 |
| Good, Ty | Gateway Grizzlies | 107 | 3.81 | 1.60 |
| Traxel, Blaine | New York Boulders | 267 | 3.66 | 2.07 |
| Melendez, Omar | Brockton Rox | 619 | 3.48 | 1.81 |
| Sanchez, Brett | Joliet Slammers | 240 | 3.44 | 3.25 |
| Ito, Ryusuke | Quebec Capitales | 155 | 3.30 | 2.67 |
| Veinbergs, Lukas | Gateway Grizzlies | 120 | 3.19 | 2.11 |
| Ragins, Darrien | Lake Erie Crushers | 34 | 3.13 | 1.93 |
| Hopewell, Chase | Lake Erie Crushers | 112 | 3.06 | 1.68 |
| Webb, Colt | New Jersey Jackals | 36 | 2.99 | 2.59 |
| Khan, Sebastian | Quebec Capitales | 108 | 2.87 | 1.99 |
| Kuzemka, Christian | Windy City ThunderBolts | 53 | 2.86 | 1.35 |
| Novis, Tino | Ottawa Titans | 89 | 2.75 | 1.64 |
| Puccetti, Dominic | Ottawa Titans | 45 | 2.70 | 1.03 |
| Primeaux, Parker | Sussex County Miners | 357 | 2.67 | 1.61 |
| Plumadore, Carsen | Windy City ThunderBolts | 322 | 2.63 | 1.77 |

### splitter (3 of 50 cells)

| Pitcher | 2026 team | Pitches | Isolation-forest score | Mahalanobis score |
|---|---|---|---|---|
| Balzan, Jackson | Sussex County Miners | 51 | 3.05 | 2.47 |
| Peters, Garrett | New York Boulders | 58 | 2.42 | 2.20 |
| Barreto, Brayhans | Tri-City ValleyCats | 43 | 2.38 | 2.10 |
