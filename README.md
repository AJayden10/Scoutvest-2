# ScoutVest backend

Django REST API over the PostgreSQL database built in Stage 1.

## Setup
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set DATABASE_URL
python manage.py migrate
python manage.py import_fbref --show-columns     # check column detection first
python manage.py import_fbref                    # from data/raw/*.csv
python manage.py import_fbref --from-table       # or from the player_stats table
python manage.py runserver
python manage.py test
```

## Endpoints (`/api/`)
| Endpoint | Purpose |
|---|---|
| `GET players/` | list; filters `q position league club signal risk min_age max_age min_value max_value min_upside`, `ordering=-upside` |
| `GET players/<id>/` | profile + value history, forecast, season stats |
| `GET players/compare/?ids=1,2,3` | 2-4 players side by side |
| `GET/POST watchlist/`, `DELETE watchlist/<id>/` | watchlist |
| `GET market-trends/` | average value by position / league / month |
| `POST simulator/` | single transfer ROI (`playerId`) or portfolio (`budget`) |

Player JSON uses the frontend's field names (`currentValue`, `predictedValue`, `upside`, `riskScore`, ...).
Fields with no data yet (value, prediction, fee) are `null`.

## Data model
`Club`, `Player`, `PlayerSeasonStat` (FBref), `MarketValue` (Transfermarkt, to be collected),
`Prediction` (written by the ML pipeline), `WatchlistEntry`.

## Data: one import from the Transfermarkt dataset (recommended)

The free, CC0-licensed [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) download
(Kaggle: `davidcariboo/player-scores`) contains everything the model needs: players, clubs, match-by-match
appearances (goals, assists, minutes, cards) and market-value history. Unzip it into one folder, then:

```bash
python manage.py migrate
python manage.py import_tm_dataset --dir path/to/unzipped --dry-run     # see what it would import
python manage.py import_tm_dataset --dir path/to/unzipped               # about a minute
```

Defaults: ten leagues (`--leagues GB1,ES1,IT1,L1,FR1,NL1,PO1,BE1,TR1,SC1`), seasons from 2015
(`--since-season`). Seasons run July-June. Player ids come from Transfermarkt, so there is no name matching.
Re-running updates rather than duplicates. This source has no xG or progressive passes; those stay empty
(the model handles missing values) until another importer fills them.

### Optional: FBref / name-matched sources
`import_fbref` and `import_transfermarkt` (name + birth-year matching) are still available for CSVs from
other sources. FBref removed its advanced (Opta) stats in January 2026, so check
`import_fbref --show-columns` before relying on it.

## Value model (XGBoost)

```bash
pip install -r requirements.txt
python manage.py train_model            # trains and prints MAE/RMSE/R² against a naive baseline
python manage.py generate_predictions   # writes Prediction rows the API serves
```

- One row per player-season, snapshotted at season end (30 June). Features: age, minutes, per-90 goals,
  assists, xG, xA, progressive passes/carries, position, current value, 12-month value change, contract
  years left. Target: the value 2 years later (`--horizon`), learned as log(future / current).
- Evaluation is a **time split**: train on older seasons, test on the newest labelled one. The command
  also reports a "value stays the same" baseline and warns if the model does not beat it.
- `generate_predictions` adds a signal (Strong buy / Undervalued / Breakout / Monitor / Overvalued), a
  risk score, a confidence score and up to three plain-English reasons per player. Thresholds live at the
  top of `players/ml/signals.py`.
- `GET /api/model/performance/` returns the latest metrics and feature importances.
- Models are saved to `models/` (git-ignored; override with `MODEL_DIR`).

## Understat (xG, xA, xGChain, xGBuildup, shots, key passes)

Price-context features (age, club, contract) mostly rebuild an age curve. Performance measured independently of price is
what lets the model say a player is better than his valuation. Understat covers the Big 5 + Russia from 2014/15.

```bash
python manage.py import_tm_dataset --dir data/raw/tm      # players, clubs, minutes, values first
python manage.py collect_understat --from-year 2015       # polite: one request per 3s, cached as JSON in data/raw/understat
python manage.py import_understat                         # matches by name + minutes, prints match % per league-season
python manage.py train_model                              # now uses xG-based features
python manage.py train_model --exclude xg90 xa90 npxg90 xgchain90 xgbuildup90 key_passes90 shots90 --name no-xg   # ablation
python manage.py generate_predictions
```

Compare the metrics of the two runs: that is the honest test of whether xG helps. Players from leagues Understat does not
cover keep blank xG (the model tolerates missing values). Check Understat's terms before any commercial use.
Note: the collector was written against Understat's public JSON endpoint but could not be tested live from the build
sandbox; if it fails, the response-shape error message will say what came back.
