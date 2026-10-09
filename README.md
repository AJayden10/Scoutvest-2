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

## Market values (Transfermarkt)

The collector reads the open [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
CSVs rather than scraping the site. Download `players.csv` and `player_valuations.csv`, then, after
`import_fbref`:

```bash
python manage.py import_transfermarkt --players players.csv --valuations player_valuations.csv --show-columns
python manage.py import_transfermarkt --players players.csv --valuations player_valuations.csv --dry-run
python manage.py import_transfermarkt --players players.csv --valuations player_valuations.csv
```

Players are matched on accent-insensitive name, with birth year breaking ties; ambiguous matches are
skipped and listed rather than guessed. Re-running is safe (values are upserted by player and date).

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
