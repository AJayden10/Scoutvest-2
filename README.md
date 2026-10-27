# Free Agent API (ScoutVest backend)

Django REST API behind [Free Agent](https://github.com/AJayden10/scoutvest-frontend), a football scouting app that
forecasts player market values and finds statistically similar players. The data stack is deliberately targeted:
one lesser-known league (**Australia's A-League Men**) with free, open, recent data.

```mermaid
flowchart LR
  TM[transfermarkt-datasets<br/>CC0, weekly] -->|import_tm_dataset| DB[(PostgreSQL)]
  SC[SkillCorner open data<br/>MIT, A-League 2024/25] -->|import_skillcorner| DB
  DB -->|build_frame| ML[XGBoost<br/>train_model]
  ML -->|generate_predictions| DB
  DB --> API[Django REST API]
  API -->|/players /similar /health| FE[React frontend<br/>Vercel]
```

## Quickstart (A-League, about 10 minutes)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # without DATABASE_URL it uses SQLite, fine for a first run
python manage.py migrate

# 1. Players, clubs, minutes, goals, market values (A-League is the default league)
#    Download the Kaggle dataset "davidcariboo/player-scores" (transfermarkt-datasets) and unzip it.
python manage.py import_tm_dataset --dir path/to/player-scores --since-season 2015

# 2. Physical, off-ball-run and passing profiles for 2024/25
git clone --depth 1 https://github.com/SkillCorner/opendata.git ../skillcorner-opendata
python manage.py import_skillcorner --dir ../skillcorner-opendata/data/aggregates

# 3. Model
python manage.py train_model          # prints metrics vs three baselines, saves model + metrics + holdout CSV
python manage.py generate_predictions # writes the predictions the API serves

python manage.py runserver
curl localhost:8000/api/health/
```

Run step 1 before step 2: SkillCorner players are matched to the Transfermarkt ones by name and birth year, so the
similarity search can show market values. Players SkillCorner has that Transfermarkt doesn't are created (use
`--no-create` to skip them). The import prints how many matched by each rule.

Other leagues: `--leagues AUS1,DK1,SE1` (any Transfermarkt competition id; the Big 5 are in `tm_dataset.BIG_LEAGUES`).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | SQLite file | e.g. `postgresql://user:pass@host:5432/scoutvest` |
| `DJANGO_SECRET_KEY` | dev key | **set in production** |
| `DJANGO_DEBUG` | `1` | `0` in production |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | your API's hostname |
| `CORS_ALLOWED_ORIGINS` | Vite dev server | your frontend, e.g. `https://scoutvest-frontend.vercel.app` |
| `MODEL_DIR` | `models/` | where trained models, metrics and holdout CSVs go |
| `LOG_LEVEL` | `INFO` | request log verbosity |

The frontend needs `VITE_API_URL=https://<your-api>/api` set **at build time** in Vercel. Without it the deployed
site calls `localhost` and shows demo data (it now says so on screen).

## Endpoints (`/api/`)

| Endpoint | Purpose |
|---|---|
| `GET health/` | status, data sources with row counts and latest dates, model version and training date (503 if the DB is down) |
| `GET players/` | list; filters `q position league club signal risk min_age max_age min_value max_value min_upside predicted`, `ordering=-upside` |
| `GET players/<id>/` | profile, value history, forecast with 80% range, risk factors, season stats and SkillCorner metrics |
| `GET players/<id>/similar/` | same-position players with the closest profile; `max_value` (budget €M), `max_age`, `min_minutes`, `season`, `limit` |
| `GET players/compare/?ids=1,2,3` | 2-4 players side by side |
| `GET/POST watchlist/`, `DELETE watchlist/<id>/` | watchlist |
| `GET market-trends/` | average value by position / league / month |
| `POST simulator/` | single transfer ROI (`playerId`) or portfolio (`budget`) |
| `GET model/performance/` | full metrics of the latest model (baselines, segments, range coverage) |

Errors always come back as `{"detail": "...", "code": "..."}`.

## Data sources and licences

| Source | What | Licence / terms | Used for |
|---|---|---|---|
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | players, clubs, appearances, market values; updated weekly | Repository published as CC0. The data is scraped from Transfermarkt, so treat reuse beyond a non-commercial portfolio project with care. | value model, values everywhere |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) | A-League 2024/25 season aggregates from broadcast tracking: physical, off-ball runs, passing | MIT; credit SkillCorner | similarity search, player profiles |
| Understat, FBref (optional) | xG, xA, shots for the Big 5 | site terms; check before commercial use | optional extra model features |

Market values are Transfermarkt's crowd estimates, not transfer fees.

## How the model is evaluated

`train_model` is one command and writes everything a reviewer needs into `MODEL_DIR`:
`<version>.json` (metrics, training years, data summary, package versions, seed) and `<version>_holdout.csv`
(every held-out player with actual, predicted and baseline values).

- **Target:** market value `--horizon` years (default 2) after a season ends (30 June), learned as log(future / current).
- **Time split:** test on one season the model never saw (default: the newest one with a known outcome).
- **Embargo:** a training row's label is a value 2 years after its season. Training only uses seasons whose labels
  were known by the test season, so no training label comes from the test period. (Without the embargo, a model
  tested on 2023 would learn from 2022 rows labelled with 2024 values.) `--no-embargo` exists only for comparison.
- **No look-ahead in features:** the "current value" must be dated on or before the season's end; a valuation from
  the following summer would already reflect transfers. Contract length is not a feature for the same reason.
- **Three baselines:** value stays the same; everyone grows by the training median; players grow like others of
  their age. The command warns if the model doesn't beat all three.
- **Segments:** error by league, position, value band and confidence band, so weak spots are visible.
- **Uncertainty:** an 80% range (10th-90th percentile of held-out errors), set on half the test rows and checked
  on the other half; the checked coverage is reported. Predictions store this range.

### Results

Fill this in from your own run (`cat models/latest.json`), so every number you quote is reproducible:

| | Value |
|---|---|
| Data | A-League Men, seasons …, … labelled player-seasons |
| Test season / training seasons | … / … |
| MAE (model) | €…M |
| MAE: no change / median growth / age curve | €…M / €…M / €…M |
| 80% range coverage (checked) | …% |

## Similarity search

`GET /api/players/<id>/similar/` compares a player with others **in the same position group and season**
(midfielders = DM/CM/AM). Features: per-90 goals and assists (plus xG etc. where available) and SkillCorner's
physical, running and passing metrics. Each is turned into a z-score within the pool; the distance is the
root-mean-square gap over shared stats and `similarity = 100 / (1 + distance)`. Every result lists the three stats
that are closest and the two that differ most, with both players' numbers. A stat is only used if at least 70% of
the pool has it. Similarity describes style and output, not quality, tactical fit or price.

## Signals, risk and confidence

`generate_predictions` adds a signal (Strong buy / Undervalued / Breakout / Monitor / Overvalued), a 0-100 risk
score with its parts (`riskFactors`: playing time, age, contract, value volatility, model error) and a confidence
score. Thresholds live at the top of `players/ml/signals.py`. **Risk and confidence are transparent rules, not
calibrated probabilities**; the "confidence band" segment in the metrics shows whether higher confidence actually
came with smaller errors.

## Monitoring and deployment checks

- Every API request logs method, path, status and duration to stdout, and adds a `Server-Timing` header.
  Requests over 1 s or with a 5xx status are logged as warnings.
- `scripts/smoke_test.py --api https://<api>/api --origin https://<frontend>` checks health, the player list,
  CORS for the frontend's origin and that the frontend loads. The **Deployment smoke test** GitHub workflow runs
  it on demand, or daily once the repository variables `API_URL` and `FRONTEND_ORIGIN` are set.
- **CI** (`.github/workflows/ci.yml`) runs Django checks, a missing-migration check and the test suite on every push.

## Limitations

- One league is a small sample (a few hundred players a season), so the value model's errors are wide; check the
  baselines before trusting a forecast.
- SkillCorner's open data covers one season (2024/25), so its metrics power similarity but not the value model.
- `pos_*` features use a player's current position for past seasons too (documented, minor leak).
- The ROI simulator assumes the fee equals current market value unless you enter one, and values the player at the
  forecast even if the holding period runs past his contract.

## Optional: FBref / Understat / name-matched sources

`import_fbref`, `import_transfermarkt` (name + birth-year matching), `collect_understat` and `import_understat`
remain available. FBref removed its advanced (Opta) stats in January 2026; check `import_fbref --show-columns`.
For an xG ablation: `train_model --exclude xg90 xag90 npxg90 xgchain90 xgbuildup90 key_passes90 shots90 --name no-xg`.

## Tests

```bash
python manage.py test     # SQLite, no network; covers API, importers, leakage, embargo, similarity ranking
```
