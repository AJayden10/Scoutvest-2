"""Train an XGBoost model on past seasons and test it on a later one, honestly.

Evaluation rules (see README "How the model is evaluated"):
- Time split: the test set is one anchor year the model never saw.
- Embargo: a training row's label is the value `horizon` years after its anchor. Training rows are only
  kept if that label was already known at the test anchor, so no training label comes from the test
  period. Without this, a 2-year model tested on 2023 would train on 2022 rows labelled with 2024 values.
- Baselines: "value stays the same", "everyone grows by the training median" and "players grow like
  others of the same age". A model that cannot beat these is not adding anything.
- Uncertainty: a 10th-90th percentile range from held-out errors, with its coverage measured on rows
  that were not used to set it.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from . import signals
from .features import FEATURES, labelled, log_ratio

MIN_TRAIN_ROWS = 30
MIN_TEST_ROWS = 10
MIN_SEGMENT_ROWS = 5
INTERVAL_QUANTILES = (0.10, 0.90)
VALUE_BANDS = [(0, 1, "under €1M"), (1, 5, "€1-5M"), (5, 20, "€5-20M"), (20, float("inf"), "€20M+")]


class NotEnoughData(Exception):
    pass


@dataclass
class TrainResult:
    model: XGBRegressor
    metrics: dict
    importances: dict
    features: list
    holdout: pd.DataFrame = field(default_factory=pd.DataFrame)


def _params(fast: bool, seed: int = 42):
    return dict(
        n_estimators=60 if fast else 400, max_depth=3 if fast else 4, learning_rate=0.08 if fast else 0.04,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=2.0,
        objective="reg:squarederror", random_state=seed, n_jobs=1,
    )


def split_by_time(df: pd.DataFrame, test_year: int | None = None, horizon_years: int = 2, embargo: bool = True):
    """Test on one anchor year (default: the newest labelled one); train on earlier anchors whose labels
    were known by then. Returns (train, test, last_train_year, test_year)."""
    years = sorted(df["anchor_year"].unique())
    if len(years) < 2:
        raise NotEnoughData("need labelled seasons from at least two different years to do a time split")
    test_year = int(test_year if test_year is not None else years[-1])
    last_train = test_year - (horizon_years if embargo else 1)
    train, test = df[df["anchor_year"] <= last_train], df[df["anchor_year"] == test_year]
    if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
        hint = " (the embargo needs labelled seasons at least `horizon` years before the test year)" if embargo else ""
        raise NotEnoughData(f"only {len(train)} training rows up to {last_train} and {len(test)} test rows in "
                            f"{test_year} (need {MIN_TRAIN_ROWS}/{MIN_TEST_ROWS}){hint}")
    return train, test, int(last_train), test_year


def _age_curve(train: pd.DataFrame):
    """Median log growth per age bucket in the training rows; used as a baseline."""
    ratios = log_ratio(train)
    buckets = train["age"].map(signals.age_bucket)
    by_bucket = ratios.groupby(buckets).median().to_dict()
    overall = float(ratios.median())
    return lambda rows: rows["age"].map(signals.age_bucket).map(by_bucket).fillna(overall).astype(float)


def _value_band(v: float) -> str:
    for lo, hi, label in VALUE_BANDS:
        if lo <= v < hi:
            return label
    return VALUE_BANDS[-1][2]


def _segments(h: pd.DataFrame, column: str) -> list[dict]:
    out = []
    for name, g in h.groupby(column):
        if len(g) < MIN_SEGMENT_ROWS or name == "":
            continue
        out.append({
            "segment": str(name), "n": int(len(g)),
            "mae": round(float(mean_absolute_error(g["actual"], g["predicted"])), 3),
            "baseline_mae": round(float(mean_absolute_error(g["actual"], g["baseline_no_change"])), 3),
            "median_abs_pct_error": round(float(np.median(np.abs(g["predicted"] - g["actual"]) / g["actual"]) * 100), 1),
        })
    return sorted(out, key=lambda s: -s["n"])


def _interval(log_resid: np.ndarray, seed: int) -> dict:
    """10th-90th percentile of log(actual / predicted), set on one half of the test rows and checked on the
    other half, so the reported coverage is not true by construction."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(log_resid))
    a, b = log_resid[idx[: len(idx) // 2]], log_resid[idx[len(idx) // 2:]]
    lo, hi = np.quantile(a, INTERVAL_QUANTILES)
    coverage = float(np.mean((b >= lo) & (b <= hi))) if len(b) else None
    full_lo, full_hi = np.quantile(log_resid, INTERVAL_QUANTILES)  # what serving uses
    return {"quantiles": list(INTERVAL_QUANTILES), "log_low": round(float(full_lo), 4), "log_high": round(float(full_hi), 4),
            "nominal_coverage": round(INTERVAL_QUANTILES[1] - INTERVAL_QUANTILES[0], 2),
            "checked_coverage": None if coverage is None else round(coverage, 3), "checked_on_rows": int(len(b))}


def train(df: pd.DataFrame, test_year: int | None = None, fast: bool = False, exclude=(),
          horizon_years: int = 2, embargo: bool = True, seed: int = 42) -> TrainResult:
    features = [f for f in FEATURES if f not in set(exclude)]
    data = labelled(df)
    tr, te, last_train, test_year = split_by_time(data, test_year, horizon_years, embargo)
    model = XGBRegressor(**_params(fast, seed))
    model.fit(tr[features], log_ratio(tr))

    pred = te["current_value"] * np.exp(model.predict(te[features]))
    actual = te["target_value"]
    no_change = te["current_value"]
    median_growth = te["current_value"] * float(np.exp(log_ratio(tr).median()))
    age_curve = te["current_value"] * np.exp(_age_curve(tr)(te))

    mae = mean_absolute_error(actual, pred)
    baselines = {
        "no_change": round(float(mean_absolute_error(actual, no_change)), 3),
        "median_growth": round(float(mean_absolute_error(actual, median_growth)), 3),
        "age_curve": round(float(mean_absolute_error(actual, age_curve)), 3),
    }
    log_err = np.abs(np.log(pred / actual))
    mle = float(log_err.mean())

    holdout = pd.DataFrame({
        "player_id": te["player_id"].to_numpy(), "season": te["season"].to_numpy(), "league": te["league"].to_numpy(),
        "position": te["position"].to_numpy() if "position" in te else "", "age": te["age"].to_numpy(),
        "minutes": te["minutes"].to_numpy(), "current_value": te["current_value"].to_numpy(),
        "actual": actual.to_numpy(), "predicted": pred.to_numpy(), "baseline_no_change": no_change.to_numpy(),
        "baseline_age_curve": age_curve.to_numpy(),
    })
    holdout["value_band"] = holdout["current_value"].map(_value_band)
    holdout["confidence"] = [signals.confidence(minutes=m, mean_abs_log_error=mle) for m in holdout["minutes"]]
    holdout["confidence_band"] = pd.cut(holdout["confidence"], [0, 50, 70, 101], right=False,
                                        labels=["under 50", "50-69", "70+"]).astype(str)

    metrics = {
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "train_years": sorted(int(y) for y in tr["anchor_year"].unique()), "test_year": test_year,
        "cutoff_year": last_train, "horizon_years": horizon_years, "embargo": embargo,
        "mae": round(float(mae), 3), "rmse": round(float(np.sqrt(np.mean((actual - pred) ** 2))), 3),
        "r2": round(float(r2_score(actual, pred)), 3),
        "median_abs_pct_error": round(float(np.median(np.abs(pred - actual) / actual) * 100), 1),
        "mean_abs_log_error": round(mle, 3),
        "baseline_mae": baselines["no_change"],
        "beats_baseline": bool(mae < baselines["no_change"]),
        "baselines": baselines,
        "beats_all_baselines": bool(all(mae < b for b in baselines.values())),
        "interval": _interval(np.log(actual.to_numpy() / pred.to_numpy()), seed),
        "segments": {
            "league": _segments(holdout, "league"), "position": _segments(holdout, "position"),
            "value_band": _segments(holdout, "value_band"),
            # If "confidence" means anything, error should fall as confidence rises.
            "confidence_band": _segments(holdout, "confidence_band"),
        },
    }
    # Final model uses every labelled row; the metrics above come from the honest time split.
    final = XGBRegressor(**_params(fast, seed))
    final.fit(data[features], log_ratio(data))
    imp = dict(sorted(zip(features, map(float, final.feature_importances_)), key=lambda kv: -kv[1]))
    return TrainResult(final, metrics, imp, features, holdout)


def _environment() -> dict:
    import sklearn
    import xgboost
    return {"python": platform.python_version(), "xgboost": xgboost.__version__, "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


def save(result: TrainResult, directory: Path, version: str, horizon_years: int, data_summary: dict | None = None,
         seed: int = 42) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{version}.joblib"
    meta = {"version": version, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "horizon_years": horizon_years, "seed": seed, "features": result.features,
            "metrics": result.metrics, "importances": result.importances,
            "data": data_summary or {}, "environment": _environment()}
    joblib.dump({"model": result.model, "meta": meta}, path)
    (directory / f"{version}.json").write_text(json.dumps(meta, indent=2))
    if not result.holdout.empty:
        result.holdout.to_csv(directory / f"{version}_holdout.csv", index=False)
    (directory / "latest.json").write_text(json.dumps(meta, indent=2))
    (directory / "latest.txt").write_text(version)
    return path


def load_latest(directory: Path):
    pointer = directory / "latest.txt"
    if not pointer.exists():
        raise FileNotFoundError(f"no trained model in {directory}; run `manage.py train_model` first")
    bundle = joblib.load(directory / f"{pointer.read_text().strip()}.joblib")
    return bundle["model"], bundle["meta"]
