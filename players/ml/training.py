"""Train an XGBoost model on past seasons and test it on later ones."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from .features import FEATURES, labelled, log_ratio

MIN_TRAIN_ROWS = 30
MIN_TEST_ROWS = 10


class NotEnoughData(Exception):
    pass


@dataclass
class TrainResult:
    model: XGBRegressor
    metrics: dict
    importances: dict


def _params(fast: bool):
    return dict(
        n_estimators=60 if fast else 400, max_depth=3 if fast else 4, learning_rate=0.08 if fast else 0.04,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=2.0,
        objective="reg:squarederror", random_state=42, n_jobs=1,
    )


def split_by_time(df: pd.DataFrame, cutoff_year: int | None = None):
    """Train on anchors up to the cutoff year, test on later ones. Default cutoff keeps the most
    recent labelled anchor year as the test set, so the model is judged on seasons it never saw."""
    years = sorted(df["anchor_year"].unique())
    if len(years) < 2:
        raise NotEnoughData("need labelled seasons from at least two different years to do a time split")
    cutoff = cutoff_year if cutoff_year is not None else years[-2]
    train, test = df[df["anchor_year"] <= cutoff], df[df["anchor_year"] > cutoff]
    if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
        raise NotEnoughData(f"only {len(train)} training and {len(test)} test rows (need {MIN_TRAIN_ROWS}/{MIN_TEST_ROWS})")
    return train, test, int(cutoff)


def train(df: pd.DataFrame, cutoff_year: int | None = None, fast: bool = False) -> TrainResult:
    data = labelled(df)
    tr, te, cutoff = split_by_time(data, cutoff_year)
    model = XGBRegressor(**_params(fast))
    model.fit(tr[FEATURES], log_ratio(tr))

    pred = te["current_value"] * np.exp(model.predict(te[FEATURES]))
    actual = te["target_value"]
    naive = te["current_value"]  # "value stays the same"
    mae, naive_mae = mean_absolute_error(actual, pred), mean_absolute_error(actual, naive)
    log_err = np.abs(np.log(pred / actual))
    metrics = {
        "train_rows": int(len(tr)), "test_rows": int(len(te)), "cutoff_year": cutoff,
        "mae": round(float(mae), 3), "rmse": round(float(np.sqrt(np.mean((actual - pred) ** 2))), 3),
        "r2": round(float(r2_score(actual, pred)), 3),
        "median_abs_pct_error": round(float(np.median(np.abs(pred - actual) / actual) * 100), 1),
        "mean_abs_log_error": round(float(log_err.mean()), 3),
        "baseline_mae": round(float(naive_mae), 3),
        "beats_baseline": bool(mae < naive_mae),
    }
    # Final model uses every labelled row; the metrics above come from the honest time split.
    final = XGBRegressor(**_params(fast))
    final.fit(data[FEATURES], log_ratio(data))
    imp = dict(sorted(zip(FEATURES, map(float, final.feature_importances_)), key=lambda kv: -kv[1]))
    return TrainResult(final, metrics, imp)


def save(result: TrainResult, directory: Path, version: str, horizon_years: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{version}.joblib"
    meta = {"version": version, "horizon_years": horizon_years, "features": FEATURES,
            "metrics": result.metrics, "importances": result.importances}
    joblib.dump({"model": result.model, "meta": meta}, path)
    (directory / f"{version}.json").write_text(json.dumps(meta, indent=2))
    (directory / "latest.json").write_text(json.dumps(meta, indent=2))
    (directory / "latest.txt").write_text(version)
    return path


def load_latest(directory: Path):
    pointer = directory / "latest.txt"
    if not pointer.exists():
        raise FileNotFoundError(f"no trained model in {directory}; run `manage.py train_model` first")
    bundle = joblib.load(directory / f"{pointer.read_text().strip()}.joblib")
    return bundle["model"], bundle["meta"]
