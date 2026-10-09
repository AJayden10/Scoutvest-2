"""Turn a raw prediction into what the app shows: signal, risk, confidence and reasons.

All rules are plain functions with named thresholds so they are easy to read, test and tune.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES

STRONG_BUY_UPSIDE = 60.0    # % upside
UNDERVALUED_UPSIDE = 35.0
BREAKOUT_UPSIDE = 30.0
BREAKOUT_MAX_AGE = 21
OVERVALUED_GAP = 15.0       # points below what players of the same age are predicted to do
STRONG_BUY_MAX_RISK = 45

AGE_BUCKETS = [(21, "u21"), (24, "22-24"), (27, "25-27"), (30, "28-30"), (99, "31+")]


def age_bucket(age) -> str:
    if age is None or age != age:
        return "unknown"
    for top, name in AGE_BUCKETS:
        if age <= top:
            return name
    return "31+"


def upside_pct(current: float, predicted: float) -> float:
    return (predicted - current) / current * 100.0


def risk_score(*, minutes, age, contract_years_left, value_growth_12m, mean_abs_log_error) -> int:
    """0-100, higher is riskier. Adds up: thin playing time, age, expiring contract, a volatile
    value history and how wrong the model usually is."""
    score = 10.0
    score += 25 * max(0.0, min(1.0, (1500 - (minutes or 0)) / 1200))          # few minutes -> noisy stats
    if age is not None and not np.isnan(age):
        score += 12 if age >= 30 else 6 if age >= 28 else 0
    if contract_years_left is not None and not np.isnan(contract_years_left):
        score += 15 if contract_years_left <= 1 else 6 if contract_years_left <= 2 else 0
    if value_growth_12m is not None and not np.isnan(value_growth_12m):
        score += min(15.0, abs(value_growth_12m) * 20)                          # big swings = volatile
    score += min(15.0, (mean_abs_log_error or 0.5) * 25)                        # model uncertainty
    return int(round(max(0, min(100, score))))


def confidence(*, minutes, mean_abs_log_error) -> int:
    """0-100: how much to trust the number. Falls with model error and thin playing time."""
    base = 100 * (1 - min(0.7, (mean_abs_log_error or 0.5)))
    base *= 0.6 + 0.4 * max(0.0, min(1.0, (minutes or 0) / 1800))
    return int(round(max(10, min(95, base))))


def classify(*, upside, age, risk, peer_median=0.0) -> str:
    """`peer_median` is the median predicted upside of players the same age. Values naturally drift
    down after the late twenties, so "overvalued" means doing clearly worse than that drift."""
    if upside < 0 and upside <= peer_median - OVERVALUED_GAP:
        return "OVERVALUED"
    if upside >= STRONG_BUY_UPSIDE and risk <= STRONG_BUY_MAX_RISK:
        return "STRONG BUY"
    if age is not None and not np.isnan(age) and age <= BREAKOUT_MAX_AGE and upside >= BREAKOUT_UPSIDE:
        return "BREAKOUT"
    if upside >= UNDERVALUED_UPSIDE:
        return "UNDERVALUED"
    return "MONITOR"


_UP = {
    "age": "Young for his level, with room to grow", "xg90": "Strong expected goals per 90",
    "xag90": "Creates plenty of chances (expected assists per 90)", "goals90": "Productive in front of goal",
    "assists90": "Strong assist output", "prog_passes90": "Moves the ball forward with progressive passing",
    "prog_carries90": "Drives the team forward with progressive carries", "value_growth_12m": "Market value is already rising",
    "log_value": "Current value is low relative to output", "minutes": "Plays regularly",
    "log_minutes": "Plays regularly", "npxg90": "Gets into good scoring positions (non-penalty xG per 90)",
    "xgchain90": "Involved in many of his team's attacking moves", "xgbuildup90": "Strong in build-up play",
    "key_passes90": "Creates chances with key passes", "shots90": "Takes plenty of shots", "contract_years_left": "Contract situation supports a move",
}
_DOWN = {
    "age": "Age limits future growth", "log_value": "Already priced high for his output",
    "value_growth_12m": "Recent value trend is weak", "minutes": "Limited playing time", "log_minutes": "Limited playing time",
}


def reasons(model, row: pd.DataFrame, top: int = 3, features=None) -> list[str]:
    """Plain-English reasons from the features that pushed this prediction up or down most."""
    from xgboost import DMatrix

    features = features or FEATURES
    contribs = model.get_booster().predict(DMatrix(row[features]), pred_contribs=True)[0][:-1]
    order = np.argsort(-np.abs(contribs))
    out = []
    for i in order:
        name, c = features[i], contribs[i]
        text = (_UP if c > 0 else _DOWN).get(name)
        if text and text not in out:
            out.append(text)
        if len(out) >= top:
            break
    return out
