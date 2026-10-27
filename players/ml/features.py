"""Turn the database into one training/prediction row per player-season.

Each row is a snapshot at the END of a season (the "anchor", 30 June of the season's second
year). Features only use information available at that date. The label is the player's market
value `horizon_years` later, so a model trained on past anchors can be asked about today's.
"""
from __future__ import annotations

import bisect
from datetime import date, timedelta

import numpy as np
import pandas as pd

from players.models import MarketValue, Player, PlayerSeasonStat, Position

NUMERIC_FEATURES = [
    "age", "minutes", "log_minutes", "goals90", "assists90", "xg90", "xag90",
    "prog_passes90", "prog_carries90", "npxg90", "xgchain90", "xgbuildup90", "key_passes90", "shots90",
    "log_value", "value_growth_12m",
]
# contract_years_left is computed per row (the risk score uses it) but is NOT a model feature: the
# source only knows each player's CURRENT contract, so for past seasons it would leak the future.
# Known, accepted leak: `pos_*` uses the player's current position, not the position he had in that
# season. Positions rarely change, so this is documented rather than modelled (see README "Limitations").
POSITIONS = [p.value for p in Position]
FEATURES = NUMERIC_FEATURES + [f"pos_{p}" for p in POSITIONS]

VALUE_TOLERANCE_DAYS = 120  # how far a valuation may be from the date we want


def season_end(season: str) -> date:
    """'2023-2024' -> 2024-06-30."""
    return date(int(season.split("-")[1]), 6, 30)


def _nearest(dates, values, target: date, tol_days: int, *, not_after: date | None = None):
    """Value whose date is closest to `target` (within tolerance), or None."""
    if not dates:
        return None
    i = bisect.bisect_left(dates, target)
    best, best_gap = None, None
    for j in (i - 1, i):
        if 0 <= j < len(dates):
            if not_after and dates[j] > not_after:
                continue
            gap = abs((dates[j] - target).days)
            if gap <= tol_days and (best_gap is None or gap < best_gap):
                best, best_gap = values[j], gap
    return best


def _per90(value, minutes):
    if value is None or not minutes:
        return np.nan  # unknown stays unknown (e.g. no xG in the source), it is not zero
    return value * 90.0 / minutes


def build_frame(horizon_years: int = 2, min_minutes: int = 450) -> pd.DataFrame:
    """One row per (player, season). `target_value` is NaN when the future value is unknown
    (which is exactly the case for the newest season we want to predict)."""
    rows = []
    stats = PlayerSeasonStat.objects.select_related("player", "club").filter(minutes__gte=min_minutes)
    values: dict[int, tuple[list, list]] = {}
    for pid, d, v in MarketValue.objects.order_by("player_id", "date").values_list("player_id", "date", "value"):
        ds, vs = values.setdefault(pid, ([], []))
        ds.append(d)
        vs.append(v)

    for s in stats:
        p: Player = s.player
        anchor = season_end(s.season)
        ds, vs = values.get(p.id, ([], []))
        # The "current" value must be known AT the anchor: a valuation from the following summer would
        # already reflect transfers and form after the season ended (look-ahead leakage).
        current = _nearest(ds, vs, anchor, VALUE_TOLERANCE_DAYS, not_after=anchor)
        if current is None or current <= 0:
            continue
        past = _nearest(ds, vs, anchor - timedelta(days=365), VALUE_TOLERANCE_DAYS, not_after=anchor)
        future_date = date(anchor.year + horizon_years, anchor.month, anchor.day)
        future = _nearest(ds, vs, future_date, VALUE_TOLERANCE_DAYS)
        age = s.age if s.age is not None else (anchor.year - p.birth_year if p.birth_year else np.nan)
        row = {
            "player_id": p.id, "season": s.season, "anchor": anchor, "anchor_year": anchor.year,
            "league": s.club.league if s.club else "", "position": p.position or "",
            "age": age, "minutes": s.minutes, "log_minutes": np.log1p(s.minutes),
            "goals90": _per90(s.goals, s.minutes), "assists90": _per90(s.assists, s.minutes),
            "xg90": _per90(s.xg, s.minutes), "xag90": _per90(s.xag, s.minutes),
            "npxg90": _per90(s.extra.get("npxg"), s.minutes), "xgchain90": _per90(s.extra.get("xgchain"), s.minutes),
            "xgbuildup90": _per90(s.extra.get("xgbuildup"), s.minutes),
            "key_passes90": _per90(s.extra.get("key_passes"), s.minutes), "shots90": _per90(s.extra.get("shots"), s.minutes),
            "prog_passes90": _per90(s.progressive_passes, s.minutes),
            "prog_carries90": _per90(s.progressive_carries, s.minutes),
            "current_value": current, "log_value": np.log(current),
            "value_growth_12m": np.log(current / past) if past and past > 0 else np.nan,
            "contract_years_left": (p.contract_expiry_year - anchor.year) if p.contract_expiry_year else np.nan,
            "target_value": future if (future and future > 0) else np.nan,
        }
        for pos in POSITIONS:
            row[f"pos_{pos}"] = 1.0 if p.position == pos else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["target_value"]).copy()


def log_ratio(df: pd.DataFrame) -> pd.Series:
    """The quantity the model learns: log(future value / current value)."""
    return np.log(df["target_value"] / df["current_value"])
