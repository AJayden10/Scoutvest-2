"""Transfermarkt market values -> our MarketValue rows.

Source: the open "transfermarkt-datasets" CSVs (players.csv + player_valuations.csv),
not live scraping, so there is nothing to rate-limit and re-runs are cheap. Column names are
resolved loosely, the same way as the FBref importer, so small schema changes do not break it.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import pandas as pd

PLAYER_COLUMNS = {
    "tm_id": ["player_id", "id", "transfermarkt_id"],
    "name": ["name", "player_name", "player"],
    "birth": ["date_of_birth", "dob", "born", "birth_date"],
    "contract": ["contract_expiration_date", "contract_expires", "contract_expiry"],
}
VALUATION_COLUMNS = {
    "tm_id": ["player_id", "id", "transfermarkt_id"],
    "date": ["date", "datetime", "valuation_date"],
    "value": ["market_value_in_eur", "market_value", "value_eur", "value"],
}


class MissingColumns(Exception):
    pass


def _norm_col(c) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def resolve(df: pd.DataFrame, wanted: dict[str, list[str]], required: list[str]) -> dict[str, str]:
    cols = {_norm_col(c): c for c in df.columns}
    out = {}
    for key, options in wanted.items():
        for o in options:
            if o in cols:
                out[key] = cols[o]
                break
    missing = [k for k in required if k not in out]
    if missing:
        raise MissingColumns(f"could not find {missing} in columns {list(df.columns)}")
    return out


def normalize_name(name) -> str:
    """Lower-case, accents and punctuation removed: 'Rémi Delacroix' -> 'remi delacroix'."""
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", "", s.lower().replace("-", " ")).strip()


def _year(value):
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else int(ts.year)


def to_millions(value):
    """Euro amount -> €M, or None when missing or non-positive."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v <= 0:
        return None
    return round(v / 1_000_000, 3)


@dataclass
class TMPlayer:
    tm_id: str
    name: str
    birth_year: int | None
    contract_expiry_year: int | None


def iter_tm_players(df: pd.DataFrame):
    m = resolve(df, PLAYER_COLUMNS, required=["tm_id", "name"])
    for _, row in df.iterrows():
        if pd.isna(row[m["tm_id"]]) or pd.isna(row[m["name"]]):
            continue
        yield TMPlayer(
            tm_id=str(int(row[m["tm_id"]])) if str(row[m["tm_id"]]).replace(".0", "").isdigit() else str(row[m["tm_id"]]),
            name=str(row[m["name"]]),
            birth_year=_year(row[m["birth"]]) if "birth" in m else None,
            contract_expiry_year=_year(row[m["contract"]]) if "contract" in m else None,
        )


def build_index(tm_players):
    """normalized name -> list of TMPlayer, for matching our FBref players."""
    idx = defaultdict(list)
    for p in tm_players:
        idx[normalize_name(p.name)].append(p)
    return idx


def match_player(name: str, birth_year: int | None, index) -> TMPlayer | None:
    """Exact normalized-name match. With several candidates, birth year decides; if it cannot,
    the match is skipped rather than guessed (a wrong value is worse than a missing one)."""
    candidates = index.get(normalize_name(name), [])
    if len(candidates) == 1:
        c = candidates[0]
        if birth_year and c.birth_year and abs(c.birth_year - birth_year) > 1:
            return None
        return c
    if birth_year:
        close = [c for c in candidates if c.birth_year and abs(c.birth_year - birth_year) <= 1]
        if len(close) == 1:
            return close[0]
    return None


def iter_valuations(df: pd.DataFrame, wanted_ids: set[str], since: date | None = None):
    """Yield (tm_id, date, value_in_millions) for the players we matched."""
    m = resolve(df, VALUATION_COLUMNS, required=["tm_id", "date", "value"])
    for row in df[[m["tm_id"], m["date"], m["value"]]].itertuples(index=False):
        tid = str(int(row[0])) if str(row[0]).replace(".0", "").isdigit() else str(row[0])
        if tid not in wanted_ids:
            continue
        d = pd.to_datetime(row[1], errors="coerce")
        v = to_millions(row[2])
        if pd.isna(d) or v is None:
            continue
        d = d.date()
        if since and d < since:
            continue
        yield tid, d, v
