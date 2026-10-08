"""Turn the raw FBref "standard stats" DataFrame (as saved by scripts/collect_stats.py)
into normalized records ready for the database.

ScraperFC/FBref column names vary a little between versions (sometimes
"Gls", sometimes "Performance Gls" or "Performance_Gls"), so columns are matched
on a normalized form and the importer reports clearly what it could not find.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# normalized target -> candidate normalized source column names, in priority order
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "name": ["player"],
    "nation": ["nation"],
    "pos": ["pos"],
    "club": ["squad", "team"],
    "age": ["age"],
    "born": ["born"],
    "matches": ["mp", "matchesplayed", "playingtimemp"],
    "starts": ["starts", "playingtimestarts"],
    "minutes": ["min", "minutes", "playingtimemin"],
    "goals": ["gls", "goals", "performancegls"],
    "assists": ["ast", "assists", "performanceast"],
    "xg": ["xg", "expectedxg"],
    "xag": ["xag", "expectedxag"],
    "prog_carries": ["prgc", "progressionprgc"],
    "prog_passes": ["prgp", "progressionprgp"],
    "season": ["season"],
}
REQUIRED = ["name", "club", "season", "minutes"]

# FBref gives coarse positions (GK, DF, MF, FW and combinations). This is a
# first approximation to the frontend's position set; Transfermarkt's detailed
# positions can refine it later (CB vs FB, CM vs DM, ST vs Winger).
POSITION_MAP = {
    "GK": "GK",
    "DF": "CB",
    "MF": "CM",
    "FW": "ST",
    "DF,MF": "DM",
    "MF,DF": "DM",
    "MF,FW": "AM",
    "FW,MF": "AM",
    "DF,FW": "FB",
    "FW,DF": "FB",
}


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


class MissingColumns(Exception):
    pass


@dataclass
class StatRecord:
    name: str
    club: str
    season: str
    nationality: str = ""
    position: str = ""
    age: int | None = None
    birth_year: int | None = None
    matches: int = 0
    starts: int = 0
    minutes: int = 0
    goals: float = 0.0
    assists: float = 0.0
    xg: float | None = None
    xag: float | None = None
    prog_passes: float | None = None
    prog_carries: float | None = None
    extra: dict = field(default_factory=dict)


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map target field -> actual DataFrame column. Raises MissingColumns for required gaps."""
    by_norm: dict[str, str] = {}
    for col in df.columns:
        by_norm.setdefault(_norm(col), col)  # first occurrence wins (totals before per-90 duplicates)
    mapping = {}
    for target, candidates in COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand in by_norm:
                mapping[target] = by_norm[cand]
                break
    missing = [t for t in REQUIRED if t not in mapping]
    if missing:
        raise MissingColumns(
            f"Could not find columns for {missing}. Columns present: {list(df.columns)}"
        )
    return mapping


def _num(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _int(value, default=None):
    n = _num(value)
    return int(n) if n is not None else default


def _age(value):
    """FBref ages look like 24 or '24-153' (years-days)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return _int(str(value).split("-")[0])


def normalize_position(pos) -> str:
    if pos is None or (not isinstance(pos, str) and pd.isna(pos)):
        return ""
    return POSITION_MAP.get(str(pos).replace(" ", "").upper(), "")


def iter_records(df: pd.DataFrame):
    """Yield StatRecords, skipping blank/repeated-header rows."""
    cols = resolve_columns(df)

    def get(row, key):
        return row[cols[key]] if key in cols else None

    for _, row in df.iterrows():
        name = get(row, "name")
        club = get(row, "club")
        season = get(row, "season")
        if not isinstance(name, str) or not name.strip() or name.strip().lower() == "player":
            continue
        if not isinstance(club, str) or not club.strip():
            continue
        nation = get(row, "nation")
        # FBref nation is like "eng ENG"; keep the readable code part.
        nationality = nation.split()[-1] if isinstance(nation, str) and nation.strip() else ""
        yield StatRecord(
            name=name.strip(),
            club=club.strip(),
            season=str(season),
            nationality=nationality,
            position=normalize_position(get(row, "pos")),
            age=_age(get(row, "age")),
            birth_year=_int(get(row, "born")),
            matches=_int(get(row, "matches"), 0),
            starts=_int(get(row, "starts"), 0),
            minutes=_int(get(row, "minutes"), 0),
            goals=_num(get(row, "goals"), 0.0),
            assists=_num(get(row, "assists"), 0.0),
            xg=_num(get(row, "xg")),
            xag=_num(get(row, "xag")),
            prog_passes=_num(get(row, "prog_passes")),
            prog_carries=_num(get(row, "prog_carries")),
        )
