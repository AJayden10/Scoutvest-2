"""Build everything from the open transfermarkt-datasets tables (players, clubs, appearances,
player_valuations). One source means one set of player ids, so no name matching is needed.

Stats come from per-match rows: goals, assists, minutes and cards per player-club-season.
There is no xG or progressive-pass data here; those columns stay empty unless another importer fills them.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1", "NL1", "PO1", "BE1", "TR1", "SC1"]
LEAGUE_NAMES = {
    "GB1": ("Premier League", "England"), "ES1": ("La Liga", "Spain"), "IT1": ("Serie A", "Italy"),
    "L1": ("Bundesliga", "Germany"), "FR1": ("Ligue 1", "France"), "NL1": ("Eredivisie", "Netherlands"),
    "PO1": ("Primeira Liga", "Portugal"), "BE1": ("Jupiler Pro League", "Belgium"),
    "TR1": ("Süper Lig", "Turkey"), "SC1": ("Scottish Premiership", "Scotland"),
    "RU1": ("Russian Premier League", "Russia"), "GB2": ("Championship", "England"),
    "ES2": ("Segunda División", "Spain"), "IT2": ("Serie B", "Italy"), "L2": ("2. Bundesliga", "Germany"),
    "FR2": ("Ligue 2", "France"), "DK1": ("Superliga", "Denmark"), "GR1": ("Super League 1", "Greece"),
    "UKR1": ("Ukrainian Premier League", "Ukraine"), "A1": ("Bundesliga (Austria)", "Austria"),
    "C1": ("Super League (Switzerland)", "Switzerland"), "TS1": ("Fortuna Liga", "Czechia"),
}

SUB_POSITION = {
    "Goalkeeper": "GK", "Centre-Back": "CB", "Left-Back": "FB", "Right-Back": "FB",
    "Defensive Midfield": "DM", "Central Midfield": "CM", "Attacking Midfield": "AM",
    "Left Midfield": "Winger", "Right Midfield": "Winger", "Left Winger": "Winger", "Right Winger": "Winger",
    "Second Striker": "ST", "Centre-Forward": "ST",
}
POSITION_FALLBACK = {"Goalkeeper": "GK", "Defender": "CB", "Midfield": "CM", "Attack": "ST"}


def map_position(sub_position, position) -> str:
    if isinstance(sub_position, str) and sub_position in SUB_POSITION:
        return SUB_POSITION[sub_position]
    return POSITION_FALLBACK.get(position, "") if isinstance(position, str) else ""


def season_label(date: pd.Series) -> pd.Series:
    """Season runs July-June: a match on 2023-09-02 or 2024-03-10 is '2023-2024'."""
    start = date.dt.year.where(date.dt.month >= 7, date.dt.year - 1)
    return start.astype(str) + "-" + (start + 1).astype(str)


def league_info(league_id: str, competitions: pd.DataFrame | None = None):
    if league_id in LEAGUE_NAMES:
        return LEAGUE_NAMES[league_id]
    if competitions is not None:
        row = competitions[competitions["competition_id"] == league_id]
        if len(row):
            r = row.iloc[0]
            country = r["country_name"] if isinstance(r.get("country_name"), str) else ""
            return str(r["name"]).replace("-", " ").title(), country
    return league_id, ""


def season_stats(appearances: pd.DataFrame, club_league: dict, leagues: list[str], since_season: int) -> pd.DataFrame:
    """Aggregate match rows to one row per (player, club, season) for clubs in the chosen leagues."""
    a = appearances.copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a = a.dropna(subset=["date"])
    a["league"] = a["player_club_id"].map(club_league)
    a = a[a["league"].isin(leagues)]
    a["season"] = season_label(a["date"])
    a = a[a["season"].str[:4].astype(int) >= since_season]
    g = a.groupby(["player_id", "player_club_id", "season"], as_index=False).agg(
        matches=("game_id", "nunique"), starts=("minutes_played", lambda s: int((s >= 45).sum())),
        minutes=("minutes_played", "sum"), goals=("goals", "sum"), assists=("assists", "sum"),
        yellow_cards=("yellow_cards", "sum"), red_cards=("red_cards", "sum"),
    )
    return g
