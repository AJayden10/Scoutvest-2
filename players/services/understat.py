"""Understat player-season stats (xG, xA, npxG, xGChain, xGBuildup, shots, key passes).

Understat covers the Big 5 leagues plus Russia from 2014/15. It has no player ids in common with
Transfermarkt, so rows are matched by normalised name, using minutes played to break ties.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

LEAGUES = {  # understat code -> league name used in our Club.league
    "EPL": "Premier League", "La_liga": "La Liga", "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A", "Ligue_1": "Ligue 1", "RFPL": "Russian Premier League",
}
USER_AGENT = "Mozilla/5.0 (compatible; FreeAgentResearch/1.0; personal project)"


def norm(name) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", "", s.lower().replace("-", " ")).strip()


def season_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_players(raw: list[dict]) -> list[dict]:
    """Understat returns every number as a string; turn one league-season into clean rows."""
    rows = []
    for p in raw:
        rows.append({
            "name": p.get("player_name", ""), "team": p.get("team_title", ""),
            "minutes": int(_num(p.get("time"))), "games": int(_num(p.get("games"))),
            "goals": _num(p.get("goals")), "assists": _num(p.get("assists")),
            "xg": _num(p.get("xG")), "xa": _num(p.get("xA")), "npxg": _num(p.get("npxG")),
            "shots": _num(p.get("shots")), "key_passes": _num(p.get("key_passes")),
            "xgchain": _num(p.get("xGChain")), "xgbuildup": _num(p.get("xGBuildup")),
        })
    return rows


def fetch_league_season(code: str, start_year: int, timeout: int = 30) -> list[dict]:
    """Ask Understat's own JSON endpoint (the one its league page calls) for one league-season."""
    data = urllib.parse.urlencode({"league": code, "season": str(start_year)}).encode()
    req = urllib.request.Request(
        "https://understat.com/main/getPlayersStats/", data=data,
        headers={"User-Agent": USER_AGENT, "X-Requested-With": "XMLHttpRequest",
                 "Referer": f"https://understat.com/league/{code}/{start_year}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    players = payload.get("players") if isinstance(payload, dict) else None
    if players is None:
        raise ValueError(f"unexpected response shape for {code} {start_year}: {str(payload)[:120]}")
    return players


def collect(out_dir: Path, codes: list[str], years: list[int], delay: float = 3.0, log=print) -> int:
    """Download league-seasons to JSON files, skipping ones already saved. Be polite: one request per `delay` seconds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for code in codes:
        for y in years:
            path = out_dir / f"{code}_{y}.json"
            if path.exists():
                continue
            try:
                players = fetch_league_season(code, y)
            except Exception as e:  # noqa: BLE001 - report and keep going
                log(f"  {code} {y}: failed ({e})")
                continue
            path.write_text(json.dumps(players))
            log(f"  {code} {y}: {len(players)} players")
            fetched += 1
            time.sleep(delay)
    return fetched


def match_rows(rows: list[dict], candidates: list[dict]):
    """Pair Understat rows with our players of the same league-season.

    `candidates`: dicts with id, name, minutes. Exact normalised name first; if several of ours share
    the name, the one whose minutes are closest wins (within 15%). Returns (pairs, unmatched_rows)."""
    by_name: dict[str, list[dict]] = {}
    for c in candidates:
        by_name.setdefault(norm(c["name"]), []).append(c)
    used, pairs, unmatched = set(), [], []
    for r in sorted(rows, key=lambda r: -r["minutes"]):
        options = [c for c in by_name.get(norm(r["name"]), []) if c["id"] not in used]
        if len(options) > 1:
            options.sort(key=lambda c: abs(c["minutes"] - r["minutes"]))
        if options and (len(options) == 1 and _close(options[0]["minutes"], r["minutes"], 0.35) or
                        _close(options[0]["minutes"], r["minutes"], 0.15)):
            used.add(options[0]["id"])
            pairs.append((options[0], r))
        else:
            unmatched.append(r)
    return pairs, unmatched


def _close(a, b, tol) -> bool:
    return abs(a - b) <= tol * max(a, b, 1)
