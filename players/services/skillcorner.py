"""SkillCorner open data (MIT licence): season aggregates for every A-League Men player in 2024/25.

Source: https://github.com/SkillCorner/opendata  (data/aggregates/*.csv). Three files:
  physical  - per-match averages (distance, sprints, top speed, accelerations); rows weighted by `count_match`
  obr       - off-ball runs per 30 minutes of team possession; rows weighted by `performance_count`
  passing   - passing per 30 minutes of team possession; rows weighted by `performance_count`

A player appears once per (team, position group) he played, so rows are combined into one profile per
player with a match-weighted average, and his main position is the group he played most.

Players are matched to existing ones (usually from the Transfermarkt import) by name + birth year: an
exact normalised name first, then a shared-name-tokens rule for names written differently
("Douglas Costa de Souza" vs "Douglas Costa"). A match must be unique; ambiguous names are skipped,
never guessed, because a wrong match silently gives one player another's stats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .transfermarkt import normalize_name

SOURCE = "skillcorner"

# stored key -> (file, source column, readable label). Kept small and balanced across physical, running
# and passing so no single category dominates a similarity search.
METRICS: dict[str, tuple[str, str, str]] = {
    "distance_per_min": ("physical", "total_metersperminute_full_all", "Distance per minute (m)"),
    "hsr_distance": ("physical", "hsr_distance_full_all", "High-speed running per match (m)"),
    "sprints": ("physical", "sprint_count_full_all", "Sprints per match"),
    "high_accels": ("physical", "highaccel_count_full_all", "High accelerations per match"),
    "top_speed": ("physical", "psv99", "Top speed (PSV-99, km/h)"),
    "runs": ("obr", "offballrun_count_p30tip", "Off-ball runs per 30' in possession"),
    "runs_in_behind": ("obr", "behindrun_count_p30tip", "Runs in behind per 30' in possession"),
    "coming_short_runs": ("obr", "comingshortrun_count_p30tip", "Coming-short runs per 30' in possession"),
    "overlap_runs": ("obr", "overlaprun_count_p30tip", "Overlapping runs per 30' in possession"),
    "dangerous_runs": ("obr", "offballrun_count_dangerous_p30tip", "Dangerous runs per 30' in possession"),
    "passes": ("passing", "pass_count_attempted_p30tip", "Passes per 30' in possession"),
    "pass_completion": ("passing", "pass_pct_completed", "Pass completion %"),
    "line_breaking_passes": ("passing", "pass_count_linebreak_attempted_p30tip", "Line-breaking passes per 30' in possession"),
    "passes_to_runs": ("passing", "pass_count_torun_attempted_p30tip", "Passes to runners per 30' in possession"),
    "long_passes": ("passing", "pass_count_longrange_attempted_p30tip", "Long passes per 30' in possession"),
}
LABELS = {k: v[2] for k, v in METRICS.items()}
FILES = {"physical": "*physicalaggregates*.csv", "obr": "*obraggregates*.csv", "passing": "*passingaggregates*.csv"}
WEIGHT = {"physical": "count_match", "obr": "performance_count", "passing": "performance_count"}
POSITION = {"Central Defender": "CB", "Full Back": "FB", "Midfield": "CM", "Wide Attacker": "Winger", "Center Forward": "ST"}


@dataclass
class Profile:
    skillcorner_id: str
    name: str
    birth_year: int | None
    team: str
    position_group: str
    season: str
    matches: int
    metrics: dict = field(default_factory=dict)

    @property
    def position(self) -> str:
        return POSITION.get(self.position_group, "")


def find_files(directory: Path) -> dict[str, Path]:
    found = {}
    for key, pattern in FILES.items():
        hits = sorted(directory.glob(pattern)) or sorted(directory.glob(f"**/{pattern}"))
        if hits:
            found[key] = hits[0]
    return found


def _season(label: str) -> str:
    """'2024/2025' -> '2024-2025' (the format the rest of the app uses)."""
    return str(label).replace("/", "-").strip()


def load_profiles(files: dict[str, Path]) -> list[Profile]:
    """One Profile per SkillCorner player, metrics match-weighted across his (team, position) rows."""
    frames = {k: pd.read_csv(p) for k, p in files.items()}
    info: dict[str, dict] = {}
    metrics: dict[str, dict] = {}
    for key, df in frames.items():
        w_col = WEIGHT[key]
        df = df.copy()
        df["_w"] = pd.to_numeric(df.get(w_col), errors="coerce").fillna(0)
        for pid, g in df.groupby("player_id"):
            pid = str(int(pid))
            w = g["_w"]
            main = g.loc[w.idxmax()]
            entry = info.setdefault(pid, {"weights": {}})
            entry.setdefault("name", str(main["player_name"]))
            entry.setdefault("birthdate", main.get("player_birthdate"))
            entry.setdefault("season", _season(main["season_name"]))
            # position/team from the physical file when present (it counts full matches), else the first seen
            if key == "physical" or "team" not in entry:
                entry["team"] = str(main["team_name"])
                entry["position_group"] = str(main["position_group"])
            if key == "physical":
                entry["matches"] = int(w.sum())
                mins = pd.to_numeric(g["minutes_full_all"], errors="coerce")
                metrics.setdefault(pid, {})["minutes_total"] = round(float((mins * w).sum()), 1)
            for stored, (file_key, col, _label) in METRICS.items():
                if file_key != key or col not in g:
                    continue
                vals = pd.to_numeric(g[col], errors="coerce")
                ok = vals.notna() & (w > 0)
                if ok.any():
                    metrics.setdefault(pid, {})[stored] = round(float((vals[ok] * w[ok]).sum() / w[ok].sum()), 3)

    out = []
    for pid, e in info.items():
        bd = pd.to_datetime(e.get("birthdate"), errors="coerce")
        out.append(Profile(
            skillcorner_id=pid, name=e["name"], birth_year=int(bd.year) if pd.notna(bd) else None,
            team=e.get("team", ""), position_group=e.get("position_group", ""), season=e["season"],
            matches=e.get("matches", 0), metrics=metrics.get(pid, {}),
        ))
    return out


def _tokens(name: str) -> set[str]:
    return set(normalize_name(name).split())


def match_player(profile: Profile, by_name_year: dict, by_year: dict):
    """Return (player, rule) or (None, reason). Exact normalised name + birth year first, then a
    shared-token rule (one name's tokens contained in the other's, at least two shared)."""
    key = (normalize_name(profile.name), profile.birth_year)
    exact = by_name_year.get(key, [])
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        return None, "ambiguous"
    if profile.birth_year is None:
        return None, "no birth year"
    mine = _tokens(profile.name)
    cands = [p for p in by_year.get(profile.birth_year, [])
             if len(mine & _tokens(p.name)) >= 2 and (_tokens(p.name) <= mine or mine <= _tokens(p.name))]
    if len(cands) == 1:
        return cands[0], "tokens"
    return None, "ambiguous" if cands else "unmatched"
