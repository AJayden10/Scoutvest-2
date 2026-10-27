"""Position-aware player similarity with explanations ("who plays like X, within my budget?").

Method (deliberately simple, so every number can be explained):
1. Pick one season. Default: the reference player's season with the most comparable stats.
2. Pool: players in the same position group that season with at least `min_minutes`.
3. Features: per-90 output from the season stats, plus provider metrics (SkillCorner physical, off-ball
   runs and passing) when present. A feature is used only if the reference has it and at least 70% of
   the pool does, so a source that covers a few players cannot skew the comparison.
4. Each feature is turned into a z-score within the pool (clipped to +-3).
5. Distance = root-mean-square gap in z-scores over the shared features; similarity = 100 / (1 + distance).
   0 gap -> 100, a one-standard-deviation average gap -> 50.
6. Each result lists its three closest stats and two biggest differences, with both players' raw values.

Similarity describes style and output. It is not a forecast of quality, fit or price.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from players.models import Player, PlayerSeasonMetrics, PlayerSeasonStat

from . import skillcorner

POSITION_GROUP = {"GK": "GK", "CB": "CB", "FB": "FB", "DM": "MID", "CM": "MID", "AM": "MID", "Winger": "W", "ST": "ST"}
GROUP_LABEL = {"GK": "Goalkeepers", "CB": "Centre-backs", "FB": "Full-backs", "MID": "Midfielders", "W": "Wingers", "ST": "Strikers"}

STAT_FEATURES = {
    "goals90": "Goals per 90", "assists90": "Assists per 90", "xg90": "xG per 90", "xag90": "xA per 90",
    "prog_passes90": "Progressive passes per 90", "prog_carries90": "Progressive carries per 90",
    "shots90": "Shots per 90", "key_passes90": "Key passes per 90",
}
LABELS = {**STAT_FEATURES, **skillcorner.LABELS}
MIN_COVERAGE = 0.7
MIN_FEATURES = 3
MAX_MISSING_SHARE = 0.3
Z_CLIP = 3.0


class SimilarityError(Exception):
    """Raised when a comparison is not possible (no stats for the player, too few comparable stats...)."""


def _per90(total, minutes):
    return None if total is None or not minutes else total * 90.0 / minutes


def season_rows(season: str, player_ids: list[int] | None = None) -> dict[int, dict]:
    """One row per player for a season: minutes, per-90 stats and provider metrics."""
    agg: dict[int, dict] = defaultdict(lambda: defaultdict(float))
    has: dict[int, set] = defaultdict(set)
    stats_qs = PlayerSeasonStat.objects.filter(season=season)
    metrics_qs = PlayerSeasonMetrics.objects.filter(season=season)
    if player_ids is not None:
        stats_qs, metrics_qs = stats_qs.filter(player_id__in=player_ids), metrics_qs.filter(player_id__in=player_ids)
    for s in stats_qs.values(
            "player_id", "minutes", "goals", "assists", "xg", "xag", "progressive_passes", "progressive_carries", "extra", "age"):
        a = agg[s["player_id"]]
        a["minutes"] += s["minutes"] or 0
        a["age"] = s["age"] if s["age"] is not None else a.get("age")
        for src, key in (("goals", "goals"), ("assists", "assists"), ("xg", "xg"), ("xag", "xag"),
                         ("progressive_passes", "prog_passes"), ("progressive_carries", "prog_carries")):
            if s[src] is not None:
                a[key] += s[src]
                has[s["player_id"]].add(key)
        for key in ("shots", "key_passes"):
            if (s["extra"] or {}).get(key) is not None:
                a[key] += s["extra"][key]
                has[s["player_id"]].add(key)

    rows: dict[int, dict] = {}
    for pid, a in agg.items():
        mins = a["minutes"]
        row = {"minutes": mins, "age": a.get("age")}
        for key in ("goals", "assists", "xg", "xag", "prog_passes", "prog_carries", "shots", "key_passes"):
            row[f"{key}90"] = _per90(a[key], mins) if key in has[pid] else None
        rows[pid] = row

    for m in metrics_qs.values("player_id", "metrics", "position_group"):
        row = rows.setdefault(m["player_id"], {"minutes": 0, "age": None})
        metrics = m["metrics"] or {}
        row.update({k: v for k, v in metrics.items() if k in skillcorner.LABELS})
        if not row["minutes"] and metrics.get("minutes_total"):
            row["minutes"] = metrics["minutes_total"]  # no stats row: use the provider's minutes
        row.setdefault("_position_group", m["position_group"])
    return rows


def _position_of(player: Player, row: dict) -> str:
    return player.position or skillcorner.POSITION.get(row.get("_position_group", ""), "")


def best_season(player: Player) -> str:
    """The reference player's season with the most comparable stats (latest on a tie)."""
    seasons = set(PlayerSeasonStat.objects.filter(player=player).values_list("season", flat=True))
    seasons |= set(PlayerSeasonMetrics.objects.filter(player=player).values_list("season", flat=True))
    if not seasons:
        raise SimilarityError("this player has no season stats to compare")

    def score(season):
        row = season_rows(season, [player.id]).get(player.id, {})
        return (sum(1 for k in LABELS if row.get(k) is not None), season)

    return max(seasons, key=score)


def find_similar(player: Player, *, season: str | None = None, limit: int = 10, min_minutes: int = 450,
                 max_value: float | None = None, max_age: int | None = None, values: dict | None = None) -> dict:
    """`values` maps player id -> current market value (€M) for the budget filter; players without a
    value are dropped when `max_value` is set, since their price is unknown."""
    season = season or best_season(player)
    rows = season_rows(season)
    ref = rows.get(player.id)
    if ref is None:
        raise SimilarityError(f"no stats for this player in {season}")

    players = {p.id: p for p in Player.objects.filter(id__in=list(rows)).select_related("club")}
    group = POSITION_GROUP.get(_position_of(player, ref))
    if group is None:
        raise SimilarityError("this player has no position, so there is no like-for-like pool")
    pool = [pid for pid, r in rows.items()
            if pid in players and POSITION_GROUP.get(_position_of(players[pid], r)) == group
            and (r["minutes"] or 0) >= min_minutes]
    if player.id not in pool:
        pool.append(player.id)  # always compare the reference, even on few minutes (flagged below)

    features = []
    for key in LABELS:
        if ref.get(key) is None:
            continue
        coverage = np.mean([rows[pid].get(key) is not None for pid in pool])
        if coverage >= MIN_COVERAGE:
            features.append(key)
    if len(features) < MIN_FEATURES:
        raise SimilarityError(f"only {len(features)} stats are available for both this player and most of the pool")

    M = np.array([[np.nan if rows[pid].get(k) is None else float(rows[pid][k]) for k in features] for pid in pool])
    mean, std = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
    keep = std > 0
    features = [f for f, k in zip(features, keep) if k]
    M, mean, std = M[:, keep], mean[keep], std[keep]
    Z = np.clip((M - mean) / std, -Z_CLIP, Z_CLIP)
    ref_i = pool.index(player.id)
    zr = Z[ref_i]

    results = []
    for i, pid in enumerate(pool):
        if pid == player.id:
            continue
        r = rows[pid]
        if max_age is not None and (r.get("age") is None or r["age"] > max_age):
            continue
        value = (values or {}).get(pid)
        if max_value is not None and (value is None or value > max_value):
            continue
        z = Z[i]
        missing = np.isnan(z)
        if missing.mean() > MAX_MISSING_SHARE:
            continue
        gap = np.where(missing, 0.0, z - zr)  # missing stats are left out of the distance below
        rms = float(np.sqrt(np.mean(gap[~missing] ** 2)))
        order = np.argsort(np.abs(gap))
        explain = lambda j: {"feature": features[j], "label": LABELS[features[j]],
                             "reference": round(float(M[ref_i, j]), 2), "candidate": round(float(M[i, j]), 2),
                             "gap": round(float(gap[j]), 2)}
        shared = [j for j in order if not missing[j]]
        results.append({
            "playerId": pid, "similarity": round(100 / (1 + rms)), "distance": round(rms, 3),
            "sharedFeatures": int((~missing).sum()), "minutes": int(r["minutes"] or 0),
            "closest": [explain(j) for j in shared[:3]],
            "differences": [explain(j) for j in shared[::-1][:2]],
        })
    results.sort(key=lambda x: (x["distance"], x["playerId"]))
    return {
        "season": season, "positionGroup": group, "positionGroupLabel": GROUP_LABEL[group],
        "poolSize": len(pool) - 1, "minMinutes": min_minutes,
        "referenceMinutes": int(ref["minutes"] or 0), "referenceBelowMinMinutes": (ref["minutes"] or 0) < min_minutes,
        "features": [{"key": k, "label": LABELS[k], "reference": round(float(M[ref_i, j]), 2)} for j, k in enumerate(features)],
        "method": "z-scores within position group and season; distance = RMS gap over shared stats; similarity = 100/(1+distance)",
        "results": results[:limit],
    }
