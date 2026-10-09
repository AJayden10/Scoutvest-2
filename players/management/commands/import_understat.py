import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from players.models import PlayerSeasonStat
from players.services import understat as us


class Command(BaseCommand):
    help = "Add Understat xG/xA and related stats to player-seasons already imported (e.g. from the Transfermarkt dataset)."

    def add_arguments(self, parser):
        parser.add_argument("--dir", default="data/raw/understat", help="folder written by collect_understat")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **o):
        files = sorted(Path(o["dir"]).glob("*_*.json"))
        if not files:
            raise CommandError(f"no Understat files in {o['dir']}; run collect_understat first")
        total_pairs = total_rows = 0
        for f in files:
            code, year = f.stem.rsplit("_", 1)
            if code not in us.LEAGUES:
                continue
            league, season = us.LEAGUES[code], us.season_label(int(year))
            rows = us.parse_players(json.loads(f.read_text()))
            stats = list(PlayerSeasonStat.objects.filter(season=season, club__league=league)
                         .values("id", "player_id", "player__name", "minutes"))
            cands = [{"id": s["id"], "name": s["player__name"], "minutes": s["minutes"]} for s in stats]
            pairs, unmatched = us.match_rows(rows, cands)
            total_pairs += len(pairs)
            total_rows += len(rows)
            self.stdout.write(f"{league} {season}: matched {len(pairs)}/{len(rows)}"
                              + (f" (e.g. unmatched: {', '.join(r['name'] for r in unmatched[:3])})" if unmatched else ""))
            if o["dry_run"]:
                continue
            by_id = {s.id: s for s in PlayerSeasonStat.objects.filter(id__in=[c["id"] for c, _ in pairs])}
            for cand, r in pairs:
                s = by_id[cand["id"]]
                s.xg, s.xag = r["xg"], r["xa"]
                s.extra = {**s.extra, "npxg": r["npxg"], "shots": r["shots"], "key_passes": r["key_passes"],
                           "xgchain": r["xgchain"], "xgbuildup": r["xgbuildup"]}
            PlayerSeasonStat.objects.bulk_update(by_id.values(), ["xg", "xag", "extra"])
        pct = 100 * total_pairs / total_rows if total_rows else 0
        self.stdout.write(self.style.SUCCESS(f"Matched {total_pairs} of {total_rows} Understat rows ({pct:.0f}%)"))
