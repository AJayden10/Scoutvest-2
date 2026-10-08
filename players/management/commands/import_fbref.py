import os
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from players.models import Club, Player, PlayerSeasonStat
from players.services.fbref import MissingColumns, iter_records, resolve_columns


class Command(BaseCommand):
    help = (
        "Import FBref standard stats into the Django tables. Reads the CSV written by "
        "scripts/collect_stats.py, or the raw player_stats table written by "
        "scripts/load_to_postgres.py (--from-table)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--csv", default="data/raw/epl_player_standard_stats.csv")
        parser.add_argument("--from-table", action="store_true", help="read player_stats via DATABASE_URL instead of the CSV")
        parser.add_argument("--league", default="Premier League")
        parser.add_argument("--country", default="England")
        parser.add_argument("--show-columns", action="store_true", help="print the source columns and the detected mapping, then exit")

    def handle(self, *args, **opts):
        df = self._load(opts)
        self.stdout.write(f"Loaded {len(df)} source rows")

        if opts["show_columns"]:
            self.stdout.write(f"Columns: {list(df.columns)}")
            try:
                self.stdout.write(f"Detected mapping: {resolve_columns(df)}")
            except MissingColumns as e:
                raise CommandError(str(e))
            return

        try:
            records = list(iter_records(df))
        except MissingColumns as e:
            raise CommandError(str(e))

        created_players = 0
        with transaction.atomic():
            clubs: dict[str, Club] = {}
            for rec in records:
                club = clubs.get(rec.club)
                if club is None:
                    club, _ = Club.objects.get_or_create(
                        name=rec.club, league=opts["league"], defaults={"country": opts["country"]}
                    )
                    clubs[rec.club] = club

                # No FBref id in the standard table, so identify a player by name + birth year.
                player, was_created = Player.objects.get_or_create(
                    name=rec.name, birth_year=rec.birth_year,
                    defaults={"nationality": rec.nationality, "position": rec.position, "club": club},
                )
                created_players += was_created

                PlayerSeasonStat.objects.update_or_create(
                    player=player, season=rec.season, club=club,
                    defaults=dict(
                        age=rec.age, matches=rec.matches, starts=rec.starts, minutes=rec.minutes,
                        goals=rec.goals, assists=rec.assists, xg=rec.xg, xag=rec.xag,
                        progressive_passes=rec.prog_passes, progressive_carries=rec.prog_carries,
                    ),
                )

            # Point each player at their most recent club/position.
            for player in Player.objects.filter(season_stats__isnull=False).distinct():
                latest = player.season_stats.order_by("-season", "-minutes").first()
                if latest and latest.club_id:
                    player.club_id = latest.club_id
                    player.save(update_fields=["club"])

        self.stdout.write(self.style.SUCCESS(
            f"Imported {len(records)} stat rows ({created_players} new players)"
        ))

    def _load(self, opts) -> pd.DataFrame:
        if opts["from_table"]:
            from sqlalchemy import create_engine

            url = os.environ.get("DATABASE_URL")
            if not url:
                raise CommandError("DATABASE_URL is not set")
            return pd.read_sql_table("player_stats", create_engine(url))
        path = Path(opts["csv"])
        if not path.exists():
            raise CommandError(f"{path} not found (run scripts/collect_stats.py first, or use --from-table)")
        return pd.read_csv(path)
