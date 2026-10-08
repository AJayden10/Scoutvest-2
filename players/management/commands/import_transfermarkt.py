from datetime import date
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from players.models import MarketValue, Player
from players.services.transfermarkt import (
    MissingColumns, build_index, iter_tm_players, iter_valuations, match_player,
)


class Command(BaseCommand):
    help = (
        "Attach Transfermarkt market values to the players already imported from FBref. "
        "Needs players.csv and player_valuations.csv from the open transfermarkt-datasets project."
    )

    def add_arguments(self, parser):
        parser.add_argument("--players", required=True, help="path to players.csv")
        parser.add_argument("--valuations", required=True, help="path to player_valuations.csv")
        parser.add_argument("--since", default="2018-01-01", help="ignore valuations before this date (YYYY-MM-DD)")
        parser.add_argument("--show-columns", action="store_true", help="print the columns of both files and exit")
        parser.add_argument("--dry-run", action="store_true", help="report matches without writing anything")

    def handle(self, *args, **opts):
        players_df, vals_df = self._read(opts["players"]), self._read(opts["valuations"])
        if opts["show_columns"]:
            self.stdout.write(f"players.csv: {list(players_df.columns)}")
            self.stdout.write(f"player_valuations.csv: {list(vals_df.columns)}")
            return
        try:
            since = date.fromisoformat(opts["since"])
            tm_players = list(iter_tm_players(players_df))
            index = build_index(tm_players)
            ours = list(Player.objects.all())
            matched: dict[str, Player] = {}
            unmatched = []
            for p in ours:
                hit = match_player(p.name, p.birth_year, index)
                if hit and hit.tm_id not in matched:
                    matched[hit.tm_id] = p
                    p._tm = hit
                else:
                    unmatched.append(p.name)
            rows = list(iter_valuations(vals_df, set(matched), since))
        except (MissingColumns, ValueError) as e:
            raise CommandError(str(e))

        self.stdout.write(
            f"Matched {len(matched)} of {len(ours)} players; {len(rows)} valuation points since {since}"
        )
        if unmatched:
            sample = ", ".join(unmatched[:10])
            self.stdout.write(f"Unmatched (first 10 of {len(unmatched)}): {sample}")
        if opts["dry_run"]:
            return

        with transaction.atomic():
            for tid, p in matched.items():
                p.transfermarkt_id = tid
                if p._tm.contract_expiry_year:
                    p.contract_expiry_year = p._tm.contract_expiry_year
                if not p.birth_year and p._tm.birth_year:
                    p.birth_year = p._tm.birth_year
                p.save(update_fields=["transfermarkt_id", "contract_expiry_year", "birth_year"])
            for tid, d, v in rows:
                MarketValue.objects.update_or_create(
                    player=matched[tid], date=d, source="transfermarkt", defaults={"value": v}
                )
        self.stdout.write(self.style.SUCCESS(f"Saved {len(rows)} market values for {len(matched)} players"))

    def _read(self, path):
        p = Path(path)
        if not p.exists():
            raise CommandError(f"{p} not found")
        return pd.read_csv(p)
