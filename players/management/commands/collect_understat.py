from pathlib import Path

from django.core.management.base import BaseCommand

from players.services import understat as us


class Command(BaseCommand):
    help = (
        "Download Understat player stats (xG, xA, xGChain...) to JSON files. Run this on your own machine; "
        "it makes one request every few seconds. Check Understat's terms before heavy or commercial use."
    )

    def add_arguments(self, parser):
        parser.add_argument("--out", default="data/raw/understat")
        parser.add_argument("--leagues", default="EPL,La_liga,Bundesliga,Serie_A,Ligue_1", help=f"any of {', '.join(us.LEAGUES)}")
        parser.add_argument("--from-year", type=int, default=2015, help="first season start year")
        parser.add_argument("--to-year", type=int, default=2025, help="last season start year")
        parser.add_argument("--delay", type=float, default=3.0, help="seconds between requests")

    def handle(self, *args, **o):
        codes = [c.strip() for c in o["leagues"].split(",") if c.strip()]
        years = list(range(o["from_year"], o["to_year"] + 1))
        self.stdout.write(f"Collecting {len(codes)} leagues x {len(years)} seasons into {o['out']}")
        n = us.collect(Path(o["out"]), codes, years, o["delay"], log=self.stdout.write)
        self.stdout.write(self.style.SUCCESS(f"Downloaded {n} new league-seasons (existing files are skipped)"))
