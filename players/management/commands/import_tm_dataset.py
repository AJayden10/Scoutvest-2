from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from players.models import Club, MarketValue, Player, PlayerSeasonStat
from players.services import tm_dataset as tm


class Command(BaseCommand):
    help = (
        "Import players, clubs, per-season stats and market values from the open transfermarkt-datasets "
        "CSVs (unzip the Kaggle download into one folder). No FBref needed."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dir", required=True, help="folder with players.csv, clubs.csv, appearances.csv, player_valuations.csv")
        parser.add_argument("--leagues", default=",".join(tm.DEFAULT_LEAGUES), help="Transfermarkt league ids, comma separated")
        parser.add_argument("--since-season", type=int, default=2015, help="first season start year to import stats for")
        parser.add_argument("--min-minutes", type=int, default=90, help="skip player-seasons with fewer minutes")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **o):
        d = Path(o["dir"])
        need = ["players.csv", "clubs.csv", "appearances.csv", "player_valuations.csv"]
        missing = [f for f in need if not (d / f).exists()]
        if missing:
            raise CommandError(f"missing in {d}: {', '.join(missing)}")
        leagues = [x.strip() for x in o["leagues"].split(",") if x.strip()]

        clubs = pd.read_csv(d / "clubs.csv", usecols=["club_id", "name", "domestic_competition_id"])
        comps = pd.read_csv(d / "competitions.csv") if (d / "competitions.csv").exists() else None
        club_league = dict(zip(clubs["club_id"], clubs["domestic_competition_id"]))
        club_name = dict(zip(clubs["club_id"], clubs["name"]))

        self.stdout.write("Reading appearances (this is the big file)...")
        app = pd.read_csv(d / "appearances.csv", usecols=[
            "game_id", "player_id", "player_club_id", "date", "yellow_cards", "red_cards", "goals", "assists", "minutes_played"])
        stats = tm.season_stats(app, club_league, leagues, o["since_season"])
        del app
        stats = stats[stats["minutes"] >= o["min_minutes"]]

        players = pd.read_csv(d / "players.csv", usecols=[
            "player_id", "name", "date_of_birth", "country_of_citizenship", "sub_position", "position",
            "contract_expiration_date", "current_club_id"]).set_index("player_id")
        stats = stats[stats["player_id"].isin(players.index)]
        vals = pd.read_csv(d / "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"])
        vals = vals[vals["player_id"].isin(stats["player_id"].unique())].dropna()
        vals = vals[vals["market_value_in_eur"] > 0]
        vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
        vals = vals.dropna(subset=["date"])
        vals = vals[vals["date"] >= f"{o['since_season'] - 3}-01-01"]  # a little history for the growth feature

        self.stdout.write(
            f"{stats['player_id'].nunique()} players, {len(stats)} player-seasons, {len(vals)} valuations "
            f"in {', '.join(leagues)} since {o['since_season']}"
        )
        if o["dry_run"]:
            return

        with transaction.atomic():
            club_objs = {}
            for cid in stats["player_club_id"].unique():
                lname, country = tm.league_info(club_league[cid], comps)
                club_objs[cid], _ = Club.objects.get_or_create(
                    name=club_name.get(cid, f"Club {cid}"), league=lname, defaults={"country": country})

            player_objs = {}
            existing = {p.transfermarkt_id: p for p in Player.objects.filter(transfermarkt_id__isnull=False)}
            latest_club = stats.sort_values("season").groupby("player_id")["player_club_id"].last()
            for pid in stats["player_id"].unique():
                r = players.loc[pid]
                dob = pd.to_datetime(r["date_of_birth"], errors="coerce")
                exp = pd.to_datetime(r["contract_expiration_date"], errors="coerce")
                fields = dict(
                    name=str(r["name"]), nationality=str(r["country_of_citizenship"]) if pd.notna(r["country_of_citizenship"]) else "",
                    position=tm.map_position(r["sub_position"], r["position"]),
                    birth_year=int(dob.year) if pd.notna(dob) else None,
                    contract_expiry_year=int(exp.year) if pd.notna(exp) else None,
                    club=club_objs[latest_club[pid]],
                )
                obj = existing.get(str(pid))
                if obj:
                    for k, v in fields.items():
                        setattr(obj, k, v)
                    obj.save()
                else:
                    obj = Player.objects.create(transfermarkt_id=str(pid), **fields)
                player_objs[pid] = obj

            PlayerSeasonStat.objects.bulk_create([
                PlayerSeasonStat(
                    player=player_objs[r.player_id], club=club_objs[r.player_club_id], season=r.season,
                    age=(int(r.season[:4]) + 1 - player_objs[r.player_id].birth_year) if player_objs[r.player_id].birth_year else None,
                    matches=int(r.matches), starts=int(r.starts), minutes=int(r.minutes), goals=float(r.goals),
                    assists=float(r.assists), extra={"yellow_cards": int(r.yellow_cards), "red_cards": int(r.red_cards)},
                ) for r in stats.itertuples()
            ], update_conflicts=True, unique_fields=["player", "season", "club"],
               update_fields=["age", "matches", "starts", "minutes", "goals", "assists", "extra"])

            MarketValue.objects.bulk_create([
                MarketValue(player=player_objs[r.player_id], date=r.date.date(), value=round(r.market_value_in_eur / 1e6, 3),
                            source="transfermarkt")
                for r in vals.itertuples()
            ], update_conflicts=True, unique_fields=["player", "date", "source"], update_fields=["value"])

        self.stdout.write(self.style.SUCCESS(
            f"Imported {len(player_objs)} players, {len(stats)} season rows, {len(vals)} market values"))
