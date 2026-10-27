from collections import Counter, defaultdict
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from players.models import Club, Player, PlayerSeasonMetrics
from players.services import skillcorner as sc
from players.services.transfermarkt import normalize_name


class Command(BaseCommand):
    help = (
        "Import SkillCorner's open A-League 2024/25 season aggregates (physical, off-ball runs, passing). "
        "Run import_tm_dataset --leagues AUS1 first so players match the ones that have market values."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dir", required=True,
                            help="folder containing the three *aggregates*.csv files (e.g. a clone of SkillCorner/opendata)")
        parser.add_argument("--league", default="A-League Men", help="league name for clubs of newly created players")
        parser.add_argument("--country", default="Australia")
        parser.add_argument("--no-create", action="store_true", help="only attach metrics to players that already exist")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **o):
        files = sc.find_files(Path(o["dir"]))
        if not files:
            raise CommandError(f"no SkillCorner aggregate CSVs found under {o['dir']} (expected {', '.join(sc.FILES.values())})")
        missing = set(sc.FILES) - set(files)
        if missing:
            self.stdout.write(self.style.WARNING(f"missing {', '.join(sorted(missing))} file(s); importing the rest"))
        profiles = sc.load_profiles(files)

        by_sc = {p.skillcorner_id: p for p in Player.objects.filter(skillcorner_id__isnull=False)}
        by_name_year, by_year = defaultdict(list), defaultdict(list)
        for p in Player.objects.exclude(skillcorner_id__in=list(by_sc)).only("id", "name", "birth_year"):
            by_name_year[(normalize_name(p.name), p.birth_year)].append(p)
            by_year[p.birth_year].append(p)

        outcome = Counter()
        plan = []
        for prof in profiles:
            player = by_sc.get(prof.skillcorner_id)
            rule = "skillcorner id" if player else None
            if player is None:
                player, rule = sc.match_player(prof, by_name_year, by_year)
            outcome[rule] += 1
            plan.append((prof, player, rule))

        self.stdout.write(f"{len(profiles)} SkillCorner players in season {profiles[0].season if profiles else '?'}: "
                          + ", ".join(f"{k}: {v}" for k, v in sorted(outcome.items())))
        if o["dry_run"]:
            return

        created = attached = skipped = 0
        with transaction.atomic():
            for prof, player, rule in plan:
                if player is None:
                    if o["no_create"] or rule == "ambiguous":
                        skipped += 1
                        continue
                    club, _ = Club.objects.get_or_create(name=prof.team, league=o["league"], defaults={"country": o["country"]})
                    player = Player.objects.create(name=prof.name, birth_year=prof.birth_year, position=prof.position,
                                                   club=club, skillcorner_id=prof.skillcorner_id)
                    created += 1
                else:
                    changed = []
                    if not player.skillcorner_id:
                        player.skillcorner_id = prof.skillcorner_id
                        changed.append("skillcorner_id")
                    if not player.position and prof.position:
                        player.position = prof.position
                        changed.append("position")
                    if changed:
                        player.save(update_fields=changed)
                    attached += 1
                PlayerSeasonMetrics.objects.update_or_create(
                    player=player, season=prof.season, source=sc.SOURCE,
                    defaults={"team_name": prof.team, "position_group": prof.position_group,
                              "matches": prof.matches, "metrics": prof.metrics},
                )
        self.stdout.write(self.style.SUCCESS(
            f"Attached metrics to {attached} existing players, created {created}, skipped {skipped} (ambiguous or --no-create)"))
