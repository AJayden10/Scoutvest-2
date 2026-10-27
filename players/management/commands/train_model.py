from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from players.ml.features import build_frame, labelled
from players.ml.training import NotEnoughData, save, train


class Command(BaseCommand):
    help = "Train the value-prediction model on past seasons and report how it does on a later one."

    def add_arguments(self, parser):
        parser.add_argument("--horizon", type=int, default=2, help="years ahead to predict (default 2)")
        parser.add_argument("--test-year", type=int, default=None,
                            help="anchor year to test on (default: newest labelled year)")
        parser.add_argument("--no-embargo", action="store_true",
                            help="also train on rows whose labels come from the test period (optimistic; for comparison only)")
        parser.add_argument("--min-minutes", type=int, default=450)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--fast", action="store_true", help="smaller model, for quick checks")
        parser.add_argument("--exclude", nargs="*", default=[], help="features to drop (ablation), e.g. --exclude xg90 npxg90")
        parser.add_argument("--name", default=None, help="model name (default: timestamped)")

    def handle(self, *args, **o):
        df = build_frame(o["horizon"], o["min_minutes"])
        lab = labelled(df)
        self.stdout.write(f"{len(df)} player-seasons, {len(lab)} with a known value {o['horizon']} years later")
        try:
            result = train(df, o["test_year"], o["fast"], o["exclude"], horizon_years=o["horizon"],
                           embargo=not o["no_embargo"], seed=o["seed"])
        except NotEnoughData as e:
            raise CommandError(f"Not enough data to train: {e}. Import stats and market values first.")
        m = result.metrics
        b = m["baselines"]
        iv = m["interval"]
        self.stdout.write(
            f"Trained on {m['train_rows']} rows from {m['train_years']}, tested on {m['test_rows']} rows from {m['test_year']}"
            f"{' (embargoed)' if m['embargo'] else ' (NO embargo: optimistic)'}\n"
            f"  MAE €{m['mae']}M  RMSE €{m['rmse']}M  R² {m['r2']}  median error {m['median_abs_pct_error']}%\n"
            f"  Baselines MAE: no change €{b['no_change']}M, median growth €{b['median_growth']}M, age curve €{b['age_curve']}M\n"
            f"  80% range: coverage {iv['checked_coverage']} on {iv['checked_on_rows']} rows not used to set it"
        )
        for seg in m["segments"]["value_band"]:
            self.stdout.write(f"    {seg['segment']:>10}: n={seg['n']:<4} MAE €{seg['mae']}M (baseline €{seg['baseline_mae']}M)")
        if not m["beats_all_baselines"]:
            self.stdout.write(self.style.WARNING("The model does NOT beat every simple baseline; treat predictions with caution."))
        version = o["name"] or "xgb-" + timezone.now().strftime("%Y%m%d-%H%M")
        summary = {
            "player_seasons": int(len(df)), "labelled_rows": int(len(lab)), "players": int(df["player_id"].nunique()),
            "rows_by_anchor_year": {str(k): int(v) for k, v in lab["anchor_year"].value_counts().sort_index().items()},
            "leagues": sorted(x for x in df["league"].unique() if x), "min_minutes": o["min_minutes"],
        }
        path = save(result, settings.MODEL_DIR, version, o["horizon"], summary, o["seed"])
        self.stdout.write(self.style.SUCCESS(f"Saved {path} (+ metrics JSON and holdout CSV next to it)"))
