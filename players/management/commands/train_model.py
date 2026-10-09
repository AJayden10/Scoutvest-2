from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from players.ml.features import build_frame, labelled
from players.ml.training import NotEnoughData, save, train


class Command(BaseCommand):
    help = "Train the value-prediction model on past seasons and report how it does on later ones."

    def add_arguments(self, parser):
        parser.add_argument("--horizon", type=int, default=2, help="years ahead to predict (default 2)")
        parser.add_argument("--cutoff-year", type=int, default=None, help="train on anchors up to this year, test after")
        parser.add_argument("--min-minutes", type=int, default=450)
        parser.add_argument("--fast", action="store_true", help="smaller model, for quick checks")
        parser.add_argument("--exclude", nargs="*", default=[], help="features to drop (ablation), e.g. --exclude xg90 npxg90")
        parser.add_argument("--name", default=None, help="model name (default: timestamped)")

    def handle(self, *args, **o):
        df = build_frame(o["horizon"], o["min_minutes"])
        lab = labelled(df)
        self.stdout.write(f"{len(df)} player-seasons, {len(lab)} with a known value {o['horizon']} years later")
        try:
            result = train(df, o["cutoff_year"], o["fast"], o["exclude"])
        except NotEnoughData as e:
            raise CommandError(f"Not enough data to train: {e}. Import FBref stats and Transfermarkt values first.")
        m = result.metrics
        self.stdout.write(
            f"Trained on {m['train_rows']} rows up to {m['cutoff_year']}, tested on {m['test_rows']} later rows\n"
            f"  MAE €{m['mae']}M (baseline 'value stays the same': €{m['baseline_mae']}M)  RMSE €{m['rmse']}M  R² {m['r2']}\n"
            f"  median error {m['median_abs_pct_error']}%"
        )
        if not m["beats_baseline"]:
            self.stdout.write(self.style.WARNING("The model does NOT beat the naive baseline yet; treat predictions with caution."))
        version = o["name"] or "xgb-" + timezone.now().strftime("%Y%m%d-%H%M")
        path = save(result, settings.MODEL_DIR, version, o["horizon"])
        self.stdout.write(self.style.SUCCESS(f"Saved {path}"))
