import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from players.ml.features import FEATURES, build_frame
from players.ml.signals import age_bucket, classify, confidence, reasons, risk_score, upside_pct
from players.ml.training import load_latest
from players.models import Player, Prediction


class Command(BaseCommand):
    help = "Predict future market value for every player's latest season and store it with signal, risk and reasons."

    def add_arguments(self, parser):
        parser.add_argument("--min-minutes", type=int, default=450)
        parser.add_argument("--keep-history", action="store_true", help="keep earlier predictions instead of replacing them")

    def handle(self, *args, **o):
        try:
            model, meta = load_latest(settings.MODEL_DIR)
        except FileNotFoundError as e:
            raise CommandError(str(e))
        horizon = meta["horizon_years"]
        mle = meta["metrics"]["mean_abs_log_error"]

        df = build_frame(horizon, o["min_minutes"])
        if df.empty:
            raise CommandError("no player-seasons with a market value; import data first")
        # Only players active in the newest season: predicting from a 2017 snapshot says nothing about today.
        latest = df[df["anchor_year"] == df["anchor_year"].max()].sort_values("minutes").groupby("player_id").tail(1)
        preds = latest["current_value"].to_numpy() * np.exp(model.predict(latest[FEATURES]))

        ups = [upside_pct(float(c), float(p)) for c, p in zip(latest["current_value"], preds)]
        buckets = [age_bucket(a) for a in latest["age"]]
        peer = {b: float(np.median([u for u, bb in zip(ups, buckets) if bb == b])) for b in set(buckets)}

        created = 0
        with transaction.atomic():
            if not o["keep_history"]:
                Prediction.objects.all().delete()
            for (_, row), pv, up, bucket in zip(latest.iterrows(), preds, ups, buckets):
                risk = risk_score(minutes=row["minutes"], age=row["age"], contract_years_left=row["contract_years_left"],
                                  value_growth_12m=row["value_growth_12m"], mean_abs_log_error=mle)
                Prediction.objects.create(
                    player_id=int(row["player_id"]), model_version=meta["version"], horizon_years=horizon,
                    predicted_value=round(float(pv), 2),
                    confidence=confidence(minutes=row["minutes"], mean_abs_log_error=mle),
                    risk_score=risk,
                    signal=classify(upside=up, age=row["age"], risk=risk, peer_median=peer[bucket]),
                    reasons=reasons(model, latest.loc[[_]]),
                )
                created += 1
        self.stdout.write(self.style.SUCCESS(f"Wrote {created} predictions with model {meta['version']}"))
