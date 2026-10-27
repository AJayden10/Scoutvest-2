"""Query helpers: annotate players with their latest valuation/prediction so the API can
filter and sort on them in SQL."""

from django.db.models import ExpressionWrapper, F, FloatField, OuterRef, Prefetch, Subquery
from django.db.models.functions import NullIf

from .models import MarketValue, Player, PlayerSeasonMetrics, PlayerSeasonStat, Prediction

RISK_LOW_MAX = 35  # risk_score < 35 -> Low
RISK_MEDIUM_MAX = 60  # 35 <= risk_score < 60 -> Medium, otherwise High


def risk_label(score):
    if score is None:
        return None
    if score < RISK_LOW_MAX:
        return "Low"
    if score < RISK_MEDIUM_MAX:
        return "Medium"
    return "High"


def annotated_players():
    """Players with latest value/prediction annotations (no prefetching; use for aggregates)."""
    latest_value = MarketValue.objects.filter(player=OuterRef("pk")).order_by("-date")
    latest_pred = Prediction.objects.filter(player=OuterRef("pk")).order_by("-created_at")
    latest_stat = PlayerSeasonStat.objects.filter(player=OuterRef("pk")).order_by("-season", "-minutes")

    return (
        Player.objects.select_related("club")
        .annotate(
            current_value=Subquery(latest_value.values("value")[:1], output_field=FloatField()),
            predicted_value=Subquery(latest_pred.values("predicted_value")[:1], output_field=FloatField()),
            signal=Subquery(latest_pred.values("signal")[:1]),
            risk_score=Subquery(latest_pred.values("risk_score")[:1]),
            confidence=Subquery(latest_pred.values("confidence")[:1]),
            latest_age=Subquery(latest_stat.values("age")[:1]),
        )
        .annotate(
            upside=ExpressionWrapper(
                (F("predicted_value") - F("current_value")) * 100.0 / NullIf(F("current_value"), 0.0),
                output_field=FloatField(),
            )
        )
    )


def players_queryset():
    """Annotated players plus the prefetches PlayerSerializer needs."""
    return annotated_players().prefetch_related(
        Prefetch(
            "season_stats",
            queryset=PlayerSeasonStat.objects.select_related("club").order_by("-season", "-minutes"),
            to_attr="stats_list",
        ),
        Prefetch("market_values", queryset=MarketValue.objects.order_by("date"), to_attr="values_list"),
        Prefetch("predictions", queryset=Prediction.objects.order_by("-created_at"), to_attr="preds_list"),
        Prefetch("season_metrics", queryset=PlayerSeasonMetrics.objects.order_by("-season"), to_attr="metrics_list"),
    )
