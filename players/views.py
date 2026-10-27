from django.db.models import Avg, Count
from django.db.models.functions import TruncMonth
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .filters import PlayerFilter
from .models import MarketValue, Player, PlayerSeasonMetrics, PlayerSeasonStat, Prediction, WatchlistEntry
from .selectors import annotated_players, players_queryset
from .serializers import PlayerDetailSerializer, PlayerSerializer
from .services.simulator import build_portfolio, simulate_transfer

# API ordering name -> annotated column, so NULLs (players without a prediction/value yet)
# always sort last instead of first.
ORDERING_COLUMNS = {
    "name": "name", "age": "latest_age", "current_value": "current_value",
    "predicted_value": "predicted_value", "upside": "upside",
    "confidence": "confidence", "risk_score": "risk_score",
}


class PlayerViewSet(viewsets.ReadOnlyModelViewSet):
    """GET /api/players/ (filter + sort), /api/players/<id>/, /api/players/compare/?ids=1,2,3"""

    filterset_class = PlayerFilter

    def get_queryset(self):
        return players_queryset()

    def get_serializer_class(self):
        return PlayerDetailSerializer if self.action == "retrieve" else PlayerSerializer

    def filter_queryset(self, queryset):
        queryset = super().filter_queryset(queryset)
        ordering = self.request.query_params.get("ordering")
        if ordering:
            exprs = []
            for part in ordering.split(","):
                column = ORDERING_COLUMNS.get(part.lstrip("-"))
                if column:
                    from django.db.models import F
                    exprs.append(F(column).desc(nulls_last=True) if part.startswith("-")
                                 else F(column).asc(nulls_last=True))
            if exprs:
                queryset = queryset.order_by(*exprs)
        return queryset

    @action(detail=False, methods=["get"])
    def compare(self, request):
        try:
            ids = [int(i) for i in request.query_params.get("ids", "").split(",") if i]
        except ValueError:
            return Response({"detail": "ids must be a comma-separated list of integers"}, status=400)
        if not 2 <= len(ids) <= 4:
            return Response({"detail": "provide between 2 and 4 player ids"}, status=400)
        found = {p.id: p for p in players_queryset().filter(id__in=ids)}
        missing = [i for i in ids if i not in found]
        if missing:
            return Response({"detail": f"unknown player ids: {missing}"}, status=404)
        return Response(PlayerSerializer([found[i] for i in ids], many=True).data)

    @action(detail=True, methods=["get"])
    def similar(self, request, pk=None):
        """GET /api/players/<id>/similar/?limit=10&max_value=5&max_age=25&min_minutes=450&season=2024-2025

        Same-position players with the closest statistical profile, each with the stats that make them
        close and the ones that differ most. `max_value` is a budget in €M on current market value."""
        from .services.similarity import SimilarityError, find_similar

        player = Player.objects.filter(pk=pk).first()
        if player is None:
            return Response({"detail": "unknown player", "code": "not_found"}, status=404)
        q = request.query_params
        try:
            limit = max(1, min(int(q.get("limit", 10)), 50))
            max_value = float(q["max_value"]) if q.get("max_value") else None
            max_age = int(q["max_age"]) if q.get("max_age") else None
            min_minutes = int(q.get("min_minutes", 450))
        except ValueError:
            return Response({"detail": "limit, max_age and min_minutes must be integers; max_value a number", "code": "invalid"}, status=400)

        values = dict(annotated_players().filter(current_value__isnull=False).values_list("id", "current_value"))
        try:
            out = find_similar(player, season=q.get("season") or None, limit=limit, min_minutes=min_minutes,
                               max_value=max_value, max_age=max_age, values=values)
        except SimilarityError as e:
            return Response({"detail": str(e), "code": "not_comparable"}, status=422)

        found = {p.id: p for p in players_queryset().filter(id__in=[r["playerId"] for r in out["results"]])}
        for r in out["results"]:
            r["player"] = PlayerSerializer(found[r["playerId"]]).data
        out["reference"] = PlayerSerializer(players_queryset().get(pk=player.pk)).data
        return Response(out)


class WatchlistView(APIView):
    """GET list, POST {"playerId": n} to add, DELETE /watchlist/<playerId>/ to remove."""

    def get(self, request):
        ids = list(WatchlistEntry.objects.values_list("player_id", flat=True))
        players = {p.id: p for p in players_queryset().filter(id__in=ids)}
        ordered = [players[i] for i in ids if i in players]
        return Response(PlayerSerializer(ordered, many=True).data)

    def post(self, request):
        player_id = request.data.get("playerId")
        if not Player.objects.filter(pk=player_id).exists():
            return Response({"detail": "unknown playerId"}, status=status.HTTP_404_NOT_FOUND)
        _, created = WatchlistEntry.objects.get_or_create(player_id=player_id)
        return Response({"playerId": player_id}, status=201 if created else 200)

    def delete(self, request, player_id):
        deleted, _ = WatchlistEntry.objects.filter(player_id=player_id).delete()
        return Response(status=204 if deleted else 404)


class MarketTrendsView(APIView):
    def get(self, request):
        with_value = annotated_players().filter(current_value__isnull=False)
        by_position = [
            {"position": r["position"], "value": round(r["avg"], 1), "players": r["n"]}
            for r in with_value.exclude(position="").values("position")
            .annotate(avg=Avg("current_value"), n=Count("id")).order_by("-avg")
        ]
        by_league = [
            {"league": r["club__league"], "value": round(r["avg"], 1), "players": r["n"]}
            for r in with_value.exclude(club__isnull=True).values("club__league")
            .annotate(avg=Avg("current_value"), n=Count("id")).order_by("-avg")
        ]
        by_month = [
            {"month": r["month"].strftime("%b %y"), "value": round(r["avg"], 1)}
            for r in MarketValue.objects.annotate(month=TruncMonth("date")).values("month")
            .annotate(avg=Avg("value")).order_by("month")
        ]
        overall = with_value.aggregate(avg=Avg("current_value"))["avg"]
        return Response({
            "averagePlayerValue": round(overall, 1) if overall is not None else None,
            "mostValuablePosition": by_position[0] if by_position else None,
            "averageValueByPosition": by_position,
            "averageValueByLeague": by_league,
            "averageValueByMonth": by_month,
        })


class SimulatorView(APIView):
    """POST /api/simulator/

    Single transfer:  {"playerId", "fee"?, "salary", "contractYears", "holdingYears", "scenario"}
    Portfolio:        {"budget", "maxPlayers"?}  (no playerId)
    """

    def post(self, request):
        data = request.data
        try:
            if data.get("playerId") is not None:
                return self._single(data)
            return self._portfolio(data)
        except (ValueError, TypeError) as e:
            return Response({"detail": str(e)}, status=400)

    def _single(self, data):
        player = annotated_players().filter(pk=data["playerId"]).first()
        if player is None:
            return Response({"detail": "unknown playerId"}, status=404)
        if player.predicted_value is None:
            return Response({"detail": "no prediction available for this player yet"}, status=409)
        fee = float(data.get("fee") or player.current_value or 0)
        result = simulate_transfer(
            fee=fee,
            predicted_value=player.predicted_value,
            risk_score=player.risk_score,
            salary=float(data.get("salary", 0)),
            contract_years=int(data.get("contractYears", 1)),
            holding_years=int(data.get("holdingYears", 1)),
            scenario=str(data.get("scenario", "base")),
        )
        result["player"] = {"id": player.id, "name": player.name}
        return Response(result)

    def _portfolio(self, data):
        budget = float(data["budget"]) if "budget" in data else None
        if not budget or budget <= 0:
            raise ValueError("budget (in €M) is required")
        candidates = [
            {"id": p.id, "name": p.name, "currentValue": p.current_value,
             "predictedValue": p.predicted_value, "riskScore": p.risk_score}
            for p in annotated_players().filter(predicted_value__isnull=False, current_value__isnull=False)
        ]
        max_players = int(data["maxPlayers"]) if data.get("maxPlayers") else None
        return Response(build_portfolio(candidates, budget=budget, max_players=max_players))


class HealthView(APIView):
    """GET /api/health/ - is the API up, what data does it hold and which model produced the predictions.

    The frontend shows this as its "Live model" badge, and scripts/smoke_test.py checks it after a deploy.
    Returns 503 if the database cannot be reached."""

    def get(self, request):
        import json

        from django.conf import settings
        from django.db import DatabaseError
        from django.db.models import Max

        try:
            latest_pred = Prediction.objects.order_by("-created_at").values("model_version", "created_at").first()
            values = MarketValue.objects.aggregate(n=Count("id"), latest=Max("date"))
            stats = PlayerSeasonStat.objects.aggregate(n=Count("id"), latest=Max("season"))
            metrics_by_source = list(PlayerSeasonMetrics.objects.values("source").annotate(n=Count("id"), latest=Max("season")))
            body = {
                "status": "ok",
                "database": "ok",
                "players": Player.objects.count(),
                "playersWithPrediction": Prediction.objects.values("player").distinct().count(),
                "dataSources": [
                    {"name": "Market values", "rows": values["n"],
                     "latest": values["latest"].isoformat() if values["latest"] else None},
                    {"name": "Season stats", "rows": stats["n"], "latest": stats["latest"]},
                    *[{"name": f"{m['source']} season metrics", "rows": m["n"], "latest": m["latest"]} for m in metrics_by_source],
                ],
                "predictions": {
                    "modelVersion": latest_pred["model_version"] if latest_pred else None,
                    "createdAt": latest_pred["created_at"].isoformat() if latest_pred else None,
                },
            }
        except DatabaseError as e:
            return Response({"status": "error", "database": "unreachable", "detail": str(e)[:200]}, status=503)

        model = {"trained": False}
        path = settings.MODEL_DIR / "latest.json"
        if path.exists():
            meta = json.loads(path.read_text())
            m = meta.get("metrics", {})
            model = {"trained": True, "version": meta.get("version"), "trainedAt": meta.get("trained_at"),
                     "horizonYears": meta.get("horizon_years"), "testYear": m.get("test_year"),
                     "beatsBaseline": m.get("beats_baseline"), "beatsAllBaselines": m.get("beats_all_baselines")}
        body["model"] = model
        return Response(body)


class ModelPerformanceView(APIView):
    """Metrics and feature importances of the latest trained model (for the Model Performance page)."""

    def get(self, request):
        import json

        from django.conf import settings

        path = settings.MODEL_DIR / "latest.json"
        if not path.exists():
            return Response({"trained": False})
        meta = json.loads(path.read_text())
        return Response({"trained": True, **meta})
