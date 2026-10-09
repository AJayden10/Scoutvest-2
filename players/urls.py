from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("players", views.PlayerViewSet, basename="player")

urlpatterns = [
    path("", include(router.urls)),
    path("watchlist/", views.WatchlistView.as_view()),
    path("watchlist/<int:player_id>/", views.WatchlistView.as_view()),
    path("market-trends/", views.MarketTrendsView.as_view()),
    path("simulator/", views.SimulatorView.as_view()),
    path("model/performance/", views.ModelPerformanceView.as_view()),
]
