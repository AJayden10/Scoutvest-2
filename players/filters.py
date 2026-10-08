import django_filters as df
from django.db.models import Q

from .models import Player
from .selectors import RISK_LOW_MAX, RISK_MEDIUM_MAX


class PlayerFilter(df.FilterSet):
    q = df.CharFilter(method="filter_q", label="Name or club contains")
    position = df.CharFilter(field_name="position", lookup_expr="iexact")
    league = df.CharFilter(field_name="club__league", lookup_expr="iexact")
    club = df.CharFilter(field_name="club__name", lookup_expr="icontains")
    signal = df.CharFilter(field_name="signal", lookup_expr="iexact")
    risk = df.CharFilter(method="filter_risk", label="Low | Medium | High")
    min_age = df.NumberFilter(field_name="latest_age", lookup_expr="gte")
    max_age = df.NumberFilter(field_name="latest_age", lookup_expr="lte")
    min_value = df.NumberFilter(field_name="current_value", lookup_expr="gte")
    max_value = df.NumberFilter(field_name="current_value", lookup_expr="lte")
    min_upside = df.NumberFilter(field_name="upside", lookup_expr="gte")

    ordering = df.OrderingFilter(
        fields=(
            ("name", "name"),
            ("latest_age", "age"),
            ("current_value", "current_value"),
            ("predicted_value", "predicted_value"),
            ("upside", "upside"),
            ("confidence", "confidence"),
            ("risk_score", "risk_score"),
        )
    )

    class Meta:
        model = Player
        fields: list[str] = []

    def filter_q(self, queryset, name, value):
        return queryset.filter(Q(name__icontains=value) | Q(club__name__icontains=value))

    def filter_risk(self, queryset, name, value):
        value = value.lower()
        if value == "low":
            return queryset.filter(risk_score__lt=RISK_LOW_MAX)
        if value == "medium":
            return queryset.filter(risk_score__gte=RISK_LOW_MAX, risk_score__lt=RISK_MEDIUM_MAX)
        if value == "high":
            return queryset.filter(risk_score__gte=RISK_MEDIUM_MAX)
        return queryset
