"""Responses use the same camelCase field names as the frontend's `Player` type
(src/data/mockData.ts) so pages can switch from mock data to the API with minimal changes."""

from datetime import date, timedelta

from rest_framework import serializers

from .selectors import risk_label


def _round(value, ndigits=1):
    return None if value is None else round(value, ndigits)


class PlayerSerializer(serializers.BaseSerializer):
    """Expects objects from selectors.players_queryset() (prefetched to_attr lists)."""

    def to_representation(self, p):
        stat = p.stats_list[0] if p.stats_list else None
        values = p.values_list
        pred = p.preds_list[0] if p.preds_list else None

        current = values[-1].value if values else None
        peak = max((v.value for v in values), default=None)
        value_change = None
        if len(values) >= 2 and values[-2].value:
            value_change = _round((values[-1].value - values[-2].value) / values[-2].value * 100)

        predicted = pred.predicted_value if pred else None
        upside = None
        if predicted is not None and current:
            upside = round((predicted - current) / current * 100)

        contract_years = None
        if p.contract_expiry_year:
            contract_years = max(p.contract_expiry_year - date.today().year, 0)

        return {
            "id": p.id,
            "name": p.name,
            "age": stat.age if stat else None,
            "position": p.position or None,
            "club": p.club.name if p.club else None,
            "league": p.club.league if p.club else None,
            "nationality": p.nationality or None,
            "currentValue": current,
            "predictedValue": _round(predicted),
            # 10th-90th percentile range from the model's held-out errors (null for older models).
            "predictedLow": _round(pred.predicted_low) if pred else None,
            "predictedHigh": _round(pred.predicted_high) if pred else None,
            "upside": upside,
            "risk": risk_label(pred.risk_score) if pred else None,
            "riskScore": pred.risk_score if pred else None,
            "signal": pred.signal if pred else None,
            "xG90": stat.per90(stat.xg) if stat else None,
            "xA90": stat.per90(stat.xag) if stat else None,
            "goals90": stat.per90(stat.goals) if stat else None,
            "assists90": stat.per90(stat.assists) if stat else None,
            "progressivePasses": stat.per90(stat.progressive_passes) if stat else None,
            "progressiveCarries": stat.per90(stat.progressive_carries) if stat else None,
            "minutes": stat.minutes if stat else 0,
            "peakValue": peak,
            "valueChange": value_change,
            # No fee data is collected yet; null rather than guessing.
            "transferFee": None,
            "contractYears": contract_years,
            "confidence": pred.confidence if pred else None,
            "reasons": pred.reasons if pred else [],
        }


class PlayerDetailSerializer(PlayerSerializer):
    def to_representation(self, p):
        data = super().to_representation(p)
        pred = p.preds_list[0] if p.preds_list else None

        data["marketValueHistory"] = [
            {"date": v.date.isoformat(), "value": v.value} for v in p.values_list
        ]
        # Only one forecast point per prediction (the model's horizon), anchored at today's value.
        forecast = []
        if pred and data["currentValue"] is not None:
            horizon_date = date.today() + timedelta(days=365 * pred.horizon_years)
            forecast = [
                {"date": date.today().isoformat(), "predicted": data["currentValue"]},
                {"date": horizon_date.isoformat(), "predicted": round(pred.predicted_value, 1)},
            ]
        data["forecast"] = forecast
        data["modelVersion"] = pred.model_version if pred else None
        data["predictedAt"] = pred.created_at.isoformat() if pred else None
        data["riskFactors"] = pred.risk_factors if pred else []
        data["seasonMetrics"] = [
            {"season": m.season, "source": m.source, "team": m.team_name, "positionGroup": m.position_group,
             "matches": m.matches, "metrics": m.metrics}
            for m in getattr(p, "metrics_list", [])
        ]
        data["seasonStats"] = [
            {
                "season": s.season,
                "club": s.club.name if s.club else None,
                "age": s.age,
                "matches": s.matches,
                "minutes": s.minutes,
                "goals": s.goals,
                "assists": s.assists,
                "xg": s.xg,
                "xag": s.xag,
            }
            for s in p.stats_list
        ]
        return data
