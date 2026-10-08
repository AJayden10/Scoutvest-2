from datetime import date

import pandas as pd
from django.core.management import call_command
from django.test import TestCase
from rest_framework.test import APIClient

from .models import Club, MarketValue, Player, PlayerSeasonStat, Prediction, WatchlistEntry
from .services.fbref import MissingColumns, iter_records, normalize_position, resolve_columns
from .services.simulator import build_portfolio, simulate_transfer


def make_player(name, position, value, predicted, risk, signal="UNDERVALUED", minutes=1800,
                league="La Liga", club="Sevilla", age=20):
    club_obj, _ = Club.objects.get_or_create(name=club, league=league)
    p = Player.objects.create(name=name, position=position, club=club_obj, nationality="ESP")
    PlayerSeasonStat.objects.create(
        player=p, club=club_obj, season="2023-2024", age=age, minutes=minutes,
        goals=6, assists=4, xg=5.4, xag=3.6, progressive_passes=90, progressive_carries=45,
    )
    MarketValue.objects.create(player=p, date=date(2023, 7, 1), value=value * 0.8)
    MarketValue.objects.create(player=p, date=date(2024, 7, 1), value=value)
    if predicted is not None:
        Prediction.objects.create(
            player=p, model_version="test-v1", horizon_years=2, predicted_value=predicted,
            confidence=80, risk_score=risk, signal=signal, reasons=["reason a", "reason b"],
        )
    return p


class PlayerApiTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.a = make_player("Lucas Fernandez", "CM", 8.0, 24.0, 22)
        cls.b = make_player("Matteo Rossi", "Winger", 12.0, 30.0, 40, "BREAKOUT",
                            league="Serie A", club="Atalanta", age=21)
        cls.c = make_player("No Prediction", "ST", 5.0, None, 0)
        cls.client = APIClient()

    def get(self, path):
        r = self.client.get(path)
        self.assertEqual(r.status_code, 200, r.content)
        return r.json()

    def test_list_matches_frontend_shape(self):
        data = self.get("/api/players/")
        self.assertEqual(data["count"], 3)
        row = next(r for r in data["results"] if r["name"] == "Lucas Fernandez")
        self.assertEqual(row["currentValue"], 8.0)
        self.assertEqual(row["predictedValue"], 24.0)
        self.assertEqual(row["upside"], 200)
        self.assertEqual(row["risk"], "Low")
        self.assertEqual(row["signal"], "UNDERVALUED")
        self.assertEqual(row["age"], 20)
        self.assertEqual(row["xG90"], 0.27)  # 5.4 xG over 20 nineties
        self.assertEqual(row["valueChange"], 25.0)  # 6.4 -> 8.0
        self.assertEqual(row["peakValue"], 8.0)
        self.assertEqual(row["league"], "La Liga")
        self.assertEqual(len(row["reasons"]), 2)

    def test_player_without_prediction_has_nulls(self):
        row = next(r for r in self.get("/api/players/")["results"] if r["name"] == "No Prediction")
        self.assertIsNone(row["predictedValue"])
        self.assertIsNone(row["upside"])
        self.assertIsNone(row["risk"])

    def test_filters(self):
        self.assertEqual(self.get("/api/players/?position=Winger")["count"], 1)
        self.assertEqual(self.get("/api/players/?league=serie a")["count"], 1)
        self.assertEqual(self.get("/api/players/?signal=BREAKOUT")["count"], 1)
        self.assertEqual(self.get("/api/players/?risk=Low")["count"], 1)
        self.assertEqual(self.get("/api/players/?risk=Medium")["count"], 1)
        self.assertEqual(self.get("/api/players/?min_age=21")["count"], 1)
        self.assertEqual(self.get("/api/players/?q=rossi")["count"], 1)
        self.assertEqual(self.get("/api/players/?q=atalanta")["count"], 1)

    def test_ordering_puts_missing_predictions_last(self):
        names = [r["name"] for r in self.get("/api/players/?ordering=-upside")["results"]]
        self.assertEqual(names[-1], "No Prediction")
        self.assertEqual(names[0], "Lucas Fernandez")  # 200% vs 150%

    def test_detail_includes_history_and_forecast(self):
        d = self.get(f"/api/players/{self.a.id}/")
        self.assertEqual(len(d["marketValueHistory"]), 2)
        self.assertEqual(d["forecast"][-1]["predicted"], 24.0)
        self.assertEqual(d["seasonStats"][0]["season"], "2023-2024")
        self.assertEqual(d["modelVersion"], "test-v1")

    def test_compare(self):
        d = self.get(f"/api/players/compare/?ids={self.a.id},{self.b.id}")
        self.assertEqual([p["name"] for p in d], ["Lucas Fernandez", "Matteo Rossi"])
        self.assertEqual(self.client.get("/api/players/compare/?ids=1").status_code, 400)
        self.assertEqual(self.client.get("/api/players/compare/?ids=1,999999").status_code, 404)

    def test_watchlist_roundtrip(self):
        self.assertEqual(self.client.post("/api/watchlist/", {"playerId": self.a.id}, format="json").status_code, 201)
        self.assertEqual(self.client.post("/api/watchlist/", {"playerId": self.a.id}, format="json").status_code, 200)
        self.assertEqual([p["id"] for p in self.get("/api/watchlist/")], [self.a.id])
        self.assertEqual(self.client.delete(f"/api/watchlist/{self.a.id}/").status_code, 204)
        self.assertEqual(self.client.delete(f"/api/watchlist/{self.a.id}/").status_code, 404)
        self.assertEqual(WatchlistEntry.objects.count(), 0)
        self.assertEqual(self.client.post("/api/watchlist/", {"playerId": 99999}, format="json").status_code, 404)

    def test_market_trends(self):
        d = self.get("/api/market-trends/")
        self.assertEqual(d["mostValuablePosition"]["position"], "Winger")
        self.assertEqual({r["league"] for r in d["averageValueByLeague"]}, {"La Liga", "Serie A"})
        self.assertEqual(d["averagePlayerValue"], 8.3)  # (8 + 12 + 5) / 3
        self.assertEqual(len(d["averageValueByMonth"]), 2)

    def test_simulator_single_and_portfolio(self):
        r = self.client.post("/api/simulator/", {
            "playerId": self.a.id, "salary": 2, "contractYears": 4, "holdingYears": 3, "scenario": "Base",
        }, format="json")
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        # fee 8, salary 2 * 3 years -> cost 14; future 24 -> ROI (24-14)/14
        self.assertEqual(body["totalCost"], 14.0)
        self.assertAlmostEqual(body["projectedROI"], 71.4, places=1)
        self.assertAlmostEqual(body["riskAdjustedROI"], 71.4 * 0.78, places=1)
        self.assertEqual(len(body["cases"]), 3)

        r = self.client.post("/api/simulator/", {"budget": 15}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual([p["name"] for p in r.json()["players"]], ["Lucas Fernandez"])  # Rossi (12) won't fit after Lucas (8)

        self.assertEqual(self.client.post("/api/simulator/", {"playerId": self.c.id}, format="json").status_code, 409)
        self.assertEqual(self.client.post("/api/simulator/", {}, format="json").status_code, 400)


class SimulatorUnitTests(TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            simulate_transfer(fee=0, predicted_value=10, risk_score=10)
        with self.assertRaises(ValueError):
            simulate_transfer(fee=5, predicted_value=10, risk_score=10, scenario="nope")

    def test_loss_when_predicted_below_fee(self):
        r = simulate_transfer(fee=10, predicted_value=6, risk_score=20)
        self.assertLess(r["projectedROI"], 0)

    def test_portfolio_skips_negative_scores_and_respects_max(self):
        cands = [
            {"id": 1, "name": "A", "currentValue": 5, "predictedValue": 20, "riskScore": 10},
            {"id": 2, "name": "B", "currentValue": 5, "predictedValue": 15, "riskScore": 10},
            {"id": 3, "name": "C", "currentValue": 5, "predictedValue": 3, "riskScore": 10},
        ]
        self.assertEqual([p["name"] for p in build_portfolio(cands, budget=100)["players"]], ["A", "B"])
        self.assertEqual(len(build_portfolio(cands, budget=100, max_players=1)["players"]), 1)


class FbrefImportTests(TestCase):
    def frame(self):
        return pd.DataFrame({
            "Player": ["Bukayo Saka", "Player", "Bukayo Saka", "Aaron Ramsdale"],
            "Nation": ["eng ENG", None, "eng ENG", "eng ENG"],
            "Pos": ["FW,MF", None, "FW", "GK"],
            "Squad": ["Arsenal", None, "Arsenal", "Arsenal"],
            "Age": ["22-110", None, "21-120", "25"],
            "Born": [2001, None, 2001, 1998],
            "MP": [38, None, 38, 37],
            "Starts": [35, None, 36, 37],
            "Min": ["3,000", None, 3100, 3330],
            "Gls": [14, None, 11, 0],
            "Ast": [11, None, 7, 0],
            "xG": [11.2, None, 10.1, 0],
            "xAG": [8.0, None, 6.5, 0],
            "PrgC": [100, None, 90, 5],
            "PrgP": [120, None, 110, 80],
            "Gls.1": [0.42, None, 0.31, 0],  # per-90 duplicate must not override totals
            "season": ["2023-2024", None, "2022-2023", "2023-2024"],
        })

    def test_parsing(self):
        records = list(iter_records(self.frame()))
        self.assertEqual(len(records), 3)  # blank/header row skipped
        saka = records[0]
        self.assertEqual((saka.position, saka.age, saka.minutes, saka.goals), ("AM", 22, 3000, 14.0))
        self.assertEqual(saka.nationality, "ENG")
        self.assertEqual(normalize_position("GK"), "GK")
        self.assertEqual(normalize_position("zz"), "")

    def test_missing_required_columns(self):
        with self.assertRaises(MissingColumns):
            resolve_columns(pd.DataFrame({"Player": ["x"]}))

    def test_command_is_idempotent(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "stats.csv")
            self.frame().to_csv(path, index=False)
            call_command("import_fbref", csv=path)
            call_command("import_fbref", csv=path)
        self.assertEqual(Player.objects.count(), 2)
        self.assertEqual(PlayerSeasonStat.objects.count(), 3)
        saka = Player.objects.get(name="Bukayo Saka")
        self.assertEqual(saka.club.name, "Arsenal")
        self.assertEqual(saka.club.league, "Premier League")
        self.assertEqual(saka.season_stats.count(), 2)
