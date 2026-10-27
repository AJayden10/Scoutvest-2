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
        total = self.get("/api/players/")["count"]
        self.assertEqual(self.get("/api/players/?predicted=true")["count"], total - 1)  # all but "No Prediction"

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

    def test_health_reports_data_and_untrained_model(self):
        with tempfile.TemporaryDirectory() as tmp, override_settings(MODEL_DIR=Path(tmp)):
            d = self.get("/api/health/")
        self.assertEqual((d["status"], d["database"], d["players"], d["playersWithPrediction"]), ("ok", "ok", 3, 2))
        self.assertEqual(d["model"], {"trained": False})
        values = next(s for s in d["dataSources"] if s["name"] == "Market values")
        self.assertEqual((values["rows"], values["latest"]), (6, "2024-07-01"))

    def test_errors_have_one_json_shape(self):
        r = self.client.get("/api/players/999999/")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["code"], "not_found")
        self.assertIn("detail", r.json())
        self.assertIn("Server-Timing", r.headers)

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


class TransfermarktImportTests(TestCase):
    def setUp(self):
        import tempfile, os
        self.dir = tempfile.mkdtemp()
        pd.DataFrame({
            "player_id": [1, 2, 3],
            "name": ["Rémi Delacroix", "Sam Lee", "Sam Lee"],
            "date_of_birth": ["2001-03-04", "1999-01-01", "2004-05-05"],
            "contract_expiration_date": ["2027-06-30", None, None],
        }).to_csv(os.path.join(self.dir, "players.csv"), index=False)
        pd.DataFrame({
            "player_id": [1, 1, 2, 3, 9],
            "date": ["2022-06-01", "2023-06-01", "2023-06-01", "2023-06-01", "2023-06-01"],
            "market_value_in_eur": [5_000_000, 8_500_000, 1_000_000, 2_000_000, 3_000_000],
        }).to_csv(os.path.join(self.dir, "player_valuations.csv"), index=False)

    def run_import(self, *extra):
        import os
        call_command("import_transfermarkt", "--players", os.path.join(self.dir, "players.csv"),
                     "--valuations", os.path.join(self.dir, "player_valuations.csv"), *extra)

    def test_accent_insensitive_match_and_values(self):
        p = Player.objects.create(name="Remi Delacroix", birth_year=2001)
        self.run_import("--since", "2020-01-01")
        p.refresh_from_db()
        self.assertEqual(p.transfermarkt_id, "1")
        self.assertEqual(p.contract_expiry_year, 2027)
        self.assertEqual([mv.value for mv in p.market_values.all()], [5.0, 8.5])

    def test_ambiguous_name_uses_birth_year_or_is_skipped(self):
        a = Player.objects.create(name="Sam Lee", birth_year=2004)
        b = Player.objects.create(name="Sam Lee", birth_year=None)
        self.run_import()
        a.refresh_from_db(); b.refresh_from_db()
        self.assertEqual(a.transfermarkt_id, "3")
        self.assertIsNone(b.transfermarkt_id)

    def test_rerun_does_not_duplicate_and_dry_run_writes_nothing(self):
        p = Player.objects.create(name="Remi Delacroix", birth_year=2001)
        self.run_import("--dry-run")
        self.assertEqual(MarketValue.objects.count(), 0)
        self.run_import(); self.run_import()
        self.assertEqual(p.market_values.count(), 2)


# ---------------------------------------------------------------- ML pipeline
import tempfile
from pathlib import Path

import numpy as np
from django.test import override_settings

from .ml import signals
from .ml.features import FEATURES, build_frame, labelled, season_end
from .ml.training import NotEnoughData, train


def make_synthetic_league(n=140, seed=7):
    """Players whose value really does follow age and output, so a model has something to learn."""
    rng = np.random.default_rng(seed)
    club = Club.objects.create(name="Synth FC", league="Premier League")
    for i in range(n):
        age0 = int(rng.integers(18, 33))
        quality = rng.uniform(0.1, 0.6)
        p = Player.objects.create(name=f"Player {i}", position=["CM", "ST", "CB", "Winger"][i % 4],
                                  club=club, birth_year=2022 - age0, contract_expiry_year=int(rng.integers(2023, 2028)))
        base = float(np.exp(rng.normal(1.8, 0.8)))
        growth = 0.35 * (24 - age0) / 6 + 1.5 * (quality - 0.3)  # young + productive -> value rises
        # Enough seasons that an embargoed time split (train labels known before the test year) still has data.
        for yr in range(2019, 2026):
            t = yr - 2019
            MarketValue.objects.create(player=p, date=date(yr, 6, 30), value=round(base * np.exp(growth * t + rng.normal(0, 0.1)), 2))
        for yr in (2020, 2021, 2022, 2023, 2024):
            PlayerSeasonStat.objects.create(
                player=p, club=club, season=f"{yr - 1}-{yr}", age=age0 + (yr - 2022), minutes=int(rng.integers(900, 3000)),
                goals=quality * 20, assists=quality * 10, xg=quality * 18, xag=quality * 9,
                progressive_passes=quality * 150, progressive_carries=quality * 80,
            )


class FeatureTests(TestCase):
    def test_season_end_and_frame_labels(self):
        self.assertEqual(season_end("2023-2024"), date(2024, 6, 30))
        make_synthetic_league(20)
        df = build_frame(2)
        self.assertEqual(set(FEATURES) - set(df.columns), set())
        # 2024 anchors have no value two years later (2026), so they stay unlabelled
        self.assertTrue(df[df.anchor_year == 2024]["target_value"].isna().all())
        self.assertFalse(labelled(df).empty)

    def test_features_use_only_the_past(self):
        make_synthetic_league(5)
        df = build_frame(2)
        row = df[(df.anchor_year == 2023)].iloc[0]
        mv = MarketValue.objects.get(player_id=row.player_id, date=date(2023, 6, 30))
        self.assertAlmostEqual(row.current_value, mv.value)

    def test_current_value_never_comes_from_after_the_season(self):
        club = Club.objects.create(name="Leak FC", league="A-League Men")
        p = Player.objects.create(name="Late Riser", position="ST", club=club, birth_year=2000)
        PlayerSeasonStat.objects.create(player=p, club=club, season="2022-2023", age=23, minutes=2000, goals=10)
        MarketValue.objects.create(player=p, date=date(2023, 8, 15), value=9.0)  # after the 30 June anchor
        self.assertTrue(build_frame(2).empty)  # no valuation known at the anchor -> no row, not a leaked one
        MarketValue.objects.create(player=p, date=date(2023, 4, 1), value=2.0)
        self.assertEqual(build_frame(2).iloc[0].current_value, 2.0)


class TrainingTests(TestCase):
    def test_model_learns_the_pattern_and_beats_the_baseline(self):
        make_synthetic_league(140)
        res = train(build_frame(2), fast=True)
        self.assertTrue(res.metrics["beats_baseline"], res.metrics)
        self.assertGreater(res.metrics["test_rows"], 10)
        self.assertEqual(res.metrics["test_year"], 2023)
        self.assertEqual(res.metrics["cutoff_year"], 2021)  # 2023 test - 2-year embargo

    def test_embargo_keeps_test_period_labels_out_of_training(self):
        make_synthetic_league(60)
        res = train(build_frame(2), fast=True)
        # every training label (anchor + horizon) must be known by the test anchor
        self.assertTrue(all(y + 2 <= res.metrics["test_year"] for y in res.metrics["train_years"]), res.metrics)
        loose = train(build_frame(2), fast=True, embargo=False)
        self.assertIn(2022, loose.metrics["train_years"])
        self.assertNotIn(2022, res.metrics["train_years"])

    def test_metrics_include_baselines_segments_and_checked_interval(self):
        make_synthetic_league(140)
        m = train(build_frame(2), fast=True).metrics
        self.assertEqual(set(m["baselines"]), {"no_change", "median_growth", "age_curve"})
        self.assertIn("beats_all_baselines", m)
        self.assertTrue(m["segments"]["position"])
        self.assertTrue(m["segments"]["value_band"])
        iv = m["interval"]
        self.assertLess(iv["log_low"], iv["log_high"])
        self.assertTrue(0 <= iv["checked_coverage"] <= 1)
        self.assertGreater(iv["checked_on_rows"], 0)

    def test_too_little_data_is_reported(self):
        make_synthetic_league(3)
        with self.assertRaises(NotEnoughData):
            train(build_frame(2), fast=True)


class SignalTests(TestCase):
    def test_classify(self):
        self.assertEqual(signals.classify(upside=-30, age=25, risk=30, peer_median=-5), "OVERVALUED")
        self.assertEqual(signals.classify(upside=-30, age=33, risk=30, peer_median=-28), "MONITOR")  # normal decline for his age
        self.assertEqual(signals.classify(upside=150, age=24, risk=30), "STRONG BUY")
        self.assertEqual(signals.classify(upside=150, age=24, risk=70), "UNDERVALUED")
        self.assertEqual(signals.classify(upside=60, age=19, risk=50), "BREAKOUT")
        self.assertEqual(signals.classify(upside=20, age=25, risk=20), "MONITOR")

    def test_risk_and_confidence_move_the_right_way(self):
        kw = dict(age=24, contract_years_left=4, value_growth_12m=0.1, mean_abs_log_error=0.3)
        self.assertGreater(signals.risk_score(minutes=500, **kw), signals.risk_score(minutes=3000, **kw))
        self.assertGreater(signals.risk_score(minutes=2000, **{**kw, "age": 32}), signals.risk_score(minutes=2000, **kw))
        self.assertGreater(signals.confidence(minutes=3000, mean_abs_log_error=0.2), signals.confidence(minutes=600, mean_abs_log_error=0.2))
        self.assertTrue(0 <= signals.risk_score(minutes=0, age=40, contract_years_left=0, value_growth_12m=3, mean_abs_log_error=2) <= 100)

    def test_risk_breakdown_adds_up_and_admits_missing_data(self):
        kw = dict(minutes=800, age=29, contract_years_left=float("nan"), value_growth_12m=None, mean_abs_log_error=0.4)
        parts = signals.risk_breakdown(**kw)
        self.assertEqual(round(sum(p["points"] for p in parts)), signals.risk_score(**kw))
        contract = next(p for p in parts if p["factor"] == "Contract")
        self.assertEqual((contract["points"], contract["detail"]), (0.0, "No contract data"))


class PipelineCommandTests(TestCase):
    def test_train_then_predict_end_to_end_and_api(self):
        make_synthetic_league(140)
        with tempfile.TemporaryDirectory() as tmp, override_settings(MODEL_DIR=Path(tmp)):
            call_command("train_model", "--fast", "--name", "test-v1")
            call_command("generate_predictions")
            self.assertEqual(Prediction.objects.count(), 140)
            pr = Prediction.objects.first()
            self.assertEqual(pr.model_version, "test-v1")
            self.assertIn(pr.signal, {"STRONG BUY", "UNDERVALUED", "BREAKOUT", "MONITOR", "OVERVALUED"})
            self.assertTrue(0 <= pr.risk_score <= 100 and 0 <= pr.confidence <= 100)
            self.assertTrue(1 <= len(pr.reasons) <= 3)
            self.assertTrue(pr.predicted_low < pr.predicted_value < pr.predicted_high)
            self.assertEqual(round(sum(f["points"] for f in pr.risk_factors)), pr.risk_score)
            self.assertTrue((Path(tmp) / "test-v1_holdout.csv").exists())
            health = APIClient().get("/api/health/").json()
            self.assertEqual(health["predictions"]["modelVersion"], "test-v1")
            self.assertTrue(health["model"]["trained"])
            self.assertEqual(health["playersWithPrediction"], 140)
            detail = APIClient().get(f"/api/players/{pr.player_id}/").json()
            self.assertEqual(detail["predictedLow"], round(pr.predicted_low, 1))
            self.assertTrue(detail["riskFactors"])
            r = APIClient().get("/api/model/performance/")
            self.assertTrue(r.json()["trained"])
            self.assertIn("mae", r.json()["metrics"])
            r = APIClient().get("/api/players/?page_size=5")
            self.assertEqual(r.status_code, 200)

    def test_predict_without_model_gives_clear_error(self):
        from django.core.management.base import CommandError
        with tempfile.TemporaryDirectory() as tmp, override_settings(MODEL_DIR=Path(tmp)):
            with self.assertRaises(CommandError):
                call_command("generate_predictions")
            self.assertEqual(APIClient().get("/api/model/performance/").json(), {"trained": False})


class TmDatasetImportTests(TestCase):
    def make_dir(self):
        d = tempfile.mkdtemp()
        pd.DataFrame({"club_id": [1, 2], "name": ["Ajax", "Some Cup Club"], "domestic_competition_id": ["NL1", "XX9"]}).to_csv(f"{d}/clubs.csv", index=False)
        pd.DataFrame({
            "player_id": [10, 11], "name": ["Jorthy Mokio", "Not In League"], "date_of_birth": ["2007-03-01", "1990-01-01"],
            "country_of_citizenship": ["Belgium", "Spain"], "sub_position": ["Defensive Midfield", "Left-Back"],
            "position": ["Midfield", "Defender"], "contract_expiration_date": ["2028-06-30", None], "current_club_id": [1, 2],
        }).to_csv(f"{d}/players.csv", index=False)
        pd.DataFrame({
            "game_id": [1, 2, 3, 4], "player_id": [10, 10, 10, 11], "player_club_id": [1, 1, 1, 2],
            "date": ["2023-09-02", "2024-03-10", "2024-07-20", "2023-09-02"],  # last one is the next season
            "yellow_cards": [0, 1, 0, 0], "red_cards": [0, 0, 0, 0], "goals": [1, 0, 2, 5], "assists": [0, 1, 0, 0],
            "minutes_played": [90, 60, 90, 90],
        }).to_csv(f"{d}/appearances.csv", index=False)
        pd.DataFrame({"player_id": [10, 10, 11], "date": ["2023-06-30", "2024-06-30", "2024-06-30"],
                      "market_value_in_eur": [2_000_000, 5_000_000, 1_000_000]}).to_csv(f"{d}/player_valuations.csv", index=False)
        return d

    def test_import_builds_players_stats_and_values(self):
        d = self.make_dir()
        call_command("import_tm_dataset", "--dir", d, "--since-season", "2022", "--min-minutes", "30", "--leagues", "NL1")
        p = Player.objects.get(transfermarkt_id="10")
        self.assertEqual((p.position, p.birth_year, p.nationality, p.club.league), ("DM", 2007, "Belgium", "Eredivisie"))
        self.assertFalse(Player.objects.filter(transfermarkt_id="11").exists())  # club is not in a chosen league
        s = {x.season: x for x in p.season_stats.all()}
        self.assertEqual(set(s), {"2023-2024", "2024-2025"})  # July starts a new season
        self.assertEqual((s["2023-2024"].minutes, s["2023-2024"].goals, s["2023-2024"].assists, s["2023-2024"].age), (150, 1, 1, 17))
        self.assertIsNone(s["2023-2024"].xg)
        self.assertEqual([m.value for m in p.market_values.all()], [2.0, 5.0])

    def test_rerun_is_idempotent_and_missing_files_are_reported(self):
        from django.core.management.base import CommandError
        d = self.make_dir()
        for _ in range(2):
            call_command("import_tm_dataset", "--dir", d, "--since-season", "2022", "--min-minutes", "30", "--leagues", "NL1")
        self.assertEqual(Player.objects.count(), 1)
        self.assertEqual(PlayerSeasonStat.objects.count(), 2)
        self.assertEqual(MarketValue.objects.count(), 2)
        with self.assertRaises(CommandError):
            call_command("import_tm_dataset", "--dir", tempfile.mkdtemp())


class UnderstatTests(TestCase):
    RAW = [
        {"player_name": "Erling Haaland", "team_title": "Manchester City", "time": "2800", "games": "33",
         "goals": "27", "assists": "5", "xG": "25.4", "xA": "3.1", "npxG": "22.0", "shots": "120",
         "key_passes": "20", "xGChain": "28.5", "xGBuildup": "3.2"},
        {"player_name": "Nobody Known", "team_title": "X", "time": "100", "games": "2", "goals": "0", "assists": "0",
         "xG": "0.1", "xA": "0", "npxG": "0.1", "shots": "1", "key_passes": "0", "xGChain": "0", "xGBuildup": "0"},
    ]

    def test_parse_converts_strings(self):
        from players.services import understat as us
        rows = us.parse_players(self.RAW)
        self.assertEqual(rows[0]["minutes"], 2800)
        self.assertAlmostEqual(rows[0]["xg"], 25.4)
        self.assertEqual(us.norm("Álvaro Morata-Ünal"), "alvaro morata unal")

    def test_import_matches_by_name_and_minutes_and_feeds_features(self):
        import json, tempfile
        from pathlib import Path
        from django.core.management import call_command
        club = Club.objects.create(name="Manchester City", league="Premier League")
        p = Player.objects.create(name="Erling Haaland", position="ST", club=club, birth_year=2000)
        s = PlayerSeasonStat.objects.create(player=p, club=club, season="2023-2024", age=23, minutes=2790, goals=27)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "EPL_2023.json").write_text(json.dumps(self.RAW))
            call_command("import_understat", dir=d, stdout=__import__("io").StringIO())
        s.refresh_from_db()
        self.assertAlmostEqual(s.xg, 25.4)
        self.assertAlmostEqual(s.extra["xgchain"], 28.5)

    def test_ablation_excludes_features(self):
        make_synthetic_league(140)
        res = train(build_frame(2), fast=True, exclude=["xg90", "npxg90"])
        self.assertNotIn("xg90", res.features)
        self.assertNotIn("xg90", res.importances)


# ---------------------------------------------------------------- similarity + SkillCorner
from .models import PlayerSeasonMetrics
from .services import skillcorner as sc
from .services.similarity import SimilarityError, find_similar


def make_mid(name, position="CM", minutes=2000, goals=2, assists=3, metrics=None, value=1.0, age=24, club="Sydney FC"):
    club_obj, _ = Club.objects.get_or_create(name=club, league="A-League Men")
    p = Player.objects.create(name=name, position=position, club=club_obj, birth_year=2024 - age)
    PlayerSeasonStat.objects.create(player=p, club=club_obj, season="2024-2025", age=age, minutes=minutes,
                                    goals=goals, assists=assists)
    if metrics is not None:
        PlayerSeasonMetrics.objects.create(player=p, season="2024-2025", source="skillcorner", metrics=metrics)
    if value is not None:
        MarketValue.objects.create(player=p, date=date(2025, 6, 1), value=value)
    return p


def runner(sprints, passes, top_speed):
    return {"sprints": sprints, "passes": passes, "top_speed": top_speed, "runs": sprints / 2}


class SimilarityTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.ref = make_mid("Reference Mid", metrics=runner(20, 40, 31), goals=2, assists=3)
        cls.twin = make_mid("Near Twin", metrics=runner(20.5, 41, 31.2), goals=2, assists=3, value=0.8)
        cls.other = make_mid("Different Mid", metrics=runner(8, 70, 27), goals=0, assists=8, value=0.5)
        cls.mid3 = make_mid("Average Mid", metrics=runner(14, 55, 29), goals=1, assists=5, value=0.6)
        cls.pricey = make_mid("Pricey Twin", position="DM", metrics=runner(19.5, 39, 31), goals=2, assists=3, value=9.0)
        cls.striker = make_mid("Same Numbers Striker", position="ST", metrics=runner(20, 40, 31), goals=2, assists=3)
        cls.bench = make_mid("Bench Twin", minutes=200, metrics=runner(20, 40, 31), goals=0, assists=0)

    def values(self):
        return {mv.player_id: mv.value for mv in MarketValue.objects.all()}

    def names(self, out):
        return [Player.objects.get(pk=r["playerId"]).name for r in out["results"]]

    def test_ranks_by_closeness_within_position_group(self):
        out = find_similar(self.ref, values=self.values())
        names = self.names(out)
        self.assertEqual(out["positionGroup"], "MID")  # CM and DM compare together
        self.assertEqual(set(names[:2]), {"Pricey Twin", "Near Twin"})
        self.assertEqual(names[-1], "Different Mid")
        self.assertNotIn("Same Numbers Striker", names)  # other position group
        self.assertNotIn("Bench Twin", names)  # below min minutes
        sims = [r["similarity"] for r in out["results"]]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertTrue(all(0 < s <= 100 for s in sims))

    def test_budget_and_explanations(self):
        out = find_similar(self.ref, values=self.values(), max_value=1.0)
        names = self.names(out)
        self.assertEqual(names[0], "Near Twin")
        self.assertNotIn("Pricey Twin", names)
        top = out["results"][0]
        self.assertEqual(len(top["closest"]), 3)
        self.assertEqual(len(top["differences"]), 2)
        self.assertTrue(all({"label", "reference", "candidate", "gap"} <= set(c) for c in top["closest"]))
        self.assertTrue({"goals90", "sprints", "passes"} <= {f["key"] for f in out["features"]})

    def test_player_without_stats_is_reported(self):
        lonely = Player.objects.create(name="No Stats", position="CM")
        with self.assertRaises(SimilarityError):
            find_similar(lonely)

    def test_api_endpoint(self):
        r = APIClient().get(f"/api/players/{self.ref.id}/similar/?max_value=1&limit=2")
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual(len(body["results"]), 2)
        self.assertEqual(body["results"][0]["player"]["name"], "Near Twin")
        self.assertEqual(body["reference"]["name"], "Reference Mid")
        self.assertEqual(APIClient().get("/api/players/999999/similar/").status_code, 404)
        self.assertEqual(APIClient().get(f"/api/players/{self.ref.id}/similar/?max_value=abc").status_code, 400)
        lonely = Player.objects.create(name="No Stats", position="CM")
        self.assertEqual(APIClient().get(f"/api/players/{lonely.id}/similar/").json()["code"], "not_comparable")


class SkillCornerImportTests(TestCase):
    def write(self, d):
        common = {"competition_name": "AUS - A-League", "season_name": "2024/2025"}
        pd.DataFrame([
            {**common, "player_id": 4322, "player_name": "Hiroki Sakai", "player_birthdate": "1990-04-12",
             "team_name": "Auckland FC", "position_group": "Full Back", "count_match": 21, "minutes_full_all": 98,
             "sprint_count_full_all": 20, "psv99": 31.0, "total_metersperminute_full_all": 110},
            {**common, "player_id": 4322, "player_name": "Hiroki Sakai", "player_birthdate": "1990-04-12",
             "team_name": "Auckland FC", "position_group": "Central Defender", "count_match": 3, "minutes_full_all": 99,
             "sprint_count_full_all": 12, "psv99": 30.0, "total_metersperminute_full_all": 100},
            {**common, "player_id": 2858, "player_name": "Douglas  Costa de Souza", "player_birthdate": "1990-09-14",
             "team_name": "Sydney Football Club", "position_group": "Wide Attacker", "count_match": 8, "minutes_full_all": 94,
             "sprint_count_full_all": 15, "psv99": 32.0, "total_metersperminute_full_all": 105},
            {**common, "player_id": 9999, "player_name": "New Youngster", "player_birthdate": "2006-01-01",
             "team_name": "Macarthur FC", "position_group": "Midfield", "count_match": 5, "minutes_full_all": 80,
             "sprint_count_full_all": 9, "psv99": 29.0, "total_metersperminute_full_all": 115},
        ]).to_csv(f"{d}/aus1league_physicalaggregates_20242025.csv", index=False)
        pd.DataFrame([
            {**common, "player_id": 4322, "player_name": "Hiroki Sakai", "player_birthdate": "1990-04-12",
             "team_name": "Auckland FC", "position_group": "Full Back", "performance_count": 13,
             "pass_count_attempted_p30tip": 30, "pass_pct_completed": 85},
        ]).to_csv(f"{d}/aus1league_passingaggregates_20242025.csv", index=False)

    def test_weighted_profiles_matching_and_idempotence(self):
        import io
        d = tempfile.mkdtemp()
        self.write(d)
        club = Club.objects.create(name="Auckland FC", league="A-League Men")
        sakai = Player.objects.create(name="Hiroki Sakai", birth_year=1990, position="FB", club=club, transfermarkt_id="1")
        costa = Player.objects.create(name="Douglas Costa", birth_year=1990, position="Winger", transfermarkt_id="2")
        Player.objects.create(name="Douglas Costa", birth_year=1985)  # same name, wrong birth year
        for _ in range(2):
            call_command("import_skillcorner", "--dir", d, stdout=io.StringIO())

        m = PlayerSeasonMetrics.objects.get(player=sakai)
        self.assertEqual((m.season, m.position_group, m.matches), ("2024-2025", "Full Back", 24))
        self.assertAlmostEqual(m.metrics["sprints"], (20 * 21 + 12 * 3) / 24, places=2)  # match-weighted
        self.assertEqual(m.metrics["pass_completion"], 85)
        self.assertEqual(PlayerSeasonMetrics.objects.get(player=costa).position_group, "Wide Attacker")  # token match
        new = Player.objects.get(skillcorner_id="9999")
        self.assertEqual((new.position, new.birth_year, new.club.league), ("CM", 2006, "A-League Men"))
        self.assertEqual(PlayerSeasonMetrics.objects.count(), 3)
        self.assertEqual(Player.objects.count(), 4)

    def test_no_create_and_missing_dir(self):
        import io
        from django.core.management.base import CommandError
        d = tempfile.mkdtemp()
        self.write(d)
        call_command("import_skillcorner", "--dir", d, "--no-create", stdout=io.StringIO())
        self.assertEqual(Player.objects.count(), 0)
        with self.assertRaises(CommandError):
            call_command("import_skillcorner", "--dir", tempfile.mkdtemp())

    def test_ambiguous_names_are_not_guessed(self):
        a = Player(name="Sam Lee", birth_year=2000)
        b = Player(name="Sam Lee", birth_year=2000)
        prof = sc.Profile("1", "Sam Lee", 2000, "X", "Midfield", "2024-2025", 3)
        player, rule = sc.match_player(prof, {("sam lee", 2000): [a, b]}, {2000: [a, b]})
        self.assertIsNone(player)
        self.assertEqual(rule, "ambiguous")
