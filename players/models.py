from django.db import models


class Position(models.TextChoices):
    """The position set the frontend uses (see src/data/mockData.ts)."""

    GK = "GK", "Goalkeeper"
    CB = "CB", "Centre-back"
    FB = "FB", "Full-back"
    DM = "DM", "Defensive midfielder"
    CM = "CM", "Central midfielder"
    AM = "AM", "Attacking midfielder"
    WINGER = "Winger", "Winger"
    ST = "ST", "Striker"


class Signal(models.TextChoices):
    UNDERVALUED = "UNDERVALUED", "Undervalued"
    BREAKOUT = "BREAKOUT", "Breakout"
    STRONG_BUY = "STRONG BUY", "Strong buy"
    OVERVALUED = "OVERVALUED", "Overvalued"
    MONITOR = "MONITOR", "Monitor"


class Club(models.Model):
    name = models.CharField(max_length=120)
    league = models.CharField(max_length=80)
    country = models.CharField(max_length=80, blank=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["name", "league"], name="uniq_club_league")]
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.league})"


class Player(models.Model):
    name = models.CharField(max_length=160, db_index=True)
    nationality = models.CharField(max_length=80, blank=True)
    position = models.CharField(max_length=10, choices=Position.choices, blank=True)
    club = models.ForeignKey(Club, null=True, blank=True, on_delete=models.SET_NULL, related_name="players")
    birth_year = models.PositiveSmallIntegerField(null=True, blank=True)
    contract_expiry_year = models.PositiveSmallIntegerField(null=True, blank=True)
    # External identifiers, so re-running the scrapers updates rather than duplicates.
    fbref_id = models.CharField(max_length=32, null=True, blank=True, unique=True)
    transfermarkt_id = models.CharField(max_length=32, null=True, blank=True, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class PlayerSeasonStat(models.Model):
    """One player-club-season row from FBref's standard stats table."""

    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="season_stats")
    club = models.ForeignKey(Club, null=True, blank=True, on_delete=models.SET_NULL, related_name="season_stats")
    season = models.CharField(max_length=9)  # e.g. "2023-2024"
    age = models.PositiveSmallIntegerField(null=True, blank=True)
    matches = models.PositiveSmallIntegerField(default=0)
    starts = models.PositiveSmallIntegerField(default=0)
    minutes = models.PositiveIntegerField(default=0)
    goals = models.FloatField(default=0)
    assists = models.FloatField(default=0)
    xg = models.FloatField(null=True, blank=True)
    xag = models.FloatField(null=True, blank=True)
    progressive_passes = models.FloatField(null=True, blank=True)
    progressive_carries = models.FloatField(null=True, blank=True)
    # Anything else from the scrape that the models may want later
    # (pressures, injuries, agent, ...) goes here without a schema change.
    extra = models.JSONField(default=dict, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["player", "season", "club"], name="uniq_player_season_club")
        ]
        ordering = ["-season"]

    @property
    def nineties(self):
        return self.minutes / 90 if self.minutes else 0

    def per90(self, value):
        n = self.nineties
        return round(value / n, 3) if (n and value is not None) else None

    def __str__(self):
        return f"{self.player} {self.season}"


class MarketValue(models.Model):
    """A dated market valuation in €M (Transfermarkt-style)."""

    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="market_values")
    date = models.DateField()
    value = models.FloatField(help_text="Market value in millions of euros")
    source = models.CharField(max_length=40, default="transfermarkt")

    class Meta:
        constraints = [models.UniqueConstraint(fields=["player", "date", "source"], name="uniq_value_point")]
        ordering = ["date"]

    def __str__(self):
        return f"{self.player} {self.date}: €{self.value}M"


class Prediction(models.Model):
    """Output of a trained model for one player (written by the ML pipeline)."""

    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="predictions")
    model_version = models.CharField(max_length=60)
    horizon_years = models.PositiveSmallIntegerField(default=2)
    predicted_value = models.FloatField(help_text="Predicted market value in €M at the horizon")
    confidence = models.PositiveSmallIntegerField(default=0, help_text="0-100")
    risk_score = models.PositiveSmallIntegerField(default=50, help_text="0-100, higher is riskier")
    signal = models.CharField(max_length=20, choices=Signal.choices, default=Signal.MONITOR)
    reasons = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["player", "-created_at"])]

    def __str__(self):
        return f"{self.player} -> €{self.predicted_value}M ({self.model_version})"


class WatchlistEntry(models.Model):
    """Single shared watchlist for now; add a user FK when auth is introduced."""

    player = models.OneToOneField(Player, on_delete=models.CASCADE, related_name="watchlist_entry")
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-added_at"]
