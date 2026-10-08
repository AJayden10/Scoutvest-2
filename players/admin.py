from django.contrib import admin

from . import models


@admin.register(models.Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ("name", "position", "club", "nationality")
    list_filter = ("position",)
    search_fields = ("name",)


admin.site.register(models.Club)
admin.site.register(models.PlayerSeasonStat)
admin.site.register(models.MarketValue)
admin.site.register(models.Prediction)
admin.site.register(models.WatchlistEntry)
