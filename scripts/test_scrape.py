import ScraperFC as sfc

fbref = sfc.FBref()

seasons = fbref.get_valid_seasons("England Premier League")
print("Valid EPL seasons (sample):", list(seasons.items())[:5])

result = fbref.scrape_stats(year="2023-2024", league="England Premier League", stat_category="standard")

player_stats = result["player"]
print(player_stats.shape)      # (rows, columns)
print(player_stats.head())     # first few rows
print(player_stats.columns.tolist())  # full column list, so we know what's available