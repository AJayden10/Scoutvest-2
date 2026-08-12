import ScraperFC as sfc
import pandas as pd
import os
import time

fbref = sfc.FBRefScraper(wait_time=6)
LEAGUE = 'England Premier League'
SEASONS = ["2021-2022", "2022-2023", "2023-2024"]
STAT_CATEGORY = 'standard'

os.makedirs('data/raw', exist_ok=True)

all_seasons = []
for season in SEASONS:
    print(f"Scraping {LEAGUE} {season}...")
    try:
        result = fbref.scrape_stats(year=season, league=LEAGUE, stat_category=STAT_CATEGORY)
        player_stats = result["player"]
        player_stats["season"] = season  # tag which season each row came from
        all_seasons.append(player_stats)
        print(f"  → got {player_stats.shape[0]} player rows")
    except Exception as e:
        print(f"  → FAILED for {season}: {e}")

combined = pd.concat(all_seasons, ignore_index=True)
output_path = "data/raw/epl_player_standard_stats.csv"
combined.to_csv(output_path, index=False)
print(f"\nSaved {combined.shape[0]} total rows to {output_path}")