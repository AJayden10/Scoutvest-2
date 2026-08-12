import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()  # reads the .env file into environment variables

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Load the CSV we already scraped
df = pd.read_csv("data/raw/epl_player_standard_stats.csv")
print(f"Loaded {df.shape[0]} rows from CSV")

# Push it into a table called player_stats
df.to_sql("player_stats", engine, if_exists="replace", index=False)
print("Successfully loaded data into PostgreSQL table: player_stats")