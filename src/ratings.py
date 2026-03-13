
import pandas as pd
import numpy as np
from src.config import CONFIG

def calculate_elo(games_df, k_factor=CONFIG["elo_k_factor"], initial_rating=CONFIG["elo_initial_rating"]):
    ratings = {}
    
    # Sort games by DayNum to process them chronologically
    games_df = games_df.sort_values(["Season", "DayNum"])
    
    elo_history = []
    
    for row in games_df.itertuples():
        season = row.Season
        team_a = row.TeamID
        team_b = row.OppTeamID
        win = row.Win
        
        # Initialize ratings if not present
        if team_a not in ratings:
            ratings[team_a] = initial_rating
        if team_b not in ratings:
            ratings[team_b] = initial_rating
            
        rating_a = ratings[team_a]
        rating_b = ratings[team_b]
        
        # Expected win probability for team A
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        
        # Update ratings
        actual_a = 1 if win == 1 else 0
        new_rating_a = rating_a + k_factor * (actual_a - expected_a)
        new_rating_b = rating_b + k_factor * ((1 - actual_a) - (1 - expected_a))
        
        ratings[team_a] = new_rating_a
        ratings[team_b] = new_rating_b
        
        elo_history.append({
            "Season": season,
            "DayNum": row.DayNum,
            "TeamID": team_a,
            "Elo": new_rating_a
        })
        
    return pd.DataFrame(elo_history), ratings

def get_latest_elo(elo_history_df):
    # Get the last Elo rating for each team in each season
    return elo_history_df.groupby(["Season", "TeamID"]).tail(1)[["Season", "TeamID", "Elo"]]
