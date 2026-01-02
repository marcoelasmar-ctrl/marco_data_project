"""Import modules """
import os
import glob
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, "data", "raw")


def get_season_id(date): 
    """Determines the season year based on the match date."""
    if date.month >= 8:
        return date.year
    else:
        return date.year - 1

def calculate_elo_ratings(match_data, k_factor=20, home_advantage=100):
    """
    Calculates ELO ratings for all teams.
    Returns the dataframe with 'HomeElo' and 'AwayElo' added.
    """
    teams = np.unique(np.concatenate([match_data['HomeTeam'], match_data['AwayTeam']]))  
    elo_dict = {team: 1500 for team in teams} 
    
    home_elos = []
    away_elos = []
    
    for _, row in match_data.iterrows():  
        home, away = row['HomeTeam'], row['AwayTeam'] 
        result = row['FTR'] 
        
        r_home = elo_dict[home]  
        r_away = elo_dict[away]
        
        home_elos.append(r_home)
        away_elos.append(r_away)
        
        dr = (r_home + home_advantage) - r_away 
        e_home = 1 / (1 + 10 ** (-dr / 400)) 
        
        if result == 'H':
            s_home = 1
        elif result == 'D':
            s_home = 0.5
        else:
            s_home = 0
            
        elo_dict[home] += k_factor * (s_home - e_home) 
        elo_dict[away] += k_factor * ((1 - s_home) - (1 - e_home)) 
        
    match_data['HomeElo'] = home_elos 
    match_data['AwayElo'] = away_elos 
    return match_data 

def load_and_process_data(data_dir=DATA_DIR_DEFAULT, window=5): 

    # --- 1. Load Data ---
    print("Looking for files in:", os.path.abspath(data_dir)) 
    all_files = glob.glob(os.path.join(data_dir, "serieA*.csv"))
    print("Found files:", all_files)
    df_list = [] 
    for filename in sorted(all_files):
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df_list.append(df) 

    raw_data = pd.concat(df_list, ignore_index=True).sort_values('Date')
    raw_data = raw_data.copy()
    raw_data['Season'] = raw_data['Date'].apply(get_season_id)


    # --- 2. Advanced Feature: ELO Ratings ---
    raw_data = calculate_elo_ratings(raw_data) # add ELO ratings to raw data 

    # --- 3. Team-Centric Features ---
    # Extract detailed stats
    home_matches = raw_data[['Date', 'Season', 'HomeTeam', 'FTHG', 'FTAG', 'FTR', 'HST', 'HC', 'HF']].copy()
    home_matches.columns = ['Date', 'Season', 'Team', 'GoalsFor', 'GoalsAgainst', 'Result', 'ShotsOnTarget', 'Corners', 'Fouls']
    home_matches['Points'] = home_matches['Result'].map({'H': 3, 'D': 1, 'A': 0}) # map results to points
    home_matches['IsHome'] = 1  

    away_matches = raw_data[['Date', 'Season', 'AwayTeam', 'FTAG', 'FTHG', 'FTR', 'AST', 'AC', 'AF']].copy()
    away_matches.columns = ['Date', 'Season', 'Team', 'GoalsFor', 'GoalsAgainst', 'Result', 'ShotsOnTarget', 'Corners', 'Fouls']
    away_matches['Points'] = away_matches['Result'].map({'A': 3, 'D': 1, 'H': 0})
    away_matches['IsHome'] = 0 

    team_stats = pd.concat([home_matches, away_matches]).sort_values(['Team', 'Date']) 


    # --- 4. Advanced Feature: Rest Days ---
    team_stats['LastMatchDate'] = team_stats.groupby('Team')['Date'].shift(1)
    team_stats['RestDays'] = (team_stats['Date'] - team_stats['LastMatchDate']).dt.days
    team_stats['RestDays'] = team_stats['RestDays'].fillna(7).clip(upper=30)


    # --- 5. Rolling Stats (Form) ---
    grouper = team_stats.groupby(['Team', 'Season']) 
    cols_to_roll = ['Points', 'GoalsFor', 'GoalsAgainst', 'ShotsOnTarget', 'Corners', 'Fouls']
    
    rolling = grouper[cols_to_roll].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )

    for col in cols_to_roll:
        team_stats[f'Avg{col}Last{window}'] = rolling[col] 

    cumulative = grouper[['Points', 'GoalsFor']].transform(lambda x: x.shift(1).expanding().mean()) 
    team_stats['AvgPointsSeason'] = cumulative['Points']
    team_stats['AvgGoalsForSeason'] = cumulative['GoalsFor'] 


    # --- 6. Merge Features back to Match View ---
    features_to_merge = [
        f'AvgPointsLast{window}', f'AvgGoalsForLast{window}', f'AvgGoalsAgainstLast{window}',
        f'AvgShotsOnTargetLast{window}', f'AvgCornersLast{window}', f'AvgFoulsLast{window}',
        'AvgPointsSeason', 'AvgGoalsForSeason',
        'RestDays'
    ] 
    
    cols_to_merge = ['Date', 'Team'] + features_to_merge 
    
    match_data = raw_data.copy() 
    
    match_data = match_data.merge(
        team_stats[cols_to_merge], 
        left_on=['Date', 'HomeTeam'], 
        right_on=['Date', 'Team'], 
        suffixes=('', '_Home')
    ).drop(columns=['Team'])
    
    match_data = match_data.merge(
        team_stats[cols_to_merge], 
        left_on=['Date', 'AwayTeam'], 
        right_on=['Date', 'Team'], 
        suffixes=('_Home', '_Away')
    ).drop(columns=['Team'])


    # --- 7. Advanced Feature: Differentials ---
    diff_features = []
    for feature in features_to_merge:
        if 'RestDays' not in feature:
            diff_col = f'Diff_{feature}'
            match_data[diff_col] = match_data[f'{feature}_Home'] - match_data[f'{feature}_Away']
            diff_features.append(diff_col) 

    match_data['Diff_Elo'] = match_data['HomeElo'] - match_data['AwayElo']
    diff_features.append('Diff_Elo') 


    # --- 8. Final Feature Selection ---
    final_features = diff_features + ['RestDays_Home', 'RestDays_Away', 'HomeElo', 'AwayElo'] 
    
    match_data = match_data.dropna(subset=final_features + ['B365H', 'B365D', 'B365A']) 

    X = match_data[final_features] 
    y = match_data['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    odds = match_data[['B365H', 'B365D', 'B365A']] 

    # --- 9. Split ---
    split_date = '2024-08-01' 
    dates = match_data['Date']

    mask_train = dates < split_date
    mask_test = dates >= split_date

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]
    odds_test = odds[mask_test]

    meta = {
        "raw_dir": os.path.abspath(data_dir),
        "files": all_files,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
    }

    return X_train, X_test, y_train, y_test, odds_test, meta