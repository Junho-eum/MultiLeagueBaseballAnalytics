import pandas as pd

def preprocess_league_data(df1_path, df2_path, team_info_path):
    # load data
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    df_team = pd.read_csv(team_info_path)  # load team info

    # Calculate wins and losses for home and away games
    df1_home = df1.groupby(['season', 'homeId']).agg({
        'homeWins': 'sum',
        'homeLosses': 'sum'
    }).reset_index().rename(columns={'homeId': 'teamId', 'homeWins': 'wins', 'homeLosses': 'losses'})

    df1_away = df1.groupby(['season', 'awayId']).agg({
        'awayWins': 'sum',
        'awayLosses': 'sum'
    }).reset_index().rename(columns={'awayId': 'teamId', 'awayWins': 'wins', 'awayLosses': 'losses'})

    # Concatenate home and away statistics
    team_season_data = pd.concat([df1_home, df1_away])

    # Aggregate wins and losses
    team_season_data = team_season_data.groupby(['season', 'teamId']).agg({'wins': 'sum', 'losses': 'sum'}).reset_index()

    # Calculate win loss percentage
    team_season_data['winLossPercentage'] = team_season_data['wins'] / (team_season_data['wins'] + team_season_data['losses'])

    # Average stats from df2 per season per team
    df2['gameDate'] = pd.to_datetime(df2['gameDate'])
    df2['season'] = df2['gameDate'].dt.year
    df2_grouped = df2.groupby(['season', 'teamId']).mean().reset_index()
    
    # Merge two dataframes
    final_df = pd.merge(team_season_data, df2_grouped, on=['season', 'teamId'])

    # Merge with team info to get team names
    final_df = pd.merge(final_df, df_team[['id', 'teamName']], left_on='teamId', right_on='id', how='left')

    # Change columns names to match required format
    column_mapping = {
        'teamId': 'team', 
        'season': 'year',
        'wins': 'wins',
        'losses': 'losses',
        'winLossPercentage': 'win_loss_percentage',
        'runsScored': 'avg_runs_scored',
        'runsPitching': 'avg_runs_allowed',
        'strikeOuts': 'strikeouts_x',
        'hits': 'hits_x',
        'plateAppearances': 'plate_appearances',
        'atBats': 'at_bats',
        'homeRuns': 'homeruns',
        'rbi': 'RBI',
        'baseOnBalls': 'bases_on_balls',
        'strikeOutsPitching': 'strikeouts_y',
        'hitsPitching': 'hits_y',
        'inningsPitched': 'innings_pitched',
        'earnedRuns': 'earned_runs',
        'teamName': 'teamName'
        # map the remaining columns...
    }
    final_df.rename(columns=column_mapping, inplace=True)

    # Remove 'id' column
    final_df = final_df.drop(columns='id')

    return final_df

if __name__ == "__main__":
    final_df = preprocess_league_data('./dataset/mlb_win_losses.csv', './dataset/mlb_data.csv', './dataset/mlb_teams.csv')
    # output the data
    final_df.to_csv('team_season_statistics.csv', index=False)
