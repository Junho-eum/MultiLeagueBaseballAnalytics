import pandas as pd

# load data
df1 = pd.read_csv('mlb_win_losses.csv')
df2 = pd.read_csv('mlb_data.csv')

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
# output the data
final_df.to_csv('team_season_statistics.csv', index=False)
