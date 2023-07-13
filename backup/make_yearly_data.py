import pandas as pd

# Load the data
data = pd.read_csv('mlb_data.csv')

# Convert the gameDate column to datetime
data['gameDate'] = pd.to_datetime(data['gameDate'])

# Extract the year from each date
data['year'] = data['gameDate'].dt.year

# Loop over each unique year
for year in data['year'].unique():
    # Select only the rows for that year
    yearly_data = data[data['year'] == year]

    # Compute yearly stats for each team
    yearly_team_stats = yearly_data.groupby('teamId').sum()

    # Save that to a CSV
    yearly_team_stats.to_csv(f'year_{year}.csv')
