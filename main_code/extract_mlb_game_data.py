import pandas as pd
import json
import numpy as np

# Load the CSV file
df = pd.read_csv('train.csv')

# Initialize an empty dataframe to store the game data
mlb_df = pd.DataFrame()

# Loop over each row in the original dataframe
for idx, row in df.iterrows():
    # Check if the data is missing or malformed
    if pd.isna(row['games']) or not isinstance(row['games'], str):
        continue

    # Parse the JSON data
    try:
        game_data = json.loads(row['games'])
    except json.JSONDecodeError:
        print(f'Error decoding JSON at index {idx}')
        continue

    # Convert the parsed JSON data into a dataframe
    game_df = pd.json_normalize(game_data)
    
    # Add it to the MLB dataframe
    mlb_df = pd.concat([mlb_df, game_df], ignore_index=True)

# Save the new dataframe to a CSV file
mlb_df.to_csv('mlb_win_losses.csv', index=False)
