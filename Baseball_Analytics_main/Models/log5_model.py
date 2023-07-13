import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_log5(df):
    # Calculate winning percentages
    if 'total_games' not in df.columns:
        df['total_games'] = df['wins'] + df['losses']
    df['winning_percentage'] = df['wins'] / df['total_games']

    # Create a dictionary with teams as keys and winning percentages as values
    winning_percentages = df.set_index('team')['winning_percentage'].to_dict()

    # Create a DataFrame for the probabilities
    teams = df['team'].unique()
    probabilities = pd.DataFrame(index=teams, columns=teams)

    # Calculate probabilities
    for team_a in teams:
        for team_b in teams:
            if team_a != team_b:
                A = winning_percentages[team_a]
                B = winning_percentages[team_b]
                result = (A - A*B) / (A + B - 2*A*B)
                # Ensure that the result is a finite number
                if np.isfinite(result):
                    probabilities.loc[team_a, team_b] = result
                else:
                    probabilities.loc[team_a, team_b] = np.nan
            else:
                probabilities.loc[team_a, team_b] = np.nan

    # Convert the dataframe to float data type
    probabilities = probabilities.astype(float)

    return probabilities


def plot_heatmap(df, title):
    plt.figure(figsize=(10,8))
    sns.heatmap(df, cmap="YlGnBu")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    df_kbo = pd.read_csv('../datasets/KBO_datasets/kbo_train.csv')
    df_mlb = pd.read_csv('../datasets/MLB_datasets/MLB_team_season_statistics.csv')
    probabilities_kbo = calculate_log5(df_kbo)
    probabilities_mlb = calculate_log5(df_mlb)
    plot_heatmap(probabilities_kbo, 'KBO Log5 Probabilities')
    plot_heatmap(probabilities_mlb, 'MLB Log5 Probabilities')
