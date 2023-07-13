import pandas as pd
import numpy as np
import statsmodels.api as sm

def predict_win_probability(df, features, dep_variable):
    # Define the dependent variable and independent variables
    y = df[dep_variable]
    X = df[features]

    # Add constant to the independent variables matrix
    X = sm.add_constant(X)

    # Fit Poisson regression model
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    # Print the model summary
    print(model.summary())

    # Predict total runs scored
    df['predicted_runs_scored'] = model.predict(X)

    # Convert runs scored to win probability (this is a simplification, real conversion may be more complex)
    df['predicted_win_probability'] = df['predicted_runs_scored'] / df[dep_variable]

    # Return DataFrame with predictions
    return df

if __name__ == "__main__":
    # Load the data
    df_kbo = pd.read_csv('../datasets/KBO_datasets/kbo_train.csv')
    df_mlb = pd.read_csv('../datasets/MLB_datasets/MLB_team_season_statistics.csv')

    # Define features for each dataset
    kbo_features = ['average_age', 'avg_runs_allowed', 'ERA', 'total_games', 'walks', 'strikeouts_x', 'batting_average', 'OBP', 'SLG', 'OPS']
    mlb_features = ['wins', 'losses', 'home', 'avg_runs_scored', 'doubles', 'triples', 'homeruns', 'strikeouts_x', 'bases_on_balls', 'hits_x', 'at_bats']

    # Predict win probabilities and print the result
    df_kbo = predict_win_probability(df_kbo, kbo_features, 'total_runs_scored')
    df_mlb = predict_win_probability(df_mlb, mlb_features, 'RBI')
    print(df_kbo)
    print(df_mlb)
