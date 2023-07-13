import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# TODO: Fix the code to work with the kbo dataset

def predict_win_probability(df, features):
    # Define dependent variable
    if 'total_runs_scored' in df.columns:
        print("Using 'total_runs_scored' as the dependent variable")
        y = df['total_runs_scored']
    else:
        print("Using 'RBI' as the dependent variable")
        y = df['RBI']
        
    # Define independent variables
    print("Defining independent variables")
    X = df[features]
    
    # Add a constant to the independent variables matrix
    print("Adding constant to the independent variables")
    X = sm.add_constant(X)
    
    # Fit the model
    print("Fitting the model, this may take a while...")
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    
    # Predict the win probabilities
    print("Predicting win probabilities")
    df['predicted_win_probability'] = poisson_model.predict(X)
    
    # Apply the sigmoid transformation to get probabilities between 0 and 1
    print("Applying sigmoid transformation")
    df['predicted_win_probability'] = 1 / (1 + np.exp(-df['predicted_win_probability']))
    
    # Print the model summary
    print("Model summary:")
    print(poisson_model.summary())
    
    return df


def plot_predicted_win_probability(df, league_name):
    plt.hist(df['predicted_win_probability'], bins=20, edgecolor='black')
    plt.title(f"Distribution of Predicted Win Probabilities - {league_name}")
    plt.xlabel('Predicted Win Probability')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    # Load the data
    df_kbo = pd.read_csv('../datasets/KBO_datasets/kbo_train.csv')
    df_mlb = pd.read_csv('../datasets/MLB_datasets/MLB_team_season_statistics.csv')

    # Define features for each dataset
    kbo_features = ['average_age', 'avg_runs_allowed', 'ERA', 'total_games', 'walks', 'strikeouts_x', 'batting_average', 'OBP', 'SLG', 'OPS']
    mlb_features = ['wins', 'losses', 'home', 'avg_runs_scored', 'doubles', 'triples', 'homeruns', 'strikeouts_x', 'bases_on_balls', 'hits_x', 'at_bats']

    # Predict win probabilities and print the result
    df_kbo = predict_win_probability(df_kbo, kbo_features)
    df_mlb = predict_win_probability(df_mlb, mlb_features)
    
    plot_predicted_win_probability(df_kbo, 'KBO')
    plot_predicted_win_probability(df_mlb, 'MLB')
