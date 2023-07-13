import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def Pythagorean_expectation_modeling(df, runsScored, runsAllowed, win_loss_probability):
    """
    Pass in average runs scored, runs allowed, win loss probability in a year using feature name from your own dataset.
    """
    # Create a new column for Pythagorean expectation
    df['pythag_expect'] = (df[runsScored] ** 2) / (df[runsScored] ** 2 + df[runsAllowed] ** 2)

    # Split the DataFrame into input features and target variable
    X = df[[runsScored, runsAllowed]]
    y = df[win_loss_probability]

    # Generate polynomial and interaction features
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict win_percentage for the test set
    y_pred = model.predict(X_test)

    # Calculate metrics for Pythagorean Expectation model
    mse_pythag = mean_squared_error(y_test, y_pred)
    r2_pythag = r2_score(y_test, y_pred)
    print(f'Pythagorean Expectation: MSE = {mse_pythag}, R^2 = {r2_pythag}')

if __name__ == "__main__":
    # Load your datasets
    win_loss_prob = pd.read_csv('./dataset/mlb_win_loss_probabilities.csv')
    teams = pd.read_csv('./dataset/mlb_teams.csv')

    # Create mapping from teamId to teamName
    team_mapping = teams.set_index('id')['name'].to_dict()

    # Loop through each year
    for year in range(2018, 2022):  # 2022 is used as the end point as the range function does not include the end point
        team_boxscore = pd.read_csv(f'./dataset/MLB_yearly_boxscore/year_{year}.csv')

        # Ensure that 'year' column matches current year
        team_boxscore['year'] = int(year)  # the int() function converts the year to an integer

        # Prepare the team boxscore data
        team_boxscore_grouped = team_boxscore.groupby(['teamId', 'year']).agg({
            'runsScored': 'sum',
            'runsPitching': 'sum',  # 'runsPitching' will serve as 'runsAllowed'
        }).reset_index()

        # Check if team_boxscore_grouped DataFrame is empty
        if team_boxscore_grouped.empty:
            print(f'team_boxscore_grouped DataFrame is empty for year {year}.')
            continue

        # Map teamId to teamName
        team_boxscore_grouped['teamName'] = team_boxscore_grouped['teamId'].map(team_mapping)

        # Merge with the win-loss probabilities dataset
        dataset = pd.merge(team_boxscore_grouped, win_loss_prob, left_on=['teamName', 'year'], right_on=['homeName', 'season'])

        # Check if the merged DataFrame is empty
        if dataset.empty:
            print(f'Merged DataFrame is empty for year {year}. Check your merge operation.')
            continue

        dataset.to_csv(f"pythagorean_output_{year}.csv")

        # Run the Pythagorean expectation modeling
        Pythagorean_expectation_modeling(dataset, 'runsScored', 'runsPitching', 'winProb')
