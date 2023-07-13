import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class PythagoreanExpectationModel:
    def __init__(self, runsScored, runsAllowed, win_loss_probability):
        self.runsScored = runsScored
        self.runsAllowed = runsAllowed
        self.win_loss_probability = win_loss_probability

    def calculate_pythag_expect(self, df):
        df['pythag_expect'] = (df[self.runsScored] ** 2) / (df[self.runsScored] ** 2 + df[self.runsAllowed] ** 2)
        return df

    def train_model(self, df):
        X = df[[self.runsScored, self.runsAllowed]]
        y = df[self.win_loss_probability]

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse_pythag = mean_squared_error(y_test, y_pred)
        r2_pythag = r2_score(y_test, y_pred)
        print(f'Pythagorean Expectation: MSE = {mse_pythag}, R^2 = {r2_pythag}')

def main():
    win_loss_prob = pd.read_csv('../datasets/MLB_datasets/mlb_win_loss_probabilities.csv')
    teams = pd.read_csv('../datasets/MLB_datasets/mlb_teams.csv')
    team_mapping = teams.set_index('id')['name'].to_dict()
    pythagorean_model = PythagoreanExpectationModel('runsScored', 'runsPitching', 'winProb')

    for year in range(2018, 2022):
        team_boxscore = pd.read_csv(f'../datasets/MLB_datasets/MLB_yearly_boxscore/year_{year}.csv')
        team_boxscore['year'] = int(year)
        team_boxscore_grouped = team_boxscore.groupby(['teamId', 'year']).agg({
            'runsScored': 'sum',
            'runsPitching': 'sum', 
        }).reset_index()

        if team_boxscore_grouped.empty:
            print(f'team_boxscore_grouped DataFrame is empty for year {year}.')
            continue

        team_boxscore_grouped['teamName'] = team_boxscore_grouped['teamId'].map(team_mapping)

        dataset = pd.merge(team_boxscore_grouped, win_loss_prob, left_on=['teamName', 'year'], right_on=['homeName', 'season'])

        if dataset.empty:
            print(f'Merged DataFrame is empty for year {year}. Check your merge operation.')
            continue

        dataset = pythagorean_model.calculate_pythag_expect(dataset)
        dataset.to_csv(f"../datasets/MLB_datasets/pe_model_yearly_outputs/pythagorean_output_{year}.csv")

        pythagorean_model.train_model(dataset)

if __name__ == "__main__":
    main()
