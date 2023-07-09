import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from preprocess_kbo_data import read_merge_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



train_df,df_numeric = read_merge_data("kbopitchingdata.csv","kbobattingdata.csv")

def Pythagorean_expectation_modeling(df):
    # Create a new column for Pythagorean expectation
    df['pythag_expect'] = (df['avg_runs_scored'] ** 2) / (df['avg_runs_scored'] ** 2 + df['avg_runs_allowed'] ** 2)

    # Split the DataFrame into input features and target variable
    X = df[['avg_runs_scored', 'avg_runs_allowed']]
    y = df['win_loss_percentage']

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

    # Print the model's coefficients
    # Calculate metrics for Pythagorean Expectation model
    mse_pythag = mean_squared_error(y_test, y_pred)

    r2_pythag = r2_score(y_test, y_pred)
    print(f'Pythagorean Expectation: MSE = {mse_pythag}, R^2 = {r2_pythag}')

if __name__ == "__main__":
    train_df = pd.read_csv('./dataset/kbo_train.csv')
    Pythagorean_expectation_modeling(train_df)