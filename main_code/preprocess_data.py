import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Missing Data Handling: 
def handle_missing_data(df, impute_strategy_dict):
    """
    This function accepts a dataframe, df, and a dictionary, impute_strategy_dict, where:
    - keys are column names, and
    - values are impute strategies: 'mean', 'median', or 'mode'.
    The function fills missing values in each column according to the specified strategy.
    """
    for column, strategy in impute_strategy_dict.items():
        if strategy == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif strategy == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif strategy == 'mode':
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df

def separate_features_target(df, target_column):
    """
    separate numerical variables and target variable
    """
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_column])
    y = df[target_column]
    return X, y

def scale_features(X):
    """
    returning the scaler because it needs to be applied the same scaling to the test set and any future data
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Make more strategy for missing values
impute_strategy_dict = {
    'average_age': 'mean',
    'avg_runs_allowed': 'median',
    'total_games': 'mode',
}

if __name__ == "__main__":
    train_df = pd.read_csv("kbo_train.csv")
    df = handle_missing_data(train_df, impute_strategy_dict)
    X, y = separate_features_target(train_df, 'win_loss_percentage')
    X_scaled, scaler = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)






