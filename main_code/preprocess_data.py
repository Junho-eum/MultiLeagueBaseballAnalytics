import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class PreprocessData:
    def __init__(self, impute_strategy_dict):
        self.impute_strategy_dict = impute_strategy_dict

    def handle_missing_data(self, df):
        for column, strategy in self.impute_strategy_dict.items():
            if column in df.columns:
                if strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
        return df

    def separate_features_target(self, df, target_column):
        if target_column == 'win_loss_percentage':
            bins = [0, 0.43, 0.5, 1]  
            labels = ['low', 'medium', 'high']  
            df['categorical_win_loss_prob'] = pd.cut(df[target_column], bins=bins, labels=labels)
            df['categorical_win_loss_prob'] = df['categorical_win_loss_prob'].cat.codes
            target_column = 'categorical_win_loss_prob'
            
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_column])
        y = df[target_column]
        return X, y

    def scale_features(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def remove_features(self, X, feature_names):
        """
        Removes specified features from the dataframe.

        Parameters:
            X (pd.DataFrame): The input DataFrame.
            feature_names (list): The list of feature names to remove.

        Returns:
            X_new (pd.DataFrame): The DataFrame with specified features removed.
        """
        X_new = X.drop(feature_names, axis=1)
        return X_new
    
