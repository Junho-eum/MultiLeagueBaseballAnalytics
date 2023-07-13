import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.stats import norm
import warnings
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from dataclasses import dataclass
matplotlib_axes_logger.setLevel('ERROR')
warnings.filterwarnings('ignore')
import seaborn as sns


def read_merge_data(df1,df2):
    pitching_data = pd.read_csv("kbopitchingdata.csv")
    batting_data = pd.read_csv("kbobattingdata.csv")

    # Rename the 'old_column_name' to 'new_column_name'
    pitching_data.rename(columns={'runs_per_game': 'avg_runs_allowed'}, inplace=True)
    batting_data.rename(columns={'runs_per_game': 'avg_runs_scored'}, inplace=True)

    # Merge the DataFrames on the specified columns
    train_df = pd.merge(pitching_data, batting_data, on=["team", "year"])

    # Remove duplicate rows
    train_df = train_df.drop_duplicates()

    # Remove columns with any missing values
    train_df = train_df.dropna(axis=1, how="any")

    # Change column names
    train_df.rename(columns={'runs_per_game': 'avg_runs_allowed'}, inplace = True)

    numeric_columns = train_df.select_dtypes(include=[float, int]).columns
    df_numeric = train_df[numeric_columns]

    return train_df, df_numeric

def column_mapping(df):
    # Define the mapping from old column names to new ones
    column_mapping = {
        'games_x': 'total_games',
        'runs_x': 'total_runs_allowed',
        'runs_y': 'total_runs_scored'
    }
    # Drop duplicate column 
    df = df.drop('run_average_9', axis=1)
    # Rename the columns
    df.rename(columns=column_mapping, inplace=True)
    return df



def create_corr_plot(df):
    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Extract the correlations of the target column with other columns
    column_correlations = correlation_matrix['win_loss_percentage'].drop('win_loss_percentage')

    # Get the top 15 variables with the highest absolute correlation with the target variable
    
    top_15_variables = column_correlations.abs().sort_values(ascending=False).head(15)
    
    top_15_variable_names = top_15_variables.index.tolist()

    # Plot the bar chart
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_15_variables.values, y=top_15_variables.index, palette="viridis")
    plt.title('Correlation: Top 15 Variables vs Win Loss Percentage')
    plt.xlabel('Correlation')
    plt.ylabel('Variable')
    plt.show()
    
    return top_15_variable_names

if  __name__ == "__main__":
    train_df, df_numeric = read_merge_data("kbopitchingdata.csv","kbobattingdata.csv")
    train_df = column_mapping(train_df)
    train_df.to_csv("kbo_train.csv")
    
    top_15_var = create_corr_plot(train_df)
    print(top_15_var)



