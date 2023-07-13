from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import seaborn as sns
matplotlib_axes_logger.setLevel('ERROR')
warnings.filterwarnings('ignore')
from sklearn.linear_model import LassoCV, RidgeCV

def perform_ridge_with_lasso_selection(df, target_col, test_size, random_state):

    # Select only numeric columns, excluding the target column
    X = df.drop(target_col, axis=1)
    X = X.select_dtypes(include=np.number)
    y = df[target_col]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define a range of alpha values
    alphas = np.logspace(-4, 4, 50)

    # Perform Lasso regression with cross-validation
    lasso = LassoCV(alphas=alphas, cv=5)
    lasso.fit(X_train_scaled, y_train)

    # Get the feature coefficients from the Lasso model
    coef = pd.Series(lasso.coef_, index = X.columns)

    # Print the sorted coefficients
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

    # Select the relevant features
    important_features = coef[coef != 0].index.tolist()

    # Create a new X_train and X_test with only the important features
    X_train_selected = X_train[important_features]
    X_test_selected = X_test[important_features]

    # Standardize the selected features
    X_train_selected_scaled = scaler.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler.transform(X_test_selected)

    # Perform Ridge regression with the selected features and cross-validation
    ridge = RidgeCV(alphas=alphas, cv=10)
    ridge.fit(X_train_selected_scaled, y_train)

    # Print the Ridge regression coefficients
    ridge_coef = pd.Series(ridge.coef_, index = X_train_selected.columns)
    print(ridge_coef)

    # Plotting
    coef_sorted = coef.sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=coef_sorted.values, y=coef_sorted.index)
    plt.title("Feature importance based on Lasso Coefficients")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature Names")
    plt.show()

    # Print the best alpha values
    print("Best alpha for Lasso: ", lasso.alpha_)
    print("Best alpha for Ridge: ", ridge.alpha_)

    return ridge, important_features, ridge_coef



if __name__ == "__main__":
    train_df = pd.read_csv('./dataset/kbo_train.csv')
    ridge_model, selected_features, coefficients = perform_ridge_with_lasso_selection(
        train_df,
        'win_loss_percentage', 
        test_size=0.3, 
        random_state=42
)


