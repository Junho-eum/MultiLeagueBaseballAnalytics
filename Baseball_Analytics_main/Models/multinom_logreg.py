import pandas as pd
import numpy as np
from patsy import dmatrices
from sklearn.metrics import confusion_matrix
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import norm


# Create function to generate the levels
def create_level(x):
    if x < 0.45:
        return 0  # "bad"
    elif x < 0.55:
        return 1  # "normal"
    else:
        return 2  # "good"
    
def multinomial_logreg_model(df):
    # Apply the function to the 'win_loss_percentage' column
    df['response'] = df['win_loss_percentage'].apply(create_level)

    # Ensure the data type is integer
    df['response'] = df['response'].astype(int)

    # Define the formula for the model
    formula = 'response ~ 0 + avg_runs_allowed + saves + WHIP + avg_runs_scored + OBP'

    # Create a numpy array for y (the dependent variable)
    y, X = dmatrices(formula, df, return_type='dataframe')  

    # Ensure the dependent variable y is a numpy array
    y = np.asarray(y).ravel()  # ravel() to convert to 1D array

    # X to numpy array
    X_arr = np.asarray(X)

    # Build the model
    model = OrderedModel(y, X_arr, distr='logit')

    result = model.fit()

    #...rest of your code
    # Print model summary
    print(result.summary())

    # Predict classes
    pred_class = result.predict().argmax(axis=1)
    print(pred_class)

    # Predict probabilities
    # pred_prob = result.predict()
    # print(pred_prob)

    # Confusion matrix
    # cm = confusion_matrix(y.values.argmax(axis=1), pred_class)
    # print(cm)

    # Misclassification error
    # misclass_rate = 1 - np.diag(cm).sum() / cm.sum()
    # print(misclass_rate)


if __name__ == "__main__":
    train_df = pd.read_csv('../datasets/KBO_datasets/kbo_train.csv')
    multinomial_logreg_model(train_df)
