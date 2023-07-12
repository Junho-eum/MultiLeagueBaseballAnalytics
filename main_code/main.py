from preprocess_data import PreprocessData
from LDA import LDA
import pandas as pd

# Make more strategy for missing values
impute_strategy_dict = {
    'average_age': 'mean',
    'avg_runs_allowed': 'median',
}
def main():
    # Instantiate PreprocessData and AnalyzeData classes
    preprocessor = PreprocessData(impute_strategy_dict)
    analyzer = LDA_explore()

    df = pd.read_csv("./dataset/kbo_train.csv")
    df = preprocessor.handle_missing_data(df)

    # Update this line to create the categorical target
    X, y = preprocessor.separate_features_target(df, 'win_loss_percentage')

    X_scaled, scaler = preprocessor.scale_features(X)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)

    lda, X_lda = analyzer.fit_lda(X_train, y_train)
    analyzer.plot_lda(X_lda, y_train)

    print('Explained variance ratio: ', lda.explained_variance_ratio_)
    feature_coef_df = analyzer.lda_coef(lda, X)
    print(feature_coef_df)
    # Save feature_coef_df to csv
    feature_coef_df.to_csv('./lda_feature_coefficients.csv', index=False)
if __name__ == "__main__":
    main()
    
