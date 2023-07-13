from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from preprocess_data_LDA import PreprocessModelData
import seaborn as sns

class LDA:
    def __init__(self,df, threshold=1):
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.impute_strategy_dict = {'avg_runs_allowed': 'median'}
        self.df = df
        
    def fit_lda(self, X_train, y_train, n_components=2):
        # Standardize the feature matrix
        X_train = self.scaler.fit_transform(X_train)
        
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X_train, y_train)
        return lda, X_lda

    def plot_lda(self, X_lda, y_train):
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_train)
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.title('Linear Discriminant Analysis')
        plt.grid(True)
        plt.legend(*scatter.legend_elements(), title='Classes')
        plt.show()

    def lda_coef(self, lda, X):
        coef = lda.coef_
        # Take absolute values
        abs_coef = abs(coef)
        # Since X has been standardized, the original column names need to be applied
        feature_coef_df = pd.DataFrame(abs_coef, columns=X.columns)
        # Filter coefficients based on the threshold
        selected_features = feature_coef_df.loc[:, (feature_coef_df > self.threshold).any()]
        return selected_features
    
    def check_collinearity(self, X):
        # VIF dataframe
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns

        # calculating VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

        return vif_data
    
    def select_features_based_on_coef(self, coef_df, threshold):
        coef_df_abs = coef_df.abs()
        selected_features = coef_df_abs.columns[(coef_df_abs > threshold).any()].tolist()
        return selected_features
    
    def plot_vif_data(self, vif_data):
        vif_data = vif_data[vif_data["VIF"] != float('inf')]  # remove inf values for plotting

        plt.figure(figsize=(12, 8))
        sns.barplot(x="VIF", y="feature", data=vif_data.sort_values(by="VIF", ascending=False))
        plt.title("VIF scores")
        plt.show()

    def run_analysis(self):
        # Instantiate PreprocessData class
        preprocessor = PreprocessModelData(self.impute_strategy_dict)

        df = self.df
        df = preprocessor.handle_missing_data(df)

        # Update this line to create the categorical target
        X, y = preprocessor.separate_features_target(df, 'win_loss_percentage')

        X_scaled, scaler = preprocessor.scale_features(X)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)

        vif_data = self.check_collinearity(X)
        lda, X_lda = self.fit_lda(X_train, y_train)

        self.plot_lda(X_lda, y_train)
        self.plot_vif_data(vif_data)

        print('Explained variance ratio: ', lda.explained_variance_ratio_)
        feature_coef_df = self.lda_coef(lda, X)
        print(feature_coef_df)

        # Select features based on absolute coefficient value
        selected_features = self.select_features_based_on_coef(feature_coef_df, 1)
        print('Selected Features: ', selected_features)

        # Save feature_coef_df to csv
        feature_coef_df.to_csv('./lda_feature_coefficients.csv', index=False)

        return vif_data, lda

    
if __name__ == "__main__":
    mlb_df = pd.read_csv("../datasets/MLB_datasets/team_season_statistics.csv")
    kbo_df = pd.read_csv("../datasets/KBO_datasets/kbo_train.csv")
    lda_kbo = LDA(kbo_df)
    lda_kbo.run_analysis()
