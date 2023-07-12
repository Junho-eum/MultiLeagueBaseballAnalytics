from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt

class LDA:
    def __init__(self):
        pass

    def fit_lda(self, X_train, y_train, n_components=2):
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
        feature_coef_df = pd.DataFrame(coef, columns=X.columns)
        return feature_coef_df
