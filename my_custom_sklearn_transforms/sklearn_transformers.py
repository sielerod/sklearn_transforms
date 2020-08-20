from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
import pandas as pd


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X[self.columns].copy()
        # Retornamos um novo dataframe com as colunas desejadas
        return data


class MinMax_Notes(BaseEstimator, TransformerMixin):

    def __init__(self, other_features, notes):
        self.other_features = other_features
        self.notes = notes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
        X_minmax = min_max_scaler.fit_transform(X[self.notes].values)
        scaled_features_df = pd.DataFrame(X_minmax, columns=self.notes)

        X = pd.concat([X[self.other_features], scaled_features_df], axis=1)

        return X