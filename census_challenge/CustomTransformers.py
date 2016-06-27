from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

class ColumnExtractor(TransformerMixin,BaseEstimator):
    "Helps extract columns from Pandas Dataframe"

    def __init__(self, columns, c_type=None):
        self.columns = columns
        self.c_type = c_type

    def transform(self, X, **transform_params):
        cs = X[self.columns]
        return(cs)

    def fit(self, X, y=None, **fit_params):
        return(self)

class FillNaTransformer(TransformerMixin,BaseEstimator):

    def transform(self, X, **transform_params):
        clean = X.replace(np.nan,"?")
        return(clean)

    def fit(self, X, y=None, **fit_params):
        return(self)

class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return(self) # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return(output)

    def fit_transform(self, X, y=None):
        return(self.fit(X, y).transform(X))
