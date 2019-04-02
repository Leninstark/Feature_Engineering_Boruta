# Implementation of boruta for feature selection
# Boruta makes explicit use of a random forest to eliminate features
# Boruta follows an all-relevant feature selection method where it captures all features
# which are in some circumstances relevant to the outcome variable

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def boruta(dsname, target):
    na = ["", " ", "-", "?", "N/A"]
    data = pd.read_csv(dsname, na_values=na)
    data[target] = data[target].astype(object)
    columns = data.columns
    target_type = data[target].dtype
    print(target_type)

    class DataFrameImputer(TransformerMixin):
        def __init__(self):
            """Impute missing values.

            Columns of dtype object are imputed with the most frequent value
            in column.

            Columns of other types are imputed with mean of column.

            """

        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0]
                                   if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                                  index=X.columns)

            return self

        def transform(self, X, y=None):
            return X.fillna(self.fill)

    cat_data = data.select_dtypes(include=["object"])
    num_data = data.select_dtypes(include=["int64", "float64"])

    num_data = num_data.fillna(num_data.mean())
    cat_data = DataFrameImputer().fit_transform(cat_data)

    # Label encoding the values - Converting into Numerical
    """ Some of the algorithms has hard constraint that it cannot process categorical data . For such algorithms,
        User has to convert categorical features into Numeric. We have few techniques for this operation and 
        we are implementing the famous and good-result methods here as  
        Label - Encoding 
    """
    cat_data = cat_data.apply(LabelEncoder().fit_transform)
    data = pd.concat([num_data, cat_data], axis=1)
    data = data[columns]

    y = data[target]
    x = data.drop([target], axis=1)
    x = x.as_matrix()
    y = y.as_matrix()
    if target_type == 'object':

        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', max_depth=20)
        feature_selection = BorutaPy(rf, n_estimators='auto', verbose=2)
        feature_selection.fit(x, y)

        # number of selected features
        print('\n Number of selected features:')
        print(feature_selection.n_features_)

        # check ranking of features
        print('\n Feature ranking:')
        print(feature_selection.ranking_)

        print('\n Initial features: ', data.columns.tolist())

        data = data.drop([target], axis=1)
        feature_df = pd.DataFrame(data.columns.tolist())
        feature_df['rank'] = feature_selection.ranking_
        feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)
        print('\n Top %d features:' % feature_selection.n_features_)
        feature_df.columns = ['feature_name', 'Rank']
        print(feature_df.head(feature_selection.n_features_))

    elif (target_type == 'int') | (target_type == 'float'):
        rf = RandomForestRegressor(n_estimators=500, max_depth=20)
        feature_selection = BorutaPy(rf, n_estimators='auto', verbose=2)
        print(feature_selection.fit(x, y))

        # number of selected features
        print('\n Number of selected features:')
        print(feature_selection.n_features_)

        # check ranking of features
        print('\n Feature ranking:')
        print(feature_selection.ranking_)

        print('\n Initial features: ', data.columns.tolist())

        data = data.drop([target], axis=1)
        feature_df = pd.DataFrame(data.columns.tolist())
        feature_df['rank'] = feature_selection.ranking_
        feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)
        print(feature_df)
        print('\n Top %d features:' % feature_selection.n_features_)
        feature_df.columns = ['feature_name', 'Rank']
        print(feature_df.head(feature_selection.n_features_))
