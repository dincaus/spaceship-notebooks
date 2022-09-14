import numpy as np
import pandas as pd

from typing import List, Text, Union
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest


def detect_outliers_isolation(
        df: pd.DataFrame,
        contamination=0.15,
        categorical_features: Union[None, List[Text]] = None
):
    assert df is not None, "Dataframe has to be provided"
    assert contamination, "Contamination has to be provided"

    train_data = df.copy()
    iso_model = IsolationForest(contamination=contamination)

    if categorical_features:
        for cat_feat in categorical_features:
            train_data[cat_feat] = LabelEncoder().fit_transform(train_data[cat_feat].to_numpy().reshape(-1, 1))

    return iso_model.fit_predict(train_data.to_numpy())


def detect_outliers_elliptic(
        df: pd.DataFrame,
        contamination=0.1,
        categorical_features: Union[None, List[Text]] = None
):
    assert df is not None, "Dataframe has to be provided"

    train_data = df.copy()
    elliptic_model = EllipticEnvelope(contamination=contamination)

    if categorical_features:
        for cat_feat in categorical_features:
            train_data[cat_feat] = LabelEncoder().fit_transform(train_data[cat_feat].to_numpy().reshape(-1, 1))

    return elliptic_model.fit_predict(train_data.to_numpy())
