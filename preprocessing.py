import pandas as pd
import numpy as np

from typing import List, Text
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


CATEGORICAL_COLUMNS = ["HomePlanet", "Destination", "CabinNumber", "Deck", "Side"]
BOOL_COLUMNS = ["CryoSleep", "VIP", "Transported"]
NUMERICAL_COLUMNS = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

FEATURES = CATEGORICAL_COLUMNS + BOOL_COLUMNS + NUMERICAL_COLUMNS


def fill_missing_categorical_columns(df: pd.DataFrame, col_name):
    mode_result = df[col_name].mode()

    return df[col_name].fillna(mode_result[0])


def fill_missing_w_mean(df: pd.DataFrame, col_name):
    mode_result = df[col_name].mode()

    return df[col_name].fillna(mode_result[0])


def expand_cabin(df: pd.DataFrame):
    result_series = df["Cabin"].str.split("/", 3, expand=True)
    result_series[1] = result_series[1].fillna(result_series[1].mode()[0]).astype(np.int32)

    return result_series


def expand_name(df: pd.DataFrame):
    result_series = df["Name"].str.lower().str.split(" ", 2, expand=True)

    return result_series


def expand_passenger_id_to_group(df: pd.DataFrame):
    return df["PassengerId"].str.split('_', expand=True)[1].astype(np.int64)


def fill_age_column(df: pd.DataFrame):
    median_ages_per_passenger_group = {}
    passengers_group = df["PassengerGroup"].unique()

    for pg in passengers_group:
        median_ages_per_passenger_group[pg] = df.loc[df["PassengerGroup"] == pg, ["Age"]].median()[0]

    for index, passenger in df.iterrows():
        if pd.isna(passenger["Age"]):
            df.at[index, "Age"] = median_ages_per_passenger_group[passenger["PassengerGroup"]]


def process_dataset(dataset: pd.DataFrame):
    dataset[["Deck", "CabinNumber", "Side"]] = expand_cabin(dataset)
    dataset["PassengerGroup"] = expand_passenger_id_to_group(dataset)

    fill_age_column(dataset)

    dataset["HomePlanet"] = fill_missing_categorical_columns(dataset, "HomePlanet")
    dataset["CryoSleep"] = fill_missing_categorical_columns(dataset, "CryoSleep")
    dataset["Destination"] = fill_missing_categorical_columns(dataset, "Destination")
    dataset["VIP"] = fill_missing_categorical_columns(dataset, "VIP")
    dataset["Deck"] = fill_missing_categorical_columns(dataset, "Deck")
    dataset["Side"] = fill_missing_categorical_columns(dataset, "Side")

    dataset["CabinNumber"] = fill_missing_categorical_columns(dataset, "CabinNumber")
    dataset["RoomService"] = fill_missing_w_mean(dataset, "RoomService")
    dataset["FoodCourt"] = fill_missing_w_mean(dataset, "FoodCourt")
    dataset["ShoppingMall"] = fill_missing_w_mean(dataset, "ShoppingMall")
    dataset["Spa"] = fill_missing_w_mean(dataset, "Spa")
    dataset["VRDeck"] = fill_missing_categorical_columns(dataset, "VRDeck")

    return dataset[FEATURES]


def get_where_all_are_non_null(df: pd.DataFrame):
    return df.dropna()


def convert_to_categorical(df: pd.DataFrame):
    categorical_mapper = {}

    for cat_col in CATEGORICAL_COLUMNS:

        if cat_col in df.columns:
            categorical_mapper[cat_col] = LabelEncoder()
            df[cat_col] = categorical_mapper[cat_col].fit_transform(df[cat_col])

    for cat_col in BOOL_COLUMNS:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(np.int32)

    return categorical_mapper


def impute_missing_values(df: pd.DataFrame, columns: List[Text], **kwargs):
    n_neighbors = kwargs.get("n_neighbors", 2)

    knn_impute = KNNImputer(n_neighbors=n_neighbors)
    return knn_impute.fit_transform(df[columns].to_numpy())
