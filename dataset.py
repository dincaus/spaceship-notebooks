import pandas as pd


# CATEGORICAL_COLUMNS = ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinNumber", "Deck", "Side"]
# NUMERICAL_COLUMNS = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
#
# FEATURES = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
# LABELS = ["Transported", ]


def extract_test_from_dataset(df: pd.DataFrame):

    test_df = df.loc[df["Transported"].isnull()]
    train_df = df.loc[~df["Transported"].isnull()]

    return train_df, test_df
