import pandas as pd
import tensorflow as tf

from typing import List, Text, Union


def extract_test_from_dataset(df: pd.DataFrame):

    test_df = df.loc[df["Transported"].isnull()]
    train_df = df.loc[~df["Transported"].isnull()]

    return train_df, test_df


def dataframe_to_dataset(
        df: pd.DataFrame,
        label_cols: Union[List[Text], None] = None,
        shuffle: bool = True
):
    assert df is not None, "Dataframe has to be provided"

    dataframe = df.copy()

    if label_cols:
        labels = dataframe[label_cols]
        dataframe = dataframe.drop(columns=label_cols)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe, )))

    feature_cols = dataframe.columns

    def _expand_dims(x_, y_):

        for fc in feature_cols:
            x_[fc] = tf.expand_dims(x_[fc], -1)

        return x_, y_

    def _expand_dims_single(x_):
        for fc in feature_cols:
            x_[fc] = tf.expand_dims(x_[fc], -1)

        return x_

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    if label_cols:
        ds = ds.map(_expand_dims)
    else:
        ds = ds.map(_expand_dims_single)

    return ds
