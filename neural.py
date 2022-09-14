import math

import numpy as np
import pandas as pd
import tensorflow as tf

from tabnet import create_tabnet_classifier

from typing import List, Union, Text
from dataset import dataframe_to_dataset

from hyperopt import fmin, tpe, space_eval
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold


layers = tf.keras.layers
optimizers = tf.keras.optimizers
losses = tf.keras.losses


def create_model_inputs(
        numerical_features: List[Text],
        categorical_features: List[Text],
        categorical_int_features: Union[None, List[Text]] = None
):

    if categorical_int_features is None:
        categorical_int_features = list()

    inputs = {}

    for feature in numerical_features + categorical_features + categorical_int_features:
        if feature in numerical_features:
            inputs[feature] = layers.Input(name=feature, shape=(), dtype=tf.float32)
        elif feature in categorical_features:
            inputs[feature] = layers.Input(name=feature, shape=(), dtype=tf.string)
        elif feature in categorical_int_features:
            inputs[feature] = layers.Input(name=feature, shape=(), dtype=tf.int64)

    return inputs


def encode_inputs_separate(
        inputs,
        numerical_features: List[Text],
        categorical_features: List[Text],
        categorical_vocabulary: dict,
        encoding_size: int
):
    encoded_categorical_feature_list, numerical_feature_list= [], []

    for feature_name in inputs:
        if feature_name in categorical_features:
            vocabulary = categorical_vocabulary[feature_name]

            lookup = layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int"
            )
            encoded_feature = lookup(inputs[feature_name])

            embedding = layers.Embedding(input_dim=len(vocabulary), output_dim=encoding_size)
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_categorical_feature_list.append(encoded_categorical_feature)
        else:
            numerical_feature = tf.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list


def encode_inputs(
        inputs,
        numerical_features: List[Text],
        categorical_features: List[Text],
        categorical_vocabulary: dict,
        use_embedding=False,
        encoding_size=None,
        return_as_list: bool = False,
        categorical_features_int: Union[None, List[Text]] = None
):
    encoded_features = []

    if categorical_features_int is None:
        categorical_features_int = list()

    for feature_name in inputs:

        if feature_name in (categorical_features + categorical_features_int):
            vocabulary = categorical_vocabulary[feature_name]

            if feature_name not in categorical_features_int:
                lookup = layers.StringLookup(
                    vocabulary=vocabulary,
                    mask_token=None,
                    num_oov_indices=0,
                    output_mode="int" if use_embedding else "binary"
                )
            else:
                lookup = layers.IntegerLookup(
                    vocabulary=vocabulary,
                    mask_token=None,
                    num_oov_indices=0,
                    output_mode="int" if use_embedding else "binary"
                )

            if use_embedding:
                encoded_feature = lookup(inputs[feature_name])
                embedding_dims = int(math.sqrt(len(vocabulary))) if encoding_size is None else encoding_size
                embedding = layers.Embedding(input_dim=len(vocabulary), output_dim=embedding_dims)
                encoded_feature = embedding(encoded_feature)
            else:
                encoded_feature = lookup(tf.expand_dims(inputs[feature_name], -1))
        elif feature_name in numerical_features:
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        else:
            print(f"Unknown feature name provided {feature_name}")
            continue

        encoded_features.append(encoded_feature)

    if not return_as_list:
        all_features = layers.concatenate(encoded_features)
    else:
        all_features = encoded_features

    return all_features


def create_classification_model(
        inputs,
        features,
        hidden_layers: List[int],
        learning_rate: float = 1e-2,
        dropout_rate: Union[None, float] = None,
        smooth_label: Union[None, float] = 0
):
    x = None

    for hidden_idx, hidden_size in enumerate(hidden_layers):
        if not hidden_idx:
            x = layers.Dense(hidden_size, activation="relu")(features)
        else:
            x = layers.Dense(hidden_size, activation="relu")(x)

    if dropout_rate is None:
        output = layers.Dense(1, activation="sigmoid")(x)
    else:
        dropout_layer = layers.Dropout(dropout_rate)(x)
        output = layers.Dense(1, activation="sigmoid")(dropout_layer)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    if smooth_label is None:
        model.compile(
            optimizer=optimizers.RMSprop(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
    else:
        model.compile(
            optimizer=optimizers.RMSprop(learning_rate=learning_rate),
            loss=losses.BinaryCrossentropy(label_smoothing=smooth_label),
            metrics=["binary_accuracy"]
        )

    return model


def create_wide_and_deep_model(
    numerical_features,
    categorical_features,
    categorical_features_int,
    categorical_features_with_vocabulary,
    hidden_units: List[int],
    encoding_dims: int,
    learning_rate: float,
    dropout_rate: float
):
    inputs = create_model_inputs(
        numerical_features,
        categorical_features,
        categorical_int_features=categorical_features_int
    )
    wide = encode_inputs(inputs, numerical_features, categorical_features, categorical_features_with_vocabulary)
    wide = layers.BatchNormalization()(wide)

    deep = encode_inputs(
        inputs,
        numerical_features,
        categorical_features,
        categorical_features_with_vocabulary,
        use_embedding=True,
        encoding_size=encoding_dims,
        categorical_features_int=categorical_features_int
    )

    for units in hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([wide, deep])
    outputs = layers.Dense(1, activation="sigmoid")(merged)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    return model


def create_deep_and_cross_model(
        numerical_features,
        categorical_features,
        categorical_features_int,
        categorical_features_with_vocabulary,
        hidden_units: List[int],
        encoding_dims: int,
        learning_rate: float,
        dropout_rate: float
):
    inputs = create_model_inputs(
        numerical_features,
        categorical_features,
        categorical_int_features=categorical_features_int
    )

    x0 = encode_inputs(
        inputs,
        numerical_features,
        categorical_features,
        categorical_features_with_vocabulary,
        use_embedding=True,
        encoding_size=encoding_dims,
        categorical_features_int=categorical_features_int
    )

    cross = x0
    for _ in hidden_units:
        units = cross.shape[-1]
        x = layers.Dense(units)(cross)
        cross = x0 * x + cross

    cross = layers.BatchNormalization()(cross)

    deep = x0
    for units in hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([cross, deep])
    outputs = layers.Dense(1, activation="sigmoid")(merged)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    return model


class GatedLinearUnit(layers.Layer):

    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()

        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)


class GatedResidualNetwork(layers.Layer):

    def __init__(self, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()

        self.units = units
        self.elu_dense = layers.Dense(units, activation="elu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)

        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)

        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)

        return x


class VariableSelection(layers.Layer):

    def __init__(self, num_features, units, dropout_rate):
        super(VariableSelection, self).__init__()

        self.grns = list()

        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)

        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(num_features, activation="softmax")

    def call(self, inputs):
        v = layers.concatenate(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, inp in enumerate(inputs):
            x.append(self.grns[idx](inp))

        x = tf.stack(x, axis=1)

        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)

        return outputs


def create_grn_and_vsn_model(
        numerical_features,
        categorical_features,
        categorical_features_int,
        categorical_features_with_vocabulary,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.15,
        encoding_size: int = 16
):
    inputs = create_model_inputs(
        numerical_features,
        categorical_features,
        categorical_int_features=categorical_features_int
    )
    feature_list = encode_inputs(
        inputs,
        numerical_features,
        categorical_features,
        categorical_features_with_vocabulary,
        categorical_features_int=categorical_features_int,
        use_embedding=True,
        encoding_size=encoding_size,
        return_as_list=True
    )
    num_features = len(feature_list)

    features = VariableSelection(num_features, encoding_size, dropout_rate)(feature_list)

    outputs = layers.Dense(1, activation="sigmoid")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    return model


def run_grn_and_vsn_model(
        train_df: pd.DataFrame,
        numerical_features,
        categorical_features,
        categorical_features_int,
        categorical_features_with_vocabulary,
        label_cols,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.15,
        encoding_size: int = 16,
        batch_size: int = 256,
        shuffle: bool = True,
        test_size: float = 0.1
):
    assert train_df is not None, "Train dataframe has to be provided"

    model = create_grn_and_vsn_model(
        numerical_features,
        categorical_features,
        categorical_features_int,
        categorical_features_with_vocabulary,
        learning_rate,
        dropout_rate,
        encoding_size
    )
    print(model.summary())

    train_data = train_df

    train_data, test_data = train_test_split(
        train_data[numerical_features + categorical_features + categorical_features_int + label_cols],
        test_size=test_size,
        shuffle=shuffle,
        stratify=train_data[label_cols]
    )
    train_ds = dataframe_to_dataset(
        train_data[numerical_features + categorical_features + categorical_features_int + label_cols],
        label_cols=label_cols
    )
    test_ds = dataframe_to_dataset(
        test_data[numerical_features + categorical_features + categorical_features_int + label_cols],
        label_cols=label_cols
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_accuracy",
            patience=10,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds.batch(batch_size),
        epochs=epochs,
        validation_data=test_ds.batch(batch_size),
        callbacks=callbacks
    )
    evaluate_result = model.evaluate(test_ds.batch(batch_size))

    return model, [history, evaluate_result]


def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
    mlp_layers = []

    for units in hidden_units:
        mlp_layers.append(normalization_layer)
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))

    return tf.keras.Sequential(mlp_layers, name=name)


def create_tabtransformer_classifier(
    numerical_features,
    categorical_features,
    categorical_features_with_vocabulary,
    num_transformer_blocks,
    num_heads,
    mlp_hidden_units_factors,
    dropout_rate,
    embedding_dims,
    learning_rate: float,
    use_column_embedding=False
):
    inputs = create_model_inputs(numerical_features, categorical_features)
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs_separate(
        inputs,
        numerical_features,
        categorical_features,
        categorical_features_with_vocabulary,
        embedding_dims
    )
    encoded_categorical_features = tf.stack(encoded_categorical_feature_list, axis=1)
    numerical_features = layers.concatenate(numerical_feature_list)

    if use_column_embedding:
        num_columns = encoded_categorical_features.shape[1]
        column_embedding = layers.Embedding(input_dim=num_columns, output_dim=embedding_dims)
        column_indices = tf.range(start=0, limit=num_columns, delta=1)
        encoded_categorical_features = encoded_categorical_features + column_embedding(column_indices)

    for block_idx in range(num_transformer_blocks):
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f"multihead_attention_{block_idx}"
        )(encoded_categorical_features, encoded_categorical_features)
        x = layers.Add(name=f"skip_connection1_{block_idx}")([attention_output, encoded_categorical_features])
        x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)

        feedforward_output = create_mlp(
            hidden_units=[embedding_dims],
            dropout_rate=dropout_rate,
            activation=tf.keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            name=f"feedforward_{block_idx}"
        )(x)
        x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
        encoded_categorical_features = layers.LayerNormalization(name=f"layer_norm2_{block_idx}", epsilon=1e-6)(x)

    categorical_features = layers.Flatten()(encoded_categorical_features)
    numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)
    features = layers.concatenate([categorical_features, numerical_features])

    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=tf.keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="MLP"
    )(features)

    outputs = layers.Dense(1, activation="sigmoid", name="sigmoid")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    return model


def run_tabtransformer_model(
    train_df: pd.DataFrame,
    numerical_features,
    categorical_features,
    categorical_features_with_vocabulary,
    label_cols,
    num_transformer_blocks,
    num_heads,
    mlp_hidden_units_factors,
    dropout_rate,
    embedding_dims,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    shuffle: bool = True,
    test_size: float = 0.1,
    number_of_splits: int = 3
):
    assert train_df is not None, "Train dataframe has to be provided"

    model = create_tabtransformer_classifier(
        numerical_features,
        categorical_features,
        categorical_features_with_vocabulary,
        num_transformer_blocks,
        num_heads,
        mlp_hidden_units_factors,
        dropout_rate,
        embedding_dims,
        learning_rate=learning_rate
    )
    print(model.summary())

    # train_data, validation_data = train_test_split(train_df, shuffle=shuffle, test_size=test_size)
    #
    # validation_ds = dataframe_to_dataset(
    #     validation_data[numerical_features + categorical_features + label_cols],
    #     label_cols=label_cols
    # )
    train_data = train_df

    train_data_x, train_data_y = train_data[numerical_features + categorical_features], train_data[label_cols]
    skf = StratifiedKFold(n_splits=number_of_splits, shuffle=shuffle)

    result = []
    for train_idx, test_idx in skf.split(train_data_x, train_data_y):
        train_ds = dataframe_to_dataset(
            train_data.iloc[train_idx][numerical_features + categorical_features + label_cols],
            label_cols=label_cols
        )
        test_ds = dataframe_to_dataset(
            train_data.iloc[test_idx][numerical_features + categorical_features + label_cols],
            label_cols=label_cols
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
        ]
        history = model.fit(
            train_ds.batch(batch_size),
            epochs=epochs,
            validation_data=test_ds.batch(batch_size),
            callbacks=callbacks
        )
        evaluate_result = model.evaluate(test_ds.batch(batch_size))

        result.append({
            "history": history.history,
            "evaluateResult": evaluate_result
        })

    return model, result


def run_wide_and_deep_model(
    train_df: pd.DataFrame,
    numerical_features,
    categorical_features,
    categorical_features_int,
    categorical_features_with_vocabulary,
    label_cols,
    hidden_units: List[int],
    encoding_size: int,
    learning_rate: float,
    epochs: int,
    dropout_rate: float,
    batch_size: int,
    test_size: float = 0.1,
    shuffle: bool = True,
    resume_model=None
):
    assert train_df is not None, "Train dataframe has to be provided"

    model = resume_model or create_wide_and_deep_model(
        numerical_features,
        categorical_features,
        categorical_features_int,
        categorical_features_with_vocabulary,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        encoding_dims=encoding_size
    )
    print(model.summary())

    train_data = train_df

    train_data, test_data = train_test_split(
        train_data[numerical_features + categorical_features + categorical_features_int + label_cols],
        test_size=test_size,
        shuffle=shuffle,
        stratify=train_data[label_cols]
    )
    train_ds = dataframe_to_dataset(
        train_data[numerical_features + categorical_features + categorical_features_int + label_cols],
        label_cols=label_cols
    )
    test_ds = dataframe_to_dataset(
        test_data[numerical_features + categorical_features + categorical_features_int + label_cols],
        label_cols=label_cols
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds.batch(batch_size),
        epochs=epochs,
        validation_data=test_ds.batch(batch_size),
        callbacks=callbacks
    )
    evaluate_result = model.evaluate(test_ds.batch(batch_size))

    return model, [history, evaluate_result]


def run_deep_and_cross_model(
        train_df: pd.DataFrame,
        numerical_features,
        categorical_features,
        categorical_features_int,
        categorical_features_with_vocabulary,
        label_cols,
        hidden_units: List[int],
        encoding_size: int,
        learning_rate: float,
        epochs: int,
        dropout_rate: float,
        batch_size: int,
        test_size: float = 0.1,
        shuffle: bool = True,
        resume_model=None
):
    assert train_df is not None, "Train dataframe has to be provided"

    model = resume_model or create_deep_and_cross_model(
        numerical_features,
        categorical_features,
        categorical_features_int,
        categorical_features_with_vocabulary,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        encoding_dims=encoding_size
    )
    print(model.summary())

    train_data = train_df

    train_data, test_data = train_test_split(
        train_data[numerical_features + categorical_features + categorical_features_int + label_cols],
        test_size=test_size,
        shuffle=shuffle,
        stratify=train_data[label_cols]
    )
    train_ds = dataframe_to_dataset(
        train_data[numerical_features + categorical_features + categorical_features_int + label_cols],
        label_cols=label_cols
    )
    test_ds = dataframe_to_dataset(
        test_data[numerical_features + categorical_features + categorical_features_int + label_cols],
        label_cols=label_cols
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_accuracy",
            patience=10,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds.batch(batch_size),
        epochs=epochs,
        validation_data=test_ds.batch(batch_size),
        callbacks=callbacks
    )
    evaluate_result = model.evaluate(test_ds.batch(batch_size))

    return model, [history, evaluate_result]


def run_tabnet_model(
    train_df: pd.DataFrame,
    feature_columns: List[Text],
    label_cols: List[Text],
    feature_dim,
    output_dim,
    n_step,
    relaxation_factor,
    sparsity_coefficient,
    n_shared,
    bn_momentum,
    learning_rate,
    epochs: int,
    batch_size: int,
    test_size: float = 0.1,
    shuffle: bool = True
):
    train_data = train_df
    train_data, test_data = train_test_split(
        train_data[feature_columns + label_cols],
        test_size=test_size,
        shuffle=shuffle,
        stratify=train_data[label_cols]
    )

    train_x, train_y = train_data[feature_columns], train_data[label_cols]
    test_x, test_y = test_data[feature_columns], test_data[label_cols]

    train_y = tf.keras.utils.to_categorical(train_y, num_classes=2)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=2)

    model = create_tabnet_classifier(
        num_features=train_x.shape[1],
        output_dim=output_dim,
        feature_dim=feature_dim,
        n_step=n_step,
        relaxation_factor=relaxation_factor,
        sparsity_coefficient=sparsity_coefficient,
        n_shared=n_shared,
        bn_momentum=bn_momentum,
        learning_rate=learning_rate
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_x,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_x, test_y),
        callbacks=callbacks,
        verbose=1
    )
    evaluation_result = model.evaluate(test_x, test_y)

    test_y = test_y.argmax(axis=-1)
    test_predictions, test_imps = model.predict(test_x)

    acc_score = accuracy_score(test_y, test_predictions.argmax(axis=-1))
    roc_score = roc_auc_score(test_y, test_predictions.argmax(axis=-1))

    return model, [history, evaluation_result, acc_score, roc_score]


def run_tabnet_model_hyperopt(
    train_df: pd.DataFrame,
    feature_columns: List[Text],
    label_cols: List[Text],
    search_space_params: dict,
    epochs: int = 100,
    batch_size: int = 256,
    number_iterations: int = 100,
    test_size: float = 0.1,
    shuffle: bool = True,
    verbose: int = 2
):
    train_data = train_df
    train_data, test_data = train_test_split(
        train_data[feature_columns + label_cols],
        test_size=test_size,
        shuffle=shuffle,
        stratify=train_data[label_cols]
    )

    train_x, train_y = train_data[feature_columns], train_data[label_cols]
    test_x, test_y = test_data[feature_columns], test_data[label_cols]

    train_y = tf.keras.utils.to_categorical(train_y, num_classes=2)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=2)

    def objective(hyperopt_params: dict):
        hyperopt_params["output_dim"] = hyperopt_params["feature_dim"]

        model = create_tabnet_classifier(
            num_features=train_x.shape[1],
            **hyperopt_params
            # output_dim=output_dim,
            # feature_dim=feature_dim,
            # n_step=n_step,
            # relaxation_factor=relaxation_factor,
            # sparsity_coefficient=sparsity_coefficient,
            # n_shared=n_shared,
            # bn_momentum=bn_momentum,
            # learning_rate=learning_rate
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=30,
                restore_best_weights=True
            )
        ]

        model.fit(
            train_x,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_x, test_y),
            callbacks=callbacks,
            verbose=verbose
        )
        test_y_preds = test_y.argmax(axis=-1)
        test_predictions, test_imps = model.predict(test_x)

        score_result = roc_auc_score(test_y_preds, test_predictions.argmax(axis=-1))

        return -score_result

    print(f"Tabnet Search Space: {search_space_params}")
    best = fmin(objective, search_space_params, algo=tpe.suggest, max_evals=number_iterations)

    return best, train_data
