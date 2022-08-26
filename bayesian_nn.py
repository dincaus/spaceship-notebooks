import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from typing import List, Text
from sklearn.model_selection import StratifiedKFold

tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tfp.layers


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size

    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )

    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size

    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )

    return posterior_model


def create_probabilistic_bnn_model(
        feature_dim: int,
        output_dim: int,
        train_size: int,
        hidden_units: List[int],
        learning_rate: float
):
    assert feature_dim, "Feature dimension has to be provided"
    assert output_dim, "Output dimension has to be provided"
    assert train_size, "Train size has to be provided"
    assert hidden_units, "Hidden units has to be provided"
    assert learning_rate is not None, "Learning rate has to be provided"

    def negative_loglikelihood(targets, estimated_distribution):
        return -estimated_distribution.log_prob(targets)

    inputs = tf.keras.layers.Input(shape=(feature_dim, ))

    for unit_idx, unit in enumerate(hidden_units):
        if unit_idx == 0:
            features = tfl.DenseVariational(
                units=unit,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1. / train_size,
                activation="sigmoid"
            )(inputs)
        else:
            features = tfl.DenseVariational(
                units=unit,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1. / train_size,
                activation="sigmoid"
            )(features)

    distribution_params = tf.keras.layers.Dense(
        tfl.IndependentBernoulli.params_size(8)
    )(features)
    outputs = tfl.IndependentBernoulli(8)(distribution_params)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=negative_loglikelihood,
        metrics=["accuracy"]
    )

    return model


def create_bnn_model(
        feature_dim: int,
        output_dim: int,
        train_size: int,
        hidden_units: List[int],

        learning_rate: float
):
    assert feature_dim, "Feature dimension has to be provided"
    assert output_dim, "Output dimension has to be provided"
    assert train_size, "Train size has to be provided"
    assert hidden_units, "Hidden units has to be provided"
    assert learning_rate is not None, "Learning rate has to be provided"

    inputs = tf.keras.layers.Input(shape=(feature_dim,))
    features = tf.keras.layers.BatchNormalization()(inputs)

    for unit in hidden_units:
        features = tfl.DenseVariational(
            units=unit,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1. / train_size,
            activation="sigmoid"
        )(features)

    outputs = tf.keras.layers.Dense(units=output_dim, activation="sigmoid")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            "accuracy",
        ]
    )

    return model


def create_bnn_vi(
        feature_dim: int,
        output_dim: int,
        train_size: int,
        hidden_units: List[int],
        learning_rate: float
):
    assert feature_dim, "Feature dimension has to be provided"
    assert output_dim, "Output dimension has to be provided"
    assert train_size, "Train size has to be provided"
    assert hidden_units, "Hidden units has to be provided"
    assert learning_rate is not None, "Learning rate has to be provided"

    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(train_size, dtype=tf.float32))

    model_layers = []
    inputs = tf.keras.layers.Input(shape=(feature_dim, ))

    model_layers.append(inputs)

    for hidden_unit in hidden_units:
        model_layers.append(
            tfl.DenseFlipout(hidden_unit, activation=tf.nn.relu, kernel_divergence_fn=kl_divergence_function)
        )

    model_layers.append(
        tfl.DenseFlipout(output_dim, kernel_divergence_fn=kl_divergence_function, activation="sigmoid")
    )
    model = tf.keras.Sequential(model_layers)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            "accuracy",
        ]
    )

    return model


def run_training_bnn_probabilistic(
        train_df: pd.DataFrame,
        hidden_units: List[int],
        feature_columns: List[Text],
        label_columns: List[Text],
        learning_rate=1e-2,
        batch_size=128,
        epochs=500,
        verbose=1,
        number_of_splits=10,
        shuffle: bool = True
):
    train_data_x, train_data_y = train_df[feature_columns], train_df[label_columns]
    train_x, train_y = train_data_x.to_numpy().astype(np.float32), train_data_y.to_numpy().astype(np.float32)

    model = create_probabilistic_bnn_model(
        feature_dim=train_x.shape[1],
        output_dim=1,
        train_size=train_x.shape[0] - (train_x.shape[0] // number_of_splits),
        learning_rate=learning_rate,
        hidden_units=hidden_units
    )
    print(f"Starting with BNN training with parameters: "
          f"epochs={epochs}, "
          f"verbose={verbose}, "
          f"number_of_splits={number_of_splits}, "
          f"feature_columns={feature_columns}, ",
          f"label_columns={label_columns}, "
          f"train_size={train_x.shape[0]}, "
          f"feature_dim={train_x.shape[1]}")
    print(model.summary())

    skf = StratifiedKFold(n_splits=number_of_splits, shuffle=shuffle)

    training_results = []
    for train_index, test_index in skf.split(train_x, train_y):
        history = model.fit(
            train_x[train_index],
            train_y[train_index],
            epochs=epochs,
            verbose=verbose,
            batch_size=batch_size,
            validation_data=(
                train_x[test_index],
                train_y[test_index]
            )
        )
        training_results.append({
            "history": history
        })

    return training_results, model


def run_training_bnn_vi(
        train_df: pd.DataFrame,
        hidden_units: List[int],
        feature_columns: List[Text],
        label_columns: List[Text],
        learning_rate=1e-2,
        epochs=500,
        verbose=1,
        number_of_splits=10,
        shuffle: bool = True
):
    train_data_x, train_data_y = train_df[feature_columns], train_df[label_columns]
    train_x, train_y = train_data_x.to_numpy().astype(np.float32), train_data_y.to_numpy().astype(np.float32)

    model = create_bnn_vi(
        feature_dim=train_x.shape[1],
        output_dim=1,
        train_size=train_x.shape[0] - (train_x.shape[0] // number_of_splits),
        learning_rate=learning_rate,
        hidden_units=hidden_units
    )
    print(f"Starting with BNN training with parameters: "
          f"epochs={epochs}, "
          f"verbose={verbose}, "
          f"number_of_splits={number_of_splits}, "
          f"feature_columns={feature_columns}, ",
          f"label_columns={label_columns}, "
          f"train_size={train_x.shape[0]}, "
          f"feature_dim={train_x.shape[1]}")
    print(model.summary())

    skf = StratifiedKFold(n_splits=number_of_splits, shuffle=shuffle)

    training_results = []
    for train_index, test_index in skf.split(train_x, train_y):
        history = model.fit(
            train_x[train_index],
            train_y[train_index],
            epochs=epochs,
            verbose=verbose,
            validation_data=(
                train_x[test_index],
                train_y[test_index]
            )
        )
        training_results.append({
            "history": history
        })

    return training_results, model


def run_training(
        train_df: pd.DataFrame,
        hidden_units: List[int],
        feature_columns: List[Text],
        label_columns: List[Text],
        learning_rate=1e-2,
        epochs=500,
        verbose=1,
        number_of_splits=10,
        shuffle: bool = True
):
    train_data_x, train_data_y = train_df[feature_columns], train_df[label_columns]
    train_x, train_y = train_data_x.to_numpy().astype(np.float32), train_data_y.to_numpy().astype(np.float32)

    model = create_bnn_model(
        feature_dim=train_x.shape[1],
        output_dim=1,
        train_size=train_x.shape[0] - (train_x.shape[0] // number_of_splits),
        learning_rate=learning_rate,
        hidden_units=hidden_units
    )
    print(f"Starting with BNN training with parameters: "
          f"epochs={epochs}, "
          f"verbose={verbose}, "
          f"number_of_splits={number_of_splits}, "
          f"feature_columns={feature_columns}, ",
          f"label_columns={label_columns}, "
          f"train_size={train_x.shape[0]}, "
          f"feature_dim={train_x.shape[1]}")
    print(model.summary())

    skf = StratifiedKFold(
        n_splits=number_of_splits,
        shuffle=shuffle
    )

    training_results = []
    for train_index, test_index in skf.split(train_x, train_y):
        history = model.fit(
            train_x[train_index],
            train_y[train_index],
            epochs=epochs,
            verbose=verbose,
            validation_data=(
                train_x[test_index],
                train_y[test_index]
            )
        )
        training_results.append({
            "history": history
        })

    return training_results, model
