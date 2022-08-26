import numpy as np
import pandas as pd

from typing import List, Text, Union

from hyperopt import fmin, tpe
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def create_xgboost_classifier(**kwargs):
    return XGBClassifier(**kwargs)


def create_ada_boost_classifier(
    base_estimator=None,
    n_estimators=50,
    learning_rate=1.0,
    algorithm="SAMME.R",
    random_state=None
):
    return AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm=algorithm,
        random_state=random_state
    )


def create_random_forest_classifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    ccp_alpha=0.0
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        ccp_alpha=ccp_alpha
    )


def run_xgboost_classifier(
    train_df: pd.DataFrame,
    feature_columns: List[Text],
    label_columns: List[Text],
    number_of_splits=5,
    shuffle: bool = True,
    xgboost_model_predefined=None,
    **xgboost_param
):
    train_data_x, train_data_y = train_df[feature_columns], train_df[label_columns]
    train_x, train_y = train_data_x.to_numpy().astype(np.float32), train_data_y.to_numpy().astype(np.float32)

    model = xgboost_model_predefined or create_xgboost_classifier(**xgboost_param)
    skf = StratifiedKFold(n_splits=number_of_splits, shuffle=shuffle)

    training_results = []
    for train_index, test_index in skf.split(train_x, train_y):
        model.fit(train_x[train_index], train_y[train_index].ravel())
        y_predicted = model.predict(train_x[test_index])
        model_score = model.score(train_x[test_index], train_y[test_index])
        acc_score = accuracy_score(train_y[test_index], y_predicted)
        roc_score = roc_auc_score(train_y[test_index], y_predicted)

        training_results.append({
            "accuracy": acc_score,
            "rocAuc": roc_score,
            "score": model_score
        })

    return training_results, model


def run_xgboost_classifier_search_cv(
    train_df: pd.DataFrame,
    feature_columns: List[Text],
    label_columns: List[Text],
    param_distributions: dict,
    scoring: Text = "roc_auc",
    number_iterations: int = 100,
    number_of_splits=5,
    shuffle: bool = True
):
    train_data_x, train_data_y = train_df[feature_columns], train_df[label_columns]
    train_x, train_y = train_data_x.to_numpy().astype(np.float32), train_data_y.to_numpy().astype(np.float32)

    skf = StratifiedKFold(n_splits=number_of_splits, shuffle=shuffle)

    xgb_model = XGBClassifier()

    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_distributions,
        n_iter=number_iterations,
        scoring=scoring,
        cv=skf,
        verbose=True
    )
    random_search.fit(train_x, train_y)

    return random_search


def run_xgboost_classifier_hyperopt(
    train_df: pd.DataFrame,
    feature_columns: List[Text],
    label_columns: List[Text],
    search_space_params: dict,
    tree_method: Text = "auto",
    number_iterations: int = 100,
    test_size: float = 0.1,
    shuffle: bool = True
):

    train_data_x, train_data_y = train_df[feature_columns], train_df[label_columns]
    # train_x, train_y = train_data_x.to_numpy().astype(np.float32), train_data_y.to_numpy().astype(np.float32)
    train_x, test_x, train_y, test_y = train_test_split(train_data_x, train_data_y, test_size=test_size, shuffle=shuffle)

    def objective(hyperopt_params: dict):
        hyperopt_params["tree_method"] = tree_method
        m = create_xgboost_classifier(**hyperopt_params)
        m.fit(train_x, train_y)

        return -m.score(test_x, test_y)

    print(f"XGBoost Search Space: {search_space_params}")
    print(f"Tree method: {tree_method}")
    best = fmin(objective, search_space_params, algo=tpe.suggest, max_evals=number_iterations)

    return best


def run_random_forest_classifier(
        train_df: pd.DataFrame,
        feature_columns: List[Text],
        label_columns: List[Text],

        number_estimators: int,
        criterion: Text = "gini",
        max_depth: Union[int, None] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,

        number_of_splits=10,
        shuffle: bool = True
):
    train_data_x, train_data_y = train_df[feature_columns], train_df[label_columns]
    train_x, train_y = train_data_x.to_numpy().astype(np.float32), train_data_y.to_numpy().astype(np.float32)

    model = create_random_forest_classifier(
        n_estimators=number_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf
    )

    skf = StratifiedKFold(n_splits=number_of_splits, shuffle=shuffle)

    training_results = []
    for train_index, test_index in skf.split(train_x, train_y):
        model.fit(train_x[train_index], train_y[train_index].ravel())
        training_results.append({
            "accuracy": model.score(train_x[test_index], train_y[test_index])
        })

    return training_results, model


def run_ada_boost_classifier(
    train_df: pd.DataFrame,
    feature_columns: List[Text],
    label_columns: List[Text],
    number_estimators=100,
    learning_rate=1.0,
    number_of_splits=10,
    shuffle: bool = True
):
    train_data_x, train_data_y = train_df[feature_columns], train_df[label_columns]
    train_x, train_y = train_data_x.to_numpy().astype(np.float32), train_data_y.to_numpy().astype(np.float32)

    model = create_ada_boost_classifier(n_estimators=number_estimators, learning_rate=learning_rate)
    skf = StratifiedKFold(n_splits=number_of_splits, shuffle=shuffle)

    training_results = []
    for train_index, test_index in skf.split(train_x, train_y):
        model.fit(train_x[train_index], train_y[train_index].ravel())
        training_results.append({
            "accuracy": model.score(train_x[test_index], train_y[test_index].ravel())
        })

    return training_results, model
