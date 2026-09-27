"""XGBoost pair-match scorer (stage 06): `hist` tree method on GPU when
available, depth 8, eta 0.05, early stopping on Vcal. Also a
leave-one-country-out diagnostic (train on one country, evaluate on
another, then the reverse) to sanity-check that features generalize rather
than overfitting to a country's quirks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_PARAMS = {
    "max_depth": 8,
    "eta": 0.05,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
}


def _device_params() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"tree_method": "hist", "device": "cuda"}
    except ImportError:
        pass
    return {"tree_method": "hist", "device": "cpu"}


def train_gbdt(
    X_train, y_train, X_val, y_val, num_boost_round: int = 2000, early_stopping_rounds: int = 50,
    feature_names: list[str] | None = None,
) -> xgb.Booster:
    params = {**DEFAULT_PARAMS, **_device_params()}
    # QuantileDMatrix stores the pre-binned (1 byte/value) matrix instead of
    # a float copy of the inputs -- the difference between fitting and not
    # fitting ~15M T rows next to everything else on a 16GB/8GB laptop.
    dtrain = xgb.QuantileDMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain, feature_names=feature_names)
    return xgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds, verbose_eval=False,
    )


def predict(booster: xgb.Booster, X):
    """X: a DataFrame with the training feature names, or a float32 array
    in the booster's feature order."""
    names = None if isinstance(X, pd.DataFrame) else booster.feature_names
    return booster.predict(xgb.DMatrix(X, feature_names=names), iteration_range=(0, booster.best_iteration + 1))


def leave_one_country_out(
    X: np.ndarray, y: np.ndarray, country: np.ndarray, feature_names: list[str], country_a: str, country_b: str,
) -> dict[str, float]:
    """Train on one country's rows of (X, y), score the other's, and the
    reverse; returns average precision for each direction."""
    from sklearn.metrics import average_precision_score

    results = {}
    for train_c, test_c in ((country_a, country_b), (country_b, country_a)):
        tr, te = country == train_c, country == test_c
        if not tr.any() or not te.any():
            continue
        booster = train_gbdt(X[tr], y[tr], X[te], y[te], feature_names=feature_names)
        preds = predict(booster, X[te])
        results[f"train_{train_c}_test_{test_c}"] = float(average_precision_score(y[te], preds))
    return results
