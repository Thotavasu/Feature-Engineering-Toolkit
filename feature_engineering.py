from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Logging configuration
logger = logging.getLogger(__name__)

# Configuration container
@dataclass(frozen=True)
class FeatureConfig:
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    scale_numeric: bool = True
    one_hot_encode: bool = True
    drop_first: bool = False
    sparse_one_hot: bool = True
    scaler_with_mean: bool = False

def handle_missing_values(
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
) -> Tuple[SimpleImputer, SimpleImputer]:
    num_imputer = SimpleImputer(strategy=numeric_strategy)
    cat_imputer = SimpleImputer(strategy=categorical_strategy)
    return num_imputer, cat_imputer

def encode_categorical_features(
        handle_unknown: str = "ignore",
        drop: Optional[str] = None,
        sparse_output: bool = True
) -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown=handle_unknown,
        drop=drop,
        sparse_output=sparse_output,
    )

def scale_numeric_features(with_mean: bool = False) -> StandardScaler:
    return StandardScaler(with_mean=with_mean)

def split_data(
        df: pd.DataFrame,
        target: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        drop_cols: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if drop_cols:
        df =df.drop(columns=drop_cols, errors = "ignore")

    y = df[target]
    X = df.drop(columns=[target])

    stratify_arg = None
    if stratify:
        if y.dtype == object or y.nunique() <= 20:
            stratify_arg = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )
    logger.info(
        "Split data: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
        X_train.shape, X_test.shape, y_train.shape, y_test.shape
    )
    return X_train, X_test, y_train, y_test
def build_feature_pipeline(
    X: pd.DataFrame,
    config: FeatureConfig = FeatureConfig(),
) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    logger.info("Dected %d numeric columns, %d categorical columns",
                len(numeric_cols), len(categorical_cols))
    
    num_imputer, cat_imputer = handle_missing_values(
        numeric_strategy=config.numeric_impute_strategy,
        categorical_strategy=config.categorical_impute_strategy,
    )

    numeric_steps = [("imputer", num_imputer)]
    if config.scale_numeric:
        numeric_steps.append(("scaler", scale_numeric_features(with_mean=config.scaler_with_mean)))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_steps = [("imputer", cat_imputer)]
    if config.one_hot_encode:
        drop = "first" if config.drop_first else None
        categorical_steps.append(
            ("encoder", encode_categorical_features(drop=drop, sparse_output=config.sparse_one_hot))
        )
    categorical_pipeline = Pipeline(steps=categorical_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor