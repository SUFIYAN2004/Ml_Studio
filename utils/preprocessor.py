import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def apply_imputer(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols:
        num_strat = strategy if strategy in ["mean", "median"] else "most_frequent"
        imp = SimpleImputer(strategy=num_strat)
        df[numeric_cols] = imp.fit_transform(df[numeric_cols])

    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

    return df


def apply_label_encoding(df: pd.DataFrame, cols: list):
    df = df.copy()
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def apply_onehot_encoding(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=cols, drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def encode_target(series: pd.Series):
    le = LabelEncoder()
    encoded = le.fit_transform(series.astype(str))
    return pd.Series(encoded, name=series.name), le


def apply_scaler(X: np.ndarray, method: str = "none"):
    if method == "standard":
        sc = StandardScaler()
        return sc.fit_transform(X), sc
    elif method == "minmax":
        sc = MinMaxScaler()
        return sc.fit_transform(X), sc
    return X, None
