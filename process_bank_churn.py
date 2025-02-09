import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.25, random_state: int = 42) -> tuple:
    """Splits the dataset into training and validation sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])


def get_feature_columns(df: pd.DataFrame) -> tuple:
    """Identifies numerical and categorical feature columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols


def scale_numeric_features(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: list) -> tuple:
    """Scales numeric features using MinMaxScaler."""
    scaler = MinMaxScaler().fit(train_df[numeric_cols])
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler


def encode_categorical_features(train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: list) -> tuple:
    """Encodes categorical features using OneHotEncoder."""
    encoder = OneHotEncoder(sparse_output=False).fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_encoded = encoder.transform(train_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols])
    train_df[encoded_cols] = train_encoded
    val_df[encoded_cols] = val_encoded
    return train_df, val_df, encoder


def process_data(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Processes raw data by cleaning, splitting, scaling, and encoding features."""
    train_df, val_df = split_data(raw_df, target_col='Exited')
    input_cols = list(train_df.columns[3:-1])

    train_inputs, train_targets = train_df[input_cols].copy(), train_df['Exited'].copy()
    val_inputs, val_targets = val_df[input_cols].copy(), val_df['Exited'].copy()

    numeric_cols, categorical_cols = get_feature_columns(train_inputs)
    train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols)
    train_inputs, val_inputs, encoder = encode_categorical_features(train_inputs, val_inputs, categorical_cols)

    return {
        'X_train': train_inputs,
        'train_targets': train_targets,
        'X_val': val_inputs,
        'val_targets': val_targets,
        'input_cols': train_inputs.columns.tolist(),
        'scaler': scaler,
        'encoder': encoder
    }
