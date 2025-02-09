import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any, Tuple, Optional


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.25, random_state: int = 42) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into training and validation sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])


def get_feature_columns(df: pd.DataFrame) -> Tuple[list, list]:
    """Identifies numerical and categorical feature columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols


def scale_numeric_features(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """Scales numeric features using a trained MinMaxScaler."""
    numeric_cols, _ = get_feature_columns(df)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


def encode_categorical_features(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """Encodes categorical features using a trained OneHotEncoder."""
    _, categorical_cols = get_feature_columns(df)
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    df[encoded_cols] = encoder.transform(df[categorical_cols])
    return df


def preprocess_new_data(df: pd.DataFrame, scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    """Preprocesses new input data using trained scaler and encoder."""
    df = scale_numeric_features(df, scaler)
    df = encode_categorical_features(df, encoder)
    return df


def process_data(raw_df: pd.DataFrame, target_col: Optional[str] = None, drop_cols: list = None) -> Dict[str, Any]:
    """Processes raw data by scaling and encoding features. Splits data if target_col is provided."""
    drop_cols = drop_cols or []

    if target_col:
        train_df, val_df = split_data(raw_df, target_col=target_col)
        input_cols = [col for col in train_df.columns if col not in drop_cols + [target_col]]
        train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
        val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()

        numeric_cols, categorical_cols = get_feature_columns(train_inputs)
        scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
        encoder = OneHotEncoder(sparse_output=False).fit(train_inputs[categorical_cols])

        train_inputs = scale_numeric_features(train_inputs, scaler)
        val_inputs = scale_numeric_features(val_inputs, scaler)
        train_inputs = encode_categorical_features(train_inputs, encoder)
        val_inputs = encode_categorical_features(val_inputs, encoder)

        return {
            'X_train': train_inputs,
            'train_targets': train_targets,
            'X_val': val_inputs,
            'val_targets': val_targets,
            'input_cols': train_inputs.columns.tolist(),
            'scaler': scaler,
            'encoder': encoder
        }
    else:
        numeric_cols, categorical_cols = get_feature_columns(raw_df)
        scaler = MinMaxScaler().fit(raw_df[numeric_cols])
        encoder = OneHotEncoder(sparse_output=False).fit(raw_df[categorical_cols])
        raw_df = scale_numeric_features(raw_df, scaler)
        raw_df = encode_categorical_features(raw_df, encoder)

        return {
            'processed_data': raw_df,
            'input_cols': raw_df.columns.tolist(),
            'scaler': scaler,
            'encoder': encoder
        }
