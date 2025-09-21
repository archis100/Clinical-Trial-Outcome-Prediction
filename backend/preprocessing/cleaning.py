# cleaning.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def select_required_columns(df: pd.DataFrame, required_cols: list)  -> pd.DataFrame:
    return df[required_cols].copy()

def transform_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sqrt transform to 'Enrollment' column
    """
    df['Enrollment'] = np.sqrt(df['Enrollment'] + 1e-6)
    return df

def fill_missing_numerical(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """
    Fill missing numerical values with the median of each column.
    """
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

def fill_missing_categorical(df: pd.DataFrame, columns_to_clean: list) -> pd.DataFrame:
    """
    Replace 'Unknown', 'NA', '', ' ' and NaN with 'Unknown' in given categorical columns.
    """
    for col in columns_to_clean:
        df[col] = df[col].replace(['Unknown', 'NA', '', ' '], 'Unknown')
        df[col] = df[col].fillna('Unknown')
    return df

def drop_irrelevant_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, errors='ignore')