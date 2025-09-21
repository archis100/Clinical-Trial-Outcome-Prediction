# categorical.py 
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel

# ------------------------
# Study Design Parsing
# ------------------------

def parse_study_design(study_design, all_attributes):
    # Initialize all allowed attributes as "Unknown"
    attributes = {attr: "Unknown" for attr in all_attributes}

    if study_design and study_design != "Unknown" and pd.notna(study_design):
        for part in study_design.split('|'):
            if ':' in part:
                key, value = part.split(':', 1)
                key, value = key.strip(), value.strip()

                # Only keep keys that are in our unique_attributes list
                if key in all_attributes:
                    attributes[key] = value
                # else: ignore unknown keys (do not create new columns)

    return attributes

def expand_study_design(df, unique_attributes):
    parsed = df['Study Design'].apply(lambda x: parse_study_design(x, unique_attributes))
    study_df = pd.DataFrame(parsed.tolist(), index=df.index)

    # Merge parsed attributes back with df
    df = pd.concat([df, study_df], axis=1)

    # Drop original Study Design column
    df = df.drop(columns=['Study Design'], errors='ignore')

    return df

# ------------------------
# Encoding Categorical Columns
# ------------------------

def encode_categorical(df, label_encoders):
    for col, le in label_encoders.items():
        # Transform using saved encoder; handle unseen labels
        df[col] = df[col].map(lambda x: x if x in le.classes_ else "Unknown")
        df[col] = le.transform(df[col])
    return df

def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize certain categorical columns for inference.
    
    Replaces missing or malformed values with 'Unknown' to match training preprocessing.
    
    Args:
        df (pd.DataFrame): Input dataframe with user data.
        
    Returns:
        pd.DataFrame: DataFrame with cleaned categorical columns.
    """
    columns_to_clean = ['Allocation', 'Intervention Model', 'Masking', 'Primary Purpose']
    
    for col in columns_to_clean:
        # Replace known missing/malformed values with 'Unknown'
        df[col] = df[col].replace(['Unknown', 'NA', '', ' '], 'Unknown')
        # Replace actual NaN values with 'Unknown'
        df[col] = df[col].fillna('Unknown')
    
    return df

