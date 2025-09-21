# scaling.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel

# ------------------------
# Scaling numeric columns
# ------------------------

def scale_numeric(df, scaler):
    """
    Standardize numerical columns using StandardScaler.
    """
    df['Enrollment'] = scaler.transform(df[['Enrollment']])
    return df