# globals.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent

# --- Load saved artifacts using the absolute path ---
scaler = joblib.load(BACKEND_DIR / "models/scaler_enrollment.pkl")
label_encoders = joblib.load(BACKEND_DIR / "models/feature_label_encoders.pkl")
unique_attributes = joblib.load(BACKEND_DIR / "models/study_design_attributes.pkl")

