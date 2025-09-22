# globals.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel
from pathlib import Path
from huggingface_hub import hf_hub_download

BACKEND_DIR = Path(__file__).parent.parent

# Download scaler from Hugging Face
scaler_path = hf_hub_download(
    repo_id="archis99/Novartis-models",
    filename="scaler_enrollment.pkl"
)
scaler = joblib.load(scaler_path)

# Download label encoders from hugging face
label_encoders_path = hf_hub_download(
    repo_id="archis99/Novartis-models",
    filename="feature_label_encoders.pkl"
)
label_encoders = joblib.load(label_encoders_path)

# Download unique attributes from hugging face
unique_attributes_path = hf_hub_download(
    repo_id="archis99/Novartis-models",
    filename="study_design_attributes.pkl"
)
unique_attributes = joblib.load(unique_attributes_path)

# # --- Load saved artifacts using the absolute path ---
# scaler = joblib.load(BACKEND_DIR / "models/scaler_enrollment.pkl")
# label_encoders = joblib.load(BACKEND_DIR / "models/feature_label_encoders.pkl")
# unique_attributes = joblib.load(BACKEND_DIR / "models/study_design_attributes.pkl")

