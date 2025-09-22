# globals.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel
from pathlib import Path
from huggingface_hub import hf_hub_download

# Base directory
BACKEND_DIR = Path(__file__).parent.parent
MODEL_DIR = BACKEND_DIR / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hugging Face repo
HF_REPO = "archis99/Novartis-models"

# Files to manage
FILES = {
    "scaler": "scaler_enrollment.pkl",
    "label_encoders": "feature_label_encoders.pkl",
    "unique_attributes": "study_design_attributes.pkl"
}

# Dictionary to hold loaded objects
loaded_artifacts = {}

for key, filename in FILES.items():
    local_path = MODEL_DIR / filename

    # If not present locally, download from HF
    if not local_path.exists():
        print(f"Downloading {filename} from Hugging Face...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )

    # Load the artifact 
    loaded_artifacts[key] = joblib.load(local_path)

# Unpack for usage
scaler = loaded_artifacts["scaler"]
label_encoders = loaded_artifacts["label_encoders"]
unique_attributes = loaded_artifacts["unique_attributes"]