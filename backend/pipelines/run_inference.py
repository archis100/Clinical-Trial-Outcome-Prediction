import torch
import os
import joblib
import pickle
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from pathlib import Path
from ..models.biobert_model import BioBERTClassifier

# Directory to store downloaded models
MODEL_DIR = Path(__file__).parent.parent / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hugging Face repo and filenames
HF_REPO = "archis99/Novartis-models"  
BIOBERT_FILE = "biobert_classifier.pth"
RF_FILE = "random_forest_model.joblib"
PREPROCESSOR_FILE = "preprocessor.pkl"

# Paths for local files
biobert_path = os.path.join(MODEL_DIR, BIOBERT_FILE)
rf_path = os.path.join(MODEL_DIR, RF_FILE)
preprocessor_path = os.path.join(MODEL_DIR, PREPROCESSOR_FILE)

# Download if not present locally
for file_name, local_path in [(BIOBERT_FILE, biobert_path), 
                              (RF_FILE, rf_path), 
                              (PREPROCESSOR_FILE, preprocessor_path)]:
    if not os.path.exists(local_path):
        print(f"Downloading {file_name} from Hugging Face...")
        hf_hub_download(repo_id=HF_REPO, filename=file_name, local_dir=MODEL_DIR, local_dir_use_symlinks=False)

# Load preprocessor
with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

# Load Random Forest model
rf_model = joblib.load(rf_path)

# Load BioBERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
biobert_model = BioBERTClassifier()
biobert_model.load_state_dict(torch.load(biobert_path, map_location=device))
biobert_model.to(device)
biobert_model.eval()

# Thresholds & weights from training
RF_THRESHOLD = 0.1
BIOBERT_THRESHOLD = 0.3
ENSEMBLE_THRESHOLD = 0.22999999999999995
W1, W2 = 2.0, 0.5

# Label mapping
LABEL_MAP = {0: "COMPLETED", 1: "NOT COMPLETED"}

# Inference function
def predict(df_new: pd.DataFrame):
    # Preprocess input
    X_tabular, embeddings = preprocessor.transform(df_new)

    # Columns to drop for RF
    textual_columns = [
        "Brief Summary",
        "Conditions",
        "Interventions",
        "Primary Outcome Measures",
        "Secondary Outcome Measures"
    ]

    # Keep only RF-relevant features
    X_tabular_rf = X_tabular.drop(columns=textual_columns, errors="ignore")

    # RF prediction (probabilities)
    rf_probs = rf_model.predict_proba(X_tabular_rf)[:, 1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BioBERT prediction
    e1, e2, e3, e4, e5 = [embeddings[col].to(device) for col in textual_columns]  # unpack embeddings
    with torch.no_grad():
        logits = biobert_model(e1, e2, e3, e4, e5)
        biobert_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # Ensemble (soft voting with weights)
    combined_probs = (W1 * rf_probs + W2 * biobert_probs) / (W1 + W2)

    # Final binary predictions using tuned threshold
    final_preds = (combined_probs > ENSEMBLE_THRESHOLD).astype(int)

    # Map to human-readable labels
    final_labels = [LABEL_MAP[p] for p in final_preds]

    return {
        # "rf_probs": rf_probs.tolist(),
        # "biobert_probs": biobert_probs.tolist(),
        # "combined_probs": combined_probs.tolist(),
        "final_predictions": final_labels
    }
