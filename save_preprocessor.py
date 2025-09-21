# save_preprocessor.py
from transformers import AutoTokenizer, AutoModel
from backend.pipelines.preprocessor_pipeline import Preprocessor
import pandas as pd

# Define dataset columns (adapt to your dataset)
sample_df = pd.DataFrame([{
    "Brief Summary": "This is a sample study.",
    "Study Results": "Has Results",
    "Conditions": "Condition A",
    "Interventions": "Drug X",
    "Primary Outcome Measures": "Outcome 1",
    "Secondary Outcome Measures": "Outcome 2",
    "Sponsor": "XYZ Corp",
    "Sex": "All",
    "Age": "Adult",
    "Funder Type": "Industry",
    "Phases": "Phase 2",
    "Enrollment": 120,
    "Study Type": "Interventional",
    "Study Design": "Intervention: Randomized|Masking: Double",
}])

required_cols = sample_df.columns.tolist()
categorical_cols = [
    "Study Results", "Sex", "Age", "Funder Type", "Phases",
    "Study Type"
]
columns_to_drop = ["Sponsor", "Observational Model", "Time Perspective"]
text_columns = [
    "Brief Summary", "Conditions", "Interventions",
    "Primary Outcome Measures", "Secondary Outcome Measures"
]

# Load BioBERT
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.cls_token or "[PAD]"

# Create and save preprocessor
preprocessor = Preprocessor(
    required_cols,
    categorical_cols,
    columns_to_drop,
    text_columns,
    tokenizer=tokenizer,
    biobert_model=model,
    device="cpu"
)
preprocessor.save("backend/models/preprocessor.pkl")
