# test_inference.py
import pandas as pd
from pipelines.preprocessor_pipeline import Preprocessor

# Load saved preprocessor
preprocessor = Preprocessor.load("models/preprocessor.pkl")

# Sample new data for inference
df_new = pd.DataFrame([{
    ""
    "NCT Number": "NCT01234567",
    "Study Title": "A Study of Drug X in Treating Lung Cancer",
    "Study URL": "https://clinicaltrials.gov/ct2/show/NCT01234567",
    "Acronym": "LUNG-X",
    "Brief Summary": "This is a phase 3 trial evaluating the effectiveness of Drug X for lung cancer.",
    "Study Results": "NO",
    "Conditions": "Lung Cancer",
    "Interventions": "Drug Y",
    "Primary Outcome Measures": "Survival rate",
    "Secondary Outcome Measures": "Side effects",
    "Other Outcome Measures": "",
    "Sponsor": "ABC Research",
    "Collaborators": "University of SFX",
    "Sex": "MALE",
    "Age": "garbage value - jhfkjahfaiueuw",
    "Phases": "Phase 3",
    "Enrollment": 500,
    "Funder Type": "Government",
    "Study Type": "Archchisman",
    "Study Design": "Intervention Model: Randomized|Masking: QUADRUPLE (PARTICIPANT, CARE_PROVIDER, INVESTIGATOR, OUTCOMES_ASSESSOR)|Observational Model: Observing|Name: Archchisman Banerjee",
    "Other IDs": "ABC-123",
    "Start Date": "January 2023",
    "Primary Completion Date": "December 2025",
    "Completion Date": "June 2026",
    "First Posted": "February 2023",
    "Results First Posted": "N/A",
    "Last Update Posted": "September 2025",
    "Locations": "New York, USA",
    "Study Documents": "Protocol PDF"
}])

X_tabular, embeddings = preprocessor.transform(df_new)

print("Processed Tabular Features:")
print(X_tabular.head())
X_tabular.to_csv("test.csv")

if embeddings:
    for col, emb in embeddings.items():
        print(f"Embeddings for {col}: {emb.shape}")
