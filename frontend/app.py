import sys
import os
from pathlib import Path

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from backend.pipelines.run_inference import predict
from io import BytesIO

st.set_page_config(page_title="Study Status Prediction", page_icon="📊", layout="wide")

st.title("📊 Study Status Prediction")
st.markdown("Upload a CSV file or manually enter study details to predict whether a study is **COMPLETED** or **NOT COMPLETED**.")

# Tabs for CSV Upload and Manual Entry
tab1, tab2 = st.tabs(["📂 Upload CSV", "✍️ Manual Entry"])

# --- Option 1: CSV Upload ---
with tab1:
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        preds = predict(df_new)

        # Only keep final predictions
        final_preds = pd.DataFrame({"Final Prediction": preds["final_predictions"]})

        # Display a preview
        st.subheader("🔎 Predictions Preview")
        st.dataframe(final_preds.head())

        # Download button for predictions
        csv_buffer = BytesIO()
        final_preds.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv"
        )

# --- Option 2: Manual Entry ---
with tab2:
    st.subheader("✍️ Enter Study Details")
    st.markdown("Fill in the fields below to predict the study status.")

    # Placeholders for all the features 
    nct_number = st.text_input("NCT Number", placeholder="e.g., NCT01234567")
    study_title = st.text_area("Study Title", placeholder="e.g., A Study of Drug X in Treating Lung Cancer")
    study_url = st.text_input("Study URL", placeholder="e.g., https://clinicaltrials.gov/ct2/show/NCT01234567")
    acronym = st.text_input("Acronym", placeholder="e.g., LUNG-X")
    brief_summary = st.text_area("Brief Summary", placeholder="e.g., This is a phase 3 trial evaluating the effectiveness of Drug X for lung cancer.")
    study_results = st.selectbox("Study Results", ["YES", "NO"])
    conditions = st.text_input("Conditions", placeholder="e.g., Lung Cancer")
    interventions = st.text_input("Interventions", placeholder="e.g., Drug Y")
    primary_outcome = st.text_input("Primary Outcome Measures", placeholder="e.g., Survival rate")
    secondary_outcome = st.text_input("Secondary Outcome Measures", placeholder="e.g., Side effects")
    other_outcome = st.text_input("Other Outcome Measures", placeholder="Optional")
    sponsor = st.text_input("Sponsor", placeholder="e.g., ABC Research")
    collaborators = st.text_input("Collaborators", placeholder="e.g., University of SFX")
    sex = st.selectbox("Sex", ["ALL", "MALE", "FEMALE"])
    age = st.selectbox("Age", ["ADULT, OLDER_ADULT",
        "ADULT",
        "CHILD, ADULT, OLDER_ADULT",
        "CHILD",
        "CHILD, ADULT",
        "OLDER_ADULT"])
    phases = st.selectbox("Phases", ["PHASE2",
        "PHASE1",
        "PHASE4",
        "PHASE3",
        "PHASE1|PHASE2",
        "PHASE2|PHASE3",
        "EARLY_PHASE1"])
    enrollment = st.number_input("Enrollment", min_value=0, step=1, placeholder="e.g., 500")
    funder_type = st.selectbox("Funder Type", ["OTHER",
        "INDUSTRY",
        "NIH",
        "OTHER_GOV",
        "NETWORK",
        "FED",
        "INDIV",
        "UNKNOWN",
        "AMBIG"])
    study_type = st.selectbox("Study Type", ["INTERVENTIONAL", "OBSERVATIONAL"])
    study_design = st.text_area("Study Design", placeholder="e.g., Intervention Model: PARALLEL | Masking: SINGLE (INVESTIGATOR)")
    other_ids = st.text_input("Other IDs", placeholder="e.g., ABC-123")
    start_date = st.text_input("Start Date", placeholder="e.g., January 2023")
    primary_completion_date = st.text_input("Primary Completion Date", placeholder="e.g., December 2025")
    completion_date = st.text_input("Completion Date", placeholder="e.g., June 2026")
    first_posted = st.text_input("First Posted", placeholder="e.g., February 2023")
    results_first_posted = st.text_input("Results First Posted", placeholder="e.g., N/A")
    last_update_posted = st.text_input("Last Update Posted", placeholder="e.g., September 2025")
    locations = st.text_area("Locations", placeholder="e.g., New York, USA")
    study_documents = st.text_area("Study Documents", placeholder="e.g., Protocol PDF")



    if st.button("🔮 Predict Status"):
        single_data = {
        "NCT Number": nct_number,
        "Study Title": study_title,
        "Study URL": study_url,
        "Acronym": acronym,
        "Brief Summary": brief_summary,
        "Study Results": study_results,
        "Conditions": conditions,
        "Interventions": interventions,
        "Primary Outcome Measures": primary_outcome,
        "Secondary Outcome Measures": secondary_outcome,
        "Other Outcome Measures": other_outcome,
        "Sponsor": sponsor,
        "Collaborators": collaborators,
        "Sex": sex,
        "Age": age,
        "Phases": phases,
        "Enrollment": enrollment,
        "Funder Type": funder_type,
        "Study Type": study_type,
        "Study Design": study_design,
        "Other IDs": other_ids,
        "Start Date": start_date,
        "Primary Completion Date": primary_completion_date,
        "Completion Date": completion_date,
        "First Posted": first_posted,
        "Results First Posted": results_first_posted,
        "Last Update Posted": last_update_posted,
        "Locations": locations,
        "Study Documents": study_documents,
    }

        df_single = pd.DataFrame([single_data])

        preds = predict(df_single)

        # Show final prediction with animation
        final_label = preds["final_predictions"][0]
        if final_label == "COMPLETED":
            st.success(f"✅ Prediction: **{final_label}**", icon="🎉")
        else:
            st.error(f"❌ Prediction: **{final_label}**", icon="⚠️")
