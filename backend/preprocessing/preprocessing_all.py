# preprocessing_all.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel

# ------------------------
# Load saved artifacts
# ------------------------

scaler = joblib.load("models\scaler_enrollment.pkl")  # StandardScaler for 'Enrollment'
label_encoders = joblib.load("models\label_encoders.pkl")  # Dict of LabelEncoders for categorical columns
unique_attributes = joblib.load("models\study_design_attributes.pkl")  # List of Study Design attributes

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

# ------------------------
# Study Design Parsing
# ------------------------

def parse_study_design(study_design, all_attributes):
    attributes = {attr: "Unknown" for attr in all_attributes}
    if study_design != "Unknown" and pd.notna(study_design):
        for part in study_design.split('|'):
            if ':' in part:
                key, value = part.split(':', 1)
                attributes[key.strip()] = value.strip()
    return attributes

def expand_study_design(df, unique_attributes):
    parsed = df['Study Design'].apply(lambda x: parse_study_design(x, unique_attributes))
    study_df = pd.DataFrame(parsed.tolist(), index=df.index)
    df = pd.concat([df, study_df], axis=1)
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

# ------------------------
# Scaling numeric columns
# ------------------------

def scale_numeric(df, scaler):
    """
    Standardize numerical columns using StandardScaler.
    """
    df['Enrollment'] = scaler.transform(df[['Enrollment']])
    return df

# ------------------------
# Text preprocessing
# ------------------------

def clean_text(text):
    if pd.isna(text):  # Handle missing values
        return ""
    text = text.lower()  # Convert to lowercase
    text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Remove special characters
    return ' '.join(text.split())  # Remove extra whitespaces

def preprocess_text_columns(df, text_columns):
    for col in text_columns:
        df[col] = df[col].fillna("No info provided")
        df[col] = df[col].apply(clean_text)
    return df

# ------------------------
# Tokenization of textual Columns
# ------------------------

def tokenize_text_columns(df, textual_columns, tokenizer, batch_size=50, max_length=256):
    """
    Tokenizes multiple textual columns in batches for inference.

    Args:
        df (pd.DataFrame): DataFrame containing textual columns.
        textual_columns (list): List of column names to tokenize.
        tokenizer: HuggingFace tokenizer.
        batch_size (int): Number of samples per batch.
        max_length (int): Maximum token length per sequence.

    Returns:
        dict: Dictionary with column names as keys and tokenized tensors as values.
    """
    def tokenize_in_batches(column_texts):
        tokenized_batches = []
        for i in range(0, len(column_texts), batch_size):
            batch = column_texts[i:i + batch_size].tolist()
            tokenized_batch = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            tokenized_batches.append(tokenized_batch)
        # Combine batches
        return {
            "input_ids": torch.cat([batch["input_ids"] for batch in tokenized_batches], dim=0),
            "attention_mask": torch.cat([batch["attention_mask"] for batch in tokenized_batches], dim=0)
        }

    tokenized_data = {}
    for col in textual_columns:
        tokenized_data[col] = tokenize_in_batches(df[col])
    return tokenized_data

# ------------------------
# Extract Embeddings
# ------------------------


def extract_text_embeddings(tokenized_data_dict, model, device=None, batch_size=32, save_to_disk=False):
    """
    Extract embeddings from tokenized textual data using BioBERT.

    Args:
        tokenized_data_dict (dict): Dictionary of tokenized columns (output of `tokenize_text_columns`).
        model (transformers.PreTrainedModel): BioBERT model (without classification head).
        device (torch.device, optional): Device to run the model on. Defaults to GPU if available.
        batch_size (int): Batch size for embedding extraction.
        save_to_disk (bool): Whether to save embeddings as .pt files for each column.

    Returns:
        dict: Dictionary of embeddings for each column.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Ensure model is in evaluation mode

    embeddings_dict = {}

    for col, tokenized_data in tokenized_data_dict.items():
        print(f"Extracting embeddings for column: {col}")

        input_ids = tokenized_data["input_ids"]
        attention_mask = tokenized_data["attention_mask"]

        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids_batch, attention_mask_batch = batch
                input_ids_batch = input_ids_batch.to(device)
                attention_mask_batch = attention_mask_batch.to(device)

                outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
                hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]

                # Mean pooling across sequence length
                embeddings = hidden_states.mean(dim=1)
                all_embeddings.append(embeddings.cpu())

        embeddings_col = torch.cat(all_embeddings, dim=0)
        embeddings_dict[col] = embeddings_col

        if save_to_disk:
            torch.save(embeddings_col, f"{col}_embeddings.pt")
            print(f"Saved embeddings for column: {col}")

        print(f"Shape of embeddings for column {col}: {embeddings_col.shape}")

    return embeddings_dict

# ------------------------
# Main preprocessing function
# ------------------------

def preprocess(df, required_cols, categorical_cols, columns_to_drop, text_columns,
               tokenizer=None, biobert_model=None, device='cpu'):
    """
    Full preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame (single row or batch).
        required_cols (list): Columns to select from df.
        categorical_cols (list): Categorical columns to encode.
        columns_to_drop (list): Columns to drop from df.
        text_columns (list): Textual columns to preprocess.
        tokenizer (transformers.AutoTokenizer, optional): BioBERT tokenizer for text.
        biobert_model (transformers.AutoModel, optional): BioBERT model (no classification head).
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        df (pd.DataFrame): Preprocessed tabular DataFrame.
        embeddings (dict or None): Dict of embeddings for text columns, if model provided.
    """
    # Tabular preprocessing
    df = drop_duplicates(df)
    df = select_required_columns(df, required_cols)
    df = transform_numeric(df)
    df = fill_missing_numerical(df, ["Enrollment"])  # median fill for Enrollment
    df = fill_missing_categorical(df, categorical_cols)
    df = drop_irrelevant_columns(df, columns_to_drop)
    df = expand_study_design(df, unique_attributes)
    df = clean_categorical_columns(df)
    df = encode_categorical(df, label_encoders)
    df = scale_numeric(df, scaler)
    df = preprocess_text_columns(df, text_columns)

    embeddings = None
    if tokenizer is not None and biobert_model is not None:
        tokenized_dict = tokenize_text_columns(df, text_columns, tokenizer)
        embeddings = extract_text_embeddings(tokenized_dict, biobert_model, device=device)

    return df, embeddings

