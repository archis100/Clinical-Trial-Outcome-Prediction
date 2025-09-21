# text_processing.py 
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel

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
