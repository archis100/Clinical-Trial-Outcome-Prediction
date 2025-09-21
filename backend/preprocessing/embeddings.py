# embeddings.py 
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel


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
