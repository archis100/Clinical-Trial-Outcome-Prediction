# preprocessor_pipeline.py
import joblib
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Import all helper funcs & globals from preprocessing package
from ..preprocessing.cleaning import (
    drop_duplicates, select_required_columns, transform_numeric,
    fill_missing_numerical, fill_missing_categorical, drop_irrelevant_columns
)
from ..preprocessing.categorical import expand_study_design, encode_categorical, clean_categorical_columns
from ..preprocessing.scaling import scale_numeric     
from ..preprocessing.text_processing import preprocess_text_columns, tokenize_text_columns
from ..preprocessing.embeddings import extract_text_embeddings
from ..preprocessing.globals import scaler, label_encoders, unique_attributes


class Preprocessor:
    def __init__(self, required_cols, categorical_cols, columns_to_drop, text_columns,
                 tokenizer=None, biobert_model=None, device="cpu"):
        self.required_cols = required_cols
        self.categorical_cols = categorical_cols
        self.columns_to_drop = columns_to_drop
        self.text_columns = text_columns
        self.tokenizer = tokenizer
        self.biobert_model = biobert_model
        self.device = device

    def transform(self, df: pd.DataFrame):
        """Run full preprocessing on a dataframe."""
        df = drop_duplicates(df)
        df = select_required_columns(df, self.required_cols)
        df = transform_numeric(df)
        df = fill_missing_numerical(df, ["Enrollment"])
        df = fill_missing_categorical(df, self.categorical_cols)
        df = expand_study_design(df, unique_attributes)
        df = drop_irrelevant_columns(df, self.columns_to_drop)
        df = clean_categorical_columns(df)
        df = encode_categorical(df, label_encoders)
        df = scale_numeric(df, scaler)
        df = preprocess_text_columns(df, self.text_columns)

        embeddings = None
        if self.tokenizer is not None and self.biobert_model is not None:
            tokenized_dict = tokenize_text_columns(df, self.text_columns, self.tokenizer)
            embeddings = extract_text_embeddings(
                tokenized_dict,
                self.biobert_model,
                device=self.device
            )

        return df, embeddings

    def save(self, path="models/preprocessor.pkl"):
        """Save preprocessor object."""
        joblib.dump(self, path)

    @staticmethod
    def load(path="models/preprocessor.pkl"):
        """Load preprocessor object."""
        return joblib.load(path)
