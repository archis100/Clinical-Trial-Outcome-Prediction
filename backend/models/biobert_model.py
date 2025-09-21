# biobert_model.py
# rebuilding the same model architecture as training

import torch
import torch.nn as nn

class BioBERTClassifier(nn.Module):
    def __init__(self, embedding_dim=768, num_embeddings=5, num_classes=2, hidden_dim=256):
        super(BioBERTClassifier, self).__init__()
        # input size = 5 * 768 = 3840
        self.fc1 = nn.Linear(embedding_dim * num_embeddings, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, e1, e2, e3, e4, e5):
        # Concatenate all embeddings
        x = torch.cat((e1, e2, e3, e4, e5), dim=1)  # shape (batch, 3840)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits