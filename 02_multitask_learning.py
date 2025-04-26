"""
Task 2: Multi-Task Learning Expansion

This script extends a Sentence Transformer to handle two tasks simultaneously:
  - Task A: Sentence Classification (binary)
  - Task B: Named Entity Recognition (4 classes: O, PER, LOC, ORG)

We share a single BERT encoder and attach two task-specific heads. Data is loaded 
from the IMDB dataset (for Task A) with synthetic NER labels (for Task B). 
"""

# Step 0: Install dependencies (if needed):
# pip install torch transformers datasets numpy

# %% [code]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from datasets import load_dataset
import numpy as np

# %% [code]
# Initialize the BERT tokenizer for 'bert-base-uncased'
# This will convert text into token IDs, handle padding/truncation, etc.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# %% [markdown]
# 1. Data Preparation
#    - Task A uses IMDB sentiment labels (0=negative,1=positive), recast as binary.
#    - Task B uses synthetic per-token NER labels [0–3] for demonstration.
#    - We tokenize text to max_length=128 and return PyTorch tensors.

# %% [code]
# Load the IMDB dataset (train + test splits)
dataset = load_dataset("imdb")

def create_ner_labels(text: str) -> list[int]:
    """
    Create synthetic NER labels for each token in the input text.
    Returns a random integer [0,3] for each token.
    """
    tokens = tokenizer.tokenize(text)
    return np.random.randint(0, 4, size=(len(tokens),)).tolist()

def preprocess(batch: dict) -> dict:
    """
    Tokenize a batch of examples and generate labels for both tasks.
    Input:
      batch['text']: list[str]
      batch['label']: list[int]
    Output dict contains:
      input_ids, attention_mask: torch.Tensor (B, 128)
      task_a_labels: list[int] (B,)
      task_b_labels: list[list[int]] (B, variable length)
    """
    # Convert IMDB labels to binary sentiment labels (0 or 1)
    task_a = [1 if lbl == 1 else 0 for lbl in batch['label']]
    # Generate synthetic NER labels per example
    task_b = [create_ner_labels(txt) for txt in batch['text']]
    # Tokenize text to fixed-length sequences
    encodings = tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    return {
        'input_ids': encodings.input_ids,
        'attention_mask': encodings.attention_mask,
        'task_a_labels': task_a,
        'task_b_labels': task_b
    }

# Apply preprocessing to the train split, removing raw text/label columns
dataset = dataset.map(preprocess, batched=True, remove_columns=['text','label'])

# %% [markdown]
# 2. Model Architecture
#    - Shared BERT encoder
#    - Task A head: linear layer → 2 classes
#    - Task B head: linear layer → 4 classes (per token)
#    - Dropout for regularization

# %% [code]
class MultiTaskTransformer(nn.Module):
    """
    Multi-task model with a shared BERT encoder and two heads:
      - classifier: sentence-level classification (2 classes)
      - ner: token-level tagging (4 classes)
    """
    def __init__(self):
        super().__init__()
        # Load pretrained BERT encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        # Sentence classification head
        self.classifier = nn.Linear(hidden_size, 2)
        # NER head
        self.ner = nn.Linear(hidden_size, 4)
        # Shared dropout layer
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
          input_ids:      (batch_size, seq_len)
          attention_mask: (batch_size, seq_len)
        Returns:
          logits_a: (batch_size, 2)
          logits_b: (batch_size, seq_len, 4)
        """
        # Encode with BERT (shared)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Pooled output for classification ([CLS] token)
        pooled = outputs.pooler_output
        # Sequence output for token-level predictions
        sequence = outputs.last_hidden_state
        # Compute logits
        logits_a = self.classifier(self.dropout(pooled))
        logits_b = self.ner(self.dropout(sequence))
        return logits_a, logits_b

# Instantiate the model
model = MultiTaskTransformer()

# %% [markdown]
# 3. DataLoader & Training Setup
#    - Loss functions: CrossEntropyLoss for both tasks
#    - Optimizer: AdamW
#    - Custom collate_fn to pad token-level labels to length=128 and ignore with -100

# %% [code]
# Define loss functions for each task
loss_fn_a = nn.CrossEntropyLoss()
loss_fn_b = nn.CrossEntropyLoss()

# AdamW optimizer for all model parameters
optimizer = AdamW(model.parameters(), lr=2e-5)

def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Custom collation function to:
      - Stack input_ids/attention_mask tensors
      - Convert sentence labels → tensor
      - Pad token labels to seq_len=128 with ignore index (-100)
    """
    # Stack input IDs and attention masks
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    # Sentence-level labels
    task_a_labels = torch.tensor([item['task_a_labels'] for item in batch], dtype=torch.long)
    # Token-level labels: pad each list to length 128
    task_b_labels = torch.stack([
        F.pad(
            torch.tensor(item['task_b_labels'], dtype=torch.long),
            pad=(0, 128 - len(item['task_b_labels'])),
            value=-100  # ignore index for loss
        ) for item in batch
    ])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'task_a_labels': task_a_labels,
        'task_b_labels': task_b_labels
    }

# Create DataLoader
train_loader = DataLoader(
    dataset['train'],
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# %% [markdown]
# 4. Inference Example
#    - Demonstrates how to use the model for both tasks on a single sentence.

# %% [code]
def predict(text: str) -> dict:
    """
    Returns predictions for:
      - sentiment classification ("Positive"/"Negative")
      - synthetic NER tags per token
    """
    # Tokenize single input
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    model.eval()
    with torch.no_grad():
        logits_a, logits_b = model(encoding.input_ids, encoding.attention_mask)
    # Task A prediction
    pred_class = torch.argmax(logits_a, dim=1).item()
    sentiment = 'Positive' if pred_class == 1 else 'Negative'
    # Task B prediction
    tags = torch.argmax(logits_b, dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
    return {'sentiment': sentiment, 'entities': list(zip(tokens, tags))}

# Example inference
if __name__ == "__main__":
    sample = "I didn't like that movie at all; it was a waste of time."
    result = predict(sample)
    print("Inference result:", result)
