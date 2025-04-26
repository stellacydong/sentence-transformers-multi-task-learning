# Task 1: Sentence Transformer Implementation
# In this notebook script, we implement a sentence transformer model using Hugging Face Transformers and PyTorch.

# %% [markdown]
## Step 1: Install Required Libraries
# !pip install torch transformers

# %% [markdown]
## Step 2: Import Libraries
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# %% [markdown]
## Step 3: Define the Sentence Transformer Model
class SentenceTransformerModel(torch.nn.Module):
    """
    A simple sentence transformer model that wraps a pretrained transformer and adds a pooling strategy.
    """
    def __init__(self, model_name='bert-base-uncased', pooling='cls'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        if self.pooling == 'cls':
            return last_hidden[:, 0]
        elif self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts
        else:
            raise ValueError("Unsupported pooling type. Use 'cls' or 'mean'.")

# %% [markdown]
## Step 4: Initialize Tokenizer and Model
model_name = 'bert-base-uncased'
pooling_strategy = 'cls'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SentenceTransformerModel(model_name=model_name, pooling=pooling_strategy)

# %% [markdown]
## Step 5: Prepare Sample Sentences
sentences = [
    "The cat sat on the mat.",
    "Artificial intelligence is transforming the world.",
    "I love machine learning!"
]

# %% [markdown]
## Step 6: Tokenize Sentences
encoded = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# %% [markdown]
## Step 7: Generate Embeddings
model.eval()
with torch.no_grad():
    embeddings = model(
        input_ids=encoded['input_ids'],
        attention_mask=encoded['attention_mask']
    )

# %% [markdown]
## Step 8: Show the Embeddings
print("Fixed-length sentence embeddings (one per sentence):")
print(embeddings)
print("\nEmbeddings shape:", embeddings.shape)

# %% [markdown]
## Step 9: (Optional) Convert to NumPy
embeddings_np = embeddings.cpu().numpy()
for i, sentence in enumerate(sentences):
    print(f"Sentence: '{sentence}' -> first 8 dims: {embeddings_np[i][:8]}")
