"""
Task 1: Sentence Transformer Implementation

In this script, we implement a simple sentence transformer model using Hugging Face Transformers and PyTorch.
The model encodes variable-length sentences into fixed-length embeddings via a pretrained transformer and a pooling strategy.
"""

# Step 1: Install Required Libraries
# If running in a fresh environment, uncomment the following:
# pip install torch transformers numpy

# Step 2: Import Libraries
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Step 3: Define the Sentence Transformer Model
class SentenceTransformerModel(torch.nn.Module):
    """
    Wraps a pretrained transformer to produce fixed-length sentence embeddings.
    
    Pooling options:
      - 'cls': use the [CLS] token embedding.
      - 'mean': average all token embeddings (masking out padding).
    """
    def __init__(self, model_name: str = 'bert-base-uncased', pooling: str = 'cls'):
        super().__init__()
        # Load the pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          input_ids:      shape (batch_size, seq_len)
          attention_mask: shape (batch_size, seq_len)
        Returns:
          embeddings:     shape (batch_size, hidden_size)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        if self.pooling == 'cls':
            # Return the [CLS] token embedding
            return last_hidden[:, 0]

        elif self.pooling == 'mean':
            # Create mask and expand to hidden size
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            # Sum valid token embeddings
            summed = torch.sum(last_hidden * mask, dim=1)
            # Count valid tokens (avoid division by zero)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts

        else:
            raise ValueError("Unsupported pooling type. Use 'cls' or 'mean'.")


def main():
    # Step 4: Initialize Tokenizer and Model
    model_name = 'bert-base-uncased'
    pooling_strategy = 'cls'  # Options: 'cls' or 'mean'

    # Create tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentenceTransformerModel(model_name=model_name, pooling=pooling_strategy)

    # Step 5: Prepare Sample Sentences
    sentences = [
        "The cat sat on the mat.",
        "Artificial intelligence is transforming the world.",
        "I love machine learning!"
    ]

    # Step 6: Tokenize Sentences
    # - padding: pad to longest in batch
    # - truncation: truncate to model max length
    # - return_tensors='pt': return PyTorch tensors
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # Step 7: Generate Embeddings
    model.eval()  # disable dropout
    with torch.no_grad():
        embeddings = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )

    # Step 8: Show the Embeddings
    print("Fixed-length sentence embeddings (one per sentence):")
    print(embeddings)
    print("\nEmbeddings shape:", embeddings.shape)

    # Step 9: Convert to NumPy
    embeddings_np = embeddings.cpu().numpy()
    for idx, sentence in enumerate(sentences):
        print(f"\nSentence: \"{sentence}\"")
        print(f"First 8 dimensions: {embeddings_np[idx][:8]} ...")


if __name__ == "__main__":
    main()
