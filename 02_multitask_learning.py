# Task 2: Multi-Task Learning Expansion
## Shared encoder + two task-specific heads (classification & NER)

# %% [code]
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# %% [markdown]
## 1. Data Preparation

# %% [code]
dataset = load_dataset("imdb")

def create_ner_labels(text):
    tokens = tokenizer.tokenize(text)
    return np.random.randint(0, 4, size=(len(tokens),)).tolist()

def preprocess(batch):
    task_a = [1 if lbl == 1 else 0 for lbl in batch['label']]
    task_b = [create_ner_labels(txt) for txt in batch['text']]
    enc = tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    return {
        'input_ids': enc.input_ids,
        'attention_mask': enc.attention_mask,
        'task_a_labels': task_a,
        'task_b_labels': task_b
    }

dataset = dataset.map(preprocess, batched=True, remove_columns=['text','label'])

# %% [markdown]
## 2. Model Architecture

# %% [code]
class MultiTaskTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.ner = nn.Linear(self.bert.config.hidden_size, 4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        sequence = outputs.last_hidden_state
        logits_a = self.classifier(self.dropout(pooled))
        logits_b = self.ner(self.dropout(sequence))
        return logits_a, logits_b

model = MultiTaskTransformer()

# %% [markdown]
## 3. DataLoader & Training Setup

# %% [code]
loss_fn_a = nn.CrossEntropyLoss()
loss_fn_b = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(ex['input_ids']) for ex in batch])
    attention_mask = torch.stack([torch.tensor(ex['attention_mask']) for ex in batch])
    task_a = torch.tensor([ex['task_a_labels'] for ex in batch], dtype=torch.long)
    task_b = torch.stack([
        F.pad(torch.tensor(ex['task_b_labels'], dtype=torch.long), (0, 128-len(ex['task_b_labels'])), value=-100)
        for ex in batch
    ])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'task_a_labels': task_a, 'task_b_labels': task_b}

train_loader = DataLoader(dataset['train'], batch_size=16, shuffle=True, collate_fn=collate_fn)

# %% [markdown]
## 4. Inference Example

# %% [code]
def predict(text):
    enc = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    model.eval()
    with torch.no_grad():
        log_a, log_b = model(enc.input_ids, enc.attention_mask)
    pred_a = torch.argmax(log_a, dim=1).item()
    sentiment = 'Positive' if pred_a==1 else 'Negative'
    tags = torch.argmax(log_b, dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[0])
    return {'sentiment': sentiment, 'entities': list(zip(tokens, tags))}

print(predict("Christopher Nolan directed Inception in London."))
