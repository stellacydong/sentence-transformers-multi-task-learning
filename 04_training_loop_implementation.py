# Task 4: Training Loop Implementation (BONUS)

# %% [markdown]
We implement a multi-task training loop: forward pass, loss computation, metrics.

# %% [code]
import torch

# Preconditions: model, train_loader, loss_fn_a, loss_fn_b, optimizer defined.

num_epochs = 3
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0.0
    correct_a, total_a = 0,0
    correct_b, total_b = 0,0

    for batch in train_loader:
        optimizer.zero_grad()
        logits_a, logits_b = model(batch['input_ids'], batch['attention_mask'])
        # Task A
        loss_a = loss_fn_a(logits_a, batch['task_a_labels'])
        preds_a = torch.argmax(logits_a, dim=1)
        correct_a += (preds_a == batch['task_a_labels']).sum().item()
        total_a += batch['task_a_labels'].size(0)
        # Task B
        B,L,C = logits_b.shape
        loss_b = loss_fn_b(logits_b.view(-1,C), batch['task_b_labels'].view(-1))
        preds_b = torch.argmax(logits_b, dim=2)
        mask = batch['task_b_labels'] != -100
        correct_b += ((preds_b == batch['task_b_labels']) & mask).sum().item()
        total_b += mask.sum().item()
        # Combine
        loss = loss_a + loss_b
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss/len(train_loader)
    acc_a = correct_a/total_a
    acc_b = correct_b/total_b
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Task A Acc: {acc_a:.4f} | Task B Acc: {acc_b:.4f}")
