# Task 3: Training Considerations

## Freezing Scenarios

# %% [markdown]
### 1. Entire Network Frozen
- No parameters updated: fastest, feature-extraction only, no forgetting.

# %% [markdown]
### 2. Freeze Transformer Backbone Only
- Freeze encoder: train only heads, fast convergence, regularization.

# %% [markdown]
### 3. Freeze One Task Head Only
- Freeze either classifier or NER head: balance tasks with imbalanced data.

## Transfer Learning Workflow

# %% [markdown]
| Stage                 | Frozen Layers              | Trainable Layers                   |
|-----------------------|----------------------------|------------------------------------|
| Head-Only Tuning      | All encoder layers         | Task heads                         |
| Partial Unfreeze      | Bottom N blocks            | Top blocks + heads                 |
| Full Fine-tuning      | None                       | Entire model                       |

# %% [markdown]
### Rationale
- Start with head-only, then gradually unfreeze deeper layers to specialize, use differential LRs.
