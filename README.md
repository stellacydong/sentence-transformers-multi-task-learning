# README.md

```markdown
# ML Apprentice Take-Home Exercise

This repository contains solutions for the **ML Apprentice** take-home exercise focused on Sentence Transformers and Multi-Task Learning.

## Repository Structure

```bash
├── 01_sentence_transformer.py         # Task 1: Sentence Transformer Implementation
├── 02_multitask_learning.py          # Task 2: Multi-Task Learning Expansion
├── 03_training_considerations.py      # Task 3: Training Considerations (write-up)
├── 04_training_loop_implementation.py # Task 4: Training Loop Implementation (BONUS)
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Containerization setup
├── README.md                          # Project overview and instructions
├── .gitignore                         # Git ignore patterns
└── LICENSE                            # MIT License
```

## Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/ml-apprentice-exercise.git
   cd ml-apprentice-exercise
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Tasks Overview

### Task 1: Sentence Transformer Implementation
- Implements `SentenceTransformerModel` using Hugging Face Transformers.
- Encodes sample sentences into fixed-length embeddings.
- See `01_sentence_transformer.py` for code and detailed comments.

### Task 2: Multi-Task Learning Expansion
- Extends the sentence transformer with two heads:
  - **Task A**: Sentence classification
  - **Task B**: Named Entity Recognition (NER)
- Shared encoder with task-specific heads and combined loss.
- See `02_multitask_learning.py` for implementation details.

### Task 3: Training Considerations
- Discusses parameter-freezing scenarios and transfer-learning strategies.
- Explains when to freeze the entire network, only the backbone, or individual heads.
- Provides rationale balancing performance, compute, and overfitting risk.
- See `03_training_considerations.py` for the full write-up and key insights.

### Task 4: Training Loop Implementation (BONUS)
- Illustrates a multi-task training loop with combined loss and per-task metrics.
- Covers data handling, forward pass, and metrics calculation.
- See `04_training_loop_implementation.py` for code and explanatory comments.

## Usage

Run each script individually:
```bash
python 01_sentence_transformer.py
python 02_multitask_learning.py
python 04_training_loop_implementation.py
```

## Docker Container (Optional)

To build and run the project in a Docker container:

1. **Build the Docker image**
   ```bash
   docker build -t ml-apprentice-exercise .
   ```

2. **Run a specific task** (e.g., Task 1)
   ```bash
   docker run --rm ml-apprentice-exercise python 01_sentence_transformer.py
   ```

3. **Enter an interactive shell** within the container:
   ```bash
   docker run --rm -it ml-apprentice-exercise bash
   ```

## Brief Summaries of Key Tasks

### Task 3 Summary
- **Freezing Scenarios:** Entire network, backbone only, or individual heads.
- **Transfer Learning Workflow:** Staged unfreeze strategy (head-only → partial → full fine-tuning).  
- **Rationale:** Balance rapid convergence, regularization, and representational specialization.

### Task 4 Summary
- **Training Loop:** Handles multi-task batches via a custom `collate_fn`.  
- **Forward Pass:** Single pass yields both sentence-level and token-level logits.  
- **Metrics:** Sentence accuracy for Task A; token accuracy (excluding padding) for Task B.  
- **MTL Dynamics:** Joint gradient flow encourages shared representations beneficial to both tasks.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
```

---

# .gitignore

```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/
.env

# Jupyter Notebook checkpoints
.ipynb_checkpoints

# Docker files
Dockerfile

# Logs and data
*.log
*.csv
*.pt
*.pth

# macOS
.DS_Store
```

---

# Dockerfile

```
# Use official Python 3.10 image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Default command (runs Task 1 example)
CMD ["python", "01_sentence_transformer.py"]
```

---

# LICENSE (MIT)

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

