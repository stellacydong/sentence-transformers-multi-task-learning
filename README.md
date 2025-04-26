# sentence-transformers-multi-task-learning

This repository contains solutions for the ML Apprentice take-home exercise on Sentence Transformers & Multi-Task Learning. It organizes code, notebooks, write-ups, and Docker support for reproducibility.

## Repository Structure
```
├── 01_sentence_transformer.py               # Task 1: Sentence Transformer script
├── 02_multitask_learning.py                # Task 2: Multi-Task Learning script
├── notebooks/                              # Jupyter notebooks mirroring each task
│   ├── 01_sentence_transformer.ipynb
│   ├── 02_multitask_learning.ipynb
│   ├── 03_training_considerations.ipynb
│   └── 04_training_loop_implementation.ipynb
├── explanations_of_task3_and_task4_in_LaTeX/ # LaTeX source for detailed write-ups
│   ├── Task3_Training_Considerations.tex
│   └── Task4_Training_Loop_Implementation.tex
├── Task3_Training_Considerations.pdf       # PDF summary for Task 3
├── Task4_Training_Loop_Implementation.pdf   # PDF summary for Task 4
├── requirements.txt                        # Python dependencies
├── Dockerfile                              # Containerization definition
├── README.md                               # Project overview (this file)
└── LICENSE                                 # MIT license
```

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sentence-transformers-multi-task-learning.git
cd sentence-transformers-multi-task-learning
```

### 2. Environment Setup

**Using virtualenv**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Using Conda** (optional):
```bash
conda env create -f environment.yml
conda activate ml_apprentice
```

## Usage

### Run scripts directly
```bash
# Task 1
python 01_sentence_transformer.py

# Task 2
python 02_multitask_learning.py

```

### Jupyter Notebooks
Launch any notebook in the `notebooks/` folder:
```bash
jupyter lab notebooks/01_sentence_transformer.ipynb
```

### Docker (Extra Credit)

1. **Build the Docker image**  
   ```bash
   docker build -t sentence-transformers-mtl .
   ```

2. **Run the default script (Task 1)**  
   ```bash
   docker run --rm sentence-transformers-mtl
   ```

3. **Run another task** (e.g. Task 2)  
   ```bash
   docker run --rm sentence-transformers-mtl python 02_multitask_learning.py
   ```

4. **Open an interactive shell in the container**  
   ```bash
   docker run --rm -it sentence-transformers-mtl bash
   ```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
