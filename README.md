# AI6102 Assignment — Question 2: SVM on a9a

Binary classification with Support Vector Machines using scikit-learn's SVC API on the [a9a dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a).

## Task Overview

1. Download the a9a dataset (train + test) in LIBSVM format.
2. Run 3-fold stratified cross-validation on the training set for:
   - **Linear SVC** over `C ∈ {0.01, 0.05, 0.1, 0.5, 1}`
   - **RBF SVC** over a 5×5 grid of `C ∈ {0.01, 0.05, 0.1, 0.5, 1}` and `γ ∈ {0.01, 0.05, 0.1, 0.5, 1}`
3. Select the best model by highest mean CV accuracy.
4. Train the best model on the full training set and report test accuracy.

## Repository Structure

```
.
├── q2_svm_a9a.py       # Main script
├── requirements.txt    # Python dependencies
├── README.md
└── data/               # Created at runtime
    ├── a9a             # Training set (downloaded)
    ├── a9a.t           # Test set (downloaded)
    ├── table1.csv      # Linear SVC CV results
    ├── table2.csv      # RBF SVC CV results (5×5 grid)
    └── final_result.txt
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays |
| `pandas` | CSV output |
| `scikit-learn` | SVC, pipelines, cross-validation, data loading |

## Setup

```bash
# (Optional) create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python q2_svm_a9a.py
```

The script will:
1. Download the dataset into `./data/` (skipped if files already exist).
2. Print progress for each CV run.
3. Print Table 1, Table 2, and Table 3.
4. Save `data/table1.csv`, `data/table2.csv`, and `data/final_result.txt`.

## Expected Output

### Table 1 — Linear SVC CV Accuracy
```
C=0.01 | C=0.05 | C=0.1 | C=0.5 | C=1
--------------------------------------
0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX
```

### Table 2 — RBF SVC CV Accuracy (rows = C, columns = γ)
```
        g=0.01  g=0.05  g=0.1   g=0.5   g=1
C=0.01  0.XXXX  0.XXXX  0.XXXX  0.XXXX  0.XXXX
C=0.05  ...
...
```

### Table 3 — Test Accuracy
```
Best setting : <kernel, C, [gamma]>
Best CV score: 0.XXXX
Test accuracy: 0.XXXX
```

## Implementation Details

| Detail | Value |
|---|---|
| Cross-validation | `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` |
| Feature scaling | `StandardScaler(with_mean=False)` (sparse-safe) |
| Labels | `±1` (as loaded from LIBSVM format) |
| Parallelism | `n_jobs=-1` in `cross_val_score` |
| Tie-break rule | linear > rbf; then smaller C; then smaller γ |
| Random seed | `42` throughout |

Both kernels use the same pipeline structure:
```
StandardScaler(with_mean=False) → SVC(kernel=..., C=..., [gamma=...])
```

## Dataset

- **Name**: a9a (derived from UCI Adult dataset)
- **Task**: Binary classification (income >50K or not)
- **Train**: 32,561 samples, 123 features
- **Test**: 16,281 samples, 123 features
- **Source**: [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a)
