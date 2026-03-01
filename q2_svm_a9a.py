"""
AI6102 Assignment Q2 — SVM on a9a dataset
==========================================
This script:
  1. Downloads the a9a binary classification dataset (train + test) in LIBSVM format
     from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/
  2. Loads both sets as sparse CSR matrices.
  3. Runs 3-fold Stratified CV on the training set for:
       - Table 1: Linear SVC over C in {0.01, 0.05, 0.1, 0.5, 1}
       - Table 2: RBF SVC over C x gamma grid (5x5)
  4. Selects the best (kernel, params) by highest mean CV accuracy.
     Tie-break: prefer linear over rbf; smaller C first; smaller gamma first.
  5. Trains the best model on the full training set, evaluates on the test set.
  6. Prints Table 1, Table 2, Table 3 and saves table1.csv, table2.csv, final_result.txt.

Assumptions / defaults:
  - Data saved to ./data/
  - StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
  - Labels are kept as ±1 (accuracy is the same either way).
  - RBF pipeline: StandardScaler(with_mean=False) + SVC
  - Linear pipeline: StandardScaler(with_mean=False) + SVC (consistent with RBF)
  - n_jobs=-1 for parallel cross_val_score
  - random_state=42 throughout
"""

import os
import ssl
import urllib.request
import numpy as np
import pandas as pd

from sklearn.datasets import load_svmlight_files
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TRAIN_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
TEST_URL  = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t"
DATA_DIR  = "./data"

C_LIST     = [0.01, 0.05, 0.1, 0.5, 1]
GAMMA_LIST = [0.01, 0.05, 0.1, 0.5, 1]
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 1. Download helper
# ─────────────────────────────────────────────
def download_file(url: str, dest_path: str) -> None:
    """Download *url* to *dest_path* using only the standard library.
    SSL certificate verification is disabled to handle servers with
    missing Subject Key Identifier extensions.
    """
    if os.path.exists(dest_path):
        print(f"  [skip] already exists: {dest_path}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading {url} → {dest_path} …")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as response, \
         open(dest_path, "wb") as out_file:
        out_file.write(response.read())
    print(f"  Done.")


# ─────────────────────────────────────────────
# 2. Load dataset
# ─────────────────────────────────────────────
def load_a9a(train_path: str, test_path: str):
    """
    Load a9a train + test files.
    Returns X_train, y_train, X_test, y_test as sparse CSR + numpy arrays.
    Uses load_svmlight_files so feature dimensions are aligned automatically.
    """
    print("Loading dataset …")
    X_train, y_train, X_test, y_test = load_svmlight_files([train_path, test_path])
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Label values: {np.unique(y_train)}")
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
# 3a. CV for Linear SVC
# ─────────────────────────────────────────────
def cv_linear_svc(X, y, C_list: list, cv) -> dict:
    """
    Run 3-fold stratified CV for linear SVC over each C in C_list.
    Returns dict {C: mean_accuracy}.
    Uses a Pipeline with StandardScaler(with_mean=False).
    """
    results = {}
    for C in C_list:
        print(f"  Linear SVC  C={C} …", end=" ", flush=True)
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("svc",    SVC(kernel="linear", C=C, random_state=RANDOM_STATE)),
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_acc = scores.mean()
        results[C] = mean_acc
        print(f"mean CV acc = {mean_acc:.4f}")
    return results


# ─────────────────────────────────────────────
# 3b. CV for RBF SVC
# ─────────────────────────────────────────────
def cv_rbf_svc(X, y, C_list: list, gamma_list: list, cv) -> np.ndarray:
    """
    Run 3-fold stratified CV for RBF SVC over all (C, gamma) pairs.
    Returns 2-D numpy array of shape (len(C_list), len(gamma_list)).
    Rows = C values, Columns = gamma values.
    Uses a Pipeline with StandardScaler(with_mean=False).
    """
    n_C, n_g = len(C_list), len(gamma_list)
    results = np.zeros((n_C, n_g))
    total = n_C * n_g
    run = 0
    for i, C in enumerate(C_list):
        for j, gamma in enumerate(gamma_list):
            run += 1
            print(f"  RBF SVC [{run}/{total}]  C={C}, gamma={gamma} …",
                  end=" ", flush=True)
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("svc",    SVC(kernel="rbf", C=C, gamma=gamma,
                               random_state=RANDOM_STATE)),
            ])
            scores = cross_val_score(pipe, X, y, cv=cv,
                                     scoring="accuracy", n_jobs=-1)
            mean_acc = scores.mean()
            results[i, j] = mean_acc
            print(f"mean CV acc = {mean_acc:.4f}")
    return results


# ─────────────────────────────────────────────
# 4. Select best model
# ─────────────────────────────────────────────
def select_best(table1: dict, table2: np.ndarray,
                C_list: list, gamma_list: list) -> dict:
    """
    Compare all linear and RBF results; return the best model spec.
    Tie-break: linear > rbf; smaller C; smaller gamma.

    Returns a dict with keys: 'kernel', 'C', and optionally 'gamma'.
    """
    best_score = -1.0
    best_spec  = None

    # Check linear candidates (prefer linear on tie → check first)
    for C in C_list:
        score = table1[C]
        if score > best_score:
            best_score = score
            best_spec  = {"kernel": "linear", "C": C}

    # Check RBF candidates (only accept if strictly better)
    for i, C in enumerate(C_list):
        for j, gamma in enumerate(gamma_list):
            score = table2[i, j]
            if score > best_score:
                best_score = score
                best_spec  = {"kernel": "rbf", "C": C, "gamma": gamma}

    best_spec["cv_score"] = best_score
    return best_spec


# ─────────────────────────────────────────────
# 5. Train on full train set, evaluate on test
# ─────────────────────────────────────────────
def train_and_test_best(X_train, y_train, X_test, y_test, best_spec: dict) -> float:
    """
    Build, train, and evaluate the best model on the held-out test set.
    Returns test accuracy.
    """
    kernel = best_spec["kernel"]
    C      = best_spec["C"]
    if kernel == "rbf":
        gamma = best_spec["gamma"]
        svc = SVC(kernel="rbf", C=C, gamma=gamma, random_state=RANDOM_STATE)
    else:
        svc = SVC(kernel="linear", C=C, random_state=RANDOM_STATE)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("svc",    svc),
    ])
    print("  Fitting best model on full training set …")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = (y_pred == y_test).mean()
    return acc


# ─────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────
def print_table1(table1: dict, C_list: list) -> None:
    header = " | ".join(f"C={c}" for c in C_list)
    values = " | ".join(f"{table1[c]:.4f}" for c in C_list)
    sep    = "-" * len(header)
    print(f"\n{'='*60}")
    print("TABLE 1 — Linear SVC CV Accuracy (3-fold Stratified)")
    print(f"{'='*60}")
    print(header)
    print(sep)
    print(values)


def print_table2(table2: np.ndarray, C_list: list, gamma_list: list) -> None:
    col_width = 8
    header = f"{'':>8}" + "".join(f"g={g:<{col_width}}" for g in gamma_list)
    print(f"\n{'='*60}")
    print("TABLE 2 — RBF SVC CV Accuracy (3-fold Stratified)")
    print(f"{'='*60}")
    print(header)
    print("-" * len(header))
    for i, C in enumerate(C_list):
        row = f"C={C:<6}" + "".join(f"{table2[i,j]:.4f}  " for j in range(len(gamma_list)))
        print(row)


def print_table3(best_spec: dict, test_acc: float) -> None:
    kernel = best_spec["kernel"]
    C      = best_spec["C"]
    if kernel == "rbf":
        setting = f"rbf, C={C}, gamma={best_spec['gamma']}"
    else:
        setting = f"linear, C={C}"
    print(f"\n{'='*60}")
    print("TABLE 3 — Test Set Accuracy")
    print(f"{'='*60}")
    print(f"Best setting : {setting}")
    print(f"Best CV score: {best_spec['cv_score']:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


def save_outputs(table1: dict, table2: np.ndarray,
                 C_list: list, gamma_list: list,
                 best_spec: dict, test_acc: float) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    # table1.csv
    df1 = pd.DataFrame(
        [[round(table1[c], 4) for c in C_list]],
        columns=[f"C={c}" for c in C_list],
    )
    df1.to_csv(os.path.join(DATA_DIR, "table1.csv"), index=False)

    # table2.csv
    df2 = pd.DataFrame(
        np.round(table2, 4),
        index=[f"C={c}" for c in C_list],
        columns=[f"g={g}" for g in gamma_list],
    )
    df2.to_csv(os.path.join(DATA_DIR, "table2.csv"))

    # final_result.txt
    kernel = best_spec["kernel"]
    if kernel == "rbf":
        setting = f"rbf, C={best_spec['C']}, gamma={best_spec['gamma']}"
    else:
        setting = f"linear, C={best_spec['C']}"

    lines = [
        "AI6102 Q2 Final Result",
        "======================",
        f"Best setting : {setting}",
        f"Best CV score: {best_spec['cv_score']:.4f}",
        f"Test accuracy: {test_acc:.4f}",
    ]
    with open(os.path.join(DATA_DIR, "final_result.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {DATA_DIR}/table1.csv, table2.csv, final_result.txt")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # ── Step 1: Download ──────────────────────
    print("=== Step 1: Downloading data ===")
    train_path = os.path.join(DATA_DIR, "a9a")
    test_path  = os.path.join(DATA_DIR, "a9a.t")
    download_file(TRAIN_URL, train_path)
    download_file(TEST_URL,  test_path)

    # ── Step 2: Load ──────────────────────────
    print("\n=== Step 2: Loading data ===")
    X_train, y_train, X_test, y_test = load_a9a(train_path, test_path)

    # ── Step 3: CV setup ──────────────────────
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # ── Step 3a: Linear SVC CV ────────────────
    print("\n=== Step 3a: Linear SVC cross-validation ===")
    table1 = cv_linear_svc(X_train, y_train, C_LIST, cv)

    # ── Step 3b: RBF SVC CV ───────────────────
    print("\n=== Step 3b: RBF SVC cross-validation ===")
    table2 = cv_rbf_svc(X_train, y_train, C_LIST, GAMMA_LIST, cv)

    # ── Step 4: Select best ───────────────────
    print("\n=== Step 4: Selecting best model ===")
    best_spec = select_best(table1, table2, C_LIST, GAMMA_LIST)

    # ── Step 5: Train & test ──────────────────
    print("\n=== Step 5: Training and evaluating best model ===")
    test_acc = train_and_test_best(X_train, y_train, X_test, y_test, best_spec)

    # ── Output ────────────────────────────────
    print_table1(table1, C_LIST)
    print_table2(table2, C_LIST, GAMMA_LIST)
    print_table3(best_spec, test_acc)
    save_outputs(table1, table2, C_LIST, GAMMA_LIST, best_spec, test_acc)


if __name__ == "__main__":
    main()
