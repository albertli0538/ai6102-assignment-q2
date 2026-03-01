# Copilot Prompt — AI6102 Assignment Q2 (SVM on a9a)

Copy-paste everything below into **Copilot Chat** (or into a comment at the top of your file) and ask Copilot to generate the code.

---

You are writing Python code (Python 3.10+) to solve **AI6102 Assignment Q2** using **scikit-learn SVC API**.

## Goal
Implement a script (or notebook) that:

1. Downloads the **a9a** dataset (train + test) in LIBSVM / SVMlight sparse format:
   - Train: `https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a`
   - Test: `https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t`

2. Loads the dataset using:
   - `sklearn.datasets.load_svmlight_file` (or `load_svmlight_files`)
   - Keep data as sparse CSR matrices.

3. Runs **3-fold cross-validation (Stratified)** on the training set to produce:
   - **Table 1**: Linear kernel SVC accuracy for \( C \in \{0.01, 0.05, 0.1, 0.5, 1\} \)
   - **Table 2**: RBF kernel SVC accuracy for:
     - \( C \in \{0.01, 0.05, 0.1, 0.5, 1\} \)
     - \( \gamma \in \{0.01, 0.05, 0.1, 0.5, 1\} \)
     - Fill a 5×5 grid with mean CV accuracy (3 folds)

4. Selects the **best kernel + parameters** based on highest mean CV accuracy among:
   - Linear kernel across the 5 C values
   - RBF kernel across all 25 (C, gamma) pairs  
   Tie-break rule (if needed): choose the model with smaller complexity (prefer linear over rbf if same score; otherwise smaller C, then smaller gamma).

5. Trains the best model on the **full training set**, predicts on **test set**, reports test accuracy (**Table 3**) and prints which kernel and parameters were used.

---

## Required details / constraints
- Use `sklearn.svm.SVC` for both linear and rbf (assignment says “SVC API”).
- Use **3-fold StratifiedKFold** with shuffle and fixed seed for reproducibility:
  - `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
- Use `cross_val_score` or manual loop, but must compute mean accuracy for each setting.
- The dataset labels are `+1` and `-1`. Convert to `0/1` or keep as `±1` consistently (accuracy works either way). Be explicit.
- Important: **train/test feature dimension must match**. Use:
  - `load_svmlight_files([train_path, test_path])` with `n_features` inferred from train,
  - OR compute `n_features` from train and pass it to loading test.
- For RBF kernel, include a pipeline to scale features appropriately with sparse data:
  - `StandardScaler(with_mean=False)` (because sparse)
  - Use `Pipeline([('scaler', StandardScaler(with_mean=False)), ('svc', SVC(...))])`
- For linear kernel, you may either scale or not; keep consistent. If you pipeline, do the same style.

---

## Output formatting requirements
Print results in a clear way:

### Table 1 (linear kernel)
A header row:
- `C=0.01 | C=0.05 | C=0.1 | C=0.5 | C=1`  
Then one row of mean accuracies (e.g. `0.8421 ...`) rounded to 4 decimals.

### Table 2 (rbf kernel)
A grid where rows are C and columns are gamma:

Columns: `g=0.01 g=0.05 g=0.1 g=0.5 g=1`  
Rows: `C=0.01, 0.05, 0.1, 0.5, 1`  
Each cell is mean CV accuracy rounded to 4 decimals.

### Table 3 (test accuracy)
Print:
- Best kernel + parameter setting (e.g. “rbf, C=0.5, gamma=0.1” or “linear, C=1”)
- Test accuracy rounded to 4 decimals.

Also print:
- The best mean CV score and which setting achieved it.

Optionally save:
- `table1.csv`, `table2.csv`, and `final_result.txt`.

---

## Implementation guidance (must do)
Write clean, modular code with these functions:

1. `download_file(url, dest_path)` using `urllib.request.urlretrieve` (no external libs).
2. `load_a9a(train_path, test_path)` returning `X_train, y_train, X_test, y_test` as sparse matrices and numpy arrays.
3. `cv_linear_svc(X, y, C_list, cv)` → returns list/dict of accuracies.
4. `cv_rbf_svc(X, y, C_list, gamma_list, cv)` → returns 2D array of accuracies.
5. `select_best(table1, table2, C_list, gamma_list)` → returns best model spec.
6. `train_and_test_best(X_train, y_train, X_test, y_test, best_spec)` → returns test accuracy.

Use:
- `n_jobs=-1` in cross_val_score where safe.
- `random_state=42` for reproducibility.
- `verbose` prints so I can see progress (especially for 25 rbf runs).

---

## Deliverables
Generate **one runnable Python file** called `q2_svm_a9a.py` that:
- downloads data into `./data/`
- runs CV
- prints Table 1, Table 2, Table 3
- prints best setting and accuracy

Include a `main()` and allow execution:

```bash
python q2_svm_a9a.py
```

Also include:
- a top comment summarizing what the script does
- any assumptions and parameter defaults

Now generate the complete code.
