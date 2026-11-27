# Project 04 — Results & Limitations Write-Up

## 1. Prediction Task Definition
Predict whether a Titanic passenger survived (`survived`, binary target: `1` = Survived, `0` = Did Not Survive) using four passenger features: ticket class (`pclass`), gender (`sex` encoded as `1` = female, `0` = male), `age`, and ticket `fare`.

---

## 2. Dataset & Train/Test Split
- Loaded the Seaborn `titanic` dataset (`891` total rows), selected the target and four features, and dropped `177` rows missing `age`, leaving **`714` complete passenger records** (`290` survived = `40.6%`, `424` did not survive = `59.4%`).
- Split the data into **80% training (`571` rows)** and **20% unseen test (`143` rows)** using `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` so both sets share the exact `40.6%` survival rate.

---

## 3. Model Results vs. Baseline (Unseen Test Set, `n = 143`)
- **Majority-Class Baseline** (always predict `0` / Did Not Survive): **59.4% accuracy** (`0.5944`), **MSE = 0.4056**.
- **Decision Tree (`max_depth=4, random_state=42`):**
  - **Training Accuracy:** `84.8%` (`0.8476`)
  - **Test Accuracy:** **`79.7%`** (`0.7972`) — outperforms the baseline by **+20.3 percentage points**.
  - **Test MSE:** `0.2028` on binary predictions (`0.1519` Brier MSE on predicted survival probabilities).
  - **Precision / Recall / F1:** Survived (`0.764` / `0.724` / `0.743`); Did Not Survive (`0.818` / `0.847` / `0.832`).

### Confusion Matrix Breakdown (`[[72, 13], [16, 42]]`)
- **True Negatives (`TN = 72`, 50.3%):** Passengers who did not survive and were correctly predicted as Did Not Survive.
- **False Positives (`FP = 13`, 9.1%):** Passengers who did not survive, but the model falsely predicted they Survived.
- **False Negatives (`FN = 16`, 11.2%):** Passengers who actually survived, but the model missed them and predicted Did Not Survive.
- **True Positives (`TP = 42`, 29.4%):** Passengers who survived and were correctly predicted as Survived.

---

## 4. Feature Importances (`model.feature_importances_`)
1. **`sex` (`54.5%`):** Strongest predictor, reflecting the *"women and children first"* evacuation priority.
2. **`pclass` (`23.5%`):** Second strongest, capturing better lifeboat access for 1st/2nd-class cabins over 3rd class.
3. **`age` (`13.8%`):** Separates young children (higher survival) from older adults.
4. **`fare` (`8.2%`):** Captures residual socio-economic/cabin-location differences within classes.

---

## 5. Depth Sweep & Overfitting (`max_depth` 1 to 12)
- At **`max_depth=1`**, training (`77.9%`) and test (`78.3%`) accuracy match closely (underfitting, using only `sex`).
- Test accuracy peaks at **`max_depth=3` (`81.1%`)** and remains strong at **`max_depth=4` (`79.7%`, 5.0 pp train–test gap)**.
- Beyond depth 4, training accuracy climbs monotonically up to **`96.9%` at `max_depth=12`**, while test accuracy drops to **`77.6%`** (**19.2 pp gap**). Deep trees memorize small quirks and noise in the `571` training passengers rather than general survival patterns, hurting performance on unseen test passengers.

---

## 6. Honest Model Limitations
1. **Small & Noisy Sample (`n = 714`, Test `n = 143`):** Each single passenger changes test accuracy by `0.7` percentage points (`1 / 143`), so differences of 1–2% between depths (e.g., depth 3 vs. depth 4) partly reflect random split noise.
2. **Dropped Missing Ages (`177` Rows Removed):** Dropping rows with missing `age` discarded `19.9%` of the dataset (`891 -> 714`). If age missingness correlates with passenger class or survival (e.g., 3rd-class passengers), dropping rows introduces selection bias; training-fold median imputation or cross-validation would be stronger next steps.
3. **Limited Feature Set & Single-Tree Instability:** Only four features were used (`embarked`, `sibsp`, `parch`, and `deck` were excluded), and single decision trees have high variance and axis-aligned split boundaries compared with ensembles like Random Forests.
