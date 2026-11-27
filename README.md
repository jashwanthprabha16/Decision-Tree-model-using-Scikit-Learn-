# Project 04 — First Machine Learning Classifier (Titanic Survival)

A complete supervised machine learning classification workflow using `scikit-learn` (`DecisionTreeClassifier`) on the Seaborn `titanic` dataset, benchmarked against a majority-class baseline, evaluated on unseen test data, and analyzed for overfitting across tree depths `1` to `12`.

---

## Project Structure

```text
project-04-titanic-classifier/
├── data/
│   └── titanic.csv                           # Local copy of sns.load_dataset("titanic") (891 rows)
├── outputs/
│   ├── 01_confusion_matrix.png               # Annotated 2x2 test confusion matrix
│   ├── 02_depth_overfitting_curve.png        # Train vs. Test accuracy across max_depth 1..12
│   ├── 03_feature_importances.png            # Decision Tree feature importances
│   └── 04_decision_tree_structure.png        # Visualized depth-4 decision tree rules
├── RESULTS.md                                # Concise write-up of results, metrics & limitations
├── titanic_classifier.ipynb                  # Executed Jupyter notebook with outputs & plots
├── titanic_classifier.py                     # Reproducible training & evaluation script
└── README.md                                 # Project overview and checklist
```

---

## Key Diagnostics

### 1. Test Confusion Matrix (`max_depth=4`, Accuracy = 79.7%)
![Confusion Matrix](outputs/01_confusion_matrix.png)

### 2. Overfitting Curve (`max_depth` 1 to 12)
![Overfitting Curve](outputs/02_depth_overfitting_curve.png)

### 3. Feature Importances (`sex` = 54.5%, `pclass` = 23.5%)
![Feature Importances](outputs/03_feature_importances.png)

### 4. Decision Tree Rules (`max_depth=4`)
![Decision Tree Structure](outputs/04_decision_tree_structure.png)

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python titanic_classifier.py
```

Open [titanic_classifier.ipynb](titanic_classifier.ipynb) for the step-by-step interactive notebook and [RESULTS.md](RESULTS.md) for the standalone results & limitations summary.
