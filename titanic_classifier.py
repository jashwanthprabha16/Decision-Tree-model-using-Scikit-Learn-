"""Project 04: First Machine Learning Classifier — InternCircle Data Science & AI.

Trains a DecisionTreeClassifier on the Seaborn `titanic` dataset to predict
passenger survival from passenger class, sex, age, and fare; compares against a
majority-class baseline; evaluates unseen test performance (Accuracy, Confusion
Matrix, Precision/Recall/F1, and MSE); tests max_depth 1..12 for overfitting;
and saves crisp 150 dpi diagnostic charts.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "titanic.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

FEATURE_COLS = ["pclass", "sex", "age", "fare"]
TARGET_COL = "survived"
RANDOM_STATE = 42

BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
SKY = "#56B4E9"
SLATE = "#1F2937"


def configure_style() -> None:
    """Configure consistent, colorblind-safe chart formatting."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "text.parse_math": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CCCCCC",
            "axes.titleweight": "bold",
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "grid.color": "#EAEAEA",
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#DDDDDD",
        }
    )


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load Titanic data, select target/features, drop missing ages, encode sex."""
    if DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
    else:
        raw = sns.load_dataset("titanic")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(DATA_PATH, index=False)

    selected = raw[[TARGET_COL] + FEATURE_COLS].copy()
    clean = selected.dropna().reset_index(drop=True)
    # Encode sex numerically: female -> 1, male -> 0
    clean["sex"] = (clean["sex"] == "female").astype(int)

    X = clean[FEATURE_COLS]
    y = clean[TARGET_COL].astype(int)
    return raw, clean, y, X


def plot_confusion_matrix_chart(cm: np.ndarray, acc: float, out_path: Path) -> None:
    """Plot annotated 2x2 confusion matrix with plain-English cell labels."""
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    cell_names = np.array(
        [
            ["True Negative (TN)\nCorrectly predicted Did Not Survive", "False Positive (FP)\nPredicted Survived, actually Did Not"],
            ["False Negative (FN)\nPredicted Did Not Survive, actually Survived", "True Positive (TP)\nCorrectly predicted Survived"],
        ]
    )
    total = cm.sum()
    annot = np.empty_like(cm, dtype=object)
    for r in range(2):
        for c in range(2):
            short_tag = ["TN", "FP", "FN", "TP"][r * 2 + c]
            pct = cm[r, c] / total * 100
            annot[r, c] = f"{short_tag}: {cm[r, c]}\n({pct:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1.2,
        linecolor="white",
        square=True,
        annot_kws={"size": 12, "weight": "bold"},
        xticklabels=["Predicted: Did Not Survive (0)", "Predicted: Survived (1)"],
        yticklabels=["Actual: Did Not Survive (0)", "Actual: Survived (1)"],
        ax=ax,
    )
    ax.set_title(f"Test Confusion Matrix (max_depth=4 | Accuracy = {acc:.1%})", fontsize=12, pad=12)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_depth_overfitting_curve(depth_df: pd.DataFrame, chosen_depth: int, out_path: Path) -> None:
    """Plot training vs. test accuracy across max_depth 1..12 to illustrate overfitting."""
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(
        depth_df["max_depth"],
        depth_df["train_accuracy"] * 100,
        marker="o",
        color=ORANGE,
        linewidth=2.3,
        label="Training Accuracy",
    )
    ax.plot(
        depth_df["max_depth"],
        depth_df["test_accuracy"] * 100,
        marker="s",
        color=BLUE,
        linewidth=2.3,
        label="Test Accuracy (Unseen Data)",
    )

    chosen_row = depth_df[depth_df["max_depth"] == chosen_depth].iloc[0]
    ax.axvline(chosen_depth, color="#555555", linestyle="--", linewidth=1.3)
    ax.annotate(
        f"Chosen depth = {chosen_depth}\nTest: {chosen_row['test_accuracy']:.1%} | Train: {chosen_row['train_accuracy']:.1%}",
        xy=(chosen_depth, chosen_row["test_accuracy"] * 100),
        xytext=(chosen_depth + 1.2, chosen_row["test_accuracy"] * 100 - 7.5),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.3},
        fontsize=9.2,
        fontweight="bold",
        color=SLATE,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": BLUE},
    )

    ax.set_xticks(depth_df["max_depth"].tolist())
    ax.set_ylim(65, 102)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
    ax.set(
        title="Training Accuracy Climbs to 95%+ While Test Accuracy Peaks Early (Overfitting Beyond Depth 4)",
        xlabel="Decision Tree max_depth",
        ylabel="Classification Accuracy (%)",
    )
    ax.legend(loc="lower right")
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importances(importances: pd.Series, out_path: Path) -> None:
    """Zero-anchored horizontal bar chart of Decision Tree feature importances."""
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    imp_sorted = importances.sort_values(ascending=True)
    pretty = {"sex": "Sex (female=1, male=0)", "pclass": "Passenger Class (1–3)", "age": "Age (Years)", "fare": "Ticket Fare"}
    labels = [pretty.get(k, k) for k in imp_sorted.index]
    vals = imp_sorted.values * 100

    colors = [SKY, SKY, ORANGE, BLUE]
    bars = ax.barh(labels, vals, color=colors, height=0.55, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9.5, fontweight="bold", color=SLATE)

    ax.set_xlim(0, max(vals) * 1.18)
    ax.axvline(0, color="#444444", linewidth=1.0)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
    ax.set(
        title=f"{labels[-1]} Dominates Survival Splits ({vals[-1]:.1f}% Importance)",
        xlabel="Feature Importance (%)",
        ylabel="Feature",
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tree_diagram(model: DecisionTreeClassifier, feature_names: list[str], out_path: Path) -> None:
    """Render the trained depth-4 decision tree rules cleanly."""
    fig, ax = plt.subplots(figsize=(22, 9.5))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No", "Survived"],
        filled=True,
        rounded=True,
        fontsize=8,
        proportion=False,
        impurity=False,
        ax=ax,
    )
    ax.set_title("Trained Decision Tree (max_depth=4) Survival Rules", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw, clean, y, X = load_and_prepare_data()
    print("=== DATA SUMMARY ===")
    print(f"Raw rows: {len(raw)} | Missing values in selected cols:\n{raw[[TARGET_COL] + FEATURE_COLS].isna().sum().to_dict()}")
    print(f"Clean rows after dropna(): {len(clean)} (dropped {len(raw) - len(clean)} rows missing age)")
    print(f"Overall survival rate (clean): {y.mean():.3f} ({y.sum()} survived, {(1 - y).sum()} did not survive)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain shape: {X_train.shape} (survival rate: {y_train.mean():.3f})")
    print(f"Test shape:  {X_test.shape} (survival rate: {y_test.mean():.3f})")

    # Majority-class baseline fit on training data only
    baseline_clf = DummyClassifier(strategy="most_frequent")
    baseline_clf.fit(X_train, y_train)
    baseline_pred = baseline_clf.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_mse = mean_squared_error(y_test, baseline_pred)

    # Decision Tree classifier (max_depth=4)
    model = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_mse_class = mean_squared_error(y_test, y_test_pred)
    test_brier_mse = mean_squared_error(y_test, y_test_prob)
    cm = confusion_matrix(y_test, y_test_pred)

    print("\n=== BASELINE VS DECISION TREE (max_depth=4) ===")
    print(f"Majority-class baseline test accuracy: {baseline_acc:.4f} ({baseline_acc:.1%}) | MSE: {baseline_mse:.4f}")
    print(f"Decision Tree train accuracy:          {train_acc:.4f} ({train_acc:.1%})")
    print(f"Decision Tree test accuracy:           {test_acc:.4f} ({test_acc:.1%}) (+{(test_acc - baseline_acc)*100:.1f} pp vs baseline)")
    print(f"Decision Tree test MSE (labels):       {test_mse_class:.4f} | Brier MSE (probabilities): {test_brier_mse:.4f}")
    print("\nConfusion Matrix (rows=Actual [0,1], cols=Predicted [0,1]):")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=["Did Not Survive", "Survived"], digits=3))

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Feature Importances:")
    print(importances.round(4).to_string())

    # Depth sweep 1..12
    depth_rows = []
    for depth in range(1, 13):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        depth_rows.append(
            {
                "max_depth": depth,
                "train_accuracy": accuracy_score(y_train, clf.predict(X_train)),
                "test_accuracy": accuracy_score(y_test, clf.predict(X_test)),
            }
        )
    depth_df = pd.DataFrame(depth_rows)
    depth_df["gap_pp"] = (depth_df["train_accuracy"] - depth_df["test_accuracy"]) * 100
    print("\n=== DEPTH SWEEP (max_depth 1 to 12) ===")
    print(depth_df.round({"train_accuracy": 4, "test_accuracy": 4, "gap_pp": 1}).to_string(index=False))

    plot_confusion_matrix_chart(cm, test_acc, OUTPUT_DIR / "01_confusion_matrix.png")
    plot_depth_overfitting_curve(depth_df, chosen_depth=4, out_path=OUTPUT_DIR / "02_depth_overfitting_curve.png")
    plot_feature_importances(importances, OUTPUT_DIR / "03_feature_importances.png")
    plot_tree_diagram(model, FEATURE_COLS, OUTPUT_DIR / "04_decision_tree_structure.png")
    print("\nSaved all 4 diagnostic charts to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
