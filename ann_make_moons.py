"""ANN binary-classification project for scikit-learn's make_moons dataset.

Run:
    python ann_make_moons.py

The script trains a small multilayer perceptron, reports held-out metrics, and
saves a four-panel evaluation figure beside this file (or at --plot-path).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def build_model() -> object:
    """Return a scaled ANN pipeline with reproducible training settings."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=0.005,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=RANDOM_STATE,
        ),
    )


def plot_results(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    plot_path: Path,
) -> None:
    """Draw decision boundary, confusion matrix, ROC curve, and loss curve."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("ANN Binary Classification — Make Moons", fontsize=17, weight="bold")

    # Decision boundary over the full feature space; test observations are rings.
    ax = axes[0, 0]
    padding = 0.6
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350))
    grid = np.c_[xx.ravel(), yy.ravel()]
    regions = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5], alpha=0.24,
                colors=["#3478c9", "#e36a5c"])
    ax.contour(xx, yy, regions, levels=[0.5], colors="#333333", linewidths=1.3)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm",
               edgecolors="white", linewidths=0.45, s=24, label="Training set")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm",
               edgecolors="#222222", linewidths=0.9, s=44, marker="o", label="Test set")
    ax.set(title="Learned decision boundary", xlabel="Feature 1", ylabel="Feature 2")
    ax.legend(loc="upper right", frameon=True)

    ax = axes[0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"]).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format="d"
    )
    ax.set_title("Test-set confusion matrix")

    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    ax.plot(fpr, tpr, color="#3478c9", linewidth=2.5, label=f"ANN (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="No-skill baseline")
    ax.set(title="Receiver operating characteristic", xlabel="False positive rate",
           ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    mlp = model.named_steps["mlpclassifier"]
    ax.plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, color="#5b9b65", lw=2)
    ax.set(title="Training loss", xlabel="Training iteration", ylabel="Log-loss")
    ax.grid(alpha=0.2)

    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Jupyter/IPython cells do not define __file__; save beside the notebook
    # (current working directory) in that case, or beside this script otherwise.
    project_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-path", type=Path,
                        default=project_dir / "make_moons_ann_results.png",
                        help="Where to save the evaluation graph.")
    # Ignore notebook/kernel arguments (for example, Jupyter's injected -f).
    args, _unknown_args = parser.parse_known_args()

    # A moderately noisy, balanced, nonlinear binary dataset.
    X, y = make_moons(n_samples=1200, noise=0.25, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        "Accuracy": accuracy,
        "Precision (macro)": precision_score(y_test, y_pred, average="macro"),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro"),
        "F1 score (macro)": f1_score(y_test, y_pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_test, y_score),
    }
    print("ANN — Make Moons binary classification")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Training iterations: {model.named_steps['mlpclassifier'].n_iter_}")
    print("\nHeld-out test metrics")
    for name, value in metrics.items():
        print(f"{name:20s}: {value:.4f}")
    print("\nClassification report")
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"], digits=4))

    # The requested target is checked against untouched test data.
    if accuracy < 0.85:
        raise RuntimeError(f"Test accuracy {accuracy:.1%} did not meet the 85% target.")
    print("Accuracy target: PASSED (at least 85% on the held-out test set).")

    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(model, X, y, X_train, y_train, X_test, y_test, y_pred, y_score, args.plot_path)
    print(f"\nEvaluation graph saved to: {args.plot_path.resolve()}")


if __name__ == "__main__":
    main()
