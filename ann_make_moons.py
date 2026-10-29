
# %pip install -q numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
)

# 1. Create reproducible Make Moons data
RANDOM_STATE = 42
X, y = make_moons(n_samples=1200, noise=0.25, random_state=RANDOM_STATE)

# 2. Split into training and held-out test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)

# 3. Build and train the ANN
model = make_pipeline(
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

model.fit(X_train, y_train)

# 4. Predict on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Print test metrics
accuracy = accuracy_score(y_test, y_pred)

print("ANN — Make Moons binary classification")
print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")
print(f"Training iterations: {model.named_steps['mlpclassifier'].n_iter_}")

print("\nHeld-out test metrics")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro precision:   {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Macro recall:      {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"Macro F1 score:    {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification report")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Class 0", "Class 1"],
    digits=4,
))

if accuracy >= 0.85:
    print("Accuracy target passed: at least 85%.")
else:
    print("Accuracy is below 85%; consider adjusting the model or data settings.")

# 6. Plot decision boundary, confusion matrix, ROC curve, and training loss
fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
fig.suptitle("ANN Binary Classification — Make Moons", fontsize=16, weight="bold")

# Decision boundary
ax = axes[0, 0]
padding = 0.6
x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 350),
    np.linspace(y_min, y_max, 350),
)
grid = np.c_[xx.ravel(), yy.ravel()]
regions = model.predict(grid).reshape(xx.shape)

ax.contourf(
    xx, yy, regions,
    levels=[-0.5, 0.5, 1.5],
    alpha=0.25,
    colors=["#3478c9", "#e36a5c"],
)
ax.contour(xx, yy, regions, levels=[0.5], colors="#333333", linewidths=1.2)
ax.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train, cmap="coolwarm", s=24,
    edgecolors="white", linewidths=0.4, label="Training set",
)
ax.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_test, cmap="coolwarm", s=42, marker="o",
    edgecolors="#222222", linewidths=0.8, label="Test set",
)
ax.set_title("Learned decision boundary")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

# Confusion matrix
ax = axes[0, 1]
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Class 0", "Class 1"],
).plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
ax.set_title("Test-set confusion matrix")

# ROC curve
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
ax.plot(fpr, tpr, linewidth=2.5, label=f"ANN (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", label="No-skill baseline")
ax.set_title("ROC curve")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.2)
ax.legend(loc="lower right")

# Training loss
ax = axes[1, 1]
loss_curve = model.named_steps["mlpclassifier"].loss_curve_
ax.plot(range(1, len(loss_curve) + 1), loss_curve, color="#5b9b65", linewidth=2)
ax.set_title("Training loss")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Log-loss")
ax.grid(alpha=0.2)

plt.show()

# Optional: save the graph in the Colab session
fig.savefig("make_moons_ann_results.png", dpi=180, bbox_inches="tight")
