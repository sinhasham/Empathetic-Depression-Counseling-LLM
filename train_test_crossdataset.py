"""
train_numeric_model.py

Numeric PHQ-9 severity classification — trains on the half PHQ-9 dataset
(no missing values) and tests on a separate dataset (labelled "Mexican"
in the class-distribution plot) to check generalization.

Models compared: Logistic Regression, SVM, Random Forest, Decision Tree, KNN
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ── PATHS — adjust these to your actual files ───────────────────
TRAIN_PATH = "data/phq9_numeric_dataset.csv"   # main PHQ-9 numeric dataset (half, no missing)
TEST_PATH  = "data/phq9_mexican_dataset.csv"   # separate test dataset ("Mexican")

FEATURE_COLS = [f"q{i}" for i in range(1, 10)]  # q1..q9 PHQ-9 item scores
TARGET_COL = "severity"

# ── LOAD ─────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH).dropna()
test_df  = pd.read_csv(TEST_PATH).dropna()

print(f"Train (PHQ, half): {len(train_df)} rows")
print(train_df[TARGET_COL].value_counts())
print(f"\nTest (Mexican): {len(test_df)} rows")
print(test_df[TARGET_COL].value_counts())

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET_COL]

# ── SCALE ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 5 MODELS ──────────────────────────────────────────────────────
models = {
    "LR" : LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42),
    "RF" : RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42),
    "DT" : DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=9),
}

print("\n" + "="*60)
print("NUMERIC MODEL COMPARISON (PHQ-9, Half Data, No Missing)")
print("="*60)
print(f"{'Model':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*52)

results = {}
fitted_models = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    fitted_models[name] = model
    yp   = model.predict(X_test_s)
    acc  = round(accuracy_score(y_test, yp), 4)
    prec = round(precision_score(y_test, yp, average="weighted", zero_division=0), 4)
    rec  = round(recall_score(y_test, yp, average="weighted", zero_division=0), 4)
    f1   = round(f1_score(y_test, yp, average="weighted", zero_division=0), 4)
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    print(f"{name:<10} {acc:>10} {prec:>10} {rec:>10} {f1:>10}")

# LR predictions used for the confusion matrix
lr_pred = fitted_models["LR"].predict(X_test_s)

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, lr_pred, zero_division=0))

# ── 5-FOLD CV ─────────────────────────────────────────────────────
print("="*60)
print("5-FOLD CROSS VALIDATION (Numeric Model)")
print("="*60)
cv_results = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring="f1_weighted")
    cv_results[name] = {"mean": round(cv.mean(), 4), "std": round(cv.std(), 4)}
    print(f"{name:<10} Mean F1: {cv_results[name]['mean']} +/- {cv_results[name]['std']}")

# =====================================================
# GRAPHS
# =====================================================
model_names = list(results.keys())   # ["LR", "SVM", "RF", "DT", "KNN"]
colors = ["#1E9EE8", "#3EA639", "#F5A200", "#E8195F", "#9C27B0"]

fig = plt.figure(figsize=(16, 20))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

fig.suptitle(
    "Numeric PHQ-9 Model Performance (Half Data, No Missing)\n"
    f"Train: {len(train_df)} rows | Test: {len(test_df)} rows",
    fontsize=16, fontweight="bold", y=0.995
)

# ax1: Accuracy
ax1 = fig.add_subplot(gs[0, 0])
accs = [results[m]["Accuracy"] for m in model_names]
bars = ax1.bar(model_names, accs, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_title("Numeric Model — Accuracy\n(Half Data, No Missing)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Accuracy"); ax1.set_ylim(0, 1.0)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

# ax2: F1 Score
ax2 = fig.add_subplot(gs[0, 1])
f1s = [results[m]["F1"] for m in model_names]
bars2 = ax2.bar(model_names, f1s, color=colors, edgecolor="black", linewidth=0.5)
ax2.set_title("Numeric Model — F1 Score\n(Half Data, No Missing)", fontsize=12, fontweight="bold")
ax2.set_ylabel("F1 Score"); ax2.set_ylim(0, 1.0)
for bar, val in zip(bars2, f1s):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

# ax3: Precision vs Recall
ax3 = fig.add_subplot(gs[1, 0])
precs = [results[m]["Precision"] for m in model_names]
recs  = [results[m]["Recall"] for m in model_names]
x = np.arange(len(model_names)); w = 0.35
ax3.bar(x - w/2, precs, w, label="Precision", color="#3F51B5", edgecolor="black", linewidth=0.5)
ax3.bar(x + w/2, recs, w, label="Recall", color="#FF5722", edgecolor="black", linewidth=0.5)
ax3.set_xticks(x); ax3.set_xticklabels(model_names)
ax3.set_title("Numeric Model — Precision vs Recall", fontsize=12, fontweight="bold")
ax3.set_ylim(0, 1.0); ax3.legend()

# ax4: 5-Fold CV
ax4 = fig.add_subplot(gs[1, 1])
cv_means = [cv_results[m]["mean"] for m in model_names]
cv_stds  = [cv_results[m]["std"]  for m in model_names]
ax4.bar(model_names, cv_means, yerr=cv_stds, color=colors, edgecolor="black", linewidth=0.5, capsize=5)
ax4.set_title("5-Fold CV — Numeric Model\n(Mean F1 +/- Std Dev)", fontsize=12, fontweight="bold")
ax4.set_ylabel("Mean F1"); ax4.set_ylim(0, 1.0)
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax4.text(i, m+s+0.01, f"{m:.4f}", ha="center", fontsize=9)

# ax5: Confusion Matrix (Logistic Regression)
ax5 = fig.add_subplot(gs[2, 0])
cm = confusion_matrix(y_test, lr_pred, labels=["Low", "Medium", "High"])
im = ax5.imshow(cm, interpolation="nearest", cmap="Blues")
ax5.set_title("Confusion Matrix — Logistic Regression\n(Numeric Model)", fontsize=12, fontweight="bold")
ax5.set_xticks([0,1,2]); ax5.set_yticks([0,1,2])
ax5.set_xticklabels(["Low","Medium","High"]); ax5.set_yticklabels(["Low","Medium","High"])
ax5.set_xlabel("Predicted"); ax5.set_ylabel("Actual")
for i in range(3):
    for j in range(3):
        ax5.text(j, i, str(cm[i,j]), ha="center", va="center",
                 color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax5)

# ax6: Class Distribution — Train vs Test
ax6 = fig.add_subplot(gs[2, 1])
classes = ["Low", "Medium", "High"]
train_counts = train_df[TARGET_COL].value_counts().reindex(classes).fillna(0)
test_counts  = test_df[TARGET_COL].value_counts().reindex(classes).fillna(0)
x6 = np.arange(len(classes)); w6 = 0.35
ax6.bar(x6 - w6/2, train_counts.values, w6, label="Train (PHQ, half)", color="#009688", edgecolor="black", linewidth=0.5)
ax6.bar(x6 + w6/2, test_counts.values, w6, label="Test (Mexican)", color="#795548", edgecolor="black", linewidth=0.5)
ax6.set_xticks(x6); ax6.set_xticklabels(classes)
ax6.set_title("Class Distribution — Train vs Test", fontsize=12, fontweight="bold")
ax6.legend()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/numeric_model_graphs.png", dpi=150, bbox_inches="tight")
print("\nGraphs saved as: outputs/numeric_model_graphs.png")

# ── SAVE BEST MODEL (Logistic Regression, per the confusion matrix) ─
os.makedirs("models", exist_ok=True)
with open("models/numeric_model.pkl", "wb") as f:
    pickle.dump(fitted_models["LR"], f)
with open("models/numeric_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved: models/numeric_model.pkl")
print("Saved: models/numeric_scaler.pkl")
print("\nDONE.")