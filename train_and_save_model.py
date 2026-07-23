import pickle
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

PHQ_PATH = "data/Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv"
MEX_PATH = "data/mexican_medical_students_mental_health_data.csv"

phq_cols = ["phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]

# ── LOAD ─────────────────────────────────────────────
phq_df = pd.read_csv(PHQ_PATH)
mex_df = pd.read_csv(MEX_PATH)

# ── PROF'S INSTRUCTION: remove missing rows, do NOT fill ──
phq_clean = phq_df.dropna(subset=phq_cols).copy()
mex_clean = mex_df.dropna(subset=phq_cols).copy()

print("="*60)
print("DATASET INFO (after removing missing rows)")
print("="*60)
print(f"PHQ Training  : {len(phq_df)} -> {len(phq_clean)} rows")
print(f"Mexican Test  : {len(mex_df)} -> {len(mex_clean)} rows")

# ── PROF'S INSTRUCTION: use only half of the numeric dataset ──
phq_half = phq_clean.sample(n=len(phq_clean)//2, random_state=42).copy()
print(f"Half PHQ Data : {len(phq_half)} rows used for training")

# ── LMH LABELS ───────────────────────────────────────
phq_half["PHQ_Total"]  = phq_half[phq_cols].sum(axis=1)
mex_clean["PHQ_Total"] = mex_clean[phq_cols].sum(axis=1)

low  = phq_half["PHQ_Total"].quantile(0.33)
high = phq_half["PHQ_Total"].quantile(0.66)

def create_lmh(score):
    if score <= low:    return "Low"
    elif score <= high: return "Medium"
    else:               return "High"

phq_half["LMH"]  = phq_half["PHQ_Total"].apply(create_lmh)
mex_clean["LMH"] = mex_clean["PHQ_Total"].apply(create_lmh)

print(f"\nTrain Labels:\n{phq_half['LMH'].value_counts()}")
print(f"\nTest Labels:\n{mex_clean['LMH'].value_counts()}")

# ── SCALE ─────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(phq_half[phq_cols])
X_test  = scaler.transform(mex_clean[phq_cols])
y_train = phq_half["LMH"]
y_test  = mex_clean["LMH"]

# ── 5 MODELS ──────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "SVM"                : SVC(kernel="linear", random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(random_state=42),
    "KNN"                : KNeighborsClassifier(n_neighbors=5),
}

print("\n" + "="*60)
print("NUMERIC MODEL COMPARISON (Half Data, No Missing)")
print("="*60)
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*62)

num_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    yp   = model.predict(X_test)
    acc  = round(accuracy_score(y_test, yp), 4)
    prec = round(precision_score(y_test, yp, average="weighted", zero_division=0), 4)
    rec  = round(recall_score(y_test, yp, average="weighted", zero_division=0), 4)
    f1   = round(f1_score(y_test, yp, average="weighted", zero_division=0), 4)
    num_results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    print(f"{name:<22} {acc:>10} {prec:>10} {rec:>10} {f1:>10}")

best_num = max(num_results, key=lambda x: num_results[x]["F1"])
print(f"\nBest Model: {best_num}")
print("\nClassification Report:")
print(classification_report(y_test, models[best_num].predict(X_test), zero_division=0))

print("="*60)
print("5-FOLD CROSS VALIDATION (Numeric)")
print("="*60)
cv_num = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
    cv_num[name] = {"mean": round(cv.mean(), 4), "std": round(cv.std(), 4)}
    print(f"{name:<22} Mean F1: {cv_num[name]['mean']} +/- {cv_num[name]['std']}")

# =====================================================
# GRAPHS
# =====================================================
model_names = list(num_results.keys())
short_names = ["LR", "SVM", "RF", "DT", "KNN"]
colors      = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

fig = plt.figure(figsize=(16, 20))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
accs = [num_results[m]["Accuracy"] for m in model_names]
bars = ax1.bar(short_names, accs, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_title("Numeric Model — Accuracy\n(Half Data, No Missing)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Accuracy"); ax1.set_ylim(0, 1.05)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
f1s = [num_results[m]["F1"] for m in model_names]
bars2 = ax2.bar(short_names, f1s, color=colors, edgecolor="black", linewidth=0.5)
ax2.set_title("Numeric Model — F1 Score\n(Half Data, No Missing)", fontsize=12, fontweight="bold")
ax2.set_ylabel("F1 Score"); ax2.set_ylim(0, 1.05)
for bar, val in zip(bars2, f1s):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

ax3 = fig.add_subplot(gs[1, 0])
precs = [num_results[m]["Precision"] for m in model_names]
recs  = [num_results[m]["Recall"] for m in model_names]
x = np.arange(len(model_names)); w = 0.35
ax3.bar(x - w/2, precs, w, label="Precision", color="#3F51B5", edgecolor="black", linewidth=0.5)
ax3.bar(x + w/2, recs, w, label="Recall", color="#FF5722", edgecolor="black", linewidth=0.5)
ax3.set_xticks(x); ax3.set_xticklabels(short_names)
ax3.set_title("Numeric Model — Precision vs Recall", fontsize=12, fontweight="bold")
ax3.set_ylim(0, 1.05); ax3.legend()

ax4 = fig.add_subplot(gs[1, 1])
cv_means = [cv_num[m]["mean"] for m in model_names]
cv_stds  = [cv_num[m]["std"]  for m in model_names]
ax4.bar(short_names, cv_means, yerr=cv_stds, color=colors, edgecolor="black", linewidth=0.5, capsize=5)
ax4.set_title("5-Fold CV — Numeric Model\n(Mean F1 +/- Std Dev)", fontsize=12, fontweight="bold")
ax4.set_ylabel("Mean F1"); ax4.set_ylim(0, 1.1)
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax4.text(i, m+s+0.01, f"{m:.4f}", ha="center", fontsize=9)

ax5 = fig.add_subplot(gs[2, 0])
cm_num = confusion_matrix(y_test, models[best_num].predict(X_test), labels=["Low", "Medium", "High"])
im = ax5.imshow(cm_num, interpolation="nearest", cmap="Blues")
ax5.set_title(f"Confusion Matrix — {best_num}\n(Numeric Model)", fontsize=12, fontweight="bold")
ax5.set_xticks([0,1,2]); ax5.set_yticks([0,1,2])
ax5.set_xticklabels(["Low","Medium","High"]); ax5.set_yticklabels(["Low","Medium","High"])
ax5.set_xlabel("Predicted"); ax5.set_ylabel("Actual")
for i in range(3):
    for j in range(3):
        ax5.text(j, i, str(cm_num[i,j]), ha="center", va="center",
                 color="white" if cm_num[i,j] > cm_num.max()/2 else "black", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax5)

ax6 = fig.add_subplot(gs[2, 1])
train_counts = phq_half["LMH"].value_counts().reindex(["Low","Medium","High"])
test_counts  = mex_clean["LMH"].value_counts().reindex(["Low","Medium","High"])
x2 = np.arange(3)
ax6.bar(x2 - w/2, train_counts.values, w, label="Train (PHQ, half)", color="#009688", edgecolor="black", linewidth=0.5)
ax6.bar(x2 + w/2, test_counts.values, w, label="Test (Mexican)", color="#795548", edgecolor="black", linewidth=0.5)
ax6.set_xticks(x2); ax6.set_xticklabels(["Low","Medium","High"])
ax6.set_title("Class Distribution — Train vs Test", fontsize=12, fontweight="bold")
ax6.legend()

fig.suptitle("Numeric PHQ-9 Model Performance (Half Data, No Missing)\n"
             f"Train: {len(phq_half)} rows | Test: {len(mex_clean)} rows",
             fontsize=14, fontweight="bold", y=1.01)

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/numeric_model_graphs.png", dpi=150, bbox_inches="tight")
print("\nGraphs saved as: outputs/numeric_model_graphs.png")

# =====================================================
# SAVE PICKLES FOR app.py (Logistic Regression + scaler)
# =====================================================
os.makedirs("models", exist_ok=True)
with open("models/lr_model.pkl", "wb") as f:
    pickle.dump(models["Logistic Regression"], f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved: models/lr_model.pkl")
print("Saved: models/scaler.pkl")
print("\nDONE.")