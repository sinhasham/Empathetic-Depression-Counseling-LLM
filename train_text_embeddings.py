"""
Text severity classification using sentence embeddings (instead of TF-IDF).
Run this LOCALLY (needs internet on first run to download the embedding model).

Install first:
    pip install sentence-transformers scikit-learn pandas matplotlib numpy
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

TEXT_PATH = "data/depression_dataset_large.csv"

# ── LOAD ─────────────────────────────────────────────
df = pd.read_csv(TEXT_PATH)
print(f"Text dataset: {len(df)} rows")
print(df['severity'].value_counts())

X_text = df['patient_text'].astype(str).tolist()
y = df['severity']

# ── EMBEDDINGS ────────────────────────────────────────
# 'all-MiniLM-L6-v2' is small, fast, and strong for short-text semantic similarity.
print("\nLoading sentence embedding model (downloads once, then cached)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding patient_text into embeddings...")
X_embed = embedder.encode(X_text, show_progress_bar=True, batch_size=32)
print(f"Embedding shape: {X_embed.shape}")   # (1600, 384)

# ── SPLIT ─────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_embed, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# ── 5 MODELS ──────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "SVM"                : SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "KNN"                : KNeighborsClassifier(n_neighbors=9),
}

print("\n" + "="*60)
print("TEXT MODEL COMPARISON (Sentence Embeddings, 1600 samples)")
print("="*60)
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*62)

results = {}
for name, model in models.items():
    model.fit(X_tr_s, y_tr)
    yp   = model.predict(X_te_s)
    acc  = round(accuracy_score(y_te, yp), 4)
    prec = round(precision_score(y_te, yp, average="weighted", zero_division=0), 4)
    rec  = round(recall_score(y_te, yp, average="weighted", zero_division=0), 4)
    f1   = round(f1_score(y_te, yp, average="weighted", zero_division=0), 4)
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    print(f"{name:<22} {acc:>10} {prec:>10} {rec:>10} {f1:>10}")

best_name = max(results, key=lambda x: results[x]["F1"])
print(f"\nBest single model: {best_name}")

# ── VOTING ENSEMBLE (often adds a few more points) ────
print("\n--- Voting Ensemble (LR + SVM + RF, soft voting) ---")
ens = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42)),
    ],
    voting="soft"
)
ens.fit(X_tr_s, y_tr)
yp_ens = ens.predict(X_te_s)
ens_acc = round(accuracy_score(y_te, yp_ens), 4)
ens_f1  = round(f1_score(y_te, yp_ens, average="weighted"), 4)
print(f"Ensemble Accuracy: {ens_acc}  F1: {ens_f1}")

if ens_f1 > results[best_name]["F1"]:
    final_model = ens
    final_name  = "Voting Ensemble"
    final_pred  = yp_ens
    results["Voting Ensemble"] = {
        "Accuracy": ens_acc,
        "Precision": round(precision_score(y_te, yp_ens, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_te, yp_ens, average="weighted", zero_division=0), 4),
        "F1": ens_f1,
    }
else:
    final_model = models[best_name]
    final_name  = best_name
    final_pred  = final_model.predict(X_te_s)

print(f"\nFINAL BEST MODEL: {final_name}")
print("\nClassification Report:")
print(classification_report(y_te, final_pred, zero_division=0))

# ── 5-FOLD CV on the final model ─────────────────────
print("="*60)
print("5-FOLD CROSS VALIDATION (Embeddings)")
print("="*60)
cv_results = {}
for name, model in models.items():
    cv = cross_val_score(model, X_tr_s, y_tr, cv=5, scoring="f1_weighted")
    cv_results[name] = {"mean": round(cv.mean(), 4), "std": round(cv.std(), 4)}
    print(f"{name:<22} Mean F1: {cv_results[name]['mean']} +/- {cv_results[name]['std']}")

# =====================================================
# GRAPHS
# =====================================================
model_names = list(results.keys())
name_map = {
    "Logistic Regression": "LR",
    "SVM": "SVM",
    "Random Forest": "RF",
    "Decision Tree": "DT",
    "KNN": "KNN",
    "Voting Ensemble": "Voting",
}
short_names = [name_map.get(n, n) for n in model_names]
colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

fig = plt.figure(figsize=(16, 20))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
accs = [results[m]["Accuracy"] for m in model_names]
bars = ax1.bar(short_names, accs, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_title("Text Model — Accuracy\n(Sentence Embeddings)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Accuracy"); ax1.set_ylim(0, 1.0)
ax1.set_xlabel("Models")
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
f1s = [results[m]["F1"] for m in model_names]
bars2 = ax2.bar(short_names, f1s, color=colors, edgecolor="black", linewidth=0.5)
ax2.set_title("Text Model — F1 Score\n(Sentence Embeddings)", fontsize=12, fontweight="bold")
ax2.set_ylabel("F1 Score"); ax2.set_ylim(0, 1.0)
ax2.set_xlabel("Models")
for bar, val in zip(bars2, f1s):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

ax3 = fig.add_subplot(gs[1, 0])
precs = [results[m]["Precision"] for m in model_names]
recs  = [results[m]["Recall"] for m in model_names]
x = np.arange(len(model_names)); w = 0.35
ax3.bar(x - w/2, precs, w, label="Precision", color="#3F51B5", edgecolor="black", linewidth=0.5)
ax3.bar(x + w/2, recs, w, label="Recall", color="#FF5722", edgecolor="black", linewidth=0.5)
ax3.set_xticks(x); ax3.set_xticklabels(short_names)
ax3.set_title("Text Model — Precision vs Recall", fontsize=12, fontweight="bold")
ax3.set_ylim(0, 1.0); ax3.legend()
ax3.set_xlabel("Models")
ax3.set_ylabel("Score")

ax4 = fig.add_subplot(gs[1, 1])
cv_names = list(cv_results.keys())
cv_short = [name_map.get(n, n) for n in cv_names]
cv_means = [cv_results[m]["mean"] for m in cv_names]
cv_stds  = [cv_results[m]["std"]  for m in cv_names]
ax4.bar(cv_short, cv_means, yerr=cv_stds, color=colors[:len(cv_names)], edgecolor="black", linewidth=0.5, capsize=5)
ax4.set_title("5-Fold CV — Text Model\n(Mean F1 +/- Std Dev)", fontsize=12, fontweight="bold")
ax4.set_ylabel("Mean F1"); ax4.set_ylim(0, 1.0)
ax4.set_xlabel("Models")
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax4.text(i, m+s+0.01, f"{m:.4f}", ha="center", fontsize=9)

ax5 = fig.add_subplot(gs[2, 0])
cm = confusion_matrix(y_te, final_pred, labels=["Low", "Medium", "High"])
im = ax5.imshow(cm, interpolation="nearest", cmap="Greens")
ax5.set_title(f"Confusion Matrix — {final_name}\n(Embeddings)", fontsize=12, fontweight="bold")
ax5.set_xticks([0,1,2]); ax5.set_yticks([0,1,2])
ax5.set_xticklabels(["Low","Medium","High"]); ax5.set_yticklabels(["Low","Medium","High"])
ax5.set_xlabel("Predicted"); ax5.set_ylabel("Actual")
for i in range(3):
    for j in range(3):
        ax5.text(j, i, str(cm[i,j]), ha="center", va="center",
                 color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax5)

ax6 = fig.add_subplot(gs[2, 1])
full_counts = df["severity"].value_counts().reindex(["Low","Medium","High"])
ax6.bar(["Low","Medium","High"], full_counts.values, color=["#8BC34A","#FFC107","#F44336"], edgecolor="black", linewidth=0.5)
ax6.set_title("Text Dataset — Class Distribution\n(1600 samples)", fontsize=12, fontweight="bold")
ax6.set_ylabel("Count")
ax6.set_xlabel("Severity Class")

fig.suptitle(f"Text Severity Classification — Sentence Embeddings\nBest: {final_name} | Test Acc: {results.get(final_name, {}).get('Accuracy', ens_acc)}",
             fontsize=14, fontweight="bold", y=1.01)

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/text_embedding_graphs.png", dpi=150, bbox_inches="tight")
print("\nGraphs saved as: outputs/text_embedding_graphs.png")

# ── SAVE MODEL ────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/text_embedding_model.pkl", "wb") as f:
    pickle.dump(final_model, f)
with open("models/text_embedding_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved: models/text_embedding_model.pkl")
print("Saved: models/text_embedding_scaler.pkl")
print("\nNOTE: to use this model later, re-encode new text with the SAME")
print("SentenceTransformer('all-MiniLM-L6-v2') before scaling + predicting.")
print("\nDONE.")