"""
train_lr_text.py

Trains the severity classifier PURELY on patient text (TF-IDF),
NOT on numeric PHQ-9 scores. The numeric phq1..phq9 columns in the
dataset are used ONLY to derive the ground-truth severity label
(Low/Medium/High) -- they are never given to the model as features.

This mirrors a real deployment scenario: the user only ever types
free text, never numeric scores, so the model should learn directly
from text.

Input : data/depression_types_dataset.csv
Output: models/lr_text_model.pkl
        models/tfidf_vectorizer.pkl
"""

import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

DATA_PATH = "data/depression_types_dataset.csv"
MODEL_DIR = "models"

PHQ_COLS = ["phq1", "phq2", "phq3", "phq4", "phq5", "phq6", "phq7", "phq8", "phq9"]


# =====================================================
# STEP 1 — Load data
# =====================================================
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows")
print(f"Depression types: {df['depression_type'].unique().tolist()}")

# =====================================================
# STEP 2 — Derive severity label from PHQ totals
#           (numbers used ONLY here, never as model input)
# =====================================================
df["phq_total"] = df[PHQ_COLS].sum(axis=1)


def clinical_severity(total):
    # Standard PHQ-9 clinical cutoffs (DSM-5 aligned), not data-dependent
    # quantiles -- this avoids the circularity that caused overfitting
    # in the earlier numeric-only model.
    if total <= 4:
        return "Low"
    elif total <= 14:
        return "Medium"
    else:
        return "High"


df["severity"] = df["phq_total"].apply(clinical_severity)

print("\nSeverity label distribution:")
print(df["severity"].value_counts())

# =====================================================
# STEP 3 — Clean text
# =====================================================
df["patient_text"] = df["patient_text"].fillna("").str.lower()

# =====================================================
# STEP 4 — Train/test split (text-only features from here on)
# =====================================================
X = df["patient_text"]
y = df["severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# =====================================================
# STEP 5 — TF-IDF vectorization (text -> numeric features for LR,
#           but these features come from the text itself, not from
#           any numeric PHQ score)
# =====================================================
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =====================================================
# STEP 6 — Train Logistic Regression on text features
# =====================================================
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_vec, y_train)

# =====================================================
# STEP 7 — Evaluate
# =====================================================
y_pred = lr.predict(X_test_vec)

print("\n===== TEXT-ONLY MODEL RESULTS =====")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4))
print("F1 Score :", round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# =====================================================
# STEP 8 — Save model + vectorizer
# =====================================================
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "lr_text_model.pkl"), "wb") as f:
    pickle.dump(lr, f)

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\nSaved: {MODEL_DIR}/lr_text_model.pkl")
print(f"Saved: {MODEL_DIR}/tfidf_vectorizer.pkl")
