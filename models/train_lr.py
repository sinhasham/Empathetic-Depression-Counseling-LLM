# Train Logistic Regression model here
# models/train_lr.py

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

phq_cols = ["phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]

df = pd.read_csv("data/Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv")
df[phq_cols] = df[phq_cols].fillna(df[phq_cols].median())
df["PHQ_Total"] = df[phq_cols].sum(axis=1)

low  = df["PHQ_Total"].quantile(0.33)
high = df["PHQ_Total"].quantile(0.66)

def create_lmh(score):
    if score <= low:   return "Low"
    elif score <= high: return "Medium"
    else:              return "High"

df["LMH"] = df["PHQ_Total"].apply(create_lmh)

X = df[phq_cols]
y = df["LMH"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=5000, random_state=42)
lr.fit(X_scaled, y)

pickle.dump(lr,     open("models/lr_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl",   "wb"))

print("Model and scaler saved.")# models/train_lr.py

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

phq_cols = ["phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]

df = pd.read_csv("data/Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv")
df[phq_cols] = df[phq_cols].fillna(df[phq_cols].median())
df["PHQ_Total"] = df[phq_cols].sum(axis=1)

low  = df["PHQ_Total"].quantile(0.33)
high = df["PHQ_Total"].quantile(0.66)

def create_lmh(score):
    if score <= low:   return "Low"
    elif score <= high: return "Medium"
    else:              return "High"

df["LMH"] = df["PHQ_Total"].apply(create_lmh)

X = df[phq_cols]
y = df["LMH"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=5000, random_state=42)
lr.fit(X_scaled, y)

pickle.dump(lr,     open("models/lr_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl",   "wb"))

print("Model and scaler saved.")