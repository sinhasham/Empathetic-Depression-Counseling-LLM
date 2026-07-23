"""
fill_healthy_baseline.py
Adds more Low severity (Healthy Baseline) samples to the existing dataset.
Run this after generate_depression_dataset.py when Low class is underrepresented.
"""

import httpx
from groq import Groq
import pandas as pd
import json
import time
import os

GROQ_API_KEY = "api_key = "gsk_your_actual_key""

client = Groq(
    api_key=GROQ_API_KEY,
    http_client=httpx.Client(verify=False)
)

CSV_PATH = "data/depression_types_dataset.csv"
TARGET_LOW_SAMPLES = 20

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling/staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself or like a failure",
    "Trouble concentrating",
    "Moving/speaking slowly, or being restless/fidgety",
    "Thoughts of self-harm or not wanting to be alive"
]

def generate_healthy_sample():
    prompt = f"""You are simulating a mentally healthy person with NO depression.
This person may have occasional mild stress (normal life stress) but NO clinical depression.

For each of the 9 PHQ-9 symptom areas below, provide:
- A short, natural first-person sentence (1 sentence) showing this person is doing fine
- A severity score from 0-3 — for a healthy person, scores should be mostly 0 and occasionally 1
  (scores of 2 or 3 would indicate depression, which this person does NOT have)

Symptom areas:
{json.dumps(PHQ9_QUESTIONS, indent=2)}

Respond ONLY with valid JSON, no extra text:
{{
  "responses": [
    {{"symptom_num": 1, "patient_sentence": "...", "score": 0}},
    {{"symptom_num": 2, "patient_sentence": "...", "score": 0}},
    {{"symptom_num": 3, "patient_sentence": "...", "score": 0}},
    {{"symptom_num": 4, "patient_sentence": "...", "score": 0}},
    {{"symptom_num": 5, "patient_sentence": "...", "score": 0}},
    {{"symptom_num": 6, "patient_sentence": "...", "score": 0}},
    {{"symptom_num": 7, "patient_sentence": "...", "score": 0}},
    {{"symptom_num": 8, "patient_sentence": "...", "score": 0}},
    {{"symptom_num": 9, "patient_sentence": "...", "score": 0}}
  ]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def main():
    # Load existing dataset
    df = pd.read_csv(CSV_PATH)
    existing_low = len(df[df["severity"] == "Low"]) if "severity" in df.columns else 0

    # Calculate severity if not already there
    if "severity" not in df.columns:
        def get_severity(total):
            if total <= 4:   return "Low"
            elif total <= 14: return "Medium"
            else:             return "High"
        df["phq_total"] = df[["phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]].sum(axis=1)
        df["severity"]  = df["phq_total"].apply(get_severity)
        existing_low = len(df[df["severity"] == "Low"])

    print(f"Existing dataset: {len(df)} rows")
    print(f"Current Low severity samples: {existing_low}")
    needed = TARGET_LOW_SAMPLES - existing_low
    print(f"Need to generate: {needed} more Low samples\n")

    if needed <= 0:
        print("Already have enough Low samples!")
        return

    new_rows = []
    for i in range(needed):
        try:
            data = generate_healthy_sample()
            scores   = [r["score"] for r in data["responses"]]
            sentences = [r["patient_sentence"] for r in data["responses"]]

            row = {
                "depression_type": "No Significant Depression (Healthy Baseline)",
                "phq1": scores[0], "phq2": scores[1], "phq3": scores[2],
                "phq4": scores[3], "phq5": scores[4], "phq6": scores[5],
                "phq7": scores[6], "phq8": scores[7], "phq9": scores[8],
                "patient_text": " ".join(sentences),
                "phq_total": sum(scores),
                "severity": "Low"
            }
            new_rows.append(row)
            print(f"Sample {i+1}/{needed} done")

        except Exception as e:
            print(f"Sample {i+1} failed: {e}, skipping")
            continue

        time.sleep(0.5)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        final_df = pd.concat([df, new_df], ignore_index=True)
        final_df.to_csv(CSV_PATH, index=False)
        print(f"\nDataset updated: {len(final_df)} total rows")
        print(f"Low samples now: {len(final_df[final_df['severity'] == 'Low'])}")
    else:
        print("No new samples generated.")

if __name__ == "__main__":
    main()