"""
generate_depression_dataset.py

Generates a synthetic patient dataset across 7 clinically distinct
profiles -- 6 depression subtypes plus a healthy/low-severity baseline --
using an LLM (Groq / Llama 3.3 70B) acting as a simulated patient for
each PHQ-9 symptom area.

The healthy baseline category is included so the dataset spans all
three severity classes (Low/Medium/High), not just Medium/High.

If data/depression_types_dataset.csv already exists (e.g. from a
previous run), this script will only generate and append rows for
depression_type categories that are NOT already present -- it will
NOT regenerate or duplicate existing rows.

Output: data/depression_types_dataset.csv

IMPORTANT: Put your own Groq API key in GROQ_API_KEY below before running.
Never share your API key publicly (e.g. in chats, GitHub, screenshots).
"""

import httpx
from groq import Groq
import pandas as pd
import json
import time
import os

# =====================================================
# CONFIG — put your own key here
# =====================================================
GROQ_API_KEY = "gsk_your_actual_key"

SAMPLES_PER_TYPE = 20          # 7 types x 20 = 140 rows total
OUTPUT_PATH = "data/depression_types_dataset.csv"

client = Groq(
    api_key=GROQ_API_KEY,
    http_client=httpx.Client(verify=False)
)

DEPRESSION_TYPES = [
    "Major Depressive Disorder (MDD) - severe, persistent sadness and loss of interest",
    "Persistent Depressive Disorder (Dysthymia) - chronic low-grade depression lasting years",
    "Postpartum Depression - depression following childbirth",
    "Trauma-linked Depression - depression following a traumatic event (PTSD-related)",
    "Seasonal Affective Disorder - depression that worsens in specific seasons",
    "Mild/Situational Depression - low-level depression triggered by a specific life event",
    "No Significant Depression (Healthy Baseline) - this person is psychologically healthy "
    "overall. They may have an occasional bad day or mild, transient stress (e.g. before an "
    "exam or a busy week), but no persistent low mood, no loss of interest, normal sleep and "
    "appetite, normal energy, and no self-critical or hopeless thinking. Almost all symptom "
    "scores should be 0, with at most one or two symptoms scoring 1 due to a recent minor "
    "stressor."
]

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


def generate_patient_sample(depression_type):
    prompt = f"""You are simulating a person described as follows:
{depression_type}

For each of the 9 PHQ-9 symptom areas below, provide:
- A short, natural first-person sentence describing how this person would
  describe that symptom (1 sentence, realistic, not exaggerated)
- A severity score 0-3 (0=not at all, 1=several days, 2=more than half the days,
  3=nearly every day) that is clinically appropriate for this specific profile

Symptom areas:
{json.dumps(PHQ9_QUESTIONS, indent=2)}

Respond ONLY with valid JSON, no extra text, no markdown fences:
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

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    [DEBUG] JSON parse failed. Raw response was:\n{raw[:500]}\n")
        raise e


def build_dataset(samples_per_type=SAMPLES_PER_TYPE, output_path=OUTPUT_PATH):
    existing_df = None
    types_to_generate = DEPRESSION_TYPES

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_types = set(existing_df["depression_type"].unique())
        types_to_generate = [
            dtype for dtype in DEPRESSION_TYPES
            if dtype.split(" - ")[0] not in existing_types
        ]
        if not types_to_generate:
            print("All depression types are already present in the dataset. Nothing to do.")
            return existing_df
        print(f"Existing dataset found with {len(existing_df)} rows.")
        print(f"Will only generate new samples for: "
              f"{[t.split(' - ')[0] for t in types_to_generate]}")

    rows = []

    for dtype in types_to_generate:
        short_name = dtype.split(" - ")[0]
        print(f"\nGenerating samples for: {short_name}")

        for i in range(samples_per_type):
            try:
                data = generate_patient_sample(dtype)
                scores = [r["score"] for r in data["responses"]]
                sentences = [r["patient_sentence"] for r in data["responses"]]

                row = {
                    "depression_type": short_name,
                    "phq1": scores[0], "phq2": scores[1], "phq3": scores[2],
                    "phq4": scores[3], "phq5": scores[4], "phq6": scores[5],
                    "phq7": scores[6], "phq8": scores[7], "phq9": scores[8],
                    "patient_text": " ".join(sentences)
                }
                rows.append(row)
                print(f"  Sample {i+1}/{samples_per_type} done")

            except Exception as e:
                print(f"  Sample {i+1} failed: {type(e).__name__}: {e}, skipping")
                continue

            time.sleep(0.5)

    new_df = pd.DataFrame(rows)

    if new_df.empty:
        print("\nERROR: No new samples were successfully generated. "
              "Check the [DEBUG] messages above for the root cause "
              "(invalid API key, network/SSL issue, rate limit, or malformed model output).")
        return existing_df if existing_df is not None else new_df

    new_df["phq_total"] = new_df[
        ["phq1", "phq2", "phq3", "phq4", "phq5", "phq6", "phq7", "phq8", "phq9"]
    ].sum(axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(output_path, index=False)
    print(f"\nDataset saved: {output_path} ({len(combined)} total rows)")
    print(f"\ndepression_type counts:\n{combined['depression_type'].value_counts()}")
    return combined


if __name__ == "__main__":
    build_dataset()