import pickle
from agents.router             import route_agent
from agents.memory              import add_user_message, add_assistant_message, get_history, clear_history
from agents.safety              import check_risk, get_crisis_response
from agents.state_machine       import update_stage, get_stage_name, get_stage_prompt
from agents.progress_tracker    import log_turn, get_summary
from agents.conversation_saver  import save_session
from agents.mood_refresher      import start_session, should_refresh, get_light_moment
from timeout_input               import get_input_with_timeout

from llm.emotion_reasoner       import analyze_emotion
from llm.cot                    import generate_cot
from llm.coe                    import generate_empathy
from llm.response_generator     import generate_response
from llm.phq_text_assessor      import run_text_phq_assessment
from rag.retriever              import retrieve

from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

lr     = pickle.load(open("models/lr_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl",   "rb"))

phq_cols = ["phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]



def show_model_accuracy():
    print("\n" + "="*55)
    print("  STEP 0 — ML MODEL EVALUATION (Cross-Dataset)")
    print("="*55)

    try:
        mex_df = pd.read_csv("data/mexican_medical_students_mental_health_data.csv")
        phq_df = pd.read_csv("data/Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv")

        phq_clean = phq_df.dropna(subset=phq_cols).copy()
        mex_clean = mex_df.dropna(subset=phq_cols).copy()

        phq_half = phq_clean.sample(n=len(phq_clean)//2, random_state=42).copy()

        phq_half["PHQ_Total"]  = phq_half[phq_cols].sum(axis=1)
        mex_clean["PHQ_Total"] = mex_clean[phq_cols].sum(axis=1)

        low  = phq_half["PHQ_Total"].quantile(0.33)
        high = phq_half["PHQ_Total"].quantile(0.66)

        def create_lmh(score):
            if score <= low:    return "Low"
            elif score <= high: return "Medium"
            else:               return "High"

        mex_clean["True_LMH"] = mex_clean["PHQ_Total"].apply(create_lmh)

        X_test  = scaler.transform(mex_clean[phq_cols])
        y_test  = mex_clean["True_LMH"]
        y_pred  = lr.predict(X_test)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        print(f"  Training Dataset : PHQ-9 AA Depression (half, no missing -> {len(phq_half)} rows)")
        print(f"  Testing Dataset  : Mexican Medical Students (no missing -> {len(mex_clean)} rows)")
        print(f"  Model            : Logistic Regression")
        print(f"  Scaler           : StandardScaler")
        print("-"*55)
        print(f"  Accuracy  : {round(accuracy_score(y_test, y_pred), 4)}")
        print(f"  Precision : {round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4)}")
        print(f"  Recall    : {round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4)}")
        print(f"  F1 Score  : {round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)}")
        print("="*55)
    except Exception as e:
        print(f"  [Model eval skipped: {e}]")
        print("="*55)



def predict_severity(scores):
    X    = scaler.transform([scores])
    lmh  = lr.predict(X)[0]
    total = sum(scores)
    return lmh, total



def check_llm_crisis(emotion_output):
    """Check if LLM detected suicidal ideation or high risk"""
    risk_level = ""
    problem_text = ""
    emotion_text = ""

    for line in emotion_output.splitlines():
        line = line.strip()
        if line.startswith("Risk:"):
            risk_level = line.split(":", 1)[1].strip().lower()
        if line.startswith("Problem:"):
            problem_text = line.split(":", 1)[1].strip().lower()
        if line.startswith("Emotion:"):
            emotion_text = line.split(":", 1)[1].strip().lower()

    crisis_problems = [
        "suicidal", "suicide", "self harm", "self-harm",
        "hurt myself", "end my life", "kill myself",
        "want to die", "no reason to live"
    ]

    for keyword in crisis_problems:
        if keyword in problem_text or keyword in emotion_text:
            return True

    return False



def main():
    print("\n" + "="*55)
    print("   EMPATHIC MENTAL HEALTH COUNSELING SYSTEM")
    print("="*55)

    show_model_accuracy()

    scores = run_text_phq_assessment()
    lmh, total = predict_severity(scores)
    agent = route_agent(lmh)

    phq_profile = dict(zip(phq_cols, scores))
    phq_profile["PHQ_Total"] = total

    print("\n" + "="*55)
    print("  STEP 2 — SEVERITY CLASSIFICATION + AGENT ROUTING")
    print("="*55)
    print(f"  PHQ Total Score : {total} / 27")
    print(f"  Severity Level  : {lmh}")
    print(f"  Assigned Agent  : {agent}")
    if lmh == "Low":
        print("  Agent Style     : Rapport building + Psychoeducation")
    elif lmh == "Medium":
        print("  Agent Style     : CBT + Cognitive Restructuring")
    else:
        print("  Agent Style     : Resilience + Professional Care")
    print("="*55)

    print("\n  Counselor is ready. Type 'quit' to exit.\n")

    start_session()

    turn_number = 0

    while True:
        user_input = get_input_with_timeout("You: ")

        if user_input.lower() == "quit":
            save_session(get_history(), [], total, agent)
            print(get_summary())
            print("  Take care. Goodbye.")
            break

        add_user_message(user_input)
        turn_number += 1

        print("\n" + "-"*55)

        # ── RULE-BASED SAFETY CHECK ──
        safety = check_risk(user_input)
        if safety["escalate"]:
            print(f"  ⚠️  SAFETY LAYER   : Crisis keyword detected → '{safety['trigger']}'")
            print(f"  ⚠️  LLM bypassed   : Direct crisis response triggered")
            print("-"*55)
            crisis_response = get_crisis_response(agent)
            print(f"\nCounselor ({agent}):\n{crisis_response}\n")
            add_assistant_message(crisis_response)
            log_turn(turn_number, "Crisis", "crisis", "crisis", user_input)
            continue

        # ── STATE MACHINE ──
        update_stage(len(get_history()))
        stage_name   = get_stage_name()
        stage_prompt = get_stage_prompt()
        print(f"  STEP 3 — STATE MACHINE  : Stage → {stage_name}")

        # ── EMOTION REASONER ──
        emotion = analyze_emotion(user_input, phq_profile)
        print(f"  STEP 4 — EMOTION REASONER OUTPUT:")
        for line in emotion.splitlines():
            if line.strip():
                print(f"           {line.strip()}")

        # ── LLM BASED CRISIS CHECK (NEW) ──
        if check_llm_crisis(emotion):
            print(f"  ⚠️  LLM CRISIS DETECTED : Suicidal ideation identified by Emotion Reasoner")
            print(f"  ⚠️  LLM bypassed        : Direct crisis response triggered")
            print("-"*55)
            crisis_response = get_crisis_response(agent)
            print(f"\nCounselor ({agent}):\n{crisis_response}\n")
            add_assistant_message(crisis_response)
            log_turn(turn_number, "Crisis", "crisis", "crisis", user_input)
            continue

        # ── CHAIN OF THOUGHT ──
        cot = generate_cot(emotion, agent)
        print(f"  STEP 5 — CHAIN OF THOUGHT (CoT):")
        for line in cot.splitlines():
            if line.strip():
                print(f"           {line.strip()}")

        # ── CHAIN OF EMPATHY ──
        empathy = generate_empathy(emotion)
        print(f"  STEP 6 — CHAIN OF EMPATHY (CoE):")
        print(f"           {empathy.strip()}")

        # ── RAG RETRIEVAL ──
        topic = "sadness"
        for line in emotion.splitlines():
            if line.startswith("Topic:"):
                topic = line.split(":", 1)[1].strip().lower()

        docs = retrieve(f"{topic} CBT", topic)
        print(f"  STEP 7 — RAG RETRIEVAL:")
        print(f"           Topic    : {topic}")
        print(f"           Top Doc  : {docs[0]['text'][:80]}...")
        print(f"           Score    : {docs[0]['score']}")

        # ── RESPONSE GENERATOR ──
        history  = get_history()
        response = generate_response(
            agent, empathy, cot, docs,
            history, user_input, stage_prompt
        )

        # ── LOG PROGRESS ──
        risk = "low"
        emotion_label = "unknown"
        for line in emotion.splitlines():
            if line.startswith("Risk:"):
                risk = line.split(":", 1)[1].strip().lower()
            if line.startswith("Emotion:"):
                emotion_label = line.split(":", 1)[1].strip()

        log_turn(turn_number, stage_name, emotion_label, risk, user_input)

        add_assistant_message(response)

        print("-"*55)
        print(f"\nCounselor ({agent}):\n{response}\n")

        # ── MOOD REFRESHER ──
        if should_refresh():
            light_moment = get_light_moment()
            print(f"Counselor ({agent}):\n{light_moment}\n")
            add_assistant_message(light_moment)

if __name__ == "__main__":
    main()