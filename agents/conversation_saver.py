# agents/conversation_saver.py

import json
import os
from datetime import datetime

SAVE_DIR = "sessions"

def save_session(history, progress_log, phq_score, agent):
    os.makedirs(SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{SAVE_DIR}/session_{timestamp}.json"

    session_data = {
        "timestamp"   : timestamp,
        "phq_score"   : phq_score,
        "agent"       : agent,
        "history"     : history,
        "progress_log": progress_log
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    print(f"\nSession saved → {filename}")

def load_last_session():
    if not os.path.exists(SAVE_DIR):
        return None

    files = sorted(os.listdir(SAVE_DIR))
    if not files:
        return None

    last_file = os.path.join(SAVE_DIR, files[-1])
    with open(last_file, "r", encoding="utf-8") as f:
        return json.load(f)