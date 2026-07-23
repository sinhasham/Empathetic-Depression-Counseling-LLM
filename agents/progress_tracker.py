# agents/progress_tracker.py

progress_log = []

def log_turn(turn_number, stage, emotion, risk, user_message):
    progress_log.append({
        "turn"   : turn_number,
        "stage"  : stage,
        "emotion": emotion,
        "risk"   : risk,
        "message": user_message
    })

def get_progress():
    return progress_log

def get_summary():
    if not progress_log:
        return "No progress data yet."

    total_turns = len(progress_log)
    stages_visited = list(set([p["stage"] for p in progress_log]))
    risk_levels = [p["risk"] for p in progress_log]

    high_risk_count   = risk_levels.count("crisis")
    medium_risk_count = risk_levels.count("high")

    summary = f"""
=== Progress Summary ===
Total Turns     : {total_turns}
Stages Visited  : {', '.join(stages_visited)}
Crisis Moments  : {high_risk_count}
High Risk Turns : {medium_risk_count}
Last Emotion    : {progress_log[-1]['emotion']}
Last Stage      : {progress_log[-1]['stage']}
========================
"""
    return summary