# agents/state_machine.py

STAGES = {
    1: "Rapport Building",
    2: "Problem Exploration",
    3: "Emotional Reasoning",
    4: "Intervention",
    5: "Action Planning"
}

STAGE_THRESHOLDS = {
    1: 3,
    2: 6,
    3: 9,
    4: 13,
    5: 999
}

current_stage = 1

def get_stage():
    return current_stage

def get_stage_name():
    return STAGES[current_stage]

def update_stage(history_length):
    global current_stage
    for stage, threshold in STAGE_THRESHOLDS.items():
        if history_length < threshold:
            current_stage = stage
            break

def get_stage_prompt():
    if current_stage == 1:
        return "Focus on building rapport. Ask open-ended questions. Do not give advice yet."
    elif current_stage == 2:
        return "Explore the user's problem deeper. Understand triggers, thoughts and feelings."
    elif current_stage == 3:
        return "Identify cognitive distortions and emotional patterns gently."
    elif current_stage == 4:
        return "Now introduce CBT-based coping strategies relevant to the user's situation."
    elif current_stage == 5:
        return "Help the user create small, concrete, achievable action steps."
    return ""